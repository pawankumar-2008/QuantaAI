"""CPU-first training engine for the custom QuantaLM V2 model.

The command-line entry point is deliberately guarded. It will not start the
real dataset experiment until both explicit configuration and command-line
approval are present.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from .checkpoint_manager import CheckpointManager
    from .model import ModelConfig, QuantaLMV2
    from .training_config import (
        DEFAULT_TRAINING_CONFIG,
        TRAINING_METRICS_DIR,
        TrainingConfig,
    )
    from .training_dataset import (
        ChunkDataset,
        DeterministicEpochSampler,
        build_training_loaders,
    )
    from .workspace_config import v2_config
except ImportError:
    from checkpoint_manager import CheckpointManager
    from model import ModelConfig, QuantaLMV2
    from training_config import (
        DEFAULT_TRAINING_CONFIG,
        TRAINING_METRICS_DIR,
        TrainingConfig,
    )
    from training_dataset import (
        ChunkDataset,
        DeterministicEpochSampler,
        build_training_loaders,
    )
    from workspace_config import v2_config


METRIC_FIELDS = (
    "epoch",
    "training_loss",
    "validation_loss",
    "learning_rate",
    "average_gradient_norm",
    "epoch_duration_seconds",
    "training_tokens_processed",
    "validation_tokens_processed",
    "global_steps",
    "best_validation_loss",
    "overfit_gap",
    "overfit_warning",
    "estimated_progress_percentage",
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"run-{timestamp}-{uuid.uuid4().hex[:8]}"


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.stem}-",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(value, temporary_file, ensure_ascii=False, indent=2, sort_keys=True)
            temporary_file.write("\n")
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


class Trainer:
    def __init__(
        self,
        train_loader: DataLoader[dict[str, Any]],
        validation_loader: DataLoader[dict[str, Any]],
        *,
        config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
        model: QuantaLMV2 | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        metrics_dir: str | Path | None = None,
        tokenizer_hash: str | None = None,
        dataset_manifest_hash: str | None = None,
    ) -> None:
        self.config = config
        if config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        self.device = torch.device(config.device)
        if train_loader is validation_loader:
            raise ValueError("Training and validation loaders must be distinct.")
        train_dataset = train_loader.dataset
        validation_dataset = validation_loader.dataset
        if getattr(train_dataset, "split", "train") != "train":
            raise ValueError("The training loader must use only the train split.")
        if getattr(validation_dataset, "split", "validation") != "validation":
            raise ValueError("The validation loader must use only the validation split.")
        if train_loader.batch_size != config.batch_size:
            raise ValueError("Training loader batch size differs from TrainingConfig.")
        if validation_loader.batch_size != config.batch_size:
            raise ValueError("Validation loader batch size differs from TrainingConfig.")
        if len(train_dataset) == 0 or len(validation_dataset) == 0:
            raise ValueError("Training and validation datasets must not be empty.")
        if isinstance(train_dataset, ChunkDataset) and isinstance(validation_dataset, ChunkDataset):
            if train_dataset.manifest_hash != validation_dataset.manifest_hash:
                raise ValueError("Training and validation manifest hashes differ.")
            if train_dataset.tokenizer_hash != validation_dataset.tokenizer_hash:
                raise ValueError("Training and validation tokenizer hashes differ.")
            if train_dataset.document_ids & validation_dataset.document_ids:
                raise ValueError("A document appears in both training and validation datasets.")
            tokenizer_hash = tokenizer_hash or train_dataset.tokenizer_hash
            dataset_manifest_hash = dataset_manifest_hash or train_dataset.manifest_hash
        self.tokenizer_hash = tokenizer_hash
        self.dataset_manifest_hash = dataset_manifest_hash
        for field, value in (
            ("tokenizer", self.tokenizer_hash),
            ("dataset manifest", self.dataset_manifest_hash),
        ):
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"A valid {field} SHA-256 hash is required.")
            try:
                int(value, 16)
            except ValueError as error:
                raise ValueError(
                    f"A valid {field} SHA-256 hash is required."
                ) from error

        _seed_everything(config.random_seed)
        self.model = model or QuantaLMV2()
        if (
            self.model.config.vocab_size != v2_config.VOCAB_SIZE
            or self.model.config.context_length != v2_config.CONTEXT_LENGTH
        ):
            raise ValueError("Model vocabulary/context differs from the V2 workspace.")
        self.model.to(self.device)
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.metrics_dir = Path(metrics_dir) if metrics_dir is not None else TRAINING_METRICS_DIR

        steps_per_epoch = math.ceil(len(train_dataset) / config.batch_size)
        total_steps = steps_per_epoch * config.epochs
        self.scheduler_total_steps = max(total_steps, 1)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self._lr_multiplier,
        )
        self.global_step = 0
        self.current_epoch = 0
        self.batch_in_epoch = 0
        self.best_validation_loss = math.inf
        self.epochs_without_improvement = 0
        self.history: list[dict[str, Any]] = []
        self.partial_epoch: dict[str, float] = {
            "training_loss_sum": 0.0,
            "training_tokens": 0.0,
            "gradient_norm_sum": 0.0,
            "gradient_steps": 0.0,
            "duration_seconds": 0.0,
        }
        self.validation_improved_this_epoch = False

    def _lr_multiplier(self, scheduler_step: int) -> float:
        warmup = min(self.config.warmup_steps, self.scheduler_total_steps)
        if warmup and scheduler_step < warmup:
            return max((scheduler_step + 1) / warmup, 1e-8)
        decay_steps = max(self.scheduler_total_steps - warmup, 1)
        progress = min(max((scheduler_step - warmup) / decay_steps, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.config.min_learning_rate_ratio + (
            1.0 - self.config.min_learning_rate_ratio
        ) * cosine

    def _batch_tensors(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = batch["input_ids"].to(self.device, dtype=torch.long, non_blocking=False)
        target_ids = batch["target_ids"].to(self.device, dtype=torch.long, non_blocking=False)
        attention_mask = batch["attention_mask"].to(
            self.device,
            dtype=torch.bool,
            non_blocking=False,
        )
        loss_mask = batch["loss_mask"].to(self.device, dtype=torch.bool, non_blocking=False)
        if input_ids.ndim != 2 or input_ids.shape != target_ids.shape:
            raise ValueError("input_ids and target_ids must have matching [batch, sequence] shapes.")
        if loss_mask.shape != target_ids.shape or attention_mask.shape != target_ids.shape:
            raise ValueError(
                "attention_mask and loss_mask must match input_ids and target_ids."
            )
        if torch.any(loss_mask & ~attention_mask):
            raise ValueError("loss_mask cannot include attention-masked padding.")
        if torch.any(input_ids[~attention_mask] != v2_config.PAD_TOKEN_ID):
            raise ValueError("Attention-masked input positions must contain PAD.")
        if torch.any(target_ids[~attention_mask] != v2_config.PAD_TOKEN_ID):
            raise ValueError("Attention-masked target positions must contain PAD.")
        if input_ids.shape[1] > self.model.config.context_length:
            raise ValueError("Batch sequence exceeds the model context length.")
        valid_targets = target_ids[loss_mask]
        if torch.any(valid_targets < 0) or torch.any(
            valid_targets >= self.model.config.vocab_size
        ):
            raise ValueError("Valid target IDs are outside the model vocabulary.")
        if not torch.any(loss_mask):
            raise ValueError("Batch contains no valid target tokens.")
        return input_ids, target_ids, loss_mask

    def _loss_sum(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        if logits.shape[:2] != target_ids.shape or loss_mask.shape != target_ids.shape:
            raise ValueError("Logits, targets, and loss mask have incompatible shapes.")
        valid_count = int(loss_mask.sum().item())
        if valid_count <= 0:
            raise ValueError("Loss mask contains no valid targets.")
        selected_logits = logits[loss_mask]
        selected_targets = target_ids[loss_mask]
        loss_sum = F.cross_entropy(selected_logits, selected_targets, reduction="sum")
        if not torch.isfinite(loss_sum):
            raise FloatingPointError("Training loss is not finite.")
        return loss_sum, valid_count

    def train_step(self, batch: dict[str, Any]) -> dict[str, float | int]:
        self.model.train()
        input_ids, target_ids, loss_mask = self._batch_tensors(batch)
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(input_ids)
        loss_sum, valid_tokens = self._loss_sum(logits, target_ids, loss_mask)
        loss = loss_sum / valid_tokens
        if not torch.isfinite(loss):
            raise FloatingPointError("Mean training loss is not finite.")
        loss.backward()
        for parameter in self.model.parameters():
            if parameter.grad is not None and not torch.all(torch.isfinite(parameter.grad)):
                self.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError("A model gradient is not finite.")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.gradient_clip_norm,
            error_if_nonfinite=True,
        )
        if not torch.isfinite(gradient_norm):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("Gradient norm is not finite.")
        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1
        return {
            "loss": float(loss.detach().item()),
            "loss_sum": float(loss_sum.detach().item()),
            "valid_tokens": valid_tokens,
            "gradient_norm": float(gradient_norm.detach().item()),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
        }

    @torch.no_grad()
    def evaluate(self) -> dict[str, float | int]:
        was_training = self.model.training
        self.model.eval()
        loss_sum_total = 0.0
        valid_token_total = 0
        try:
            for batch in self.validation_loader:
                input_ids, target_ids, loss_mask = self._batch_tensors(batch)
                logits = self.model(input_ids)
                loss_sum, valid_tokens = self._loss_sum(logits, target_ids, loss_mask)
                loss_sum_total += float(loss_sum.item())
                valid_token_total += valid_tokens
        finally:
            self.model.train(was_training)
        if valid_token_total == 0:
            raise ValueError("Validation loader contains no valid target tokens.")
        loss = loss_sum_total / valid_token_total
        if not math.isfinite(loss):
            raise FloatingPointError("Validation loss is not finite.")
        return {
            "loss": loss,
            "loss_sum": loss_sum_total,
            "valid_tokens": valid_token_total,
        }

    def _checkpoint_payload(self) -> dict[str, Any]:
        return {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "epoch": self.current_epoch,
            "batch_in_epoch": self.batch_in_epoch,
            "global_step": self.global_step,
            "training_config": self.config.to_dict(),
            "model_config": self.model.config.to_dict(),
            "best_validation_loss": self.best_validation_loss,
            "epochs_without_improvement": self.epochs_without_improvement,
            "random_seed": self.config.random_seed,
            "tokenizer_hash": self.tokenizer_hash,
            "dataset_manifest_hash": self.dataset_manifest_hash,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "history": self.history,
            "partial_epoch": self.partial_epoch,
            "validation_improved_this_epoch": self.validation_improved_this_epoch,
        }

    def _save_checkpoint(self, kind: str) -> Path:
        return self.checkpoint_manager.save(kind, self._checkpoint_payload())

    def resume(self, kind: str = "resume") -> None:
        checkpoint = self.checkpoint_manager.load(kind, map_location=self.device)
        expected = {
            "training_config": self.config.to_dict(),
            "model_config": self.model.config.to_dict(),
            "random_seed": self.config.random_seed,
            "tokenizer_hash": self.tokenizer_hash,
            "dataset_manifest_hash": self.dataset_manifest_hash,
        }
        for field, value in expected.items():
            if checkpoint.get(field) != value:
                raise ValueError(f"Checkpoint {field} is incompatible with this run.")
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.current_epoch = int(checkpoint["epoch"])
        self.batch_in_epoch = int(checkpoint["batch_in_epoch"])
        self.global_step = int(checkpoint["global_step"])
        self.best_validation_loss = float(checkpoint["best_validation_loss"])
        self.epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        self.history = list(checkpoint["history"])
        self.partial_epoch = dict(checkpoint["partial_epoch"])
        self.validation_improved_this_epoch = bool(
            checkpoint["validation_improved_this_epoch"]
        )
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if self.device.type == "cuda" and checkpoint["cuda_rng_state"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        self.model.to(self.device)

    def _write_metrics(self) -> None:
        metrics_dir = self.metrics_dir
        metrics_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(
            metrics_dir / "training_metrics.json",
            {
                "model": v2_config.MODEL_NAME,
                "training_config": self.config.to_dict(),
                "dataset_manifest_sha256": self.dataset_manifest_hash,
                "tokenizer_sha256": self.tokenizer_hash,
                "test_split_evaluated": False,
                "overfitting_warning": (
                    "The corpus is small and imbalanced. Lower training loss alone "
                    "does not establish generalization or model quality."
                ),
                "epochs": self.history,
            },
        )
        csv_path = metrics_dir / "training_metrics.csv"
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                prefix=".training_metrics-",
                suffix=".tmp",
                dir=metrics_dir,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                writer = csv.DictWriter(temporary_file, fieldnames=METRIC_FIELDS)
                writer.writeheader()
                writer.writerows(self.history)
            os.replace(temporary_path, csv_path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _train_batch(self, batch: dict[str, Any]) -> dict[str, float | int]:
        started = time.perf_counter()
        result = self.train_step(batch)
        self.partial_epoch["training_loss_sum"] += float(result["loss_sum"])
        self.partial_epoch["training_tokens"] += int(result["valid_tokens"])
        self.partial_epoch["gradient_norm_sum"] += float(result["gradient_norm"])
        self.partial_epoch["gradient_steps"] += 1
        self.partial_epoch["duration_seconds"] += time.perf_counter() - started
        self.batch_in_epoch += 1
        return result

    def fit(self, *, max_steps: int | None = None) -> list[dict[str, Any]]:
        """Fit for configured epochs; max_steps is intended for bounded smoke tests."""
        if max_steps is not None and (
            not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1
        ):
            raise ValueError("max_steps must be a positive integer.")
        if self.current_epoch == 0 and self.global_step == 0:
            training_documents = getattr(
                self.train_loader.dataset,
                "document_ids",
                None,
            )
            if training_documents is not None and len(training_documents) < 100:
                print(
                    "WARNING: training uses only "
                    f"{len(training_documents)} unique documents; expect high "
                    "overfitting risk and weak generalization."
                )
        steps_this_call = 0
        stop = False

        while self.current_epoch < self.config.epochs and not stop:
            epoch_started = time.perf_counter()
            duration_before_epoch = self.partial_epoch["duration_seconds"]
            sampler = self.train_loader.sampler
            if isinstance(sampler, DeterministicEpochSampler):
                sampler.set_epoch(
                    self.current_epoch,
                    start_index=min(
                        self.batch_in_epoch * self.config.batch_size,
                        len(self.train_loader.dataset),
                    ),
                )
            elif self.batch_in_epoch:
                raise ValueError("Resume requires a DeterministicEpochSampler.")
            for batch in self.train_loader:
                self._train_batch(batch)
                steps_this_call += 1

                if self.global_step % self.config.evaluation_interval == 0:
                    validation = self.evaluate()
                    self._consider_best(validation["loss"])

                if self.global_step % self.config.checkpoint_interval == 0:
                    self._save_checkpoint("latest")
                    self._save_checkpoint("resume")

                if max_steps is not None and steps_this_call >= max_steps:
                    self._save_checkpoint("latest")
                    self._save_checkpoint("resume")
                    self._write_metrics()
                    return self.history

            validation = self.evaluate()
            training_loss = (
                self.partial_epoch["training_loss_sum"]
                / self.partial_epoch["training_tokens"]
            )
            validation_loss = float(validation["loss"])
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            epoch_number = self.current_epoch + 1
            record = {
                "epoch": epoch_number,
                "training_loss": training_loss,
                "validation_loss": validation_loss,
                "learning_rate": learning_rate,
                "average_gradient_norm": (
                    self.partial_epoch["gradient_norm_sum"]
                    / max(self.partial_epoch["gradient_steps"], 1)
                ),
                "epoch_duration_seconds": duration_before_epoch
                + max(0.0, time.perf_counter() - epoch_started),
                "training_tokens_processed": int(self.partial_epoch["training_tokens"]),
                "validation_tokens_processed": int(validation["valid_tokens"]),
                "global_steps": self.global_step,
                "best_validation_loss": min(self.best_validation_loss, validation_loss),
                "overfit_gap": validation_loss - training_loss,
                "overfit_warning": validation_loss > training_loss,
                "estimated_progress_percentage": min(
                    100.0 * self.global_step / self.scheduler_total_steps,
                    100.0,
                ),
            }
            self.history.append(record)
            self._consider_best(validation_loss)
            if not self.validation_improved_this_epoch:
                self.epochs_without_improvement += 1
            self.current_epoch += 1
            self.batch_in_epoch = 0
            self.partial_epoch = {
                "training_loss_sum": 0.0,
                "training_tokens": 0.0,
                "gradient_norm_sum": 0.0,
                "gradient_steps": 0.0,
                "duration_seconds": 0.0,
            }
            self.validation_improved_this_epoch = False
            self._write_metrics()
            self._save_checkpoint("latest")
            self._save_checkpoint("resume")
            eta_seconds = record["epoch_duration_seconds"] * max(
                self.config.epochs - epoch_number,
                0,
            )
            warning = (
                " | validation loss exceeds training loss"
                if record["overfit_warning"]
                else ""
            )
            print(
                f"Epoch {epoch_number}/{self.config.epochs} | "
                f"train loss {training_loss:.4f} | "
                f"validation loss {validation_loss:.4f} | "
                f"learning rate {learning_rate:.3g} | "
                f"time {record['epoch_duration_seconds']:.1f}s | "
                f"tokens train/validation "
                f"{record['training_tokens_processed']}/"
                f"{record['validation_tokens_processed']} | "
                f"progress {record['estimated_progress_percentage']:.1f}% | "
                f"estimated remaining {eta_seconds:.1f}s{warning}"
            )
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                stop = True

        return self.history

    def _consider_best(self, validation_loss: float | int) -> None:
        value = float(validation_loss)
        if not math.isfinite(value):
            raise FloatingPointError("Cannot select a checkpoint from non-finite validation loss.")
        if value < self.best_validation_loss:
            self.best_validation_loss = value
            self.epochs_without_improvement = 0
            self.validation_improved_this_epoch = True
            self._save_checkpoint("best")


def run_training(
    config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
    *,
    model_config: ModelConfig | None = None,
    resume_kind: str | None = None,
    checkpoint_dir: str | Path | None = None,
    confirm_training: bool = False,
) -> list[dict[str, Any]]:
    if not v2_config.TRAINING_ENABLED:
        raise RuntimeError("Training is disabled by the V2 workspace configuration.")
    if not confirm_training:
        raise RuntimeError(
            "Training requires the explicit --confirm-training approval flag."
        )
    if resume_kind is not None and checkpoint_dir is None:
        raise ValueError("Resuming requires the original --checkpoint-dir.")
    checkpoint_dir = (
        Path(checkpoint_dir)
        if checkpoint_dir is not None
        else v2_config.CHECKPOINT_DIR / _new_run_id()
    )
    checkpoint_manager = CheckpointManager(checkpoint_dir)
    if resume_kind is None and any(
        checkpoint_manager.path_for(kind).exists()
        for kind in ("latest", "best", "resume")
    ):
        raise FileExistsError(
            f"Checkpoint directory already contains a run: {checkpoint_manager.checkpoint_dir}"
        )
    train_loader, validation_loader = build_training_loaders(config)
    trainer = Trainer(
        train_loader,
        validation_loader,
        config=config,
        model=QuantaLMV2(model_config) if model_config is not None else None,
        checkpoint_manager=checkpoint_manager,
    )
    print(f"Training checkpoint directory: {checkpoint_manager.checkpoint_dir}")
    if resume_kind is not None:
        trainer.resume(resume_kind)
    return trainer.fit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the custom QuantaLM V2 model.")
    parser.add_argument(
        "--confirm-training",
        action="store_true",
        help="Required explicit approval to train on the V2 corpus.",
    )
    parser.add_argument(
        "--resume",
        choices=("latest", "best", "resume"),
        help="Resume from an existing V2 training checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Existing V2 run directory to resume, or an empty directory for a new run.",
    )
    parser.add_argument(
        "--run-id",
        help="Unique name for a new versioned checkpoint directory.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_TRAINING_CONFIG.batch_size)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_TRAINING_CONFIG.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_TRAINING_CONFIG.weight_decay)
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAINING_CONFIG.epochs)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_TRAINING_CONFIG.warmup_steps)
    parser.add_argument("--gradient-clip-norm", type=float, default=DEFAULT_TRAINING_CONFIG.gradient_clip_norm)
    parser.add_argument("--evaluation-interval", type=int, default=DEFAULT_TRAINING_CONFIG.evaluation_interval)
    parser.add_argument("--checkpoint-interval", type=int, default=DEFAULT_TRAINING_CONFIG.checkpoint_interval)
    parser.add_argument("--early-stopping-patience", type=int, default=DEFAULT_TRAINING_CONFIG.early_stopping_patience)
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAINING_CONFIG.random_seed)
    parser.add_argument("--device", default=DEFAULT_TRAINING_CONFIG.device)
    parser.add_argument("--context-length", type=int, default=v2_config.CONTEXT_LENGTH)
    parser.add_argument("--embedding-dim", type=int, default=v2_config.EMBEDDING_DIM)
    parser.add_argument("--num-layers", type=int, default=v2_config.NUM_LAYERS)
    parser.add_argument("--num-heads", type=int, default=v2_config.NUM_HEADS)
    parser.add_argument("--feed-forward-dim", type=int, default=v2_config.FEED_FORWARD_DIM)
    parser.add_argument("--dropout", type=float, default=v2_config.DROPOUT)
    args = parser.parse_args()
    if not args.confirm_training:
        parser.error("Pass --confirm-training only after approving a full training run.")
    if args.resume and (args.checkpoint_dir is None or args.run_id is not None):
        parser.error("Resume requires --checkpoint-dir and cannot use --run-id.")
    if args.run_id is not None and args.checkpoint_dir is not None:
        parser.error("Use either --run-id or --checkpoint-dir for a new run, not both.")
    if args.run_id and (
        args.run_id in {".", ".."}
        or Path(args.run_id).name != args.run_id
        or any(not (character.isalnum() or character in "-_") for character in args.run_id)
    ):
        parser.error("--run-id may contain only letters, numbers, hyphens, and underscores.")
    if args.context_length != v2_config.CONTEXT_LENGTH:
        parser.error(
            "--context-length must match the prepared chunk dataset "
            f"({v2_config.CONTEXT_LENGTH}); rebuild the dataset before changing it."
        )
    try:
        training_config = TrainingConfig(
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            warmup_steps=args.warmup_steps,
            gradient_clip_norm=args.gradient_clip_norm,
            random_seed=args.seed,
            evaluation_interval=args.evaluation_interval,
            checkpoint_interval=args.checkpoint_interval,
            early_stopping_patience=args.early_stopping_patience,
            device=args.device,
        )
        model_config = ModelConfig(
            context_length=args.context_length,
            embedding_dim=args.embedding_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            feed_forward_dim=args.feed_forward_dim,
            dropout=args.dropout,
        )
    except ValueError as error:
        parser.error(str(error))

    estimated_parameters = (
        model_config.vocab_size * model_config.embedding_dim
        + model_config.context_length * model_config.embedding_dim
        + model_config.num_layers
        * (
            4 * model_config.embedding_dim**2
            + 2 * model_config.embedding_dim * model_config.feed_forward_dim
            + 2 * model_config.feed_forward_dim
            + 6 * model_config.embedding_dim
        )
        + 2 * model_config.embedding_dim
    )
    estimated_memory_bytes = (
        estimated_parameters * 16
        + args.batch_size
        * model_config.num_layers
        * model_config.context_length
        * model_config.embedding_dim
        * 48
    )
    if estimated_memory_bytes > 2 * 1024**3:
        parser.error(
            "Estimated CPU training memory exceeds the 2 GiB safety limit; "
            "reduce model dimensions, context, or batch size."
        )
    history = run_training(
        config=training_config,
        model_config=model_config,
        resume_kind=args.resume,
        checkpoint_dir=(
            args.checkpoint_dir
            if args.checkpoint_dir is not None
            else (
                v2_config.CHECKPOINT_DIR / args.run_id
                if args.run_id
                else None
            )
        ),
        confirm_training=True,
    )
    print(json.dumps(history[-1] if history else {}, indent=2))


if __name__ == "__main__":
    main()
