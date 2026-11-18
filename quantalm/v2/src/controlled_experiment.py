"""Explicitly bounded V2 experiments using the existing training pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from .checkpoint_manager import CheckpointManager
from .bpe_tokenizer import QuantaBPETokenizer
from .evaluator import (
    GenerationConfig,
    _memory_snapshot,
    _model_state_hash,
    _tokenizer_artifact_hashes,
    benchmark_generation,
    calculate_perplexity,
    generation_quality,
)
from .generation import generate
from .model import QuantaLMV2
from .trainer import Trainer
from .training_config import TrainingConfig
from .training_dataset import (
    DeterministicEpochSampler,
    build_training_loaders,
    read_json,
    sha256_file,
)
from .workspace_config import v2_config


V2_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = V2_ROOT / "experiments" / "phase_3_6_controlled_cpu.json"
TRAINING_MANIFEST = v2_config.DATA_DIR / "manifests" / "dataset_manifest.json"
EXPERIMENTS_ROOT = v2_config.DATA_DIR / "reports" / "experiments"
CHECKPOINT_EXPERIMENTS_ROOT = v2_config.CHECKPOINT_DIR / "experiments"


def load_experiment_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Experiment configuration not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as source:
        config = json.load(source)
    required = {
        "experiment_id",
        "description",
        "training",
        "max_steps",
        "generation",
        "benchmark",
        "data_policy",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("Controlled experiment configuration has unexpected fields.")
    experiment_id = config["experiment_id"]
    if (
        not isinstance(experiment_id, str)
        or not experiment_id
        or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in experiment_id)
    ):
        raise ValueError("experiment_id must use only letters, digits, hyphens, and underscores.")
    max_steps = config["max_steps"]
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or not 1 <= max_steps <= 10:
        raise ValueError("Controlled experiment max_steps must be between 1 and 10.")
    if config["data_policy"] != {
        "train_split": "train",
        "monitoring_split": "validation",
        "test_split_used": False,
        "external_data_used": False,
        "source_license_status": "unknown; research-use authorization is not established by repository presence",
        "experiment_scope": "controlled research probe on existing local corpus only",
    }:
        raise ValueError("Experiment data policy must prohibit test and external data.")
    training = config["training"]
    if not isinstance(training, dict):
        raise ValueError("training must be an object.")
    training_config = TrainingConfig(**training)
    if training_config.device != "cpu":
        raise ValueError("The initial controlled experiment must use CPU.")
    generation = config["generation"]
    if not isinstance(generation, dict):
        raise ValueError("generation must be an object.")
    generation_config = GenerationConfig(
        max_new_tokens=generation.get("max_new_tokens"),
        temperature=generation.get("temperature"),
        top_k=generation.get("top_k"),
        random_seed=generation.get("random_seed"),
    )
    if not isinstance(generation.get("prompt"), str) or not generation["prompt"].strip():
        raise ValueError("generation.prompt must be non-empty.")
    benchmark = config["benchmark"]
    if (
        not isinstance(benchmark, dict)
        or set(benchmark) != {"warmup_runs", "measured_runs"}
        or not isinstance(benchmark["warmup_runs"], int)
        or isinstance(benchmark["warmup_runs"], bool)
        or benchmark["warmup_runs"] < 0
        or not isinstance(benchmark["measured_runs"], int)
        or isinstance(benchmark["measured_runs"], bool)
        or benchmark["measured_runs"] < 1
    ):
        raise ValueError("benchmark must specify valid warmup_runs and measured_runs.")
    return config


def _atomic_json(path: Path, payload: Any) -> None:
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
        ) as target:
            temporary_path = Path(target.name)
            json.dump(payload, target, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            target.write("\n")
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _source_audit() -> dict[str, Any]:
    manifest = read_json(TRAINING_MANIFEST)
    report_path = v2_config.DATA_DIR / "reports" / "data_quality_report.json"
    duplicate_report_path = v2_config.DATA_DIR / "reports" / "duplicate_report.json"
    quality_report = read_json(report_path)
    duplicate_report = read_json(duplicate_report_path)
    split_files: dict[str, str] = {}
    source_ids: dict[str, set[str]] = {}
    for split in ("train", "validation", "test"):
        split_info = manifest["split_files"][split]
        document_path = v2_config.DATA_DIR / split_info["path"]
        split_files[split] = sha256_file(document_path)
        with document_path.open("r", encoding="utf-8") as source:
            docs = [json.loads(line) for line in source if line.strip()]
        source_ids[split] = {record["document_id"] for record in docs}
    if source_ids["train"] & source_ids["validation"]:
        raise ValueError("Train and validation document IDs overlap.")
    if source_ids["train"] & source_ids["test"]:
        raise ValueError("Train and test document IDs overlap.")
    if source_ids["validation"] & source_ids["test"]:
        raise ValueError("Validation and test document IDs overlap.")
    provenance = quality_report["provenance_and_licensing"]
    license_status = provenance["license_status"]
    if license_status != "unknown for all source documents":
        raise ValueError(f"Unexpected source licensing status: {license_status}")
    leak_check = quality_report["split_leakage_validation"]
    if leak_check.get("status") != "passed":
        raise ValueError("Dataset split leakage report has not passed.")
    return {
        "dataset_manifest_sha256": sha256_file(TRAINING_MANIFEST),
        "data_quality_report_sha256": sha256_file(report_path),
        "duplicate_report_sha256": sha256_file(duplicate_report_path),
        "source_document_split_sha256": split_files,
        "document_counts": {
            split: len(ids) for split, ids in source_ids.items()
        },
        "split_document_overlap_counts": {
            "train_validation": len(source_ids["train"] & source_ids["validation"]),
            "train_test": len(source_ids["train"] & source_ids["test"]),
            "validation_test": len(source_ids["validation"] & source_ids["test"]),
        },
        "duplicate_and_shared_passage_audit": leak_check,
        "corpus_duplicate_content_summary": {
            "exact_duplicate_documents_removed": duplicate_report[
                "exact_duplicate_documents_removed"
            ],
            "normalized_duplicate_documents_removed": duplicate_report[
                "normalized_duplicate_documents_removed"
            ],
            "near_duplicate_documents_removed": duplicate_report[
                "near_duplicate_detection"
            ]["near_duplicate_documents_removed"],
            "shared_substantive_line_count": duplicate_report[
                "repeated_lines"
            ]["shared_substantive_line_count"],
            "shared_passage_count": duplicate_report[
                "repeated_passages"
            ]["shared_passage_count"],
            "duplicate_documents_are_removed_or_grouped_before_split": True,
        },
        "licensing": license_status,
        "permitted_scope": (
            "No external material added. The run is a controlled experiment on "
            "existing local data only; source reuse authorization remains unresolved."
        ),
    }


def _generation_sample(
    model: QuantaLMV2,
    tokenizer: Any,
    generation: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    prompt = generation["prompt"]
    prompt_ids = [tokenizer.bos_token_id, *tokenizer.encode(prompt)]
    if len(prompt_ids) > model.config.context_length:
        raise ValueError("Controlled experiment generation prompt exceeds model context.")
    config = GenerationConfig(
        max_new_tokens=generation["max_new_tokens"],
        temperature=generation["temperature"],
        top_k=generation["top_k"],
        random_seed=generation["random_seed"],
    )
    output = generate(
        model,
        torch.tensor([prompt_ids], dtype=torch.long, device=device),
        config.max_new_tokens,
        config.temperature,
        config.top_k,
        eos_token_id=tokenizer.eos_token_id,
        seed=config.random_seed,
    )
    generated_ids = output[0, len(prompt_ids) :].detach().cpu().tolist()
    answer_ids = generated_ids[:-1] if generated_ids and generated_ids[-1] == tokenizer.eos_token_id else generated_ids
    answer = tokenizer.decode(answer_ids)
    quality = generation_quality(
        generated_ids,
        answer,
        eos_token_id=tokenizer.eos_token_id,
        vocab_size=model.config.vocab_size,
        tokenizer=tokenizer,
    )
    return {
        "prompt": prompt,
        "prompt_token_count": len(prompt_ids),
        "answer": answer,
        "generated_token_ids": generated_ids,
        "generation_config": {
            **config.__dict__,
        },
        "quality": quality,
    }


def _write_step_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError("No experiment training steps were recorded.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "step",
        "training_loss",
        "training_valid_tokens",
        "validation_loss",
        "validation_valid_tokens",
        "learning_rate",
        "gradient_norm",
        "step_duration_seconds",
        "cumulative_training_seconds",
    )
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: record[field] for field in fields} for record in records)


def run_controlled_experiment(
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    experiment_id = config["experiment_id"]
    report_dir = EXPERIMENTS_ROOT / experiment_id
    checkpoint_dir = CHECKPOINT_EXPERIMENTS_ROOT / experiment_id
    if report_dir.exists() or checkpoint_dir.exists():
        raise FileExistsError(
            "Experiment outputs already exist; refusing to overwrite: "
            f"{report_dir} or {checkpoint_dir}"
        )
    report_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)

    audit = _source_audit()
    training_config = TrainingConfig(**config["training"])
    train_loader, validation_loader = build_training_loaders(training_config)
    train_docs = getattr(train_loader.dataset, "document_ids", set())
    validation_docs = getattr(validation_loader.dataset, "document_ids", set())
    if train_docs & validation_docs:
        raise ValueError("Train and validation chunks contain overlapping document IDs.")
    if len(train_loader.dataset) != 285 or len(validation_loader.dataset) != 38:
        raise ValueError("Dataset chunk counts differ from the inspected Phase 3.5 data.")
    sampler = train_loader.sampler
    if not isinstance(sampler, DeterministicEpochSampler):
        raise TypeError("Controlled experiment requires DeterministicEpochSampler.")

    initial_memory, initial_memory_method = _memory_snapshot()
    torch.manual_seed(training_config.random_seed)
    model = QuantaLMV2()
    trainer = Trainer(
        train_loader,
        validation_loader,
        config=training_config,
        model=model,
        checkpoint_manager=CheckpointManager(checkpoint_dir),
        metrics_dir=report_dir / "training_metrics",
        tokenizer_hash=train_loader.dataset.tokenizer_hash,
        dataset_manifest_hash=train_loader.dataset.manifest_hash,
    )
    tokenizer = QuantaBPETokenizer.load(v2_config.TOKENIZER_DIR)
    validation_records = validation_loader.dataset.records
    baseline_model_hash = _model_state_hash(model)
    baseline_validation = calculate_perplexity(
        model,
        validation_records,
        device=training_config.device,
    )
    baseline_generation = _generation_sample(
        model,
        tokenizer,
        config["generation"],
        device=torch.device(training_config.device),
    )
    baseline_benchmark = benchmark_generation(
        model,
        torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long),
        max_new_tokens=8,
        temperature=config["generation"]["temperature"],
        top_k=config["generation"]["top_k"],
        random_seed=config["generation"]["random_seed"],
        warmup_runs=config["benchmark"]["warmup_runs"],
        measured_runs=config["benchmark"]["measured_runs"],
    )

    training_steps: list[dict[str, Any]] = []
    experiment_started = time.perf_counter()
    sampler.set_epoch(0, start_index=0)
    cumulative_training_seconds = 0.0
    for step_index in range(config["max_steps"]):
        sampler.set_epoch(0, start_index=step_index * training_config.batch_size)
        step_iterator = iter(train_loader)
        try:
            batch = next(step_iterator)
        except StopIteration as error:
            raise RuntimeError("Training loader ended before the configured step limit.") from error
        step_started = time.perf_counter()
        train_metrics = trainer.train_step(batch)
        step_duration = time.perf_counter() - step_started
        cumulative_training_seconds += step_duration
        trainer.batch_in_epoch = step_index + 1
        trainer.partial_epoch["training_loss_sum"] += float(train_metrics["loss_sum"])
        trainer.partial_epoch["training_tokens"] += int(train_metrics["valid_tokens"])
        trainer.partial_epoch["gradient_norm_sum"] += float(train_metrics["gradient_norm"])
        trainer.partial_epoch["gradient_steps"] += 1
        trainer.partial_epoch["duration_seconds"] += step_duration

        validation_metrics = trainer.evaluate()
        if validation_metrics["loss"] < trainer.best_validation_loss:
            trainer.best_validation_loss = float(validation_metrics["loss"])
            trainer.epochs_without_improvement = 0
            trainer.validation_improved_this_epoch = True
            trainer.checkpoint_manager.save("best", trainer._checkpoint_payload())
        else:
            trainer.epochs_without_improvement += 1
        training_steps.append(
            {
                "step": step_index + 1,
                "training_loss": float(train_metrics["loss"]),
                "training_valid_tokens": int(train_metrics["valid_tokens"]),
                "validation_loss": float(validation_metrics["loss"]),
                "validation_valid_tokens": int(validation_metrics["valid_tokens"]),
                "learning_rate": float(train_metrics["learning_rate"]),
                "gradient_norm": float(train_metrics["gradient_norm"]),
                "step_duration_seconds": step_duration,
                "cumulative_training_seconds": cumulative_training_seconds,
            }
        )

    total_experiment_seconds = time.perf_counter() - experiment_started
    if not training_steps:
        raise RuntimeError("Controlled experiment performed no optimizer updates.")
    trainer.checkpoint_manager.save("latest", trainer._checkpoint_payload())
    trainer.checkpoint_manager.save("resume", trainer._checkpoint_payload())
    best_checkpoint = trainer.checkpoint_manager.path_for("best")
    if not best_checkpoint.is_file():
        raise RuntimeError("No best validation checkpoint was saved.")
    best_checkpoint_hash = sha256_file(best_checkpoint)

    from .evaluator import _load_checkpoint_state

    _load_checkpoint_state(model, best_checkpoint)
    model.to(training_config.device)
    trained_model_hash = _model_state_hash(model)
    trained_validation = calculate_perplexity(
        model,
        validation_records,
        device=training_config.device,
    )
    trained_generation = _generation_sample(
        model,
        tokenizer,
        config["generation"],
        device=torch.device(training_config.device),
    )
    trained_benchmark = benchmark_generation(
        model,
        torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long),
        max_new_tokens=8,
        temperature=config["generation"]["temperature"],
        top_k=config["generation"]["top_k"],
        random_seed=config["generation"]["random_seed"],
        warmup_runs=config["benchmark"]["warmup_runs"],
        measured_runs=config["benchmark"]["measured_runs"],
    )
    final_memory, memory_method = _memory_snapshot()
    tokenizer_hash, tokenizer_artifact_hashes = _tokenizer_artifact_hashes(
        v2_config.TOKENIZER_DIR
    )
    report = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "controlled_experiment_complete",
        "scope": config["description"],
        "model_label": "three_step_experimental_checkpoint_not_trained_model_claim",
        "training_enabled_setting_remains": v2_config.TRAINING_ENABLED,
        "experiment_configuration": config,
        "training_configuration": training_config.to_dict(),
        "model_configuration": model.config.to_dict(),
        "model_parameter_count": model.parameter_count(),
        "dataset_audit": audit,
        "dataset_usage": {
            "training_chunks": len(train_loader.dataset),
            "validation_chunks": len(validation_loader.dataset),
            "training_documents": len(train_docs),
            "validation_documents": len(validation_docs),
            "validation_used_for_gradients": False,
            "held_out_test_evaluated": False,
            "external_data_used": False,
            "training_chunk_sha256": sha256_file(train_loader.dataset.path),
            "validation_chunk_sha256": sha256_file(validation_loader.dataset.path),
            "tokenizer_sha256": tokenizer_hash,
            "tokenizer_artifact_sha256": tokenizer_artifact_hashes,
        },
        "baseline": {
            "label": "untrained_random_initialization",
            "model_state_sha256": baseline_model_hash,
            "validation_perplexity": baseline_validation,
            "generation": baseline_generation,
            "performance": baseline_benchmark,
        },
        "training": {
            "optimizer_steps": len(training_steps),
            "training_loss_by_step": [record["training_loss"] for record in training_steps],
            "validation_loss_by_step": [record["validation_loss"] for record in training_steps],
            "step_metrics": training_steps,
            "training_time_seconds": cumulative_training_seconds,
            "total_experiment_time_seconds": total_experiment_seconds,
            "best_validation_loss": trainer.best_validation_loss,
            "cpu_peak_working_set_bytes_after_run": final_memory,
            "cpu_memory_measurement": memory_method,
            "cpu_peak_working_set_bytes_before_training": initial_memory,
            "cpu_initial_memory_measurement": initial_memory_method,
            "training_metrics_csv": str((report_dir / "training_steps.csv").relative_to(V2_ROOT)),
            "training_metrics_json": str((report_dir / "training_metrics.json").relative_to(V2_ROOT)),
        },
        "experimental_checkpoint": {
            "best_checkpoint_path": str(best_checkpoint.relative_to(V2_ROOT)),
            "best_checkpoint_sha256": best_checkpoint_hash,
            "latest_checkpoint_path": str(
                trainer.checkpoint_manager.path_for("latest").relative_to(V2_ROOT)
            ),
            "resume_checkpoint_path": str(
                trainer.checkpoint_manager.path_for("resume").relative_to(V2_ROOT)
            ),
            "model_state_sha256": trained_model_hash,
            "training_global_step": trainer.global_step,
            "label": "controlled three-step experiment; not a production or quality claim",
        },
        "trained_experiment": {
            "validation_perplexity": trained_validation,
            "generation": trained_generation,
            "performance": trained_benchmark,
        },
        "comparison": {
            "validation_perplexity_delta": (
                trained_validation["perplexity"] - baseline_validation["perplexity"]
            ),
            "validation_nll_delta": (
                trained_validation["average_negative_log_likelihood"]
                - baseline_validation["average_negative_log_likelihood"]
            ),
            "generation_text_changed": (
                baseline_generation["answer"] != trained_generation["answer"]
            ),
            "quality_metrics_are_human_assessed": False,
            "human_evaluation_score": None,
        },
        "limitations": [
            "All source-document license statuses are unknown; this experiment adds no external data and does not establish reuse authorization.",
            "The 11-document training corpus and one-document validation set are too small and imbalanced for meaningful generalization claims.",
            "A three-step run is only an optimization-pipeline experiment, not evidence that the model is capable or intelligent.",
            "There are no human-verified evaluation questions; generation comparison is a fixed diagnostic prompt, not correctness evaluation.",
            "Validation perplexity reuses one document and overlapping chunks; it is monitoring only and can be overfit.",
            "Peak process working set is process-wide and not an isolated measurement of model-only memory.",
            "No held-out test evaluation, full-corpus training, or GPU execution was performed.",
        ],
    }
    _atomic_json(report_dir / "experiment_report.json", report)
    _atomic_json(report_dir / "training_metrics.json", {
        "experiment_id": experiment_id,
        "test_split_evaluated": False,
        "training_steps": training_steps,
        "training_time_seconds": cumulative_training_seconds,
        "total_experiment_time_seconds": total_experiment_seconds,
        "best_validation_loss": trainer.best_validation_loss,
        "training_config": training_config.to_dict(),
    })
    _write_step_csv(report_dir / "training_steps.csv", training_steps)
    _write_readable_report(report_dir / "experiment_report.md", report)
    return report


def _write_readable_report(path: Path, result: dict[str, Any]) -> None:
    baseline = result["baseline"]
    trained = result["trained_experiment"]
    training = result["training"]
    lines = [
        f"# Controlled Training Experiment: {result['experiment_id']}",
        "",
        f"- **Status:** {result['status']}",
        f"- **Scope:** {result['scope']}",
        f"- **Model label:** {result['model_label']}",
        f"- **Steps:** {result['training']['optimizer_steps']}",
        f"- **Device:** {trained['performance']['device_information']['device']}",
        f"- **Training time:** {training['training_time_seconds']:.3f} seconds",
        f"- **Total run time:** {training['total_experiment_time_seconds']:.3f} seconds",
        f"- **Process peak working set:** {training['cpu_peak_working_set_bytes_after_run']} bytes ({training['cpu_memory_measurement']})",
        f"- **Experimental best checkpoint:** `{result['experimental_checkpoint']['best_checkpoint_path']}`",
        f"- **Checkpoint SHA-256:** `{result['experimental_checkpoint']['best_checkpoint_sha256']}`",
        "",
        "## Dataset audit",
        "",
        f"- Training documents/chunks: {result['dataset_usage']['training_documents']} / {result['dataset_usage']['training_chunks']}",
        f"- Validation documents/chunks: {result['dataset_usage']['validation_documents']} / {result['dataset_usage']['validation_chunks']}",
        "- Validation data was used only for loss monitoring and checkpoint selection, never for gradient updates.",
        "- Held-out test split was not evaluated.",
        "- Split leakage audit: passed; source licensing remains unknown.",
        "- No new or external data was used.",
        "",
        "## Loss and perplexity",
        "",
        f"- Baseline validation perplexity: {baseline['validation_perplexity']['perplexity']:.6g}",
        f"- Experimental checkpoint validation perplexity: {trained['validation_perplexity']['perplexity']:.6g}",
        f"- Perplexity delta: {result['comparison']['validation_perplexity_delta']:.6g}",
        f"- Baseline validation NLL: {baseline['validation_perplexity']['average_negative_log_likelihood']:.6f}",
        f"- Experimental validation NLL: {trained['validation_perplexity']['average_negative_log_likelihood']:.6f}",
        "",
        "| Step | Training loss | Validation loss | Learning rate | Gradient norm | Seconds |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        *[
            f"| {step['step']} | {step['training_loss']:.6f} | {step['validation_loss']:.6f} | {step['learning_rate']:.3g} | {step['gradient_norm']:.6f} | {step['step_duration_seconds']:.3f} |"
            for step in training["step_metrics"]
        ],
        "",
        "## Generation probe (not a correctness test)",
        "",
        f"Prompt: {result['experiment_configuration']['generation']['prompt']}",
        "",
        "Untrained baseline output:",
        "",
        "```text",
        baseline["generation"]["answer"],
        "```",
        "",
        "Experimental checkpoint output:",
        "",
        "```text",
        trained["generation"]["answer"],
        "```",
        "",
        f"- Repeated-token ratio: {baseline['generation']['quality']['repeated_token_ratio']:.4f} → {trained['generation']['quality']['repeated_token_ratio']:.4f}",
        f"- Repeated n-gram ratio: {baseline['generation']['quality']['repeated_ngram_ratio']:.4f} → {trained['generation']['quality']['repeated_ngram_ratio']:.4f}",
        f"- EOS terminated: {baseline['generation']['quality']['eos_terminated']} → {trained['generation']['quality']['eos_terminated']}",
        "- Outputs are diagnostic samples only; no human quality or factuality score is available.",
        "",
        "## Limitations",
        "",
        *[f"- {limitation}" for limitation in result["limitations"]],
        "",
        "No full-corpus training or test evaluation was run. Obtain data-license clarification and human-reviewed validation examples before designing a larger experiment.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an explicitly bounded QuantaLM V2 experiment.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--confirm-controlled-experiment",
        action="store_true",
        help="Confirm the bounded optimizer-step experiment defined by the config.",
    )
    args = parser.parse_args()
    if not args.confirm_controlled_experiment:
        parser.error("Pass --confirm-controlled-experiment to run the bounded experiment.")
    report = run_controlled_experiment(args.config)
    print(json.dumps({
        "experiment_id": report["experiment_id"],
        "optimizer_steps": report["training"]["optimizer_steps"],
        "baseline_validation_perplexity": report["baseline"]["validation_perplexity"]["perplexity"],
        "experimental_validation_perplexity": report["trained_experiment"]["validation_perplexity"]["perplexity"],
        "checkpoint": report["experimental_checkpoint"]["best_checkpoint_path"],
        "report": str(
            (EXPERIMENTS_ROOT / report["experiment_id"] / "experiment_report.json")
        ),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
