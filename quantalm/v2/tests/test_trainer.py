import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.checkpoint_manager import CheckpointManager
from src.model import ModelConfig, QuantaLMV2
from src.trainer import Trainer
from src.trainer import run_training
from src.training_config import TrainingConfig
from src.training_dataset import DeterministicEpochSampler


V2_ROOT = Path(__file__).resolve().parents[1]


class SyntheticDataset(Dataset):
    split = "train"

    def __init__(self):
        self.examples = [
            {
                "input_ids": torch.tensor([1, 2, 3, 256, 256, 256], dtype=torch.long),
                "target_ids": torch.tensor([2, 3, 4, 256, 256, 256], dtype=torch.long),
                "attention_mask": torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool),
                "loss_mask": torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool),
            },
            {
                "input_ids": torch.tensor([5, 6, 7, 8, 256, 256], dtype=torch.long),
                "target_ids": torch.tensor([6, 7, 8, 9, 256, 256], dtype=torch.long),
                "attention_mask": torch.tensor([1, 1, 1, 1, 0, 0], dtype=torch.bool),
                "loss_mask": torch.tensor([1, 1, 1, 1, 0, 0], dtype=torch.bool),
            },
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def loaders(seed=42):
    train_dataset = SyntheticDataset()
    validation_dataset = SyntheticDataset()
    validation_dataset.split = "validation"
    train_sampler = DeterministicEpochSampler(train_dataset, seed)
    train_loader = DataLoader(train_dataset, batch_size=1, sampler=train_sampler)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
    return train_loader, validation_loader


def small_model():
    return QuantaLMV2(
        ModelConfig(
            vocab_size=2048,
            context_length=256,
            embedding_dim=16,
            num_layers=1,
            num_heads=2,
            feed_forward_dim=32,
            dropout=0.0,
        )
    )


def small_config(seed=42):
    return TrainingConfig(
        batch_size=1,
        learning_rate=1e-3,
        weight_decay=0.01,
        epochs=1,
        warmup_steps=1,
        gradient_clip_norm=1.0,
        random_seed=seed,
        evaluation_interval=10,
        checkpoint_interval=10,
        early_stopping_patience=2,
        device="cpu",
    )


def new_trainer(temp_path, seed=42):
    train_loader, validation_loader = loaders(seed)
    config = small_config(seed)
    torch.manual_seed(seed)
    return Trainer(
        train_loader,
        validation_loader,
        config=config,
        model=small_model(),
        checkpoint_manager=CheckpointManager(temp_path / "checkpoints"),
        metrics_dir=temp_path / "metrics",
        tokenizer_hash=hashlib.sha256(b"synthetic-tokenizer").hexdigest(),
        dataset_manifest_hash=hashlib.sha256(b"synthetic-manifest").hexdigest(),
    )


class TrainerTests(unittest.TestCase):
    def test_loss_uses_mask_and_normalizes_by_valid_targets(self):
        with tempfile.TemporaryDirectory(
            prefix="quanta-v2-trainer-test-",
            dir=V2_ROOT / "checkpoints",
        ) as directory:
            trainer = new_trainer(Path(directory))
            logits = torch.zeros((1, 3, 2048), dtype=torch.float32)
            targets = torch.tensor([[2, 999, 256]], dtype=torch.long)
            mask = torch.tensor([[True, False, False]])
            loss_sum, valid = trainer._loss_sum(logits, targets, mask)
            self.assertEqual(valid, 1)
            self.assertAlmostEqual(float(loss_sum), torch.log(torch.tensor(2048.0)).item(), places=5)

    def test_padding_targets_do_not_affect_loss(self):
        with tempfile.TemporaryDirectory(
            prefix="quanta-v2-trainer-test-",
            dir=V2_ROOT / "checkpoints",
        ) as directory:
            trainer = new_trainer(Path(directory))
            logits = torch.randn((1, 3, 2048))
            mask = torch.tensor([[True, False, False]])
            first, count_first = trainer._loss_sum(
                logits,
                torch.tensor([[2, 3, 256]]),
                mask,
            )
            second, count_second = trainer._loss_sum(
                logits,
                torch.tensor([[2, 2047, 1]]),
                mask,
            )
            self.assertEqual(count_first, count_second)
            torch.testing.assert_close(first, second)

    def test_optimizer_scheduler_and_cpu_step(self):
        with tempfile.TemporaryDirectory(
            prefix="quanta-v2-trainer-test-",
            dir=V2_ROOT / "checkpoints",
        ) as directory:
            trainer = new_trainer(Path(directory))
            parameter_before = next(trainer.model.parameters()).detach().clone()
            learning_rate_before = trainer.optimizer.param_groups[0]["lr"]
            result = trainer.train_step(next(iter(trainer.train_loader)))
            trainer.train_step(next(iter(trainer.train_loader)))
            parameter_after = next(trainer.model.parameters()).detach()
            self.assertTrue(torch.isfinite(torch.tensor(result["loss"])))
            self.assertTrue(torch.isfinite(torch.tensor(result["gradient_norm"])))
            self.assertEqual(result["valid_tokens"], 3)
            self.assertFalse(torch.equal(parameter_before, parameter_after))
            self.assertNotEqual(trainer.optimizer.param_groups[0]["lr"], learning_rate_before)
            self.assertEqual(trainer.device.type, "cpu")
            self.assertTrue(
                all(
                    parameter.grad is None or torch.all(torch.isfinite(parameter.grad))
                    for parameter in trainer.model.parameters()
                )
            )

    def test_initialization_is_reproducible(self):
        with tempfile.TemporaryDirectory(
            prefix="quanta-v2-trainer-determinism-",
            dir=V2_ROOT / "checkpoints",
        ) as directory:
            first = new_trainer(Path(directory) / "first")
            second = new_trainer(Path(directory) / "second")
            for name, parameter in first.model.state_dict().items():
                torch.testing.assert_close(parameter, second.model.state_dict()[name])

    def test_short_cpu_fit_records_metrics_and_checkpoints(self):
        with tempfile.TemporaryDirectory(
            prefix="quanta-v2-trainer-test-",
            dir=V2_ROOT / "checkpoints",
        ) as directory:
            trainer = new_trainer(Path(directory))
            history = trainer.fit()
            self.assertEqual(len(history), 1)
            self.assertGreater(history[0]["training_tokens_processed"], 0)
            self.assertGreater(history[0]["validation_tokens_processed"], 0)
            self.assertTrue((Path(directory) / "checkpoints" / "latest.pt").is_file())
            self.assertTrue((Path(directory) / "checkpoints" / "best.pt").is_file())
            self.assertTrue((Path(directory) / "checkpoints" / "resume.pt").is_file())
            metrics = json.loads(
                (Path(directory) / "metrics" / "training_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(metrics["test_split_evaluated"])
            self.assertTrue((Path(directory) / "metrics" / "training_metrics.csv").is_file())

    def test_resume_matches_uninterrupted_synthetic_training(self):
        with tempfile.TemporaryDirectory(
            prefix="quanta-v2-trainer-resume-",
            dir=V2_ROOT / "checkpoints",
        ) as directory:
            root = Path(directory)
            torch.manual_seed(42)
            uninterrupted = Trainer(
                *loaders(),
                config=small_config(),
                model=small_model(),
                checkpoint_manager=CheckpointManager(root / "full-checkpoints"),
                metrics_dir=root / "full-metrics",
                tokenizer_hash=hashlib.sha256(b"synthetic-tokenizer").hexdigest(),
                dataset_manifest_hash=hashlib.sha256(b"synthetic-manifest").hexdigest(),
            )
            uninterrupted.fit()

            interrupted = new_trainer(root / "resumed")
            interrupted.fit(max_steps=1)
            resumed = new_trainer(root / "resumed")
            resumed.resume("resume")
            self.assertEqual(resumed.global_step, 1)
            self.assertEqual(resumed.batch_in_epoch, 1)
            resumed.fit()
            self.assertEqual(resumed.global_step, uninterrupted.global_step)
            for name, parameter in uninterrupted.model.state_dict().items():
                torch.testing.assert_close(parameter, resumed.model.state_dict()[name])

    def test_resume_after_partial_final_batch_completes_epoch(self):
        with tempfile.TemporaryDirectory(
            prefix="quanta-v2-trainer-partial-resume-",
            dir=V2_ROOT / "checkpoints",
        ) as directory:
            root = Path(directory)
            dataset = SyntheticDataset()
            dataset.examples.append(dataset.examples[-1])
            validation_dataset = SyntheticDataset()
            validation_dataset.split = "validation"
            config = TrainingConfig(
                batch_size=2,
                learning_rate=1e-3,
                epochs=1,
                warmup_steps=1,
                random_seed=42,
                evaluation_interval=10,
                checkpoint_interval=10,
                early_stopping_patience=2,
                device="cpu",
            )
            train_loader = DataLoader(
                dataset,
                batch_size=2,
                sampler=DeterministicEpochSampler(dataset, 42),
            )
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=2,
                shuffle=False,
            )
            torch.manual_seed(42)
            interrupted = Trainer(
                train_loader,
                validation_loader,
                config=config,
                model=small_model(),
                checkpoint_manager=CheckpointManager(root / "checkpoints"),
                metrics_dir=root / "metrics",
                tokenizer_hash=hashlib.sha256(b"synthetic-tokenizer").hexdigest(),
                dataset_manifest_hash=hashlib.sha256(b"synthetic-manifest").hexdigest(),
            )
            interrupted.fit(max_steps=2)
            self.assertEqual(interrupted.batch_in_epoch, 2)

            resumed_train = DataLoader(
                dataset,
                batch_size=2,
                sampler=DeterministicEpochSampler(dataset, 42),
            )
            resumed_validation = DataLoader(
                validation_dataset,
                batch_size=2,
                shuffle=False,
            )
            torch.manual_seed(42)
            resumed = Trainer(
                resumed_train,
                resumed_validation,
                config=config,
                model=small_model(),
                checkpoint_manager=interrupted.checkpoint_manager,
                metrics_dir=root / "metrics-resumed",
                tokenizer_hash=hashlib.sha256(b"synthetic-tokenizer").hexdigest(),
                dataset_manifest_hash=hashlib.sha256(b"synthetic-manifest").hexdigest(),
            )
            resumed.resume("resume")
            history = resumed.fit()
            self.assertEqual(resumed.global_step, 2)
            self.assertEqual(resumed.current_epoch, 1)
            self.assertEqual(history[-1]["training_tokens_processed"], 11)

    def test_resume_rejects_configuration_mismatch(self):
        with tempfile.TemporaryDirectory(
            prefix="quanta-v2-trainer-test-",
            dir=V2_ROOT / "checkpoints",
        ) as directory:
            root = Path(directory)
            original = new_trainer(root)
            original.fit(max_steps=1)
            incompatible = new_trainer(root, seed=43)
            with self.assertRaisesRegex(ValueError, "training_config"):
                incompatible.resume("resume")

    def test_full_dataset_training_is_guarded(self):
        with self.assertRaisesRegex(RuntimeError, "--confirm-training"):
            run_training()

    def test_non_finite_gradients_are_rejected(self):
        with tempfile.TemporaryDirectory(
            prefix="quanta-v2-trainer-test-",
            dir=V2_ROOT / "checkpoints",
        ) as directory:
            trainer = new_trainer(Path(directory))
            batch = next(iter(trainer.train_loader))
            trainer.model.register_forward_hook(
                lambda _module, _inputs, output: output * torch.tensor(float("nan"))
            )
            with self.assertRaisesRegex(FloatingPointError, "loss is not finite"):
                trainer.train_step(batch)


if __name__ == "__main__":
    unittest.main()
