import hashlib
import tempfile
import unittest
from pathlib import Path

import torch

from src.checkpoint_manager import CheckpointManager
from src.model import ModelConfig, QuantaLMV2
from src.trainer import Trainer
from src.training_config import TrainingConfig
from src.training_dataset import DeterministicEpochSampler


V2_ROOT = Path(__file__).resolve().parents[1]


class SyntheticDataset(torch.utils.data.Dataset):
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


def small_config():
    return TrainingConfig(
        batch_size=1,
        learning_rate=1e-3,
        epochs=1,
        warmup_steps=1,
        random_seed=42,
        evaluation_interval=10,
        checkpoint_interval=10,
        device="cpu",
    )


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


def loaders():
    train_dataset = SyntheticDataset()
    validation_dataset = SyntheticDataset()
    validation_dataset.split = "validation"
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        sampler=DeterministicEpochSampler(train_dataset, 42),
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
    )
    return train_loader, validation_loader


def new_trainer(directory):
    torch.manual_seed(42)
    train_loader, validation_loader = loaders()
    checkpoint_manager = CheckpointManager(Path(directory) / "checkpoints")
    return Trainer(
        train_loader,
        validation_loader,
        config=small_config(),
        model=small_model(),
        checkpoint_manager=checkpoint_manager,
        metrics_dir=Path(directory) / "metrics",
        tokenizer_hash=hashlib.sha256(b"synthetic-tokenizer").hexdigest(),
        dataset_manifest_hash=hashlib.sha256(b"synthetic-manifest").hexdigest(),
    )


class CheckpointManagerTests(unittest.TestCase):
    def test_checkpoint_round_trip_contains_training_state(self):
        with tempfile.TemporaryDirectory(
            prefix="quanta-v2-checkpoint-manager-test-",
            dir=V2_ROOT / "checkpoints",
        ) as directory:
            trainer = new_trainer(Path(directory))
            trainer.train_step(next(iter(trainer.train_loader)))
            path = trainer._save_checkpoint("resume")
            payload = trainer.checkpoint_manager.load("resume")
            self.assertTrue(path.is_file())
            for field in (
                "model_state",
                "optimizer_state",
                "scheduler_state",
                "epoch",
                "global_step",
                "training_config",
                "best_validation_loss",
                "random_seed",
                "tokenizer_hash",
                "dataset_manifest_hash",
            ):
                self.assertIn(field, payload)

            restored = new_trainer(Path(directory) / "restored")
            restored.checkpoint_manager = trainer.checkpoint_manager
            restored.resume("resume")
            self.assertEqual(restored.global_step, 1)
            self.assertEqual(restored.scheduler.state_dict(), trainer.scheduler.state_dict())
            for name, value in trainer.model.state_dict().items():
                torch.testing.assert_close(value, restored.model.state_dict()[name])

    def test_checkpoint_kind_and_path_are_restricted(self):
        with tempfile.TemporaryDirectory(
            prefix="quanta-v2-checkpoint-manager-test-",
            dir=V2_ROOT / "logs",
        ) as directory:
            with self.assertRaisesRegex(ValueError, "inside quantalm/v2/checkpoints"):
                CheckpointManager(Path(directory) / "outside")
        manager = CheckpointManager()
        with self.assertRaisesRegex(ValueError, "Checkpoint kind"):
            manager.path_for("test")

    def test_load_missing_checkpoint_reports_error(self):
        with tempfile.TemporaryDirectory(
            prefix="quanta-v2-checkpoint-manager-test-",
            dir=V2_ROOT / "checkpoints",
        ) as directory:
            manager = CheckpointManager(directory)
            with self.assertRaises(FileNotFoundError):
                manager.load("latest")


if __name__ == "__main__":
    unittest.main()
