import tempfile
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.checkpoint import load_checkpoint, save_checkpoint
from src.model import ModelConfig, QuantaLMV2


class CheckpointTests(unittest.TestCase):
    def test_save_and_load_round_trip(self):
        model = QuantaLMV2(ModelConfig(dropout=0.0))
        model.eval()
        inputs = torch.randint(0, 2048, (2, 5))
        with torch.no_grad():
            expected = model(inputs)
        checkpoint_root = Path(__file__).resolve().parents[1] / "checkpoints"
        with tempfile.TemporaryDirectory(
            prefix="quanta-v2-checkpoint-test-",
            dir=checkpoint_root,
        ) as directory:
            checkpoint_path = Path(directory) / "model.pt"
            save_checkpoint(model, checkpoint_path)
            restored = QuantaLMV2(model.config)
            metadata = load_checkpoint(restored, checkpoint_path)
            restored.eval()
            with torch.no_grad():
                actual = restored(inputs)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(metadata["architecture_version"], "2.0")
        self.assertEqual(metadata["vocab_size"], 2048)
        self.assertEqual(metadata["context_length"], 256)
        self.assertTrue(metadata["created_at"])

    def test_refuses_to_overwrite_existing_checkpoint_by_default(self):
        model = QuantaLMV2(ModelConfig(dropout=0.0))
        checkpoint_root = Path(__file__).resolve().parents[1] / "checkpoints"
        with tempfile.TemporaryDirectory(
            prefix="quanta-v2-checkpoint-test-",
            dir=checkpoint_root,
        ) as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(model, path)
            with self.assertRaises(FileExistsError):
                save_checkpoint(model, path)

    def test_rejects_path_outside_v2_checkpoint_directory(self):
        model = QuantaLMV2(ModelConfig(dropout=0.0))
        with tempfile.TemporaryDirectory() as directory:
            outside_path = Path(directory) / "outside.pt"
            with self.assertRaisesRegex(ValueError, "inside the V2 checkpoint"):
                save_checkpoint(model, outside_path)

    def test_rejects_incompatible_configuration(self):
        model = QuantaLMV2(ModelConfig(dropout=0.0))
        incompatible = QuantaLMV2(
            ModelConfig(dropout=0.0, context_length=128)
        )
        checkpoint_root = Path(__file__).resolve().parents[1] / "checkpoints"
        with tempfile.TemporaryDirectory(
            prefix="quanta-v2-checkpoint-test-",
            dir=checkpoint_root,
        ) as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(model, path)
            with self.assertRaisesRegex(ValueError, "configuration is incompatible"):
                load_checkpoint(incompatible, path)


if __name__ == "__main__":
    unittest.main()
