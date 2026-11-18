import unittest

import torch

from src.training_dataset import ChunkDataset, build_training_loaders
from src.training_config import TrainingConfig


class TrainingDatasetTests(unittest.TestCase):
    def test_loads_train_and_validation_only(self):
        config = TrainingConfig(batch_size=2)
        train_loader, validation_loader = build_training_loaders(config)
        self.assertEqual(train_loader.dataset.split, "train")
        self.assertEqual(validation_loader.dataset.split, "validation")
        self.assertFalse(hasattr(train_loader.dataset, "test"))
        self.assertGreater(len(train_loader.dataset), 0)
        self.assertGreater(len(validation_loader.dataset), 0)

    def test_chunk_tensors_have_consistent_shapes_and_types(self):
        dataset = ChunkDataset("train")
        sample = dataset[0]
        self.assertEqual(sample["input_ids"].shape, (256,))
        self.assertEqual(sample["target_ids"].shape, (256,))
        self.assertEqual(sample["attention_mask"].shape, (256,))
        self.assertEqual(sample["loss_mask"].shape, (256,))
        self.assertEqual(sample["input_ids"].dtype, torch.int64)
        self.assertEqual(sample["attention_mask"].dtype, torch.bool)
        self.assertEqual(sample["loss_mask"].dtype, torch.bool)
        self.assertEqual(sample["input_ids"][1:].tolist(), sample["target_ids"][:-1].tolist())
        self.assertEqual(sample["document_id"], dataset.records[0]["document_id"])
        self.assertEqual(sample["chunk_id"], dataset.records[0]["chunk_id"])

    def test_test_split_is_rejected_by_training_dataset(self):
        with self.assertRaisesRegex(ValueError, "only 'train' or 'validation'"):
            ChunkDataset("test")

    def test_sampler_order_is_reproducible(self):
        first, _ = build_training_loaders(TrainingConfig(random_seed=123))
        second, _ = build_training_loaders(TrainingConfig(random_seed=123))
        first_order = list(iter(first.sampler))
        second_order = list(iter(second.sampler))
        self.assertEqual(first_order, second_order)
        self.assertNotEqual(first_order, list(range(len(first_order))))


if __name__ == "__main__":
    unittest.main()
