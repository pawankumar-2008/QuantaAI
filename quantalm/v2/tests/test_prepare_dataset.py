import unittest
from collections import Counter
import sys
from pathlib import Path

V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))

from src.prepare_dataset import assign_splits, count_words


class PrepareDatasetTests(unittest.TestCase):
    def test_splits_document_groups_with_90_10_train_validation_targets(self):
        documents = [
            {
                "doc_id": f"doc-{index:03d}",
                "leakage_group_id": f"group-{index:03d}",
                "category": "Computer Science",
            }
            for index in range(100)
        ]

        splits = assign_splits(documents)

        self.assertEqual(
            {name: len(records) for name, records in splits.items()},
            {"train": 81, "validation": 9, "test": 10},
        )
        self.assertAlmostEqual(
            len(splits["train"]) / (len(splits["train"]) + len(splits["validation"])),
            0.9,
        )
        self.assertEqual(
            Counter(record["split"] for records in splits.values() for record in records),
            Counter({"train": 81, "validation": 9, "test": 10}),
        )

    def test_related_documents_remain_in_one_split(self):
        documents = [
            {
                "doc_id": f"doc-{index:03d}",
                "leakage_group_id": f"group-{index // 2:03d}",
                "category": "Computer Science",
            }
            for index in range(20)
        ]

        splits = assign_splits(documents)

        split_by_group = {
            record["leakage_group_id"]: record["split"]
            for records in splits.values()
            for record in records
        }
        for records in splits.values():
            for record in records:
                self.assertEqual(
                    split_by_group[record["leakage_group_id"]],
                    record["split"],
                )

    def test_word_count_uses_unicode_word_runs(self):
        self.assertEqual(count_words("QuantaLM V2: café, next-token."), 5)

    def test_insufficient_independent_groups_is_reported(self):
        with self.assertRaisesRegex(ValueError, "three independent document groups"):
            assign_splits(
                [
                    {"doc_id": "one", "leakage_group_id": "same"},
                    {"doc_id": "two", "leakage_group_id": "same"},
                ]
            )


if __name__ == "__main__":
    unittest.main()
