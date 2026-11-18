import json
import tempfile
import unittest
from pathlib import Path

from src.controlled_experiment import (
    DEFAULT_CONFIG,
    _source_audit,
    load_experiment_config,
)


V2_ROOT = Path(__file__).resolve().parents[1]


class ControlledExperimentConfigTests(unittest.TestCase):
    def test_checked_in_experiment_is_cpu_bounded_and_test_isolated(self):
        config = load_experiment_config(DEFAULT_CONFIG)
        self.assertEqual(config["max_steps"], 3)
        self.assertEqual(config["training"]["device"], "cpu")
        self.assertEqual(config["training"]["evaluation_interval"], 1)
        self.assertFalse(config["data_policy"]["test_split_used"])
        self.assertFalse(config["data_policy"]["external_data_used"])

    def test_full_training_step_count_is_rejected(self):
        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        config["max_steps"] = 11
        with tempfile.TemporaryDirectory(
            prefix="quanta-v2-experiment-config-test-",
            dir=V2_ROOT / "logs",
        ) as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "between 1 and 10"):
                load_experiment_config(path)

    def test_test_or_external_data_policy_cannot_be_enabled(self):
        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        config["data_policy"]["test_split_used"] = True
        with tempfile.TemporaryDirectory(
            prefix="quanta-v2-experiment-config-test-",
            dir=V2_ROOT / "logs",
        ) as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prohibit test"):
                load_experiment_config(path)

    def test_dataset_audit_reports_unknown_license_and_deduplication(self):
        audit = _source_audit()
        self.assertEqual(audit["licensing"], "unknown for all source documents")
        self.assertEqual(audit["document_counts"], {
            "train": 11,
            "validation": 1,
            "test": 1,
        })
        self.assertEqual(audit["split_document_overlap_counts"], {
            "train_validation": 0,
            "train_test": 0,
            "validation_test": 0,
        })
        duplicates = audit["corpus_duplicate_content_summary"]
        self.assertEqual(duplicates["exact_duplicate_documents_removed"], 2)
        self.assertEqual(duplicates["near_duplicate_documents_removed"], 0)
        self.assertEqual(duplicates["shared_passage_count"], 8)


if __name__ == "__main__":
    unittest.main()
