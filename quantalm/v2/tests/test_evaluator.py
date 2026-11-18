import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.checkpoint import save_checkpoint
from src.compare_checkpoints import summarize_comparison
from src.evaluator import (
    EVALUATION_CATEGORIES,
    EVALUATION_REPORT_DIR,
    Evaluator,
    GenerationConfig,
    _load_chunk_records,
    load_evaluation_dataset,
    _tokenizer_artifact_hashes,
)
from src.model import ModelConfig, QuantaLMV2


V2_ROOT = Path(__file__).resolve().parents[1]


def evaluation_record():
    return {
        "question_id": "validation-question-1",
        "category": "Computer Science",
        "question": "Explain a computer-science concept from the source.",
        "reference_answer": "A source-grounded reference awaiting review.",
        "acceptable_key_points": [],
        "source_document_id": "validation-document",
        "source_document_hash": hashlib.sha256(b"source").hexdigest(),
        "source_text_reference": "data/splits/validation/dataset.jsonl",
        "difficulty": "introductory",
        "human_verified": False,
    }


class EvaluatorTests(unittest.TestCase):
    def test_checkpoint_comparison_reports_validation_metric_deltas(self):
        shared = {
            "split": "validation",
            "evaluation_dataset_sha256": "a" * 64,
            "perplexity_chunk_dataset_sha256": "b" * 64,
            "tokenizer_sha256": "c" * 64,
        }
        previous = {
            **shared,
            "model_label": "previous",
            "checkpoint_sha256": "d" * 64,
            "perplexity": {
                "perplexity": 10.0,
                "average_negative_log_likelihood": 2.302585,
            },
        }
        candidate = {
            **shared,
            "model_label": "candidate",
            "checkpoint_sha256": "e" * 64,
            "perplexity": {
                "perplexity": 8.0,
                "average_negative_log_likelihood": 2.079442,
            },
        }

        comparison = summarize_comparison(previous, candidate)

        self.assertEqual(comparison["candidate_minus_previous_perplexity"], -2.0)
        self.assertEqual(comparison["relative_perplexity_change_percentage"], -20.0)

    def test_checkpoint_comparison_rejects_different_tokenizers(self):
        previous = {
            "split": "validation",
            "evaluation_dataset_sha256": "a" * 64,
            "perplexity_chunk_dataset_sha256": "b" * 64,
            "tokenizer_sha256": "c" * 64,
        }
        candidate = {**previous, "tokenizer_sha256": "d" * 64}
        with self.assertRaisesRegex(ValueError, "tokenizer_sha256"):
            summarize_comparison(previous, candidate)

    def test_categories_and_empty_review_dataset_are_explicit(self):
        self.assertEqual(len(EVALUATION_CATEGORIES), 6)
        self.assertTrue((EVALUATION_REPORT_DIR.parent.parent / "evaluation").exists())

    def test_dataset_loader_accepts_valid_candidate_but_no_verified_reference(self):
        with tempfile.TemporaryDirectory(
            prefix="quanta-v2-eval-data-test-",
            dir=V2_ROOT / "logs",
        ) as directory:
            path = Path(directory) / "questions.jsonl"
            path.write_text(
                json.dumps(evaluation_record(), ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            records = load_evaluation_dataset(path)
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0]["human_verified"])

    def test_dataset_loader_rejects_invalid_record_and_duplicate_id(self):
        with tempfile.TemporaryDirectory(
            prefix="quanta-v2-eval-data-test-",
            dir=V2_ROOT / "logs",
        ) as directory:
            path = Path(directory) / "questions.jsonl"
            invalid = evaluation_record()
            invalid["source_document_hash"] = "not-a-hash"
            path.write_text(json.dumps(invalid) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source_document_hash"):
                load_evaluation_dataset(path)

            record = evaluation_record()
            path.write_text(
                json.dumps(record) + "\n" + json.dumps(record) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_evaluation_dataset(path)

    def test_test_split_requires_explicit_approval_before_reading_any_test_data(self):
        with self.assertRaisesRegex(PermissionError, "disabled by default"):
            _load_chunk_records("test", allow_test_evaluation=False)
        evaluator = Evaluator(device="cpu", random_seed=2)
        with self.assertRaisesRegex(PermissionError, "disabled by default"):
            evaluator.evaluate("test")

    def test_checkpoint_hash_is_recorded_and_safe_model_checkpoint_loads(self):
        with tempfile.TemporaryDirectory(
            prefix="quanta-v2-evaluator-checkpoint-test-",
            dir=V2_ROOT / "checkpoints",
        ) as directory:
            model = QuantaLMV2()
            checkpoint_path = save_checkpoint(
                model,
                Path(directory) / "selected.pt",
            )
            expected_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            evaluator = Evaluator(
                checkpoint_path=checkpoint_path,
                device="cpu",
                random_seed=13,
                generation_config=GenerationConfig(max_new_tokens=1, top_k=1),
            )
            training_checkpoint_path = Path(directory) / "trained.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model.config.to_dict(),
                    "global_step": 1,
                    "tokenizer_hash": _tokenizer_artifact_hashes(
                        V2_ROOT / "tokenizer"
                    )[0],
                },
                training_checkpoint_path,
            )
            trained_evaluator = Evaluator(
                checkpoint_path=training_checkpoint_path,
                device="cpu",
                random_seed=13,
            )
        self.assertEqual(evaluator.checkpoint_sha256, expected_hash)
        self.assertEqual(
            evaluator.model_label,
            "checkpoint_training_status_unverified",
        )
        self.assertEqual(trained_evaluator.model_label, "trained_checkpoint")

    def test_checkpoint_architecture_is_used_for_custom_model_dimensions(self):
        with tempfile.TemporaryDirectory(
            prefix="quanta-v2-evaluator-config-test-",
            dir=V2_ROOT / "checkpoints",
        ) as directory:
            config = ModelConfig(
                embedding_dim=16,
                num_layers=1,
                num_heads=2,
                feed_forward_dim=32,
                dropout=0.0,
            )
            model = QuantaLMV2(config)
            checkpoint = save_checkpoint(model, Path(directory) / "custom.pt")
            evaluator = Evaluator(checkpoint_path=checkpoint, device="cpu")

        self.assertEqual(evaluator.model.config, config)

    def test_untrained_baseline_question_generation_is_seeded(self):
        evaluator = Evaluator(
            device="cpu",
            random_seed=31,
            generation_config=GenerationConfig(max_new_tokens=2, top_k=1, random_seed=31),
        )
        question = evaluation_record()
        first = evaluator._generate_answer(question, 0)
        second = evaluator._generate_answer(question, 0)
        self.assertEqual(first["generated_token_ids"], second["generated_token_ids"])
        self.assertFalse(first["human_verified_reference"])
        self.assertNotIn("reference_metrics", first)
        self.assertEqual(evaluator.model_label, "untrained_baseline")
        self.assertIsNone(evaluator.checkpoint_sha256)

    def test_validation_integration_writes_all_reports_without_test_access(self):
        evaluator = Evaluator(
            device="cpu",
            random_seed=5,
            generation_config=GenerationConfig(max_new_tokens=1, top_k=1, random_seed=5),
        )
        with tempfile.TemporaryDirectory(
            prefix="quanta-v2-eval-report-test-",
            dir=V2_ROOT / "logs",
        ) as directory:
            result = evaluator.evaluate(
                "validation",
                benchmark_warmup_runs=0,
                benchmark_measured_runs=1,
                output_dir=directory,
            )
            output = Path(directory)
            for name in (
                "evaluation_summary.json",
                "evaluation_metrics.csv",
                "generation_samples.jsonl",
                "performance_report.json",
                "human_review_summary.json",
                "evaluation_report.md",
            ):
                self.assertTrue((output / name).is_file(), name)
        self.assertEqual(result["split"], "validation")
        self.assertEqual(result["model_label"], "untrained_baseline")
        self.assertEqual(result["question_count"], 0)
        self.assertEqual(result["valid_answer_count"], 0)
        self.assertEqual(result["perplexity"]["chunk_count"], 38)
        self.assertGreater(result["perplexity"]["evaluated_token_count"], 0)
        self.assertIsNone(result["overall_human_quality_score"]["overall_human_quality_percentage"])
        self.assertIsNone(result["reference_metrics"]["exact_match_percentage"])
        self.assertEqual(result["device"]["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
