import unittest

from src.evaluator import aggregate_generation_quality, generation_quality


class GenerationQualityTests(unittest.TestCase):
    def test_repeated_token_and_ngram_ratios_are_reported(self):
        result = generation_quality([10, 10, 10, 11, 11, 259], "xxx", eos_token_id=259)
        self.assertEqual(result["answer_length_tokens"], 5)
        self.assertAlmostEqual(result["repeated_token_ratio"], 3 / 5)
        self.assertTrue(result["eos_terminated"])
        self.assertGreater(result["repeated_ngram_ratio"], 0.0)
        self.assertTrue(result["repetition_patterns"])

    def test_empty_response_and_missing_eos_are_detected(self):
        result = generation_quality([], "", eos_token_id=259)
        self.assertTrue(result["empty_response"])
        self.assertFalse(result["eos_terminated"])
        self.assertEqual(result["answer_length_tokens"], 0)
        self.assertEqual(result["invalid_token_rate"], 0.0)

    def test_invalid_token_rate_ignores_eos_but_counts_invalid_answer_ids(self):
        result = generation_quality(
            [4, 5000, 259],
            "answer",
            eos_token_id=259,
            vocab_size=2048,
        )
        self.assertEqual(result["invalid_token_count"], 1)
        self.assertEqual(result["invalid_token_rate"], 0.5)

    def test_empty_aggregate_marks_metrics_unavailable(self):
        summary = aggregate_generation_quality([])
        self.assertEqual(summary["sample_count"], 0)
        self.assertIsNone(summary["empty_response_rate_percentage"])
        self.assertIsNone(summary["repetition_quality_percentage"])

    def test_aggregate_rates_and_repetition_quality(self):
        samples = [
            {"quality": generation_quality([1, 1, 259], "repeat", eos_token_id=259)},
            {"quality": generation_quality([2, 3], "", eos_token_id=259)},
        ]
        summary = aggregate_generation_quality(samples)
        self.assertEqual(summary["empty_response_rate_percentage"], 50.0)
        self.assertEqual(summary["eos_termination_rate_percentage"], 50.0)
        self.assertIsNotNone(summary["repetition_quality_percentage"])


if __name__ == "__main__":
    unittest.main()
