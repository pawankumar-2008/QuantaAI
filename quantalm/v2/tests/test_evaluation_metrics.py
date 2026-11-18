import unittest

from src.evaluator import (
    aggregate_reference_metrics,
    reference_metrics,
)


class ReferenceMetricTests(unittest.TestCase):
    def test_exact_match_normalizes_unicode_case_and_whitespace(self):
        result = reference_metrics("  CAFÉ \n DBMS ", "cafe\u0301 dbms")
        self.assertEqual(result["exact_match"], 1.0)

    def test_token_metrics_count_duplicate_tokens_as_a_multiset(self):
        result = reference_metrics("cat cat dog", "cat dog dog")
        self.assertAlmostEqual(result["token_precision"], 2 / 3)
        self.assertAlmostEqual(result["token_recall"], 2 / 3)
        self.assertAlmostEqual(result["token_f1"], 2 / 3)

    def test_empty_answers_do_not_gain_token_credit(self):
        empty_generated = reference_metrics("", "an answer")
        self.assertEqual(empty_generated["exact_match"], 0.0)
        self.assertEqual(empty_generated["token_precision"], 0.0)
        self.assertEqual(empty_generated["token_recall"], 0.0)
        self.assertEqual(empty_generated["token_f1"], 0.0)

        both_empty = reference_metrics("", "")
        self.assertEqual(both_empty["exact_match"], 1.0)
        self.assertEqual(both_empty["token_precision"], 0.0)
        self.assertEqual(both_empty["token_recall"], 0.0)
        self.assertEqual(both_empty["token_f1"], 0.0)

    def test_reference_summary_macro_averages_and_marks_missing_data(self):
        summary = aggregate_reference_metrics(
            [
                reference_metrics("a", "a"),
                reference_metrics("wrong", "right"),
            ]
        )
        self.assertEqual(summary["answer_count"], 2)
        self.assertEqual(summary["exact_match_percentage"], 50.0)
        self.assertEqual(aggregate_reference_metrics([])["token_f1_percentage"], None)

    def test_non_string_answers_are_rejected(self):
        with self.assertRaises(TypeError):
            reference_metrics(None, "answer")


if __name__ == "__main__":
    unittest.main()
