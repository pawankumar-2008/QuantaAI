import unittest

from src.evaluator import (
    HUMAN_SCORE_FIELDS,
    summarize_human_reviews,
    validate_human_review,
    weighted_human_quality_score,
)


def valid_review(question_id="q1", reviewer_id="reviewer-a"):
    return {
        "question_id": question_id,
        "factual_accuracy_score": 5,
        "relevance_score": 4,
        "coherence_score": 3,
        "instruction_following_score": 2,
        "explanation": "Reviewed against the supplied source and question.",
        "reviewer_id": reviewer_id,
        "reviewed": True,
    }


class HumanEvaluationTests(unittest.TestCase):
    def test_valid_review_scores_map_to_percentages(self):
        review = valid_review()
        self.assertEqual(set(HUMAN_SCORE_FIELDS), {
            "factual_accuracy_score",
            "relevance_score",
            "coherence_score",
            "instruction_following_score",
        })
        validate_human_review(review)
        summary = summarize_human_reviews([review], valid_question_ids={"q1"})
        self.assertEqual(summary["valid_review_count"], 1)
        self.assertEqual(summary["scores"]["factual_accuracy_percentage"], 100.0)
        self.assertEqual(summary["scores"]["relevance_percentage"], 75.0)
        self.assertEqual(summary["scores"]["coherence_percentage"], 50.0)
        self.assertEqual(summary["scores"]["instruction_following_percentage"], 25.0)

    def test_invalid_scores_and_incomplete_reviews_are_not_counted(self):
        invalid = valid_review()
        invalid["factual_accuracy_score"] = 6
        incomplete = {
            "question_id": "",
            "reviewed": False,
        }
        summary = summarize_human_reviews([invalid, incomplete])
        self.assertEqual(summary["valid_review_count"], 0)
        self.assertEqual(summary["invalid_review_count"], 1)
        self.assertEqual(summary["incomplete_review_count"], 1)

    def test_boolean_is_not_accepted_as_numeric_score(self):
        review = valid_review()
        review["factual_accuracy_score"] = True
        with self.assertRaisesRegex(ValueError, "integer score"):
            validate_human_review(review)

    def test_unknown_question_and_duplicate_same_reviewer_are_rejected(self):
        duplicate = valid_review()
        summary = summarize_human_reviews(
            [valid_review(), duplicate, valid_review("unknown", "reviewer-b")],
            valid_question_ids={"q1"},
        )
        self.assertEqual(summary["valid_review_count"], 1)
        self.assertEqual(summary["invalid_review_count"], 2)

    def test_weighted_score_is_unavailable_without_human_reviews(self):
        summary = summarize_human_reviews([])
        score = weighted_human_quality_score(summary, repetition_quality_percentage=100.0)
        self.assertFalse(score["available"])
        self.assertIsNone(score["overall_human_quality_percentage"])

    def test_weighted_score_uses_only_available_metrics_and_normalizes_weights(self):
        summary = summarize_human_reviews([valid_review()])
        score = weighted_human_quality_score(summary, repetition_quality_percentage=None)
        self.assertTrue(score["available"])
        self.assertAlmostEqual(
            sum(score["effective_weights"].values()),
            1.0,
        )
        self.assertNotIn("repetition_quality_percentage", score["components"])
        self.assertAlmostEqual(
            score["overall_human_quality_percentage"],
            (100 * 0.35 + 75 * 0.25 + 50 * 0.15 + 25 * 0.15) / 0.9,
        )


if __name__ == "__main__":
    unittest.main()
