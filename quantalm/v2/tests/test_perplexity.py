import math
import unittest

import torch
from torch import nn

from src.evaluator import calculate_perplexity


class FixedLogitModel(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor(logits, dtype=torch.float32))

    def forward(self, input_ids):
        return self.logits.expand(input_ids.shape[0], input_ids.shape[1], -1)


class PerplexityTests(unittest.TestCase):
    def test_uniform_logits_have_perplexity_equal_to_vocabulary_size(self):
        model = FixedLogitModel([0.0, 0.0, 0.0, 0.0])
        record = {
            "input_ids": [1, 2, 256],
            "target_ids": [2, 3, 256],
            "loss_mask": [1, 1, 0],
        }
        result = calculate_perplexity(
            model,
            [record],
            vocab_size=4,
            context_length=3,
        )
        self.assertAlmostEqual(result["perplexity"], 4.0, places=5)
        self.assertEqual(result["evaluated_token_count"], 2)
        self.assertEqual(result["chunk_count"], 1)

    def test_uses_existing_shifted_targets_without_shifting_again(self):
        class AlignedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(()))

            def forward(self, input_ids):
                logits = torch.zeros(
                    input_ids.shape[0],
                    input_ids.shape[1],
                    4,
                )
                logits[:, 0, 2] = 100
                logits[:, 1, 3] = 100
                return logits

        result = calculate_perplexity(
            AlignedModel(),
            [{"input_ids": [1, 2], "target_ids": [2, 3], "loss_mask": [1, 1]}],
            vocab_size=4,
            context_length=2,
        )
        self.assertLess(result["average_negative_log_likelihood"], 1e-6)
        self.assertAlmostEqual(result["perplexity"], 1.0, places=5)

    def test_padding_positions_are_excluded_from_nll(self):
        model = FixedLogitModel([0.0, 0.0, 0.0, 0.0])
        two_valid = {
            "input_ids": [1, 2, 256],
            "target_ids": [1, 2, 256],
            "loss_mask": [1, 1, 0],
        }
        result = calculate_perplexity(
            model,
            [two_valid],
            vocab_size=4,
            context_length=3,
        )
        self.assertEqual(result["evaluated_token_count"], 2)
        self.assertAlmostEqual(result["average_negative_log_likelihood"], math.log(4), places=5)

    def test_perplexity_overflow_is_reported_without_infinity(self):
        model = FixedLogitModel([0.0, 0.0, -1000.0, 0.0])
        result = calculate_perplexity(
            model,
            [{"input_ids": [1], "target_ids": [2], "loss_mask": [1]}],
            vocab_size=4,
            context_length=1,
        )
        self.assertTrue(result["perplexity_overflow"])
        self.assertIsNone(result["perplexity"])

    def test_empty_or_invalid_masked_data_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no valid targets"):
            calculate_perplexity(
                FixedLogitModel([0.0, 0.0]),
                [{"input_ids": [256], "target_ids": [256], "loss_mask": [0]}],
                vocab_size=2,
                context_length=1,
            )
        with self.assertRaisesRegex(ValueError, "non-PAD"):
            calculate_perplexity(
                FixedLogitModel([0.0, 0.0]),
                [{"input_ids": [0], "target_ids": [0], "loss_mask": [0]}],
                vocab_size=2,
                context_length=1,
            )

    def test_invalid_token_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid token IDs"):
            calculate_perplexity(
                FixedLogitModel([0.0, 0.0]),
                [{"input_ids": [3], "target_ids": [0], "loss_mask": [1]}],
                vocab_size=2,
                context_length=1,
            )


if __name__ == "__main__":
    unittest.main()
