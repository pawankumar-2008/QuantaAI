import unittest
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attention import CausalSelfAttention


class CausalAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.attention = CausalSelfAttention(embedding_dim=16, num_heads=4, dropout=0.0)
        self.attention.eval()

    def test_attention_preserves_shape_and_batch(self):
        values = torch.randn(3, 7, 16)
        self.assertEqual(self.attention(values).shape, (3, 7, 16))

    def test_attention_has_no_future_token_leakage(self):
        values = torch.randn(2, 6, 16)
        changed = values.clone()
        changed[:, 4:, :] = torch.randn_like(changed[:, 4:, :]) * 10
        with torch.no_grad():
            original_output = self.attention(values)
            changed_output = self.attention(changed)
        torch.testing.assert_close(
            original_output[:, :4, :],
            changed_output[:, :4, :],
            rtol=1e-5,
            atol=1e-6,
        )

    def test_single_position_attention(self):
        values = torch.randn(2, 1, 16)
        self.assertEqual(self.attention(values).shape, (2, 1, 16))

    def test_rejects_non_3d_input(self):
        with self.assertRaises(ValueError):
            self.attention(torch.randn(2, 16))

    def test_rejects_incompatible_embedding_dimension(self):
        with self.assertRaises(ValueError):
            self.attention(torch.randn(2, 3, 12))

    def test_rejects_non_divisible_head_configuration(self):
        with self.assertRaises(ValueError):
            CausalSelfAttention(embedding_dim=15, num_heads=4)


if __name__ == "__main__":
    unittest.main()
