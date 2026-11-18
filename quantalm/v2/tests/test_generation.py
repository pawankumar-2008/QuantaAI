import unittest
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generation import generate
from src.model import ModelConfig, QuantaLMV2


class GenerationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.model = QuantaLMV2(ModelConfig(dropout=0.0))
        self.input_ids = torch.tensor([[10, 11], [20, 21]], dtype=torch.long)

    def test_generates_requested_number_of_tokens_with_context_truncation(self):
        output = generate(
            self.model,
            torch.randint(0, 2048, (1, 255)),
            max_new_tokens=3,
            temperature=1.0,
            top_k=20,
            seed=9,
        )
        self.assertEqual(output.shape, (1, 258))

    def test_generation_is_reproducible_with_seed(self):
        first = generate(self.model, self.input_ids, 5, seed=111)
        second = generate(self.model, self.input_ids, 5, seed=111)
        self.assertTrue(torch.equal(first, second))

    def test_top_k_limits_sampled_tokens(self):
        prompt = self.input_ids[:1]
        self.model.eval()
        with torch.no_grad():
            expected_token = self.model(prompt)[:, -1, :].argmax(dim=-1)
        output = generate(self.model, prompt, max_new_tokens=1, top_k=1, seed=4)
        self.assertEqual(output.shape, (1, 3))
        self.assertTrue(torch.equal(output[:, -1], expected_token))

    def test_top_p_keeps_only_the_minimum_probability_prefix(self):
        class UniformModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = ModelConfig(vocab_size=8, context_length=4)

            def forward(self, token_ids):
                return torch.zeros(
                    token_ids.shape[0],
                    token_ids.shape[1],
                    self.config.vocab_size,
                )

        model = UniformModel()

        def capture_probabilities(probabilities, num_samples, generator=None):
            self.assertEqual(num_samples, 1)
            self.assertEqual(int(torch.count_nonzero(probabilities).item()), 2)
            return probabilities.argmax(dim=-1, keepdim=True)

        with unittest.mock.patch(
            "src.generation.torch.multinomial",
            side_effect=capture_probabilities,
        ):
            generate(
                model,
                torch.tensor([[1]], dtype=torch.long),
                max_new_tokens=1,
                top_p=0.2,
                seed=1,
            )

    def test_repetition_penalty_can_change_the_next_token(self):
        class RepeatingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = ModelConfig(vocab_size=8, context_length=4)

            def forward(self, token_ids):
                logits = torch.zeros(
                    token_ids.shape[0],
                    token_ids.shape[1],
                    self.config.vocab_size,
                )
                logits[..., 2] = 5.0
                logits[..., 3] = 4.0
                return logits

        output = generate(
            RepeatingModel(),
            torch.tensor([[2]], dtype=torch.long),
            max_new_tokens=1,
            top_k=1,
            repetition_penalty=2.0,
            seed=1,
        )
        self.assertEqual(output[0, -1].item(), 3)

    def test_zero_new_tokens_returns_input(self):
        output = generate(self.model, self.input_ids, max_new_tokens=0)
        self.assertTrue(torch.equal(output, self.input_ids))

    def test_temperature_must_be_positive(self):
        for temperature in (0, -1.0, float("nan"), float("inf")):
            with self.subTest(temperature=temperature):
                with self.assertRaisesRegex(ValueError, "temperature"):
                    generate(
                        self.model,
                        self.input_ids,
                        max_new_tokens=1,
                        temperature=temperature,
                    )

    def test_top_p_and_repetition_penalty_must_be_valid(self):
        for value in (0, -0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(top_p=value):
                with self.assertRaisesRegex(ValueError, "top_p"):
                    generate(
                        self.model,
                        self.input_ids,
                        max_new_tokens=1,
                        top_p=value,
                    )
        for value in (0.9, float("nan"), float("inf")):
            with self.subTest(repetition_penalty=value):
                with self.assertRaisesRegex(ValueError, "repetition_penalty"):
                    generate(
                        self.model,
                        self.input_ids,
                        max_new_tokens=1,
                        repetition_penalty=value,
                    )

    def test_top_k_must_be_valid(self):
        for top_k in (0, -2, 2049):
            with self.subTest(top_k=top_k):
                with self.assertRaisesRegex(ValueError, "top_k"):
                    generate(
                        self.model,
                        self.input_ids,
                        max_new_tokens=1,
                        top_k=top_k,
                    )

    def test_max_new_tokens_must_be_non_negative_integer(self):
        for value in (-1, 1.5):
            with self.subTest(value=value):
                with self.assertRaises((ValueError, TypeError)):
                    generate(self.model, self.input_ids, max_new_tokens=value)

    def test_stops_after_eos_for_all_batch_items(self):
        class AlwaysEosModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = ModelConfig(vocab_size=8, context_length=4)

            def forward(self, token_ids):
                logits = torch.zeros(
                    token_ids.shape[0],
                    token_ids.shape[1],
                    self.config.vocab_size,
                    device=token_ids.device,
                )
                logits[..., 3] = 100
                return logits

        model = AlwaysEosModel()
        prompt = torch.tensor([[1], [2]])
        result = generate(
            model,
            prompt,
            max_new_tokens=10,
            eos_token_id=3,
            seed=7,
        )
        self.assertEqual(result.shape, (2, 2))
        self.assertTrue(torch.all(result[:, -1] == 3))

    def test_generation_restores_training_mode(self):
        self.model.train()
        generate(self.model, self.input_ids[:1], max_new_tokens=1, seed=2)
        self.assertTrue(self.model.training)


if __name__ == "__main__":
    unittest.main()
