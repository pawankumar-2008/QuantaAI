import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import ModelConfig, QuantaLMV2


class ModelTests(unittest.TestCase):
    def test_default_configuration_and_parameter_count(self):
        model = QuantaLMV2()
        self.assertEqual(model.config.vocab_size, 2048)
        self.assertEqual(model.config.context_length, 256)
        self.assertEqual(model.config.embedding_dim, 128)
        self.assertEqual(model.config.num_layers, 4)
        self.assertEqual(model.config.num_heads, 4)
        self.assertEqual(model.config.feed_forward_dim, 512)
        self.assertIs(
            model.token_embedding.weight,
            model.output_head.weight,
        )
        self.assertEqual(model.parameter_count(), 1_088_256)

    def test_forward_pass_single_item_batch(self):
        model = QuantaLMV2(ModelConfig(dropout=0.0))
        logits = model(torch.randint(0, 2048, (1, 12)))
        self.assertEqual(logits.shape, (1, 12, 2048))

    def test_forward_pass_batch_size_greater_than_one(self):
        model = QuantaLMV2(ModelConfig(dropout=0.0))
        logits = model(torch.randint(0, 2048, (3, 9)))
        self.assertEqual(logits.shape, (3, 9, 2048))

    def test_sequence_length_one(self):
        model = QuantaLMV2(ModelConfig(dropout=0.0))
        self.assertEqual(model(torch.tensor([[17]])).shape, (1, 1, 2048))

    def test_maximum_context_length(self):
        model = QuantaLMV2(ModelConfig(dropout=0.0))
        with torch.no_grad():
            logits = model(torch.randint(0, 2048, (1, 256)))
        self.assertEqual(logits.shape, (1, 256, 2048))

    def test_invalid_context_length_is_rejected(self):
        model = QuantaLMV2()
        with self.assertRaisesRegex(ValueError, "exceeds context length"):
            model(torch.randint(0, 2048, (1, 257)))

    def test_invalid_input_shapes_and_ids_are_rejected(self):
        model = QuantaLMV2()
        with self.assertRaises(ValueError):
            model(torch.tensor([1, 2]))
        with self.assertRaises(TypeError):
            model(torch.ones((1, 2), dtype=torch.float32))
        with self.assertRaises(ValueError):
            model(torch.tensor([[2048]]))
        with self.assertRaises(ValueError):
            model(torch.empty((1, 0), dtype=torch.long))

    def test_model_has_no_future_token_leakage(self):
        torch.manual_seed(31)
        model = QuantaLMV2(ModelConfig(dropout=0.0))
        model.eval()
        original = torch.randint(0, 2048, (2, 10))
        changed = original.clone()
        changed[:, 6:] = torch.randint(0, 2048, changed[:, 6:].shape)
        with torch.no_grad():
            original_logits = model(original)
            changed_logits = model(changed)
        torch.testing.assert_close(
            original_logits[:, :6, :],
            changed_logits[:, :6, :],
            rtol=1e-5,
            atol=1e-6,
        )

    def test_backpropagation_produces_finite_gradients(self):
        model = QuantaLMV2(ModelConfig(dropout=0.0))
        input_ids = torch.randint(0, 2048, (2, 8))
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, 2048),
            input_ids[:, 1:].reshape(-1),
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )

    def test_cpu_execution(self):
        model = QuantaLMV2(ModelConfig(dropout=0.0)).cpu()
        logits = model(torch.randint(0, 2048, (2, 4)).cpu())
        self.assertEqual(logits.device.type, "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available.")
    def test_gpu_execution_when_available(self):
        model = QuantaLMV2(ModelConfig(dropout=0.0)).cuda()
        logits = model(torch.randint(0, 2048, (2, 4), device="cuda"))
        self.assertEqual(logits.device.type, "cuda")

    def test_independent_bpe_tokenizer_compatibility_and_round_trip(self):
        from src.bpe_tokenizer import QuantaBPETokenizer

        tokenizer = QuantaBPETokenizer.load()
        self.assertEqual(tokenizer.vocab_size, ModelConfig().vocab_size)
        examples = (
            "Unicode: भारत — café.",
            "def add(a, b):\n    return a + b\n",
            "∑_{i=1}^{n} x_i^2 ≤ 10; Δx → 0",
        )
        for text in examples:
            with self.subTest(text=text):
                self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_rejects_non_divisible_embedding_configuration(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            ModelConfig(embedding_dim=127, num_heads=4)


if __name__ == "__main__":
    unittest.main()
