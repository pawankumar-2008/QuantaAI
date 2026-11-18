import unittest

import torch

from src.evaluator import benchmark_generation, device_information
from src.model import ModelConfig, QuantaLMV2


class BenchmarkTests(unittest.TestCase):
    def test_cpu_benchmark_reports_warmup_and_measured_runs(self):
        model = QuantaLMV2(
            ModelConfig(
                vocab_size=2048,
                context_length=256,
                embedding_dim=16,
                num_layers=1,
                num_heads=2,
                feed_forward_dim=32,
                dropout=0.0,
            )
        )
        result = benchmark_generation(
            model,
            torch.tensor([[258]], dtype=torch.long),
            max_new_tokens=2,
            top_k=1,
            warmup_runs=1,
            measured_runs=2,
        )
        self.assertEqual(result["warmup_runs"], 1)
        self.assertEqual(result["measured_runs"], 2)
        self.assertGreater(result["first_token_latency_seconds_mean"], 0.0)
        self.assertGreater(result["average_response_latency_seconds"], 0.0)
        self.assertGreater(result["tokens_per_second"], 0.0)
        self.assertIn("peak_cpu_memory_bytes", result)
        self.assertEqual(result["device_information"]["device_type"], "cpu")
        self.assertFalse(result["device_information"]["gpu_execution_verified"])
        self.assertIn("not an evaluation question", result["prompt_source"])

    def test_benchmark_rejects_bad_run_count_and_context(self):
        model = QuantaLMV2(
            ModelConfig(
                vocab_size=2048,
                context_length=2,
                embedding_dim=8,
                num_layers=1,
                num_heads=2,
                feed_forward_dim=16,
                dropout=0.0,
            )
        )
        with self.assertRaisesRegex(ValueError, "measured_runs"):
            benchmark_generation(
                model,
                torch.tensor([[1]], dtype=torch.long),
                measured_runs=0,
            )
        with self.assertRaisesRegex(ValueError, "context length"):
            benchmark_generation(
                model,
                torch.tensor([[1, 2, 3]], dtype=torch.long),
            )

    def test_cpu_device_information_does_not_claim_gpu_verification(self):
        info = device_information("cpu")
        self.assertEqual(info["device"], "cpu")
        self.assertFalse(info["gpu_execution_verified"])


if __name__ == "__main__":
    unittest.main()
