import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bpe_tokenizer import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    UNK_ID,
    QuantaBPETokenizer,
)
from src.model import ModelConfig
from src.workspace_config import v2_config


V2_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_DIR = V2_ROOT / "tokenizer"


class V2TokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = QuantaBPETokenizer.load(TOKENIZER_DIR)

    def test_vocabulary_matches_model_and_ids_are_in_range(self):
        self.assertEqual(self.tokenizer.vocab_size, 2048)
        self.assertEqual(self.tokenizer.vocab_size, ModelConfig().vocab_size)
        self.assertEqual(
            (PAD_ID, UNK_ID, BOS_ID, EOS_ID),
            (256, 257, 258, 259),
        )
        self.assertEqual(self.tokenizer.eos_token_id, EOS_ID)
        self.assertEqual(v2_config.EOS_TOKEN_ID, EOS_ID)
        for sample in self.samples():
            token_ids = self.tokenizer.encode(sample, add_special_tokens=True)
            self.assertTrue(all(0 <= token_id < 2048 for token_id in token_ids))

    @staticmethod
    def samples():
        return (
            "A language model predicts the next token.",
            "भारत में विज्ञान और गणित का अध्ययन।",
            "∑_{i=1}^{n} x_i^2 ≤ 10; Δx → 0",
            "def add(a, b):\n    return a + b\n",
            "public class Main {\n    public static void main(String[] args) {}\n}",
            "Avogadro's constant is approximately 6.022e23 mol^-1.",
            "First line.\n\n    indented code\n\tTabbed line\n",
            "   \t",
        )

    def test_unicode_code_math_science_and_whitespace_round_trip(self):
        for text in self.samples():
            with self.subTest(text=text):
                self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)

    def test_model_specification_special_tokens_match_tokenizer(self):
        specification = json.loads(
            (V2_ROOT / "model_config.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            specification["tokenizer"]["vocabulary_size"],
            self.tokenizer.vocab_size,
        )
        self.assertEqual(
            specification["tokenizer"]["special_tokens"],
            self.tokenizer.special_token_ids,
        )

    def test_empty_input_and_special_token_behavior(self):
        self.assertEqual(self.tokenizer.encode(""), [])
        self.assertEqual(
            self.tokenizer.encode("", add_special_tokens=True),
            [BOS_ID, EOS_ID],
        )
        self.assertEqual(
            self.tokenizer.decode([BOS_ID, EOS_ID]),
            "",
        )
        sample = "EOS follows this sentence."
        encoded = self.tokenizer.encode(sample, add_special_tokens=True)
        self.assertEqual(encoded[0], BOS_ID)
        self.assertEqual(encoded[-1], EOS_ID)
        self.assertEqual(self.tokenizer.decode(encoded), sample)

    def test_unknown_token_is_not_emitted_for_unicode_input(self):
        text = "नमस्ते — λ ∑ 😀"
        self.assertNotIn(UNK_ID, self.tokenizer.encode(text))
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)

    def test_tokenizer_provenance_uses_training_documents_only(self):
        report = json.loads(
            (TOKENIZER_DIR / "tokenizer_training_report.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(report["vocabulary_size"], ModelConfig().vocab_size)
        self.assertEqual(report["training_source"]["split"], "train")
        self.assertEqual(report["training_source"]["validation_documents_read_for_fitting"], 0)
        self.assertEqual(report["training_source"]["test_documents_read_for_fitting"], 0)
        self.assertFalse(report["v1_tokenizer_modified"])
        self.assertTrue(report["model_compatibility"]["vocabulary_matches_model"])

    def test_fresh_python_process_can_load_saved_tokenizer(self):
        code = (
            "from src.bpe_tokenizer import QuantaBPETokenizer; "
            "t=QuantaBPETokenizer.load(); "
            "s='fresh process — ∑ x²\\n'; "
            "assert t.vocab_size == 2048; "
            "assert t.decode(t.encode(s)) == s; "
            "print(t.eos_token_id)"
        )
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=V2_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), str(EOS_ID))


if __name__ == "__main__":
    unittest.main()
