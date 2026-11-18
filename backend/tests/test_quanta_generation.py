import unittest
import hashlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from backend.services.quanta_service import QuantaService, V2_CHECKPOINT_DIR
from quantalm.src.model import QuantaLM
from quantalm.v2.src.model import QuantaLMV2


class QuantaGenerationTests(unittest.TestCase):
    def _save_v2_training_checkpoint(self, path, *, tokenizer_hash=None):
        model = QuantaLMV2()
        torch.save(
            {
                "schema_version": 1,
                "model_state": model.state_dict(),
                "model_config": model.config.to_dict(),
                "global_step": 1,
                "tokenizer_hash": tokenizer_hash or QuantaService._v2_tokenizer_hash(),
                "dataset_manifest_hash": hashlib.sha256(b"dataset").hexdigest(),
            },
            path,
        )

    def test_v2_trained_checkpoint_loads_through_existing_quanta_service(self):
        with tempfile.TemporaryDirectory(
            prefix="quanta-v2-backend-test-",
            dir=V2_CHECKPOINT_DIR,
        ) as directory:
            checkpoint_path = Path(directory) / "best.pt"
            self._save_v2_training_checkpoint(checkpoint_path)
            service = QuantaService(model_version="v2", checkpoint_path=checkpoint_path)

            def fake_generate(_model, input_ids, **kwargs):
                self.assertEqual(kwargs["top_p"], 0.9)
                self.assertEqual(kwargs["repetition_penalty"], 1.15)
                next_token = torch.tensor([[65]], dtype=torch.long)
                return torch.cat((input_ids, next_token), dim=1)

            with patch(
                "quantalm.v2.src.generation.generate",
                side_effect=fake_generate,
            ):
                reply = service.generate("A test prompt", max_new_tokens=1)

        self.assertTrue(service.loaded)
        self.assertEqual(reply, "A.")

    def test_v2_checkpoint_rejects_tokenizer_mismatch(self):
        with tempfile.TemporaryDirectory(
            prefix="quanta-v2-backend-test-",
            dir=V2_CHECKPOINT_DIR,
        ) as directory:
            checkpoint_path = Path(directory) / "best.pt"
            self._save_v2_training_checkpoint(
                checkpoint_path,
                tokenizer_hash="0" * 64,
            )
            service = QuantaService(model_version="v2", checkpoint_path=checkpoint_path)

            with self.assertRaisesRegex(ValueError, "tokenizer hash"):
                service.load_model()

    def test_incoherent_auxiliary_verb_output_fails_reliability_check(self):
        response = (
            "Delivery. Logic arguments does describes how far a language model "
            "learns from examples and patterns."
        )

        self.assertFalse(
            QuantaService.is_response_reliable(
                "How does a language model learn?",
                response,
            )
        )

    def test_decoded_responses_are_cleaned_and_split_into_paragraphs(self):
        raw = (
            "database management systems organize data. they store, retrieve, "
            "and update records. a primary key identifies each row. "
            "a foreign key links rows across tables."
        )

        formatted = QuantaService.format_response(raw)

        self.assertEqual(
            formatted,
            (
                "Database management systems organize data. They store, retrieve, "
                "and update records. A primary key identifies each row.\n\n"
                "A foreign key links rows across tables."
            ),
        )

    def test_existing_headings_and_ordered_and_bullet_lists_are_normalized(self):
        formatted = QuantaService.format_response(
            "steps:\n1 . choose a process\n2 . choose a scheduling policy\n"
            "- compare waiting time\n* compare throughput"
        )

        self.assertEqual(
            formatted,
            (
                "### Steps\n\n1. Choose a process\n2. Choose a scheduling policy\n\n"
                "- Compare waiting time\n- Compare throughput"
            ),
        )

    def test_repeated_sentences_and_word_runs_are_removed(self):
        formatted = QuantaService.format_response(
            "the key identifies a row. the key identifies a row. "
            "the the key is unique."
        )

        self.assertEqual(
            formatted,
            "The key identifies a row. The key is unique.",
        )

    def test_truncated_output_does_not_end_on_a_partial_sentence(self):
        formatted = QuantaService.format_response(
            "the first point is complete. the next point is cut off\n\n",
            truncated=True,
        )

        self.assertEqual(formatted, "The first point is complete.")

        only_partial_sentence = QuantaService.format_response(
            "an unfinished answer",
            truncated=True,
        )
        self.assertEqual(only_partial_sentence, "An unfinished answer…")

    def test_plain_answer_does_not_receive_invented_headings(self):
        formatted = QuantaService.format_response(
            "a primary key uniquely identifies a row"
        )

        self.assertEqual(formatted, "A primary key uniquely identifies a row.")
        self.assertNotIn("#", formatted)

    def test_fenced_code_spacing_is_preserved(self):
        formatted = QuantaService.format_response(
            "example:\n```python\nif value  ==  1:\n    print(  value  )\n```"
        )

        self.assertEqual(
            formatted,
            "### Example\n\n```python\nif value  ==  1:\n    print(  value  )\n```",
        )

    def test_generation_formats_the_v1_decoded_text_without_changing_provider(self):
        service = QuantaService()
        service.loaded = True
        service.tokenizer = MagicMock()
        service.tokenizer.encode.return_value = [1, 2]
        service.tokenizer.decode.return_value = (
            "java inheritance allows a class to reuse behavior. "
            "it can override inherited methods"
        )
        service.tokenizer.eos_id = 0
        service.model = MagicMock()
        service.model.generate.return_value = torch.tensor([[1, 2, 3, 4]])

        response = service.generate("Explain Java inheritance", max_new_tokens=8)

        self.assertEqual(
            response,
            (
                "Java inheritance allows a class to reuse behavior. "
                "It can override inherited methods."
            ),
        )
        service.tokenizer.decode.assert_called_once_with([3, 4])
        service.model.generate.assert_called_once()

    def test_requested_study_prompts_share_only_formatting_instructions(self):
        from backend.agents.profiles import build_offline_prompt

        prompts = [
            "Explain DBMS.",
            "What is a primary key?",
            "Explain process scheduling.",
            "Explain Java inheritance.",
        ]
        for request in prompts:
            with self.subTest(request=request):
                prompt = build_offline_prompt("Study", request)
                self.assertIn(request, prompt)
                self.assertIn("short paragraphs", prompt)
                self.assertIn("Use headings only", prompt)

    def test_repetition_penalty_is_applied_and_generation_stops_at_eos(self):
        model = QuantaLM(vocab_size=5)
        forward_steps = 0

        def scripted_forward(token_ids):
            nonlocal forward_steps
            logits = torch.full(
                (token_ids.size(0), token_ids.size(1), 5),
                -100.0,
            )
            if forward_steps == 0:
                logits[:, -1, 4] = 10.0
            else:
                logits[:, -1, 4] = 4.0
                logits[:, -1, 3] = 3.5
            forward_steps += 1
            return logits, None

        with patch.object(model, "forward", side_effect=scripted_forward) as forward:
            output = model.generate(
                torch.tensor([[2]], dtype=torch.long),
                max_new_tokens=8,
                temperature=0.1,
                top_p=0.9,
                repetition_penalty=2.0,
                eos_token_id=3,
            )

        self.assertEqual(output.tolist(), [[2, 4, 3]])
        self.assertEqual(forward.call_count, 2)


if __name__ == "__main__":
    unittest.main()