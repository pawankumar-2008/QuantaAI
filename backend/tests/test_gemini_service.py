import json
import os
import unittest
from unittest.mock import MagicMock, patch

from backend.services.gemini_service import GeminiProviderError, GeminiService


class GeminiServiceTests(unittest.TestCase):
    def test_generate_sends_system_prompt_history_and_key_header(self):
        response = MagicMock()
        response.read.return_value = json.dumps(
            {
                "candidates": [
                    {"content": {"parts": [{"text": "A recursive function calls itself."}]}}
                ]
            }
        ).encode()
        response.__enter__.return_value = response
        history = [
            {"role": "user", "content": "What is recursion?"},
            {"role": "assistant", "content": "A function calling itself."},
        ]

        with (
            patch.dict(
                os.environ,
                {"GEMINI_API_KEY": "mock-gemini-key", "GEMINI_MODEL": "mock-gemini-model"},
            ),
            patch(
                "backend.services.gemini_service.urllib.request.urlopen",
                return_value=response,
            ) as urlopen,
        ):
            result = GeminiService().generate(
                "Give an example",
                "Coding",
                history,
                system_prompt="You are a coding tutor.",
                max_tokens=128,
                temperature=0.2,
            )

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.get_header("X-goog-api-key"), "mock-gemini-key")
        self.assertNotIn("mock-gemini-key", request.full_url)
        self.assertEqual(payload["systemInstruction"]["parts"][0]["text"], "You are a coding tutor.")
        self.assertEqual(
            [item["role"] for item in payload["contents"]],
            ["user", "model", "user"],
        )
        self.assertEqual(payload["generationConfig"]["maxOutputTokens"], 128)
        self.assertEqual(result["provider"], "Google Gemini")
        self.assertEqual(result["model"], "mock-gemini-model")
        self.assertEqual(result["agent"], "Coding")
        self.assertEqual(result["reply"], "A recursive function calls itself.")

    def test_gemini_uses_the_stable_default_model_when_unset(self):
        with patch.dict(
            os.environ,
            {"GEMINI_API_KEY": "mock-gemini-key", "GEMINI_MODEL": ""},
        ):
            service = GeminiService()

        self.assertTrue(service.is_available())
        self.assertEqual(service.model, "gemini-3.5-flash-lite")

    def test_gemini_is_unavailable_without_a_key(self):
        with patch.dict(
            os.environ,
            {"GEMINI_API_KEY": "", "GEMINI_MODEL": ""},
        ):
            service = GeminiService()

        self.assertFalse(service.is_available())
        with self.assertRaises(GeminiProviderError) as raised:
            service.generate("Question", system_prompt="System")
        self.assertEqual(raised.exception.category, "not_configured")

    def test_http_failure_is_normalized_without_response_secrets(self):
        with (
            patch.dict(
                os.environ,
                {"GEMINI_API_KEY": "mock-gemini-key", "GEMINI_MODEL": "mock-gemini-model"},
            ),
            patch(
                "backend.services.gemini_service.urllib.request.urlopen",
                side_effect=OSError("connection failed"),
            ),
        ):
            with self.assertRaises(GeminiProviderError) as raised:
                GeminiService().generate("Question", system_prompt="System")

        self.assertEqual(raised.exception.category, "network_error")
        self.assertNotIn("mock-gemini-key", str(raised.exception))


if __name__ == "__main__":
    unittest.main()