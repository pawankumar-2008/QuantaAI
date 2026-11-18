import json
import unittest
from unittest.mock import MagicMock, Mock, patch

from backend.services.gemini_service import GeminiProviderError, GeminiService
from backend.services.provider_errors import ProviderError, ProviderUnavailableError
from backend.services.provider_manager import (
    OpenRouterModelProvider,
    ProviderManager,
    QuantaLMProvider,
    build_provider_manager,
    select_provider,
)
from backend.services.teacher_service import TeacherService


class FakeProvider:
    def __init__(self, name, model, *, available=True, error=None):
        self.name = name
        self.model = model
        self.available = available
        self.error = error
        self.is_local = name == "QuantaLM Offline"
        self.calls = []

    def is_available(self):
        return self.available

    def generate(
        self,
        message,
        agent,
        conversation_history=None,
        *,
        system_prompt,
        max_tokens,
        temperature,
    ):
        self.calls.append(
            {
                "message": message,
                "agent": agent,
                "history": conversation_history,
                "system_prompt": system_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        if self.error:
            raise self.error
        return {"reply": f"reply from {self.name}", "model": self.model}


class ProviderManagerTests(unittest.TestCase):
    def test_provider_mode_selection_keeps_offline_local_and_auto_configurable(self):
        self.assertEqual(select_provider("offline"), "quanta")
        self.assertEqual(select_provider("teacher"), "teacher")
        with patch.dict("os.environ", {"QUANTAAI_AUTO_PROVIDER": "quanta"}):
            self.assertEqual(select_provider("auto"), "quanta")
        with patch.dict("os.environ", {"QUANTAAI_AUTO_PROVIDER": "invalid"}):
            with self.assertRaisesRegex(ValueError, "QUANTAAI_AUTO_PROVIDER"):
                select_provider("auto")

    def test_openrouter_secondary_sends_its_bound_model_in_the_request(self):
        service = TeacherService()
        service.api_key = "mock-openrouter-key"
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = (
            b'{"choices":[{"message":{"content":"mocked answer"}}]}'
        )
        provider = OpenRouterModelProvider(
            service,
            "OpenRouter Secondary",
            "nemotron-model",
        )
        manager = ProviderManager((provider,))

        with patch(
            "backend.services.teacher_service.urllib.request.urlopen",
            return_value=response,
        ) as urlopen:
            result = manager.generate("Question", "Study")

        request = urlopen.call_args.args[0]
        self.assertEqual(json.loads(request.data)["model"], "nemotron-model")
        self.assertEqual(result["provider"], "OpenRouter Secondary")
        self.assertEqual(result["model"], "nemotron-model")

    @staticmethod
    def _configured_gemini():
        gemini = GeminiService()
        gemini.api_key = "mock-gemini-key"
        gemini.model = "gemini-model"
        return gemini

    def test_openrouter_model_adapter_binds_the_configured_model(self):
        service = Mock()
        service.is_available.return_value = True
        service.generate.return_value = {
            "reply": "secondary response",
            "model": "nemotron-model",
        }
        provider = OpenRouterModelProvider(
            service,
            "OpenRouter Secondary",
            "nemotron-model",
        )
        manager = ProviderManager((provider,))

        result = manager.generate("Explain this", "Research")

        self.assertEqual(result["provider"], "OpenRouter Secondary")
        self.assertEqual(result["model"], "nemotron-model")
        self.assertEqual(
            service.generate.call_args.kwargs["model_override"],
            "nemotron-model",
        )

    def test_provider_builder_uses_a_distinct_secondary_openrouter_key(self):
        primary = TeacherService(
            api_key="primary-test-key",
            model="primary-test-model",
        )
        with patch.dict(
            "os.environ",
            {
                "OPENROUTER_API_KEY_SECONDARY": "secondary-test-key",
                "OPENROUTER_MODEL_SECONDARY": "secondary-test-model",
            },
        ):
            manager = build_provider_manager(
                teacher=primary,
                gemini=FakeProvider("Google Gemini", "gemini-model"),
                quanta=Mock(),
            )

        primary_provider, secondary_provider = manager.providers[:2]
        self.assertIsNot(primary_provider.service, secondary_provider.service)
        self.assertEqual(primary_provider.service.api_key, "primary-test-key")
        self.assertEqual(secondary_provider.service.api_key, "secondary-test-key")
        self.assertEqual(primary_provider.model, "primary-test-model")
        self.assertEqual(secondary_provider.model, "secondary-test-model")

    def test_primary_success_returns_provider_and_model_without_fallback(self):
        primary = FakeProvider("OpenRouter Primary", "xing-model")
        secondary = FakeProvider("OpenRouter Secondary", "nemotron-model")
        gemini = FakeProvider("Google Gemini", "gemini-model")
        manager = ProviderManager((primary, secondary, gemini))

        result = manager.generate("Explain recursion", "Coding")

        self.assertEqual(result["provider"], "OpenRouter Primary")
        self.assertEqual(result["model"], "xing-model")
        self.assertEqual(result["agent"], "Coding")
        self.assertEqual(len(primary.calls), 1)
        self.assertEqual(secondary.calls, [])
        self.assertEqual(gemini.calls, [])

    def test_falls_back_in_order_and_preserves_history_and_prompt(self):
        primary = FakeProvider(
            "OpenRouter Primary",
            "xing-model",
            error=ProviderError(
                "primary failed",
                provider="OpenRouter Primary",
                category="http_error",
            ),
        )
        secondary = FakeProvider(
            "OpenRouter Secondary",
            "nemotron-model",
            error=ProviderError(
                "secondary failed",
                provider="OpenRouter Secondary",
                category="http_error",
            ),
        )
        gemini = self._configured_gemini()
        manager = ProviderManager((primary, secondary, gemini))
        history = [{"role": "user", "content": "Earlier question"}]

        with patch.object(
            gemini,
            "generate",
            return_value={"reply": "reply from Google Gemini", "model": "gemini-model"},
        ) as gemini_generate:
            result = manager.generate(
                "Follow up",
                "Study",
                history,
                system_prompt="Custom tutor prompt",
            )

        self.assertEqual(result["provider"], "Google Gemini")
        self.assertEqual(result["model"], "gemini-model")
        self.assertEqual([len(item.calls) for item in (primary, secondary)], [1, 1])
        gemini_generate.assert_called_once()
        self.assertEqual(gemini_generate.call_args.args[2], history)
        self.assertEqual(
            gemini_generate.call_args.kwargs["system_prompt"],
            "Custom tutor prompt",
        )

    def test_primary_failure_uses_secondary_once_and_stops_on_success(self):
        primary = FakeProvider(
            "OpenRouter Primary",
            "xing-model",
            error=ProviderError(
                "primary failed",
                provider="OpenRouter Primary",
                category="http_error",
            ),
        )
        secondary = FakeProvider("OpenRouter Secondary", "nemotron-model")
        gemini = FakeProvider("Google Gemini", "gemini-model")
        quantalm = FakeProvider("QuantaLM Offline", "QuantaLM")
        manager = ProviderManager((primary, secondary, gemini, quantalm))

        result = manager.generate("Question", "Career")

        self.assertEqual(result["provider"], "OpenRouter Secondary")
        self.assertEqual(result["model"], "nemotron-model")
        self.assertEqual([len(provider.calls) for provider in manager.providers], [1, 1, 0, 0])

    def test_falls_back_to_quantalm_after_all_cloud_providers_fail(self):
        primary = FakeProvider(
            "OpenRouter Primary",
            "xing-model",
            error=ProviderError(
                "primary failed",
                provider="OpenRouter Primary",
                category="network_error",
            ),
        )
        secondary = FakeProvider(
            "OpenRouter Secondary",
            "nemotron-model",
            error=ProviderError(
                "secondary failed",
                provider="OpenRouter Secondary",
                category="network_error",
            ),
        )
        gemini = self._configured_gemini()
        quantalm = FakeProvider("QuantaLM Offline", "QuantaLM")
        manager = ProviderManager((primary, secondary, gemini, quantalm))
        history = [{"role": "user", "content": "Earlier question"}]

        with patch.object(
            gemini,
            "generate",
            side_effect=GeminiProviderError(
                "Gemini failed",
                category="network_error",
            ),
        ) as gemini_generate:
            result = manager.generate(
                "Follow up",
                "Study",
                history,
                system_prompt="Study tutor prompt",
            )

        self.assertEqual(result["provider"], "QuantaLM Offline")
        self.assertEqual(result["model"], "QuantaLM")
        self.assertEqual([len(provider.calls) for provider in (primary, secondary, quantalm)], [1, 1, 1])
        gemini_generate.assert_called_once()
        self.assertEqual(gemini_generate.call_args.args[2], history)
        self.assertEqual(
            gemini_generate.call_args.kwargs["system_prompt"],
            "Study tutor prompt",
        )
        self.assertEqual(quantalm.calls[0]["history"], history)
        self.assertEqual(quantalm.calls[0]["system_prompt"], "Study tutor prompt")

    def test_skips_unavailable_providers(self):
        primary = FakeProvider("OpenRouter Primary", "xing-model", available=False)
        secondary = FakeProvider("OpenRouter Secondary", "nemotron-model")
        manager = ProviderManager((primary, secondary))

        result = manager.generate("Explain", "Guide")

        self.assertEqual(result["provider"], "OpenRouter Secondary")
        self.assertEqual(primary.calls, [])
        self.assertEqual(len(secondary.calls), 1)

    def test_raises_clear_error_when_no_provider_is_configured(self):
        manager = ProviderManager(
            (FakeProvider("OpenRouter Primary", "", available=False),)
        )

        with self.assertRaisesRegex(
            ProviderUnavailableError,
            "No AI provider is configured or available",
        ):
            manager.generate("Explain", "Exam")

    def test_raises_clear_error_when_all_providers_fail(self):
        primary = FakeProvider(
            "OpenRouter Primary",
            "xing-model",
            error=ProviderError(
                "primary failed",
                provider="OpenRouter Primary",
                category="network_error",
            ),
        )
        secondary = FakeProvider(
            "OpenRouter Secondary",
            "nemotron-model",
            error=ProviderError(
                "secondary failed",
                provider="OpenRouter Secondary",
                category="network_error",
            ),
        )
        gemini = self._configured_gemini()
        quantalm = FakeProvider(
            "QuantaLM Offline",
            "QuantaLM",
            error=ProviderError(
                "QuantaLM failed",
                provider="QuantaLM Offline",
                category="generation_error",
            ),
        )
        manager = ProviderManager((primary, secondary, gemini, quantalm))

        with patch.object(
            gemini,
            "generate",
            side_effect=GeminiProviderError(
                "Gemini failed",
                category="network_error",
            ),
        ) as gemini_generate:
            with self.assertRaisesRegex(
                ProviderUnavailableError,
                "All configured AI providers failed, including QuantaLM",
            ):
                manager.generate("Explain", "Planner")

        self.assertEqual([len(provider.calls) for provider in (primary, secondary, quantalm)], [1, 1, 1])
        gemini_generate.assert_called_once()

    def test_quantalm_adapter_preserves_prompt_history_and_reliability_guard(self):
        service = Mock()
        service.loaded = True
        service.generate.return_value = "A safe response with enough distinct words."
        service.is_response_reliable.return_value = True
        provider = QuantaLMProvider(service)
        history = [{"role": "assistant", "content": "Prior answer"}]

        result = provider.generate(
            "Current question",
            "Study",
            history,
            system_prompt="Study instructions",
            max_tokens=1500,
            temperature=0.4,
        )

        prompt = service.generate.call_args.args[0]
        self.assertIn("Study instructions", prompt)
        self.assertIn("[ASSISTANT] Prior answer", prompt)
        self.assertIn("Current question", prompt)
        self.assertEqual(service.generate.call_args.kwargs["max_new_tokens"], 80)
        self.assertEqual(result["provider"], "QuantaLM Offline")
        self.assertEqual(result["model"], "QuantaLM")
        self.assertEqual(result["reliability"], "reliable")
        service.is_response_reliable.assert_called_once_with(
            "Current question",
            "A safe response with enough distinct words.",
        )

    def test_quantalm_adapter_returns_unreliable_output_with_quality_metadata(self):
        service = Mock()
        service.loaded = True
        service.generate.return_value = "unreliable output"
        service.is_response_reliable.return_value = False
        provider = QuantaLMProvider(service)

        result = provider.generate(
            "Question",
            "Coding",
            system_prompt="Coding instructions",
            max_tokens=80,
            temperature=0.4,
        )

        self.assertEqual(result["reply"], "unreliable output")
        self.assertEqual(result["reliability"], "unreliable")
        self.assertEqual(result["provider"], "QuantaLM Offline")
        self.assertEqual(result["model"], "QuantaLM")

    def test_manager_returns_unreliable_quantalm_fallback_without_retrying(self):
        primary = FakeProvider(
            "OpenRouter Primary",
            "xing-model",
            error=ProviderError(
                "primary failed",
                provider="OpenRouter Primary",
                category="network_error",
            ),
        )
        service = Mock()
        service.loaded = True
        service.generate.return_value = "A questionable but displayable response."
        service.is_response_reliable.return_value = False
        quantalm = QuantaLMProvider(service)
        manager = ProviderManager((primary, quantalm))

        result = manager.generate("Question", "Coding")

        self.assertEqual(result["reply"], "A questionable but displayable response.")
        self.assertEqual(result["provider"], "QuantaLM Offline")
        self.assertEqual(result["model"], "QuantaLM")
        self.assertEqual(result["reliability"], "unreliable")
        self.assertEqual(len(primary.calls), 1)
        service.generate.assert_called_once()

    def test_quantalm_adapter_keeps_invalid_and_failed_generations_as_errors(self):
        for invalid_output in ("", None, []):
            with self.subTest(invalid_output=invalid_output):
                service = Mock()
                service.loaded = True
                service.generate.return_value = invalid_output
                provider = QuantaLMProvider(service)

                with self.assertRaisesRegex(ProviderError, "empty response"):
                    provider.generate(
                        "Question",
                        "Coding",
                        system_prompt="Coding instructions",
                        max_tokens=80,
                        temperature=0.4,
                    )

        service = Mock()
        service.loaded = True
        service.generate.side_effect = RuntimeError("private model-loading detail")
        provider = QuantaLMProvider(service)

        with self.assertRaisesRegex(ProviderError, "generation failed") as raised:
            provider.generate(
                "Question",
                "Coding",
                system_prompt="Coding instructions",
                max_tokens=80,
                temperature=0.4,
            )

        self.assertNotIn("private model-loading detail", str(raised.exception))

    def test_rejects_unknown_agents_before_provider_calls(self):
        provider = FakeProvider("OpenRouter Primary", "xing-model")
        manager = ProviderManager((provider,))

        with self.assertRaisesRegex(ValueError, "Unknown agent"):
            manager.generate("Explain", "Unknown")

        self.assertEqual(provider.calls, [])


if __name__ == "__main__":
    unittest.main()