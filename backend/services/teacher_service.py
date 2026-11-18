import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence

from backend import config as _config
from backend.agents.profiles import AGENT_PROMPTS
from backend.services.provider_errors import ProviderError


class TeacherProviderError(ProviderError):
    def __init__(
        self,
        message: str,
        *,
        category: str,
        status_code: int | None = None,
        diagnostic: str = "",
    ):
        super().__init__(
            message,
            provider="OpenRouter",
            category=category,
            status_code=status_code,
            diagnostic=diagnostic,
        )


class TeacherService:
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    MODEL = "xingchen-agi/xing4.0-29b-a4b:free"
    AGENT_PROMPTS = AGENT_PROMPTS

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
    ):
        configured_key = (
            os.getenv("OPENROUTER_API_KEY", "")
            if api_key is None
            else api_key
        )
        configured_model = (
            os.getenv("OPENROUTER_MODEL", self.MODEL)
            if model is None
            else model
        )
        self.api_key = configured_key.strip()
        self.model = configured_model.strip() or (
            self.MODEL if model is None else ""
        )

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(
        self,
        message: str,
        agent: str = "Study",
        conversation_history: Sequence[Mapping[str, str]] | None = None,
        *,
        system_prompt: str | None = None,
        model_override: str | None = None,
        max_tokens: int = 1500,
        temperature: float = 0.4,
    ) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not configured.")
        if agent not in self.AGENT_PROMPTS:
            raise ValueError(f"Unknown agent: {agent}")

        selected_model = model_override or self.model
        messages = [{
            "role": "system",
            "content": system_prompt or self.AGENT_PROMPTS[agent],
        }]
        if conversation_history:
            messages.extend(
                {"role": entry["role"], "content": entry["content"]}
                for entry in conversation_history
                if entry.get("role") in {"user", "assistant"}
                and isinstance(entry.get("content"), str)
                and entry["content"].strip()
            )
        messages.append({"role": "user", "content": message})

        payload = {
            "model": selected_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        request = urllib.request.Request(
            self.API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-Title": "QuantaAI",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            try:
                details = error.read().decode("utf-8", errors="replace")
            finally:
                error.close()
            diagnostic = self._redact_diagnostic(
                self._provider_error_details(details),
                self.api_key,
            )
            raise TeacherProviderError(
                f"OpenRouter request failed (HTTP {error.code}).",
                category="http_error",
                status_code=error.code,
                diagnostic=diagnostic,
            ) from None
        except urllib.error.URLError as error:
            diagnostic = self._redact_diagnostic(str(error.reason), self.api_key)
            raise TeacherProviderError(
                "Could not connect to OpenRouter.",
                category="network_error",
                diagnostic=diagnostic,
            ) from None
        except json.JSONDecodeError:
            raise TeacherProviderError(
                "OpenRouter returned invalid JSON.",
                category="invalid_json",
            ) from None
        except UnicodeDecodeError:
            raise TeacherProviderError(
                "OpenRouter returned an invalidly encoded response.",
                category="invalid_response",
            ) from None
        except OSError as error:
            diagnostic = self._redact_diagnostic(str(error), self.api_key)
            raise TeacherProviderError(
                "Could not connect to OpenRouter.",
                category="network_error",
                diagnostic=diagnostic,
            ) from None

        if not isinstance(result, dict):
            raise TeacherProviderError(
                "OpenRouter returned an invalid response.",
                category="invalid_response",
            )
        if "error" in result:
            error_info = result["error"]
            if isinstance(error_info, dict):
                provider_message = error_info.get("message", "Unknown provider error")
                provider_code = error_info.get("code")
            else:
                provider_message = str(error_info)
                provider_code = None
            diagnostic = self._redact_diagnostic(
                f"code={provider_code}; {provider_message}",
                self.api_key,
            )
            raise TeacherProviderError(
                "OpenRouter rejected the generation request.",
                category="provider_error",
                diagnostic=diagnostic,
            )

        choices = result.get("choices")
        if not isinstance(choices, list) or not choices:
            raise TeacherProviderError(
                "OpenRouter returned no response choices.",
                category="invalid_response",
            )

        choice = choices[0]
        message_data = choice.get("message", {}) if isinstance(choice, dict) else {}
        reply = message_data.get("content") if isinstance(message_data, dict) else None
        if isinstance(reply, list):
            reply = "\n".join(
                item.get("text", "")
                for item in reply
                if isinstance(item, dict)
            )
        if not isinstance(reply, str) or not reply.strip():
            raise TeacherProviderError(
                "OpenRouter returned an empty response.",
                category="invalid_response",
            )

        return {
            "reply": reply.strip(),
            "provider": "OpenRouter",
            "model": result.get("model", selected_model),
            "agent": agent,
            "status": "success",
        }

    def generate_youtube_query(
        self,
        message: str,
        conversation_history: Sequence[Mapping[str, str]] | None = None,
    ) -> str | None:
        result = self.generate(
            (
                f"Student's latest request: {message}\n"
                "Create a YouTube search query for educational videos about this "
                "request. Return only one topic-specific query of at most 12 words, "
                "with no quotes or explanation. If the latest request is a greeting "
                "or is not asking about a learnable topic, return exactly NONE."
            ),
            "Study",
            conversation_history,
            system_prompt=(
                "You generate concise search queries, not answers. Use prior "
                "conversation only to resolve references such as 'that' or 'it'. "
                "Never return a URL, a video title, or multiple alternatives."
            ),
            max_tokens=48,
            temperature=0,
        )
        query = result["reply"].strip().strip('"\'`')
        query = re.sub(r"^(?:search query|query)\s*:\s*", "", query, flags=re.I)
        if not query or query.casefold() == "none":
            return None
        if "\n" in query or len(query) > 160 or len(query.split()) > 12:
            raise TeacherProviderError(
                "The provider returned an invalid YouTube search query.",
                category="invalid_response",
            )
        return query

    @staticmethod
    def _provider_error_details(response_text: str) -> str:
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            return response_text[:500]

        error = result.get("error") if isinstance(result, dict) else None
        if isinstance(error, dict):
            return f"code={error.get('code')}; {error.get('message', '')}"
        return str(error)[:500] if error else response_text[:500]

    @staticmethod
    def _redact_diagnostic(value: object, api_key: str | None) -> str:
        diagnostic = str(value)
        if api_key:
            diagnostic = diagnostic.replace(api_key, "[REDACTED]")
        diagnostic = re.sub(
            r"(?i)bearer\s+\S+",
            "Bearer [REDACTED]",
            diagnostic,
        )
        diagnostic = re.sub(
            r"(?i)sk-or-v1-[a-z0-9]+",
            "[REDACTED]",
            diagnostic,
        )
        return diagnostic[:500]


teacher_service = TeacherService()