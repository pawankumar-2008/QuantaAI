import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence

from backend import config as _config
from backend.services.provider_errors import ProviderError


class GeminiProviderError(ProviderError):
    def __init__(
        self,
        message: str,
        *,
        category: str,
        status_code: int | None = None,
    ):
        super().__init__(
            message,
            provider="Google Gemini",
            category=category,
            status_code=status_code,
        )


class GeminiService:
    name = "Google Gemini"
    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    DEFAULT_MODEL = "gemini-3.5-flash-lite"

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.model = os.getenv("GEMINI_MODEL", "").strip() or self.DEFAULT_MODEL

    def is_available(self) -> bool:
        return bool(self.api_key and self.model)

    def generate(
        self,
        message: str,
        agent: str = "Study",
        conversation_history: Sequence[Mapping[str, str]] | None = None,
        *,
        system_prompt: str,
        max_tokens: int = 1500,
        temperature: float = 0.4,
    ) -> dict[str, str]:
        if not self.api_key or not self.model:
            raise GeminiProviderError(
                "Gemini API key or model is not configured.",
                category="not_configured",
            )

        contents = []
        for entry in conversation_history or ():
            role = entry.get("role")
            content = entry.get("content")
            if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
                contents.append(
                    {
                        "role": "model" if role == "assistant" else "user",
                        "parts": [{"text": content}],
                    }
                )
        contents.append({"role": "user", "parts": [{"text": message}]})

        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        url = self.API_URL.format(model=urllib.parse.quote(self.model, safe=""))
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            status_code = error.code
            error.close()
            raise GeminiProviderError(
                "Gemini request failed.",
                category="http_error",
                status_code=status_code,
            ) from None
        except urllib.error.URLError:
            raise GeminiProviderError(
                "Could not connect to Gemini.",
                category="network_error",
            ) from None
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise GeminiProviderError(
                "Gemini returned an unreadable response.",
                category="invalid_response",
            ) from None
        except OSError:
            raise GeminiProviderError(
                "Could not connect to Gemini.",
                category="network_error",
            ) from None

        candidates = result.get("candidates") if isinstance(result, dict) else None
        if not isinstance(candidates, list) or not candidates:
            raise GeminiProviderError(
                "Gemini returned no response candidates.",
                category="invalid_response",
            )

        candidate = candidates[0]
        content = candidate.get("content") if isinstance(candidate, dict) else None
        parts = content.get("parts") if isinstance(content, dict) else None
        reply = "\n".join(
            part["text"]
            for part in parts or ()
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
        if not reply.strip():
            raise GeminiProviderError(
                "Gemini returned an empty response.",
                category="invalid_response",
            )

        return {
            "reply": reply.strip(),
            "provider": "Google Gemini",
            "model": self.model,
            "agent": agent,
            "status": "success",
        }


gemini_service = GeminiService()