import os
import re
import logging
from collections.abc import Mapping, Sequence
from typing import Protocol

from backend import config as _config
from backend.agents.profiles import AGENT_PROMPTS, build_offline_prompt
from backend.services.gemini_service import GeminiService, gemini_service
from backend.services.provider_errors import ProviderError, ProviderUnavailableError
from backend.services.quanta_service import MODEL_DIR, QuantaService, quanta_service
from backend.services.teacher_service import TeacherService, teacher_service


logger = logging.getLogger(__name__)


class Provider(Protocol):
    name: str

    def is_available(self) -> bool: ...

    def generate(
        self,
        message: str,
        agent: str,
        conversation_history: Sequence[Mapping[str, str]] | None = None,
        *,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, str]: ...


class OpenRouterModelProvider:
    def __init__(self, service: TeacherService, name: str, model: str):
        self.service = service
        self.name = name
        self.model = model.strip()

    def is_available(self) -> bool:
        return bool(self.model and self.service.is_available())

    def generate(
        self,
        message: str,
        agent: str,
        conversation_history: Sequence[Mapping[str, str]] | None = None,
        *,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, str]:
        return self.service.generate(
            message,
            agent,
            conversation_history,
            system_prompt=system_prompt,
            model_override=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
        )


class QuantaLMProvider:
    name = "QuantaLM Offline"
    is_local = True

    def __init__(self, service: QuantaService):
        self.service = service

    def is_available(self) -> bool:
        availability_check = getattr(type(self.service), "is_available", None)
        if callable(availability_check):
            return bool(availability_check(self.service))
        return bool(
            self.service.loaded
            or (
                (MODEL_DIR / "tokenizer" / "vocab.json").is_file()
                and (MODEL_DIR / "checkpoints" / "quantalm.pth").is_file()
            )
        )

    def generate(
        self,
        message: str,
        agent: str,
        conversation_history: Sequence[Mapping[str, str]] | None = None,
        *,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, str]:
        prompt = build_offline_prompt(
            agent,
            message,
            conversation_history=[dict(entry) for entry in conversation_history or ()],
            system_prompt=system_prompt,
        )
        try:
            token_limit = 900 if agent == "Planner" else 80
            reply = self.service.generate(
                prompt,
                max_new_tokens=min(max_tokens, token_limit),
            )
        except Exception:
            raise ProviderError(
                "QuantaLM generation failed.",
                provider=self.name,
                category="generation_error",
            ) from None

        if not isinstance(reply, str) or not reply.strip():
            raise ProviderError(
                "QuantaLM returned an empty response.",
                provider=self.name,
                category="invalid_response",
            )
        is_reliable = self.service.is_response_reliable(message, reply)
        return {
            "reply": reply.strip(),
            "provider": self.name,
            "model": "QuantaLM",
            "agent": agent,
            "status": "success",
            "reliability": "reliable" if is_reliable else "unreliable",
        }


class ProviderManager:
    AGENT_PROMPTS = AGENT_PROMPTS

    def __init__(self, providers: Sequence[Provider]):
        self.providers = tuple(providers)

    def available_provider_names(self) -> list[str]:
        return [provider.name for provider in self.providers if provider.is_available()]

    def online_provider_names(self) -> list[str]:
        return [
            provider.name
            for provider in self.providers
            if not getattr(provider, "is_local", False) and provider.is_available()
        ]

    def is_available(self) -> bool:
        return bool(self.online_provider_names())

    def generate(
        self,
        message: str,
        agent: str = "Study",
        conversation_history: Sequence[Mapping[str, str]] | None = None,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 1500,
        temperature: float = 0.4,
    ) -> dict[str, str]:
        if agent not in AGENT_PROMPTS:
            raise ValueError(f"Unknown agent: {agent}")

        prompt = system_prompt or AGENT_PROMPTS[agent]
        attempted = False
        for provider in self.providers:
            if not provider.is_available():
                logger.info(
                    "AI provider skipped provider=%s model=%s reason=not_configured",
                    provider.name,
                    getattr(provider, "model", ""),
                )
                continue
            attempted = True
            try:
                result = provider.generate(
                    message,
                    agent,
                    conversation_history,
                    system_prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except ProviderError as error:
                diagnostic = error.diagnostic
                if isinstance(provider, OpenRouterModelProvider):
                    diagnostic = TeacherService._redact_diagnostic(
                        diagnostic,
                        provider.service.api_key,
                    )
                logger.warning(
                    "AI provider failed provider=%s model=%s category=%s "
                    "http_status=%s diagnostic=%s",
                    provider.name,
                    getattr(provider, "model", ""),
                    error.category,
                    error.status_code,
                    diagnostic or "-",
                )
                continue

            reply = result.get("reply") if isinstance(result, dict) else None
            model = result.get("model") if isinstance(result, dict) else None
            if not isinstance(reply, str) or not reply.strip() or not isinstance(model, str):
                continue
            return {
                **result,
                "reply": reply.strip(),
                "provider": provider.name,
                "model": model,
                "agent": agent,
                "status": "success",
            }

        if attempted:
            raise ProviderUnavailableError(
                "All configured AI providers failed, including QuantaLM."
            )
        raise ProviderUnavailableError("No AI provider is configured or available.")

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
            raise ProviderError(
                "The provider returned an invalid YouTube search query.",
                provider=result["provider"],
                category="invalid_response",
            )
        return query


def select_provider(mode: str) -> str:
    if mode == "offline":
        return "quanta"
    if mode == "teacher":
        return "teacher"
    if mode != "auto":
        raise ValueError(f"Unknown provider mode: {mode}")

    provider = os.getenv("QUANTAAI_AUTO_PROVIDER", "teacher").strip().lower()
    if provider not in {"teacher", "quanta"}:
        raise ValueError("Invalid QUANTAAI_AUTO_PROVIDER configuration.")
    return provider


def build_provider_manager(
    teacher: TeacherService = teacher_service,
    gemini: GeminiService = gemini_service,
    quanta: QuantaService = quanta_service,
    secondary_teacher: TeacherService | None = None,
) -> ProviderManager:
    secondary_model = os.getenv("OPENROUTER_MODEL_SECONDARY", "").strip()
    if secondary_teacher is None:
        secondary_teacher = TeacherService(
            api_key=os.getenv("OPENROUTER_API_KEY_SECONDARY", ""),
            model=secondary_model,
        )
    return ProviderManager(
        (
            OpenRouterModelProvider(teacher, "OpenRouter Primary", teacher.model),
            OpenRouterModelProvider(
                secondary_teacher,
                "OpenRouter Secondary",
                secondary_model,
            ),
            gemini,
            QuantaLMProvider(quanta),
        )
    )


provider_manager = build_provider_manager()