from backend.agents.profiles import STUDY_TUTOR_MODE_PROMPTS
from backend.services.provider_manager import ProviderManager, provider_manager


class StudyAgent:
    def __init__(self, provider: ProviderManager = provider_manager):
        self.name = "Study Agent"
        self.provider = provider

    def generate(
        self,
        message: str,
        level: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        document_context: str | None = None,
        tutor_mode: str = "direct",
    ) -> dict[str, str]:
        message = message.strip()
        if not message:
            raise ValueError("Please enter a topic or question you want to learn.")
        if tutor_mode not in STUDY_TUTOR_MODE_PROMPTS:
            raise ValueError("Unknown Study tutor mode.")

        if level is not None and level not in {
            "beginner",
            "intermediate",
            "advanced",
        }:
            level = "beginner"

        system_prompt = (
            f"{self.provider.AGENT_PROMPTS['Study']}\n\n"
            f"{STUDY_TUTOR_MODE_PROMPTS[tutor_mode]}"
        )
        if level is not None:
            system_prompt = f"{system_prompt}\nRequested learning level: {level}."
        if document_context is not None:
            system_prompt = f"{system_prompt}\n\n{document_context}"

        return self.provider.generate(
            message,
            "Study",
            conversation_history,
            system_prompt=system_prompt,
        )

    def respond(
        self,
        message: str,
        level: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> str:
        return self.generate(message, level, conversation_history)["reply"]
