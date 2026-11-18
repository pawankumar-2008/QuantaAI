from unittest.mock import patch

from backend.services.provider_manager import provider_manager
from backend.services.quanta_service import quanta_service
from backend.tests.auth_support import AuthTestCase


class CollaborationApiTests(AuthTestCase):
    def test_synthesis_uses_real_agent_contributions_and_reports_provider(self):
        self.register_account()
        history = [{"role": "user", "content": "Explain trees"}]
        with patch.object(
            provider_manager,
            "generate",
            return_value={
                "reply": "Both agents agree that a tree is hierarchical.",
                "provider": "Google Gemini",
                "model": "gemini-test-model",
            },
        ) as generate:
            response = self.client.post(
                "/collaboration/synthesize",
                json={
                    "message": "Compare these explanations.",
                    "contributions": [
                        {
                            "agent": "Study",
                            "response": "A tree has a root and connected branches.",
                        },
                        {
                            "agent": "Coding",
                            "response": "A tree is a connected acyclic graph.",
                        },
                    ],
                    "failed_agents": ["Career"],
                    "mode": "teacher",
                    "conversation_history": history,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "reply": "Both agents agree that a tree is hierarchical.",
                "provider": "Google Gemini",
                "model": "gemini-test-model",
            },
        )
        self.assertEqual(generate.call_args.args[1], "Guide")
        self.assertEqual(generate.call_args.args[2], history)
        self.assertIn("A tree has a root", generate.call_args.args[0])
        self.assertIn("A tree is a connected", generate.call_args.args[0])
        self.assertIn("Career", generate.call_args.args[0])
        self.assertIn("untrusted reference data", generate.call_args.kwargs["system_prompt"])

    def test_offline_synthesis_uses_only_quantalmlocal(self):
        self.register_account()
        with (
            patch.object(
                quanta_service,
                "generate",
                return_value="The agents' ideas share a common theme.",
            ) as local_generate,
            patch.object(provider_manager, "generate") as online_generate,
        ):
            response = self.client.post(
                "/collaboration/synthesize",
                json={
                    "message": "Combine the answers.",
                    "contributions": [
                        {"agent": "Study", "response": "First contribution."},
                        {"agent": "Exam", "response": "Second contribution."},
                    ],
                    "mode": "offline",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["provider"], "QuantaLM Offline")
        local_generate.assert_called_once()
        online_generate.assert_not_called()

    def test_synthesis_validates_contributions_and_requires_csrf(self):
        self.register_account()
        body = {
            "message": "Combine the answers.",
            "contributions": [
                {"agent": "Study", "response": "A contribution."},
                {"agent": "Study", "response": "A duplicate contribution."},
            ],
        }
        duplicate_agent = self.client.post("/collaboration/synthesize", json=body)
        self.assertEqual(duplicate_agent.status_code, 422)

        csrf_header = self.client.headers.pop("X-CSRF-Token")
        valid_body = {
            "message": "Combine the answers.",
            "contributions": [
                {"agent": "Study", "response": "First contribution."},
                {"agent": "Exam", "response": "Second contribution."},
            ],
        }
        missing_csrf = self.client.post(
            "/collaboration/synthesize",
            json=valid_body,
        )
        self.client.headers["X-CSRF-Token"] = csrf_header
        self.assertEqual(missing_csrf.status_code, 403)
