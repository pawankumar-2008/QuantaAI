from backend.conversations import Conversation, ConversationMessage
from backend.documents import StudyDocument
from backend.main import app
from backend.tests.auth_support import AuthTestCase


class ConversationApiTests(AuthTestCase):
    def setUp(self):
        super().setUp()
        self.assertEqual(self.register_account().status_code, 201)

    def test_conversation_messages_persist_with_agent_attribution_and_timestamps(self):
        created = self.client.post("/conversations", json={"title": "Linear algebra"})
        self.assertEqual(created.status_code, 201)
        conversation_id = created.json()["id"]
        with self.session_factory() as session:
            conversation = session.get(Conversation, conversation_id)
            self.assertEqual(conversation.conversation_type, "chat")
            self.assertIsNone(conversation.agent_id)

        appended = self.client.post(
            f"/conversations/{conversation_id}/messages",
            json={
                "messages": [
                    {"role": "user", "content": "Explain eigenvectors"},
                    {
                        "role": "assistant",
                        "content": "An eigenvector keeps its direction under a transformation.",
                        "agent": "Study",
                        "model": "QuantaLM",
                        "provider": "QuantaLM Offline",
                        "document_source": "uploaded_pdf",
                        "document_pages": [1, 3],
                        "experimental": True,
                        "reliability_warning": True,
                        "search_performed": True,
                        "search_agent": "Research",
                        "sources": [
                            {
                                "id": "S1",
                                "title": "Reference source",
                                "url": "https://example.com/reference",
                                "published_date": "2026-10-15",
                            }
                        ],
                    },
                ]
            },
        )
        self.assertEqual(appended.status_code, 201)
        self.assertEqual(len(appended.json()["messages"]), 2)

        with self.session_factory() as session:
            conversation = session.get(Conversation, conversation_id)
            messages = (
                session.query(ConversationMessage)
                .filter_by(conversation_id=conversation_id)
                .order_by(ConversationMessage.id)
                .all()
            )
            self.assertIsNotNone(conversation.created_at)
            self.assertIsNotNone(conversation.updated_at)
            self.assertEqual(len(messages), 2)
            self.assertEqual(messages[1].agent, "Study")
            self.assertEqual(messages[1].model, "QuantaLM")
            self.assertEqual(messages[1].message_metadata["provider"], "QuantaLM Offline")
            self.assertTrue(messages[1].message_metadata["experimental"])
            self.assertTrue(messages[1].message_metadata["reliability_warning"])
            self.assertEqual(
                messages[1].message_metadata["document_source"],
                "uploaded_pdf",
            )
            self.assertEqual(
                messages[1].message_metadata["document_pages"],
                [1, 3],
            )
            self.assertTrue(messages[1].message_metadata["search_performed"])
            self.assertEqual(messages[1].message_metadata["search_agent"], "Research")
            self.assertIsNotNone(messages[1].created_at)

        detail = self.client.get(f"/conversations/{conversation_id}")
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(
            [message["role"] for message in detail.json()["messages"]],
            ["user", "assistant"],
        )
        self.assertEqual(detail.json()["messages"][1]["provider"], "QuantaLM Offline")
        self.assertEqual(
            detail.json()["messages"][1]["document_source"],
            "uploaded_pdf",
        )
        self.assertEqual(detail.json()["messages"][1]["document_pages"], [1, 3])
        self.assertTrue(detail.json()["messages"][1]["experimental"])
        self.assertTrue(detail.json()["messages"][1]["reliability_warning"])
        self.assertEqual(
            detail.json()["messages"][1]["sources"][0]["url"],
            "https://example.com/reference",
        )
        self.assertEqual(detail.json()["messages"][1]["search_agent"], "Research")

    def test_surface_pdf_attachments_persist_in_message_metadata(self):
        created = self.client.post("/conversations", json={"title": "PDF notes"})
        conversation_id = created.json()["id"]
        user_id = self.client.get("/auth/me").json()["id"]
        with self.session_factory() as session:
            document = StudyDocument(
                user_id=user_id,
                filename="surface-notes.pdf",
                size_bytes=128,
                page_count=1,
                pages=["Surface PDF context."],
                status="indexed",
            )
            session.add(document)
            session.commit()
            session.refresh(document)
            attachment = {"id": document.id, "filename": document.filename}

        appended = self.client.post(
            f"/conversations/{conversation_id}/messages",
            json={
                "messages": [
                    {
                        "role": "user",
                        "agent": "Surface",
                        "content": "Summarize my PDF.",
                        "document_attachments": [attachment],
                    }
                ]
            },
        )
        self.assertEqual(appended.status_code, 201)
        self.assertEqual(
            appended.json()["messages"][0]["document_attachments"],
            [attachment],
        )

        reloaded = self.client.get(f"/conversations/{conversation_id}")
        self.assertEqual(
            reloaded.json()["messages"][0]["document_attachments"],
            [attachment],
        )

    def test_surface_attachments_require_owned_indexed_documents(self):
        created = self.client.post("/conversations", json={"title": "PDF notes"})
        conversation_id = created.json()["id"]
        response = self.client.post(
            f"/conversations/{conversation_id}/messages",
            json={
                "messages": [
                    {
                        "role": "user",
                        "agent": "Surface",
                        "content": "Summarize my PDF.",
                        "document_attachments": [
                            {"id": 9999, "filename": "not-owned.pdf"}
                        ],
                    }
                ]
            },
        )
        self.assertEqual(response.status_code, 404)

    def test_collaboration_summary_can_be_persisted_and_reloaded(self):
        created = self.client.post("/conversations", json={"title": "Collaborative study"})
        conversation_id = created.json()["id"]

        appended = self.client.post(
            f"/conversations/{conversation_id}/messages",
            json={
                "messages": [
                    {
                        "role": "assistant",
                        "content": "The agents agree on the key idea.",
                        "agent": "Collaboration",
                        "model": "gemini-test-model",
                        "provider": "Google Gemini",
                    }
                ]
            },
        )

        self.assertEqual(appended.status_code, 201)
        self.assertEqual(appended.json()["messages"][0]["agent"], "Collaboration")
        self.assertEqual(appended.json()["messages"][0]["provider"], "Google Gemini")
        reloaded = self.client.get(f"/conversations/{conversation_id}")
        self.assertEqual(
            reloaded.json()["messages"][0]["agent"],
            "Collaboration",
        )
        self.assertEqual(
            reloaded.json()["messages"][0]["model"],
            "gemini-test-model",
        )

    def test_conversation_routes_require_authentication(self):
        self.client.cookies.delete("quantaai_session")
        self.assertEqual(self.client.get("/conversations").status_code, 401)
        self.assertEqual(
            self.client.post("/conversations", json={"title": "Private"}).status_code,
            401,
        )

    def test_browser_preflight_allows_conversation_rename_and_delete(self):
        for method in ("PATCH", "DELETE"):
            with self.subTest(method=method):
                response = self.client.options(
                    "/conversations/1",
                    headers={
                        "Origin": "http://localhost:5173",
                        "Access-Control-Request-Method": method,
                        "Access-Control-Request-Headers": "content-type,x-csrf-token",
                    },
                )
                self.assertEqual(response.status_code, 200)
                self.assertIn(
                    method,
                    response.headers["access-control-allow-methods"],
                )
                self.assertEqual(
                    response.headers["access-control-allow-origin"],
                    "http://localhost:5173",
                )

    def test_conversation_is_hidden_from_other_users(self):
        created = self.client.post("/conversations", json={"title": "Private"})
        conversation_id = created.json()["id"]

        other_client = self.make_other_user_client()
        try:
            self.assertEqual(
                other_client.get(f"/conversations/{conversation_id}").status_code,
                404,
            )
            self.assertEqual(other_client.get("/conversations").json()["conversations"], [])
        finally:
            other_client.close()

    def test_message_batches_reject_invalid_roles_and_unknown_agents(self):
        created = self.client.post("/conversations", json={})
        conversation_id = created.json()["id"]

        for message in (
            {"role": "system", "content": "internal"},
            {"role": "assistant", "content": "answer", "agent": "Unknown"},
            {"role": "user", "content": "   "},
        ):
            with self.subTest(message=message):
                response = self.client.post(
                    f"/conversations/{conversation_id}/messages",
                    json={"messages": [message]},
                )
                self.assertEqual(response.status_code, 422)

    def test_delete_removes_only_owned_conversation_and_messages(self):
        created = self.client.post("/conversations", json={"title": "Delete me"})
        conversation_id = created.json()["id"]
        self.client.post(
            f"/conversations/{conversation_id}/messages",
            json={"messages": [{"role": "user", "content": "A message"}]},
        )

        deleted = self.client.delete(f"/conversations/{conversation_id}")
        self.assertEqual(deleted.status_code, 204)
        self.assertEqual(self.client.get(f"/conversations/{conversation_id}").status_code, 404)
        with self.session_factory() as session:
            self.assertIsNone(session.get(Conversation, conversation_id))
            self.assertEqual(
                session.query(ConversationMessage)
                .filter_by(conversation_id=conversation_id)
                .count(),
                0,
            )

    def make_other_user_client(self):
        from fastapi.testclient import TestClient

        client = TestClient(app)
        csrf = client.get("/auth/csrf").json()["csrf_token"]
        client.headers.update({"X-CSRF-Token": csrf})
        response = client.post(
            "/auth/register",
            json={
                "name": "Other Learner",
                "email": "other@example.com",
                "password": "OtherPassword123!",
            },
        )
        self.assertEqual(response.status_code, 201)
        return client
