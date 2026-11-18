import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.documents import (
    DocumentProcessingError,
    MAX_DOCUMENT_FILE_BYTES,
    StudyDocument,
    extract_pdf_pages,
    get_study_document_context,
)
from backend.learning import FlashcardDeck, QuizAttempt
from backend.main import app, provider_manager, study_agent
from backend.services.quanta_service import quanta_service
from backend.tests.auth_support import AuthTestCase


PDF_TEXT = (
    "The Orion observatory stores telescope images in the Aurora archive."
)


def make_text_pdf(text: str = PDF_TEXT) -> bytes:
    escaped_text = (
        text.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped_text}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        (
            b"<< /Length "
            + str(len(stream)).encode("ascii")
            + b" >>\nstream\n"
            + stream
            + b"\nendstream"
        ),
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{number} 0 obj\n".encode("ascii"))
        pdf.extend(body + b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(pdf)


class StudyDocumentApiTests(AuthTestCase):
    def setUp(self):
        super().setUp()
        response = self.register_account()
        self.assertEqual(response.status_code, 201)

    def upload_pdf(self, filename="lesson.pdf", content=None):
        return self.client.post(
            "/study-documents",
            files={
                "file": (
                    filename,
                    make_text_pdf() if content is None else content,
                    "application/pdf",
                )
            },
        )

    def test_upload_indexes_pdf_and_study_chat_receives_relevant_context(self):
        uploaded = self.upload_pdf()
        self.assertEqual(uploaded.status_code, 201)
        document = uploaded.json()
        self.assertEqual(document["status"], "indexed")
        self.assertTrue(document["original_file_available"])
        self.assertEqual(document["filename"], "lesson.pdf")
        self.assertEqual(document["page_count"], 1)

        listed = self.client.get("/study-documents")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual([item["id"] for item in listed.json()], [document["id"]])

        history = [
            {"role": "user", "content": "What does the report cover?"},
            {"role": "assistant", "content": "It covers telescope storage."},
        ]
        result = {
            "reply": (
                "The images are stored in the Aurora archive.\n"
                "[[QUANTAAI_SOURCE:uploaded_pdf]]"
            ),
            "model": "mock-study-model",
            "provider": "OpenRouter Secondary",
        }
        with (
            patch.object(provider_manager, "generate", return_value=result) as generate,
            patch.object(provider_manager, "generate_youtube_query", return_value=None),
        ):
            response = self.client.post(
                "/chat",
                json={
                    "message": "Where are telescope images stored?",
                    "agent": "Study",
                    "mode": "teacher",
                    "conversation_history": history,
                    "document_id": document["id"],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["reply"],
            "The images are stored in the Aurora archive.",
        )
        self.assertEqual(response.json()["document_source"], "uploaded_pdf")
        self.assertEqual(response.json()["document_pages"], [1])
        self.assertEqual(response.json()["model"], result["model"])
        self.assertEqual(response.json()["provider"], result["provider"])
        self.assertEqual(
            generate.call_args.args[:3],
            ("Where are telescope images stored?", "Study", history),
        )
        context = generate.call_args.kwargs["system_prompt"]
        self.assertIn("Aurora archive", context)
        self.assertIn("[Page 1]", context)
        self.assertIn("untrusted reference data", context)

        with self.session_factory() as session:
            stored = session.get(StudyDocument, document["id"])
            self.assertEqual(stored.pages, [PDF_TEXT])
            self.assertEqual(stored.original_file, make_text_pdf())
        with self.session_factory() as restarted_session:
            reloaded = restarted_session.get(StudyDocument, document["id"])
            self.assertEqual(reloaded.pages, [PDF_TEXT])
            self.assertEqual(reloaded.status, "indexed")
            learning_context = get_study_document_context(
                restarted_session,
                reloaded.user_id,
                document["id"],
                "Where are telescope images stored?",
            )
            self.assertIn("Aurora archive", learning_context)
            self.assertNotIn("QUANTAAI_SOURCE:", learning_context)

    def test_chat_without_a_selected_pdf_uses_normal_study_context(self):
        history = [{"role": "user", "content": "Explain a related topic."}]
        result = {
            "reply": "A general Study Agent response.",
            "model": "mock-study-model",
            "provider": "OpenRouter Primary",
        }
        with (
            patch.object(provider_manager, "generate", return_value=result) as generate,
            patch.object(provider_manager, "generate_youtube_query", return_value=None),
        ):
            response = self.client.post(
                "/chat",
                json={
                    "message": "Explain telescope observatories.",
                    "agent": "Study",
                    "mode": "teacher",
                    "conversation_history": history,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["reply"], result["reply"])
        self.assertIsNone(response.json()["document_source"])
        self.assertEqual(response.json()["document_pages"], [])
        self.assertEqual(generate.call_args.args[:3], (
            "Explain telescope observatories.",
            "Study",
            history,
        ))
        self.assertIn(
            "Tutor interaction mode: Direct.",
            generate.call_args.kwargs["system_prompt"],
        )
        self.assertNotIn(
            "Retrieved PDF excerpts:",
            generate.call_args.kwargs["system_prompt"],
        )

    def test_surface_uses_conversation_pdf_attachments_for_followups(self):
        first = self.upload_pdf(
            "orion-surface.pdf",
            make_text_pdf(
                "The Orion observatory stores telescope images in the Aurora archive."
            ),
        ).json()
        second = self.upload_pdf(
            "vega-surface.pdf",
            make_text_pdf(
                "The Vega observatory stores telescope images in the Nebula vault."
            ),
        ).json()
        conversation_id = self.client.post(
            "/conversations",
            json={"title": "Surface document questions"},
        ).json()["id"]
        attachments = [
            {"id": first["id"], "filename": first["filename"]},
            {"id": second["id"], "filename": second["filename"]},
        ]

        def append_surface_message(content, current_attachments):
            return self.client.post(
                f"/conversations/{conversation_id}/messages",
                json={
                    "messages": [
                        {
                            "role": "user",
                            "agent": "Surface",
                            "content": content,
                            "document_attachments": current_attachments,
                        }
                    ]
                },
            )

        self.assertEqual(
            append_surface_message("Compare these PDFs.", attachments).status_code,
            201,
        )
        replies = [
            {
                "reply": "The documents describe two storage locations.\n"
                "[[QUANTAAI_SOURCE:uploaded_pdf]]",
                "model": "surface-test-model",
                "provider": "OpenRouter Primary",
            },
            {
                "reply": "The PDF names the Nebula vault.\n"
                "[[QUANTAAI_SOURCE:uploaded_pdf]]",
                "model": "surface-test-model",
                "provider": "OpenRouter Primary",
            },
            {
                "reply": "The Orion PDF names the Aurora archive.\n"
                "[[QUANTAAI_SOURCE:uploaded_pdf]]",
                "model": "surface-test-model",
                "provider": "OpenRouter Primary",
            },
            {
                "reply": "I can help with that.",
                "model": "surface-test-model",
                "provider": "OpenRouter Primary",
            },
        ]
        with (
            patch.object(provider_manager, "generate", side_effect=replies) as generate,
            patch.object(provider_manager, "generate_youtube_query") as youtube_query,
            patch.object(study_agent, "generate") as study_generate,
        ):
            compare = self.client.post(
                "/chat",
                json={
                    "message": "Compare where Orion and Vega store telescope images.",
                    "agent": "Surface",
                    "mode": "teacher",
                    "conversation_id": conversation_id,
                },
            )
            self.assertEqual(compare.status_code, 200)
            compare_prompt = generate.call_args.kwargs["system_prompt"]
            self.assertIn("Aurora archive", compare_prompt)
            self.assertIn("Nebula vault", compare_prompt)
            self.assertEqual(generate.call_args.args[1], "Surface")

            append_surface_message("Where are Vega images stored?", attachments)
            follow_up = self.client.post(
                "/chat",
                json={
                    "message": "Where are Vega telescope images stored?",
                    "agent": "Surface",
                    "mode": "teacher",
                    "conversation_id": conversation_id,
                },
            )
            self.assertEqual(follow_up.status_code, 200)
            self.assertIn("Nebula vault", generate.call_args.kwargs["system_prompt"])

            append_surface_message("Focus on Orion only.", [attachments[0]])
            removed = self.client.post(
                "/chat",
                json={
                    "message": "Where are Orion telescope images stored?",
                    "agent": "Surface",
                    "mode": "teacher",
                    "conversation_id": conversation_id,
                },
            )
            self.assertEqual(removed.status_code, 200)
            removed_prompt = generate.call_args.kwargs["system_prompt"]
            self.assertIn("Aurora archive", removed_prompt)
            self.assertNotIn("Nebula vault", removed_prompt)

            unrelated_conversation = self.client.post(
                "/conversations",
                json={"title": "Unrelated Surface chat"},
            ).json()["id"]
            normal = self.client.post(
                "/chat",
                json={
                    "message": "What can you help me with?",
                    "agent": "Surface",
                    "mode": "teacher",
                    "conversation_id": unrelated_conversation,
                },
            )
            self.assertEqual(normal.status_code, 200)
            self.assertNotIn(
                "Retrieved PDF excerpts:",
                generate.call_args.kwargs["system_prompt"],
            )

        youtube_query.assert_not_called()
        study_generate.assert_not_called()
        self.assertEqual(generate.call_count, 4)

    def test_deleted_surface_attachment_does_not_break_conversation_followups(self):
        document = self.upload_pdf(
            "deleted-surface.pdf",
            make_text_pdf("The deleted document contains private project details."),
        ).json()
        conversation_id = self.client.post(
            "/conversations",
            json={"title": "Deleted PDF follow-up"},
        ).json()["id"]
        self.assertEqual(
            self.client.post(
                f"/conversations/{conversation_id}/messages",
                json={
                    "messages": [
                        {
                            "role": "user",
                            "agent": "Surface",
                            "content": "Please review this PDF.",
                            "document_attachments": [
                                {
                                    "id": document["id"],
                                    "filename": document["filename"],
                                }
                            ],
                        }
                    ]
                },
            ).status_code,
            201,
        )
        self.assertEqual(
            self.client.delete(f"/study-documents/{document['id']}").status_code,
            204,
        )
        with (
            patch.object(
                provider_manager,
                "generate",
                return_value={
                    "reply": "I can still help without the deleted PDF.",
                    "model": "surface-test-model",
                    "provider": "OpenRouter Primary",
                },
            ) as generate,
            patch.object(provider_manager, "generate_youtube_query", return_value=None),
        ):
            response = self.client.post(
                "/chat",
                json={
                    "message": "Can you answer another question?",
                    "agent": "Surface",
                    "mode": "teacher",
                    "conversation_id": conversation_id,
                },
            )

        self.assertEqual(response.status_code, 200)
        prompt = generate.call_args.kwargs["system_prompt"]
        self.assertNotIn("Retrieved PDF excerpts:", prompt)
        self.assertNotIn("private project details", prompt)

    def test_selected_pdf_context_is_available_in_offline_study_mode(self):
        uploaded = self.upload_pdf()
        document_id = uploaded.json()["id"]
        with (
            patch.object(
                quanta_service,
                "generate",
                return_value=(
                    "The PDF names the Aurora archive.\n"
                    "[[QUANTAAI_SOURCE:uploaded_pdf]]"
                ),
            ) as generate,
            patch.object(quanta_service, "is_response_reliable", return_value=True),
            patch.object(study_agent, "generate") as online_generate,
        ):
            response = self.client.post(
                "/chat",
                json={
                    "message": "Where are the telescope images stored?",
                    "agent": "Study",
                    "mode": "offline",
                    "document_id": document_id,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["reply"],
            "The PDF names the Aurora archive.",
        )
        self.assertEqual(response.json()["provider"], "QuantaLM Offline")
        self.assertEqual(response.json()["document_source"], "uploaded_pdf")
        self.assertEqual(response.json()["document_pages"], [1])
        self.assertIn("Aurora archive", generate.call_args.args[0])
        self.assertIn("[Page 1]", generate.call_args.args[0])
        online_generate.assert_not_called()

    def test_partial_and_missing_answers_have_distinct_sources(self):
        document = self.upload_pdf().json()
        partial = {
            "reply": (
                "The PDF places the images in the Aurora archive. General knowledge "
                "suggests archives also support long-term preservation.\n"
                "[[QUANTAAI_SOURCE:pdf_ai_knowledge]]"
            ),
            "model": "mock-study-model",
            "provider": "Google Gemini",
        }
        missing = {
            "reply": (
                "Ozone is composed of three oxygen atoms.\n"
                "[[QUANTAAI_SOURCE:ai_knowledge]]"
            ),
            "model": "mock-study-model",
            "provider": "Google Gemini",
        }
        with (
            patch.object(provider_manager, "generate", side_effect=[partial, missing]) as generate,
            patch.object(provider_manager, "generate_youtube_query", return_value=None),
        ):
            partial_response = self.client.post(
                "/chat",
                json={
                    "message": "Where are telescope images stored and why archive them?",
                    "agent": "Study",
                    "mode": "teacher",
                    "document_id": document["id"],
                },
            )
            missing_response = self.client.post(
                "/chat",
                json={
                    "message": "What is the chemical formula for ozone?",
                    "agent": "Study",
                    "mode": "teacher",
                    "document_id": document["id"],
                },
            )

        self.assertEqual(partial_response.status_code, 200)
        self.assertEqual(
            partial_response.json()["document_source"],
            "pdf_ai_knowledge",
        )
        self.assertEqual(partial_response.json()["document_pages"], [1])
        self.assertEqual(missing_response.status_code, 200)
        self.assertEqual(missing_response.json()["document_source"], "ai_knowledge")
        self.assertEqual(missing_response.json()["document_pages"], [])
        self.assertIn("No relevant passages were retrieved", generate.call_args_list[1].kwargs["system_prompt"])

    def test_multiple_documents_retrieve_only_the_selected_pdf_without_reextracting(self):
        first_pdf = make_text_pdf(PDF_TEXT)
        second_text = "The Vega library stores ocean maps in the Tidal catalog."
        second_pdf = make_text_pdf(second_text)
        with patch(
            "backend.documents.extract_pdf_pages",
            wraps=extract_pdf_pages,
        ) as extract:
            first = self.upload_pdf("orion.pdf", first_pdf).json()
            second = self.upload_pdf("vega.pdf", second_pdf).json()
            result = {
                "reply": "Ocean maps are in the Tidal catalog.\n"
                "[[QUANTAAI_SOURCE:uploaded_pdf]]",
                "model": "mock-study-model",
                "provider": "OpenRouter Primary",
            }
            with (
                patch.object(provider_manager, "generate", return_value=result) as generate,
                patch.object(provider_manager, "generate_youtube_query", return_value=None),
            ):
                response = self.client.post(
                    "/chat",
                    json={
                        "message": "Where are ocean maps stored?",
                        "agent": "Study",
                        "mode": "teacher",
                        "document_id": second["id"],
                    },
                )

        self.assertNotEqual(first["id"], second["id"])
        self.assertEqual(extract.call_count, 2)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["document_source"], "uploaded_pdf")
        context = generate.call_args.kwargs["system_prompt"]
        self.assertIn("Tidal catalog", context)
        self.assertNotIn("Aurora archive", context)

    def test_offline_document_failure_does_not_fall_back_to_online_provider(self):
        document = self.upload_pdf().json()
        with (
            patch.object(
                quanta_service,
                "generate",
                side_effect=RuntimeError("local generation unavailable"),
            ),
            patch.object(study_agent, "generate") as online_study,
            patch.object(provider_manager, "generate") as online_provider,
        ):
            response = self.client.post(
                "/chat",
                json={
                    "message": "Where are the telescope images stored?",
                    "agent": "Study",
                    "mode": "offline",
                    "document_id": document["id"],
                },
            )

        self.assertEqual(response.status_code, 502)
        self.assertIn("No online provider was used", response.json()["detail"])
        online_study.assert_not_called()
        online_provider.assert_not_called()

    def test_invalid_and_oversized_pdf_uploads_are_rejected(self):
        wrong_extension = self.upload_pdf("notes.txt")
        self.assertEqual(wrong_extension.status_code, 415)
        self.assertEqual(
            wrong_extension.json()["detail"],
            "Only PDF files can be uploaded.",
        )

        invalid_pdf = self.upload_pdf("notes.pdf", b"not a PDF")
        self.assertEqual(invalid_pdf.status_code, 415)
        self.assertEqual(
            invalid_pdf.json()["detail"],
            "The uploaded file is not a valid PDF.",
        )

        oversized = self.upload_pdf(
            "large.pdf",
            b"%PDF-" + b"x" * MAX_DOCUMENT_FILE_BYTES,
        )
        self.assertEqual(oversized.status_code, 413)
        self.assertEqual(
            oversized.json()["detail"],
            "PDF files must be 10 MB or smaller.",
        )

    def test_processing_failure_is_visible_and_cannot_be_used_for_chat(self):
        with patch(
            "backend.documents.extract_pdf_pages",
            side_effect=DocumentProcessingError(
                "No selectable text was found. Scanned PDFs without selectable text "
                "are not supported."
            ),
        ):
            uploaded = self.upload_pdf()

        self.assertEqual(uploaded.status_code, 201)
        document = uploaded.json()
        self.assertEqual(document["status"], "failed")
        self.assertIn("No selectable text", document["status_message"])

        response = self.client.post(
            "/chat",
            json={
                "message": "Summarize this document.",
                "agent": "Study",
                "document_id": document["id"],
            },
        )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"], document["status_message"])

    def test_unexpected_processing_failure_returns_a_safe_status(self):
        with patch(
            "backend.documents.extract_pdf_pages",
            side_effect=RuntimeError("private parser implementation detail"),
        ):
            uploaded = self.upload_pdf()

        self.assertEqual(uploaded.status_code, 201)
        document = uploaded.json()
        self.assertEqual(document["status"], "failed")
        self.assertEqual(
            document["status_message"],
            "This PDF could not be processed.",
        )
        self.assertNotIn("private parser implementation detail", uploaded.text)

    def test_file_endpoint_reports_missing_original_pdf(self):
        uploaded = self.upload_pdf()
        document_id = uploaded.json()["id"]
        with self.session_factory() as session:
            document = session.get(StudyDocument, document_id)
            document.original_file = None
            session.commit()

        response = self.client.get(f"/study-documents/{document_id}/file")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(
            response.json()["detail"],
            "The original PDF is not available for this document.",
        )

    def test_documents_are_private_and_owner_can_remove_them(self):
        uploaded = self.upload_pdf()
        document_id = uploaded.json()["id"]
        original_pdf = make_text_pdf()
        preview = self.client.get(f"/study-documents/{document_id}/file")
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(preview.headers["content-type"], "application/pdf")
        self.assertIn("inline;", preview.headers["content-disposition"])
        self.assertIn("filename*=UTF-8''lesson.pdf", preview.headers["content-disposition"])
        self.assertEqual(preview.content, original_pdf)

        renamed = self.client.patch(
            f"/study-documents/{document_id}",
            json={"filename": "renamed lesson.pdf"},
        )
        self.assertEqual(renamed.status_code, 200)
        self.assertEqual(renamed.json()["filename"], "renamed lesson.pdf")
        self.assertTrue(renamed.json()["original_file_available"])
        download = self.client.get(
            f"/study-documents/{document_id}/file",
            params={"download": "true"},
        )
        self.assertEqual(download.status_code, 200)
        self.assertEqual(download.headers["content-type"], "application/pdf")
        self.assertIn(
            "attachment;",
            download.headers["content-disposition"],
        )
        self.assertIn(
            "filename*=UTF-8''renamed%20lesson.pdf",
            download.headers["content-disposition"],
        )
        self.assertEqual(download.content, original_pdf)
        self.assertEqual(
            self.client.patch(
                f"/study-documents/{document_id}",
                json={"filename": "not-a-pdf.txt"},
            ).status_code,
            415,
        )

        other_client = TestClient(app)
        try:
            csrf_token = other_client.get("/auth/csrf").json()["csrf_token"]
            other_client.headers.update({"X-CSRF-Token": csrf_token})
            registration = other_client.post(
                "/auth/register",
                json={
                    "name": "Other Learner",
                    "email": "other-learner@example.com",
                    "password": "OtherPassword123!",
                },
            )
            self.assertEqual(registration.status_code, 201)
            self.assertEqual(other_client.get("/study-documents").json(), [])
            self.assertEqual(
                other_client.delete(f"/study-documents/{document_id}").status_code,
                404,
            )
            self.assertEqual(
                other_client.get(
                    f"/study-documents/{document_id}/file"
                ).status_code,
                404,
            )
            self.assertEqual(
                other_client.get(
                    f"/study-documents/{document_id}/file",
                    params={"download": "true"},
                ).status_code,
                404,
            )
            self.assertEqual(
                other_client.patch(
                    f"/study-documents/{document_id}",
                    json={"filename": "private.pdf"},
                ).status_code,
                404,
            )

            response = other_client.post(
                "/chat",
                json={
                    "message": "Summarize this document.",
                    "agent": "Study",
                    "document_id": document_id,
                },
            )
            self.assertEqual(response.status_code, 404)
            self.assertEqual(response.json()["detail"], "Study document not found.")
        finally:
            other_client.close()

        user_id = self.client.get("/auth/me").json()["id"]
        with self.session_factory() as session:
            session.add_all(
                [
                    QuizAttempt(
                        user_id=user_id,
                        document_id=document_id,
                        topic="Document quiz",
                        questions=[
                            {
                                "question": "Q?",
                                "choices": ["A", "B"],
                                "correct_index": 0,
                                "explanation": "Reason",
                            }
                        ],
                        question_count=1,
                        provider="test",
                        model="test",
                    ),
                    FlashcardDeck(
                        user_id=user_id,
                        document_id=document_id,
                        title="Document deck",
                        topic="Document",
                        source_type="pdf",
                        total_cards=1,
                        legacy_cards=[{"front": "Q", "back": "A"}],
                        provider="test",
                        model="test",
                    ),
                ]
            )
            session.commit()

        removed = self.client.delete(f"/study-documents/{document_id}")
        self.assertEqual(removed.status_code, 204)
        self.assertEqual(self.client.get("/study-documents").json(), [])
        with self.session_factory() as session:
            self.assertIsNone(session.query(QuizAttempt).one().document_id)
            self.assertIsNone(session.query(FlashcardDeck).one().document_id)


if __name__ == "__main__":
    unittest.main()
