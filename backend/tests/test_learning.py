import json
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from backend.documents import StudyDocument
from backend.learning import Flashcard, FlashcardDeck, QuizAttempt
from backend.main import app
from backend.services.provider_manager import provider_manager
from backend.services.quanta_service import quanta_service
from backend.tests.auth_support import AuthTestCase


def quiz_payload(question_count=3):
    questions = [
        "What condition stops a recursive call from continuing?",
        "How does each recursive call reduce the original problem?",
        "Which situation is best solved by splitting work recursively?",
    ]
    return json.dumps(
        {
            "questions": [
                {
                    "question": questions[index],
                    "choices": ["Correct", "Alternative A", "Alternative B", "Alternative C"],
                    "correct_index": 0,
                    "explanation": f"Explanation {index + 1}.",
                }
                for index in range(question_count)
            ]
        }
    )


def flashcard_payload(card_count=3):
    return json.dumps(
        {
            "cards": [
                {
                    "front": f"Term {index + 1}",
                    "back": f"Definition {index + 1}",
                }
                for index in range(card_count)
            ]
        }
    )


class LearningApiTests(AuthTestCase):
    def test_topic_quiz_can_be_submitted_and_appears_in_history_and_progress(self):
        self.register_account()
        with patch.object(
            provider_manager,
            "generate",
            return_value={
                "reply": quiz_payload(),
                "provider": "Google Gemini",
                "model": "gemini-test-model",
            },
        ) as generate:
            response = self.client.post(
                "/learning/quizzes",
                json={
                    "topic": "Recursion",
                    "question_count": 3,
                    "mode": "teacher",
                },
            )

        self.assertEqual(response.status_code, 201)
        quiz = response.json()
        self.assertEqual(quiz["topic"], "Recursion")
        self.assertEqual(quiz["provider"], "Google Gemini")
        self.assertEqual(quiz["difficulty"], "basic")
        self.assertIsNone(quiz["document_id"])
        self.assertEqual(len(quiz["questions"]), 3)
        self.assertNotIn("correct_index", quiz["questions"][0])
        self.assertIn("recursion", generate.call_args.args[0].lower())
        self.assertIn("cover distinct important concepts", generate.call_args.kwargs["system_prompt"])
        self.assertIn("do not reproduce or copy flashcard", generate.call_args.kwargs["system_prompt"])

        result = self.client.post(
            f"/learning/quizzes/{quiz['id']}/submit",
            json={"answers": [1, 0, 0]},
        )
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["score"], 2)
        self.assertEqual(result.json()["question_count"], 3)
        self.assertFalse(result.json()["feedback"][0]["correct"])
        self.assertEqual(result.json()["feedback"][0]["correct_index"], 0)
        self.assertEqual(
            self.client.post(
                f"/learning/quizzes/{quiz['id']}/submit",
                json={"answers": [0, 0, 0]},
            ).status_code,
            409,
        )

        history = self.client.get("/learning/quizzes")
        self.assertEqual(history.status_code, 200)
        self.assertEqual(history.json()["quizzes"][0]["score"], 2)
        progress = self.client.get("/learning/progress")
        self.assertEqual(progress.status_code, 200)
        self.assertEqual(progress.json()["completed_quizzes"], 1)
        self.assertEqual(progress.json()["average_score"], 66.7)
        self.assertEqual(progress.json()["weak_topics"][0]["topic"], "Recursion")
        self.assertIn("Recursion", progress.json()["recommendations"][0]["message"])
        self.assertEqual(
            [item["type"] for item in progress.json()["recent_activity"]],
            ["quiz_completed", "quiz_created"],
        )
        with self.session_factory() as session:
            stored = session.query(QuizAttempt).one()
            self.assertEqual(stored.answers, [1, 0, 0])
            self.assertEqual(stored.score, 2)

    def test_document_quiz_uses_owned_rag_context(self):
        self.register_account()
        user_id = self.client.get("/auth/me").json()["id"]
        with self.session_factory() as session:
            document = StudyDocument(
                user_id=user_id,
                filename="recursion-notes.pdf",
                size_bytes=128,
                page_count=1,
                pages=[
                    "Key concepts and definitions: recursion is when a function "
                    "calls itself and has a base case."
                ],
                status="indexed",
            )
            session.add(document)
            session.commit()
            session.refresh(document)
            document_id = document.id

        with patch.object(
            provider_manager,
            "generate",
            return_value={
                "reply": quiz_payload(),
                "provider": "OpenRouter Primary",
                "model": "primary-test-model",
            },
        ) as generate:
            response = self.client.post(
                "/learning/quizzes",
                json={
                    "document_id": document_id,
                    "question_count": 3,
                    "difficulty": "medium",
                },
            )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["document_id"], document_id)
        self.assertEqual(response.json()["difficulty"], "medium")
        self.assertIn("recursion-notes.pdf", response.json()["topic"])
        self.assertIn("Retrieved PDF excerpts", generate.call_args.args[0])
        self.assertIn("recursion is when a function", generate.call_args.args[0])
        self.assertIn("analytical reasoning", generate.call_args.kwargs["system_prompt"])
        self.assertEqual(
            self.client.get("/learning/quizzes").json()["quizzes"][0]["document_id"],
            document_id,
        )

    def test_quiz_difficulty_changes_generation_instructions_and_is_persisted(self):
        self.register_account()
        for difficulty, expected_instruction in (
            ("basic", "accessible to a beginner"),
            ("medium", "analytical reasoning"),
        ):
            with patch.object(
                provider_manager,
                "generate",
                return_value={
                    "reply": quiz_payload(),
                    "provider": "Google Gemini",
                    "model": "gemini-test-model",
                },
            ) as generate:
                response = self.client.post(
                    "/learning/quizzes",
                    json={
                        "topic": "Recursion",
                        "question_count": 3,
                        "difficulty": difficulty,
                    },
                )
            self.assertEqual(response.status_code, 201)
            self.assertEqual(response.json()["difficulty"], difficulty)
            self.assertIn(
                expected_instruction,
                generate.call_args.kwargs["system_prompt"],
            )

        rejected = self.client.post(
            "/learning/quizzes",
            json={"topic": "Recursion", "difficulty": "hard"},
        )
        self.assertEqual(rejected.status_code, 422)
        self.assertEqual(
            [item["difficulty"] for item in self.client.get(
                "/learning/quizzes"
            ).json()["quizzes"]],
            ["medium", "basic"],
        )

    def test_quiz_requires_a_topic_or_document_and_rejects_bad_provider_output(self):
        self.register_account()
        no_source = self.client.post("/learning/quizzes", json={})
        self.assertEqual(no_source.status_code, 422)

        with patch.object(
            provider_manager,
            "generate",
            return_value={
                "reply": "Not JSON",
                "provider": "OpenRouter Primary",
                "model": "primary-test-model",
            },
        ):
            invalid_output = self.client.post(
                "/learning/quizzes",
                json={"topic": "Algorithms", "question_count": 3},
            )
        self.assertEqual(invalid_output.status_code, 502)
        with self.session_factory() as session:
            self.assertEqual(session.query(QuizAttempt).count(), 0)

    def test_quiz_rejects_repetitive_questions(self):
        self.register_account()
        repeated = json.loads(quiz_payload())
        repeated["questions"][1]["question"] = repeated["questions"][0]["question"]
        with patch.object(
            provider_manager,
            "generate",
            return_value={
                "reply": json.dumps(repeated),
                "provider": "Google Gemini",
                "model": "gemini-test-model",
            },
        ):
            response = self.client.post(
                "/learning/quizzes",
                json={"topic": "Recursion", "question_count": 3},
            )
        self.assertEqual(response.status_code, 502)
        with self.session_factory() as session:
            self.assertEqual(session.query(QuizAttempt).count(), 0)

    def test_offline_quiz_uses_only_quantalmlocal_generation(self):
        self.register_account()
        with (
            patch.object(
                quanta_service,
                "generate",
                return_value=quiz_payload(),
            ) as local_generate,
            patch.object(provider_manager, "generate") as online_generate,
        ):
            response = self.client.post(
                "/learning/quizzes",
                json={
                    "topic": "Data structures",
                    "question_count": 3,
                    "mode": "offline",
                },
            )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["provider"], "QuantaLM Offline")
        local_generate.assert_called_once()
        online_generate.assert_not_called()

    def test_flashcard_deck_progress_is_persistent_and_user_scoped(self):
        self.register_account()
        with patch.object(
            provider_manager,
            "generate",
            return_value={
                "reply": flashcard_payload(),
                "provider": "Google Gemini",
                "model": "gemini-test-model",
            },
        ):
            created = self.client.post(
                "/learning/flashcards",
                json={"topic": "Binary trees", "card_count": 3},
            )

        self.assertEqual(created.status_code, 201)
        deck = created.json()
        deck_id = deck["id"]
        self.assertEqual(len(deck["cards"]), 3)
        self.assertEqual(deck["cards"][0]["front"], "Term 1")
        self.assertEqual(deck["cards"][0]["back"], "Definition 1")
        self.assertEqual(deck["title"], "Binary trees")
        self.assertEqual(deck["topic"], "Binary trees")
        self.assertEqual(deck["difficulty"], "basic")
        self.assertEqual(deck["source_type"], "topic")
        self.assertEqual(deck["total_cards"], 3)
        self.assertEqual(deck["cards"][0]["times_reviewed"], 0)
        self.assertEqual(deck["cards"][0]["times_remembered"], 0)

        listed_decks = self.client.get("/learning/flashcards")
        self.assertEqual(listed_decks.status_code, 200)
        self.assertEqual(listed_decks.json()["decks"][0]["id"], deck_id)
        self.assertEqual(listed_decks.json()["decks"][0]["difficulty"], "basic")
        retrieved = self.client.get(f"/learning/flashcards/{deck_id}")
        self.assertEqual(retrieved.status_code, 200)
        self.assertEqual(len(retrieved.json()["cards"]), 3)
        self.assertEqual(retrieved.json()["difficulty"], "basic")

        with self.session_factory() as session:
            stored_deck = session.get(FlashcardDeck, deck_id)
            self.assertEqual(stored_deck.total_cards, 3)
            self.assertEqual(
                [(card.question, card.answer, card.card_order) for card in stored_deck.flashcards],
                [
                    ("Term 1", "Definition 1", 0),
                    ("Term 2", "Definition 2", 1),
                    ("Term 3", "Definition 3", 2),
                ],
            )

        updated_deck = self.client.patch(
            f"/learning/flashcards/{deck_id}",
            json={"title": "Binary tree review", "topic": "Trees"},
        )
        self.assertEqual(updated_deck.status_code, 200)
        self.assertEqual(updated_deck.json()["title"], "Binary tree review")
        self.assertEqual(updated_deck.json()["topic"], "Trees")

        card_id = deck["cards"][0]["id"]
        remembered = self.client.post(
            f"/learning/flashcards/{deck_id}/cards/{card_id}/review",
            json={"remembered": True},
        )
        needs_revision = self.client.post(
            f"/learning/flashcards/{deck_id}/cards/{card_id}/review",
            json={"remembered": False},
        )
        self.assertEqual(remembered.status_code, 200)
        self.assertEqual(needs_revision.status_code, 200)
        self.assertEqual(needs_revision.json()["times_reviewed"], 2)
        self.assertEqual(needs_revision.json()["times_remembered"], 1)
        with self.session_factory() as session:
            reviewed_card = session.get(Flashcard, card_id)
            self.assertEqual(reviewed_card.times_reviewed, 2)
            self.assertEqual(reviewed_card.times_remembered, 1)

        updated = self.client.patch(
            f"/learning/flashcards/{deck_id}/progress",
            json={"current_index": 1, "known": True},
        )
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.json()["current_index"], 1)
        self.assertEqual(updated.json()["known_cards"], [1])
        listed = self.client.get("/learning/flashcards")
        self.assertEqual(listed.json()["decks"][0]["known_cards"], [1])

        self.assertEqual(
            self.client.patch(
                f"/learning/flashcards/{deck_id}/progress",
                json={"current_index": 9},
            ).status_code,
            422,
        )
        self.assertEqual(
            self.client.delete(f"/learning/flashcards/{deck_id}").status_code,
            204,
        )
        self.assertEqual(self.client.get("/learning/flashcards").json()["decks"], [])
        progress = self.client.get("/learning/progress").json()
        self.assertEqual(progress["completed_quizzes"], 0)
        self.assertEqual(
            [item["type"] for item in progress["recent_activity"]],
            [
                "flashcard_reviewed",
                "flashcard_reviewed",
                "flashcard_reviewed",
                "flashcards_created",
            ],
        )
        with self.session_factory() as session:
            self.assertEqual(session.query(FlashcardDeck).count(), 0)
            self.assertEqual(session.query(Flashcard).count(), 0)

    def test_flashcard_decks_are_private_and_failed_persistence_rolls_back(self):
        self.register_account()
        with patch.object(
            provider_manager,
            "generate",
            return_value={
                "reply": flashcard_payload(),
                "provider": "Google Gemini",
                "model": "gemini-test-model",
            },
        ):
            created = self.client.post(
                "/learning/flashcards",
                json={"topic": "Private deck", "card_count": 3},
            )
        deck_id = created.json()["id"]
        other_client = TestClient(app)
        csrf = other_client.get("/auth/csrf").json()["csrf_token"]
        other_client.headers.update({"X-CSRF-Token": csrf})
        registration = other_client.post(
            "/auth/register",
            json={
                "name": "Other Learner",
                "email": "other-flashcard@example.com",
                "password": "OtherPassword123!",
            },
        )
        self.assertEqual(registration.status_code, 201)
        try:
            self.assertEqual(other_client.get("/learning/flashcards").json()["decks"], [])
            self.assertEqual(other_client.get(f"/learning/flashcards/{deck_id}").status_code, 404)
            self.assertEqual(
                other_client.patch(
                    f"/learning/flashcards/{deck_id}",
                    json={"title": "Not mine"},
                ).status_code,
                404,
            )
            card_id = created.json()["cards"][0]["id"]
            self.assertEqual(
                other_client.post(
                    f"/learning/flashcards/{deck_id}/cards/{card_id}/review",
                    json={"remembered": True},
                ).status_code,
                404,
            )
        finally:
            other_client.close()

        with (
            patch.object(
                provider_manager,
                "generate",
                return_value={
                    "reply": flashcard_payload(),
                    "provider": "Google Gemini",
                    "model": "gemini-test-model",
                },
            ),
            patch("backend.learning.Session.commit", side_effect=SQLAlchemyError("write failed")),
        ):
            failed = self.client.post(
                "/learning/flashcards",
                json={"topic": "Rollback deck", "card_count": 3},
            )
        self.assertEqual(failed.status_code, 503)
        with self.session_factory() as session:
            self.assertEqual(session.query(FlashcardDeck).count(), 1)
            self.assertEqual(session.query(Flashcard).count(), 3)

    def test_topic_flashcards_use_general_knowledge_without_document_context(self):
        self.register_account()
        with patch.object(
            provider_manager,
            "generate",
            return_value={
                "reply": flashcard_payload(),
                "provider": "Google Gemini",
                "model": "gemini-test-model",
            },
        ) as generate:
            response = self.client.post(
                "/learning/flashcards",
                json={"topic": "Binary trees", "card_count": 3, "mode": "teacher"},
            )

        self.assertEqual(response.status_code, 201)
        prompt = generate.call_args.args[0]
        self.assertIn("Binary trees", prompt)
        self.assertNotIn("Retrieved PDF excerpts", prompt)
        system_prompt = generate.call_args.kwargs["system_prompt"]
        self.assertIn("BASIC", system_prompt)
        self.assertIn("identify its important subtopics", system_prompt)

    def test_medium_flashcards_request_conceptual_coverage_and_persist_difficulty(self):
        self.register_account()
        medium_cards = {
            "cards": [
                {
                    "front": "How does a base case prevent unbounded recursion?",
                    "back": "It stops further calls once the smallest input is reached.",
                },
                {
                    "front": "How would you trace the call stack for a recursive tree traversal?",
                    "back": "Track each pending call and return value as child nodes are visited.",
                },
                {
                    "front": "When is an iterative traversal preferable to recursion?",
                    "back": "Use iteration when stack depth could exceed limits or explicit stack control is needed.",
                },
            ]
        }
        with patch.object(
            provider_manager,
            "generate",
            return_value={
                "reply": json.dumps(medium_cards),
                "provider": "Google Gemini",
                "model": "gemini-test-model",
            },
        ) as generate:
            response = self.client.post(
                "/learning/flashcards",
                json={
                    "topic": "Tree traversal",
                    "card_count": 3,
                    "difficulty": "medium",
                },
            )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["difficulty"], "medium")
        self.assertIn("When is an iterative traversal preferable", response.json()["cards"][2]["front"])
        message = generate.call_args.args[0]
        system_prompt = generate.call_args.kwargs["system_prompt"]
        self.assertIn("MEDIUM difficulty", message)
        self.assertIn("comparisons, applications", system_prompt)
        self.assertIn("Avoid beginner-only definition recall", system_prompt)
        self.assertIn("highlight important concepts", system_prompt)
        listed = self.client.get("/learning/flashcards").json()["decks"][0]
        self.assertEqual(listed["difficulty"], "medium")
        unsupported = self.client.post(
            "/learning/flashcards",
            json={
                "topic": "Tree traversal",
                "card_count": 3,
                "difficulty": "hard",
            },
        )
        self.assertEqual(unsupported.status_code, 422)
        generate.assert_called_once()

    def test_flashcard_generation_rejects_repeated_questions_without_saving(self):
        self.register_account()
        repeated = json.dumps(
            {
                "cards": [
                    {"front": "What is recursion?", "back": "A function calls itself."},
                    {"front": "What is recursion!", "back": "A function calls itself."},
                    {"front": "How does a base case stop recursion?", "back": "It ends calls."},
                ]
            }
        )
        with patch.object(
            provider_manager,
            "generate",
            return_value={
                "reply": repeated,
                "provider": "Google Gemini",
                "model": "gemini-test-model",
            },
        ):
            response = self.client.post(
                "/learning/flashcards",
                json={"topic": "Recursion", "card_count": 3},
            )

        self.assertEqual(response.status_code, 502)
        self.assertIn("repetitive", response.json()["detail"])
        self.assertEqual(self.client.get("/learning/flashcards").json()["decks"], [])

    def test_document_flashcards_are_generated_from_owned_pdf_content(self):
        self.register_account()
        user_id = self.client.get("/auth/me").json()["id"]
        pdf_text = (
            "The Orion observatory stores telescope images in the Aurora archive."
        )
        with self.session_factory() as session:
            document = StudyDocument(
                user_id=user_id,
                filename="orion-notes.pdf",
                size_bytes=128,
                page_count=1,
                pages=[pdf_text],
                status="indexed",
            )
            session.add(document)
            session.commit()
            session.refresh(document)
            document_id = document.id

        with patch.object(
            provider_manager,
            "generate",
            return_value={
                "reply": flashcard_payload(),
                "provider": "OpenRouter Primary",
                "model": "openrouter-test-model",
            },
        ) as generate:
            response = self.client.post(
                "/learning/flashcards",
                json={
                    "document_id": document_id,
                    "card_count": 3,
                    "difficulty": "medium",
                    "mode": "teacher",
                },
            )

        self.assertEqual(response.status_code, 201)
        prompt = generate.call_args.args[0]
        self.assertIn("Aurora archive", prompt)
        self.assertIn("[Page 1]", prompt)
        self.assertIn("Retrieved PDF excerpts", prompt)
        self.assertIn("MEDIUM difficulty", prompt)
        with self.session_factory() as session:
            deck = session.query(FlashcardDeck).one()
            self.assertEqual(deck.document_id, document_id)
            self.assertEqual(deck.legacy_cards[0]["difficulty"], "medium")

    def test_learning_mutations_require_csrf(self):
        self.register_account()
        csrf_header = self.client.headers.pop("X-CSRF-Token")
        quiz = self.client.post(
            "/learning/quizzes",
            json={"topic": "Algorithms", "question_count": 3},
        )
        deck = self.client.post(
            "/learning/flashcards",
            json={"topic": "Algorithms", "card_count": 3},
        )
        self.client.headers["X-CSRF-Token"] = csrf_header

        self.assertEqual(quiz.status_code, 403)
        self.assertEqual(deck.status_code, 403)
