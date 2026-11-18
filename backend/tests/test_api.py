import os
import io
import json
import unittest
import urllib.error
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from backend.agents.profiles import AGENT_INSTRUCTIONS
from backend.agents.study_agent import StudyAgent
from backend.auth import User
from backend.main import (
    AGENTS,
    _current_planner_date,
    _is_schedule_request,
    _needs_exam_deadline_clarification,
    _parse_schedule_generation,
)
from backend.planner import PlannerTask
from backend.services.quanta_service import quanta_service
from backend.services.provider_errors import ProviderUnavailableError
from backend.services.provider_manager import provider_manager
from backend.services.web_search_service import web_search_service
from backend.services.gemini_service import GeminiProviderError, GeminiService
from backend.services.teacher_service import (
    TeacherProviderError,
    TeacherService,
    teacher_service,
)
from backend.services.youtube_service import youtube_service
from backend.tests.auth_support import AuthTestCase


class BackendApiTests(AuthTestCase):
    def setUp(self):
        super().setUp()
        response = self.register_account()
        self.assertEqual(response.status_code, 201)

    def test_planner_date_uses_asia_kolkata_wall_date(self):
        with patch("backend.main.datetime") as clock:
            clock.now.side_effect = lambda tz: datetime(
                2026,
                11,
                12,
                18,
                30,
                tzinfo=timezone.utc,
            ).astimezone(tz)

            current_date = _current_planner_date()

        self.assertEqual(current_date, date(2026, 11, 13))
        clock.now.assert_called_once()
        self.assertEqual(clock.now.call_args.args[0].key, "Asia/Kolkata")

    def test_planner_structured_json_decodes_html_entities_and_escaped_markdown(self):
        generated = _parse_schedule_generation(
            r'{"tasks":[{"title":"Review &quot;key&quot; topics",'
            r'"description":"Practice \*key\* concepts &amp; examples.",'
            r'"due_date":"2026-11-13"}],"clarification":null}'
        )

        self.assertEqual(generated.tasks[0]["title"], 'Review "key" topics')
        self.assertEqual(
            generated.tasks[0]["description"],
            "Practice *key* concepts & examples.",
        )

    def test_planner_deadline_detection_ignores_historical_milestone_dates(self):
        self.assertTrue(
            _needs_exam_deadline_clarification(
                "Make a full GATE 2027 plan through the exam based on my old "
                "schedule with a 2026-03-01 milestone.",
                [],
            )
        )
        self.assertFalse(
            _needs_exam_deadline_clarification(
                "Make a GATE 2027 plan; the confirmed exam date is "
                "7 February 2027.",
                [],
            )
        )
        self.assertFalse(
            _needs_exam_deadline_clarification(
                "Create a 7-day GATE 2027 study timetable starting today.",
                [],
            )
        )

    def test_planner_detects_natural_schedule_intents(self):
        requests = (
            "Create a GATE preparation plan.",
            "Make a study timetable.",
            "Plan my next 30 days.",
            "Add these activities to my planner.",
            "Schedule DBMS revision for tomorrow.",
            "Create a weekly study routine.",
            "Make a timetable with dates.",
            "Organize my exam preparation.",
            "Add daily PYQ practice.",
            "Plan my mock tests.",
            "Create a schedule for my semester examinations.",
        )
        for request in requests:
            with self.subTest(request=request):
                self.assertTrue(_is_schedule_request(request))

    def test_health_and_agent_routes(self):
        home = self.client.get("/")
        self.assertEqual(home.status_code, 200)
        self.assertEqual(home.json()["teacher"], "Online providers")

        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "healthy")

        agents = self.client.get("/agents")
        self.assertEqual(agents.status_code, 200)
        self.assertEqual(agents.json()["agents"], AGENTS)

    def test_simple_greetings_return_short_replies_without_provider_generation(self):
        greetings = (
            "hi",
            "Hello",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
            "How are you?",
        )
        with (
            patch.object(
                quanta_service,
                "generate",
                side_effect=AssertionError("QuantaLM should not handle greetings"),
            ) as offline_generate,
            patch.object(
                provider_manager,
                "generate",
                side_effect=AssertionError("OpenRouter should not handle greetings"),
            ) as online_generate,
        ):
            for index, message in enumerate(greetings):
                with self.subTest(message=message):
                    response = self.client.post(
                        "/chat",
                        json={
                            "message": message,
                            "mode": ("offline", "auto", "teacher")[index % 3],
                        },
                    )
                    self.assertEqual(response.status_code, 200)
                    self.assertLess(len(response.json()["reply"]), 80)
                    self.assertEqual(response.json()["status"], "success")

        offline_generate.assert_not_called()
        online_generate.assert_not_called()

    def test_context_free_detail_request_asks_for_a_topic(self):
        with (
            patch.object(quanta_service, "generate") as offline_generate,
            patch.object(provider_manager, "generate") as online_generate,
        ):
            response = self.client.post(
                "/chat",
                json={"message": "Explain this topic in detail.", "mode": "offline"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["reply"],
            "What topic would you like me to explain in detail?",
        )
        offline_generate.assert_not_called()
        online_generate.assert_not_called()

    def test_unreliable_local_reply_is_returned_with_a_warning(self):
        noisy_reply = (
            "delivery delivery delivery. protects from its target. "
            "database rules to runnable tasks. memory from its target. "
            "database rules to runnable tasks."
        )
        with (
            patch.object(quanta_service, "generate", return_value=noisy_reply),
            patch.object(
                quanta_service,
                "is_response_reliable",
                return_value=False,
            ),
            patch.object(provider_manager, "generate") as online_generate,
            patch.object(provider_manager, "is_available") as online_available,
        ):
            response = self.client.post(
                "/chat",
                json={
                    "message": "What is the difference between RAM and ROM?",
                    "agent": "Coding",
                    "mode": "offline",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "limited")
        self.assertEqual(response.json()["reply"], noisy_reply)
        self.assertEqual(response.json()["provider"], "QuantaLM Offline")
        self.assertEqual(response.json()["model"], "QuantaLM")
        self.assertTrue(response.json()["experimental"])
        self.assertTrue(response.json()["reliability_warning"])
        online_generate.assert_not_called()
        online_available.assert_not_called()

    def test_relevant_local_reply_is_returned(self):
        relevant_reply = (
            "A DBMS manages databases and provides tools for storing, querying, "
            "and protecting data."
        )
        history = [{"role": "user", "content": "Earlier question"}]
        with (
            patch.object(quanta_service, "generate", return_value=relevant_reply) as generate,
            patch.object(
                quanta_service,
                "is_response_reliable",
                return_value=True,
            ),
        ):
            response = self.client.post(
                "/chat",
                json={
                    "message": "What is DBMS?",
                    "agent": "Coding",
                    "mode": "offline",
                    "conversation_history": history,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "success")
        self.assertEqual(response.json()["reply"], relevant_reply)
        self.assertEqual(response.json()["provider"], "QuantaLM Offline")
        self.assertTrue(response.json()["experimental"])
        self.assertFalse(response.json()["reliability_warning"])
        self.assertIn("[USER] Earlier question", generate.call_args.args[0])

    def test_teacher_mode_supports_all_agents(self):
        with (
            patch.object(teacher_service, "is_available", return_value=True),
            patch.object(
                provider_manager,
                "generate",
                side_effect=lambda message, agent, *args, **kwargs: {
                    "reply": f"mocked {agent} reply",
                    "model": "xingchen-agi/xing4.0-29b-a4b:free",
                    "provider": "OpenRouter Primary",
                },
            ) as generate,
            patch.object(
                provider_manager,
                "generate_youtube_query",
                return_value=None,
            ),
        ):
            for agent in AGENTS:
                with self.subTest(agent=agent):
                    response = self.client.post(
                        "/chat",
                        json={
                            "message": "Help me learn",
                            "agent": agent,
                            "mode": "teacher",
                        },
                    )
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json()["agent"], agent)
                    self.assertEqual(
                        response.json()["model"],
                        "xingchen-agi/xing4.0-29b-a4b:free",
                    )
                    self.assertEqual(response.json()["provider"], "OpenRouter Primary")
                    self.assertEqual(response.json()["status"], "success")
                    self.assertFalse(response.json()["experimental"])
                    self.assertFalse(response.json()["reliability_warning"])
                    self.assertIn(agent, response.json()["reply"])
                    self.assertEqual(generate.call_count, AGENTS.index(agent) + 1)

    def test_web_research_integrates_with_all_agents_and_returns_citations(self):
        message = "Search the web for current Python documentation and verify the release."
        source = {
            "id": "S1",
            "title": "Python documentation",
            "url": "https://docs.python.org/3/",
            "content": "Official current documentation excerpt.",
            "published_date": "2026-10-15",
        }

        def answer(_message, agent, *args, **kwargs):
            prompt = kwargs.get("system_prompt", "")
            self.assertIn("[S1] Python documentation", prompt)
            self.assertIn("conflicting reports", prompt)
            return {
                "reply": f"The source supports this answer [S1]. ({agent})",
                "model": "mock-online-model",
                "provider": "OpenRouter Primary",
            }

        with (
            patch.object(
                web_search_service,
                "search",
                return_value={"status": "success", "sources": [source], "message": ""},
            ) as search,
            patch.object(provider_manager, "generate", side_effect=answer) as generate,
            patch.object(provider_manager, "generate_youtube_query", return_value=None),
        ):
            for agent in AGENTS:
                with self.subTest(agent=agent):
                    response = self.client.post(
                        "/chat",
                        json={"message": message, "agent": agent, "mode": "teacher"},
                    )
                    self.assertEqual(response.status_code, 200)
                    body = response.json()
                    self.assertIn("[S1]", body["reply"])
                    self.assertTrue(body["search_performed"])
                    self.assertEqual(body["search_agent"], agent)
                    self.assertEqual(body["sources"][0]["title"], "Python documentation")
                    self.assertEqual(body["sources"][0]["url"], source["url"])

        self.assertEqual(search.call_count, len(AGENTS))
        self.assertEqual(generate.call_count, len(AGENTS))
        self.assertEqual(
            [call.kwargs["deep"] for call in search.call_args_list],
            [agent == "Research" for agent in AGENTS],
        )

    def test_offline_mode_never_calls_web_search(self):
        with (
            patch.object(web_search_service, "search") as search,
            patch.object(quanta_service, "generate", return_value="Offline answer."),
            patch.object(quanta_service, "is_response_reliable", return_value=True),
        ):
            response = self.client.post(
                "/chat",
                json={
                    "message": "Search the web for current Python documentation.",
                    "agent": "Coding",
                    "mode": "offline",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["reply"], "Offline answer.")
        self.assertFalse(response.json()["search_performed"])
        search.assert_not_called()

    def test_search_failure_does_not_block_online_answer(self):
        with (
            patch.object(
                web_search_service,
                "search",
                return_value={
                    "status": "unavailable",
                    "sources": [],
                    "message": "Web search is not configured.",
                },
            ),
            patch.object(
                provider_manager,
                "generate",
                return_value={
                    "reply": "Here is the best available answer.",
                    "model": "mock-online-model",
                    "provider": "OpenRouter Primary",
                },
            ),
            patch.object(provider_manager, "generate_youtube_query", return_value=None),
        ):
            response = self.client.post(
                "/chat",
                json={
                    "message": "Search the web for current Python documentation.",
                    "agent": "Coding",
                    "mode": "teacher",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["reply"], "Here is the best available answer.")
        self.assertFalse(response.json()["search_performed"])
        self.assertEqual(response.json()["search_message"], "Web search is not configured.")

    def test_teacher_answers_the_requested_educational_questions(self):
        expected_replies = {
            "What is DBMS?": (
                "A DBMS manages databases and provides tools for storing, querying, "
                "and protecting data."
            ),
            "Explain normalization.": (
                "Normalization organizes database tables to reduce duplicated data "
                "and improve consistency."
            ),
            "What is the difference between RAM and ROM?": (
                "RAM is volatile working memory; ROM is non-volatile memory that "
                "retains its contents without power."
            ),
        }
        with (
            patch.object(teacher_service, "is_available", return_value=True),
            patch.object(
                provider_manager,
                "generate",
                side_effect=lambda message, agent, *args, **kwargs: {
                    "reply": expected_replies[message],
                    "model": "mock-teacher",
                    "provider": "OpenRouter Primary",
                },
            ) as generate,
            patch.object(
                provider_manager,
                "generate_youtube_query",
                return_value=None,
            ),
            patch.object(quanta_service, "generate") as offline_generate,
        ):
            for question, expected_reply in expected_replies.items():
                with self.subTest(question=question):
                    response = self.client.post(
                        "/chat",
                        json={"message": question, "mode": "teacher"},
                    )
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json()["reply"], expected_reply)

        self.assertEqual(generate.call_count, len(expected_replies))
        offline_generate.assert_not_called()

    def test_planner_schedule_tasks_use_owned_task_api_and_persist_exact_dates(self):
        structured_tasks = (
            '```json\n{"tasks":['
            '{"title":"Review algebra","description":"Review linear equations.",'
            '"due_date":"2026-11-13"},'
            '{"title":"Practice equations","description":"Solve practice problems.",'
            '"due_date":"2026-11-13"},'
            '{"title":"Review functions","description":"Study function notation.",'
            '"due_date":"2026-11-14"},'
            '{"title":"Take a practice quiz","description":"Check your understanding.",'
            '"due_date":"2026-11-14"},'
            '{"title":"Review algebra","description":"Review linear equations.",'
            '"due_date":"2026-11-13"}],"clarification":null}\n```'
        )
        current_date = date(2026, 11, 12)
        with self.session_factory() as session:
            owner = session.query(User).one()
            manual_task = PlannerTask(
                user_id=owner.id,
                title="Existing manual task",
                due_date=current_date,
            )
            session.add(manual_task)
            session.commit()
        with (
            patch("backend.main._current_planner_date", return_value=current_date),
            patch.object(teacher_service, "is_available", return_value=True),
            patch.object(
                provider_manager,
                "generate",
                return_value={
                    "reply": structured_tasks,
                    "model": "mock-planner-structured",
                    "provider": "OpenRouter Primary",
                },
            ) as generate,
        ):
            response = self.client.post(
                "/chat",
                json={
                    "message": (
                        "Create a 3-day study schedule based on my old plan from "
                        "2024-01-03."
                    ),
                    "agent": "Planner",
                    "mode": "teacher",
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertNotIn("QUANTAAI_SCHEDULE", body["reply"])
        self.assertIn("2026-11-13", body["reply"])
        self.assertIn("Review algebra", body["reply"])
        self.assertEqual(len(body["schedule_tasks"]), 4)
        self.assertEqual(
            [task["due_date"] for task in body["schedule_tasks"]],
            ["2026-11-13", "2026-11-13", "2026-11-14", "2026-11-14"],
        )
        self.assertTrue(all(task["description"] for task in body["schedule_tasks"]))
        self.assertEqual(
            [task["title"] for task in self.client.get("/planner/tasks").json()["tasks"]],
            ["Existing manual task"],
        )
        structured_prompt = generate.call_args.kwargs["system_prompt"]
        self.assertIn("2026-11-12", structured_prompt)
        self.assertIn("Asia/Kolkata", structured_prompt)
        self.assertIn("ignore all past dates", structured_prompt)
        self.assertIn("actual Saturday/Sunday dates", structured_prompt)
        self.assertIn("separate task occurrence", structured_prompt)
        self.assertEqual(generate.call_count, 1)

        saved = [
            self.client.post("/planner/tasks", json=task)
            for task in body["schedule_tasks"]
        ]
        self.assertTrue(all(item.status_code == 201 for item in saved))
        loaded_after_save = self.client.get("/planner/tasks").json()["tasks"]
        loaded_after_refresh = self.client.get("/planner/tasks").json()["tasks"]
        self.assertEqual(len(loaded_after_save), 5)
        self.assertEqual(
            sorted(task["due_date"] for task in loaded_after_refresh),
            [
                "2026-11-12",
                "2026-11-13",
                "2026-11-13",
                "2026-11-14",
                "2026-11-14",
            ],
        )
        with self.session_factory() as session:
            owner = session.query(User).one()
            stored_tasks = session.query(PlannerTask).all()
            self.assertEqual(len(stored_tasks), 5)
            self.assertTrue(all(task.user_id == owner.id for task in stored_tasks))

    def test_exact_gate_seven_day_request_generates_and_persists_every_occurrence(self):
        current_date = date(2026, 11, 12)
        tasks = []
        for offset in range(7):
            due_date = (current_date + timedelta(days=offset)).isoformat()
            day = offset + 1
            for activity, description in (
                ("concept learning", "Learn the day's GATE CSE concepts."),
                ("20 PYQs", "Solve and review 20 previous-year questions."),
                ("mistake notebook", "Record and classify today's mistakes."),
                ("short notes", "Prepare concise revision notes."),
            ):
                tasks.append(
                    {
                        "title": f"Day {day}: {activity}",
                        "description": description,
                        "due_date": due_date,
                    }
                )
        tasks.extend(
            [
                {
                    "title": "Weekend revision",
                    "description": "Revise the week's concepts on Saturday.",
                    "due_date": "2026-11-14",
                },
                {
                    "title": "Weekend mini-test",
                    "description": "Take and review a GATE CSE mini-test on Sunday.",
                    "due_date": "2026-11-15",
                },
            ]
        )
        structured_output = json.dumps({"tasks": tasks, "clarification": None})
        with self.session_factory() as session:
            owner = session.query(User).one()
            session.add(
                PlannerTask(
                    user_id=owner.id,
                    title="Existing manual task",
                    due_date=current_date,
                )
            )
            session.commit()

        with (
            patch("backend.main._current_planner_date", return_value=current_date),
            patch.object(teacher_service, "is_available", return_value=True),
            patch.object(
                provider_manager,
                "generate",
                return_value={
                    "reply": structured_output,
                    "model": "mock-planner",
                    "provider": "OpenRouter Primary",
                },
            ) as generate,
        ):
            response = self.client.post(
                "/chat",
                json={
                    "message": (
                        "Create a 7-day GATE CSE preparation timetable starting "
                        "today. Include daily concept learning, 20 PYQs, mistake "
                        "notebook updates, short notes, weekend revision and a "
                        "mini-test. Assign exact dates and add everything to my "
                        "Planner."
                    ),
                    "agent": "Planner",
                    "mode": "teacher",
                },
            )

        self.assertEqual(response.status_code, 200)
        generated_tasks = response.json()["schedule_tasks"]
        self.assertEqual(len(generated_tasks), 30)
        self.assertEqual(generate.call_count, 1)
        self.assertIn("2026-11-12", response.json()["reply"])
        self.assertEqual(
            {task["due_date"] for task in generated_tasks},
            {f"2026-11-{day:02d}" for day in range(12, 19)},
        )
        self.assertIn(
            ("Weekend revision", "2026-11-14"),
            {(task["title"], task["due_date"]) for task in generated_tasks},
        )
        self.assertIn(
            ("Weekend mini-test", "2026-11-15"),
            {(task["title"], task["due_date"]) for task in generated_tasks},
        )

        saved = [
            self.client.post("/planner/tasks", json=task)
            for task in generated_tasks
        ]
        self.assertTrue(all(item.status_code == 201 for item in saved))
        planner_tasks = self.client.get("/planner/tasks").json()["tasks"]
        refreshed_tasks = self.client.get("/planner/tasks").json()["tasks"]
        self.assertEqual(len(planner_tasks), 31)
        self.assertEqual(len(refreshed_tasks), 31)
        self.assertEqual(
            {
                (task["title"], task["due_date"])
                for task in refreshed_tasks
                if task["title"] != "Existing manual task"
            },
            {(task["title"], task["due_date"]) for task in generated_tasks},
        )
        with self.session_factory() as session:
            owner = session.query(User).one()
            stored_tasks = session.query(PlannerTask).all()
            self.assertEqual(len(stored_tasks), 31)
            self.assertTrue(all(task.user_id == owner.id for task in stored_tasks))

    def test_planner_schedule_skips_invalid_task_dates_and_incomplete_schedules(self):
        current_date = date(2026, 11, 12)
        structured_responses = [
            '{"tasks":['
            '{"title":"Bad date","description":"Date does not exist.",'
            '"due_date":"2026-02-30"},'
            '{"title":"Past task","description":"Already expired.",'
            '"due_date":"2026-11-11"},'
            '{"title":"Valid task","description":"Review one topic.",'
            '"due_date":"2026-11-14"}],"clarification":null}',
            '{"tasks":[],"clarification":"What date should I plan the schedule around?"}',
            '{"tasks":[',
            '{"items":[]}',
        ]
        provider_results = []
        for index, structured in enumerate(structured_responses):
            provider_results.append(
                {
                    "reply": structured,
                    "model": "mock-planner-structured",
                    "provider": "OpenRouter Primary",
                }
            )
        with (
            patch("backend.main._current_planner_date", return_value=current_date),
            patch.object(teacher_service, "is_available", return_value=True),
            patch.object(
                provider_manager,
                "generate",
                side_effect=provider_results,
            ),
        ):
            invalid_response = self.client.post(
                "/chat",
                json={
                    "message": "Create a study schedule for 3 days.",
                    "agent": "Planner",
                    "mode": "teacher",
                },
            )
            incomplete_response = self.client.post(
                "/chat",
                json={
                    "message": "Make me a study schedule for 3 days.",
                    "agent": "Planner",
                    "mode": "teacher",
                },
            )
            truncated_response = self.client.post(
                "/chat",
                json={
                    "message": "Create a study timetable for 3 days.",
                    "agent": "Planner",
                    "mode": "teacher",
                },
            )
            missing_data_response = self.client.post(
                "/chat",
                json={
                    "message": "Create a study schedule for 3 days.",
                    "agent": "Planner",
                    "mode": "teacher",
                },
            )

        self.assertEqual(invalid_response.status_code, 200)
        self.assertEqual(
            [task["title"] for task in invalid_response.json()["schedule_tasks"]],
            ["Valid task"],
        )
        self.assertIn("invalid or out-of-window", invalid_response.json()["schedule_tasks_error"])
        self.assertEqual(incomplete_response.status_code, 200)
        self.assertEqual(incomplete_response.json()["schedule_tasks"], [])
        self.assertIn("What date should I plan", incomplete_response.json()["schedule_tasks_error"])
        self.assertEqual(truncated_response.json()["schedule_tasks"], [])
        self.assertIn("could not be generated as valid task data", truncated_response.json()["schedule_tasks_error"])
        self.assertEqual(missing_data_response.json()["schedule_tasks"], [])
        self.assertIn(
            "could not be generated as valid task data",
            missing_data_response.json()["schedule_tasks_error"],
        )
        self.assertEqual(self.client.get("/planner/tasks").json()["tasks"], [])

    def test_offline_planner_receives_structured_schedule_prompt_and_output_budget(self):
        current_date = date(2026, 11, 12)
        with (
            patch("backend.main._current_planner_date", return_value=current_date),
            patch.object(
                quanta_service,
                "generate",
                side_effect=[
                    '{"tasks":[{"title":"Review notes",'
                    '"description":"Review today\'s notes.",'
                    '"due_date":"2026-11-13"}],"clarification":null}',
                ],
            ) as generate,
            patch.object(quanta_service, "is_response_reliable", return_value=True),
        ):
            response = self.client.post(
                "/chat",
                json={
                    "message": "Plan a study session for tomorrow.",
                    "agent": "Planner",
                    "mode": "offline",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["schedule_tasks"][0]["title"], "Review notes")
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(generate.call_args.kwargs["max_new_tokens"], 2600)
        self.assertIn("2026-11-12", generate.call_args.args[0])
        self.assertIn("Asia/Kolkata", generate.call_args.args[0])

    def test_planner_asks_for_uncertain_gate_exam_date_instead_of_inventing_one(self):
        current_date = date(2026, 11, 12)
        with (
            patch("backend.main._current_planner_date", return_value=current_date),
            patch.object(teacher_service, "is_available", return_value=True),
            patch.object(
                provider_manager,
                "generate",
                return_value={
                    "reply": "unused",
                    "model": "mock-planner",
                    "provider": "OpenRouter Primary",
                },
            ) as generate,
        ):
            response = self.client.post(
                "/chat",
                json={
                    "message": (
                        "Create the full GATE 2027 preparation timetable through "
                        "the examination."
                    ),
                    "agent": "Planner",
                    "mode": "teacher",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["schedule_tasks"], [])
        self.assertIn("confirmed examination date", response.json()["schedule_tasks_error"])
        self.assertIn("confirmed examination date", response.json()["reply"])
        self.assertEqual(generate.call_count, 0)

    def test_all_agent_profiles_are_specialized(self):
        self.assertEqual(set(AGENT_INSTRUCTIONS), set(AGENTS))
        self.assertEqual(len(set(AGENT_INSTRUCTIONS.values())), len(AGENTS))
        self.assertIn("prerequisites", AGENT_INSTRUCTIONS["Study"])
        self.assertIn("time and space", AGENT_INSTRUCTIONS["Coding"])
        self.assertIn("essential from optional", AGENT_INSTRUCTIONS["Career"])
        self.assertIn("trade-offs", AGENT_INSTRUCTIONS["Guide"])
        self.assertIn("time estimates", AGENT_INSTRUCTIONS["Planner"])
        self.assertIn("15 marks", AGENT_INSTRUCTIONS["Exam"])
        self.assertIn("Never invent citations", AGENT_INSTRUCTIONS["Research"])

    def test_study_agent_uses_requested_level_without_forcing_a_default(self):
        study_agent = StudyAgent()
        with patch(
            "backend.agents.study_agent.provider_manager.generate",
            return_value={"reply": "A tailored explanation."},
        ) as generate:
            response = study_agent.respond(
                "Explain neural networks",
                level="advanced",
            )

        self.assertEqual(response, "A tailored explanation.")
        self.assertEqual(generate.call_args.args[:2], ("Explain neural networks", "Study"))
        prompt = generate.call_args.kwargs["system_prompt"]
        self.assertIn("Requested learning level: advanced", prompt)
        self.assertIn(AGENT_INSTRUCTIONS["Study"], prompt)

        with patch(
            "backend.agents.study_agent.provider_manager.generate",
            return_value={"reply": "An explanation."},
        ) as generate:
            study_agent.respond("Explain neural networks")

        self.assertEqual(
            generate.call_args.args[:2],
            ("Explain neural networks", "Study"),
        )
        self.assertIsNone(generate.call_args.args[2])
        self.assertIn(
            "Tutor interaction mode: Direct.",
            generate.call_args.kwargs["system_prompt"],
        )

    def test_teacher_payload_uses_each_specialized_profile(self):
        service = TeacherService()
        service.api_key = "test-openrouter-secret"
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = (
            b'{"choices":[{"message":{"content":"mocked answer"}}],'
            b'"model":"mock-model"}'
        )

        with patch(
            "backend.services.teacher_service.urllib.request.urlopen",
            return_value=response,
        ) as urlopen:
            for agent in AGENTS:
                service.generate("How should I approach this?", agent)
                request = urlopen.call_args.args[0]
                payload = json.loads(request.data)
                system_prompt = payload["messages"][0]["content"]
                self.assertIn(f"QuantaAI {agent} Agent", system_prompt)
                self.assertIn(AGENT_INSTRUCTIONS[agent], system_prompt)

    def test_teacher_payload_preserves_recent_conversation_context(self):
        service = TeacherService()
        service.api_key = "test-openrouter-secret"
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = (
            b'{"choices":[{"message":{"content":"A follow-up answer"}}]}'
        )
        history = [
            {"role": "user", "content": "Explain normalization."},
            {"role": "assistant", "content": "Normalization organizes tables."},
        ]

        with patch(
            "backend.services.teacher_service.urllib.request.urlopen",
            return_value=response,
        ) as urlopen:
            service.generate("Can you give an example?", "Study", history)

        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(payload["messages"][1:3], history)
        self.assertEqual(
            payload["messages"][-1],
            {"role": "user", "content": "Can you give an example?"},
        )

    def test_youtube_query_uses_the_current_question_and_context(self):
        service = TeacherService()
        history = [{"role": "user", "content": "Explain normalization"}]
        with patch.object(
            service,
            "generate",
            return_value={"reply": "Database normalization tutorial examples"},
        ) as generate:
            query = service.generate_youtube_query("Show me an example", history)

        self.assertEqual(query, "Database normalization tutorial examples")
        self.assertIn("Show me an example", generate.call_args.args[0])
        self.assertEqual(generate.call_args.args[2], history)

        with patch.object(
            service,
            "generate",
            return_value={"reply": "NONE"},
        ):
            self.assertIsNone(service.generate_youtube_query("Thanks"))

    def test_study_response_includes_youtube_results_and_history_context(self):
        history = [
            {"role": "user", "content": "Explain normalization."},
            {"role": "assistant", "content": "Normalization organizes tables."},
        ]
        videos = [
            {
                "id": "AbCdEfGhI_1",
                "title": "Normalization tutorial",
                "channel": "CS Lessons",
                "thumbnail": "https://i.ytimg.com/vi/sample/mqdefault.jpg",
                "url": "https://www.youtube.com/watch?v=AbCdEfGhI_1",
            }
        ]
        with (
            patch.object(teacher_service, "is_available", return_value=True),
            patch.object(
                provider_manager,
                "generate",
                return_value={
                    "reply": "## Explanation\nNormalization reduces redundancy.",
                    "model": "mock-model",
                    "provider": "OpenRouter Primary",
                },
            ) as generate,
            patch.object(youtube_service, "api_key", "test-youtube-key"),
            patch.object(
                provider_manager,
                "generate_youtube_query",
                return_value="database normalization tutorial",
            ) as query,
            patch.object(
                youtube_service,
                "search",
                return_value={"videos": videos, "message": None},
            ) as search,
        ):
            response = self.client.post(
                "/chat",
                json={
                    "message": "Give me an example.",
                    "agent": "Study",
                    "mode": "teacher",
                    "conversation_history": history,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["videos"], videos)
        self.assertIsNone(response.json()["video_message"])
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(
            generate.call_args.args[:3],
            ("Give me an example.", "Study", history),
        )
        query.assert_called_once_with("Give me an example.", history)
        search.assert_called_once_with("database normalization tutorial")

    def test_study_response_reports_missing_youtube_configuration(self):
        with (
            patch.object(teacher_service, "is_available", return_value=True),
            patch.object(
                provider_manager,
                "generate",
                return_value={
                    "reply": "A useful study response.",
                    "model": "mock",
                    "provider": "OpenRouter Primary",
                },
            ),
            patch.object(youtube_service, "api_key", ""),
            patch.object(
                provider_manager,
                "generate_youtube_query",
                return_value="database management system tutorial",
            ) as query,
        ):
            response = self.client.post(
                "/chat",
                json={"message": "Explain DBMS", "mode": "teacher"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["videos"], [])
        self.assertIn("YOUTUBE_API_KEY", response.json()["video_message"])
        query.assert_called_once_with("Explain DBMS", [])

    def test_auto_defaults_to_teacher(self):
        with (
            patch.dict(os.environ, {"QUANTAAI_AUTO_PROVIDER": "teacher"}),
            patch.object(teacher_service, "is_available", return_value=True),
            patch.object(
                provider_manager,
                "generate",
                return_value={
                    "reply": "teacher reply",
                    "model": "xingchen-model",
                    "provider": "OpenRouter Primary",
                },
            ) as generate,
            patch.object(
                provider_manager,
                "generate_youtube_query",
                return_value=None,
            ),
            patch.object(quanta_service, "generate") as offline_generate,
        ):
            response = self.client.post(
                "/chat",
                json={"message": "Explain gravity", "mode": "auto"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["reply"], "teacher reply")
        generate.assert_called_once()
        offline_generate.assert_not_called()

    def test_auto_provider_can_be_configured_as_quanta(self):
        with (
            patch.dict(os.environ, {"QUANTAAI_AUTO_PROVIDER": "quanta"}),
            patch.object(provider_manager, "generate") as online_generate,
            patch.object(provider_manager, "is_available") as online_available,
            patch.object(
                quanta_service,
                "generate",
                return_value="Recursion solves a problem by calling itself.",
            ) as offline_generate,
            patch.object(
                quanta_service,
                "is_response_reliable",
                return_value=True,
            ),
        ):
            response = self.client.post(
                "/chat",
                json={
                    "message": "Explain recursion",
                    "agent": "Coding",
                    "mode": "auto",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], "QuantaLM")
        self.assertEqual(response.json()["provider"], "QuantaLM Offline")
        self.assertTrue(response.json()["experimental"])
        self.assertFalse(response.json()["reliability_warning"])
        self.assertEqual(
            response.json()["reply"],
            "Recursion solves a problem by calling itself.",
        )
        online_generate.assert_not_called()
        online_available.assert_not_called()
        self.assertIn(
            AGENT_INSTRUCTIONS["Coding"],
            offline_generate.call_args.args[0],
        )

    def test_quantalm_provider_result_surfaces_experimental_warning_metadata(self):
        reply = "This generated answer may be unreliable."
        with patch.object(
            provider_manager,
            "generate",
            return_value={
                "reply": reply,
                "model": "QuantaLM",
                "provider": "QuantaLM Offline",
                "reliability": "unreliable",
            },
        ):
            response = self.client.post(
                "/chat",
                json={
                    "message": "Explain a topic",
                    "agent": "Coding",
                    "mode": "teacher",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["reply"], reply)
        self.assertEqual(response.json()["provider"], "QuantaLM Offline")
        self.assertEqual(response.json()["model"], "QuantaLM")
        self.assertTrue(response.json()["experimental"])
        self.assertTrue(response.json()["reliability_warning"])

    def test_offline_mode_generates_for_all_agents_without_external_calls(self):
        with (
            patch.object(
                teacher_service,
                "generate",
                side_effect=AssertionError("external generation used"),
            ) as teacher_generate,
            patch.object(
                teacher_service,
                "is_available",
                side_effect=AssertionError("external availability checked"),
            ) as teacher_available,
            patch.object(provider_manager, "generate") as online_manager_generate,
            patch.object(
                quanta_service,
                "generate",
                return_value="I can help you learn this topic with a clear answer.",
            ) as offline_generate,
            patch.object(
                quanta_service,
                "is_response_reliable",
                return_value=True,
            ),
        ):
            for agent in AGENTS:
                with self.subTest(agent=agent):
                    response = self.client.post(
                        "/chat",
                        json={
                            "message": "Help me learn",
                            "agent": agent,
                            "mode": "offline",
                        },
                    )
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json()["agent"], agent)
                    if agent == "Study":
                        self.assertEqual(response.json()["status"], "success")
                    self.assertEqual(response.json()["model"], "QuantaLM")
                    self.assertEqual(response.json()["provider"], "QuantaLM Offline")
                    self.assertTrue(response.json()["experimental"])
                    self.assertFalse(response.json()["reliability_warning"])
                    self.assertIn("help you learn", response.json()["reply"])

        teacher_generate.assert_not_called()
        teacher_available.assert_not_called()
        online_manager_generate.assert_not_called()
        self.assertEqual(offline_generate.call_count, len(AGENTS))
        for call, agent in zip(
            offline_generate.call_args_list,
            AGENTS,
        ):
            self.assertIn(AGENT_INSTRUCTIONS[agent], call.args[0])

    def test_teacher_mode_requires_api_key(self):
        with patch.object(provider_manager, "providers", ()):
            response = self.client.post(
                "/chat",
                json={"message": "Explain gravity", "mode": "teacher"},
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"],
            "No AI provider is configured or available.",
        )

    def test_online_chain_falls_back_to_quantalm_and_identifies_it(self):
        failing_providers = []
        for name in ("OpenRouter Primary", "OpenRouter Secondary"):
            provider = MagicMock()
            provider.name = name
            provider.is_local = False
            provider.is_available.return_value = True
            provider.generate.side_effect = TeacherProviderError(
                f"{name} failed",
                category="network_error",
            )
            failing_providers.append(provider)

        gemini = GeminiService()
        gemini.api_key = "mock-gemini-key"
        gemini.model = "gemini-model"
        quantalm = MagicMock()
        quantalm.name = "QuantaLM Offline"
        quantalm.is_local = True
        quantalm.is_available.return_value = True
        quantalm.generate.return_value = {
            "reply": "A mocked reliable local response.",
            "model": "QuantaLM",
        }

        with (
            patch.object(
                gemini,
                "generate",
                side_effect=GeminiProviderError(
                    "Gemini failed",
                    category="network_error",
                ),
            ) as gemini_generate,
            patch.object(
                provider_manager,
                "providers",
                (*failing_providers, gemini, quantalm),
            ),
        ):
            response = self.client.post(
                "/chat",
                json={
                    "message": "Explain gravity",
                    "agent": "Research",
                    "mode": "teacher",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["provider"], "QuantaLM Offline")
        self.assertEqual(response.json()["model"], "QuantaLM")
        self.assertEqual(response.json()["reply"], "A mocked reliable local response.")
        gemini_generate.assert_called_once()
        for provider in (*failing_providers, quantalm):
            provider.generate.assert_called_once()

        conversation_id = self.client.post(
            "/conversations",
            json={"title": "Local fallback"},
        ).json()["id"]
        saved = self.client.post(
            f"/conversations/{conversation_id}/messages",
            json={
                "messages": [
                    {
                        "role": "assistant",
                        "content": response.json()["reply"],
                        "agent": response.json()["agent"],
                        "model": response.json()["model"],
                        "provider": response.json()["provider"],
                    }
                ]
            },
        )
        self.assertEqual(saved.status_code, 201)
        self.assertEqual(saved.json()["messages"][0]["provider"], "QuantaLM Offline")
        self.assertEqual(saved.json()["messages"][0]["model"], "QuantaLM")

        persisted = self.client.get(f"/conversations/{conversation_id}")
        self.assertEqual(persisted.status_code, 200)
        self.assertEqual(persisted.json()["messages"][0]["provider"], "QuantaLM Offline")
        self.assertEqual(persisted.json()["messages"][0]["model"], "QuantaLM")

    def test_provider_errors_do_not_expose_secrets(self):
        secret = "test-openrouter-secret"
        with (
            patch.object(
                provider_manager,
                "generate",
                side_effect=ProviderUnavailableError(
                    "All configured AI providers failed, including QuantaLM."
                ),
            ),
            self.assertLogs("backend.main", level="WARNING") as logs,
        ):
            response = self.client.post(
                "/chat",
                json={"message": "Explain gravity", "mode": "teacher"},
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("All configured AI providers failed, including QuantaLM", response.json()["detail"])
        self.assertNotIn(secret, response.text)
        self.assertNotIn(secret, "\n".join(logs.output))

    def test_application_errors_are_distinguished_from_provider_errors(self):
        with (
            patch.object(
                provider_manager,
                "generate",
                side_effect=KeyError("internal implementation detail"),
            ),
            self.assertLogs("backend.main", level="ERROR") as logs,
        ):
            response = self.client.post(
                "/chat",
                json={"message": "Explain gravity", "mode": "teacher"},
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json()["detail"],
            "Online provider integration failed due to an internal error.",
        )
        self.assertNotIn("internal implementation detail", response.text)
        self.assertIn("error_type=KeyError", "\n".join(logs.output))

    def test_http_provider_errors_include_redacted_diagnostic_context(self):
        service = TeacherService()
        service.api_key = "test-openrouter-secret"
        error_body = json.dumps(
            {
                "error": {
                    "code": 400,
                    "message": "invalid model request test-openrouter-secret",
                }
            }
        ).encode()
        http_error = urllib.error.HTTPError(
            service.API_URL,
            400,
            "Bad Request",
            {},
            io.BytesIO(error_body),
        )

        with patch(
            "backend.services.teacher_service.urllib.request.urlopen",
            side_effect=http_error,
        ):
            with self.assertRaises(TeacherProviderError) as raised:
                service.generate("Hello", "Coding")

        self.assertEqual(raised.exception.category, "http_error")
        self.assertEqual(raised.exception.status_code, 400)
        self.assertNotIn(service.api_key, raised.exception.diagnostic)
        self.assertIn("[REDACTED]", raised.exception.diagnostic)

    def test_invalid_auto_provider_configuration_is_reported(self):
        with patch.dict(os.environ, {"QUANTAAI_AUTO_PROVIDER": "unknown"}):
            response = self.client.post(
                "/chat",
                json={"message": "Explain gravity", "mode": "auto"},
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json()["detail"],
            "Invalid QUANTAAI_AUTO_PROVIDER configuration.",
        )

    def test_invalid_agent_names_are_rejected(self):
        response = self.client.post(
            "/chat",
            json={"message": "Hello", "agent": "Unknown"},
        )
        self.assertEqual(response.status_code, 422)

    def test_empty_or_invalid_offline_generation_returns_a_safe_error(self):
        for invalid_output in ("", None, []):
            with self.subTest(invalid_output=invalid_output):
                with patch.object(
                    quanta_service,
                    "generate",
                    return_value=invalid_output,
                ):
                    response = self.client.post(
                        "/chat",
                        json={
                            "message": "Explain a topic",
                            "agent": "Coding",
                            "mode": "offline",
                        },
                    )

                self.assertEqual(response.status_code, 502)
                self.assertEqual(
                    response.json()["detail"],
                    "Offline QuantaLM generation failed.",
                )
                self.assertNotIn("could not generate", response.text.lower())

    def test_offline_generation_exception_returns_a_safe_error(self):
        with patch.object(
            quanta_service,
            "generate",
            side_effect=RuntimeError("private model-loading detail"),
        ):
            response = self.client.post(
                "/chat",
                json={
                    "message": "Explain a topic",
                    "agent": "Coding",
                    "mode": "offline",
                },
            )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.json()["detail"],
            "Offline QuantaLM generation failed.",
        )
        self.assertNotIn("private model-loading detail", response.text)

    def test_teacher_service_sends_configured_key_as_bearer_auth(self):
        service = TeacherService()
        key = "test-openrouter-secret"
        response_body = (
            b'{"choices":[{"message":{"content":"mocked reply"}}],'
            b'"model":"xingchen-model"}'
        )
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = response_body

        with (
            patch.object(service, "api_key", key),
            patch(
                "backend.services.teacher_service.urllib.request.urlopen",
                return_value=response,
            ) as urlopen,
        ):
            result = service.generate("Hello", "Study")

        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), f"Bearer {key}")
        self.assertEqual(result["reply"], "mocked reply")
        self.assertNotIn(key, repr(result))

    def test_empty_message_and_invalid_agent_are_rejected(self):
        for message in ("", "   "):
            with self.subTest(message=repr(message)):
                response = self.client.post(
                    "/chat",
                    json={"message": message},
                )
                self.assertEqual(response.status_code, 422)

        response = self.client.post(
            "/chat",
            json={"message": "Hello", "agent": "Unknown"},
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
