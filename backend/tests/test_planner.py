from backend.auth import User
from backend.planner import PlannerTask
from backend.tests.auth_support import AuthTestCase


class PlannerApiTests(AuthTestCase):
    def test_task_crud_persists_updates_and_completion(self):
        self.register_account()
        created = self.client.post(
            "/planner/tasks",
            json={
                "title": "  Review recursion  ",
                "description": "  Revisit the call stack and base case.  ",
                "due_date": "2026-03-10",
            },
        )

        self.assertEqual(created.status_code, 201)
        task = created.json()
        self.assertEqual(task["title"], "Review recursion")
        self.assertEqual(task["description"], "Revisit the call stack and base case.")
        self.assertEqual(task["due_date"], "2026-03-10")
        self.assertFalse(task["completed"])

        listed = self.client.get("/planner/tasks")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual([item["id"] for item in listed.json()["tasks"]], [task["id"]])

        updated = self.client.patch(
            f"/planner/tasks/{task['id']}",
            json={
                "title": "Practice recursion",
                "description": "Solve two recursive problems.",
                "completed": True,
            },
        )
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.json()["title"], "Practice recursion")
        self.assertEqual(
            updated.json()["description"],
            "Solve two recursive problems.",
        )
        self.assertTrue(updated.json()["completed"])
        self.assertEqual(
            self.client.get("/planner/tasks").json()["tasks"][0]["title"],
            "Practice recursion",
        )
        self.assertEqual(
            self.client.get("/planner/tasks").json()["tasks"][0]["description"],
            "Solve two recursive problems.",
        )

        reopened = self.client.patch(
            f"/planner/tasks/{task['id']}",
            json={"completed": False},
        )
        self.assertEqual(reopened.status_code, 200)
        self.assertFalse(reopened.json()["completed"])

        deleted = self.client.delete(f"/planner/tasks/{task['id']}")
        self.assertEqual(deleted.status_code, 204)
        self.assertEqual(self.client.get("/planner/tasks").json()["tasks"], [])
        with self.session_factory() as session:
            self.assertEqual(session.query(PlannerTask).count(), 0)

    def test_task_crud_validates_inputs_and_enforces_user_ownership(self):
        self.register_account()
        empty_title = self.client.post(
            "/planner/tasks",
            json={"title": "  ", "due_date": "2026-03-10"},
        )
        missing_date = self.client.post(
            "/planner/tasks",
            json={"title": "Review recursion"},
        )
        invalid_date = self.client.post(
            "/planner/tasks",
            json={"title": "Review recursion", "due_date": "2026-02-30"},
        )
        self.assertEqual(empty_title.status_code, 422)
        self.assertEqual(missing_date.status_code, 422)
        self.assertEqual(invalid_date.status_code, 422)

        created = self.client.post(
            "/planner/tasks",
            json={"title": "Private task", "due_date": "2026-03-10"},
        )
        self.assertEqual(created.status_code, 201)
        task_id = created.json()["id"]

        self.register_account(
            name="Another Learner",
            email="another@example.com",
        )
        self.assertEqual(
            self.client.patch(
                f"/planner/tasks/{task_id}",
                json={"completed": True},
            ).status_code,
            404,
        )
        self.assertEqual(self.client.delete(f"/planner/tasks/{task_id}").status_code, 404)

        with self.session_factory() as session:
            self.assertEqual(session.query(User).count(), 2)
            self.assertEqual(session.query(PlannerTask).count(), 1)

    def test_task_writes_require_csrf(self):
        self.register_account()
        csrf_header = self.client.headers.pop("X-CSRF-Token")
        response = self.client.post(
            "/planner/tasks",
            json={"title": "Review recursion", "due_date": "2026-03-10"},
        )
        self.client.headers["X-CSRF-Token"] = csrf_header

        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.client.get("/planner/tasks").json()["tasks"], [])
