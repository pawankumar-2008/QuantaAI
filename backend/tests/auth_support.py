import unittest

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.auth import Base, get_db
from backend.main import app


class AuthTestCase(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        event.listen(
            self.engine,
            "connect",
            lambda connection, _: connection.execute("PRAGMA foreign_keys=ON"),
        )
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
        )
        app.dependency_overrides[get_db] = self.override_get_db
        self.client = TestClient(app)
        csrf = self.client.get("/auth/csrf")
        self.assertEqual(csrf.status_code, 200)
        self.client.headers.update(
            {"X-CSRF-Token": csrf.json()["csrf_token"]}
        )

    def tearDown(self):
        self.client.close()
        app.dependency_overrides.clear()
        self.engine.dispose()

    def override_get_db(self):
        with self.session_factory() as session:
            yield session

    def register_account(self, **overrides):
        account = {
            "name": "Test Learner",
            "email": "learner@example.com",
            "password": "StrongPassword123!",
            "profile": {"careerGoal": "AI Engineer"},
        }
        account.update(overrides)
        return self.client.post("/auth/register", json=account)
