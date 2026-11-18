import os
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from backend.auth import (
    ACCESS_TOKEN_TTL,
    AuthSession,
    User,
    _database_url,
    get_db,
    password_hasher,
)
from backend.main import app
from backend.tests.auth_support import AuthTestCase
from backend.services.quanta_service import quanta_service


class AuthenticationApiTests(AuthTestCase):
    def test_registration_creates_persistent_hashed_account_and_cookie_session(self):
        response = self.register_account()

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["user"]["email"], "learner@example.com")
        self.assertNotIn("password_hash", response.json()["user"])
        self.assertNotIn("access_token", response.json())
        self.assertTrue(response.cookies.get("quantaai_session"))

        with self.session_factory() as session:
            user = session.query(User).filter_by(email="learner@example.com").one()
            self.assertNotEqual(user.password_hash, "StrongPassword123!")
            self.assertTrue(
                password_hasher.verify(user.password_hash, "StrongPassword123!")
            )
            self.assertEqual(user.profile["careerGoal"], "AI Engineer")
            self.assertIsNotNone(user.created_at)
            self.assertIsNotNone(user.updated_at)
            auth_session = session.query(AuthSession).one()
            self.assertNotEqual(
                auth_session.token_hash,
                response.cookies.get("quantaai_session"),
            )
            expires_at = auth_session.expires_at
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            self.assertGreater(expires_at, now)
            self.assertLessEqual(expires_at, now + ACCESS_TOKEN_TTL)

    def test_duplicate_registration_is_rejected(self):
        self.assertEqual(self.register_account().status_code, 201)
        response = self.register_account()
        self.assertEqual(response.status_code, 409)
        self.assertIn("already exists", response.json()["detail"])

    def test_registration_rejects_case_insensitive_duplicate_email(self):
        self.assertEqual(self.register_account().status_code, 201)
        response = self.register_account(email="LEARNER@EXAMPLE.COM")
        self.assertEqual(response.status_code, 409)

    def test_state_changing_requests_require_csrf_token(self):
        csrf_header = self.client.headers.pop("X-CSRF-Token")
        response = self.register_account()
        self.client.headers["X-CSRF-Token"] = csrf_header
        self.assertEqual(response.status_code, 403)

    def test_email_validation_is_enforced(self):
        response = self.register_account(email="not-an-email")
        self.assertEqual(response.status_code, 422)

    def test_weak_password_is_rejected(self):
        response = self.register_account(password="weak")
        self.assertEqual(response.status_code, 422)

    def test_successful_login_sets_authenticated_cookie(self):
        self.register_account()
        self.client.cookies.delete("quantaai_session")

        with patch.dict(os.environ, {"AUTH_COOKIE_SECURE": "true"}):
            response = self.client.post(
                "/auth/login",
                json={"email": "LEARNER@example.com", "password": "StrongPassword123!"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["user"]["name"], "Test Learner")
        self.assertTrue(self.client.cookies.get("quantaai_session"))
        set_cookie = response.headers["set-cookie"].lower()
        self.assertIn("httponly", set_cookie)
        self.assertIn("samesite=lax", set_cookie)
        self.assertIn("secure", set_cookie)

    def test_postgres_configuration_matches_java_service_environment(self):
        settings = {
            "DATABASE_URL": "",
            "NEON_DB_HOST": "database.example",
            "NEON_DB_NAME": "quanta",
            "NEON_DB_USER": "quanta-user",
            "NEON_DB_PASSWORD": "not-logged",
        }
        with patch.dict(os.environ, settings, clear=True):
            url = _database_url()

        self.assertEqual(url.drivername, "postgresql+psycopg")
        self.assertEqual(url.query["sslmode"], "require")
        self.assertEqual(url.host, "database.example")
        self.assertEqual(url.username, "quanta-user")

    def test_postgres_configuration_retains_username_variable_compatibility(self):
        settings = {
            "DATABASE_URL": "",
            "NEON_DB_HOST": "database.example",
            "NEON_DB_NAME": "quanta",
            "NEON_DB_USERNAME": "legacy-user",
            "NEON_DB_PASSWORD": "not-logged",
        }
        with patch.dict(os.environ, settings, clear=True):
            url = _database_url()

        self.assertEqual(url.username, "legacy-user")
        self.assertEqual(url.drivername, "postgresql+psycopg")
        self.assertEqual(url.query["sslmode"], "require")

    def test_database_url_parses_postgresql_credentials_and_enforces_neon_ssl(self):
        with patch.dict(
            os.environ,
            {
                "DATABASE_URL": (
                    "postgresql://quanta-user:p%40ss%2Fword@ep-example.neon.tech/"
                    "quanta?application_name=tests"
                )
            },
        ):
            url = _database_url()

        self.assertEqual(url.drivername, "postgresql+psycopg")
        self.assertEqual(url.password, "p@ss/word")
        self.assertEqual(url.query["application_name"], "tests")
        self.assertEqual(url.query["sslmode"], "require")

    def test_database_url_supports_legacy_postgres_scheme_and_sqlite(self):
        with patch.dict(os.environ, {"DATABASE_URL": "postgres://user:pass@db.local/quanta"}):
            postgres_url = _database_url()
        with patch.dict(os.environ, {"DATABASE_URL": "sqlite:///./test.db"}):
            sqlite_url = _database_url()

        self.assertEqual(postgres_url.drivername, "postgresql+psycopg")
        self.assertEqual(sqlite_url.drivername, "sqlite")

    def test_database_engine_uses_bounded_postgres_connection_timeouts(self):
        from backend.auth import _engine

        _engine.cache_clear()
        try:
            with patch.dict(
                os.environ,
                {"DATABASE_URL": "postgresql://user:pass@db.local/quanta"},
            ), patch("backend.auth.create_engine") as create_engine_mock:
                _engine()

            options = create_engine_mock.call_args.kwargs
            self.assertTrue(options["pool_pre_ping"])
            self.assertEqual(options["connect_args"], {"connect_timeout": 5})
            self.assertEqual(options["pool_timeout"], 5)
        finally:
            _engine.cache_clear()

    def test_database_connection_failure_returns_sanitized_service_unavailable(self):
        def fail_connection():
            raise OperationalError(
                "connect", {}, RuntimeError("do-not-disclose-this-detail")
            )

        engine = create_engine("sqlite://", creator=fail_connection)
        failing_factory = sessionmaker(bind=engine, expire_on_commit=False)

        def failing_database():
            with failing_factory() as session:
                yield session

        app.dependency_overrides[get_db] = failing_database
        try:
            response = self.register_account()
        finally:
            app.dependency_overrides[get_db] = self.override_get_db
            engine.dispose()

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"],
            "Account registration is unavailable.",
        )
        self.assertNotIn("do-not-disclose-this-detail", response.text)

    def test_incorrect_password_is_rejected_without_disclosing_lookup_result(self):
        self.register_account()
        self.client.cookies.delete("quantaai_session")

        response = self.client.post(
            "/auth/login",
            json={"email": "learner@example.com", "password": "WrongPassword123!"},
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Invalid email or password.")

    def test_nonexistent_account_uses_same_login_error(self):
        response = self.client.post(
            "/auth/login",
            json={"email": "missing@example.com", "password": "WrongPassword123!"},
        )
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Invalid email or password.")

    def test_protected_routes_reject_missing_authentication(self):
        self.assertEqual(self.client.get("/auth/me").status_code, 401)
        response = self.client.post(
            "/chat",
            json={"message": "Hello", "agent": "Study", "mode": "offline"},
        )
        self.assertEqual(response.status_code, 401)

    def test_missing_authentication_is_rejected_before_database_access(self):
        def unexpected_database_access():
            raise AssertionError("unauthenticated requests must not access the database")

        app.dependency_overrides[get_db] = unexpected_database_access
        try:
            response = self.client.get("/auth/me")
        finally:
            app.dependency_overrides[get_db] = self.override_get_db

        self.assertEqual(response.status_code, 401)

    def test_valid_session_accesses_current_user_and_existing_agent_chat(self):
        self.register_account()
        self.assertEqual(
            self.client.get("/auth/me").json()["email"],
            "learner@example.com",
        )
        with patch(
            "backend.services.quanta_service.quanta_service.generate",
            return_value="Recursion is when a function calls itself.",
        ):
            response = self.client.post(
                "/chat",
                    json={"message": "Explain recursion", "agent": "Coding", "mode": "offline"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["agent"], "Coding")
        self.assertEqual(
            response.json()["reply"],
            "Recursion is when a function calls itself.",
        )

    def test_profile_update_persists_name_and_preferences(self):
        self.register_account()

        response = self.client.patch(
            "/auth/me",
            json={
                "name": "  Ada Lovelace  ",
                "profile": {
                    "careerGoal": "Research scientist",
                    "learningStyle": "examples",
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["name"], "Ada Lovelace")
        self.assertEqual(
            response.json()["profile"],
            {
                "careerGoal": "Research scientist",
                "learningStyle": "examples",
            },
        )
        self.assertEqual(
            self.client.get("/auth/me").json(),
            response.json(),
        )
        with self.session_factory() as session:
            user = session.query(User).filter_by(email="learner@example.com").one()
            self.assertEqual(user.name, "Ada Lovelace")
            self.assertEqual(user.profile["careerGoal"], "Research scientist")

    def test_profile_update_requires_valid_fields_and_csrf(self):
        self.register_account()

        csrf_header = self.client.headers.pop("X-CSRF-Token")
        missing_csrf = self.client.patch("/auth/me", json={"name": "Ada"})
        self.client.headers["X-CSRF-Token"] = csrf_header
        empty_update = self.client.patch("/auth/me", json={})
        empty_name = self.client.patch("/auth/me", json={"name": "  "})

        self.assertEqual(missing_csrf.status_code, 403)
        self.assertEqual(empty_update.status_code, 422)
        self.assertEqual(empty_name.status_code, 422)

    def test_expired_session_is_rejected(self):
        self.register_account()
        with self.session_factory() as session:
            auth_session = session.query(AuthSession).one()
            auth_session.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
            session.commit()

        response = self.client.get("/auth/me")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Invalid or expired session.")

    def test_logout_revokes_session(self):
        self.register_account()
        response = self.client.post("/auth/logout")
        self.assertEqual(response.status_code, 204)
        self.assertEqual(self.client.get("/auth/me").status_code, 401)
        with self.session_factory() as session:
            self.assertEqual(session.query(AuthSession).count(), 0)

    def test_database_persists_user_across_sessions(self):
        self.register_account()
        with self.session_factory() as session:
            first_user_id = session.query(User.id).one()[0]
        with self.session_factory() as new_session:
            user = new_session.get(User, first_user_id)
            self.assertIsNotNone(user)
            self.assertEqual(user.email, "learner@example.com")
