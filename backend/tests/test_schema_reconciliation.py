import json
import unittest

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    inspect,
    insert,
    text,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.auth import AuthSession, User
from backend.conversations import Conversation, ConversationMessage
from backend.documents import StudyDocument
from backend.migrate import (
    _migration_ledger,
    migrate_connection,
    preflight_connection,
)
from backend.migrations.v003_fastapi_schema_reconciliation import (
    SchemaMismatchError,
    VERSION,
    _parse_conversation_type_check,
    _parse_membership_check,
    preflight,
    upgrade,
)
from backend.migrations.v004_study_documents import (
    VERSION as STUDY_DOCUMENTS_VERSION,
    preflight as study_documents_preflight,
)
from backend.migrations.v005_learning_and_planner import (
    VERSION as USER_FEATURES_VERSION,
    preflight as user_features_preflight,
)
from backend.migrations.v006_document_original_files import (
    VERSION as DOCUMENT_ORIGINAL_FILES_VERSION,
    upgrade as original_files_upgrade,
)
from backend.migrations.v007_normalized_flashcards import (
    VERSION as NORMALIZED_FLASHCARDS_VERSION,
    upgrade as normalized_flashcards_upgrade,
)
from backend.migrations.v008_planner_task_descriptions import (
    VERSION as PLANNER_TASK_DESCRIPTIONS_VERSION,
    upgrade as planner_task_descriptions_upgrade,
)


def create_existing_neon_schema(
    engine,
    *,
    user_id_type=Integer,
    duplicate_emails=False,
    short_columns=False,
    nullable_title=False,
    check_values=None,
    null_title=False,
    semantic_conversation_check=True,
):
    has_custom_check_values = check_values is not None
    check_values = check_values or {
        "conversation_type": "'chat', 'study'",
        "status": "'active', 'archived'",
        "sender_type": "'user', 'assistant'",
    }
    metadata = MetaData()
    users = Table(
        "users",
        metadata,
        Column(
            "user_id",
            user_id_type,
            primary_key=True,
            autoincrement=user_id_type is Integer,
        ),
        Column("email", String(40 if short_columns else 320), nullable=False, unique=True),
        Column("password_hash", String(40 if short_columns else 255), nullable=False),
        Column("full_name", String(40 if short_columns else 100), nullable=False),
        Column("is_active", Boolean, nullable=False, server_default=text("1")),
        Column("created_at", DateTime(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP")),
        Column("updated_at", DateTime(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP")),
    )
    conversations = Table(
        "conversations",
        metadata,
        Column("conversation_id", Integer, primary_key=True, autoincrement=True),
        Column("user_id", Integer, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False),
        Column("conversation_type", String(8 if short_columns else 32), nullable=False),
        Column("agent_id", Integer, nullable=True),
        Column("title", String(40 if short_columns else 200), nullable=nullable_title),
        Column("status", String(8 if short_columns else 32), nullable=False),
        Column("created_at", DateTime(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP")),
        Column("updated_at", DateTime(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP")),
        CheckConstraint(
            (
                "(conversation_type = 'surface' AND agent_id IS NULL) "
                "OR (conversation_type = 'agent' AND agent_id IS NOT NULL) "
                "OR (conversation_type = 'chat' AND agent_id IS NULL)"
                if semantic_conversation_check
                else "(conversation_type = 'surface' AND agent_id IS NULL) "
                "OR (conversation_type = 'agent' AND agent_id IS NOT NULL)"
            ),
            name="ck_conversations_conversation_type_agent",
        )
        if not has_custom_check_values
        else CheckConstraint(f"conversation_type IN ({check_values['conversation_type']})"),
        CheckConstraint(f"status IN ({check_values['status']})"),
    )
    messages = Table(
        "messages",
        metadata,
        Column("message_id", Integer, primary_key=True, autoincrement=True),
        Column("conversation_id", Integer, ForeignKey("conversations.conversation_id", ondelete="CASCADE"), nullable=False),
        Column("sender_type", String(8 if short_columns else 20), nullable=False),
        Column("content", Text, nullable=False),
        Column("model_key", String(40 if short_columns else 200), nullable=True),
        Column("metadata", JSON, nullable=True),
        Column("created_at", DateTime(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP")),
        CheckConstraint(f"sender_type IN ({check_values['sender_type']})"),
    )
    Index("ix_conversations_user_updated", conversations.c.user_id, conversations.c.updated_at)
    Index("ix_messages_conversation_created", messages.c.conversation_id, messages.c.message_id)
    metadata.create_all(engine)

    if null_title:
        with engine.begin() as connection:
            connection.execute(
                users.insert().values(
                    email="null-title@example.com",
                    password_hash="synthetic-hash",
                    full_name="Synthetic Learner",
                )
            )
            connection.execute(
                conversations.insert().values(
                    user_id=1,
                    conversation_type="chat",
                    title=None,
                    status="active",
                )
            )

    if duplicate_emails:
        with engine.begin() as connection:
            connection.execute(
                users.insert(),
                [
                    {
                        "email": "learner@example.com",
                        "password_hash": "synthetic-hash",
                        "full_name": "Synthetic Learner",
                    },
                    {
                        "email": "LEARNER@example.com",
                        "password_hash": "synthetic-hash",
                        "full_name": "Synthetic Duplicate",
                    },
                ],
            )


class SchemaReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://")

    def tearDown(self):
        self.engine.dispose()

    def test_orm_maps_to_existing_neon_names(self):
        self.assertIn("user_id", User.__table__.columns)
        self.assertIn("full_name", User.__table__.columns)
        self.assertIn("conversation_id", Conversation.__table__.columns)
        self.assertEqual(ConversationMessage.__tablename__, "messages")
        self.assertIn("message_id", ConversationMessage.__table__.columns)
        self.assertIn("sender_type", ConversationMessage.__table__.columns)
        self.assertIn("model_key", ConversationMessage.__table__.columns)
        self.assertIn("metadata", ConversationMessage.__table__.columns)
        self.assertIn("agent_id", Conversation.__table__.columns)

    def test_reconciliation_creates_fastapi_tables_on_empty_sqlite(self):
        with self.engine.begin() as connection:
            migrate_connection(connection)

        tables = set(inspect(self.engine).get_table_names())
        self.assertTrue(
            {
                "users",
                "auth_sessions",
                "conversations",
                "messages",
                "study_documents",
                "planner_tasks",
                "quiz_attempts",
                "flashcard_decks",
                "learning_activities",
                "quanta_schema_migrations",
            }.issubset(tables)
        )
        self.assertNotIn("conversation_messages", tables)
        planner_columns = {
            column["name"]
            for column in inspect(self.engine).get_columns("planner_tasks")
        }
        self.assertIn("description", planner_columns)

    def test_reconciliation_reuses_existing_tables_and_preserves_rows(self):
        create_existing_neon_schema(self.engine)
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    "CREATE TABLE agent_sessions "
                    "(session_id INTEGER PRIMARY KEY, session_data TEXT NOT NULL)"
                )
            )
            connection.execute(
                text("INSERT INTO agent_sessions (session_id, session_data) VALUES (9, 'keep')")
            )
            user_id = connection.execute(
                text(
                    "INSERT INTO users (email, password_hash, full_name, is_active) "
                    "VALUES ('legacy@example.com', 'synthetic-hash', 'Legacy Learner', 1) "
                    "RETURNING user_id"
                )
            ).scalar_one()
            conversation_id = connection.execute(
                text(
                    "INSERT INTO conversations "
                    "(user_id, conversation_type, title, status) "
                    "VALUES (:user_id, 'chat', 'Existing chat', 'active') "
                    "RETURNING conversation_id"
                ),
                {"user_id": user_id},
            ).scalar_one()
            connection.execute(
                text(
                    "INSERT INTO messages "
                    "(conversation_id, sender_type, content, model_key, metadata) "
                    "VALUES (:conversation_id, 'assistant', 'Persisted reply', "
                    "'QuantaLM', '{\"agent\": \"Study\"}')"
                ),
                {"conversation_id": conversation_id},
            )

        with self.engine.begin() as connection:
            migrate_connection(connection)

        with Session(self.engine) as session:
            user = session.get(User, user_id)
            conversation = session.get(Conversation, conversation_id)
            messages = session.query(ConversationMessage).all()
            sessions = session.query(AuthSession).count()
            self.assertEqual(user.name, "Legacy Learner")
            self.assertEqual(user.profile, {})
            self.assertEqual(conversation.title, "Existing chat")
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0].role, "assistant")
            self.assertEqual(messages[0].agent, "Study")
            self.assertEqual(messages[0].model, "QuantaLM")
            self.assertEqual(sessions, 0)

        table_names = set(inspect(self.engine).get_table_names())
        self.assertIn("messages", table_names)
        self.assertIn("agent_sessions", table_names)
        self.assertNotIn("conversation_messages", table_names)
        with self.engine.connect() as connection:
            self.assertEqual(
                connection.execute(
                    text("SELECT session_data FROM agent_sessions WHERE session_id = 9")
                ).scalar_one(),
                "keep",
            )

    def test_migrations_are_repeatable_and_record_each_version_once(self):
        create_existing_neon_schema(self.engine)

        for _ in range(2):
            with self.engine.begin() as connection:
                migrate_connection(connection)

        with self.engine.connect() as connection:
            versions = connection.execute(
                text("SELECT version FROM quanta_schema_migrations")
            ).scalars().all()
        self.assertEqual(
            versions,
            [
                VERSION,
                STUDY_DOCUMENTS_VERSION,
                USER_FEATURES_VERSION,
                DOCUMENT_ORIGINAL_FILES_VERSION,
                NORMALIZED_FLASHCARDS_VERSION,
                PLANNER_TASK_DESCRIPTIONS_VERSION,
            ],
        )

    def test_preflight_allows_pending_v004_when_v003_is_already_applied(self):
        create_existing_neon_schema(self.engine)
        with self.engine.begin() as connection:
            upgrade(connection)
            StudyDocument.__table__.drop(connection, checkfirst=True)
            ledger, _ = _migration_ledger(connection)
            connection.execute(insert(ledger).values(version=VERSION))

        before_tables = set(inspect(self.engine).get_table_names())
        with self.engine.connect() as connection:
            reports = preflight_connection(connection)

        self.assertTrue(all(report["passed"] for report in reports), reports)
        self.assertTrue(
            any(report["label"] == "Study document storage" for report in reports)
        )
        self.assertNotIn("study_documents", before_tables)
        self.assertEqual(set(inspect(self.engine).get_table_names()), before_tables)

        with self.engine.begin() as connection:
            migrate_connection(connection)
        self.assertIn("study_documents", inspect(self.engine).get_table_names())
        with self.engine.connect() as connection:
            versions = connection.execute(
                text("SELECT version FROM quanta_schema_migrations ORDER BY version")
            ).scalars().all()
        self.assertEqual(
            versions,
            [
                VERSION,
                STUDY_DOCUMENTS_VERSION,
                USER_FEATURES_VERSION,
                DOCUMENT_ORIGINAL_FILES_VERSION,
                NORMALIZED_FLASHCARDS_VERSION,
                PLANNER_TASK_DESCRIPTIONS_VERSION,
            ],
        )

    def test_normalized_flashcards_migration_backfills_legacy_json_without_losing_rows(self):
        with self.engine.begin() as connection:
            connection.exec_driver_sql(
                "CREATE TABLE users (user_id INTEGER PRIMARY KEY)"
            )
            connection.exec_driver_sql(
                "CREATE TABLE study_documents (document_id INTEGER PRIMARY KEY)"
            )
            connection.exec_driver_sql(
                "CREATE TABLE flashcard_decks ("
                "deck_id INTEGER PRIMARY KEY, "
                "user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE, "
                "document_id INTEGER REFERENCES study_documents(document_id) ON DELETE SET NULL, "
                "topic VARCHAR(200) NOT NULL, cards JSON NOT NULL, "
                "current_index INTEGER NOT NULL DEFAULT 0, "
                "known_cards JSON NOT NULL DEFAULT '[]', "
                "provider VARCHAR(100) NOT NULL, model VARCHAR(200) NOT NULL, "
                "created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL)"
            )
            connection.exec_driver_sql(
                "CREATE INDEX ix_flashcard_decks_user_created "
                "ON flashcard_decks (user_id, created_at)"
            )
            connection.exec_driver_sql(
                "INSERT INTO users (user_id) VALUES (7)"
            )
            connection.exec_driver_sql(
                "INSERT INTO study_documents (document_id) VALUES (11)"
            )
            connection.execute(
                text(
                    "INSERT INTO flashcard_decks "
                    "(deck_id, user_id, document_id, topic, cards, provider, model, created_at, updated_at) "
                    "VALUES (23, 7, 11, 'Review: notes.pdf', :cards, 'Google Gemini', "
                    "'test-model', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                ),
                {
                    "cards": json.dumps([
                        {"front": "What is X?", "back": "X is the answer."},
                        {"front": "What is Y?", "back": "Y is another answer."},
                    ])
                },
            )
            normalized_flashcards_upgrade(connection)
            normalized_flashcards_upgrade(connection)

        with self.engine.connect() as connection:
            deck = connection.execute(
                text(
                    "SELECT deck_id, user_id, document_id, title, source_type, "
                    "total_cards, cards FROM flashcard_decks WHERE deck_id = 23"
                )
            ).one()
            cards = connection.execute(
                text(
                    "SELECT question, answer, card_order, times_reviewed, times_remembered "
                    "FROM flashcards WHERE deck_id = 23 ORDER BY card_order"
                )
            ).all()
        self.assertEqual(deck.deck_id, 23)
        self.assertEqual(deck.user_id, 7)
        self.assertEqual(deck.document_id, 11)
        self.assertEqual(deck.title, "Review: notes.pdf")
        self.assertEqual(deck.source_type, "pdf")
        self.assertEqual(deck.total_cards, 2)
        self.assertEqual(len(json.loads(deck.cards)), 2)
        self.assertEqual(
            [(card.question, card.answer, card.card_order) for card in cards],
            [
                ("What is X?", "X is the answer.", 0),
                ("What is Y?", "Y is another answer.", 1),
            ],
        )
        self.assertTrue(all(card.times_reviewed == card.times_remembered == 0 for card in cards))

    def test_original_pdf_migration_preserves_existing_document_rows(self):
        with self.engine.begin() as connection:
            connection.exec_driver_sql(
                "CREATE TABLE study_documents "
                "(document_id INTEGER PRIMARY KEY, filename VARCHAR(255) NOT NULL)"
            )
            connection.exec_driver_sql(
                "INSERT INTO study_documents (document_id, filename) "
                "VALUES (17, 'existing.pdf')"
            )
            original_files_upgrade(connection)

        columns = {
            column["name"]: column
            for column in inspect(self.engine).get_columns("study_documents")
        }
        self.assertIn("original_file", columns)
        self.assertTrue(columns["original_file"]["nullable"])
        with self.engine.connect() as connection:
            row = connection.execute(
                text("SELECT document_id, filename, original_file FROM study_documents")
            ).one()
        self.assertEqual(row.document_id, 17)
        self.assertEqual(row.filename, "existing.pdf")
        self.assertIsNone(row.original_file)

    def test_planner_task_description_migration_preserves_existing_tasks(self):
        with self.engine.begin() as connection:
            connection.exec_driver_sql(
                "CREATE TABLE planner_tasks "
                "(task_id INTEGER PRIMARY KEY, title VARCHAR(200) NOT NULL, "
                "due_date DATE NOT NULL)"
            )
            connection.exec_driver_sql(
                "INSERT INTO planner_tasks (task_id, title, due_date) "
                "VALUES (23, 'Existing task', '2026-11-13')"
            )
            planner_task_descriptions_upgrade(connection)

        columns = {
            column["name"]: column
            for column in inspect(self.engine).get_columns("planner_tasks")
        }
        self.assertIn("description", columns)
        self.assertTrue(columns["description"]["nullable"])
        with self.engine.connect() as connection:
            row = connection.execute(
                text(
                    "SELECT task_id, title, due_date, description "
                    "FROM planner_tasks WHERE task_id = 23"
                )
            ).one()
        self.assertEqual(row.task_id, 23)
        self.assertEqual(row.title, "Existing task")
        self.assertEqual(row.due_date, "2026-11-13")
        self.assertIsNone(row.description)

    def test_user_features_preflight_is_read_only_when_tables_are_missing(self):
        create_existing_neon_schema(self.engine)
        before_tables = set(inspect(self.engine).get_table_names())

        with self.engine.connect() as connection:
            reports = user_features_preflight(connection)

        self.assertEqual(len(reports), 4)
        self.assertTrue(all(report["passed"] for report in reports), reports)
        self.assertEqual(set(inspect(self.engine).get_table_names()), before_tables)
        self.assertNotIn("planner_tasks", before_tables)

    def test_study_document_preflight_is_read_only_when_table_is_missing(self):
        create_existing_neon_schema(self.engine)
        before_tables = set(inspect(self.engine).get_table_names())

        with self.engine.connect() as connection:
            reports = study_documents_preflight(connection)

        self.assertEqual(len(reports), 1)
        self.assertTrue(reports[0]["passed"])
        self.assertEqual(set(inspect(self.engine).get_table_names()), before_tables)
        self.assertNotIn("study_documents", before_tables)

    def test_preflight_is_read_only_for_compatible_existing_schema(self):
        create_existing_neon_schema(self.engine)
        before_tables = set(inspect(self.engine).get_table_names())
        before_indexes = {
            table: tuple(index["name"] for index in inspect(self.engine).get_indexes(table))
            for table in ("users", "conversations", "messages")
        }
        before_checks = {
            table: inspect(self.engine).get_check_constraints(table)
            for table in ("users", "conversations", "messages")
        }

        with self.engine.connect() as connection:
            reports = preflight(connection)

        self.assertTrue(all(report["passed"] for report in reports), reports)
        self.assertEqual(set(inspect(self.engine).get_table_names()), before_tables)
        self.assertEqual(
            {
                table: tuple(index["name"] for index in inspect(self.engine).get_indexes(table))
                for table in ("users", "conversations", "messages")
            },
            before_indexes,
        )
        self.assertEqual(
            {
                table: inspect(self.engine).get_check_constraints(table)
                for table in ("users", "conversations", "messages")
            },
            before_checks,
        )
        self.assertNotIn("auth_sessions", before_tables)
        self.assertNotIn("quanta_schema_migrations", before_tables)

    def test_preflight_reports_mismatch_without_writes(self):
        create_existing_neon_schema(self.engine, user_id_type=String)
        before_tables = set(inspect(self.engine).get_table_names())

        with self.engine.connect() as connection:
            reports = preflight(connection)

        self.assertTrue(any(not report["passed"] for report in reports))
        self.assertEqual(set(inspect(self.engine).get_table_names()), before_tables)

    def test_preflight_reports_short_varchar_capacities_and_upgrade_preserves_long_values(self):
        create_existing_neon_schema(self.engine, short_columns=True)
        with self.engine.connect() as connection:
            reports = preflight(connection)

        capacity_reports = {
            report["label"]: report["detail"]
            for report in reports
            if "varchar capacity" in report["label"]
        }
        self.assertIn("varchar(40)", capacity_reports["users.email varchar capacity"])
        self.assertIn("320", capacity_reports["users.email varchar capacity"])
        self.assertTrue(all(report["passed"] for report in reports), reports)

        with self.engine.begin() as connection:
            migrate_connection(connection)
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO users (email, password_hash, full_name, is_active, profile) "
                    "VALUES ('long@example.com', 'synthetic-hash', :name, 1, '{}')"
                ),
                {"name": "A" * 120},
            )
            self.assertEqual(
                connection.execute(
                    text("SELECT length(full_name) FROM users WHERE email = 'long@example.com'")
                ).scalar_one(),
                120,
            )

    def test_existing_check_values_are_reported_and_incompatible_values_fail_closed(self):
        create_existing_neon_schema(self.engine)
        with self.engine.connect() as connection:
            reports = preflight(connection)
        check_report = next(
            report for report in reports
            if report["label"] == "conversations.conversation_type CHECK definitions and distinct values"
        )
        self.assertIn("surface", check_report["detail"])
        self.assertIn("agent", check_report["detail"])
        self.assertIn("chat", check_report["detail"])
        self.assertIn("NULL agent_id", check_report["detail"])

        incompatible_engine = create_engine("sqlite://")
        try:
            create_existing_neon_schema(
                incompatible_engine,
                check_values={
                    "conversation_type": "'study'",
                    "status": "'archived'",
                    "sender_type": "'user'",
                },
            )
            with incompatible_engine.begin() as connection:
                with self.assertRaisesRegex(
                    SchemaMismatchError,
                    "does not preserve the supported type/agent_id rules",
                ):
                    migration_module = __import__(
                        "backend.migrations.v003_fastapi_schema_reconciliation",
                        fromlist=["upgrade"],
                    )
                    migration_module.upgrade(connection)
        finally:
            incompatible_engine.dispose()

    def test_existing_surface_and_agent_conversation_relationships_are_preserved(self):
        create_existing_neon_schema(self.engine)
        with self.engine.begin() as connection:
            user_id = connection.execute(
                text(
                    "INSERT INTO users (email, password_hash, full_name, is_active) "
                    "VALUES ('types@example.com', 'synthetic-hash', 'Type Test', 1) "
                    "RETURNING user_id"
                )
            ).scalar_one()
            connection.execute(
                text(
                    "INSERT INTO conversations "
                    "(user_id, conversation_type, agent_id, title, status) "
                    "VALUES (:user_id, 'surface', NULL, 'Surface', 'active'), "
                    "(:user_id, 'agent', 17, 'Agent', 'active'), "
                    "(:user_id, 'chat', NULL, 'FastAPI chat', 'active')"
                ),
                {"user_id": user_id},
            )

        with self.engine.connect() as connection:
            rows = connection.execute(
                text("SELECT conversation_type, agent_id FROM conversations ORDER BY conversation_id")
            ).all()
        self.assertEqual(rows, [("surface", None), ("agent", 17), ("chat", None)])

    def test_invalid_type_agent_id_pairs_are_rejected_by_database_check(self):
        create_existing_neon_schema(self.engine)
        with self.engine.begin() as connection:
            user_id = connection.execute(
                text(
                    "INSERT INTO users (email, password_hash, full_name, is_active) "
                    "VALUES ('invalid@example.com', 'synthetic-hash', 'Invalid Test', 1) "
                    "RETURNING user_id"
                )
            ).scalar_one()
            invalid_pairs = (
                ("surface", 17),
                ("agent", None),
                ("chat", 17),
                ("other", None),
            )
        for conversation_type, agent_id in invalid_pairs:
            with self.subTest(conversation_type=conversation_type, agent_id=agent_id):
                with self.assertRaises(IntegrityError):
                    with self.engine.begin() as connection:
                        connection.execute(
                            text(
                                "INSERT INTO conversations "
                                "(user_id, conversation_type, agent_id, title, status) "
                                "VALUES (:user_id, :kind, :agent_id, 'Invalid', 'active')"
                            ),
                            {"user_id": user_id, "kind": conversation_type, "agent_id": agent_id},
                        )

    def test_legacy_semantic_check_fails_migration_without_schema_changes(self):
        create_existing_neon_schema(
            self.engine,
            semantic_conversation_check=False,
        )
        before_tables = set(inspect(self.engine).get_table_names())
        with self.assertRaises(SchemaMismatchError):
            with self.engine.begin() as connection:
                upgrade(connection)
        self.assertEqual(set(inspect(self.engine).get_table_names()), before_tables)
        self.assertNotIn("auth_sessions", before_tables)
        self.assertNotIn("quanta_schema_migrations", before_tables)

    def test_migration_transaction_rolls_back_earlier_schema_changes_on_late_failure(self):
        create_existing_neon_schema(self.engine)
        with self.engine.begin() as connection:
            connection.execute(text(
                "CREATE TABLE auth_sessions ("
                "token_hash VARCHAR(64) PRIMARY KEY, "
                "user_id INTEGER NOT NULL, "
                "expires_at DATETIME NOT NULL, "
                "created_at DATETIME NOT NULL)"
            ))

        before_tables = set(inspect(self.engine).get_table_names())
        with self.engine.connect() as connection:
            connection.exec_driver_sql("BEGIN")
            with self.assertRaisesRegex(
                SchemaMismatchError,
                "SQLite cannot add the missing foreign key",
            ):
                upgrade(connection)
            connection.rollback()

        self.assertEqual(set(inspect(self.engine).get_table_names()), before_tables)
        self.assertNotIn(
            "profile",
            {column["name"] for column in inspect(self.engine).get_columns("users")},
        )
        self.assertEqual(
            inspect(self.engine).get_indexes("auth_sessions"), []
        )

    def test_preflight_reports_legacy_semantics_as_addable_and_is_read_only(self):
        create_existing_neon_schema(
            self.engine,
            semantic_conversation_check=False,
        )
        before = inspect(self.engine).get_check_constraints("conversations")
        with self.engine.connect() as connection:
            reports = preflight(connection)
        type_report = next(
            report for report in reports
            if report["label"] == "conversations.conversation_type CHECK definitions and distinct values"
        )
        self.assertFalse(type_report["passed"])
        self.assertIn("SQLite cannot widen", type_report["detail"])
        self.assertEqual(
            inspect(self.engine).get_check_constraints("conversations"), before
        )

    def test_null_conversation_titles_fail_preflight_without_rewrite(self):
        create_existing_neon_schema(
            self.engine,
            nullable_title=True,
            null_title=True,
        )
        with self.engine.connect() as connection:
            reports = preflight(connection)
        title_report = next(
            report for report in reports
            if report["label"] == "conversations.title NULL rows and nullability"
        )
        self.assertFalse(title_report["passed"])
        self.assertIn("1 titles are NULL", title_report["detail"])
        with self.engine.connect() as connection:
            self.assertEqual(
                connection.execute(text("SELECT count(*) FROM conversations WHERE title IS NULL"))
                .scalar_one(),
                1,
            )

    def test_null_conversation_titles_abort_migration_without_schema_or_data_changes(self):
        create_existing_neon_schema(
            self.engine,
            nullable_title=True,
            null_title=True,
        )
        tables_before = set(inspect(self.engine).get_table_names())
        with self.assertRaisesRegex(SchemaMismatchError, "NULL conversation titles"):
            with self.engine.begin() as connection:
                upgrade(connection)

        self.assertEqual(set(inspect(self.engine).get_table_names()), tables_before)
        with self.engine.connect() as connection:
            self.assertEqual(
                connection.execute(
                    text("SELECT count(*) FROM conversations WHERE title IS NULL")
                ).scalar_one(),
                1,
            )

    def test_membership_check_parser_preserves_existing_literals_exactly(self):
        values = _parse_membership_check(
            "CHECK ((sender_type)::text = ANY "
            "((ARRAY['User'::character varying, 'assistant'::character varying])::text[]))",
            "sender_type",
        )
        self.assertEqual(values, {"User", "assistant"})

    def test_conversation_check_parser_accepts_postgresql_constraint_format(self):
        rules = _parse_conversation_type_check(
            "CHECK (((conversation_type)::text = 'surface'::text AND "
            "(agent_id IS NULL)) OR ((conversation_type)::text = 'agent'::text "
            "AND (agent_id IS NOT NULL)) OR ((conversation_type)::text = 'chat'::text "
            "AND (agent_id IS NULL)))"
        )
        self.assertEqual(
            rules,
            {("surface", True), ("agent", False), ("chat", True)},
        )

    def test_schema_mismatch_fails_before_adding_structures(self):
        create_existing_neon_schema(self.engine, user_id_type=String)

        with self.assertRaisesRegex(SchemaMismatchError, "data type"):
            with self.engine.begin() as connection:
                migrate_connection(connection)

        table_names = set(inspect(self.engine).get_table_names())
        self.assertNotIn("auth_sessions", table_names)
        self.assertNotIn("quanta_schema_migrations", table_names)
        self.assertNotIn("profile", {column["name"] for column in inspect(self.engine).get_columns("users")})

    def test_case_insensitive_duplicates_block_unique_index_creation(self):
        create_existing_neon_schema(self.engine, duplicate_emails=True)

        with self.engine.begin() as connection:
            with self.assertRaisesRegex(SchemaMismatchError, "duplicate emails"):
                migrate_connection(connection)

        self.assertNotIn("auth_sessions", inspect(self.engine).get_table_names())
        self.assertEqual(
            self.engine.connect()
            .execute(text("SELECT count(*) FROM users"))
            .scalar_one(),
            2,
        )


if __name__ == "__main__":
    unittest.main()