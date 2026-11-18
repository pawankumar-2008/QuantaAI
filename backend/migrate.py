import importlib
import logging
import argparse
import os

from sqlalchemy import Column, DateTime, MetaData, String, Table, insert, inspect, text
from sqlalchemy.exc import SQLAlchemyError

from backend.auth import _engine


logger = logging.getLogger(__name__)
RECONCILIATION_MODULE = "backend.migrations.v003_fastapi_schema_reconciliation"
STUDY_DOCUMENTS_MODULE = "backend.migrations.v004_study_documents"
USER_FEATURES_MODULE = "backend.migrations.v005_learning_and_planner"
DOCUMENT_ORIGINAL_FILES_MODULE = "backend.migrations.v006_document_original_files"
NORMALIZED_FLASHCARDS_MODULE = "backend.migrations.v007_normalized_flashcards"
PLANNER_TASK_DESCRIPTIONS_MODULE = "backend.migrations.v008_planner_task_descriptions"
MIGRATION_MODULES = (
    RECONCILIATION_MODULE,
    STUDY_DOCUMENTS_MODULE,
    USER_FEATURES_MODULE,
    DOCUMENT_ORIGINAL_FILES_MODULE,
    NORMALIZED_FLASHCARDS_MODULE,
    PLANNER_TASK_DESCRIPTIONS_MODULE,
)


def _migration_ledger(connection):
    schema = "public" if connection.dialect.name == "postgresql" else None
    ledger_metadata = MetaData()
    ledger = Table(
        "quanta_schema_migrations",
        ledger_metadata,
        Column("version", String(255), primary_key=True),
        Column(
            "applied_at",
            DateTime(timezone=True),
            nullable=False,
            server_default=text("CURRENT_TIMESTAMP"),
        ),
        schema=schema,
    )
    ledger.create(connection, checkfirst=True)
    ledger_columns = {
        column["name"]: column
        for column in inspect(connection).get_columns(
            "quanta_schema_migrations", schema=schema
        )
    }
    if not {"version", "applied_at"}.issubset(ledger_columns):
        raise RuntimeError("Migration ledger schema is incompatible.")
    if inspect(connection).get_pk_constraint(
        "quanta_schema_migrations", schema=schema
    ).get("constrained_columns") != ["version"]:
        raise RuntimeError("Migration ledger primary key is incompatible.")
    if not isinstance(ledger_columns["version"]["type"], String):
        raise RuntimeError("Migration ledger version column type is incompatible.")
    if (
        ledger_columns["version"]["type"].length is not None
        and ledger_columns["version"]["type"].length < 255
    ):
        raise RuntimeError("Migration ledger version column is too short.")
    if not isinstance(ledger_columns["applied_at"]["type"], DateTime):
        raise RuntimeError("Migration ledger timestamp column type is incompatible.")
    if (
        connection.dialect.name == "postgresql"
        and not ledger_columns["applied_at"]["type"].timezone
    ):
        raise RuntimeError("Migration ledger timestamp must be timezone-aware.")
    if ledger_columns["applied_at"].get("nullable", True):
        raise RuntimeError("Migration ledger timestamp must be NOT NULL.")
    if ledger_columns["applied_at"].get("default") is None:
        raise RuntimeError("Migration ledger timestamp requires a server default.")
    for name, column in ledger_columns.items():
        if name in {"version", "applied_at"}:
            continue
        if column.get("nullable", True) or column.get("default") is not None:
            continue
        if column.get("identity"):
            continue
        raise RuntimeError(
            f"Migration ledger column {name} has no server default."
        )

    ledger_name = f"{schema}.quanta_schema_migrations" if schema else "quanta_schema_migrations"
    return ledger, ledger_name


def _applied_migration_versions(connection) -> set[str]:
    schema = "public" if connection.dialect.name == "postgresql" else None
    tables = set(inspect(connection).get_table_names(schema=schema))
    if "quanta_schema_migrations" not in tables:
        return set()

    _, ledger_name = _migration_ledger(connection)
    return set(
        connection.execute(
            text(f"SELECT version FROM {ledger_name}")
        ).scalars().all()
    )


def migrate_connection(connection) -> None:
    applied_versions = _applied_migration_versions(connection)
    for migration_module in MIGRATION_MODULES:
        migration = importlib.import_module(migration_module)
        if migration.VERSION in applied_versions:
            migration.validate_schema(connection)
            continue

        migration.upgrade(connection)
        migration.validate_schema(connection)

        ledger, ledger_name = _migration_ledger(connection)
        connection.execute(insert(ledger).values(version=migration.VERSION))
        applied_versions.add(migration.VERSION)


def preflight_connection(connection) -> list[dict[str, str | bool]]:
    applied_versions = _applied_migration_versions(connection)
    reports = []
    for migration_module in MIGRATION_MODULES:
        migration = importlib.import_module(migration_module)
        if migration.VERSION in applied_versions:
            migration.validate_schema(connection)
            reports.append(
                {
                    "passed": True,
                    "label": migration.VERSION,
                    "detail": "already applied; schema validated",
                }
            )
        else:
            reports.extend(migration.preflight(connection))
    return reports


def migrate() -> None:
    with _engine().begin() as connection:
        migrate_connection(connection)


def preflight() -> bool:
    migrations = [
        importlib.import_module(module_name)
        for module_name in MIGRATION_MODULES
    ]
    has_database_url = bool(os.getenv("DATABASE_URL", "").strip())
    required_neon_variables = (
        "NEON_DB_HOST",
        "NEON_DB_NAME",
        "NEON_DB_PASSWORD",
    )
    has_neon_settings = all(os.getenv(name) for name in required_neon_variables) and bool(
        os.getenv("NEON_DB_USER") or os.getenv("NEON_DB_USERNAME")
    )
    if not has_database_url and not has_neon_settings:
        print(
            "FAIL database configuration: DATABASE_URL and complete NEON_DB_* "
            "settings are unavailable in this process"
        )
        print("NOT SAFE TO APPLY")
        return False
    try:
        with _engine().connect() as connection:
            if connection.dialect.name == "postgresql":
                connection.exec_driver_sql("SET TRANSACTION READ ONLY")
            elif connection.dialect.name == "sqlite":
                connection.exec_driver_sql("PRAGMA query_only = ON")
            else:
                print("FAIL database dialect: preflight supports PostgreSQL and SQLite only")
                print("NOT SAFE TO APPLY")
                return False
            reports = preflight_connection(connection)
    except Exception as error:
        print(
            "FAIL database connection/configuration: "
            f"{type(error).__name__}; details redacted"
        )
        print("NOT SAFE TO APPLY")
        return False

    for item in reports:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"{status} {item['label']}: {item['detail']}")

    safe_to_apply = all(bool(item["passed"]) for item in reports)
    print("SAFE TO APPLY" if safe_to_apply else "NOT SAFE TO APPLY")
    return safe_to_apply


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage the FastAPI database schema.")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="inspect schema read-only and report whether reconciliation is safe",
    )
    arguments = parser.parse_args()
    if arguments.preflight:
        raise SystemExit(0 if preflight() else 1)

    try:
        migrate()
    except (RuntimeError, SQLAlchemyError) as error:
        logger.error("Database migration failed error_type=%s", type(error).__name__)
        raise SystemExit(
            "Database migration failed; check configuration and connectivity."
        ) from None
