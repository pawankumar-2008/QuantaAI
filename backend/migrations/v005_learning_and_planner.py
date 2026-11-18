from sqlalchemy import DateTime, String, inspect
from sqlalchemy.engine import Connection

from backend.learning import FlashcardDeck, LearningActivity, QuizAttempt
from backend.migrations.v003_fastapi_schema_reconciliation import (
    SchemaMismatchError,
    _schema,
)
from backend.planner import PlannerTask


VERSION = "005_learning_and_planner"
TABLE_MODELS = (
    PlannerTask,
    QuizAttempt,
    FlashcardDeck,
    LearningActivity,
)


def _validate_table(
    connection: Connection,
    model,
    *,
    allow_missing_columns: frozenset[str] = frozenset(),
    allow_missing_checks: frozenset[str] = frozenset(),
) -> None:
    table = model.__table__
    schema = _schema(connection)
    inspector = inspect(connection)
    table_names = set(inspector.get_table_names(schema=schema))
    if table.name not in table_names:
        raise SchemaMismatchError(f"Required {table.name} table is missing.")

    columns = {
        column["name"]: column
        for column in inspector.get_columns(table.name, schema=schema)
    }
    for expected in table.columns:
        actual = columns.get(expected.name)
        if actual is None and expected.name in allow_missing_columns:
            continue
        if actual is None or not isinstance(actual["type"], type(expected.type)):
            raise SchemaMismatchError(
                f"{table.name}.{expected.name} has an incompatible data type."
            )
        if (
            isinstance(expected.type, String)
            and expected.type.length is not None
            and actual["type"].length is not None
            and actual["type"].length < expected.type.length
        ):
            raise SchemaMismatchError(
                f"{table.name}.{expected.name} is too short."
            )
        if not expected.nullable and actual.get("nullable", True):
            raise SchemaMismatchError(
                f"{table.name}.{expected.name} must be NOT NULL."
            )
        if (
            isinstance(expected.type, DateTime)
            and connection.dialect.name == "postgresql"
            and not getattr(actual["type"], "timezone", False)
        ):
            raise SchemaMismatchError(
                f"{table.name}.{expected.name} must preserve timezone-aware timestamps."
            )

    expected_primary_key = [column.name for column in table.primary_key.columns]
    primary_key = inspector.get_pk_constraint(table.name, schema=schema)
    if primary_key.get("constrained_columns") != expected_primary_key:
        raise SchemaMismatchError(
            f"{table.name} has an incompatible primary key."
        )

    actual_foreign_keys = inspector.get_foreign_keys(table.name, schema=schema)
    for foreign_key in table.foreign_keys:
        matching = any(
            item.get("constrained_columns") == [foreign_key.parent.name]
            and item.get("referred_table") == foreign_key.column.table.name
            and item.get("referred_columns") == [foreign_key.column.name]
            and str((item.get("options") or {}).get("ondelete", "")).upper()
            == str(foreign_key.ondelete or "").upper()
            for item in actual_foreign_keys
        )
        if not matching:
            raise SchemaMismatchError(
                f"{table.name}.{foreign_key.parent.name} has an incompatible foreign key."
            )

    actual_indexes = inspector.get_indexes(table.name, schema=schema)
    for expected_index in table.indexes:
        if not any(
            index.get("name") == expected_index.name
            and index.get("column_names")
            == [column.name for column in expected_index.columns]
            and bool(index.get("unique")) == bool(expected_index.unique)
            for index in actual_indexes
        ):
            raise SchemaMismatchError(
                f"Required index {expected_index.name} is missing or incompatible."
            )

    actual_checks = {
        item.get("name")
        for item in inspector.get_check_constraints(table.name, schema=schema)
    }
    for expected_check in table.constraints:
        if expected_check.__class__.__name__ == "CheckConstraint":
            if (
                expected_check.name not in actual_checks
                and expected_check.name not in allow_missing_checks
            ):
                raise SchemaMismatchError(
                    f"Required check constraint {expected_check.name} is missing."
                )


def validate_schema(connection: Connection) -> None:
    for model in TABLE_MODELS:
        _validate_table(
            connection,
            model,
            allow_missing_columns=(
                frozenset({"title", "source_type", "total_cards"})
                if model is FlashcardDeck
                else frozenset({"description"})
                if model is PlannerTask
                else frozenset()
            ),
            allow_missing_checks=(
                frozenset({
                    "ck_flashcard_decks_title",
                    "ck_flashcard_decks_source_type",
                    "ck_flashcard_decks_total_cards",
                })
                if model is FlashcardDeck
                else frozenset()
            ),
        )


def upgrade(connection: Connection) -> None:
    schema = _schema(connection)
    tables = set(inspect(connection).get_table_names(schema=schema))
    for model in TABLE_MODELS:
        if model.__tablename__ not in tables:
            model.__table__.create(connection)
            tables.add(model.__tablename__)
    validate_schema(connection)


def preflight(connection: Connection) -> list[dict[str, str | bool]]:
    schema = _schema(connection)
    tables = set(inspect(connection).get_table_names(schema=schema))
    reports = []
    for model in TABLE_MODELS:
        table_name = model.__tablename__
        if table_name not in tables:
            reports.append(
                {
                    "passed": True,
                    "label": table_name,
                    "detail": (
                        f"{table_name} is absent; v005 will add it without "
                        "changing existing tables"
                    ),
                }
            )
            continue
        try:
            _validate_table(
                connection,
                model,
                allow_missing_columns=(
                    frozenset({"title", "source_type", "total_cards"})
                    if model is FlashcardDeck
                    else frozenset({"description"})
                    if model is PlannerTask
                    else frozenset()
                ),
                allow_missing_checks=(
                    frozenset({
                        "ck_flashcard_decks_title",
                        "ck_flashcard_decks_source_type",
                        "ck_flashcard_decks_total_cards",
                    })
                    if model is FlashcardDeck
                    else frozenset()
                ),
            )
        except SchemaMismatchError as error:
            reports.append(
                {
                    "passed": False,
                    "label": table_name,
                    "detail": str(error),
                }
            )
        else:
            reports.append(
                {
                    "passed": True,
                    "label": table_name,
                    "detail": f"existing {table_name} table is compatible",
                }
            )
    return reports
