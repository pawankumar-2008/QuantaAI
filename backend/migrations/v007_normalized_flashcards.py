import json

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Integer,
    JSON,
    MetaData,
    String,
    Table,
    inspect,
    select,
    text,
)
from sqlalchemy.engine import Connection
from sqlalchemy.schema import CreateColumn, CreateTable

from sqlalchemy.dialects.postgresql import JSONB

from backend.documents import StudyDocument
from backend.learning import Flashcard, FlashcardDeck
from backend.migrations.v003_fastapi_schema_reconciliation import (
    SchemaMismatchError,
    _schema,
)


VERSION = "007_normalized_flashcards"
DECK_METADATA_COLUMNS = {
    "title": Column(
        "title",
        String(200),
        nullable=False,
        server_default=text("'Untitled deck'"),
    ),
    "source_type": Column(
        "source_type",
        String(10),
        nullable=False,
        server_default=text("'topic'"),
    ),
    "total_cards": Column(
        "total_cards",
        Integer,
        nullable=False,
        server_default=text("0"),
    ),
}
DECK_CHECKS = {
    "ck_flashcard_decks_title": "length(trim(title)) > 0",
    "ck_flashcard_decks_source_type": "source_type IN ('topic', 'pdf')",
    "ck_flashcard_decks_total_cards": "total_cards >= 0",
}
SQLITE_GUARD_INSERT = "trg_flashcard_decks_validate_insert"
SQLITE_GUARD_UPDATE = "trg_flashcard_decks_validate_update"


def _tables(connection: Connection) -> set[str]:
    return set(inspect(connection).get_table_names(schema=_schema(connection)))


def _column_map(connection: Connection, table_name: str) -> dict[str, dict]:
    return {
        column["name"]: column
        for column in inspect(connection).get_columns(
            table_name,
            schema=_schema(connection),
        )
    }


def _has_foreign_key(
    connection: Connection,
    table_name: str,
    column: str,
    target_table: str,
    target_column: str,
    delete_rule: str,
) -> bool:
    if connection.dialect.name == "sqlite":
        rows = connection.exec_driver_sql(
            f"PRAGMA foreign_key_list('{table_name}')"
        ).all()
        return any(
            row[2] == target_table
            and row[3] == column
            and row[4] == target_column
            and str(row[6]).upper() == delete_rule
            for row in rows
        )

    schema = _schema(connection)
    return any(
        fk.get("constrained_columns") == [column]
        and fk.get("referred_table") == target_table
        and fk.get("referred_columns") == [target_column]
        and str((fk.get("options") or {}).get("ondelete", "")).upper() == delete_rule
        for fk in inspect(connection).get_foreign_keys(table_name, schema=schema)
    )


def _table(connection: Connection, table_name: str) -> Table:
    return Table(
        table_name,
        MetaData(),
        autoload_with=connection,
        resolve_fks=False,
        schema=_schema(connection),
    )


def _add_column(connection: Connection, table: Table, column: Column) -> None:
    preparer = connection.dialect.identifier_preparer
    connection.exec_driver_sql(
        f"ALTER TABLE {preparer.format_table(table)} ADD COLUMN "
        f"{CreateColumn(column).compile(dialect=connection.dialect)}"
    )


def _ensure_deck_checks(connection: Connection, table: Table) -> None:
    schema = _schema(connection)
    if connection.dialect.name == "postgresql":
        present = {
            item.get("name"): item.get("sqltext", "")
            for item in inspect(connection).get_check_constraints(
                table.name,
                schema=schema,
            )
        }
        preparer = connection.dialect.identifier_preparer
        qualified_table = preparer.format_table(table)
        for name, expression in DECK_CHECKS.items():
            if name in present:
                if expression.split("(")[0].split(" ")[0] not in present[name]:
                    raise SchemaMismatchError(
                        f"{table.name}.{name} has an incompatible check constraint."
                    )
                continue
            connection.exec_driver_sql(
                f"ALTER TABLE {qualified_table} ADD CONSTRAINT "
                f"{preparer.quote(name)} CHECK ({expression})"
            )
        return

    if connection.dialect.name == "sqlite":
        preparer = connection.dialect.identifier_preparer
        qualified_table = preparer.format_table(table)
        for trigger_name, operation in (
            (SQLITE_GUARD_INSERT, "INSERT"),
            (SQLITE_GUARD_UPDATE, "UPDATE"),
        ):
            connection.exec_driver_sql(
                f"CREATE TRIGGER IF NOT EXISTS {preparer.quote(trigger_name)} "
                f"BEFORE {operation} ON {qualified_table} "
                "WHEN length(trim(NEW.title)) = 0 "
                "OR NEW.source_type NOT IN ('topic', 'pdf') "
                "OR NEW.total_cards < 0 "
                "BEGIN SELECT RAISE(ABORT, 'invalid flashcard deck metadata'); END"
            )
        return

    raise SchemaMismatchError("Normalized flashcard migration supports PostgreSQL and SQLite only.")


def _cards_from_legacy(value) -> list[dict[str, str]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []
    cards = []
    for item in value:
        if not isinstance(item, dict):
            continue
        question = item.get("front", item.get("question"))
        answer = item.get("back", item.get("answer"))
        if isinstance(question, str) and question.strip() and isinstance(answer, str) and answer.strip():
            cards.append({"question": question.strip(), "answer": answer.strip()})
    return cards


def _backfill_cards(
    connection: Connection,
    *,
    title_was_added: bool,
    source_type_was_added: bool,
    total_cards_was_added: bool,
) -> None:
    decks = _table(connection, FlashcardDeck.__tablename__)
    cards = _table(connection, Flashcard.__tablename__)
    for row in connection.execute(select(decks)).mappings():
        deck_id = row["deck_id"]
        legacy_cards = _cards_from_legacy(row.get("cards"))
        updates = {}
        if title_was_added or not (row.get("title") or "").strip():
            updates["title"] = (row.get("topic") or "Flashcard deck")[:200]
        if source_type_was_added:
            updates["source_type"] = "pdf" if row.get("document_id") else "topic"
        if total_cards_was_added or not row.get("total_cards"):
            updates["total_cards"] = len(legacy_cards)
        if updates:
            connection.execute(
                decks.update().where(decks.c.deck_id == deck_id).values(**updates)
            )

        existing_card = connection.execute(
            select(cards.c.id)
            .where(cards.c.deck_id == deck_id)
            .limit(1)
        ).first()
        if existing_card is not None:
            continue
        for order, card in enumerate(legacy_cards):
            connection.execute(
                cards.insert().values(
                    deck_id=deck_id,
                    question=card["question"],
                    answer=card["answer"],
                    card_order=order,
                    times_reviewed=0,
                    times_remembered=0,
                    created_at=row.get("created_at"),
                )
            )


def validate_schema(connection: Connection) -> None:
    schema = _schema(connection)
    tables = _tables(connection)
    deck_name = FlashcardDeck.__tablename__
    card_name = Flashcard.__tablename__
    if deck_name not in tables or card_name not in tables:
        raise SchemaMismatchError("Flashcard deck and card tables are required.")

    deck_columns = _column_map(connection, deck_name)
    expected_deck_types = {
        "deck_id": Integer,
        "user_id": Integer,
        "title": String,
        "topic": String,
        "source_type": String,
        "document_id": Integer,
        "total_cards": Integer,
        "cards": (JSON, JSONB),
        "created_at": DateTime,
        "updated_at": DateTime,
    }
    for name, expected_type in expected_deck_types.items():
        column = deck_columns.get(name)
        if column is None or not isinstance(column["type"], expected_type):
            raise SchemaMismatchError(f"flashcard_decks.{name} has an incompatible type.")
    for name in ("user_id", "title", "source_type", "total_cards", "created_at", "updated_at"):
        if deck_columns[name].get("nullable", True):
            raise SchemaMismatchError(f"flashcard_decks.{name} must be NOT NULL.")

    deck_pk = inspect(connection).get_pk_constraint(deck_name, schema=schema)
    if deck_pk.get("constrained_columns") != ["deck_id"]:
        raise SchemaMismatchError("flashcard_decks.deck_id must be the primary key.")
    required_deck_fks = (
        ("user_id", "users", "user_id", "CASCADE"),
        ("document_id", "study_documents", "document_id", "SET NULL"),
    )
    for column, target_table, target_column, delete_rule in required_deck_fks:
        if not _has_foreign_key(
            connection,
            deck_name,
            column,
            target_table,
            target_column,
            delete_rule,
        ):
            raise SchemaMismatchError(f"flashcard_decks.{column} has an incompatible foreign key.")

    deck_indexes = inspect(connection).get_indexes(deck_name, schema=schema)
    if not any(
        item.get("name") == "ix_flashcard_decks_user_created"
        and item.get("column_names") == ["user_id", "created_at"]
        for item in deck_indexes
    ):
        raise SchemaMismatchError("The flashcard deck user/created index is missing.")

    card_columns = _column_map(connection, card_name)
    expected_card_types = {
        "id": Integer,
        "deck_id": Integer,
        "question": String,
        "answer": String,
        "card_order": Integer,
        "times_reviewed": Integer,
        "times_remembered": Integer,
        "created_at": DateTime,
    }
    for name, expected_type in expected_card_types.items():
        column = card_columns.get(name)
        if column is None or not isinstance(column["type"], expected_type):
            raise SchemaMismatchError(f"flashcards.{name} has an incompatible type.")
    for name in (
        "deck_id",
        "question",
        "answer",
        "card_order",
        "times_reviewed",
        "times_remembered",
        "created_at",
    ):
        if card_columns[name].get("nullable", True):
            raise SchemaMismatchError(f"flashcards.{name} must be NOT NULL.")

    card_pk = inspect(connection).get_pk_constraint(card_name, schema=schema)
    if card_pk.get("constrained_columns") != ["id"]:
        raise SchemaMismatchError("flashcards.id must be the primary key.")
    if not _has_foreign_key(
        connection,
        card_name,
        "deck_id",
        deck_name,
        "deck_id",
        "CASCADE",
    ):
        raise SchemaMismatchError("flashcards.deck_id requires a cascading deck foreign key.")

    card_indexes = inspect(connection).get_indexes(card_name, schema=schema)
    for name, columns in (
        ("ix_flashcards_deck_created", ["deck_id", "created_at"]),
        ("ix_flashcards_created", ["created_at"]),
    ):
        if not any(item.get("name") == name and item.get("column_names") == columns for item in card_indexes):
            raise SchemaMismatchError(f"Required index {name} is missing.")

    if connection.dialect.name == "postgresql":
        checks = {
            item.get("name"): item.get("sqltext", "")
            for item in inspect(connection).get_check_constraints(deck_name, schema=schema)
        }
        for name, expression in DECK_CHECKS.items():
            if name not in checks or expression.split("(")[0].split(" ")[0] not in checks[name]:
                raise SchemaMismatchError(f"flashcard_decks.{name} is missing or incompatible.")


def upgrade(connection: Connection) -> None:
    schema = _schema(connection)
    tables = _tables(connection)
    deck_name = FlashcardDeck.__tablename__
    if deck_name not in tables:
        raise SchemaMismatchError("Required flashcard_decks table is missing; v005 must run first.")
    deck = _table(connection, deck_name)
    existing_columns = set(_column_map(connection, deck_name))
    added = set()
    for name, column in DECK_METADATA_COLUMNS.items():
        if name not in existing_columns:
            _add_column(connection, deck, column)
            added.add(name)
    if connection.dialect.name == "postgresql":
        topic = _column_map(connection, deck_name).get("topic")
        if topic is not None and not topic.get("nullable", True):
            preparer = connection.dialect.identifier_preparer
            connection.exec_driver_sql(
                f"ALTER TABLE {preparer.format_table(deck)} ALTER COLUMN "
                f"{preparer.quote('topic')} DROP NOT NULL"
            )
    if Flashcard.__tablename__ not in _tables(connection):
        Flashcard.__table__.create(connection)
    card_table = _table(connection, Flashcard.__tablename__)
    for index in Flashcard.__table__.indexes:
        index.create(connection, checkfirst=True)
    for index in FlashcardDeck.__table__.indexes:
        index.create(connection, checkfirst=True)
    _ensure_deck_checks(connection, deck)
    _backfill_cards(
        connection,
        title_was_added="title" in added,
        source_type_was_added="source_type" in added,
        total_cards_was_added="total_cards" in added,
    )
    validate_schema(connection)


def preflight(connection: Connection) -> list[dict[str, str | bool]]:
    tables = _tables(connection)
    deck_name = FlashcardDeck.__tablename__
    card_name = Flashcard.__tablename__
    if deck_name not in tables:
        if card_name in tables:
            return [{
                "passed": False,
                "label": "Normalized flashcard storage",
                "detail": "flashcards exists without its required flashcard_decks parent",
            }]
        return [{
            "passed": True,
            "label": "Normalized flashcard storage",
            "detail": "v005 will create flashcard_decks; v007 will add normalized flashcards without replacing existing tables",
        }]
    try:
        validate_schema(connection)
    except SchemaMismatchError as error:
        current_columns = set(_column_map(connection, deck_name))
        required_additions = set(DECK_METADATA_COLUMNS) - current_columns
        if required_additions and card_name not in tables:
            return [{
                "passed": True,
                "label": "Normalized flashcard storage",
                "detail": "v007 will add missing deck metadata and create the normalized card table; existing deck rows and JSON cards will be preserved",
            }]
        return [{
            "passed": False,
            "label": "Normalized flashcard storage",
            "detail": str(error),
        }]
    return [{
        "passed": True,
        "label": "Normalized flashcard storage",
        "detail": "flashcard deck metadata and normalized card storage are compatible",
    }]