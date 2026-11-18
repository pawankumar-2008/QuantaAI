from sqlalchemy import DateTime, Integer, JSON, String, inspect
from sqlalchemy.engine import Connection

from backend.documents import StudyDocument
from backend.migrations.v003_fastapi_schema_reconciliation import (
    SchemaMismatchError,
    _schema,
)


VERSION = "004_study_documents"


def validate_schema(connection: Connection) -> None:
    table = StudyDocument.__tablename__
    tables = set(inspect(connection).get_table_names(schema=_schema(connection)))
    if table not in tables:
        raise SchemaMismatchError("Required study_documents table is missing.")

    columns = {
        column["name"]: column
        for column in inspect(connection).get_columns(
            table,
            schema=_schema(connection),
        )
    }
    expected_types = {
        "document_id": Integer,
        "user_id": Integer,
        "filename": String,
        "size_bytes": Integer,
        "page_count": Integer,
        "pages": JSON,
        "status": String,
        "status_message": String,
        "created_at": DateTime,
    }
    for name, expected_type in expected_types.items():
        column = columns.get(name)
        if column is None or not isinstance(column["type"], expected_type):
            raise SchemaMismatchError(
                f"study_documents.{name} has an incompatible data type."
            )
        if name != "status_message" and column.get("nullable", True):
            raise SchemaMismatchError(
                f"study_documents.{name} must be NOT NULL."
            )

    for name, minimum_length in (
        ("filename", 255),
        ("status", 20),
        ("status_message", 255),
    ):
        length = columns[name]["type"].length
        if length is not None and length < minimum_length:
            raise SchemaMismatchError(
                f"study_documents.{name} is too short."
            )

    primary_key = inspect(connection).get_pk_constraint(
        table,
        schema=_schema(connection),
    )
    if primary_key.get("constrained_columns") != ["document_id"]:
        raise SchemaMismatchError(
            "study_documents.document_id must be the primary key."
        )

    foreign_keys = inspect(connection).get_foreign_keys(
        table,
        schema=_schema(connection),
    )
    matching_foreign_keys = [
        key
        for key in foreign_keys
        if key.get("constrained_columns") == ["user_id"]
        and key.get("referred_table") == "users"
        and key.get("referred_columns") == ["user_id"]
        and str((key.get("options") or {}).get("ondelete", "")).upper()
        == "CASCADE"
    ]
    if not matching_foreign_keys:
        raise SchemaMismatchError(
            "study_documents.user_id requires a cascading users foreign key."
        )

    indexes = inspect(connection).get_indexes(
        table,
        schema=_schema(connection),
    )
    if not any(
        index.get("column_names") == ["user_id", "created_at"]
        and not index.get("unique")
        for index in indexes
    ):
        raise SchemaMismatchError(
            "The study document user/created index is missing."
        )

    checks = inspect(connection).get_check_constraints(
        table,
        schema=_schema(connection),
    )
    status_check = next(
        (
            constraint.get("sqltext", "").lower()
            for constraint in checks
            if constraint.get("name") == "ck_study_documents_status"
        ),
        "",
    )
    if not all(
        f"'{status}'" in status_check
        for status in ("processing", "indexed", "failed")
    ):
        raise SchemaMismatchError(
            "study_documents.status has an incompatible status constraint."
        )


def upgrade(connection: Connection) -> None:
    tables = set(inspect(connection).get_table_names(schema=_schema(connection)))
    if StudyDocument.__tablename__ not in tables:
        StudyDocument.__table__.create(connection)
    validate_schema(connection)


def preflight(connection: Connection) -> list[dict[str, str | bool]]:
    tables = set(inspect(connection).get_table_names(schema=_schema(connection)))
    if StudyDocument.__tablename__ not in tables:
        return [
            {
                "passed": True,
                "label": "Study document storage",
                "detail": "study_documents is absent; v004 will add the table without changing existing tables",
            }
        ]

    try:
        validate_schema(connection)
    except SchemaMismatchError as error:
        return [
            {
                "passed": False,
                "label": "Study document storage",
                "detail": str(error),
            }
        ]
    return [
        {
            "passed": True,
            "label": "Study document storage",
            "detail": "existing study_documents table is compatible",
        }
    ]
