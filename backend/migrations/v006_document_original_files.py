from sqlalchemy import Column, LargeBinary, MetaData, Table, inspect
from sqlalchemy.engine import Connection
from sqlalchemy.schema import CreateColumn

from backend.documents import StudyDocument
from backend.migrations.v003_fastapi_schema_reconciliation import (
    SchemaMismatchError,
    _schema,
)


VERSION = "006_document_original_files"


def validate_schema(connection: Connection) -> None:
    table = StudyDocument.__tablename__
    schema = _schema(connection)
    tables = set(inspect(connection).get_table_names(schema=schema))
    if table not in tables:
        raise SchemaMismatchError("Required study_documents table is missing.")

    columns = {
        column["name"]: column
        for column in inspect(connection).get_columns(table, schema=schema)
    }
    original_file = columns.get("original_file")
    if (
        original_file is None
        or not isinstance(original_file["type"], LargeBinary)
        or original_file.get("nullable", True) is False
    ):
        raise SchemaMismatchError(
            "study_documents.original_file must be a nullable binary column."
        )


def upgrade(connection: Connection) -> None:
    schema = _schema(connection)
    tables = set(inspect(connection).get_table_names(schema=schema))
    table_name = StudyDocument.__tablename__
    if table_name not in tables:
        raise SchemaMismatchError("Required study_documents table is missing.")

    columns = {
        column["name"]
        for column in inspect(connection).get_columns(table_name, schema=schema)
    }
    if "original_file" not in columns:
        table = Table(
            table_name,
            MetaData(),
            autoload_with=connection,
            schema=schema,
        )
        column = Column("original_file", LargeBinary(), nullable=True)
        preparer = connection.dialect.identifier_preparer
        connection.exec_driver_sql(
            f"ALTER TABLE {preparer.format_table(table)} ADD COLUMN "
            f"{CreateColumn(column).compile(dialect=connection.dialect)}"
        )
    validate_schema(connection)


def preflight(connection: Connection) -> list[dict[str, str | bool]]:
    schema = _schema(connection)
    table = StudyDocument.__tablename__
    tables = set(inspect(connection).get_table_names(schema=schema))
    if table not in tables:
        return [
            {
                "passed": True,
                "label": "Original study PDF storage",
                "detail": "study_documents will be created by v004 with a nullable original_file column",
            }
        ]

    try:
        validate_schema(connection)
    except SchemaMismatchError as error:
        columns = {
            column["name"]
            for column in inspect(connection).get_columns(table, schema=schema)
        }
        if "original_file" in columns:
            return [
                {
                    "passed": False,
                    "label": "Original study PDF storage",
                    "detail": str(error),
                }
            ]
        return [
            {
                "passed": True,
                "label": "Original study PDF storage",
                "detail": "v006 will add a nullable binary column without changing existing document rows",
            }
        ]
    return [
        {
            "passed": True,
            "label": "Original study PDF storage",
            "detail": "nullable original PDF storage is compatible",
        }
    ]