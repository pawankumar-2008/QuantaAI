from sqlalchemy import Column, MetaData, Table, Text, inspect
from sqlalchemy.engine import Connection
from sqlalchemy.schema import CreateColumn

from backend.migrations.v003_fastapi_schema_reconciliation import (
    SchemaMismatchError,
    _schema,
)


VERSION = "008_planner_task_descriptions"
TABLE_NAME = "planner_tasks"
COLUMN_NAME = "description"


def validate_schema(connection: Connection) -> None:
    schema = _schema(connection)
    if TABLE_NAME not in set(inspect(connection).get_table_names(schema=schema)):
        raise SchemaMismatchError("Required planner_tasks table is missing.")

    columns = {
        column["name"]: column
        for column in inspect(connection).get_columns(TABLE_NAME, schema=schema)
    }
    description = columns.get(COLUMN_NAME)
    if (
        description is None
        or not isinstance(description["type"], Text)
        or not description.get("nullable", True)
    ):
        raise SchemaMismatchError(
            "planner_tasks.description must be a nullable text column."
        )


def upgrade(connection: Connection) -> None:
    schema = _schema(connection)
    tables = set(inspect(connection).get_table_names(schema=schema))
    if TABLE_NAME not in tables:
        raise SchemaMismatchError("Required planner_tasks table is missing; v005 must run first.")

    columns = {
        column["name"]
        for column in inspect(connection).get_columns(TABLE_NAME, schema=schema)
    }
    if COLUMN_NAME not in columns:
        table = Table(
            TABLE_NAME,
            MetaData(),
            autoload_with=connection,
            schema=schema,
        )
        column = Column(COLUMN_NAME, Text(), nullable=True)
        preparer = connection.dialect.identifier_preparer
        connection.exec_driver_sql(
            f"ALTER TABLE {preparer.format_table(table)} ADD COLUMN "
            f"{CreateColumn(column).compile(dialect=connection.dialect)}"
        )
    validate_schema(connection)


def preflight(connection: Connection) -> list[dict[str, str | bool]]:
    schema = _schema(connection)
    if TABLE_NAME not in set(inspect(connection).get_table_names(schema=schema)):
        return [
            {
                "passed": True,
                "label": "Planner task descriptions",
                "detail": (
                    "v005 will create planner_tasks with the nullable description "
                    "column without changing existing tables"
                ),
            }
        ]

    try:
        validate_schema(connection)
    except SchemaMismatchError as error:
        columns = {
            column["name"]
            for column in inspect(connection).get_columns(TABLE_NAME, schema=schema)
        }
        if COLUMN_NAME in columns:
            return [
                {
                    "passed": False,
                    "label": "Planner task descriptions",
                    "detail": str(error),
                }
            ]
        return [
            {
                "passed": True,
                "label": "Planner task descriptions",
                "detail": (
                    "v008 will add a nullable description column while preserving "
                    "existing planner tasks"
                ),
            }
        ]
    return [
        {
            "passed": True,
            "label": "Planner task descriptions",
            "detail": "planner_tasks.description is compatible",
        }
    ]
