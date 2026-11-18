import re

from sqlalchemy import Boolean, DateTime, Enum, Integer, JSON, String, Text, inspect, text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import SQLAlchemyError

from backend.auth import AuthSession, Base, User
from backend.conversations import Conversation, ConversationMessage


VERSION = "003_fastapi_schema_reconciliation"
CORE_TABLES = ("users", "conversations", "messages")
REQUIRED_VARCHAR_LENGTHS = {
    ("users", "email"): 320,
    ("users", "password_hash"): 255,
    ("users", "full_name"): 100,
    ("conversations", "conversation_type"): 32,
    ("conversations", "title"): 200,
    ("conversations", "status"): 32,
    ("messages", "sender_type"): 20,
    ("messages", "model_key"): 200,
}
REQUIRED_CHECK_VALUES = {
    ("conversations", "status"): {"active"},
    ("messages", "sender_type"): {"user", "assistant"},
}
CONVERSATION_TYPE_RULES = {
    ("surface", True),
    ("agent", False),
    ("chat", True),
}
LEGACY_CONVERSATION_TYPE_RULES = CONVERSATION_TYPE_RULES - {
    ("chat", True)
}
CONVERSATION_TYPE_CHECK_SQL = (
    "(conversation_type = 'surface' AND agent_id IS NULL) "
    "OR (conversation_type = 'agent' AND agent_id IS NOT NULL) "
    "OR (conversation_type = 'chat' AND agent_id IS NULL)"
)


class SchemaMismatchError(RuntimeError):
    pass


def _schema(connection: Connection) -> str | None:
    return "public" if connection.dialect.name == "postgresql" else None


def _column_map(connection: Connection, table: str) -> dict[str, dict]:
    return {
        column["name"]: column
        for column in inspect(connection).get_columns(table, schema=_schema(connection))
    }


def _is_type(column_type, expected: str) -> bool:
    if expected == "integer":
        return isinstance(column_type, Integer) and not isinstance(column_type, Boolean)
    if expected == "string":
        return isinstance(column_type, (String, Text, Enum))
    if expected == "boolean":
        return isinstance(column_type, Boolean)
    if expected == "datetime":
        return isinstance(column_type, DateTime)
    if expected == "text":
        return isinstance(column_type, Text) or (
            isinstance(column_type, String)
            and (column_type.length is None or column_type.length >= 20000)
        )
    if expected == "json":
        return isinstance(column_type, JSON)
    return False


def _require_column(
    connection: Connection,
    table: str,
    columns: dict[str, dict],
    name: str,
    expected_type: str,
    *,
    nullable: bool | None = None,
    allow_shorter: bool = False,
) -> None:
    column = columns.get(name)
    if column is None:
        raise SchemaMismatchError(f"Required column {table}.{name} is missing.")
    if not _is_type(column["type"], expected_type):
        raise SchemaMismatchError(
            f"Column {table}.{name} has an incompatible data type."
        )
    required_length = REQUIRED_VARCHAR_LENGTHS.get((table, name))
    actual_length = getattr(column["type"], "length", None)
    if (
        required_length
        and actual_length is not None
        and actual_length < required_length
        and connection.dialect.name != "sqlite"
        and not allow_shorter
    ):
        raise SchemaMismatchError(
            f"Column {table}.{name} is too short for the FastAPI contract."
        )
    if (
        expected_type == "datetime"
        and connection.dialect.name == "postgresql"
        and not getattr(column["type"], "timezone", False)
    ):
        raise SchemaMismatchError(
            f"Column {table}.{name} must preserve timezone-aware timestamps."
        )
    if nullable is False and column.get("nullable", True):
        raise SchemaMismatchError(
            f"Column {table}.{name} must be NOT NULL for FastAPI compatibility."
        )
    if nullable is True and not column.get("nullable", True):
        raise SchemaMismatchError(
            f"Column {table}.{name} must remain nullable for semantic compatibility."
        )


def _require_primary_key(
    connection: Connection,
    table: str,
    column_name: str,
    *,
    generated: bool = True,
) -> None:
    inspector = inspect(connection)
    primary_key = inspector.get_pk_constraint(table, schema=_schema(connection))
    if primary_key.get("constrained_columns") != [column_name]:
        raise SchemaMismatchError(
            f"Table {table} must use {column_name} as its sole primary key."
        )
    if not generated:
        return
    column = _column_map(connection, table)[column_name]
    default = str(column.get("default") or "").lower()
    identity = column.get("identity")
    sqlite_integer_key = (
        connection.dialect.name == "sqlite"
        and isinstance(column["type"], Integer)
        and primary_key.get("constrained_columns") == [column_name]
    )
    if not (identity or "nextval(" in default or sqlite_integer_key):
        raise SchemaMismatchError(
            f"Primary key {table}.{column_name} is not database-generated."
        )


def _strip_outer_parentheses(expression: str) -> str:
    expression = expression.strip()
    while expression.startswith("(") and expression.endswith(")"):
        depth = 0
        in_string = False
        closes_at_end = False
        index = 0
        while index < len(expression):
            character = expression[index]
            if character == "'":
                if in_string and index + 1 < len(expression) and expression[index + 1] == "'":
                    index += 2
                    continue
                in_string = not in_string
            elif not in_string:
                if character == "(":
                    depth += 1
                elif character == ")":
                    depth -= 1
                    if depth == 0:
                        closes_at_end = index == len(expression) - 1
                        break
            index += 1
        if not closes_at_end:
            break
        expression = expression[1:-1].strip()
    return expression


def _parse_sql_string_list(literals: str) -> set[str] | None:
    """Parse a comma-separated PostgreSQL string-literal list.

    PostgreSQL may rewrite ARRAY literals during ALTER COLUMN TYPE. For
    example, an original ARRAY['active'::varchar] can become
    ARRAY[('active'::character varying)::text]. Accept harmless casts and
    parentheses around each literal while remaining fail-closed for any
    non-literal expression.
    """
    values: set[str] = set()
    remaining = literals.strip()
    type_cast = (
        r"(?:::\s*(?:text|varchar(?:\s*\(\s*\d+\s*\))?|"
        r"character varying(?:\s*\(\s*\d+\s*\))?))"
    )
    literal_pattern = re.compile(
        r"^\(*\s*'((?:''|[^'])*)'"
        r"(?:(?:\s*" + type_cast + r")|(?:\s*\)))*\s*(?:,|$)",
        re.I,
    )
    while remaining:
        match = literal_pattern.match(remaining)
        if not match:
            return None
        values.add(match.group(1).replace("''", "'"))
        remaining = remaining[match.end():].strip()
    return values


def _parse_membership_check(definition: str, column: str) -> set[str] | None:
    expression = re.sub(r"^\s*CHECK\s*", "", definition, flags=re.I).strip()
    expression = _strip_outer_parentheses(expression)
    column_ref = rf'(?:(?:"[^"]+"|[A-Za-z_][\w$]*)\s*\.\s*)?"?{re.escape(column)}"?'
    checked_column = rf"\(?\s*{column_ref}\s*\)?"

    in_match = re.fullmatch(
        rf"{checked_column}\s*(?:::\s*(?:text|varchar|character varying))?\s+IN\s*\((.*)\)",
        expression,
        flags=re.I | re.S,
    )
    if in_match:
        return _parse_sql_string_list(in_match.group(1))

    any_match = re.fullmatch(
        rf"{checked_column}\s*(?:::\s*(?:text|varchar|character varying))?\s*=\s*ANY\s*\((.*)\)",
        expression,
        flags=re.I | re.S,
    )
    if any_match:
        array_expression = _strip_outer_parentheses(any_match.group(1))
        array_expression = re.sub(
            r"^\(\s*(ARRAY\s*\[.*\])\s*\)\s*::\s*(?:text|varchar|character varying)\[\]$",
            r"\1",
            array_expression,
            flags=re.I | re.S,
        )
        array_expression = re.sub(r"^ARRAY\s*", "", array_expression, flags=re.I)
        array_match = re.fullmatch(r"\[(.*)\](?:\s*::\s*(?:text|varchar|character varying)\[\])?", array_expression, re.I | re.S)
        if array_match:
            return _parse_sql_string_list(array_match.group(1))
    return None


def _split_boolean_expression(expression: str, operator: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    in_string = False
    start = 0
    index = 0
    upper = expression.upper()
    while index < len(expression):
        character = expression[index]
        if character == "'":
            if in_string and index + 1 < len(expression) and expression[index + 1] == "'":
                index += 2
                continue
            in_string = not in_string
        elif not in_string:
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
            elif (
                depth == 0
                and upper.startswith(operator, index)
                and (index == 0 or expression[index - 1].isspace())
                and (index + len(operator) == len(expression)
                     or expression[index + len(operator)].isspace())
            ):
                parts.append(expression[start:index].strip())
                index += len(operator)
                start = index
        index += 1
    parts.append(expression[start:].strip())
    return parts


def _parse_conversation_type_check(definition: str) -> set[tuple[str, str]] | None:
    expression = re.sub(r"^\s*CHECK\s*", "", definition, flags=re.I).strip()
    rules: set[tuple[str, str]] = set()
    branches = _split_boolean_expression(_strip_outer_parentheses(expression), "OR")
    for branch in branches:
        terms = _split_boolean_expression(_strip_outer_parentheses(branch), "AND")
        normalized = []
        for term in terms:
            value = re.sub(r"::\s*(?:text|character varying|varchar)(?:\s*\(\s*\d+\s*\))?", "", term, flags=re.I)
            value = re.sub(r"[\s\"()]", "", value).lower()
            value = re.sub(r"^(?:public\.)", "", value)
            normalized.append(value)
        type_terms = [
            term for term in normalized
            if re.fullmatch(r"(?:[a-z_][\w$]*\.)?conversation_type='(?:surface|agent|chat)'", term)
        ]
        agent_terms = [
            term for term in normalized
            if term in {"agent_idisnull", "agent_idisnotnull"}
        ]
        if len(normalized) != 2 or len(type_terms) != 1 or len(agent_terms) != 1:
            return None
        conversation_type = type_terms[0].split("='")[1][:-1]
        rules.add((conversation_type, agent_terms[0] == "agent_idisnull"))
    if len(rules) != len(branches):
        return None
    return rules


def _conversation_type_rule_detail(
    connection: Connection,
) -> tuple[str, str, set[tuple[str, str]]]:
    """Validate conversation type rules while supporting legacy SQLite fixtures.

    Production PostgreSQL schemas must have both the semantic type/agent_id
    CHECK and the allowed-values CHECK. SQLite test fixtures historically
    model the semantic CHECK alone, so that shape is accepted there.
    """
    definitions = _check_definitions(connection, "conversations", "conversation_type")
    semantic = []
    membership = []
    unsupported = []

    for name, definition in definitions:
        rules = _parse_conversation_type_check(definition)
        values = _parse_membership_check(definition, "conversation_type")
        if rules is not None:
            semantic.append((name, definition, rules))
        elif values is not None:
            membership.append((name, definition, values))
        else:
            unsupported.append(name)

    if unsupported:
        raise SchemaMismatchError(
            "Unsupported conversations.conversation_type CHECK constraint(s): "
            + ", ".join(unsupported)
            + ". Refusing to alter unknown constraints."
        )
    if len(semantic) != 1:
        raise SchemaMismatchError(
            "conversations.conversation_type CHECK does not preserve the supported type/agent_id rules."
        )
    if len(membership) > 1:
        raise SchemaMismatchError(
            "conversations.conversation_type must have at most one allowed-values CHECK constraint."
        )
    if connection.dialect.name == "postgresql" and len(membership) != 1:
        raise SchemaMismatchError(
            "PostgreSQL conversations.conversation_type must have one allowed-values CHECK constraint."
        )

    semantic_name, semantic_definition, rules = semantic[0]
    if rules not in (LEGACY_CONVERSATION_TYPE_RULES, CONVERSATION_TYPE_RULES):
        raise SchemaMismatchError(
            f"CHECK constraint conversations.{semantic_name} does not preserve the supported type/agent_id rules."
        )

    if membership:
        membership_name, _, allowed_values = membership[0]
        if allowed_values not in ({"surface", "agent"}, {"surface", "agent", "chat"}):
            raise SchemaMismatchError(
                f"CHECK constraint conversations.{membership_name} has unsupported allowed values: "
                f"{sorted(allowed_values)}."
            )
    return semantic_name, semantic_definition, rules

def _invalid_conversation_type_count(connection: Connection) -> int:
    schema = _schema(connection)
    prefix = f"{_quoted(connection, schema)}." if schema else ""
    return int(connection.execute(text(
        "SELECT count(*) FROM "
        f"{prefix}{_quoted(connection, 'conversations')} "
        "WHERE NOT ((conversation_type = 'surface' AND agent_id IS NULL) "
        "OR (conversation_type = 'agent' AND agent_id IS NOT NULL) "
        "OR (conversation_type = 'chat' AND agent_id IS NULL))"
    )).scalar_one())


def _reconcile_conversation_type_check(connection: Connection) -> None:
    semantic_name, _, rules = _conversation_type_rule_detail(connection)
    definitions = _check_definitions(connection, "conversations", "conversation_type")
    membership_constraints = [
        (name, definition, values)
        for name, definition in definitions
        if (values := _parse_membership_check(definition, "conversation_type")) is not None
    ]
    membership_name = membership_constraints[0][0] if membership_constraints else None
    membership_values = membership_constraints[0][2] if membership_constraints else None

    invalid_count = _invalid_conversation_type_count(connection)
    if invalid_count:
        raise SchemaMismatchError(
            f"{invalid_count} conversations violate the planned conversation_type/agent_id rules."
        )

    semantic_needs_update = rules != CONVERSATION_TYPE_RULES
    membership_needs_update = (
        membership_values is not None and membership_values != {"surface", "agent", "chat"}
    )
    if not semantic_needs_update and not membership_needs_update:
        return
    if connection.dialect.name != "postgresql":
        raise SchemaMismatchError(
            "SQLite cannot widen the existing conversation_type CHECK constraints in place."
        )

    schema = _schema(connection)
    prefix = f"{_quoted(connection, schema)}." if schema else ""
    table_name = f"{prefix}{_quoted(connection, 'conversations')}"

    if semantic_needs_update:
        connection.execute(text(
            f"ALTER TABLE {table_name} DROP CONSTRAINT {_quoted(connection, semantic_name)}"
        ))
        connection.execute(text(
            f"ALTER TABLE {table_name} ADD CONSTRAINT {_quoted(connection, semantic_name)} "
            f"CHECK ({CONVERSATION_TYPE_CHECK_SQL})"
        ))

    if membership_needs_update and membership_name is not None:
        final_values = membership_values | {"chat"} | _distinct_values(
            connection, "conversations", "conversation_type"
        )
        quoted_values = ", ".join(
            "'" + value.replace("'", "''") + "'" for value in sorted(final_values)
        )
        connection.execute(text(
            f"ALTER TABLE {table_name} DROP CONSTRAINT {_quoted(connection, membership_name)}"
        ))
        connection.execute(text(
            f"ALTER TABLE {table_name} ADD CONSTRAINT {_quoted(connection, membership_name)} "
            f"CHECK ({_quoted(connection, 'conversation_type')} IN ({quoted_values}))"
        ))


def _conversation_type_check_preflight_detail(connection: Connection) -> str:
    name, definition, rules = _conversation_type_rule_detail(connection)
    definitions = _check_definitions(connection, "conversations", "conversation_type")
    membership = [
        (constraint_name, constraint_definition, values)
        for constraint_name, constraint_definition in definitions
        if (values := _parse_membership_check(constraint_definition, "conversation_type")) is not None
    ]
    membership_name, membership_definition, membership_values = (
        membership[0] if membership else (None, None, None)
    )

    needs_update = rules == LEGACY_CONVERSATION_TYPE_RULES or (
        membership_values is not None and membership_values != {"surface", "agent", "chat"}
    )
    if needs_update and connection.dialect.name != "postgresql":
        raise SchemaMismatchError(
            "SQLite cannot widen the existing conversation_type CHECK constraints in place."
        )

    invalid_count = _invalid_conversation_type_count(connection)
    if invalid_count:
        raise SchemaMismatchError(
            f"{invalid_count} conversations violate the planned type/agent_id rules."
        )

    schema = _schema(connection)
    prefix = f"{_quoted(connection, schema)}." if schema else ""
    combinations = connection.execute(text(
        "SELECT DISTINCT conversation_type, agent_id IS NULL "
        f"FROM {prefix}{_quoted(connection, 'conversations')} "
        "ORDER BY conversation_type, agent_id IS NULL"
    )).all()
    semantic_status = (
        "will add chat with NULL agent_id"
        if rules == LEGACY_CONVERSATION_TYPE_RULES
        else "already includes chat with NULL agent_id"
    )
    membership_status = (
        "membership CHECK absent in SQLite fixture"
        if membership_values is None
        else (
            "will add chat to allowed values"
            if membership_values != {"surface", "agent", "chat"}
            else "already allows surface, agent, and chat"
        )
    )
    membership_description = (
        "no separate membership constraint"
        if membership_name is None
        else f"membership {membership_name}={membership_definition}"
    )
    return (
        f"semantic {name}={definition}; {membership_description}; "
        f"distinct type/agent_id-null combinations: "
        f"{[(value, is_null) for value, is_null in combinations]}; "
        f"{semantic_status}; {membership_status}; invalid rows: 0"
    )


def _check_definitions(connection: Connection, table: str, column: str) -> list[tuple[str, str]]:
    if connection.dialect.name == "postgresql":
        return [
            (name, definition)
            for name, definition in connection.execute(
                text(
                    "SELECT constraint_row.conname, "
                    "pg_get_constraintdef(constraint_row.oid) "
                    "FROM pg_constraint AS constraint_row "
                    "JOIN pg_class AS relation ON relation.oid = constraint_row.conrelid "
                    "JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace "
                    "JOIN LATERAL unnest(constraint_row.conkey) AS key_column(attnum) ON TRUE "
                    "JOIN pg_attribute AS attribute "
                    "ON attribute.attrelid = relation.oid AND attribute.attnum = key_column.attnum "
                    "WHERE constraint_row.contype = 'c' AND namespace.nspname = :schema "
                    "AND relation.relname = :table AND attribute.attname = :column"
                ),
                {"schema": _schema(connection), "table": table, "column": column},
            ).all()
        ]
    if connection.dialect.name == "sqlite":
        return [
            (constraint.get("name") or "<unnamed>", str(constraint.get("sqltext") or ""))
            for constraint in inspect(connection).get_check_constraints(table)
            if re.search(rf"\b{re.escape(column)}\b", str(constraint.get("sqltext") or ""), re.I)
        ]
    raise SchemaMismatchError("CHECK constraint inspection supports PostgreSQL and SQLite only.")


def _distinct_values(connection: Connection, table: str, column: str) -> set[str]:
    schema = _schema(connection)
    prefix = f"{_quoted(connection, schema)}." if schema else ""
    values = connection.execute(
        text(
            f"SELECT DISTINCT {_quoted(connection, column)} "
            f"FROM {prefix}{_quoted(connection, table)} "
            f"WHERE {_quoted(connection, column)} IS NOT NULL"
        )
    ).scalars().all()
    return {str(value).lower() for value in values}


def _allowed_check_values(connection: Connection, table: str, column: str) -> set[str] | None:
    enum_values = getattr(_column_map(connection, table)[column]["type"], "enums", None)
    if enum_values:
        return set(enum_values)
    definitions = _check_definitions(connection, table, column)
    if not definitions:
        return None
    allowed_sets: list[set[str]] = []
    for name, definition in definitions:
        values = _parse_membership_check(definition, column)
        if values is None:
            raise SchemaMismatchError(
                f"CHECK constraint {table}.{name} has an unsupported expression; refusing to alter it."
            )
        allowed_sets.append(values)
    return set.intersection(*allowed_sets) if allowed_sets else set()


def _reconcile_value_checks(
    connection: Connection,
    table: str,
    column: str,
    required_values: set[str],
) -> None:
    definitions = _check_definitions(connection, table, column)
    if not definitions:
        return
    allowed_sets: list[set[str]] = []
    for name, definition in definitions:
        values = _parse_membership_check(definition, column)
        if values is None:
            raise SchemaMismatchError(
                f"CHECK constraint {table}.{name} has an unsupported expression; refusing to alter it."
            )
        allowed_sets.append(values)
    effective_values = set.intersection(*allowed_sets) if allowed_sets else set()
    preserved_values = effective_values | _distinct_values(connection, table, column)
    final_values = preserved_values | required_values
    if connection.dialect.name != "postgresql":
        if not required_values.issubset(effective_values):
            raise SchemaMismatchError(
                f"SQLite cannot reconcile existing CHECK constraints on {table}.{column} in place."
            )
        return

    schema = _schema(connection)
    prefix = f"{_quoted(connection, schema)}." if schema else ""
    table_name = f"{prefix}{_quoted(connection, table)}"
    surviving_name = definitions[0][0]
    for name, _ in definitions:
        connection.execute(
            text(
                f"ALTER TABLE {table_name} DROP CONSTRAINT {_quoted(connection, name)}"
            )
        )
    quoted_values = ", ".join(
        "'" + value.replace("'", "''") + "'" for value in sorted(final_values)
    )
    connection.execute(
        text(
            f"ALTER TABLE {table_name} ADD CONSTRAINT {_quoted(connection, surviving_name)} "
            f"CHECK ({_quoted(connection, column)} IN ({quoted_values}))"
        )
    )


def _check_reconciliation_detail(
    connection: Connection,
    table: str,
    column: str,
    required_values: set[str],
) -> str:
    definitions = _check_definitions(connection, table, column)
    distinct_values = _distinct_values(connection, table, column)
    parsed: list[tuple[str, str, set[str]]] = []
    for name, definition in definitions:
        values = _parse_membership_check(definition, column)
        if values is None:
            raise SchemaMismatchError(
                f"CHECK constraint {table}.{name} cannot be safely reconciled: {definition}"
            )
        parsed.append((name, definition, values))

    effective_values = (
        set.intersection(*(values for _, _, values in parsed)) if parsed else set()
    )
    if (
        connection.dialect.name != "postgresql"
        and parsed
        and not required_values.issubset(effective_values)
    ):
        raise SchemaMismatchError(
            f"SQLite cannot widen CHECK constraints on {table}.{column} in place."
        )

    if not definitions:
        constraint_summary = "no CHECK constraint"
    else:
        constraint_summary = "; ".join(
            f"{name}={definition}" for name, definition, _ in parsed
        )
    return (
        f"constraints: {constraint_summary}; distinct values: "
        f"{sorted(distinct_values)}; required values: {sorted(required_values)}; "
        "v003 preserves current values and adds required values"
    )


def _validate_values(connection: Connection, table: str, column: str, accepted: set[str]) -> None:
    allowed = _allowed_check_values(connection, table, column)
    if allowed is not None and not accepted.issubset(allowed):
        raise SchemaMismatchError(
            f"Existing constraints on {table}.{column} reject required FastAPI values."
        )


def _validate_extra_columns(
    connection: Connection, table: str, mapped_columns: set[str]
) -> None:
    for column in _column_map(connection, table).values():
        if column["name"] in mapped_columns:
            continue
        if column.get("nullable", True) or column.get("default") is not None:
            continue
        if column.get("identity"):
            continue
        raise SchemaMismatchError(
            f"Unmapped required column {table}.{column['name']} has no server default."
        )


def _varchar_width_detail(connection: Connection, table: str, column_name: str) -> str:
    required = REQUIRED_VARCHAR_LENGTHS.get((table, column_name))
    if required is None:
        return "no bounded width requirement"
    column = _column_map(connection, table).get(column_name)
    if column is None:
        raise SchemaMismatchError(f"Required column {table}.{column_name} is missing.")
    column_type = column["type"]
    actual = getattr(column_type, "length", None)
    if actual is None:
        if isinstance(column_type, Text):
            return f"unbounded text supports required length {required}"
        raise SchemaMismatchError(f"{table}.{column_name} has no safe string capacity.")
    if actual >= required:
        return f"varchar({actual}) satisfies required capacity {required}"
    if connection.dialect.name == "sqlite":
        return f"PASS: SQLite does not enforce varchar({actual}); application validates capacity {required}"
    return f"varchar({actual}) is short; v003 widens to varchar({required}) without truncation"


def _widen_varchar_columns(connection: Connection) -> None:
    if connection.dialect.name == "sqlite":
        return
    if connection.dialect.name != "postgresql":
        raise SchemaMismatchError("VARCHAR widening supports PostgreSQL and SQLite only.")
    schema = _schema(connection)
    prefix = f"{_quoted(connection, schema)}." if schema else ""
    for (table, column_name), required in REQUIRED_VARCHAR_LENGTHS.items():
        columns = _column_map(connection, table)
        if column_name not in columns:
            continue
        column_type = columns[column_name]["type"]
        actual = getattr(column_type, "length", None)
        if actual is None or actual >= required:
            continue
        connection.execute(
            text(
                f"ALTER TABLE {prefix}{_quoted(connection, table)} "
                f"ALTER COLUMN {_quoted(connection, column_name)} "
                f"TYPE VARCHAR({required})"
            )
        )


def _null_title_count(connection: Connection) -> int:
    schema = _schema(connection)
    prefix = f"{_quoted(connection, schema)}." if schema else ""
    return int(
        connection.execute(
            text(
                f"SELECT count(*) FROM {prefix}{_quoted(connection, 'conversations')} "
                f"WHERE {_quoted(connection, 'title')} IS NULL"
            )
        ).scalar_one()
    )


def _null_profile_count(connection: Connection) -> int:
    schema = _schema(connection)
    prefix = f"{_quoted(connection, schema)}." if schema else ""
    return int(
        connection.execute(
            text(
                f"SELECT count(*) FROM {prefix}{_quoted(connection, 'users')} "
                f"WHERE {_quoted(connection, 'profile')} IS NULL"
            )
        ).scalar_one()
    )


def _ensure_title_not_null(connection: Connection) -> None:
    column = _column_map(connection, "conversations")["title"]
    if not column.get("nullable", True):
        return
    null_count = _null_title_count(connection)
    if null_count:
        raise SchemaMismatchError(
            f"{null_count} conversations have NULL titles; refusing to rewrite existing records."
        )
    if connection.dialect.name != "postgresql":
        raise SchemaMismatchError(
            "SQLite cannot tighten the existing title nullability in place."
        )
    schema = _schema(connection)
    prefix = f"{_quoted(connection, schema)}." if schema else ""
    connection.execute(
        text(
            f"ALTER TABLE {prefix}{_quoted(connection, 'conversations')} "
            f"ALTER COLUMN {_quoted(connection, 'title')} SET NOT NULL"
        )
    )


def _validate_columns(
    connection: Connection,
    *,
    allow_reconciliation: bool = False,
) -> None:
    expected = {
        "users": {
            "user_id": ("integer", False),
            "email": ("string", False),
            "password_hash": ("string", False),
            "full_name": ("string", False),
            "is_active": ("boolean", False),
            "created_at": ("datetime", False),
            "updated_at": ("datetime", False),
        },
        "conversations": {
            "conversation_id": ("integer", False),
            "user_id": ("integer", False),
            "conversation_type": ("string", False),
            "agent_id": ("integer", True),
            "title": ("string", False),
            "status": ("string", False),
            "created_at": ("datetime", False),
            "updated_at": ("datetime", False),
        },
        "messages": {
            "message_id": ("integer", False),
            "conversation_id": ("integer", False),
            "sender_type": ("string", False),
            "content": ("text", False),
            "model_key": ("string", True),
            "metadata": ("json", False),
            "created_at": ("datetime", False),
        },
    }
    mapped = {
        "users": set(expected["users"]) | {"profile"},
        "conversations": set(expected["conversations"]),
        "messages": set(expected["messages"]),
    }
    for table, table_columns in expected.items():
        columns = _column_map(connection, table)
        for name, (column_type, nullable) in table_columns.items():
            _require_column(
                connection,
                table,
                columns,
                name,
                column_type,
                nullable=(
                    None
                    if (
                        (allow_reconciliation and table == "conversations" and name == "title")
                        or (connection.dialect.name == "sqlite" and table == "messages" and name == "metadata")
                    )
                    else nullable
                ),
                allow_shorter=allow_reconciliation,
            )
        _validate_extra_columns(connection, table, mapped[table])

    _require_primary_key(connection, "users", "user_id")
    _require_primary_key(connection, "conversations", "conversation_id")
    _require_primary_key(connection, "messages", "message_id")
    _conversation_type_rule_detail(connection)
    if allow_reconciliation:
        for (table, column), _ in REQUIRED_CHECK_VALUES.items():
            for name, definition in _check_definitions(connection, table, column):
                if _parse_membership_check(definition, column) is None:
                    raise SchemaMismatchError(
                        f"CHECK constraint {table}.{name} has an unsupported expression."
                    )
    else:
        for (table, column), values in REQUIRED_CHECK_VALUES.items():
            _validate_values(connection, table, column, values)


def _quoted(connection: Connection, identifier: str) -> str:
    return connection.dialect.identifier_preparer.quote(identifier)


def _orphan_count(
    connection: Connection,
    child_table: str,
    child_column: str,
    parent_table: str,
    parent_column: str,
) -> int:
    schema = _schema(connection)
    prefix = f"{_quoted(connection, schema)}." if schema else ""
    result = connection.execute(
        text(
            f"SELECT count(*) FROM {prefix}{_quoted(connection, child_table)} child "
            f"LEFT JOIN {prefix}{_quoted(connection, parent_table)} parent "
            f"ON child.{_quoted(connection, child_column)} = "
            f"parent.{_quoted(connection, parent_column)} "
            f"WHERE child.{_quoted(connection, child_column)} IS NOT NULL "
            f"AND parent.{_quoted(connection, parent_column)} IS NULL"
        )
    )
    return int(result.scalar_one())


def _ensure_foreign_key(
    connection: Connection,
    table: str,
    column: str,
    referred_table: str,
    referred_column: str,
    constraint_name: str,
) -> None:
    inspector = inspect(connection)
    foreign_keys = inspector.get_foreign_keys(table, schema=_schema(connection))
    same_source = [
        item
        for item in foreign_keys
        if column in (item.get("constrained_columns") or [])
    ]
    compatible = [
        item
        for item in same_source
        if item.get("referred_table") == referred_table
        and item.get("referred_columns") == [referred_column]
    ]
    if compatible:
        options = compatible[0].get("options") or {}
        if str(options.get("ondelete", "")).upper() != "CASCADE":
            raise SchemaMismatchError(
                f"Foreign key {table}.{column} must cascade deletes to {referred_table}."
            )
        return
    if same_source:
        raise SchemaMismatchError(
            f"Column {table}.{column} has an incompatible existing foreign key."
        )
    if _orphan_count(connection, table, column, referred_table, referred_column):
        raise SchemaMismatchError(
            f"Orphaned {table}.{column} values prevent adding the required foreign key."
        )
    if connection.dialect.name != "postgresql":
        raise SchemaMismatchError(
            f"SQLite cannot add the missing foreign key on {table}.{column} in place."
        )

    schema = _schema(connection)
    prefix = f"{_quoted(connection, schema)}." if schema else ""
    connection.execute(
        text(
            f"ALTER TABLE {prefix}{_quoted(connection, table)} "
            f"ADD CONSTRAINT {_quoted(connection, constraint_name)} "
            f"FOREIGN KEY ({_quoted(connection, column)}) "
            f"REFERENCES {prefix}{_quoted(connection, referred_table)} "
            f"({ _quoted(connection, referred_column) }) ON DELETE CASCADE"
        )
    )


def _lower_email_index_exists(connection: Connection) -> bool:
    schema = _schema(connection)
    if connection.dialect.name == "postgresql":
        rows = connection.execute(
            text(
                "SELECT index_class.relname, pg_get_indexdef(index_class.oid), "
                "idx.indisunique, idx.indisvalid, idx.indpred IS NULL, "
                "idx.indnatts, pg_get_expr(idx.indexprs, idx.indrelid) "
                "FROM pg_index AS idx "
                "JOIN pg_class AS index_class ON index_class.oid = idx.indexrelid "
                "JOIN pg_class AS table_class ON table_class.oid = idx.indrelid "
                "JOIN pg_namespace AS namespace ON namespace.oid = table_class.relnamespace "
                "WHERE namespace.nspname = :schema AND table_class.relname = 'users'"
            ),
            {"schema": schema},
        ).all()
        for name, definition, unique, valid, non_partial, column_count, expression in rows:
            if name == "uq_users_email_lower" and not (
                unique and valid and non_partial and column_count == 1
                and expression and "lower" in expression.lower()
                and "email" in expression.lower()
            ):
                raise SchemaMismatchError(
                    "An incompatible uq_users_email_lower index already exists."
                )
            if (
                unique and valid and non_partial and column_count == 1
                and expression and "lower" in expression.lower()
                and "email" in expression.lower()
            ):
                return True
        return False
    elif connection.dialect.name == "sqlite":
        rows = connection.execute(
            text("SELECT name, sql FROM sqlite_master WHERE type = 'index' AND tbl_name = 'users'")
        ).all()
        definitions = [(name, definition or "") for name, definition in rows]
    else:
        raise SchemaMismatchError("The reconciliation migration supports PostgreSQL and SQLite only.")

    for name, definition in definitions:
        normalized = re.sub(r'["\s]', "", definition).lower()
        if name == "uq_users_email_lower" and not (
            "createuniqueindex" in normalized
            and "lower(email)" in normalized
            and " where " not in definition.lower()
        ):
            raise SchemaMismatchError(
                "An incompatible uq_users_email_lower index already exists."
            )
        if (
            "createuniqueindex" in normalized
            and "lower(email)" in normalized
            and " where " not in definition.lower()
        ):
            return True
    return False


def _duplicate_email_groups(connection: Connection) -> int:
    schema = _schema(connection)
    prefix = f"{_quoted(connection, schema)}." if schema else ""
    return int(
        connection.execute(
            text(
                "SELECT count(*) FROM ("
                f"SELECT lower(email) FROM {prefix}{_quoted(connection, 'users')} "
                f"WHERE {_quoted(connection, 'email')} IS NOT NULL "
                f"GROUP BY lower({_quoted(connection, 'email')}) "
                f"HAVING count(*) > 1"
                ") duplicates"
            )
        ).scalar_one()
    )


def _ensure_indexes(connection: Connection) -> None:
    if not _lower_email_index_exists(connection):
        if _duplicate_email_groups(connection):
            raise SchemaMismatchError(
                "Case-insensitive duplicate emails prevent the required unique index."
            )
        for index in User.__table__.indexes:
            if index.name == "uq_users_email_lower":
                index.create(connection, checkfirst=True)
                break

    for model, index_name, expected_columns in (
        (AuthSession, "ix_auth_sessions_user_id", ["user_id"]),
        (Conversation, "ix_conversations_user_updated", ["user_id", "updated_at"]),
        (ConversationMessage, "ix_messages_conversation_created", ["conversation_id", "message_id"]),
    ):
        table = model.__tablename__
        indexes = inspect(connection).get_indexes(table, schema=_schema(connection))
        named = next((index for index in indexes if index.get("name") == index_name), None)
        if named and (
            named.get("column_names") != expected_columns or named.get("unique")
        ):
            raise SchemaMismatchError(f"Index {index_name} has an incompatible definition.")
        equivalent = any(
            index.get("column_names") == expected_columns and not index.get("unique")
            for index in indexes
        )
        if not equivalent:
            next(index for index in model.__table__.indexes if index.name == index_name).create(
                connection, checkfirst=True
            )


def validate_schema(connection: Connection) -> None:
    tables = set(inspect(connection).get_table_names(schema=_schema(connection)))
    missing = (set(CORE_TABLES) | {"auth_sessions"}) - tables
    if missing:
        raise SchemaMismatchError(
            "Required FastAPI tables are missing: " + ", ".join(sorted(missing))
        )
    if "conversation_messages" in tables:
        raise SchemaMismatchError(
            "conversation_messages exists alongside the Neon messages mapping; refusing duplicate history storage."
        )

    _validate_columns(connection)
    _conversation_type_rule_detail(connection)
    if _invalid_conversation_type_count(connection):
        raise SchemaMismatchError(
            "Existing conversations violate the conversation_type/agent_id relationship."
        )
    user_columns = _column_map(connection, "users")
    _require_column(
        connection,
        "users",
        user_columns,
        "profile",
        "json",
        nullable=False,
    )
    _require_primary_key(connection, "auth_sessions", "token_hash", generated=False)
    session_columns = _column_map(connection, "auth_sessions")
    for column, kind in (
        ("token_hash", "string"),
        ("user_id", "integer"),
        ("expires_at", "datetime"),
        ("created_at", "datetime"),
    ):
        _require_column(
            connection,
            "auth_sessions",
            session_columns,
            column,
            kind,
            nullable=False,
        )
    _validate_extra_columns(
        connection,
        "auth_sessions",
        {"token_hash", "user_id", "expires_at", "created_at"},
    )

    for table, column, parent, parent_column in (
        ("conversations", "user_id", "users", "user_id"),
        ("messages", "conversation_id", "conversations", "conversation_id"),
        ("auth_sessions", "user_id", "users", "user_id"),
    ):
        foreign_keys = inspect(connection).get_foreign_keys(table, schema=_schema(connection))
        matching = [
            key
            for key in foreign_keys
            if key.get("constrained_columns") == [column]
            and key.get("referred_table") == parent
            and key.get("referred_columns") == [parent_column]
        ]
        if not matching or str((matching[0].get("options") or {}).get("ondelete", "")).upper() != "CASCADE":
            raise SchemaMismatchError(
                f"Required cascading foreign key {table}.{column} -> {parent}.{parent_column} is missing."
            )

    _ensure_indexes(connection)


def upgrade(connection: Connection) -> None:
    """Reconcile only compatible FastAPI structures; never drop or rewrite data."""
    schema = _schema(connection)
    tables = set(inspect(connection).get_table_names(schema=schema))
    if "conversation_messages" in tables:
        raise SchemaMismatchError(
            "conversation_messages exists; refusing to create or duplicate message history."
        )

    existing_core = set(CORE_TABLES).intersection(tables)
    if existing_core and existing_core != set(CORE_TABLES):
        raise SchemaMismatchError(
            "Only part of the FastAPI core schema exists; refusing automatic reconciliation."
        )
    if existing_core:
        _validate_columns(connection, allow_reconciliation=True)

    if "users" in tables:
        user_columns = _column_map(connection, "users")
        required_user_columns = {
            "user_id": "integer",
            "email": "string",
            "password_hash": "string",
            "full_name": "string",
            "is_active": "boolean",
            "created_at": "datetime",
            "updated_at": "datetime",
        }
        for name, kind in required_user_columns.items():
            _require_column(
                connection,
                "users",
                user_columns,
                name,
                kind,
                nullable=False,
                allow_shorter=True,
            )
        _require_primary_key(connection, "users", "user_id")
        if "profile" in user_columns:
            _require_column(connection, "users", user_columns, "profile", "json")
            if user_columns["profile"].get("nullable", True) and _null_profile_count(connection):
                raise SchemaMismatchError(
                    "Existing NULL users.profile values prevent safe NOT NULL enforcement."
                )
        if _duplicate_email_groups(connection) and not _lower_email_index_exists(connection):
            raise SchemaMismatchError(
                "Case-insensitive duplicate emails prevent the required unique index."
            )

    if "messages" in tables and "conversations" not in tables:
        raise SchemaMismatchError("messages exists but its parent conversations table is missing.")
    if "conversations" in tables and "users" not in tables:
        raise SchemaMismatchError("conversations exists but its parent users table is missing.")

    if "conversations" in tables:
        title_column = _column_map(connection, "conversations")["title"]
        if title_column.get("nullable", True) and _null_title_count(connection):
            raise SchemaMismatchError(
                "Existing NULL conversation titles must be resolved before tightening nullability."
            )
    if "users" in tables and "profile" in _column_map(connection, "users"):
        profile_column = _column_map(connection, "users")["profile"]
        if profile_column.get("nullable", True) and _null_profile_count(connection):
            raise SchemaMismatchError(
                "Existing NULL users.profile values prevent safe NOT NULL enforcement."
            )

    for (table, column), required_values in REQUIRED_CHECK_VALUES.items():
        if table in tables:
            _allowed_check_values(connection, table, column)
    if "conversations" in tables:
        _conversation_type_rule_detail(connection)

    _widen_varchar_columns(connection)
    for (table, column), required_values in REQUIRED_CHECK_VALUES.items():
        if table in tables:
            _reconcile_value_checks(connection, table, column, required_values)
    if "conversations" in tables:
        _reconcile_conversation_type_check(connection)
    if "conversations" in tables:
        _ensure_title_not_null(connection)

    if "users" in tables and "profile" not in _column_map(connection, "users"):
        if connection.dialect.name == "postgresql":
            connection.execute(
                text("ALTER TABLE public.users ADD COLUMN profile JSONB NOT NULL DEFAULT '{}'::jsonb")
            )
        elif connection.dialect.name == "sqlite":
            connection.execute(
                text("ALTER TABLE users ADD COLUMN profile JSON NOT NULL DEFAULT '{}'" )
            )
        else:
            raise SchemaMismatchError("The reconciliation migration supports PostgreSQL and SQLite only.")
    elif "users" in tables:
        profile_column = _column_map(connection, "users")["profile"]
        if profile_column.get("nullable", True):
            if connection.dialect.name != "postgresql":
                raise SchemaMismatchError(
                    "SQLite cannot tighten users.profile nullability in place."
                )
            connection.execute(
                text("ALTER TABLE public.users ALTER COLUMN profile SET NOT NULL")
            )

    Base.metadata.create_all(connection, checkfirst=True)

    for table, column, parent, parent_column, name in (
        ("conversations", "user_id", "users", "user_id", "fk_conversations_user_id_users"),
        ("messages", "conversation_id", "conversations", "conversation_id", "fk_messages_conversation_id_conversations"),
    ):
        _ensure_foreign_key(connection, table, column, parent, parent_column, name)

    if "auth_sessions" in set(inspect(connection).get_table_names(schema=schema)):
        _ensure_foreign_key(
            connection,
            "auth_sessions",
            "user_id",
            "users",
            "user_id",
            "fk_auth_sessions_user_id_users",
        )

    _ensure_indexes(connection)
    validate_schema(connection)


def preflight(connection: Connection) -> list[dict[str, str | bool]]:
    """Inspect schema and data without executing DDL or DML."""
    reports: list[dict[str, str | bool]] = []

    def report(label: str, action, *, success_detail: str = "compatible") -> None:
        try:
            detail = action()
            reports.append({"passed": True, "label": label, "detail": detail or success_detail})
        except Exception as error:
            detail = (
                error.args[0]
                if isinstance(error, SchemaMismatchError) and error.args
                else f"check failed ({type(error).__name__}; details redacted)"
            )
            reports.append({"passed": False, "label": label, "detail": str(detail)})

    def check_foreign_key(table: str, column: str, parent: str, parent_column: str) -> str:
        if table not in tables:
            return "will be created with table"
        if parent not in tables:
            raise SchemaMismatchError(f"Referenced table {parent} is missing.")
        if column not in _column_map(connection, table):
            raise SchemaMismatchError(f"Required column {table}.{column} is missing.")
        foreign_keys = inspect(connection).get_foreign_keys(table, schema=schema)
        same_source = [
            fk for fk in foreign_keys
            if column in (fk.get("constrained_columns") or [])
        ]
        matching = [
            fk for fk in same_source
            if fk.get("constrained_columns") == [column]
            and fk.get("referred_table") == parent
            and fk.get("referred_columns") == [parent_column]
        ]
        if matching:
            if str((matching[0].get("options") or {}).get("ondelete", "")).upper() != "CASCADE":
                raise SchemaMismatchError("Existing foreign key does not cascade deletes.")
            return "existing cascading foreign key"
        if same_source:
            raise SchemaMismatchError("An incompatible foreign key already uses this column.")
        if _orphan_count(connection, table, column, parent, parent_column):
            raise SchemaMismatchError("Orphaned rows prevent adding the foreign key.")
        if connection.dialect.name != "postgresql":
            raise SchemaMismatchError("SQLite cannot add this foreign key in place.")
        return "missing but addable; no orphan rows found"

    def check_index(table: str, columns: list[str], name: str) -> str:
        if table not in tables:
            return "will be created with table"
        indexes = inspect(connection).get_indexes(table, schema=schema)
        named = next((index for index in indexes if index.get("name") == name), None)
        if named and (named.get("column_names") != columns or named.get("unique")):
            raise SchemaMismatchError("Existing same-name index has an incompatible definition.")
        if any(
            index.get("column_names") == columns and not index.get("unique")
            for index in indexes
        ):
            return "existing compatible index"
        return "missing; v003 will create"

    schema = _schema(connection)
    tables = set(inspect(connection).get_table_names(schema=schema))
    core = set(CORE_TABLES)
    existing_core = core.intersection(tables)
    if existing_core and existing_core != core:
        report(
            "Core table set",
            lambda: (_ for _ in ()).throw(
                SchemaMismatchError("Only part of users/conversations/messages exists.")
            ),
        )
    else:
        report(
            "Core table set",
            lambda: "all present" if existing_core else "none present; v003 will create them",
        )
    report(
        "Duplicate conversation_messages table",
        lambda: (
            (_ for _ in ()).throw(
                SchemaMismatchError("conversation_messages exists; duplicate history storage is forbidden.")
            )
            if "conversation_messages" in tables
            else "absent; existing messages table will be reused"
        ),
    )

    expected_columns = {
        "users": {
            "user_id": ("integer", False),
            "email": ("string", False),
            "password_hash": ("string", False),
            "full_name": ("string", False),
            "is_active": ("boolean", False),
            "created_at": ("datetime", False),
            "updated_at": ("datetime", False),
        },
        "conversations": {
            "conversation_id": ("integer", False),
            "user_id": ("integer", False),
            "conversation_type": ("string", False),
            "agent_id": ("integer", True),
            "title": ("string", False),
            "status": ("string", False),
            "created_at": ("datetime", False),
            "updated_at": ("datetime", False),
        },
        "messages": {
            "message_id": ("integer", False),
            "conversation_id": ("integer", False),
            "sender_type": ("string", False),
            "content": ("text", False),
            "model_key": ("string", True),
            "metadata": ("json", False),
            "created_at": ("datetime", False),
        },
    }
    for table, columns in expected_columns.items():
        if table not in tables:
            report(
                f"{table} table and mapped schema",
                lambda table=table: (
                    (_ for _ in ()).throw(
                        SchemaMismatchError("A partial core schema cannot be reconciled automatically.")
                    )
                    if existing_core
                    else f"missing; v003 will create {table}"
                ),
            )
            continue
        actual = _column_map(connection, table)
        for name, (kind, nullable) in columns.items():
            allow_nullable_title = (
                table == "conversations" and name == "title"
            )
            report(
                f"{table}.{name} column/type/nullability",
                lambda table=table, actual=actual, name=name, kind=kind, nullable=nullable, allow_nullable_title=allow_nullable_title: _require_column(
                    connection,
                    table,
                    actual,
                    name,
                    kind,
                    nullable=(
                        None
                        if allow_nullable_title or (
                            connection.dialect.name == "sqlite"
                            and table == "messages"
                            and name == "metadata"
                        )
                        else nullable
                    ),
                    allow_shorter=True,
                ),
            )
            if (table, name) in REQUIRED_VARCHAR_LENGTHS:
                report(
                    f"{table}.{name} varchar capacity",
                    lambda table=table, name=name: _varchar_width_detail(
                        connection, table, name
                    ),
                )
            if name in actual:
                default_present = actual[name].get("default") is not None
                reports.append({
                    "passed": True,
                    "label": f"{table}.{name} server default",
                    "detail": (
                        "present"
                        if default_present
                        else "absent; v003/SQLAlchemy supplies required values where applicable"
                    ),
                })
            else:
                reports.append({
                    "passed": False,
                    "label": f"{table}.{name} server default",
                    "detail": "column is missing",
                })
        if table == "conversations":
            def title_nullability_detail():
                column = actual["title"]
                null_count = _null_title_count(connection)
                if null_count:
                    raise SchemaMismatchError(
                        f"{null_count} titles are NULL; v003 refuses to rewrite existing records."
                    )
                if column.get("nullable", True) and connection.dialect.name != "postgresql":
                    raise SchemaMismatchError(
                        "SQLite cannot enforce NOT NULL on the existing title column in place."
                    )
                return (
                    "already NOT NULL"
                    if not column.get("nullable", True)
                    else "zero NULL titles; v003 can safely enforce NOT NULL"
                )
            report("conversations.title NULL rows and nullability", title_nullability_detail)
        report(
            f"{table} unmapped required columns",
            lambda table=table, columns=columns: _validate_extra_columns(
                connection,
                table,
                set(columns) | ({"profile"} if table == "users" else set()),
            ),
        )
        primary_key_column = {
            "users": "user_id",
            "conversations": "conversation_id",
            "messages": "message_id",
        }[table]
        report(
            f"{table}.{primary_key_column} primary key and generation",
            lambda table=table, primary_key_column=primary_key_column: _require_primary_key(
                connection, table, primary_key_column
            ),
        )

    if "users" in tables:
        users = _column_map(connection, "users")
        if "profile" in users:
            report(
                "users.profile JSON structure",
                lambda: _require_column(connection, "users", users, "profile", "json"),
            )
            if users["profile"].get("nullable", True):
                def profile_nullability_detail():
                    null_count = _null_profile_count(connection)
                    if null_count:
                        raise SchemaMismatchError(
                            f"{null_count} users.profile values are NULL; v003 will not rewrite them."
                        )
                    if connection.dialect.name != "postgresql":
                        raise SchemaMismatchError(
                            "SQLite cannot enforce users.profile NOT NULL in place."
                        )
                    return "zero NULL values; v003 can safely enforce NOT NULL"
                report("users.profile nullability", profile_nullability_detail)
            else:
                report("users.profile nullability", lambda: "already NOT NULL")
        else:
            report("users.profile JSON structure", lambda: "missing; v003 will add JSONB with an empty-object default")
        report(
            "Case-insensitive duplicate emails",
            lambda: (
                (_ for _ in ()).throw(
                    SchemaMismatchError("Duplicate email groups prevent a unique lower(email) index.")
                )
                if _duplicate_email_groups(connection)
                else "none found"
            ),
        )
        report(
            "Unique lower(email) index",
            lambda: (
                "existing compatible unique index"
                if _lower_email_index_exists(connection)
                else "missing; v003 will create after duplicate check"
            ),
        )
    else:
        report("users.profile JSON structure", lambda: "users table will be created with profile JSONB")
        report("Case-insensitive duplicate emails", lambda: "no existing users table")
        report("Unique lower(email) index", lambda: "will be created with users table")

    for table, key in (("users", "user_id"), ("conversations", "conversation_id"), ("messages", "message_id")):
        if table not in tables:
            report(f"{table}.{key} generated identity/sequence", lambda: "table will be created with generated primary key")

    if "conversations" in tables:
        report(
            "conversations.conversation_type CHECK definitions and distinct values",
            lambda: _conversation_type_check_preflight_detail(connection),
        )

    for (table, column), required_values in REQUIRED_CHECK_VALUES.items():
        if table in tables:
            report(
                f"{table}.{column} CHECK definitions and distinct values",
                lambda table=table, column=column, required_values=required_values: _check_reconciliation_detail(
                    connection, table, column, required_values
                ),
            )

    for table, column, parent, parent_column in (
        ("conversations", "user_id", "users", "user_id"),
        ("messages", "conversation_id", "conversations", "conversation_id"),
        ("auth_sessions", "user_id", "users", "user_id"),
    ):
        report(
            f"FK {table}.{column} -> {parent}.{parent_column}",
            lambda table=table, column=column, parent=parent, parent_column=parent_column: check_foreign_key(
                table, column, parent, parent_column
            ),
        )

    if "auth_sessions" not in tables:
        report("auth_sessions table/columns/PK", lambda: "missing; v003 will create with hashed-token primary key")
        report("FK auth_sessions.user_id -> users.user_id", lambda: "will be created with auth_sessions")
        report("auth_sessions user lookup index", lambda: "will be created with auth_sessions")
    else:
        session_columns = _column_map(connection, "auth_sessions")
        for name, kind in (
            ("token_hash", "string"),
            ("user_id", "integer"),
            ("expires_at", "datetime"),
            ("created_at", "datetime"),
        ):
            report(
                f"auth_sessions.{name} column/type/nullability",
                lambda name=name, kind=kind: _require_column(
                    connection, "auth_sessions", session_columns, name, kind, nullable=False
                ),
            )
            reports.append({
                "passed": True,
                "label": f"auth_sessions.{name} server default",
                "detail": (
                    "present"
                    if session_columns[name].get("default") is not None
                    else "absent; SQLAlchemy supplies required values on insert"
                ),
            })
        report(
            "auth_sessions.token_hash primary key",
            lambda: _require_primary_key(connection, "auth_sessions", "token_hash", generated=False),
        )
        report(
            "auth_sessions unmapped required columns",
            lambda: _validate_extra_columns(
                connection, "auth_sessions", {"token_hash", "user_id", "expires_at", "created_at"}
            ),
        )
        report(
            "auth_sessions user lookup index",
            lambda: check_index("auth_sessions", ["user_id"], "ix_auth_sessions_user_id"),
        )

    for table, columns, name in (
        ("conversations", ["user_id", "updated_at"], "ix_conversations_user_updated"),
        ("messages", ["conversation_id", "message_id"], "ix_messages_conversation_created"),
    ):
        report(
            f"Index {name}",
            lambda table=table, columns=columns, name=name: check_index(table, columns, name),
        )

    if "quanta_schema_migrations" not in tables:
        report("Migration ledger shape", lambda: "missing; v003 will create it after schema validation")
        report("v003 already applied", lambda: "not recorded")
    else:
        ledger_columns = _column_map(connection, "quanta_schema_migrations")
        def check_ledger():
            if not {"version", "applied_at"}.issubset(ledger_columns):
                raise SchemaMismatchError("Ledger is missing required columns.")
            if inspect(connection).get_pk_constraint(
                "quanta_schema_migrations", schema=schema
            ).get("constrained_columns") != ["version"]:
                raise SchemaMismatchError("Ledger version is not its primary key.")
            if not isinstance(ledger_columns["version"]["type"], String):
                raise SchemaMismatchError("Ledger version is not a string type.")
            if (
                ledger_columns["version"]["type"].length is not None
                and ledger_columns["version"]["type"].length < 255
            ):
                raise SchemaMismatchError("Ledger version column is too short.")
            if not isinstance(ledger_columns["applied_at"]["type"], DateTime):
                raise SchemaMismatchError("Ledger applied_at is not a timestamp.")
            if (
                connection.dialect.name == "postgresql"
                and not ledger_columns["applied_at"]["type"].timezone
            ):
                raise SchemaMismatchError("Ledger applied_at must be timezone-aware.")
            if ledger_columns["applied_at"].get("nullable", True) or ledger_columns["applied_at"].get("default") is None:
                raise SchemaMismatchError("Ledger applied_at must be NOT NULL with a server default.")
            _validate_extra_columns(
                connection,
                "quanta_schema_migrations",
                {"version", "applied_at"},
            )
            return "compatible"
        report("Migration ledger shape", check_ledger)
        reports.append({
            "passed": bool(
                ledger_columns.get("applied_at", {}).get("default") is not None
            ),
            "label": "Migration ledger applied_at server default",
            "detail": (
                "present"
                if ledger_columns.get("applied_at", {}).get("default") is not None
                else "missing"
            ),
        })
        if "version" not in ledger_columns:
            report(
                "v003 already applied",
                lambda: (_ for _ in ()).throw(
                    SchemaMismatchError("Migration ledger has no version column.")
                ),
            )
        else:
            ledger_name = f"{schema}.quanta_schema_migrations" if schema else "quanta_schema_migrations"
            applied = connection.execute(text(f"SELECT version FROM {ledger_name}")).scalars().all()
            report(
                "v003 already applied",
                lambda: (
                    (_ for _ in ()).throw(SchemaMismatchError("v003 is already recorded; do not apply again."))
                    if VERSION in applied
                    else "not recorded"
                ),
            )

    return reports