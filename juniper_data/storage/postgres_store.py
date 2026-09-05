"""PostgreSQL-backed dataset storage for metadata with file system artifacts.

The table schema and both row mappers are DERIVED FROM ``DatasetMeta`` rather than
hand-maintained (juniper-data#320). They used to be three independent transcriptions of the
same field set, and all three had drifted from the model in different directions:

* ``SCHEMA_SQL`` declared ``n_classes`` and ``class_distribution`` ``NOT NULL`` long after
  the model made both nullable for regression and sequence artifacts (WS-1 /
  juniper-data#168) -- so the first regression dataset written would have failed the INSERT;
* ``_meta_to_row`` / ``_row_to_meta`` enumerated 23 of the model's 30 fields, silently
  dropping ``task_type``, ``sequence``, ``lookback``, ``time_unit``, ``dt_scaling``,
  ``target_scaling`` and ``truncation`` on every round trip;
* ``_row_to_meta`` raised ``TypeError`` on a NULL ``class_distribution`` because its
  ``isinstance(..., dict) else json.loads(...)`` fallback did not admit ``None``.

Deriving them makes that class of drift unrepresentable: a new field on ``DatasetMeta``
appears in the DDL, the INSERT and the read without anyone remembering to add it.
"""

import io
import json
import re
from datetime import datetime
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin
from uuid import uuid4

import numpy as np

from juniper_data.core.models import DatasetMeta
from juniper_data.storage.constants import (
    NPZ_FILE_SUFFIX,
    POSTGRES_DEFAULT_HOST,
    POSTGRES_DEFAULT_PORT,
)

from .base import DatasetStore

# ─── Model-derived schema (juniper-data#320) ─────────────────────────────────

#: A bare SQL identifier. Used to gate the one interpolated position (the table name).
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

#: Column that is the table's primary key.
_PRIMARY_KEY = "dataset_id"

#: Columns that carry a JSON document. Stored JSONB, dumped on write, loaded on read.
_JSONB_FIELDS = frozenset({"params", "class_distribution", "dt_scaling", "target_scaling", "truncation"})

#: Columns that carry a list of strings. Stored as a native TEXT[] rather than JSON so the
#: existing ``tags`` containment queries keep working.
_TEXT_ARRAY_FIELDS = frozenset({"artifact_formats", "tags"})

#: Scalar Python type -> PostgreSQL column type.
_SCALAR_SQL_TYPES: dict[type, str] = {
    bool: "BOOLEAN",
    int: "INTEGER",
    float: "DOUBLE PRECISION",
    str: "TEXT",
    datetime: "TIMESTAMP WITH TIME ZONE",
}

#: SQL defaults for non-nullable columns whose model field has a default. Without these an
#: ``ADD COLUMN ... NOT NULL`` against a populated table fails.
_SQL_DEFAULTS: dict[str, str] = {
    "artifact_formats": "ARRAY['npz']",
    "tags": "ARRAY[]::TEXT[]",
    "access_count": "0",
    "n_val": "0",
    "sequence": "FALSE",
    "task_type": "'classification'",
}


def _safe_identifier(name: str) -> str:
    """Return ``name`` if it is a bare SQL identifier, else raise.

    The DDL and statement builders interpolate a table name, which bandit correctly flags as
    a B608 string-construction pattern. Values cannot be parameterised into an identifier
    position, so the defence is to prove the identifier is not attacker-shaped: this admits
    only ``[A-Za-z_][A-Za-z0-9_]*``, which cannot carry a quote, a semicolon, a comment
    marker or whitespace.

    Column names need no such check -- they come from ``DatasetMeta.model_fields``, which are
    Python identifiers by construction and never derived from a request.
    """
    if not _IDENTIFIER_RE.fullmatch(name):
        raise ValueError(f"unsafe SQL identifier: {name!r}")
    return name


def _admits_none(annotation: Any) -> bool:
    """True when ``None`` is a valid value for this annotation."""
    if get_origin(annotation) in (Union, UnionType):
        return type(None) in get_args(annotation)
    return annotation is type(None)


def _non_none_type(annotation: Any) -> Any:
    """The annotation with ``None`` stripped out of a union."""
    if get_origin(annotation) in (Union, UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        return args[0] if args else annotation
    return annotation


def _sql_type_for(name: str, annotation: Any) -> str:
    """PostgreSQL column type for one model field."""
    if name in _JSONB_FIELDS:
        return "JSONB"
    if name in _TEXT_ARRAY_FIELDS:
        return "TEXT[]"
    base = _non_none_type(annotation)
    if get_origin(base) is dict:
        return "JSONB"
    if get_origin(base) is list:
        return "TEXT[]"
    return _SCALAR_SQL_TYPES.get(base, "TEXT")


def _column_specs() -> list[tuple[str, str, bool]]:
    """``(name, sql_type, nullable)`` for every ``DatasetMeta`` field, in declaration order.

    A column is nullable exactly when the model admits ``None`` for it. A field that is
    optional-but-not-``None``-able (``task_type``, ``sequence``, ``access_count`` and the two
    list fields) stays NOT NULL and carries a SQL default, because the model would reject a
    ``None`` read back out of it.
    """
    specs: list[tuple[str, str, bool]] = []
    for name, field in DatasetMeta.model_fields.items():
        specs.append((name, _sql_type_for(name, field.annotation), _admits_none(field.annotation)))
    return specs


def build_schema_sql(table: str = "datasets") -> str:
    """Generate the full DDL from ``DatasetMeta``.

    Emits ``CREATE TABLE IF NOT EXISTS`` plus an ``ALTER TABLE ... ADD COLUMN IF NOT EXISTS``
    for every column. The ALTERs are what migrate an existing deployment: ``SCHEMA_SQL`` runs
    on every init, so a table created before a field existed gains it without a manual
    migration -- and ``IF NOT EXISTS`` is what keeps the second boot from erroring.
    """
    table = _safe_identifier(table)
    specs = _column_specs()
    lines = []
    for name, sql_type, nullable in specs:
        if name == _PRIMARY_KEY:
            lines.append(f"    {name} {sql_type} PRIMARY KEY")
            continue
        parts = [f"    {name} {sql_type}"]
        if not nullable:
            parts.append("NOT NULL")
            if name in _SQL_DEFAULTS:
                parts.append(f"DEFAULT {_SQL_DEFAULTS[name]}")
        lines.append(" ".join(parts))

    create = f"CREATE TABLE IF NOT EXISTS {table} (\n" + ",\n".join(lines) + "\n);"

    alters = []
    for name, sql_type, nullable in specs:
        if name == _PRIMARY_KEY:
            continue
        clause = f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {name} {sql_type}"
        # An ADD COLUMN on a POPULATED table cannot be NOT NULL without a default, so only
        # the defaulted columns carry the constraint here. The CREATE TABLE above still
        # declares it for a fresh database.
        if not nullable and name in _SQL_DEFAULTS:
            clause += f" NOT NULL DEFAULT {_SQL_DEFAULTS[name]}"
        alters.append(clause + ";")

    indexes = [
        f"CREATE INDEX IF NOT EXISTS idx_{table}_generator ON {table}(generator);",
        f"CREATE INDEX IF NOT EXISTS idx_{table}_created_at ON {table}(created_at);",
        f"CREATE INDEX IF NOT EXISTS idx_{table}_expires_at ON {table}(expires_at);",
        f"CREATE INDEX IF NOT EXISTS idx_{table}_name ON {table}(dataset_name);",
    ]
    return "\n".join([create, "", *alters, "", *indexes, ""])


#: Columns an upsert must NOT overwrite. ``created_at`` records when the dataset first
#: existed; a re-save is not a re-creation. This mirrors the previous hand-written statement.
_UPSERT_PRESERVED = frozenset({_PRIMARY_KEY, "created_at"})


def build_update_sql(table: str = "datasets") -> str:
    """Generate the UPDATE ... WHERE from ``DatasetMeta``.

    The FIFTH transcription of the field list. Same 23-of-30 omission as the others, so
    ``update_meta`` silently discarded the sequence and scaling metadata on every edit.
    """
    table = _safe_identifier(table)
    names = [name for name, _, _ in _column_specs() if name not in _UPSERT_PRESERVED]
    assignments = ",\n            ".join(f"{n} = %({n})s::jsonb" if n in _JSONB_FIELDS else f"{n} = %({n})s" for n in names)
    # nosec B608 - the only interpolated identifier is `table`, gated by _safe_identifier;
    # column names come from DatasetMeta.model_fields and are never request-derived. Values
    # are bound by the driver through %(name)s placeholders, not interpolated.
    statement = f"""
        UPDATE {table} SET
            {assignments}
        WHERE {_PRIMARY_KEY} = %({_PRIMARY_KEY})s
        """  # nosec B608
    return statement


def build_upsert_sql(table: str = "datasets") -> str:
    """Generate the INSERT ... ON CONFLICT DO UPDATE from ``DatasetMeta``.

    This was a FOURTH independent transcription of the field list (after the DDL and the two
    row mappers), enumerating the same 23 of 30 columns -- so deriving the mappers alone would
    still have dropped the other seven at the INSERT. All four now come from one source.
    """
    table = _safe_identifier(table)
    names = [name for name, _, _ in _column_specs()]
    columns = ", ".join(names)
    values = ", ".join(f"%({n})s::jsonb" if n in _JSONB_FIELDS else f"%({n})s" for n in names)
    updates = ",\n            ".join(f"{n} = EXCLUDED.{n}" for n in names if n not in _UPSERT_PRESERVED)
    # nosec B608 - same reasoning as build_update_sql: `table` is gated by _safe_identifier,
    # column names come from DatasetMeta.model_fields, and every value is driver-bound.
    statement = f"""
        INSERT INTO {table} (
            {columns}
        ) VALUES (
            {values}
        ) ON CONFLICT ({_PRIMARY_KEY}) DO UPDATE SET
            {updates}
        """  # nosec B608
    return statement


try:
    import psycopg2
    from psycopg2.extras import RealDictCursor

    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    psycopg2 = None  # type: ignore[assignment]


class PostgresDatasetStore(DatasetStore):
    """PostgreSQL-backed dataset storage.

    Stores metadata in PostgreSQL and artifacts on the local filesystem.
    Suitable for production deployments with database-backed metadata.

    Requires the `psycopg2` package: pip install psycopg2-binary
    """

    #: Full DDL, DERIVED from ``DatasetMeta`` (juniper-data#320) rather than transcribed.
    #: Adding a field to the model adds its column here automatically.
    SCHEMA_SQL = build_schema_sql()

    def __init__(
        self,
        host: str = POSTGRES_DEFAULT_HOST,
        port: int = POSTGRES_DEFAULT_PORT,
        database: str = "juniper_data",
        user: str = "postgres",
        password: str | None = None,
        artifact_path: Path | None = None,
        connection_string: str | None = None,
        auto_create_schema: bool = True,
    ) -> None:
        """Initialize PostgreSQL connection.

        Args:
            host: PostgreSQL server hostname.
            port: PostgreSQL server port.
            database: Database name.
            user: Database user.
            password: Database password.
            artifact_path: Path for storing NPZ artifacts.
            connection_string: Optional full connection string (overrides other params).
            auto_create_schema: Automatically create tables if they don't exist.

        Raises:
            ImportError: If psycopg2 package is not installed.
        """
        if not POSTGRES_AVAILABLE:
            raise ImportError("psycopg2 package not installed. Install with: pip install psycopg2-binary")

        self._artifact_path = artifact_path or Path("./data/datasets")
        self._artifact_path.mkdir(parents=True, exist_ok=True)

        if connection_string:
            self._conn_params: dict[str, Any] = {"dsn": connection_string}
        else:
            self._conn_params = {
                "host": host,
                "port": str(port),
                "database": database,
                "user": user,
                "password": password or "",
            }

        if auto_create_schema:
            self._create_schema()

    def _get_connection(self) -> Any:
        """Get a new database connection."""
        return psycopg2.connect(**self._conn_params)

    def _create_schema(self) -> None:
        """Create database schema if it doesn't exist."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(self.SCHEMA_SQL)
            conn.commit()

    def _artifact_file(self, dataset_id: str) -> Path:
        """Get the artifact file path for a dataset."""
        return self._artifact_path / f"{dataset_id}{NPZ_FILE_SUFFIX}"

    def _meta_to_row(self, meta: DatasetMeta) -> dict:
        """Convert ``DatasetMeta`` to a database row dict, DERIVED from the model.

        Iterating ``model_fields`` is what stops this drifting: the previous hand-written
        version enumerated 23 of 30 fields and silently dropped the other seven on every
        write (juniper-data#320).
        """
        row: dict[str, Any] = {}
        for name in DatasetMeta.model_fields:
            value = getattr(meta, name)
            # ``json.dumps(None)`` yields the string "null", which round-trips as a JSON null
            # rather than a SQL NULL -- keep None as None so the column is genuinely NULL.
            row[name] = json.dumps(value) if (name in _JSONB_FIELDS and value is not None) else value
        return row

    def _row_to_meta(self, row: dict) -> DatasetMeta:
        """Convert a database row to ``DatasetMeta``, DERIVED from the model.

        A column that is absent (an older table not yet migrated) or NULL for a field that
        does not admit ``None`` is OMITTED from the constructor call, so the model's own
        default applies. That makes the read tolerant of a schema behind the code in either
        direction, which is what the previous version got wrong: it indexed ``row[...]``
        unconditionally and its ``isinstance(..., dict) else json.loads(...)`` fallback raised
        ``TypeError`` on a NULL ``class_distribution``.
        """
        kwargs: dict[str, Any] = {}
        for name, field in DatasetMeta.model_fields.items():
            if name not in row:
                continue
            value = row[name]
            if value is None:
                # Only pass an explicit None where the model actually accepts one.
                if _admits_none(field.annotation):
                    kwargs[name] = None
                continue
            if name in _JSONB_FIELDS and not isinstance(value, (dict, list)):
                value = json.loads(value)
            elif name in _TEXT_ARRAY_FIELDS:
                value = list(value)
            kwargs[name] = value
        return DatasetMeta(**kwargs)

    def save(
        self,
        dataset_id: str,
        meta: DatasetMeta,
        arrays: dict[str, np.ndarray],
    ) -> None:
        """Save dataset to PostgreSQL and filesystem.

        Args:
            dataset_id: Unique identifier for the dataset.
            meta: Dataset metadata.
            arrays: Dictionary of numpy arrays.
        """
        # DERIVED from DatasetMeta (juniper-data#320) -- see build_upsert_sql. The
        # hand-written version here enumerated 23 of 30 columns.
        insert_sql = build_upsert_sql()

        artifact_path = self._artifact_file(dataset_id)
        # Use a per-save temp file so concurrent writes for the same dataset_id
        # cannot clobber each other's staging artifact.
        tmp_artifact_path = artifact_path.with_name(f"{artifact_path.name}.{uuid4().hex}.tmp")
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)  # type: ignore[arg-type]
        tmp_artifact_path.write_bytes(buffer.getvalue())

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # Serialize writes for the same dataset_id to avoid racy upserts mutating metadata.
                    cur.execute("SELECT pg_advisory_xact_lock(%s, hashtext(%s))", (1, dataset_id))

                    cur.execute(
                        "SELECT dataset_name, dataset_version FROM datasets WHERE dataset_id = %s",
                        (dataset_id,),
                    )
                    existing = cur.fetchone()
                    existing_name: str | None = None
                    existing_version: int | None = None
                    if isinstance(existing, dict):
                        existing_name = existing.get("dataset_name")
                        existing_version = existing.get("dataset_version")
                    elif isinstance(existing, tuple) and len(existing) >= 2:
                        existing_name, existing_version = existing[0], existing[1]

                    if existing_name is not None:
                        meta.dataset_name = existing_name
                    if existing_version is not None:
                        meta.dataset_version = int(existing_version)
                    elif meta.dataset_name is not None:
                        # Serialize version allocation per logical dataset name.
                        cur.execute("SELECT pg_advisory_xact_lock(%s, hashtext(%s))", (2, meta.dataset_name))
                        cur.execute(
                            "SELECT COALESCE(MAX(dataset_version), 0) + 1 FROM datasets WHERE dataset_name = %s",
                            (meta.dataset_name,),
                        )
                        next_version_row = cur.fetchone()
                        if next_version_row is not None:
                            meta.dataset_version = int(next_version_row[0])

                    row = self._meta_to_row(meta)
                    cur.execute(insert_sql, row)

                # Keep metadata and artifact writes consistent: if artifact replace fails,
                # transaction is rolled back and no metadata-only record is committed.
                tmp_artifact_path.replace(artifact_path)
                conn.commit()
        except Exception:
            tmp_artifact_path.unlink(missing_ok=True)
            raise

    def get_meta(self, dataset_id: str) -> DatasetMeta | None:
        """Get dataset metadata from PostgreSQL.

        Args:
            dataset_id: Unique identifier for the dataset.

        Returns:
            Dataset metadata if found, None otherwise.
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM datasets WHERE dataset_id = %s", (dataset_id,))
                row = cur.fetchone()

        if row is None:
            return None

        return self._row_to_meta(dict(row))

    def get_artifact_bytes(self, dataset_id: str) -> bytes | None:
        """Get dataset artifact bytes from filesystem.

        Args:
            dataset_id: Unique identifier for the dataset.

        Returns:
            NPZ bytes if found, None otherwise.
        """
        artifact_path = self._artifact_file(dataset_id)
        if not artifact_path.exists():
            return None
        return artifact_path.read_bytes()

    def exists(self, dataset_id: str) -> bool:
        """Check if dataset exists in PostgreSQL.

        Args:
            dataset_id: Unique identifier for the dataset.

        Returns:
            True if the dataset exists, False otherwise.
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM datasets WHERE dataset_id = %s", (dataset_id,))
                return cur.fetchone() is not None

    def delete(self, dataset_id: str) -> bool:
        """Delete dataset from PostgreSQL and filesystem.

        Args:
            dataset_id: Unique identifier for the dataset.

        Returns:
            True if the dataset was deleted, False if it didn't exist.
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM datasets WHERE dataset_id = %s RETURNING dataset_id",
                    (dataset_id,),
                )
                deleted = cur.fetchone() is not None
            conn.commit()

        artifact_path = self._artifact_file(dataset_id)
        if artifact_path.exists():
            artifact_path.unlink()

        return deleted

    def list_datasets(self, limit: int = 100, offset: int = 0) -> list[str]:
        """List dataset IDs from PostgreSQL.

        Args:
            limit: Maximum number of dataset IDs to return.
            offset: Number of dataset IDs to skip.

        Returns:
            List of dataset IDs.
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT dataset_id FROM datasets ORDER BY created_at DESC LIMIT %s OFFSET %s",
                    (limit, offset),
                )
                rows = cur.fetchall()

        return [row[0] for row in rows]

    def update_meta(self, dataset_id: str, meta: DatasetMeta) -> bool:
        """Update dataset metadata in PostgreSQL.

        Args:
            dataset_id: Unique identifier for the dataset.
            meta: Updated dataset metadata.

        Returns:
            True if the dataset was updated, False if it didn't exist.
        """
        row = self._meta_to_row(meta)

        # DERIVED from DatasetMeta (juniper-data#320) -- see build_update_sql. The
        # hand-written version here enumerated 23 of 30 columns, so update_meta
        # silently discarded the sequence and scaling metadata on every edit.
        update_sql = build_update_sql()

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(update_sql, row)
                updated = cur.rowcount > 0
            conn.commit()

        return updated

    def list_all_metadata(self) -> list[DatasetMeta]:
        """List all dataset metadata from PostgreSQL.

        Returns:
            List of all DatasetMeta objects.
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM datasets ORDER BY created_at DESC")
                rows = cur.fetchall()

        return [self._row_to_meta(dict(row)) for row in rows]

    def close(self) -> None:
        """Close database connections (no-op for connection-per-request pattern)."""
        pass
