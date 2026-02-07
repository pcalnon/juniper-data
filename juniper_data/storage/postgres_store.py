"""PostgreSQL-backed dataset storage for metadata with file system artifacts."""

import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from juniper_data.core.models import DatasetMeta

from .base import DatasetStore

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

    SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS datasets (
        dataset_id VARCHAR(255) PRIMARY KEY,
        generator VARCHAR(100) NOT NULL,
        generator_version VARCHAR(50) NOT NULL,
        params JSONB NOT NULL,
        n_samples INTEGER NOT NULL,
        n_features INTEGER NOT NULL,
        n_classes INTEGER NOT NULL,
        n_train INTEGER NOT NULL,
        n_test INTEGER NOT NULL,
        class_distribution JSONB NOT NULL,
        artifact_formats TEXT[] NOT NULL DEFAULT ARRAY['npz'],
        created_at TIMESTAMP WITH TIME ZONE NOT NULL,
        checksum VARCHAR(64),
        tags TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
        ttl_seconds INTEGER,
        expires_at TIMESTAMP WITH TIME ZONE,
        last_accessed_at TIMESTAMP WITH TIME ZONE,
        access_count INTEGER NOT NULL DEFAULT 0
    );
    
    CREATE INDEX IF NOT EXISTS idx_datasets_generator ON datasets(generator);
    CREATE INDEX IF NOT EXISTS idx_datasets_created_at ON datasets(created_at);
    CREATE INDEX IF NOT EXISTS idx_datasets_expires_at ON datasets(expires_at);
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "juniper_data",
        user: str = "postgres",
        password: Optional[str] = None,
        artifact_path: Optional[Path] = None,
        connection_string: Optional[str] = None,
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
            raise ImportError(
                "psycopg2 package not installed. "
                "Install with: pip install psycopg2-binary"
            )

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
        return self._artifact_path / f"{dataset_id}.npz"

    def _meta_to_row(self, meta: DatasetMeta) -> dict:
        """Convert DatasetMeta to database row dict."""
        return {
            "dataset_id": meta.dataset_id,
            "generator": meta.generator,
            "generator_version": meta.generator_version,
            "params": json.dumps(meta.params),
            "n_samples": meta.n_samples,
            "n_features": meta.n_features,
            "n_classes": meta.n_classes,
            "n_train": meta.n_train,
            "n_test": meta.n_test,
            "class_distribution": json.dumps(meta.class_distribution),
            "artifact_formats": meta.artifact_formats,
            "created_at": meta.created_at,
            "checksum": meta.checksum,
            "tags": meta.tags,
            "ttl_seconds": meta.ttl_seconds,
            "expires_at": meta.expires_at,
            "last_accessed_at": meta.last_accessed_at,
            "access_count": meta.access_count,
        }

    def _row_to_meta(self, row: dict) -> DatasetMeta:
        """Convert database row to DatasetMeta."""
        return DatasetMeta(
            dataset_id=row["dataset_id"],
            generator=row["generator"],
            generator_version=row["generator_version"],
            params=row["params"] if isinstance(row["params"], dict) else json.loads(row["params"]),
            n_samples=row["n_samples"],
            n_features=row["n_features"],
            n_classes=row["n_classes"],
            n_train=row["n_train"],
            n_test=row["n_test"],
            class_distribution=row["class_distribution"]
            if isinstance(row["class_distribution"], dict)
            else json.loads(row["class_distribution"]),
            artifact_formats=list(row["artifact_formats"]),
            created_at=row["created_at"],
            checksum=row["checksum"],
            tags=list(row["tags"]) if row["tags"] else [],
            ttl_seconds=row["ttl_seconds"],
            expires_at=row["expires_at"],
            last_accessed_at=row["last_accessed_at"],
            access_count=row["access_count"],
        )

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
        row = self._meta_to_row(meta)

        insert_sql = """
        INSERT INTO datasets (
            dataset_id, generator, generator_version, params, n_samples,
            n_features, n_classes, n_train, n_test, class_distribution,
            artifact_formats, created_at, checksum, tags, ttl_seconds,
            expires_at, last_accessed_at, access_count
        ) VALUES (
            %(dataset_id)s, %(generator)s, %(generator_version)s, %(params)s::jsonb,
            %(n_samples)s, %(n_features)s, %(n_classes)s, %(n_train)s, %(n_test)s,
            %(class_distribution)s::jsonb, %(artifact_formats)s, %(created_at)s,
            %(checksum)s, %(tags)s, %(ttl_seconds)s, %(expires_at)s,
            %(last_accessed_at)s, %(access_count)s
        ) ON CONFLICT (dataset_id) DO UPDATE SET
            generator = EXCLUDED.generator,
            generator_version = EXCLUDED.generator_version,
            params = EXCLUDED.params,
            n_samples = EXCLUDED.n_samples,
            n_features = EXCLUDED.n_features,
            n_classes = EXCLUDED.n_classes,
            n_train = EXCLUDED.n_train,
            n_test = EXCLUDED.n_test,
            class_distribution = EXCLUDED.class_distribution,
            artifact_formats = EXCLUDED.artifact_formats,
            checksum = EXCLUDED.checksum,
            tags = EXCLUDED.tags,
            ttl_seconds = EXCLUDED.ttl_seconds,
            expires_at = EXCLUDED.expires_at,
            last_accessed_at = EXCLUDED.last_accessed_at,
            access_count = EXCLUDED.access_count
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(insert_sql, row)
            conn.commit()

        artifact_path = self._artifact_file(dataset_id)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)  # type: ignore[arg-type]
        artifact_path.write_bytes(buffer.getvalue())

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
                cur.execute(
                    "SELECT 1 FROM datasets WHERE dataset_id = %s", (dataset_id,)
                )
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

        update_sql = """
        UPDATE datasets SET
            generator = %(generator)s,
            generator_version = %(generator_version)s,
            params = %(params)s::jsonb,
            n_samples = %(n_samples)s,
            n_features = %(n_features)s,
            n_classes = %(n_classes)s,
            n_train = %(n_train)s,
            n_test = %(n_test)s,
            class_distribution = %(class_distribution)s::jsonb,
            artifact_formats = %(artifact_formats)s,
            checksum = %(checksum)s,
            tags = %(tags)s,
            ttl_seconds = %(ttl_seconds)s,
            expires_at = %(expires_at)s,
            last_accessed_at = %(last_accessed_at)s,
            access_count = %(access_count)s
        WHERE dataset_id = %(dataset_id)s
        """

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
