"""Unit tests for PostgresDatasetStore."""

import io
import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from juniper_data.core.models import DatasetMeta


@pytest.fixture
def sample_meta() -> DatasetMeta:
    """Create sample metadata."""
    return DatasetMeta(
        dataset_id="test-dataset",
        generator="test",
        generator_version="1.0.0",
        params={"seed": 42},
        n_samples=100,
        n_features=2,
        n_classes=2,
        n_train=80,
        n_test=20,
        class_distribution={"0": 50, "1": 50},
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        dataset_name="experiment-a",
        dataset_version=7,
        parent_dataset_id="parent-dataset",
        description="Regression metadata",
        created_by="qa-bot",
    )


@pytest.fixture
def sample_arrays() -> dict[str, np.ndarray]:
    """Create sample arrays."""
    rng = np.random.default_rng(42)
    return {
        "X_train": rng.standard_normal((80, 2)).astype(np.float32),
        "y_train": rng.standard_normal((80, 2)).astype(np.float32),
        "X_test": rng.standard_normal((20, 2)).astype(np.float32),
        "y_test": rng.standard_normal((20, 2)).astype(np.float32),
    }


@pytest.fixture
def mock_psycopg2():
    """Create a mock psycopg2 module and patch it into postgres_store."""
    mock_pg = MagicMock()
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    _mock_dict_cursor = MagicMock()

    mock_pg.connect.return_value = mock_conn
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)

    mock_pg.extras = MagicMock()
    mock_pg.extras.RealDictCursor = MagicMock()

    with patch.dict("sys.modules", {"psycopg2": mock_pg, "psycopg2.extras": mock_pg.extras}):
        with patch("juniper_data.storage.postgres_store.POSTGRES_AVAILABLE", True):
            with patch("juniper_data.storage.postgres_store.psycopg2", mock_pg):
                with patch("juniper_data.storage.postgres_store.RealDictCursor", mock_pg.extras.RealDictCursor, create=True):
                    yield mock_pg, mock_conn, mock_cursor


@pytest.mark.unit
@pytest.mark.storage
class TestPostgresDatasetStoreInit:
    """Tests for PostgresDatasetStore initialization."""

    def test_init_default_params(self, mock_psycopg2, tmp_path) -> None:
        """Initialize with default parameters."""
        mock_pg, mock_conn, mock_cursor = mock_psycopg2
        from juniper_data.storage.postgres_store import PostgresDatasetStore

        with patch.object(Path, "mkdir"):
            store = PostgresDatasetStore(artifact_path=tmp_path / "artifacts")
        assert store._conn_params["host"] == "localhost"
        assert store._conn_params["port"] == "5432"

    def test_init_custom_params(self, mock_psycopg2, tmp_path) -> None:
        """Initialize with custom parameters."""
        mock_pg, _, _ = mock_psycopg2
        from juniper_data.storage.postgres_store import PostgresDatasetStore

        store = PostgresDatasetStore(
            host="db.example.com",
            port=5433,
            database="mydb",
            user="admin",
            password="secret",
            artifact_path=tmp_path / "data",
        )
        assert store._conn_params["host"] == "db.example.com"
        assert store._conn_params["port"] == "5433"
        assert store._conn_params["database"] == "mydb"
        assert store._conn_params["user"] == "admin"
        assert store._conn_params["password"] == "secret"

    def test_init_with_connection_string(self, mock_psycopg2, tmp_path) -> None:
        """Initialize with connection string overrides individual params."""
        from juniper_data.storage.postgres_store import PostgresDatasetStore

        store = PostgresDatasetStore(connection_string="postgresql://user:pass@host/db", artifact_path=tmp_path / "data")
        assert store._conn_params == {"dsn": "postgresql://user:pass@host/db"}

    def test_init_without_auto_schema(self, mock_psycopg2, tmp_path) -> None:
        """Initialize without auto-creating schema."""
        mock_pg, mock_conn, _ = mock_psycopg2
        from juniper_data.storage.postgres_store import PostgresDatasetStore

        call_count_before = mock_pg.connect.call_count
        _store = PostgresDatasetStore(auto_create_schema=False, artifact_path=tmp_path / "data")
        assert mock_pg.connect.call_count == call_count_before

    def test_init_raises_without_psycopg2(self) -> None:
        """Raises ImportError when psycopg2 is not available."""
        with patch("juniper_data.storage.postgres_store.POSTGRES_AVAILABLE", False):
            from juniper_data.storage.postgres_store import PostgresDatasetStore

            with pytest.raises(ImportError, match="psycopg2 package not installed"):
                PostgresDatasetStore()


@pytest.mark.unit
@pytest.mark.storage
class TestPostgresDatasetStoreMetaConversion:
    """Tests for metadata <-> row conversion."""

    def test_meta_to_row(self, mock_psycopg2, tmp_path, sample_meta) -> None:
        """_meta_to_row converts DatasetMeta to dict."""
        from juniper_data.storage.postgres_store import PostgresDatasetStore

        store = PostgresDatasetStore(auto_create_schema=False, artifact_path=tmp_path / "data")
        row = store._meta_to_row(sample_meta)

        assert row["dataset_id"] == "test-dataset"
        assert row["generator"] == "test"
        assert row["n_samples"] == 100
        assert isinstance(row["params"], str)
        assert json.loads(row["params"]) == {"seed": 42}
        assert row["dataset_name"] == "experiment-a"
        assert row["dataset_version"] == 7
        assert row["parent_dataset_id"] == "parent-dataset"
        assert row["description"] == "Regression metadata"
        assert row["created_by"] == "qa-bot"

    def test_meta_to_row_includes_versioning_fields(self, mock_psycopg2, tmp_path, sample_meta) -> None:
        """_meta_to_row includes versioning metadata fields."""
        from juniper_data.storage.postgres_store import PostgresDatasetStore

        store = PostgresDatasetStore(auto_create_schema=False, artifact_path=tmp_path / "data")
        versioned_meta = sample_meta.model_copy(
            update={
                "dataset_name": "experiment-a",
                "dataset_version": 3,
                "parent_dataset_id": "parent-ds",
                "description": "Versioned metadata",
                "created_by": "integration-test",
            }
        )

        row = store._meta_to_row(versioned_meta)

        assert row["dataset_name"] == "experiment-a"
        assert row["dataset_version"] == 3
        assert row["parent_dataset_id"] == "parent-ds"
        assert row["description"] == "Versioned metadata"
        assert row["created_by"] == "integration-test"

    def test_row_to_meta_with_dict_params(self, mock_psycopg2, tmp_path, sample_meta) -> None:
        """_row_to_meta handles dict params (already parsed JSON)."""
        from juniper_data.storage.postgres_store import PostgresDatasetStore

        store = PostgresDatasetStore(auto_create_schema=False, artifact_path=tmp_path / "data")
        row = {
            "dataset_id": "test-dataset",
            "generator": "test",
            "generator_version": "1.0.0",
            "params": {"seed": 42},
            "n_samples": 100,
            "n_features": 2,
            "n_classes": 2,
            "n_train": 80,
            "n_test": 20,
            "class_distribution": {"0": 50, "1": 50},
            "artifact_formats": ["npz"],
            "created_at": datetime(2026, 1, 1, tzinfo=UTC),
            "checksum": None,
            "dataset_name": "experiment-a",
            "dataset_version": 3,
            "parent_dataset_id": "parent-ds",
            "description": "Versioned dataset for regression tests",
            "created_by": "test-user",
            "tags": ["test"],
            "ttl_seconds": None,
            "expires_at": None,
            "last_accessed_at": None,
            "access_count": 0,
        }

        meta = store._row_to_meta(row)
        assert meta.dataset_id == "test-dataset"
        assert meta.params == {"seed": 42}
        assert meta.dataset_name == "experiment-a"
        assert meta.dataset_version == 3
        assert meta.parent_dataset_id == "parent-ds"
        assert meta.description == "Versioned dataset for regression tests"
        assert meta.created_by == "test-user"

    def test_row_to_meta_with_string_params(self, mock_psycopg2, tmp_path) -> None:
        """_row_to_meta handles string params (JSON string from DB)."""
        from juniper_data.storage.postgres_store import PostgresDatasetStore

        store = PostgresDatasetStore(auto_create_schema=False, artifact_path=tmp_path / "data")
        row = {
            "dataset_id": "test-dataset",
            "generator": "test",
            "generator_version": "1.0.0",
            "params": '{"seed": 42}',
            "n_samples": 100,
            "n_features": 2,
            "n_classes": 2,
            "n_train": 80,
            "n_test": 20,
            "class_distribution": '{"0": 50, "1": 50}',
            "artifact_formats": ["npz"],
            "created_at": datetime(2026, 1, 1, tzinfo=UTC),
            "checksum": None,
            "dataset_name": "experiment-a",
            "dataset_version": 7,
            "parent_dataset_id": "parent-dataset",
            "description": "Regression metadata",
            "created_by": "qa-bot",
            "tags": None,
            "ttl_seconds": None,
            "expires_at": None,
            "last_accessed_at": None,
            "access_count": 0,
        }

        meta = store._row_to_meta(row)
        assert meta.params == {"seed": 42}
        assert meta.class_distribution == {"0": 50, "1": 50}
        assert meta.tags == []
        assert meta.dataset_name == "experiment-a"
        assert meta.dataset_version == 7
        assert meta.parent_dataset_id == "parent-dataset"
        assert meta.description == "Regression metadata"
        assert meta.created_by == "qa-bot"


@pytest.mark.unit
@pytest.mark.storage
class TestPostgresDatasetStoreSave:
    """Tests for save operation."""

    def test_save(self, mock_psycopg2, tmp_path, sample_meta, sample_arrays) -> None:
        """save writes metadata to DB and artifact to filesystem."""
        mock_pg, mock_conn, mock_cursor = mock_psycopg2
        from juniper_data.storage.postgres_store import PostgresDatasetStore

        store = PostgresDatasetStore(auto_create_schema=False, artifact_path=tmp_path / "data")
        store.save("test-dataset", sample_meta, sample_arrays)

        mock_cursor.execute.assert_called()
        executed_sql, executed_params = mock_cursor.execute.call_args.args
        assert "dataset_name" in executed_sql
        assert "dataset_version" in executed_sql
        assert executed_params["dataset_name"] == "experiment-a"
        assert executed_params["dataset_version"] == 3
        artifact_path = tmp_path / "data" / "test-dataset.npz"
        assert artifact_path.exists()

    def test_save_assigns_next_version_in_db_transaction(self, mock_psycopg2, tmp_path, sample_meta, sample_arrays) -> None:
        """save computes next version from DB for named datasets."""
        _, _, mock_cursor = mock_psycopg2
        from juniper_data.storage.postgres_store import PostgresDatasetStore

        store = PostgresDatasetStore(auto_create_schema=False, artifact_path=tmp_path / "data")
        sample_meta.dataset_name = "experiment-a"
        sample_meta.dataset_version = 999  # caller-provided value should be overridden

        # Existing row check -> none, next version query -> 3
        mock_cursor.fetchone.side_effect = [None, (3,)]

        store.save("test-dataset", sample_meta, sample_arrays)

        assert sample_meta.dataset_version == 3
        assert any(
            "MAX(dataset_version)" in call.args[0] for call in mock_cursor.execute.call_args_list if call.args
        )
        assert mock_cursor.execute.call_args_list[-1].args[1]["dataset_version"] == 3

    def test_save_preserves_existing_dataset_version_on_upsert(self, mock_psycopg2, tmp_path, sample_meta, sample_arrays) -> None:
        """save keeps existing version metadata for cached dataset_id upserts."""
        _, _, mock_cursor = mock_psycopg2
        from juniper_data.storage.postgres_store import PostgresDatasetStore

        store = PostgresDatasetStore(auto_create_schema=False, artifact_path=tmp_path / "data")
        sample_meta.dataset_name = "request-name"
        sample_meta.dataset_version = 10

        # Existing dataset row in DB owns canonical version metadata.
        mock_cursor.fetchone.return_value = ("canonical-name", 2)

        store.save("test-dataset", sample_meta, sample_arrays)

        assert sample_meta.dataset_name == "canonical-name"
        assert sample_meta.dataset_version == 2
        assert mock_cursor.execute.call_args_list[-1].args[1]["dataset_version"] == 2
    def test_save_persists_versioning_fields(self, mock_psycopg2, tmp_path, sample_meta, sample_arrays) -> None:
        """save sends versioning metadata fields in DB write parameters."""
        _, _, mock_cursor = mock_psycopg2
        from juniper_data.storage.postgres_store import PostgresDatasetStore

        versioned_meta = sample_meta.model_copy(
            update={
                "dataset_name": "experiment-a",
                "dataset_version": 4,
                "parent_dataset_id": "parent-ds",
                "description": "Versioned save",
                "created_by": "integration-test",
            }
        )
        store = PostgresDatasetStore(auto_create_schema=False, artifact_path=tmp_path / "data")
        store.save("test-dataset", versioned_meta, sample_arrays)

        execute_args = mock_cursor.execute.call_args_list[-1][0]
        sql = execute_args[0]
        params = execute_args[1]

        assert "dataset_name" in sql
        assert params["dataset_name"] == "experiment-a"
        assert params["dataset_version"] == 4
        assert params["parent_dataset_id"] == "parent-ds"
        assert params["description"] == "Versioned save"
        assert params["created_by"] == "integration-test"

    def test_save_does_not_commit_metadata_when_artifact_replace_fails(
        self, mock_psycopg2, tmp_path, sample_meta, sample_arrays
    ) -> None:
        """save rolls back DB transaction if artifact file cannot be finalized."""
        _, mock_conn, _ = mock_psycopg2
        from juniper_data.storage.postgres_store import PostgresDatasetStore

        store = PostgresDatasetStore(auto_create_schema=False, artifact_path=tmp_path / "data")
        artifact_path = tmp_path / "data" / "test-dataset.npz"
        tmp_artifact_path = artifact_path.with_suffix(".npz.tmp")

        with patch.object(Path, "replace", side_effect=OSError("rename failed")):
            with pytest.raises(OSError, match="rename failed"):
                store.save("test-dataset", sample_meta, sample_arrays)

        assert mock_conn.commit.call_count == 0
        assert not tmp_artifact_path.exists()

    def test_save_uses_distinct_temp_artifact_per_call(
        self, mock_psycopg2, tmp_path, sample_meta, sample_arrays
    ) -> None:
        """Each save call should stage artifact bytes in a unique temp file."""
        _, _, mock_cursor = mock_psycopg2
        from juniper_data.storage.postgres_store import PostgresDatasetStore

        store = PostgresDatasetStore(auto_create_schema=False, artifact_path=tmp_path / "data")
        mock_cursor.fetchone.return_value = None

        meta_a = sample_meta.model_copy(update={"dataset_name": None, "dataset_version": None})
        meta_b = sample_meta.model_copy(update={"dataset_name": None, "dataset_version": None})
        replaced_sources: list[str] = []
        original_replace = Path.replace

        def record_replace(path_obj: Path, target: Path) -> Path:
            replaced_sources.append(str(path_obj))
            return original_replace(path_obj, target)

        with patch.object(Path, "replace", autospec=True, side_effect=record_replace):
            store.save("test-dataset", meta_a, sample_arrays)
            store.save("test-dataset", meta_b, sample_arrays)

        assert len(replaced_sources) == 2
        assert replaced_sources[0] != replaced_sources[1]


@pytest.mark.unit
@pytest.mark.storage
class TestPostgresDatasetStoreGetMeta:
    """Tests for get_meta operation."""

    def test_get_meta_found(self, mock_psycopg2, tmp_path, sample_meta) -> None:
        """get_meta returns metadata when found."""
        _, mock_conn, mock_cursor = mock_psycopg2
        from juniper_data.storage.postgres_store import PostgresDatasetStore

        store = PostgresDatasetStore(auto_create_schema=False, artifact_path=tmp_path / "data")

        row_data = {
            "dataset_id": "test-dataset",
            "generator": "test",
            "generator_version": "1.0.0",
            "params": {"seed": 42},
            "n_samples": 100,
            "n_features": 2,
            "n_classes": 2,
            "n_train": 80,
            "n_test": 20,
            "class_distribution": {"0": 50, "1": 50},
            "artifact_formats": ["npz"],
            "created_at": datetime(2026, 1, 1, tzinfo=UTC),
            "checksum": None,
            "dataset_name": "experiment-a",
            "dataset_version": 7,
            "parent_dataset_id": "parent-dataset",
            "description": "Regression metadata",
            "created_by": "qa-bot",
            "tags": [],
            "ttl_seconds": None,
            "expires_at": None,
            "last_accessed_at": None,
            "access_count": 0,
        }
        mock_cursor.fetchone.return_value = row_data

        result = store.get_meta("test-dataset")
        assert result is not None
        assert result.dataset_id == "test-dataset"
        assert result.dataset_name == "experiment-a"
        assert result.dataset_version == 7

    def test_get_meta_not_found(self, mock_psycopg2, tmp_path) -> None:
        """get_meta returns None when not found."""
        _, _, mock_cursor = mock_psycopg2
        from juniper_data.storage.postgres_store import PostgresDatasetStore

        store = PostgresDatasetStore(auto_create_schema=False, artifact_path=tmp_path / "data")
        mock_cursor.fetchone.return_value = None

        result = store.get_meta("nonexistent")
        assert result is None


@pytest.mark.unit
@pytest.mark.storage
class TestPostgresDatasetStoreGetArtifact:
    """Tests for get_artifact_bytes operation."""

    def test_get_artifact_bytes_found(self, mock_psycopg2, tmp_path, sample_arrays) -> None:
        """get_artifact_bytes returns bytes when file exists."""
        from juniper_data.storage.postgres_store import PostgresDatasetStore

        store = PostgresDatasetStore(auto_create_schema=False, artifact_path=tmp_path / "data")
        (tmp_path / "data").mkdir(parents=True, exist_ok=True)

        artifact_path = tmp_path / "data" / "test-dataset.npz"
        buf = io.BytesIO()
        np.savez_compressed(buf, **sample_arrays)
        artifact_path.write_bytes(buf.getvalue())

        result = store.get_artifact_bytes("test-dataset")
        assert result is not None
        assert len(result) > 0

    def test_get_artifact_bytes_not_found(self, mock_psycopg2, tmp_path) -> None:
        """get_artifact_bytes returns None when file doesn't exist."""
        from juniper_data.storage.postgres_store import PostgresDatasetStore

        store = PostgresDatasetStore(auto_create_schema=False, artifact_path=tmp_path / "data")

        result = store.get_artifact_bytes("nonexistent")
        assert result is None


@pytest.mark.unit
@pytest.mark.storage
class TestPostgresDatasetStoreExists:
    """Tests for exists operation."""

    def test_exists_true(self, mock_psycopg2, tmp_path) -> None:
        """exists returns True when dataset is in DB."""
        _, _, mock_cursor = mock_psycopg2
        from juniper_data.storage.postgres_store import PostgresDatasetStore

        store = PostgresDatasetStore(auto_create_schema=False, artifact_path=tmp_path / "data")
        mock_cursor.fetchone.return_value = (1,)

        assert store.exists("test-dataset") is True

    def test_exists_false(self, mock_psycopg2, tmp_path) -> None:
        """exists returns False when dataset is not in DB."""
        _, _, mock_cursor = mock_psycopg2
        from juniper_data.storage.postgres_store import PostgresDatasetStore

        store = PostgresDatasetStore(auto_create_schema=False, artifact_path=tmp_path / "data")
        mock_cursor.fetchone.return_value = None

        assert store.exists("nonexistent") is False


@pytest.mark.unit
@pytest.mark.storage
class TestPostgresDatasetStoreDelete:
    """Tests for delete operation."""

    def test_delete_existing_with_artifact(self, mock_psycopg2, tmp_path) -> None:
        """delete removes DB row and artifact file."""
        _, _, mock_cursor = mock_psycopg2
        from juniper_data.storage.postgres_store import PostgresDatasetStore

        store = PostgresDatasetStore(auto_create_schema=False, artifact_path=tmp_path / "data")
        (tmp_path / "data").mkdir(parents=True, exist_ok=True)
        artifact_path = tmp_path / "data" / "test-dataset.npz"
        artifact_path.write_bytes(b"dummy")

        mock_cursor.fetchone.return_value = ("test-dataset",)

        result = store.delete("test-dataset")
        assert result is True
        assert not artifact_path.exists()

    def test_delete_existing_no_artifact(self, mock_psycopg2, tmp_path) -> None:
        """delete works even when artifact file is missing."""
        _, _, mock_cursor = mock_psycopg2
        from juniper_data.storage.postgres_store import PostgresDatasetStore

        store = PostgresDatasetStore(auto_create_schema=False, artifact_path=tmp_path / "data")
        mock_cursor.fetchone.return_value = ("test-dataset",)

        result = store.delete("test-dataset")
        assert result is True

    def test_delete_nonexistent(self, mock_psycopg2, tmp_path) -> None:
        """delete returns False when dataset doesn't exist."""
        _, _, mock_cursor = mock_psycopg2
        from juniper_data.storage.postgres_store import PostgresDatasetStore

        store = PostgresDatasetStore(auto_create_schema=False, artifact_path=tmp_path / "data")
        mock_cursor.fetchone.return_value = None

        result = store.delete("nonexistent")
        assert result is False


@pytest.mark.unit
@pytest.mark.storage
class TestPostgresDatasetStoreListDatasets:
    """Tests for list_datasets operation."""

    def test_list_datasets(self, mock_psycopg2, tmp_path) -> None:
        """list_datasets returns dataset IDs."""
        _, _, mock_cursor = mock_psycopg2
        from juniper_data.storage.postgres_store import PostgresDatasetStore

        store = PostgresDatasetStore(auto_create_schema=False, artifact_path=tmp_path / "data")
        mock_cursor.fetchall.return_value = [("ds-1",), ("ds-2",)]

        result = store.list_datasets()
        assert result == ["ds-1", "ds-2"]


@pytest.mark.unit
@pytest.mark.storage
class TestPostgresDatasetStoreUpdateMeta:
    """Tests for update_meta operation."""

    def test_update_meta_found(self, mock_psycopg2, tmp_path, sample_meta) -> None:
        """update_meta returns True when dataset was updated."""
        _, _, mock_cursor = mock_psycopg2
        from juniper_data.storage.postgres_store import PostgresDatasetStore

        store = PostgresDatasetStore(auto_create_schema=False, artifact_path=tmp_path / "data")
        mock_cursor.rowcount = 1

        result = store.update_meta("test-dataset", sample_meta)
        assert result is True

    def test_update_meta_not_found(self, mock_psycopg2, tmp_path, sample_meta) -> None:
        """update_meta returns False when dataset doesn't exist."""
        _, _, mock_cursor = mock_psycopg2
        from juniper_data.storage.postgres_store import PostgresDatasetStore

        store = PostgresDatasetStore(auto_create_schema=False, artifact_path=tmp_path / "data")
        mock_cursor.rowcount = 0

        result = store.update_meta("nonexistent", sample_meta)
        assert result is False


@pytest.mark.unit
@pytest.mark.storage
class TestPostgresDatasetStoreListAllMetadata:
    """Tests for list_all_metadata operation."""

    def test_list_all_metadata(self, mock_psycopg2, tmp_path) -> None:
        """list_all_metadata returns all metadata objects."""
        _, _, mock_cursor = mock_psycopg2
        from juniper_data.storage.postgres_store import PostgresDatasetStore

        store = PostgresDatasetStore(auto_create_schema=False, artifact_path=tmp_path / "data")
        mock_cursor.fetchall.return_value = [
            {
                "dataset_id": "ds-1",
                "generator": "test",
                "generator_version": "1.0.0",
                "params": {"seed": 42},
                "n_samples": 100,
                "n_features": 2,
                "n_classes": 2,
                "n_train": 80,
                "n_test": 20,
                "class_distribution": {"0": 50, "1": 50},
                "artifact_formats": ["npz"],
                "created_at": datetime(2026, 1, 1, tzinfo=UTC),
                "checksum": None,
                "dataset_name": "experiment-a",
                "dataset_version": 7,
                "parent_dataset_id": "parent-dataset",
                "description": "Regression metadata",
                "created_by": "qa-bot",
                "tags": [],
                "ttl_seconds": None,
                "expires_at": None,
                "last_accessed_at": None,
                "access_count": 0,
            }
        ]

        result = store.list_all_metadata()
        assert len(result) == 1
        assert result[0].dataset_id == "ds-1"
        assert result[0].dataset_name == "experiment-a"
        assert result[0].dataset_version == 7


@pytest.mark.unit
@pytest.mark.storage
class TestPostgresDatasetStoreClose:
    """Tests for close operation."""

    def test_close(self, mock_psycopg2, tmp_path) -> None:
        """close is a no-op that doesn't raise."""
        from juniper_data.storage.postgres_store import PostgresDatasetStore

        store = PostgresDatasetStore(auto_create_schema=False, artifact_path=tmp_path / "data")
        store.close()


@pytest.mark.unit
@pytest.mark.storage
class TestPostgresDatasetStoreArtifactFile:
    """Tests for _artifact_file helper."""

    def test_artifact_file_path(self, mock_psycopg2, tmp_path) -> None:
        """_artifact_file returns correct path."""
        from juniper_data.storage.postgres_store import PostgresDatasetStore

        store = PostgresDatasetStore(auto_create_schema=False, artifact_path=tmp_path / "data")
        path = store._artifact_file("my-dataset")
        assert path == tmp_path / "data" / "my-dataset.npz"
