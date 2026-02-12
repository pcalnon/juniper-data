"""Unit tests for RedisDatasetStore."""

from datetime import datetime, timezone
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
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_meta_with_ttl() -> DatasetMeta:
    """Create sample metadata with TTL."""
    return DatasetMeta(
        dataset_id="test-ttl",
        generator="test",
        generator_version="1.0.0",
        params={"seed": 42},
        n_samples=100,
        n_features=2,
        n_classes=2,
        n_train=80,
        n_test=20,
        class_distribution={"0": 50, "1": 50},
        created_at=datetime.now(timezone.utc),
        ttl_seconds=3600,
    )


@pytest.fixture
def sample_arrays() -> dict[str, np.ndarray]:
    """Create sample arrays."""
    return {
        "X_train": np.random.randn(80, 2).astype(np.float32),
        "y_train": np.random.randn(80, 2).astype(np.float32),
        "X_test": np.random.randn(20, 2).astype(np.float32),
        "y_test": np.random.randn(20, 2).astype(np.float32),
    }


@pytest.fixture
def mock_redis_module():
    """Create a mock redis module and patch it into redis_store."""
    mock_redis = MagicMock()
    mock_client = MagicMock()
    mock_redis.Redis.return_value = mock_client
    mock_pipeline = MagicMock()
    mock_client.pipeline.return_value = mock_pipeline

    with patch.dict("sys.modules", {"redis": mock_redis}):
        with patch("juniper_data.storage.redis_store.REDIS_AVAILABLE", True):
            with patch("juniper_data.storage.redis_store.redis", mock_redis):
                yield mock_redis, mock_client, mock_pipeline


@pytest.mark.unit
@pytest.mark.storage
class TestRedisDatasetStoreInit:
    """Tests for RedisDatasetStore initialization."""

    def test_init_default_params(self, mock_redis_module) -> None:
        """Initialize with default parameters."""
        mock_redis, mock_client, _ = mock_redis_module
        from juniper_data.storage.redis_store import RedisDatasetStore

        store = RedisDatasetStore()
        mock_redis.Redis.assert_called_once_with(host="localhost", port=6379, db=0, password=None, decode_responses=False)
        assert store._key_prefix == "juniper:dataset:"
        assert store._default_ttl is None

    def test_init_custom_params(self, mock_redis_module) -> None:
        """Initialize with custom parameters."""
        mock_redis, _, _ = mock_redis_module
        from juniper_data.storage.redis_store import RedisDatasetStore

        store = RedisDatasetStore(host="redis.example.com", port=6380, db=1, password="secret", key_prefix="myapp:", default_ttl=600)
        mock_redis.Redis.assert_called_once_with(host="redis.example.com", port=6380, db=1, password="secret", decode_responses=False)
        assert store._key_prefix == "myapp:"
        assert store._default_ttl == 600

    def test_init_with_connection_pool(self, mock_redis_module) -> None:
        """Initialize with an existing connection pool."""
        mock_redis, _, _ = mock_redis_module
        from juniper_data.storage.redis_store import RedisDatasetStore

        mock_pool = MagicMock()
        _store = RedisDatasetStore(connection_pool=mock_pool)
        mock_redis.Redis.assert_called_once_with(connection_pool=mock_pool)

    def test_init_raises_without_redis(self) -> None:
        """Raises ImportError when redis is not available."""
        with patch("juniper_data.storage.redis_store.REDIS_AVAILABLE", False):
            from juniper_data.storage.redis_store import RedisDatasetStore

            with pytest.raises(ImportError, match="Redis package not installed"):
                RedisDatasetStore()


@pytest.mark.unit
@pytest.mark.storage
class TestRedisDatasetStoreKeys:
    """Tests for Redis key generation."""

    def test_meta_key(self, mock_redis_module) -> None:
        """Meta key includes prefix and :meta suffix."""
        from juniper_data.storage.redis_store import RedisDatasetStore

        store = RedisDatasetStore(key_prefix="test:")
        assert store._meta_key("ds-123") == "test:ds-123:meta"

    def test_artifact_key(self, mock_redis_module) -> None:
        """Artifact key includes prefix and :artifact suffix."""
        from juniper_data.storage.redis_store import RedisDatasetStore

        store = RedisDatasetStore(key_prefix="test:")
        assert store._artifact_key("ds-123") == "test:ds-123:artifact"


@pytest.mark.unit
@pytest.mark.storage
class TestRedisDatasetStoreEncoding:
    """Tests for metadata and array encoding/decoding."""

    def test_encode_decode_meta_roundtrip(self, mock_redis_module, sample_meta) -> None:
        """Encoding then decoding metadata preserves data."""
        from juniper_data.storage.redis_store import RedisDatasetStore

        store = RedisDatasetStore()
        encoded = store._encode_meta(sample_meta)
        assert isinstance(encoded, bytes)
        decoded = store._decode_meta(encoded)
        assert decoded.dataset_id == sample_meta.dataset_id
        assert decoded.generator == sample_meta.generator
        assert decoded.n_samples == sample_meta.n_samples

    def test_encode_arrays_returns_bytes(self, mock_redis_module, sample_arrays) -> None:
        """Encoding arrays returns bytes (NPZ format)."""
        from juniper_data.storage.redis_store import RedisDatasetStore

        store = RedisDatasetStore()
        encoded = store._encode_arrays(sample_arrays)
        assert isinstance(encoded, bytes)
        assert len(encoded) > 0


@pytest.mark.unit
@pytest.mark.storage
class TestRedisDatasetStoreSave:
    """Tests for save operation."""

    def test_save_without_ttl(self, mock_redis_module, sample_meta, sample_arrays) -> None:
        """Save without TTL uses SET."""
        _, mock_client, mock_pipeline = mock_redis_module
        from juniper_data.storage.redis_store import RedisDatasetStore

        store = RedisDatasetStore()
        store.save("ds-1", sample_meta, sample_arrays)

        mock_client.pipeline.assert_called_once()
        assert mock_pipeline.set.call_count == 2
        assert mock_pipeline.setex.call_count == 0
        mock_pipeline.execute.assert_called_once()

    def test_save_with_meta_ttl(self, mock_redis_module, sample_meta_with_ttl, sample_arrays) -> None:
        """Save with metadata TTL uses SETEX."""
        _, mock_client, mock_pipeline = mock_redis_module
        from juniper_data.storage.redis_store import RedisDatasetStore

        store = RedisDatasetStore()
        store.save("ds-1", sample_meta_with_ttl, sample_arrays)

        assert mock_pipeline.setex.call_count == 2
        assert mock_pipeline.set.call_count == 0
        mock_pipeline.execute.assert_called_once()

    def test_save_with_default_ttl(self, mock_redis_module, sample_meta, sample_arrays) -> None:
        """Save with store default TTL uses SETEX."""
        _, mock_client, mock_pipeline = mock_redis_module
        from juniper_data.storage.redis_store import RedisDatasetStore

        store = RedisDatasetStore(default_ttl=300)
        store.save("ds-1", sample_meta, sample_arrays)

        assert mock_pipeline.setex.call_count == 2
        mock_pipeline.execute.assert_called_once()


@pytest.mark.unit
@pytest.mark.storage
class TestRedisDatasetStoreGetMeta:
    """Tests for get_meta operation."""

    def test_get_meta_found(self, mock_redis_module, sample_meta) -> None:
        """get_meta returns metadata when found."""
        _, mock_client, _ = mock_redis_module
        from juniper_data.storage.redis_store import RedisDatasetStore

        store = RedisDatasetStore()
        encoded = store._encode_meta(sample_meta)
        mock_client.get.return_value = encoded

        result = store.get_meta("ds-1")
        assert result is not None
        assert result.dataset_id == sample_meta.dataset_id

    def test_get_meta_not_found(self, mock_redis_module) -> None:
        """get_meta returns None when not found."""
        _, mock_client, _ = mock_redis_module
        from juniper_data.storage.redis_store import RedisDatasetStore

        store = RedisDatasetStore()
        mock_client.get.return_value = None

        result = store.get_meta("nonexistent")
        assert result is None


@pytest.mark.unit
@pytest.mark.storage
class TestRedisDatasetStoreGetArtifact:
    """Tests for get_artifact_bytes operation."""

    def test_get_artifact_bytes_found(self, mock_redis_module) -> None:
        """get_artifact_bytes returns bytes when found."""
        _, mock_client, _ = mock_redis_module
        from juniper_data.storage.redis_store import RedisDatasetStore

        store = RedisDatasetStore()
        mock_client.get.return_value = b"npz-data-bytes"

        result = store.get_artifact_bytes("ds-1")
        assert result == b"npz-data-bytes"

    def test_get_artifact_bytes_not_found(self, mock_redis_module) -> None:
        """get_artifact_bytes returns None when not found."""
        _, mock_client, _ = mock_redis_module
        from juniper_data.storage.redis_store import RedisDatasetStore

        store = RedisDatasetStore()
        mock_client.get.return_value = None

        result = store.get_artifact_bytes("nonexistent")
        assert result is None


@pytest.mark.unit
@pytest.mark.storage
class TestRedisDatasetStoreExists:
    """Tests for exists operation."""

    def test_exists_true(self, mock_redis_module) -> None:
        """exists returns True when key exists."""
        _, mock_client, _ = mock_redis_module
        from juniper_data.storage.redis_store import RedisDatasetStore

        store = RedisDatasetStore()
        mock_client.exists.return_value = 1

        assert store.exists("ds-1") is True

    def test_exists_false(self, mock_redis_module) -> None:
        """exists returns False when key doesn't exist."""
        _, mock_client, _ = mock_redis_module
        from juniper_data.storage.redis_store import RedisDatasetStore

        store = RedisDatasetStore()
        mock_client.exists.return_value = 0

        assert store.exists("ds-1") is False


@pytest.mark.unit
@pytest.mark.storage
class TestRedisDatasetStoreDelete:
    """Tests for delete operation."""

    def test_delete_existing(self, mock_redis_module) -> None:
        """delete returns True when keys were deleted."""
        _, mock_client, _ = mock_redis_module
        from juniper_data.storage.redis_store import RedisDatasetStore

        store = RedisDatasetStore()
        mock_client.delete.return_value = 2

        assert store.delete("ds-1") is True

    def test_delete_nonexistent(self, mock_redis_module) -> None:
        """delete returns False when no keys were deleted."""
        _, mock_client, _ = mock_redis_module
        from juniper_data.storage.redis_store import RedisDatasetStore

        store = RedisDatasetStore()
        mock_client.delete.return_value = 0

        assert store.delete("nonexistent") is False


@pytest.mark.unit
@pytest.mark.storage
class TestRedisDatasetStoreListDatasets:
    """Tests for list_datasets operation."""

    def test_list_datasets(self, mock_redis_module) -> None:
        """list_datasets returns sorted dataset IDs."""
        _, mock_client, _ = mock_redis_module
        from juniper_data.storage.redis_store import RedisDatasetStore

        store = RedisDatasetStore()
        mock_client.scan_iter.return_value = [
            b"juniper:dataset:ds-b:meta",
            b"juniper:dataset:ds-a:meta",
            b"juniper:dataset:ds-c:meta",
        ]

        result = store.list_datasets()
        assert result == ["ds-a", "ds-b", "ds-c"]

    def test_list_datasets_with_limit_and_offset(self, mock_redis_module) -> None:
        """list_datasets respects limit and offset."""
        _, mock_client, _ = mock_redis_module
        from juniper_data.storage.redis_store import RedisDatasetStore

        store = RedisDatasetStore()
        mock_client.scan_iter.return_value = [
            b"juniper:dataset:ds-a:meta",
            b"juniper:dataset:ds-b:meta",
            b"juniper:dataset:ds-c:meta",
        ]

        result = store.list_datasets(limit=1, offset=1)
        assert result == ["ds-b"]

    def test_list_datasets_with_string_keys(self, mock_redis_module) -> None:
        """list_datasets handles string keys (non-bytes)."""
        _, mock_client, _ = mock_redis_module
        from juniper_data.storage.redis_store import RedisDatasetStore

        store = RedisDatasetStore()
        mock_client.scan_iter.return_value = [
            "juniper:dataset:ds-a:meta",
        ]

        result = store.list_datasets()
        assert result == ["ds-a"]


@pytest.mark.unit
@pytest.mark.storage
class TestRedisDatasetStoreUpdateMeta:
    """Tests for update_meta operation."""

    def test_update_meta_found_with_ttl(self, mock_redis_module, sample_meta) -> None:
        """update_meta updates and preserves TTL when positive."""
        _, mock_client, _ = mock_redis_module
        from juniper_data.storage.redis_store import RedisDatasetStore

        store = RedisDatasetStore()
        mock_client.exists.return_value = True
        mock_client.ttl.return_value = 300

        result = store.update_meta("ds-1", sample_meta)
        assert result is True
        mock_client.setex.assert_called_once()

    def test_update_meta_found_no_ttl(self, mock_redis_module, sample_meta) -> None:
        """update_meta uses SET when no TTL."""
        _, mock_client, _ = mock_redis_module
        from juniper_data.storage.redis_store import RedisDatasetStore

        store = RedisDatasetStore()
        mock_client.exists.return_value = True
        mock_client.ttl.return_value = -1

        result = store.update_meta("ds-1", sample_meta)
        assert result is True
        mock_client.set.assert_called_once()

    def test_update_meta_not_found(self, mock_redis_module, sample_meta) -> None:
        """update_meta returns False when dataset doesn't exist."""
        _, mock_client, _ = mock_redis_module
        from juniper_data.storage.redis_store import RedisDatasetStore

        store = RedisDatasetStore()
        mock_client.exists.return_value = False

        result = store.update_meta("nonexistent", sample_meta)
        assert result is False


@pytest.mark.unit
@pytest.mark.storage
class TestRedisDatasetStoreListAllMetadata:
    """Tests for list_all_metadata operation."""

    def test_list_all_metadata(self, mock_redis_module, sample_meta) -> None:
        """list_all_metadata returns all metadata objects."""
        _, mock_client, _ = mock_redis_module
        from juniper_data.storage.redis_store import RedisDatasetStore

        store = RedisDatasetStore()
        encoded = store._encode_meta(sample_meta)
        mock_client.scan_iter.return_value = [b"juniper:dataset:ds-1:meta"]
        mock_client.get.return_value = encoded

        result = store.list_all_metadata()
        assert len(result) == 1
        assert result[0].dataset_id == sample_meta.dataset_id

    def test_list_all_metadata_skips_none(self, mock_redis_module) -> None:
        """list_all_metadata skips keys with None data."""
        _, mock_client, _ = mock_redis_module
        from juniper_data.storage.redis_store import RedisDatasetStore

        store = RedisDatasetStore()
        mock_client.scan_iter.return_value = [b"juniper:dataset:ds-1:meta"]
        mock_client.get.return_value = None

        result = store.list_all_metadata()
        assert len(result) == 0


@pytest.mark.unit
@pytest.mark.storage
class TestRedisDatasetStorePing:
    """Tests for ping operation."""

    def test_ping_success(self, mock_redis_module) -> None:
        """ping returns True when Redis is connected."""
        _, mock_client, _ = mock_redis_module
        from juniper_data.storage.redis_store import RedisDatasetStore

        store = RedisDatasetStore()
        mock_client.ping.return_value = True

        assert store.ping() is True

    def test_ping_failure(self, mock_redis_module) -> None:
        """ping returns False when connection fails."""
        _, mock_client, _ = mock_redis_module
        from juniper_data.storage.redis_store import RedisDatasetStore

        store = RedisDatasetStore()
        mock_client.ping.side_effect = ConnectionError("Connection refused")

        assert store.ping() is False


@pytest.mark.unit
@pytest.mark.storage
class TestRedisDatasetStoreFlush:
    """Tests for flush_prefix operation."""

    def test_flush_prefix_with_keys(self, mock_redis_module) -> None:
        """flush_prefix deletes matching keys."""
        _, mock_client, _ = mock_redis_module
        from juniper_data.storage.redis_store import RedisDatasetStore

        store = RedisDatasetStore()
        mock_client.scan_iter.return_value = [b"key1", b"key2"]
        mock_client.delete.return_value = 2

        result = store.flush_prefix()
        assert result == 2
        mock_client.delete.assert_called_once_with(b"key1", b"key2")

    def test_flush_prefix_no_keys(self, mock_redis_module) -> None:
        """flush_prefix returns 0 when no matching keys."""
        _, mock_client, _ = mock_redis_module
        from juniper_data.storage.redis_store import RedisDatasetStore

        store = RedisDatasetStore()
        mock_client.scan_iter.return_value = []

        result = store.flush_prefix()
        assert result == 0
