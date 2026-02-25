"""Redis-backed dataset storage for caching and distributed deployments."""

import io
import json
from datetime import datetime  # noqa: F401 - used by DatasetMeta serialization
from typing import Any

import numpy as np

from juniper_data.core.models import DatasetMeta

from .base import DatasetStore

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None  # type: ignore[assignment]


class RedisDatasetStore(DatasetStore):
    """Redis-backed dataset storage.

    Uses Redis for both metadata (as JSON) and artifact storage (as bytes).
    Suitable for caching layers and distributed deployments.

    Requires the `redis` package: pip install redis
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        key_prefix: str = "juniper:dataset:",
        default_ttl: int | None = None,
        connection_pool: Any | None = None,
    ) -> None:
        """Initialize Redis connection.

        Args:
            host: Redis server hostname.
            port: Redis server port.
            db: Redis database number.
            password: Redis password (optional).
            key_prefix: Prefix for all Redis keys.
            default_ttl: Default TTL for stored datasets in seconds (optional).
            connection_pool: Optional existing Redis connection pool.

        Raises:
            ImportError: If redis package is not installed.
        """
        if not REDIS_AVAILABLE:
            raise ImportError("Redis package not installed. Install with: pip install redis")

        self._key_prefix = key_prefix
        self._default_ttl = default_ttl

        if connection_pool:
            self._client: redis.Redis[bytes] = redis.Redis(connection_pool=connection_pool)
        else:
            self._client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=False,
            )

    def _meta_key(self, dataset_id: str) -> str:
        """Get Redis key for metadata."""
        return f"{self._key_prefix}{dataset_id}:meta"

    def _artifact_key(self, dataset_id: str) -> str:
        """Get Redis key for artifact data."""
        return f"{self._key_prefix}{dataset_id}:artifact"

    def _encode_meta(self, meta: DatasetMeta) -> bytes:
        """Encode metadata to JSON bytes."""
        data = meta.model_dump(mode="json")
        return json.dumps(data).encode("utf-8")

    def _decode_meta(self, data: bytes) -> DatasetMeta:
        """Decode metadata from JSON bytes."""
        parsed = json.loads(data.decode("utf-8"))
        return DatasetMeta(**parsed)

    def _encode_arrays(self, arrays: dict[str, np.ndarray]) -> bytes:
        """Encode arrays to NPZ bytes."""
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)  # type: ignore[arg-type]
        return buffer.getvalue()

    def save(
        self,
        dataset_id: str,
        meta: DatasetMeta,
        arrays: dict[str, np.ndarray],
    ) -> None:
        """Save dataset to Redis.

        Args:
            dataset_id: Unique identifier for the dataset.
            meta: Dataset metadata.
            arrays: Dictionary of numpy arrays.
        """
        meta_bytes = self._encode_meta(meta)
        artifact_bytes = self._encode_arrays(arrays)

        meta_key = self._meta_key(dataset_id)
        artifact_key = self._artifact_key(dataset_id)

        ttl = meta.ttl_seconds or self._default_ttl

        pipe = self._client.pipeline()
        if ttl:
            pipe.setex(meta_key, ttl, meta_bytes)
            pipe.setex(artifact_key, ttl, artifact_bytes)
        else:
            pipe.set(meta_key, meta_bytes)
            pipe.set(artifact_key, artifact_bytes)
        pipe.execute()

    def get_meta(self, dataset_id: str) -> DatasetMeta | None:
        """Get dataset metadata from Redis.

        Args:
            dataset_id: Unique identifier for the dataset.

        Returns:
            Dataset metadata if found, None otherwise.
        """
        data = self._client.get(self._meta_key(dataset_id))
        if data is None:
            return None
        return self._decode_meta(data)

    def get_artifact_bytes(self, dataset_id: str) -> bytes | None:
        """Get dataset artifact bytes from Redis.

        Args:
            dataset_id: Unique identifier for the dataset.

        Returns:
            NPZ bytes if found, None otherwise.
        """
        return self._client.get(self._artifact_key(dataset_id))

    def exists(self, dataset_id: str) -> bool:
        """Check if dataset exists in Redis.

        Args:
            dataset_id: Unique identifier for the dataset.

        Returns:
            True if the dataset exists, False otherwise.
        """
        return bool(self._client.exists(self._meta_key(dataset_id)))

    def delete(self, dataset_id: str) -> bool:
        """Delete dataset from Redis.

        Args:
            dataset_id: Unique identifier for the dataset.

        Returns:
            True if the dataset was deleted, False if it didn't exist.
        """
        meta_key = self._meta_key(dataset_id)
        artifact_key = self._artifact_key(dataset_id)

        deleted = self._client.delete(meta_key, artifact_key)
        return deleted > 0

    def list_datasets(self, limit: int = 100, offset: int = 0) -> list[str]:
        """List dataset IDs from Redis.

        Args:
            limit: Maximum number of dataset IDs to return.
            offset: Number of dataset IDs to skip.

        Returns:
            List of dataset IDs.
        """
        pattern = f"{self._key_prefix}*:meta"
        keys = list(self._client.scan_iter(match=pattern))

        dataset_ids = []
        for key in keys:
            key_str = key.decode("utf-8") if isinstance(key, bytes) else key
            dataset_id = key_str[len(self._key_prefix) : -5]
            dataset_ids.append(dataset_id)

        dataset_ids.sort()
        return dataset_ids[offset : offset + limit]

    def update_meta(self, dataset_id: str, meta: DatasetMeta) -> bool:
        """Update dataset metadata in Redis.

        Args:
            dataset_id: Unique identifier for the dataset.
            meta: Updated dataset metadata.

        Returns:
            True if the dataset was updated, False if it didn't exist.
        """
        meta_key = self._meta_key(dataset_id)
        if not self._client.exists(meta_key):
            return False

        ttl = self._client.ttl(meta_key)
        meta_bytes = self._encode_meta(meta)

        if ttl > 0:
            self._client.setex(meta_key, ttl, meta_bytes)
        else:
            self._client.set(meta_key, meta_bytes)

        return True

    def list_all_metadata(self) -> list[DatasetMeta]:
        """List all dataset metadata from Redis.

        Returns:
            List of all DatasetMeta objects.
        """
        pattern = f"{self._key_prefix}*:meta"
        keys = list(self._client.scan_iter(match=pattern))

        metadata: list[DatasetMeta] = []
        for key in keys:
            data = self._client.get(key)
            if data is not None:
                metadata.append(self._decode_meta(data))

        return metadata

    def ping(self) -> bool:
        """Check if Redis connection is alive.

        Returns:
            True if connected, False otherwise.
        """
        try:
            return self._client.ping()
        except Exception:
            return False

    def flush_prefix(self) -> int:
        """Delete all keys with this store's prefix.

        WARNING: This deletes all datasets in this store.

        Returns:
            Number of keys deleted.
        """
        pattern = f"{self._key_prefix}*"
        keys = list(self._client.scan_iter(match=pattern))
        if keys:
            return self._client.delete(*keys)
        return 0
