"""Storage module for dataset persistence."""

from juniper_data.storage.base import DatasetStore
from juniper_data.storage.cached import CachedDatasetStore
from juniper_data.storage.local_fs import LocalFSDatasetStore
from juniper_data.storage.memory import InMemoryDatasetStore

__all__ = [
    "DatasetStore",
    "CachedDatasetStore",
    "InMemoryDatasetStore",
    "LocalFSDatasetStore",
]


def get_redis_store(**kwargs):  # type: ignore[no-untyped-def]
    """Get a Redis dataset store (requires redis package).

    Args:
        **kwargs: Arguments passed to RedisDatasetStore.

    Returns:
        RedisDatasetStore instance.

    Raises:
        ImportError: If redis package is not installed.
    """
    from juniper_data.storage.redis_store import RedisDatasetStore
    return RedisDatasetStore(**kwargs)


def get_hf_store(**kwargs):  # type: ignore[no-untyped-def]
    """Get a Hugging Face dataset store (requires datasets package).

    Args:
        **kwargs: Arguments passed to HuggingFaceDatasetStore.

    Returns:
        HuggingFaceDatasetStore instance.

    Raises:
        ImportError: If datasets package is not installed.
    """
    from juniper_data.storage.hf_store import HuggingFaceDatasetStore
    return HuggingFaceDatasetStore(**kwargs)
