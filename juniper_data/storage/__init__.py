"""Storage module for dataset persistence."""

from typing import TYPE_CHECKING

from juniper_data.storage.base import DatasetStore
from juniper_data.storage.cached import CachedDatasetStore
from juniper_data.storage.local_fs import LocalFSDatasetStore
from juniper_data.storage.memory import InMemoryDatasetStore

if TYPE_CHECKING:
    from juniper_data.storage.redis_store import RedisDatasetStore
    from juniper_data.storage.hf_store import HuggingFaceDatasetStore
    from juniper_data.storage.postgres_store import PostgresDatasetStore
    from juniper_data.storage.kaggle_store import KaggleDatasetStore
else:
    try:
        from juniper_data.storage.redis_store import RedisDatasetStore
    except ImportError:
        RedisDatasetStore = None

    try:
        from juniper_data.storage.hf_store import HuggingFaceDatasetStore
    except ImportError:
        HuggingFaceDatasetStore = None

    try:
        from juniper_data.storage.postgres_store import PostgresDatasetStore
    except ImportError:
        PostgresDatasetStore = None

    try:
        from juniper_data.storage.kaggle_store import KaggleDatasetStore
    except ImportError:
        KaggleDatasetStore = None
__all__ = [
    "DatasetStore",
    "CachedDatasetStore",
    "LocalFSDatasetStore",
    "InMemoryDatasetStore",
]

if "RedisDatasetStore" in globals() and RedisDatasetStore is not None:
    __all__.append("RedisDatasetStore")

if "HuggingFaceDatasetStore" in globals() and HuggingFaceDatasetStore is not None:
    __all__.append("HuggingFaceDatasetStore")

if "PostgresDatasetStore" in globals() and PostgresDatasetStore is not None:
    __all__.append("PostgresDatasetStore")

if "KaggleDatasetStore" in globals() and KaggleDatasetStore is not None:
    __all__.append("KaggleDatasetStore")


def get_redis_store(**kwargs) -> "RedisDatasetStore":  # type: ignore[no-untyped-def]
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


def get_hf_store(**kwargs) -> "HuggingFaceDatasetStore":  # type: ignore[no-untyped-def]
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


def get_postgres_store(**kwargs) -> "PostgresDatasetStore":  # type: ignore[no-untyped-def]
    """Get a PostgreSQL dataset store (requires psycopg2 package).

    Args:
        **kwargs: Arguments passed to PostgresDatasetStore.

    Returns:
        PostgresDatasetStore instance.

    Raises:
        ImportError: If psycopg2 package is not installed.
    """
    from juniper_data.storage.postgres_store import PostgresDatasetStore

    return PostgresDatasetStore(**kwargs)


def get_kaggle_store(**kwargs) -> "KaggleDatasetStore":  # type: ignore[no-untyped-def]
    """Get a Kaggle dataset store (requires kaggle package).

    Args:
        **kwargs: Arguments passed to KaggleDatasetStore.

    Returns:
        KaggleDatasetStore instance.

    Raises:
        ImportError: If kaggle package is not installed.
    """
    from juniper_data.storage.kaggle_store import KaggleDatasetStore

    return KaggleDatasetStore(**kwargs)
