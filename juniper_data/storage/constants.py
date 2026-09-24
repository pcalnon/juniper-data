"""Constants for the JuniperData storage layer.

Centralizes hardcoded literals used by ``local_fs.py``, ``redis_store.py``,
and ``postgres_store.py``: connection defaults, file suffixes, key prefixes,
and pagination defaults.

Project: Juniper
Sub-Project: juniper-data
Application: JuniperData Storage
Author: Paul Calnon
Version: 0.6.0
License: MIT License
"""

# ─── Redis Defaults ──────────────────────────────────────────────────────────

REDIS_DEFAULT_HOST: str = "localhost"
REDIS_DEFAULT_PORT: int = 6379
REDIS_DEFAULT_DB: int = 0
REDIS_DEFAULT_KEY_PREFIX: str = "juniper:dataset:"
REDIS_META_KEY_SUFFIX: str = ":meta"
REDIS_ARTIFACT_KEY_SUFFIX: str = ":artifact"

# ─── PostgreSQL Defaults ─────────────────────────────────────────────────────

POSTGRES_DEFAULT_HOST: str = "localhost"
POSTGRES_DEFAULT_PORT: int = 5432
POSTGRES_DATASETS_TABLE: str = "datasets"

# ─── Local Filesystem Suffixes ───────────────────────────────────────────────

META_FILE_SUFFIX: str = ".meta.json"
NPZ_FILE_SUFFIX: str = ".npz"
TMP_FILE_SUFFIX: str = ".tmp"
# APD-DATA-007: advisory cross-process lock guarding a dataset's metadata
# read-modify-write. A lock file's name never ends in ``.meta.json``, so it is
# invisible to the ``*.meta.json`` globs that enumerate datasets.
LOCK_FILE_SUFFIX: str = ".lock"
# The lock files are a FIXED set of stripes in this subdirectory of the storage root,
# created when the store is: a dataset locks stripe ``sha256(id)`` mod
# ``LOCK_STRIPE_COUNT``. A lock file per dataset id was created on first use and never
# removed, so every request naming an absent id left one behind, and a delete needed a
# free inode before it could free anything.
LOCK_DIR_NAME: str = "locks"
# 16: in-process writers are already serialised by ``DatasetStore._version_lock``, so a
# stripe is contended only between worker processes, and the service runs one. Each
# storage directory costs 17 inodes (the directory and 16 files); a test suite builds
# hundreds of stores, which at 256 stripes would cost a hundred thousand.
LOCK_STRIPE_COUNT: int = 16

# ─── JSON Serialization ──────────────────────────────────────────────────────

JSON_INDENT_DEFAULT: int = 2

# ─── Pagination Defaults ─────────────────────────────────────────────────────

DEFAULT_LIST_LIMIT: int = 100
DEFAULT_LIST_OFFSET: int = 0

# ─── Default Artifact Format ─────────────────────────────────────────────────

DEFAULT_ARTIFACT_FORMAT: str = "npz"

# ─── Artifact Streaming ──────────────────────────────────────────────────────

# Chunk size for ``DatasetStore.open_artifact_stream`` (defect-register
# APD-DATA-016). 1 MiB: large enough that per-chunk overhead stays negligible on
# a multi-hundred-MB NPZ, small enough that peak resident bytes per concurrent
# download is bounded by a CONSTANT rather than by artifact size -- which is the
# whole point of the method. Backends overriding the default should honour it.
ARTIFACT_STREAM_CHUNK_SIZE: int = 1024 * 1024
