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
# read-modify-write. Appended AFTER ``META_FILE_SUFFIX`` so the resulting name does
# not end in ``.meta.json`` and is therefore invisible to the ``*.meta.json`` globs
# that enumerate datasets.
LOCK_FILE_SUFFIX: str = ".lock"

# ─── JSON Serialization ──────────────────────────────────────────────────────

JSON_INDENT_DEFAULT: int = 2

# ─── Pagination Defaults ─────────────────────────────────────────────────────

DEFAULT_LIST_LIMIT: int = 100
DEFAULT_LIST_OFFSET: int = 0

# ─── Default Artifact Format ─────────────────────────────────────────────────

DEFAULT_ARTIFACT_FORMAT: str = "npz"
