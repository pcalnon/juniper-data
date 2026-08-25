"""Constants for the JuniperData core layer.

Centralizes hardcoded literals used by ``dataset_id.py`` and ``models.py``:
the UTF-8 encoding string (used in 7+ places across the codebase),
Pydantic field length constraints, and dataset-id formatting.

Project: Juniper
Sub-Project: juniper-data
Application: JuniperData Core
Author: Paul Calnon
Version: 0.6.0
License: MIT License
"""

# ─── Encoding ────────────────────────────────────────────────────────────────

# Used by dataset_id hashing, redis_store JSON encoding, csv_import file
# reads, and arc_agi file reads.
CHARSET_UTF8: str = "utf-8"

# ─── Hashing / Dataset ID ────────────────────────────────────────────────────

# Length of the SHA-256 hash prefix used in generated dataset IDs.
# ``f"{generator}-{version}-{hash_digest[:16]}"``
DATASET_ID_HASH_PREFIX_LENGTH: int = 16

# ─── Pydantic Field Length Constraints (core/models.py) ──────────────────────

DESCRIPTION_MAX_LENGTH: int = 500
CREATED_BY_MAX_LENGTH: int = 100

# Batch operation size constraints (BatchDeleteRequest, BatchCreateRequest, etc.)
BATCH_DELETE_MAX_ITEMS: int = 100
BATCH_CREATE_MAX_ITEMS: int = 50
BATCH_UPDATE_TAGS_MAX_ITEMS: int = 100
BATCH_EXPORT_MAX_ITEMS: int = 50
BATCH_MIN_ITEMS: int = 1

# ``tags_match`` default and pattern for the datasets ``/filter`` route (APD-DATA-021: the
# route is the contract of record; the never-wired ``DatasetListFilter`` model that used
# to share these was removed, and ``tests/unit/test_filter_contract.py`` pins the route
# to them at the call site).
TAGS_MATCH_DEFAULT: str = "any"
TAGS_MATCH_PATTERN: str = "^(any|all)$"
