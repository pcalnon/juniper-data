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

import os

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

# ─── Generator seeding (juniper-data#319) ────────────────────────────────────

# Nine generators previously declared ``seed: int | None = Field(default=None)`` and were
# therefore NOT REPRODUCIBLE at their documented defaults: two identical calls produced
# different data, and ``shuffle_and_split`` re-drew the partition boundaries from OS
# entropy as well (``core/split.py``). ``spiral`` was the only 2-D generator with a
# concrete default, and its value (42) is adopted here.
#
# The default resolves from the environment AT IMPORT TIME, falling back to the constant,
# so a deployment can pin it without editing code. Import-time resolution rather than a
# pydantic ``default_factory`` is deliberate: it keeps the value visible as a concrete
# ``default`` in ``model_json_schema()``, which published clients read. The cost is that a
# mid-process env change has no effect -- normal for configuration.
#
# Passing ``seed=None`` EXPLICITLY remains legal and still opts into non-determinism, and
# still receives the BUG-JD-04 cache nonce in ``dataset_id.generate_dataset_id``. Only the
# DEFAULT changed.
DEFAULT_GENERATOR_SEED_FALLBACK: int = 42
DEFAULT_GENERATOR_SEED_ENV_VAR: str = "JUNIPER_DATA_DEFAULT_GENERATOR_SEED"

# ``mackey_glass`` is deterministic unless this is > 0 -- its seed is consumed only inside
# ``if params.init_noise_std > 0`` -- so this is the knob that decides whether that
# generator's ``seed`` has any effect at all. Exposed the same way for the same reason.
DEFAULT_MACKEY_GLASS_INIT_NOISE_STD_FALLBACK: float = 0.0
DEFAULT_MACKEY_GLASS_INIT_NOISE_STD_ENV_VAR: str = "JUNIPER_DATA_DEFAULT_MACKEY_GLASS_INIT_NOISE_STD"


def _resolve_env_number(env_var: str, fallback: float, cast: type) -> float:
    """Read a non-negative number from the environment, falling back on any problem.

    Never raises. A malformed, negative or empty override falls back to the compiled-in
    default rather than breaking every generator at import time -- a configuration error
    must not make the package unimportable.
    """
    raw = os.environ.get(env_var)
    if raw is None or not raw.strip():
        return fallback
    try:
        value = cast(raw.strip())
    except (TypeError, ValueError):
        return fallback
    return value if value >= 0 else fallback


DEFAULT_GENERATOR_SEED: int = int(_resolve_env_number(DEFAULT_GENERATOR_SEED_ENV_VAR, DEFAULT_GENERATOR_SEED_FALLBACK, int))
DEFAULT_MACKEY_GLASS_INIT_NOISE_STD: float = float(_resolve_env_number(DEFAULT_MACKEY_GLASS_INIT_NOISE_STD_ENV_VAR, DEFAULT_MACKEY_GLASS_INIT_NOISE_STD_FALLBACK, float))
