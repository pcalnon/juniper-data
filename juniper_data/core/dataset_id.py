"""Dataset ID generation utilities.

This module provides deterministic ID generation for datasets based on
generator name, version, and parameters. When ``params['seed']`` is ``None``
the ID incorporates a per-call nonce so that repeated non-deterministic
generation requests do not collide on a cached result (BUG-JD-04).
"""

import hashlib
import json
import uuid
from typing import Any

from juniper_data.core.constants import CHARSET_UTF8, DATASET_ID_HASH_PREFIX_LENGTH

# BUG-JD-04: Length of the UUID-derived nonce included in the ID hash when
# ``params['seed']`` is missing or ``None``. 8 hex chars = 32 bits, which is
# ample to avoid cache collisions in realistic request volumes while keeping
# the canonical-JSON payload compact.
_DATASET_ID_NONCE_LENGTH = 8


def generate_dataset_id(generator: str, version: str, params: dict[str, Any]) -> str:
    """Generate a hash-based ID from generator metadata and params.

    Creates a reproducible identifier for a dataset configuration by hashing
    the canonical JSON representation of the generator name, version, and
    parameters.

    When ``params['seed']`` is present and not ``None``, the ID is fully
    deterministic: identical inputs produce identical IDs, enabling result
    caching. When the seed is absent or ``None``, generation is itself
    non-deterministic and the ID is mixed with a short UUID nonce so distinct
    requests do not collide on a stale cached artifact.

    Args:
        generator: Name of the generator (e.g., "spiral").
        version: Version string (e.g., "v1.0.0").
        params: Dictionary of generator parameters. ``seed`` is treated as the
            determinism marker.

    Returns:
        Dataset ID in format "{generator}-{version}-{hash[:16]}".
        Example: "spiral-v1.0.0-a3f8e12b4c567890"
    """
    canonical_data: dict[str, Any] = {
        "generator": generator,
        "version": version,
        "params": params,
    }

    # BUG-JD-04: Seedless requests are non-deterministic; add a nonce so the
    # hashed ID cannot collide with a previous (now-stale) seedless artifact.
    if params.get("seed") is None:
        canonical_data["_nonce"] = uuid.uuid4().hex[:_DATASET_ID_NONCE_LENGTH]

    canonical_json = json.dumps(canonical_data, sort_keys=True, separators=(",", ":"))

    hash_digest = hashlib.sha256(canonical_json.encode(CHARSET_UTF8)).hexdigest()

    return f"{generator}-{version}-{hash_digest[:DATASET_ID_HASH_PREFIX_LENGTH]}"
