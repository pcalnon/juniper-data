"""Dataset ID generation utilities.

This module provides deterministic ID generation for datasets based on
generator name, version, and parameters.
"""

import hashlib
import json
from typing import Any


def generate_dataset_id(generator: str, version: str, params: dict[str, Any]) -> str:
    """Generate a deterministic hash-based ID from generator metadata and params.

    Creates a unique, reproducible identifier for a dataset configuration by
    hashing the canonical JSON representation of the generator name, version,
    and parameters.

    Args:
        generator: Name of the generator (e.g., "spiral").
        version: Version string (e.g., "v1.0.0").
        params: Dictionary of generator parameters.

    Returns:
        Dataset ID in format "{generator}-{version}-{hash[:16]}".
        Example: "spiral-v1.0.0-a3f8e12b4c567890"
    """
    canonical_data = {
        "generator": generator,
        "version": version,
        "params": params,
    }

    canonical_json = json.dumps(canonical_data, sort_keys=True, separators=(",", ":"))

    hash_digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    return f"{generator}-{version}-{hash_digest[:16]}"
