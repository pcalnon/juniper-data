"""Unit tests for dataset ID generation.

Tests cover:
- Deterministic ID generation
- Different params produce different IDs
- ID format matches expected pattern
"""

import pytest

from juniper_data.core.dataset_id import generate_dataset_id


@pytest.mark.unit
class TestDatasetIdGeneration:
    """Tests for deterministic dataset ID generation."""

    def test_deterministic_id_generation(self) -> None:
        """Verify same inputs produce identical IDs."""
        params = {
            "n_spirals": 2,
            "n_points_per_spiral": 100,
            "seed": 42,
        }

        id1 = generate_dataset_id("spiral", "v1.0.0", params)
        id2 = generate_dataset_id("spiral", "v1.0.0", params)

        assert id1 == id2

    def test_multiple_calls_identical(self) -> None:
        """Verify multiple sequential calls produce identical IDs."""
        params = {"n_spirals": 3, "n_points": 50}

        ids = [generate_dataset_id("spiral", "v1.0.0", params) for _ in range(5)]

        assert all(id_ == ids[0] for id_ in ids)

    def test_different_params_produce_different_ids(self) -> None:
        """Verify different params produce different IDs."""
        params1 = {"n_spirals": 2, "seed": 42}
        params2 = {"n_spirals": 3, "seed": 42}

        id1 = generate_dataset_id("spiral", "v1.0.0", params1)
        id2 = generate_dataset_id("spiral", "v1.0.0", params2)

        assert id1 != id2

    def test_different_generator_produces_different_id(self) -> None:
        """Verify different generator names produce different IDs."""
        params = {"n_spirals": 2}

        id1 = generate_dataset_id("spiral", "v1.0.0", params)
        id2 = generate_dataset_id("circle", "v1.0.0", params)

        assert id1 != id2

    def test_different_version_produces_different_id(self) -> None:
        """Verify different versions produce different IDs."""
        params = {"n_spirals": 2}

        id1 = generate_dataset_id("spiral", "v1.0.0", params)
        id2 = generate_dataset_id("spiral", "v2.0.0", params)

        assert id1 != id2


@pytest.mark.unit
class TestDatasetIdFormat:
    """Tests for dataset ID format validation."""

    def test_id_format_matches_pattern(self) -> None:
        """Verify ID format is '{generator}-{version}-{hash[:16]}'."""
        params = {"n_spirals": 2}

        dataset_id = generate_dataset_id("spiral", "v1.0.0", params)

        assert dataset_id.startswith("spiral-v1.0.0-")
        parts = dataset_id.split("-")
        assert len(parts) == 3
        assert parts[0] == "spiral"
        assert parts[1] == "v1.0.0"
        assert len(parts[2]) == 16

    def test_hash_is_hex(self) -> None:
        """Verify hash portion is valid hexadecimal."""
        params = {"n_spirals": 2}

        dataset_id = generate_dataset_id("spiral", "v1.0.0", params)
        hash_part = dataset_id.split("-")[-1]

        int(hash_part, 16)

    def test_id_length_consistent(self) -> None:
        """Verify ID has consistent length structure."""
        params1 = {"n_spirals": 2}
        params2 = {"n_spirals": 3, "noise": 0.5, "seed": 12345}

        id1 = generate_dataset_id("spiral", "v1.0.0", params1)
        id2 = generate_dataset_id("spiral", "v1.0.0", params2)

        hash1 = id1.split("-")[-1]
        hash2 = id2.split("-")[-1]

        assert len(hash1) == 16
        assert len(hash2) == 16


@pytest.mark.unit
class TestDatasetIdEdgeCases:
    """Tests for edge cases in dataset ID generation."""

    def test_empty_params(self) -> None:
        """Verify empty params dict works."""
        dataset_id = generate_dataset_id("spiral", "v1.0.0", {})

        assert dataset_id.startswith("spiral-v1.0.0-")
        assert len(dataset_id.split("-")[-1]) == 16

    def test_nested_params(self) -> None:
        """Verify nested params are handled correctly."""
        params = {
            "n_spirals": 2,
            "advanced": {"noise": 0.5, "scale": 1.0},
        }

        dataset_id = generate_dataset_id("spiral", "v1.0.0", params)

        assert dataset_id.startswith("spiral-v1.0.0-")

    def test_params_order_independent(self) -> None:
        """Verify param order doesn't affect ID (sorted keys)."""
        params1 = {"a": 1, "b": 2, "c": 3}
        params2 = {"c": 3, "a": 1, "b": 2}

        id1 = generate_dataset_id("spiral", "v1.0.0", params1)
        id2 = generate_dataset_id("spiral", "v1.0.0", params2)

        assert id1 == id2

    def test_float_params(self) -> None:
        """Verify float params work correctly."""
        params = {"noise": 0.25, "ratio": 0.8}

        dataset_id = generate_dataset_id("spiral", "v1.0.0", params)

        assert dataset_id.startswith("spiral-v1.0.0-")

    def test_boolean_params(self) -> None:
        """Verify boolean params work correctly."""
        params = {"clockwise": True, "shuffle": False}

        dataset_id = generate_dataset_id("spiral", "v1.0.0", params)

        assert dataset_id.startswith("spiral-v1.0.0-")

    def test_none_param_value(self) -> None:
        """Verify None param values work correctly."""
        params = {"seed": None, "n_spirals": 2}

        dataset_id = generate_dataset_id("spiral", "v1.0.0", params)

        assert dataset_id.startswith("spiral-v1.0.0-")

    def test_special_characters_in_generator_name(self) -> None:
        """Verify special characters in generator name are handled."""
        params = {"n_spirals": 2}

        dataset_id = generate_dataset_id("spiral_v2", "v1.0.0", params)

        assert dataset_id.startswith("spiral_v2-v1.0.0-")

    def test_different_float_precision_different_id(self) -> None:
        """Verify different float values produce different IDs."""
        params1 = {"noise": 0.25}
        params2 = {"noise": 0.250001}

        id1 = generate_dataset_id("spiral", "v1.0.0", params1)
        id2 = generate_dataset_id("spiral", "v1.0.0", params2)

        assert id1 != id2
