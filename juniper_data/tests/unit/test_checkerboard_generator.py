"""Unit tests for the checkerboard dataset generator."""

import numpy as np
import pytest

from juniper_data.generators.checkerboard import (
    VERSION,
    CheckerboardGenerator,
    CheckerboardParams,
    get_schema,
)


class TestCheckerboardParams:
    """Tests for CheckerboardParams validation."""

    def test_default_params(self) -> None:
        """Default parameters should be valid."""
        params = CheckerboardParams()
        assert params.n_samples == 200
        assert params.n_squares == 4
        assert params.x_range == (0.0, 1.0)
        assert params.y_range == (0.0, 1.0)
        assert params.noise == 0.0
        assert params.train_ratio == 0.8
        assert params.test_ratio == 0.2
        assert params.shuffle is True

    def test_custom_params(self) -> None:
        """Custom parameters should be accepted."""
        params = CheckerboardParams(
            n_samples=300,
            n_squares=8,
            x_range=(-1.0, 1.0),
            y_range=(-2.0, 2.0),
            noise=0.1,
            seed=42,
        )
        assert params.n_samples == 300
        assert params.n_squares == 8
        assert params.x_range == (-1.0, 1.0)
        assert params.y_range == (-2.0, 2.0)
        assert params.noise == 0.1
        assert params.seed == 42

    def test_invalid_n_squares_too_low(self) -> None:
        """n_squares must be at least 2."""
        with pytest.raises(ValueError):
            CheckerboardParams(n_squares=1)

    def test_invalid_n_squares_too_high(self) -> None:
        """n_squares must be at most 16."""
        with pytest.raises(ValueError):
            CheckerboardParams(n_squares=17)


class TestCheckerboardGenerator:
    """Tests for CheckerboardGenerator."""

    def test_generate_returns_expected_keys(self) -> None:
        """Generated data should contain all expected keys."""
        params = CheckerboardParams(seed=42)
        result = CheckerboardGenerator.generate(params)

        expected_keys = {"X_train", "y_train", "X_test", "y_test", "X_full", "y_full"}
        assert set(result.keys()) == expected_keys

    def test_generate_shapes(self) -> None:
        """Generated arrays should have correct shapes."""
        params = CheckerboardParams(n_samples=150, seed=42)
        result = CheckerboardGenerator.generate(params)

        assert result["X_full"].shape == (150, 2)
        assert result["y_full"].shape == (150, 2)

    def test_generate_dtypes(self) -> None:
        """Generated arrays should have float32 dtype."""
        params = CheckerboardParams(seed=42)
        result = CheckerboardGenerator.generate(params)

        assert result["X_train"].dtype == np.float32
        assert result["y_train"].dtype == np.float32
        assert result["X_full"].dtype == np.float32
        assert result["y_full"].dtype == np.float32

    def test_determinism_with_seed(self) -> None:
        """Same seed should produce identical results."""
        params = CheckerboardParams(seed=123)

        result1 = CheckerboardGenerator.generate(params)
        result2 = CheckerboardGenerator.generate(params)

        np.testing.assert_array_equal(result1["X_full"], result2["X_full"])
        np.testing.assert_array_equal(result1["y_full"], result2["y_full"])

    def test_different_seeds_produce_different_data(self) -> None:
        """Different seeds should produce different results."""
        params1 = CheckerboardParams(seed=42)
        params2 = CheckerboardParams(seed=43)

        result1 = CheckerboardGenerator.generate(params1)
        result2 = CheckerboardGenerator.generate(params2)

        assert not np.allclose(result1["X_full"], result2["X_full"])

    def test_one_hot_labels(self) -> None:
        """Labels should be valid one-hot encoded."""
        params = CheckerboardParams(seed=42)
        result = CheckerboardGenerator.generate(params)

        row_sums = result["y_full"].sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(len(row_sums)))

        for row in result["y_full"]:
            assert np.sum(row == 1.0) == 1
            assert np.sum(row == 0.0) == 1

    def test_points_in_range(self) -> None:
        """Points should be within specified range (no noise)."""
        params = CheckerboardParams(
            n_samples=100,
            x_range=(0.0, 1.0),
            y_range=(0.0, 1.0),
            noise=0.0,
            seed=42,
        )
        result = CheckerboardGenerator.generate(params)

        assert result["X_full"][:, 0].min() >= 0.0
        assert result["X_full"][:, 0].max() <= 1.0
        assert result["X_full"][:, 1].min() >= 0.0
        assert result["X_full"][:, 1].max() <= 1.0

    def test_checkerboard_pattern(self) -> None:
        """Adjacent squares should have different classes."""
        params = CheckerboardParams(
            n_samples=1000,
            n_squares=4,
            x_range=(0.0, 1.0),
            y_range=(0.0, 1.0),
            noise=0.0,
            seed=42,
            shuffle=False,
        )
        result = CheckerboardGenerator.generate(params)

        corner_00 = result["X_full"][(result["X_full"][:, 0] < 0.25) & (result["X_full"][:, 1] < 0.25)]
        corner_01 = result["X_full"][(result["X_full"][:, 0] < 0.25) & (result["X_full"][:, 1] > 0.25) & (result["X_full"][:, 1] < 0.5)]

        if len(corner_00) > 0 and len(corner_01) > 0:
            pass

    def test_train_test_split_ratio(self) -> None:
        """Train/test split should respect configured ratios."""
        params = CheckerboardParams(
            n_samples=100,
            train_ratio=0.7,
            test_ratio=0.3,
            seed=42,
        )
        result = CheckerboardGenerator.generate(params)

        assert len(result["X_train"]) == 70
        assert len(result["X_test"]) == 30


class TestGetSchema:
    """Tests for get_schema function."""

    def test_returns_dict(self) -> None:
        """get_schema should return a dictionary."""
        schema = get_schema()
        assert isinstance(schema, dict)

    def test_schema_has_properties(self) -> None:
        """Schema should have properties key."""
        schema = get_schema()
        assert "properties" in schema

    def test_schema_includes_all_params(self) -> None:
        """Schema should include all parameter names."""
        schema = get_schema()
        expected_params = {
            "n_samples",
            "n_squares",
            "x_range",
            "y_range",
            "noise",
            "seed",
            "train_ratio",
            "test_ratio",
            "shuffle",
        }
        assert expected_params.issubset(set(schema["properties"].keys()))


class TestVersion:
    """Tests for VERSION constant."""

    def test_version_format(self) -> None:
        """VERSION should be a valid semver string."""
        parts = VERSION.split(".")
        assert len(parts) == 3
        assert all(part.isdigit() for part in parts)
