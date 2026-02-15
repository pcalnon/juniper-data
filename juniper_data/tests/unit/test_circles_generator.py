"""Unit tests for the concentric circles dataset generator."""

import numpy as np
import pytest

from juniper_data.generators.circles import VERSION, CirclesGenerator, CirclesParams, get_schema


class TestCirclesParams:
    """Tests for CirclesParams validation."""

    def test_default_params(self) -> None:
        """Default parameters should be valid."""
        params = CirclesParams()
        assert params.n_samples == 100
        assert params.outer_radius == 1.0
        assert params.factor == 0.5
        assert params.noise == 0.0
        assert params.inner_ratio == 0.5
        assert params.train_ratio == 0.8
        assert params.test_ratio == 0.2
        assert params.shuffle is True

    def test_custom_params(self) -> None:
        """Custom parameters should be accepted."""
        params = CirclesParams(
            n_samples=200,
            outer_radius=2.0,
            factor=0.3,
            noise=0.1,
            seed=42,
        )
        assert params.n_samples == 200
        assert params.outer_radius == 2.0
        assert params.factor == 0.3
        assert params.noise == 0.1
        assert params.seed == 42

    def test_invalid_factor_too_low(self) -> None:
        """Factor must be greater than 0."""
        with pytest.raises(ValueError):
            CirclesParams(factor=0.0)

    def test_invalid_factor_too_high(self) -> None:
        """Factor must be less than 1."""
        with pytest.raises(ValueError):
            CirclesParams(factor=1.0)

    def test_invalid_outer_radius_negative(self) -> None:
        """Outer radius must be positive."""
        with pytest.raises(ValueError):
            CirclesParams(outer_radius=-1.0)

    def test_invalid_n_samples_too_low(self) -> None:
        """n_samples must be at least 2."""
        with pytest.raises(ValueError):
            CirclesParams(n_samples=1)


class TestCirclesGenerator:
    """Tests for CirclesGenerator."""

    def test_generate_returns_expected_keys(self) -> None:
        """Generated data should contain all expected keys."""
        params = CirclesParams(seed=42)
        result = CirclesGenerator.generate(params)

        expected_keys = {"X_train", "y_train", "X_test", "y_test", "X_full", "y_full"}
        assert set(result.keys()) == expected_keys

    def test_generate_shapes(self) -> None:
        """Generated arrays should have correct shapes."""
        params = CirclesParams(n_samples=150, seed=42)
        result = CirclesGenerator.generate(params)

        assert result["X_full"].shape == (150, 2)
        assert result["y_full"].shape == (150, 2)

    def test_generate_dtypes(self) -> None:
        """Generated arrays should have float32 dtype."""
        params = CirclesParams(seed=42)
        result = CirclesGenerator.generate(params)

        assert result["X_train"].dtype == np.float32
        assert result["y_train"].dtype == np.float32
        assert result["X_full"].dtype == np.float32
        assert result["y_full"].dtype == np.float32

    def test_determinism_with_seed(self) -> None:
        """Same seed should produce identical results."""
        params = CirclesParams(seed=123)

        result1 = CirclesGenerator.generate(params)
        result2 = CirclesGenerator.generate(params)

        np.testing.assert_array_equal(result1["X_full"], result2["X_full"])
        np.testing.assert_array_equal(result1["y_full"], result2["y_full"])

    def test_different_seeds_produce_different_data(self) -> None:
        """Different seeds should produce different results."""
        params1 = CirclesParams(seed=42)
        params2 = CirclesParams(seed=43)

        result1 = CirclesGenerator.generate(params1)
        result2 = CirclesGenerator.generate(params2)

        assert not np.allclose(result1["X_full"], result2["X_full"])

    def test_one_hot_labels(self) -> None:
        """Labels should be valid one-hot encoded."""
        params = CirclesParams(seed=42)
        result = CirclesGenerator.generate(params)

        row_sums = result["y_full"].sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(len(row_sums)))

        for row in result["y_full"]:
            assert np.sum(row == 1.0) == 1
            assert np.sum(row == 0.0) == 1

    def test_class_distribution(self) -> None:
        """Classes should be distributed according to inner_ratio."""
        params = CirclesParams(n_samples=100, inner_ratio=0.5, seed=42)
        result = CirclesGenerator.generate(params)

        class_counts = result["y_full"].sum(axis=0)
        assert class_counts[0] == 50
        assert class_counts[1] == 50

    def test_class_distribution_custom_ratio(self) -> None:
        """Custom inner_ratio should be respected."""
        params = CirclesParams(n_samples=100, inner_ratio=0.3, seed=42)
        result = CirclesGenerator.generate(params)

        class_counts = result["y_full"].sum(axis=0)
        assert class_counts[0] == 70
        assert class_counts[1] == 30

    def test_train_test_split_ratio(self) -> None:
        """Train/test split should respect configured ratios."""
        params = CirclesParams(
            n_samples=100,
            train_ratio=0.7,
            test_ratio=0.3,
            seed=42,
        )
        result = CirclesGenerator.generate(params)

        assert len(result["X_train"]) == 70
        assert len(result["X_test"]) == 30

    def test_points_on_circles_no_noise(self) -> None:
        """Without noise, points should lie exactly on circles."""
        params = CirclesParams(
            n_samples=100,
            outer_radius=2.0,
            factor=0.5,
            noise=0.0,
            inner_ratio=0.5,
            seed=42,
            shuffle=False,
        )
        result = CirclesGenerator.generate(params)

        outer_points = result["X_full"][:50]
        inner_points = result["X_full"][50:]

        outer_distances = np.linalg.norm(outer_points, axis=1)
        inner_distances = np.linalg.norm(inner_points, axis=1)

        np.testing.assert_array_almost_equal(outer_distances, np.full(50, 2.0))
        np.testing.assert_array_almost_equal(inner_distances, np.full(50, 1.0))

    def test_noise_adds_variation(self) -> None:
        """Noise parameter should add variation to circle radii."""
        params_no_noise = CirclesParams(n_samples=100, noise=0.0, seed=42)
        params_with_noise = CirclesParams(n_samples=100, noise=0.5, seed=42)

        result_no_noise = CirclesGenerator.generate(params_no_noise)
        result_with_noise = CirclesGenerator.generate(params_with_noise)

        var_no_noise = np.var(np.linalg.norm(result_no_noise["X_full"], axis=1))
        var_with_noise = np.var(np.linalg.norm(result_with_noise["X_full"], axis=1))

        assert var_with_noise > var_no_noise

    def test_factor_affects_inner_radius(self) -> None:
        """Factor should control inner circle radius."""
        params = CirclesParams(
            n_samples=100,
            outer_radius=4.0,
            factor=0.25,
            noise=0.0,
            inner_ratio=0.5,
            seed=42,
            shuffle=False,
        )
        result = CirclesGenerator.generate(params)

        inner_points = result["X_full"][50:]
        inner_distances = np.linalg.norm(inner_points, axis=1)

        np.testing.assert_array_almost_equal(inner_distances, np.full(50, 1.0))

    def test_generate_with_noise_covers_branch(self) -> None:
        """Noise > 0 should exercise the noise addition branch."""
        params = CirclesParams(
            n_samples=100,
            noise=0.3,
            seed=42,
            shuffle=False,
        )
        result = CirclesGenerator.generate(params)

        outer_distances = np.linalg.norm(result["X_full"][:50], axis=1)
        assert not np.allclose(outer_distances, np.full(50, 1.0))


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
            "outer_radius",
            "factor",
            "noise",
            "inner_ratio",
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
