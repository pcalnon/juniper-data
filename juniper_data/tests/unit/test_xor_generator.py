"""Unit tests for the XOR dataset generator."""

import numpy as np
import pytest

from juniper_data.generators.xor import VERSION, XorGenerator, XorParams, get_schema


@pytest.mark.unit
@pytest.mark.generators
class TestXorParams:
    """Tests for XorParams validation."""

    def test_default_params(self) -> None:
        """Default parameters are valid."""
        params = XorParams()
        assert params.n_points_per_quadrant == 50
        assert params.x_range == 1.0
        assert params.y_range == 1.0
        assert params.margin == 0.1
        assert params.noise == 0.0
        assert params.train_ratio == 0.8
        assert params.test_ratio == 0.2

    def test_custom_params(self) -> None:
        """Custom parameters are accepted."""
        params = XorParams(
            n_points_per_quadrant=100,
            x_range=2.0,
            y_range=3.0,
            margin=0.2,
            noise=0.1,
            seed=42,
            train_ratio=0.7,
            test_ratio=0.3,
        )
        assert params.n_points_per_quadrant == 100
        assert params.x_range == 2.0
        assert params.seed == 42

    def test_invalid_n_points(self) -> None:
        """n_points_per_quadrant must be >= 1."""
        with pytest.raises(ValueError):
            XorParams(n_points_per_quadrant=0)

    def test_invalid_x_range(self) -> None:
        """x_range must be > 0."""
        with pytest.raises(ValueError):
            XorParams(x_range=0)

    def test_invalid_train_ratio(self) -> None:
        """train_ratio must be in (0, 1]."""
        with pytest.raises(ValueError):
            XorParams(train_ratio=0)
        with pytest.raises(ValueError):
            XorParams(train_ratio=1.5)


@pytest.mark.unit
@pytest.mark.generators
class TestXorGenerator:
    """Tests for XorGenerator functionality."""

    def test_generate_correct_shapes(self) -> None:
        """Generated arrays have correct shapes."""
        params = XorParams(n_points_per_quadrant=25, seed=42)
        result = XorGenerator.generate(params)

        n_total = 4 * 25
        n_train = int(n_total * 0.8)
        n_test = n_total - n_train

        assert result["X_train"].shape == (n_train, 2)
        assert result["y_train"].shape == (n_train, 2)
        assert result["X_test"].shape == (n_test, 2)
        assert result["y_test"].shape == (n_test, 2)
        assert result["X_full"].shape == (n_total, 2)
        assert result["y_full"].shape == (n_total, 2)

    def test_generate_correct_dtypes(self) -> None:
        """Generated arrays have float32 dtype."""
        params = XorParams(n_points_per_quadrant=10, seed=42)
        result = XorGenerator.generate(params)

        for key in result:
            assert result[key].dtype == np.float32

    def test_generate_deterministic_with_seed(self) -> None:
        """Same seed produces identical data."""
        params = XorParams(n_points_per_quadrant=20, seed=42)

        result1 = XorGenerator.generate(params)
        result2 = XorGenerator.generate(params)

        np.testing.assert_array_equal(result1["X_full"], result2["X_full"])
        np.testing.assert_array_equal(result1["y_full"], result2["y_full"])

    def test_generate_different_seeds_different_data(self) -> None:
        """Different seeds produce different data."""
        params1 = XorParams(n_points_per_quadrant=20, seed=42)
        params2 = XorParams(n_points_per_quadrant=20, seed=123)

        result1 = XorGenerator.generate(params1)
        result2 = XorGenerator.generate(params2)

        assert not np.array_equal(result1["X_full"], result2["X_full"])

    def test_generate_one_hot_labels(self) -> None:
        """Labels are valid one-hot encodings."""
        params = XorParams(n_points_per_quadrant=10, seed=42)
        result = XorGenerator.generate(params)

        y_full = result["y_full"]
        row_sums = y_full.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(len(y_full)))

        assert set(np.unique(y_full)) == {0.0, 1.0}

    def test_generate_class_distribution(self) -> None:
        """Classes are balanced (equal quadrants for each class)."""
        params = XorParams(n_points_per_quadrant=25, seed=42)
        result = XorGenerator.generate(params)

        y_full = result["y_full"]
        class_0_count = y_full[:, 0].sum()
        class_1_count = y_full[:, 1].sum()

        assert class_0_count == 50
        assert class_1_count == 50

    def test_generate_quadrant_distribution(self) -> None:
        """Points are in correct quadrants based on class."""
        params = XorParams(n_points_per_quadrant=50, margin=0.1, seed=42, shuffle=False, noise=0)
        result = XorGenerator.generate(params)

        X = result["X_full"]
        y = result["y_full"]
        n = 50

        q1_x = X[0:n, 0]
        q1_y = X[0:n, 1]
        assert np.all(q1_x > 0)
        assert np.all(q1_y > 0)
        assert np.all(y[0:n, 0] == 1)

        q2_x = X[n : 2 * n, 0]
        q2_y = X[n : 2 * n, 1]
        assert np.all(q2_x < 0)
        assert np.all(q2_y > 0)
        assert np.all(y[n : 2 * n, 1] == 1)

        q3_x = X[2 * n : 3 * n, 0]
        q3_y = X[2 * n : 3 * n, 1]
        assert np.all(q3_x < 0)
        assert np.all(q3_y < 0)
        assert np.all(y[2 * n : 3 * n, 0] == 1)

        q4_x = X[3 * n : 4 * n, 0]
        q4_y = X[3 * n : 4 * n, 1]
        assert np.all(q4_x > 0)
        assert np.all(q4_y < 0)
        assert np.all(y[3 * n : 4 * n, 1] == 1)

    def test_generate_with_noise(self) -> None:
        """Noise is applied when specified."""
        params_no_noise = XorParams(n_points_per_quadrant=50, noise=0, seed=42)
        params_with_noise = XorParams(n_points_per_quadrant=50, noise=0.5, seed=42)

        result_no_noise = XorGenerator.generate(params_no_noise)
        result_with_noise = XorGenerator.generate(params_with_noise)

        assert not np.array_equal(result_no_noise["X_full"], result_with_noise["X_full"])

    def test_generate_respects_range(self) -> None:
        """Points are within specified range (before noise)."""
        params = XorParams(n_points_per_quadrant=100, x_range=2.0, y_range=3.0, margin=0.2, noise=0, seed=42)
        result = XorGenerator.generate(params)

        X = result["X_full"]
        assert np.all(np.abs(X[:, 0]) <= 2.0)
        assert np.all(np.abs(X[:, 1]) <= 3.0)

    def test_generate_respects_margin(self) -> None:
        """Points respect margin around axes."""
        params = XorParams(n_points_per_quadrant=100, margin=0.2, noise=0, seed=42)
        result = XorGenerator.generate(params)

        X = result["X_full"]
        assert np.all(np.abs(X[:, 0]) >= 0.2)
        assert np.all(np.abs(X[:, 1]) >= 0.2)


@pytest.mark.unit
@pytest.mark.generators
class TestXorGetSchema:
    """Tests for get_schema function."""

    def test_get_schema_returns_dict(self) -> None:
        """get_schema returns a dictionary."""
        schema = get_schema()
        assert isinstance(schema, dict)

    def test_get_schema_has_properties(self) -> None:
        """Schema has properties field."""
        schema = get_schema()
        assert "properties" in schema
        assert "n_points_per_quadrant" in schema["properties"]
        assert "x_range" in schema["properties"]
        assert "margin" in schema["properties"]
        assert "noise" in schema["properties"]


@pytest.mark.unit
@pytest.mark.generators
class TestXorVersion:
    """Tests for version constant."""

    def test_version_format(self) -> None:
        """Version follows semver format."""
        parts = VERSION.split(".")
        assert len(parts) == 3
        for part in parts:
            assert part.isdigit()
