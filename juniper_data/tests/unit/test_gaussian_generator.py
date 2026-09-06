"""Unit tests for the Gaussian blobs dataset generator."""

import numpy as np
import pytest

from juniper_data.generators.gaussian import VERSION, GaussianGenerator, GaussianParams, get_schema
from juniper_data.tests.partitions import whole

pytestmark = [pytest.mark.unit, pytest.mark.generators]


class TestGaussianParams:
    """Tests for GaussianParams validation."""

    def test_default_params(self) -> None:
        """Default parameters should be valid."""
        params = GaussianParams()
        assert params.n_classes == 2
        assert params.n_samples_per_class == 50
        assert params.n_features == 2
        assert params.class_std == 1.0
        assert params.centers is None
        assert params.center_radius == 3.0
        assert params.noise == 0.0
        assert params.train_ratio == 0.8
        assert params.test_ratio == 0.2
        assert params.shuffle is True

    def test_custom_params(self) -> None:
        """Custom parameters should be accepted."""
        params = GaussianParams(
            n_classes=3,
            n_samples_per_class=100,
            n_features=4,
            class_std=0.5,
            seed=42,
        )
        assert params.n_classes == 3
        assert params.n_samples_per_class == 100
        assert params.n_features == 4
        assert params.class_std == 0.5
        assert params.seed == 42

    def test_list_class_std(self) -> None:
        """List of class_std values should be accepted."""
        params = GaussianParams(n_classes=3, class_std=[0.5, 1.0, 1.5])
        assert params.class_std == [0.5, 1.0, 1.5]

    def test_invalid_class_std_negative(self) -> None:
        """Negative class_std should raise validation error."""
        with pytest.raises(ValueError, match="positive"):
            GaussianParams(class_std=-0.5)

    def test_invalid_class_std_list_negative(self) -> None:
        """List with negative class_std should raise validation error."""
        with pytest.raises(ValueError, match="positive"):
            GaussianParams(class_std=[0.5, -1.0, 1.5])

    def test_custom_centers(self) -> None:
        """Custom centers should be accepted."""
        centers = [[0.0, 0.0], [5.0, 5.0]]
        params = GaussianParams(n_classes=2, n_features=2, centers=centers)
        assert params.centers == centers

    def test_empty_centers_invalid(self) -> None:
        """Empty centers list should raise validation error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            GaussianParams(centers=[])

    def test_invalid_n_classes_too_low(self) -> None:
        """n_classes less than 2 should raise validation error."""
        with pytest.raises(ValueError):
            GaussianParams(n_classes=1)

    def test_invalid_n_classes_too_high(self) -> None:
        """n_classes greater than 10 should raise validation error."""
        with pytest.raises(ValueError):
            GaussianParams(n_classes=11)


class TestGaussianGenerator:
    """Tests for GaussianGenerator."""

    def test_generate_returns_expected_keys(self) -> None:
        """Generated data should contain all expected keys."""
        params = GaussianParams(seed=42)
        result = GaussianGenerator.generate(params)

        expected_keys = {"X_train", "y_train", "X_val", "y_val", "X_test", "y_test"}
        assert set(result.keys()) == expected_keys

    def test_generate_shapes(self) -> None:
        """Generated arrays should have correct shapes."""
        params = GaussianParams(
            n_classes=3,
            n_samples_per_class=40,
            n_features=5,
            seed=42,
        )
        result = GaussianGenerator.generate(params)

        # 3 x 40 = 120 TRAIN samples under additive sizing, plus 48 val and 36 test.
        n_train = 3 * 40
        total_samples = n_train + 48 + 36
        assert result["X_train"].shape == (n_train, 5)
        assert whole(result, "X").shape == (total_samples, 5)
        assert whole(result, "y").shape == (total_samples, 3)

    def test_generate_dtypes(self) -> None:
        """Generated arrays should have float32 dtype."""
        params = GaussianParams(seed=42)
        result = GaussianGenerator.generate(params)

        assert result["X_train"].dtype == np.float32
        assert result["y_train"].dtype == np.float32
        assert whole(result, "X").dtype == np.float32
        assert whole(result, "y").dtype == np.float32

    def test_determinism_with_seed(self) -> None:
        """Same seed should produce identical results."""
        params = GaussianParams(seed=123)

        result1 = GaussianGenerator.generate(params)
        result2 = GaussianGenerator.generate(params)

        np.testing.assert_array_equal(whole(result1, "X"), whole(result2, "X"))
        np.testing.assert_array_equal(whole(result1, "y"), whole(result2, "y"))

    def test_different_seeds_produce_different_data(self) -> None:
        """Different seeds should produce different results."""
        params1 = GaussianParams(seed=42)
        params2 = GaussianParams(seed=43)

        result1 = GaussianGenerator.generate(params1)
        result2 = GaussianGenerator.generate(params2)

        assert not np.allclose(whole(result1, "X"), whole(result2, "X"))

    def test_one_hot_labels(self) -> None:
        """Labels should be valid one-hot encoded."""
        params = GaussianParams(n_classes=4, seed=42)
        result = GaussianGenerator.generate(params)

        row_sums = whole(result, "y").sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(len(row_sums)))

        for row in whole(result, "y"):
            assert np.sum(row == 1.0) == 1
            assert np.sum(row == 0.0) == params.n_classes - 1

    def test_class_distribution(self) -> None:
        """Each class should have n_samples_per_class samples."""
        params = GaussianParams(n_classes=3, n_samples_per_class=50, seed=42)
        result = GaussianGenerator.generate(params)

        class_counts = whole(result, "y").sum(axis=0)
        # 3 x 50 = 150 train + 60 val + 45 test = 255 realised rows, 85 per class.
        np.testing.assert_array_equal(class_counts, [85, 85, 85])

    def test_train_test_split_ratio(self) -> None:
        """Train/test split should respect configured ratios."""
        params = GaussianParams(
            n_samples_per_class=50,
            train_ratio=0.7,
            test_ratio=0.3,
            seed=42,
            # Ratios divide a fixed N -- that is carve mode by definition.
            sizing_mode="carve",
        )
        result = GaussianGenerator.generate(params)

        total = 2 * 50
        expected_train = int(total * 0.7)
        expected_test = int(total * 0.3)

        assert len(result["X_train"]) == expected_train
        assert len(result["X_test"]) == expected_test

    def test_custom_centers(self) -> None:
        """Custom centers should position class means correctly."""
        centers = [[0.0, 0.0], [10.0, 10.0]]
        params = GaussianParams(
            n_classes=2,
            n_samples_per_class=100,
            centers=centers,
            class_std=0.1,
            noise=0.0,
            seed=42,
        )
        result = GaussianGenerator.generate(params)

        # Select by CLASS rather than position: the per-class block size follows
        # the realised row count, and a mask asks the question directly.
        labels = np.argmax(whole(result, "y"), axis=1)
        class_0_samples = whole(result, "X")[labels == 0]
        class_1_samples = whole(result, "X")[labels == 1]

        assert class_0_samples.shape[0] > 0 and class_1_samples.shape[0] > 0

        class_0_mean = class_0_samples.mean(axis=0)
        class_1_mean = class_1_samples.mean(axis=0)

        np.testing.assert_array_almost_equal(class_0_mean, [0.0, 0.0], decimal=0)
        np.testing.assert_array_almost_equal(class_1_mean, [10.0, 10.0], decimal=0)

    def test_centers_dimension_mismatch_raises_error(self) -> None:
        """Centers with wrong dimensions should raise error."""
        centers = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        params = GaussianParams(
            n_classes=2,
            n_features=2,
            centers=centers,
            seed=42,
        )
        with pytest.raises(ValueError, match="n_features"):
            GaussianGenerator.generate(params)

    def test_centers_count_mismatch_raises_error(self) -> None:
        """Wrong number of centers should raise error."""
        centers = [[0.0, 0.0]]
        params = GaussianParams(
            n_classes=2,
            n_features=2,
            centers=centers,
            seed=42,
        )
        with pytest.raises(ValueError, match="n_classes"):
            GaussianGenerator.generate(params)

    def test_noise_adds_variation(self) -> None:
        """Noise parameter should increase data variance."""
        params_no_noise = GaussianParams(
            n_samples_per_class=100,
            class_std=0.5,
            noise=0.0,
            seed=42,
        )
        params_with_noise = GaussianParams(
            n_samples_per_class=100,
            class_std=0.5,
            noise=1.0,
            seed=42,
        )

        result_no_noise = GaussianGenerator.generate(params_no_noise)
        result_with_noise = GaussianGenerator.generate(params_with_noise)

        var_no_noise = np.var(whole(result_no_noise, "X"))
        var_with_noise = np.var(whole(result_with_noise, "X"))

        assert var_with_noise > var_no_noise

    def test_auto_center_placement(self) -> None:
        """Auto-placed centers should be on a circle."""
        params = GaussianParams(
            n_classes=4,
            n_samples_per_class=100,
            center_radius=5.0,
            class_std=0.1,
            seed=42,
        )
        result = GaussianGenerator.generate(params)

        labels = np.argmax(whole(result, "y"), axis=1)
        for i in range(4):
            class_rows = whole(result, "X")[labels == i]
            assert class_rows.shape[0] > 0
            class_mean = class_rows.mean(axis=0)
            distance_from_origin = np.linalg.norm(class_mean)
            np.testing.assert_almost_equal(distance_from_origin, 5.0, decimal=0)

    def test_generate_with_list_class_std(self) -> None:
        """Per-class std list should apply different stds to each class."""
        params = GaussianParams(
            n_classes=3,
            n_samples_per_class=100,
            class_std=[0.1, 0.5, 2.0],
            seed=42,
        )
        result = GaussianGenerator.generate(params)

        # 3 x 100 = 300 TRAIN samples, plus 120 val and 90 test.
        assert whole(result, "X").shape == (510, 2)

    def test_generate_single_feature(self) -> None:
        """Single feature should skip sin component in center placement."""
        params = GaussianParams(
            n_classes=2,
            n_samples_per_class=50,
            n_features=1,
            seed=42,
        )
        result = GaussianGenerator.generate(params)

        # 100 TRAIN samples, plus 40 val and 30 test.
        assert whole(result, "X").shape == (170, 1)

    def test_get_stds_scalar(self) -> None:
        """Scalar class_std should return a list of repeated values."""
        params = GaussianParams(n_classes=3, class_std=0.5)
        stds = GaussianGenerator._get_stds(params)
        assert stds == [0.5, 0.5, 0.5]

    def test_get_stds_list(self) -> None:
        """List class_std should be returned as-is."""
        params = GaussianParams(n_classes=3, class_std=[0.1, 0.5, 2.0])
        stds = GaussianGenerator._get_stds(params)
        assert stds == [0.1, 0.5, 2.0]


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
            "n_classes",
            "n_samples_per_class",
            "n_features",
            "class_std",
            "centers",
            "center_radius",
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
