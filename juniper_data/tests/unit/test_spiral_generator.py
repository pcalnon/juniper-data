"""Unit tests for the SpiralGenerator and related modules.

Tests cover:
- Output shapes and dimensions
- One-hot encoding correctness
- Deterministic reproducibility
- Parameter validation
"""

from typing import Dict

import numpy as np
import pytest
from pydantic import ValidationError

from juniper_data.generators.spiral import SpiralGenerator, SpiralParams


@pytest.mark.unit
@pytest.mark.spiral
@pytest.mark.generators
class TestSpiralShapes:
    """Tests for spiral dataset output shapes and dimensions."""

    def test_2_spiral_shapes(self, two_spiral_params: SpiralParams) -> None:
        """Verify X is (200, 2) and y is (200, 2) for n_spirals=2, n_points=100."""
        result = SpiralGenerator.generate(two_spiral_params)

        assert result["X_full"].shape == (200, 2)
        assert result["y_full"].shape == (200, 2)

    def test_3_spiral_shapes(self, three_spiral_params: SpiralParams) -> None:
        """Verify X is (150, 2) and y is (150, 3) for n_spirals=3, n_points=50."""
        result = SpiralGenerator.generate(three_spiral_params)

        assert result["X_full"].shape == (150, 2)
        assert result["y_full"].shape == (150, 3)

    def test_train_test_split_sizes(self, two_spiral_params: SpiralParams) -> None:
        """Verify train/test sizes match ratios within ±1."""
        result = SpiralGenerator.generate(two_spiral_params)

        total_points = two_spiral_params.total_points()
        expected_train = int(np.round(total_points * two_spiral_params.train_ratio))
        expected_test = int(np.round(total_points * two_spiral_params.test_ratio))

        assert abs(result["X_train"].shape[0] - expected_train) <= 1
        assert abs(result["y_train"].shape[0] - expected_train) <= 1
        assert abs(result["X_test"].shape[0] - expected_test) <= 1
        assert abs(result["y_test"].shape[0] - expected_test) <= 1

    def test_custom_split_ratios(self) -> None:
        """Verify custom train/test ratios are honored."""
        params = SpiralParams(
            n_spirals=2,
            n_points_per_spiral=100,
            train_ratio=0.6,
            test_ratio=0.3,
            seed=42,
        )
        result = SpiralGenerator.generate(params)

        total_points = params.total_points()
        expected_train = int(np.round(total_points * params.train_ratio))
        expected_test = int(np.round(total_points * params.test_ratio))

        assert abs(result["X_train"].shape[0] - expected_train) <= 1
        assert abs(result["X_test"].shape[0] - expected_test) <= 1

    def test_output_keys_present(self, generated_minimal_dataset: Dict[str, np.ndarray]) -> None:
        """Verify all expected keys are present in output."""
        expected_keys = {"X_train", "y_train", "X_test", "y_test", "X_full", "y_full"}
        assert set(generated_minimal_dataset.keys()) == expected_keys


@pytest.mark.unit
@pytest.mark.spiral
@pytest.mark.generators
class TestOneHotEncoding:
    """Tests for one-hot label encoding correctness."""

    def test_row_sums_to_one(self, generated_two_spiral_dataset: Dict[str, np.ndarray]) -> None:
        """Verify each row of y sums to 1.0."""
        y_full = generated_two_spiral_dataset["y_full"]

        row_sums = y_full.sum(axis=1)
        expected = np.ones(y_full.shape[0], dtype=np.float32)

        np.testing.assert_allclose(row_sums, expected, rtol=1e-6)

    def test_class_distribution(self) -> None:
        """Verify each class has n_points_per_spiral samples in full dataset."""
        params = SpiralParams(
            n_spirals=3,
            n_points_per_spiral=50,
            seed=42,
        )
        result = SpiralGenerator.generate(params)

        y_full = result["y_full"]
        class_counts = y_full.sum(axis=0).astype(int)

        expected_counts = np.array([50, 50, 50])
        np.testing.assert_array_equal(class_counts, expected_counts)

    def test_dtype_is_float32(self, generated_two_spiral_dataset: Dict[str, np.ndarray]) -> None:
        """Verify arrays are float32 dtype."""
        assert generated_two_spiral_dataset["X_full"].dtype == np.float32
        assert generated_two_spiral_dataset["y_full"].dtype == np.float32
        assert generated_two_spiral_dataset["X_train"].dtype == np.float32
        assert generated_two_spiral_dataset["y_train"].dtype == np.float32
        assert generated_two_spiral_dataset["X_test"].dtype == np.float32
        assert generated_two_spiral_dataset["y_test"].dtype == np.float32

    def test_one_hot_values_binary(self, generated_minimal_dataset: Dict[str, np.ndarray]) -> None:
        """Verify one-hot encoding contains only 0.0 and 1.0 values."""
        y_full = generated_minimal_dataset["y_full"]
        unique_values = np.unique(y_full)

        assert len(unique_values) == 2
        assert 0.0 in unique_values
        assert 1.0 in unique_values


@pytest.mark.unit
@pytest.mark.spiral
@pytest.mark.generators
class TestDeterminism:
    """Tests for deterministic reproducibility."""

    def test_same_seed_identical_output(self) -> None:
        """Verify same params+seed produces bitwise identical arrays."""
        params1 = SpiralParams(
            n_spirals=2,
            n_points_per_spiral=50,
            seed=12345,
        )
        params2 = SpiralParams(
            n_spirals=2,
            n_points_per_spiral=50,
            seed=12345,
        )

        result1 = SpiralGenerator.generate(params1)
        result2 = SpiralGenerator.generate(params2)

        np.testing.assert_array_equal(result1["X_full"], result2["X_full"])
        np.testing.assert_array_equal(result1["y_full"], result2["y_full"])
        np.testing.assert_array_equal(result1["X_train"], result2["X_train"])
        np.testing.assert_array_equal(result1["y_train"], result2["y_train"])
        np.testing.assert_array_equal(result1["X_test"], result2["X_test"])
        np.testing.assert_array_equal(result1["y_test"], result2["y_test"])

    def test_different_seed_different_output(self) -> None:
        """Verify different seeds produce different arrays."""
        params1 = SpiralParams(
            n_spirals=2,
            n_points_per_spiral=50,
            seed=12345,
        )
        params2 = SpiralParams(
            n_spirals=2,
            n_points_per_spiral=50,
            seed=54321,
        )

        result1 = SpiralGenerator.generate(params1)
        result2 = SpiralGenerator.generate(params2)

        assert not np.allclose(result1["X_full"], result2["X_full"])

    def test_multiple_calls_same_seed_identical(self) -> None:
        """Verify multiple sequential calls with same seed are identical."""
        params = SpiralParams(n_spirals=2, n_points_per_spiral=30, seed=999)

        results = [SpiralGenerator.generate(params) for _ in range(3)]

        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0]["X_full"], results[i]["X_full"])
            np.testing.assert_array_equal(results[0]["y_full"], results[i]["y_full"])


@pytest.mark.unit
@pytest.mark.spiral
@pytest.mark.generators
class TestParamValidation:
    """Tests for parameter validation errors."""

    def test_invalid_n_spirals_too_low(self) -> None:
        """Verify n_spirals < 2 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SpiralParams(n_spirals=1)

        error_str = str(exc_info.value)
        assert "n_spirals" in error_str or "greater than or equal to" in error_str

    def test_invalid_n_spirals_zero(self) -> None:
        """Verify n_spirals=0 raises ValidationError."""
        with pytest.raises(ValidationError):
            SpiralParams(n_spirals=0)

    def test_invalid_n_spirals_negative(self) -> None:
        """Verify negative n_spirals raises ValidationError."""
        with pytest.raises(ValidationError):
            SpiralParams(n_spirals=-1)

    def test_invalid_n_points_too_low(self) -> None:
        """Verify n_points < 10 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SpiralParams(n_points_per_spiral=5)

        error_str = str(exc_info.value)
        assert "n_points_per_spiral" in error_str or "greater than or equal to" in error_str

    def test_invalid_n_points_zero(self) -> None:
        """Verify n_points=0 raises ValidationError."""
        with pytest.raises(ValidationError):
            SpiralParams(n_points_per_spiral=0)

    def test_invalid_ratios_exceed_one(self) -> None:
        """Verify train_ratio + test_ratio > 1.0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SpiralParams(train_ratio=0.7, test_ratio=0.5)

        error_str = str(exc_info.value)
        assert "train_ratio" in error_str or "test_ratio" in error_str or "<= 1.0" in error_str

    def test_invalid_train_ratio_negative(self) -> None:
        """Verify negative train_ratio raises ValidationError."""
        with pytest.raises(ValidationError):
            SpiralParams(train_ratio=-0.1)

    def test_invalid_test_ratio_negative(self) -> None:
        """Verify negative test_ratio raises ValidationError."""
        with pytest.raises(ValidationError):
            SpiralParams(test_ratio=-0.1)

    def test_invalid_noise_negative(self) -> None:
        """Verify negative noise raises ValidationError."""
        with pytest.raises(ValidationError):
            SpiralParams(noise=-0.1)

    def test_invalid_noise_too_high(self) -> None:
        """Verify noise > MAX_NOISE raises ValidationError."""
        with pytest.raises(ValidationError):
            SpiralParams(noise=10.0)

    def test_invalid_n_rotations_too_low(self) -> None:
        """Verify n_rotations < MIN_ROTATIONS raises ValidationError."""
        with pytest.raises(ValidationError):
            SpiralParams(n_rotations=0.1)

    def test_valid_edge_case_min_values(self) -> None:
        """Verify minimum valid values are accepted."""
        params = SpiralParams(
            n_spirals=2,
            n_points_per_spiral=10,
            n_rotations=0.5,
            noise=0.0,
        )
        assert params.n_spirals == 2
        assert params.n_points_per_spiral == 10
        assert params.n_rotations == 0.5
        assert params.noise == 0.0


@pytest.mark.unit
@pytest.mark.spiral
@pytest.mark.generators
class TestSpiralGeometry:
    """Tests for spiral geometric properties."""

    def test_coordinates_centered_near_origin(self, generated_two_spiral_dataset: Dict[str, np.ndarray]) -> None:
        """Verify spiral coordinates are centered roughly around origin."""
        X_full = generated_two_spiral_dataset["X_full"]
        mean_x = X_full[:, 0].mean()
        mean_y = X_full[:, 1].mean()

        assert abs(mean_x) < 2.0
        assert abs(mean_y) < 2.0

    def test_coordinates_within_expected_radius(self) -> None:
        """Verify coordinates fall within expected radius bounds."""
        params = SpiralParams(
            n_spirals=2,
            n_points_per_spiral=100,
            noise=0.0,
            seed=42,
        )
        result = SpiralGenerator.generate(params)

        X_full = result["X_full"]
        distances = np.sqrt(X_full[:, 0] ** 2 + X_full[:, 1] ** 2)
        max_distance = distances.max()

        assert max_distance <= 12.0

    def test_noise_increases_variance(self) -> None:
        """Verify adding noise increases coordinate variance."""
        params_no_noise = SpiralParams(
            n_spirals=2,
            n_points_per_spiral=100,
            noise=0.0,
            seed=42,
        )
        params_with_noise = SpiralParams(
            n_spirals=2,
            n_points_per_spiral=100,
            noise=1.0,
            seed=42,
        )

        result_no_noise = SpiralGenerator.generate(params_no_noise)
        result_with_noise = SpiralGenerator.generate(params_with_noise)

        var_no_noise = result_no_noise["X_full"].var()
        var_with_noise = result_with_noise["X_full"].var()

        assert var_with_noise > var_no_noise


@pytest.mark.unit
@pytest.mark.spiral
@pytest.mark.generators
class TestSpiralGeneratorLegacyMode:
    """Tests for legacy_cascor algorithm mode."""

    def test_legacy_mode_generates_correct_shapes(self) -> None:
        """Verify legacy_cascor mode generates arrays with correct shapes."""
        params = SpiralParams(
            n_spirals=2,
            n_points_per_spiral=50,
            algorithm="legacy_cascor",
            seed=42,
        )
        result = SpiralGenerator.generate(params)

        assert result["X_full"].shape == (100, 2)
        assert result["y_full"].shape == (100, 2)
        assert result["X_train"].shape[1] == 2
        assert result["y_train"].shape[1] == 2

    def test_legacy_mode_deterministic_with_seed(self) -> None:
        """Verify same seed produces identical arrays in legacy mode."""
        params1 = SpiralParams(
            n_spirals=2,
            n_points_per_spiral=50,
            algorithm="legacy_cascor",
            seed=12345,
        )
        params2 = SpiralParams(
            n_spirals=2,
            n_points_per_spiral=50,
            algorithm="legacy_cascor",
            seed=12345,
        )

        result1 = SpiralGenerator.generate(params1)
        result2 = SpiralGenerator.generate(params2)

        np.testing.assert_array_equal(result1["X_full"], result2["X_full"])
        np.testing.assert_array_equal(result1["y_full"], result2["y_full"])

    def test_legacy_mode_different_from_modern(self) -> None:
        """Verify legacy_cascor produces different results than modern algorithm."""
        params_modern = SpiralParams(
            n_spirals=2,
            n_points_per_spiral=50,
            algorithm="modern",
            seed=42,
        )
        params_legacy = SpiralParams(
            n_spirals=2,
            n_points_per_spiral=50,
            algorithm="legacy_cascor",
            seed=42,
        )

        result_modern = SpiralGenerator.generate(params_modern)
        result_legacy = SpiralGenerator.generate(params_legacy)

        assert not np.allclose(result_modern["X_full"], result_legacy["X_full"])

    def test_legacy_mode_uniform_noise_range(self) -> None:
        """Verify legacy mode uses uniform noise in [0, noise) range."""
        params = SpiralParams(
            n_spirals=2,
            n_points_per_spiral=100,
            algorithm="legacy_cascor",
            noise=1.0,
            seed=42,
        )
        result = SpiralGenerator.generate(params)

        params_no_noise = SpiralParams(
            n_spirals=2,
            n_points_per_spiral=100,
            algorithm="legacy_cascor",
            noise=0.0,
            seed=42,
        )
        result_no_noise = SpiralGenerator.generate(params_no_noise)

        noise_x = result["X_full"][:, 0] - result_no_noise["X_full"][:, 0]
        noise_y = result["X_full"][:, 1] - result_no_noise["X_full"][:, 1]

        assert noise_x.min() >= 0.0
        assert noise_x.max() < 1.0
        assert noise_y.min() >= 0.0
        assert noise_y.max() < 1.0

    def test_legacy_mode_radii_distribution(self) -> None:
        """Verify legacy mode uses sqrt-uniform radii distribution."""
        params = SpiralParams(
            n_spirals=2,
            n_points_per_spiral=1000,
            algorithm="legacy_cascor",
            noise=0.0,
            seed=42,
        )
        result = SpiralGenerator.generate(params)

        X = result["X_full"]
        radii = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)

        radii_squared = radii**2
        radii_squared_normalized = radii_squared / radii_squared.max()

        assert radii_squared_normalized.mean() < 0.6

    def test_origin_offset_works(self) -> None:
        """Verify origin parameter shifts the dataset center."""
        params_centered = SpiralParams(
            n_spirals=2,
            n_points_per_spiral=100,
            algorithm="legacy_cascor",
            origin=(0.0, 0.0),
            seed=42,
        )
        params_offset = SpiralParams(
            n_spirals=2,
            n_points_per_spiral=100,
            algorithm="legacy_cascor",
            origin=(5.0, 10.0),
            seed=42,
        )

        result_centered = SpiralGenerator.generate(params_centered)
        result_offset = SpiralGenerator.generate(params_offset)

        mean_centered = result_centered["X_full"].mean(axis=0)
        mean_offset = result_offset["X_full"].mean(axis=0)

        np.testing.assert_allclose(mean_offset[0] - mean_centered[0], 5.0, atol=0.1)
        np.testing.assert_allclose(mean_offset[1] - mean_centered[1], 10.0, atol=0.1)

    def test_radius_parameter_controls_spread(self) -> None:
        """Verify radius parameter controls the spread of points."""
        params_small = SpiralParams(
            n_spirals=2,
            n_points_per_spiral=100,
            algorithm="legacy_cascor",
            radius=5.0,
            noise=0.0,
            seed=42,
        )
        params_large = SpiralParams(
            n_spirals=2,
            n_points_per_spiral=100,
            algorithm="legacy_cascor",
            radius=20.0,
            noise=0.0,
            seed=42,
        )

        result_small = SpiralGenerator.generate(params_small)
        result_large = SpiralGenerator.generate(params_large)

        radii_small = np.sqrt(result_small["X_full"][:, 0] ** 2 + result_small["X_full"][:, 1] ** 2)
        radii_large = np.sqrt(result_large["X_full"][:, 0] ** 2 + result_large["X_full"][:, 1] ** 2)

        max_small = radii_small.max()
        max_large = radii_large.max()

        ratio = max_large / max_small
        assert 3.0 < ratio < 5.0

    def test_algorithm_param_validation(self) -> None:
        """Verify invalid algorithm values raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SpiralParams(algorithm="invalid_algorithm")  # type: ignore[arg-type]  # negative test: ensure runtime validation rejects invalid algorithm

        error_str = str(exc_info.value)
        assert "algorithm" in error_str or "Input should be" in error_str


@pytest.mark.unit
@pytest.mark.spiral
class TestGetSchema:
    """Tests for the get_schema function."""

    def test_get_schema_returns_dict(self) -> None:
        """Verify get_schema returns a dictionary."""
        from juniper_data.generators.spiral.generator import get_schema

        schema = get_schema()
        assert isinstance(schema, dict)

    def test_get_schema_contains_properties(self) -> None:
        """Verify schema contains expected properties."""
        from juniper_data.generators.spiral.generator import get_schema

        schema = get_schema()
        assert "properties" in schema
        assert "n_spirals" in schema["properties"]
        assert "n_points_per_spiral" in schema["properties"]
        assert "noise" in schema["properties"]
        assert "seed" in schema["properties"]

    def test_get_schema_contains_title(self) -> None:
        """Verify schema contains title."""
        from juniper_data.generators.spiral.generator import get_schema

        schema = get_schema()
        assert "title" in schema
        assert schema["title"] == "SpiralParams"


@pytest.mark.unit
@pytest.mark.spiral
@pytest.mark.generators
class TestParameterAliases:
    """Tests for parameter aliases for consumer compatibility."""

    def test_n_points_alias(self) -> None:
        """Verify n_points is accepted as alias for n_points_per_spiral."""
        params = SpiralParams.model_validate({"n_points": 50, "n_spirals": 2, "seed": 42})
        assert params.n_points_per_spiral == 50

    def test_noise_level_alias(self) -> None:
        """Verify noise_level is accepted as alias for noise."""
        params = SpiralParams.model_validate({"noise_level": 0.5, "n_spirals": 2, "seed": 42})
        assert params.noise == 0.5

    def test_canonical_name_takes_precedence(self) -> None:
        """Verify canonical name is used when both are provided."""
        params = SpiralParams(n_points_per_spiral=100, noise=0.2, n_spirals=2, seed=42)
        assert params.n_points_per_spiral == 100
        assert params.noise == 0.2

    def test_alias_generates_correct_dataset(self) -> None:
        """Verify dataset generation works with alias parameters."""
        params = SpiralParams.model_validate({"n_points": 25, "noise_level": 0.15, "n_spirals": 2, "seed": 42})
        result = SpiralGenerator.generate(params)

        assert result["X_full"].shape == (50, 2)
        assert result["y_full"].shape == (50, 2)

    def test_alias_determinism(self) -> None:
        """Verify same seed produces same results regardless of alias usage."""
        params1 = SpiralParams(n_points_per_spiral=30, noise=0.1, n_spirals=2, seed=123)
        params2 = SpiralParams.model_validate({"n_points": 30, "noise_level": 0.1, "n_spirals": 2, "seed": 123})

        result1 = SpiralGenerator.generate(params1)
        result2 = SpiralGenerator.generate(params2)

        np.testing.assert_array_equal(result1["X_full"], result2["X_full"])
        np.testing.assert_array_equal(result1["y_full"], result2["y_full"])
