"""Unit tests for the two-moons dataset generator.

Introduced to cover the new server-side ``MoonGenerator`` that
resolves XREPO-01b / DC-02 (client previously referenced a
``"moon"`` generator that did not exist on the server).
"""

import numpy as np
import pytest

from juniper_data.generators.moon import VERSION, MoonGenerator, MoonParams, get_schema

pytestmark = [pytest.mark.unit, pytest.mark.generators]


class TestMoonParams:
    """Tests for ``MoonParams`` validation."""

    def test_default_params(self) -> None:
        params = MoonParams()
        assert params.n_samples == 200
        assert params.noise == 0.1
        assert params.train_ratio == 0.8
        assert params.test_ratio == 0.2
        assert params.shuffle is True

    def test_custom_params(self) -> None:
        params = MoonParams(n_samples=500, noise=0.05, seed=7)
        assert params.n_samples == 500
        assert params.noise == 0.05
        assert params.seed == 7

    def test_invalid_n_samples_too_low(self) -> None:
        with pytest.raises(ValueError):
            MoonParams(n_samples=1)

    def test_invalid_noise_negative(self) -> None:
        with pytest.raises(ValueError):
            MoonParams(noise=-0.01)


class TestMoonGenerator:
    """Tests for ``MoonGenerator.generate``."""

    def test_generate_returns_expected_keys(self) -> None:
        params = MoonParams(seed=42)
        result = MoonGenerator.generate(params)
        assert set(result.keys()) == {"X_train", "y_train", "X_val", "y_val", "X_test", "y_test", "X_full", "y_full"}

    def test_generate_shapes(self) -> None:
        params = MoonParams(n_samples=150, seed=42)
        result = MoonGenerator.generate(params)
        # n_samples is the TRAIN count under additive sizing: 150 + 60 + 45 = 255.
        assert result["X_train"].shape == (150, 2)
        assert result["X_val"].shape == (60, 2)
        assert result["X_test"].shape == (45, 2)
        assert result["X_full"].shape == (255, 2)
        assert result["y_full"].shape == (255, 2)

    def test_generate_dtypes(self) -> None:
        params = MoonParams(seed=42)
        result = MoonGenerator.generate(params)
        for key in ("X_train", "y_train", "X_test", "y_test", "X_full", "y_full"):
            assert result[key].dtype == np.float32

    def test_determinism_with_seed(self) -> None:
        params = MoonParams(seed=123)
        result1 = MoonGenerator.generate(params)
        result2 = MoonGenerator.generate(params)
        np.testing.assert_array_equal(result1["X_full"], result2["X_full"])
        np.testing.assert_array_equal(result1["y_full"], result2["y_full"])

    def test_different_seeds_produce_different_data(self) -> None:
        params_a = MoonParams(seed=1)
        params_b = MoonParams(seed=2)
        result_a = MoonGenerator.generate(params_a)
        result_b = MoonGenerator.generate(params_b)
        # Any noise jitter under a distinct seed is enough — compare with allclose.
        assert not np.allclose(result_a["X_full"], result_b["X_full"])

    def test_one_hot_labels(self) -> None:
        params = MoonParams(seed=42)
        result = MoonGenerator.generate(params)
        row_sums = result["y_full"].sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(len(row_sums)))

    def test_class_balance(self) -> None:
        params = MoonParams(n_samples=200, seed=42)
        result = MoonGenerator.generate(params)
        counts = result["y_full"].sum(axis=0)
        # 200 train + 80 val + 60 test = 340 realised rows, evenly halved.
        assert counts[0] == 170
        assert counts[1] == 170

    def test_train_test_split_ratio(self) -> None:
        # Ratios divide a fixed N -- that is carve mode by definition.
        params = MoonParams(n_samples=100, train_ratio=0.7, test_ratio=0.3, seed=42, sizing_mode="carve")
        result = MoonGenerator.generate(params)
        assert len(result["X_train"]) == 70
        assert len(result["X_test"]) == 30

    def test_noise_adds_variation(self) -> None:
        params_clean = MoonParams(n_samples=200, noise=0.0, seed=42)
        params_noisy = MoonParams(n_samples=200, noise=0.5, seed=42)
        result_clean = MoonGenerator.generate(params_clean)
        result_noisy = MoonGenerator.generate(params_noisy)
        assert np.var(result_noisy["X_full"]) > np.var(result_clean["X_full"])

    def test_geometry_without_noise(self) -> None:
        """With zero noise the two moons should lie on their analytic curves."""
        params = MoonParams(n_samples=100, noise=0.0, seed=42, shuffle=False)
        result = MoonGenerator.generate(params)

        # The boundary is derived from the realised array rather than hardcoded:
        # n_samples now denotes TRAIN, so the realised row count is larger and the
        # two moons meet at its midpoint, not at row 50.
        n_upper = result["X_full"].shape[0] // 2
        upper = result["X_full"][:n_upper]
        # Upper moon: y = sin(theta), x = cos(theta) — satisfies x^2 + y^2 == 1
        radii = np.linalg.norm(upper, axis=1)
        np.testing.assert_array_almost_equal(radii, np.ones(n_upper), decimal=5)

        lower = result["X_full"][n_upper:]
        # Lower moon: x = 1 - cos, y = 0.5 - sin — so (x-1)^2 + (y-0.5)^2 == 1
        centered = lower - np.array([1.0, 0.5], dtype=np.float32)
        lower_radii = np.linalg.norm(centered, axis=1)
        np.testing.assert_array_almost_equal(lower_radii, np.ones(lower.shape[0]), decimal=5)


class TestGetSchema:
    def test_returns_dict(self) -> None:
        schema = get_schema()
        assert isinstance(schema, dict)

    def test_schema_has_properties(self) -> None:
        schema = get_schema()
        assert "properties" in schema

    def test_schema_includes_all_params(self) -> None:
        schema = get_schema()
        expected = {"n_samples", "noise", "seed", "train_ratio", "test_ratio", "shuffle"}
        assert expected.issubset(set(schema["properties"].keys()))


class TestVersion:
    def test_version_format(self) -> None:
        parts = VERSION.split(".")
        assert len(parts) == 3
        assert all(part.isdigit() for part in parts)
