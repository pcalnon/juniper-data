#####################################################################################################################################################################################################
# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     test_generator_benchmarks.py
# Author:        Paul Calnon
# Version:       0.4.2
#
# Date Created:  2026-02-25
# Last Modified: 2026-02-25
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2026 Paul Calnon
#
# Description:
#    Performance benchmarks for dataset generators.
#    Measures generation throughput (points/second) for each synthetic generator
#    and scaling behavior across dataset sizes.
#
# Usage:
#    # Run benchmarks with timing (disabled by default in addopts):
#    pytest juniper_data/tests/performance/test_generator_benchmarks.py --benchmark-enable -v
#
#    # Run with autosave for regression tracking:
#    pytest juniper_data/tests/performance/test_generator_benchmarks.py --benchmark-enable --benchmark-autosave
#
#    # Compare against saved baseline:
#    pytest juniper_data/tests/performance/test_generator_benchmarks.py --benchmark-enable --benchmark-compare
#
# References:
#    - RD-009: Performance Test Infrastructure
#    - pytest-benchmark: https://pytest-benchmark.readthedocs.io/
#####################################################################################################################################################################################################

"""Performance benchmarks for dataset generators.

Benchmarks measure generation throughput for each synthetic generator
and scaling behavior with increasing dataset sizes. External-dependency
generators (MNIST, ARC-AGI, CSV import) are excluded as they measure
I/O rather than generation performance.
"""

import numpy as np
import pytest

from juniper_data.generators.checkerboard.generator import CheckerboardGenerator
from juniper_data.generators.checkerboard.params import CheckerboardParams
from juniper_data.generators.circles.generator import CirclesGenerator
from juniper_data.generators.circles.params import CirclesParams
from juniper_data.generators.gaussian.generator import GaussianGenerator
from juniper_data.generators.gaussian.params import GaussianParams
from juniper_data.generators.spiral.generator import SpiralGenerator
from juniper_data.generators.spiral.params import SpiralParams
from juniper_data.generators.xor.generator import XorGenerator
from juniper_data.generators.xor.params import XorParams

# ═══════════════════════════════════════════════════════════════════════════════
# Generator Throughput Benchmarks
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.performance
class TestGeneratorThroughput:
    """Benchmark each synthetic generator with standard parameters.

    Each test generates a dataset of ~1000 total points and validates
    the output structure. The benchmark fixture handles timing and
    iteration count automatically.
    """

    def test_spiral_generator(self, benchmark):
        """Benchmark spiral dataset generation (1000 points)."""
        params = SpiralParams(n_spirals=2, n_points_per_spiral=500, seed=42)
        result = benchmark(SpiralGenerator.generate, params)
        assert "X_train" in result
        assert result["X_full"].shape[0] == 1000
        assert result["X_train"].dtype == np.float32

    def test_xor_generator(self, benchmark):
        """Benchmark XOR dataset generation (1000 points)."""
        params = XorParams(n_points_per_quadrant=250, seed=42)
        result = benchmark(XorGenerator.generate, params)
        assert "X_train" in result
        assert result["X_full"].shape[0] == 1000
        assert result["X_train"].dtype == np.float32

    def test_gaussian_generator(self, benchmark):
        """Benchmark Gaussian blobs dataset generation (1000 points)."""
        params = GaussianParams(n_classes=2, n_samples_per_class=500, seed=42)
        result = benchmark(GaussianGenerator.generate, params)
        assert "X_train" in result
        assert result["X_full"].shape[0] == 1000
        assert result["X_train"].dtype == np.float32

    def test_circles_generator(self, benchmark):
        """Benchmark concentric circles dataset generation (1000 points)."""
        params = CirclesParams(n_samples=1000, seed=42)
        result = benchmark(CirclesGenerator.generate, params)
        assert "X_train" in result
        assert result["X_full"].shape[0] == 1000
        assert result["X_train"].dtype == np.float32

    def test_checkerboard_generator(self, benchmark):
        """Benchmark checkerboard dataset generation (1000 points)."""
        params = CheckerboardParams(n_samples=1000, seed=42)
        result = benchmark(CheckerboardGenerator.generate, params)
        assert "X_train" in result
        assert result["X_full"].shape[0] == 1000
        assert result["X_train"].dtype == np.float32


# ═══════════════════════════════════════════════════════════════════════════════
# Scaling Benchmarks
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.performance
class TestGeneratorScaling:
    """Benchmark generation throughput across dataset sizes.

    Tests the spiral generator (representative of numpy-based generators)
    with increasing point counts to characterize scaling behavior.
    """

    @pytest.mark.parametrize(
        "n_points_per_spiral",
        [100, 500, 1000, 5000],
        ids=["200pts", "1000pts", "2000pts", "10000pts"],
    )
    def test_spiral_scaling(self, benchmark, n_points_per_spiral):
        """Benchmark spiral generation at various dataset sizes."""
        params = SpiralParams(n_spirals=2, n_points_per_spiral=n_points_per_spiral, seed=42)
        result = benchmark(SpiralGenerator.generate, params)
        assert result["X_full"].shape[0] == n_points_per_spiral * 2

    @pytest.mark.parametrize(
        "n_samples",
        [100, 500, 1000, 5000],
        ids=["100pts", "500pts", "1000pts", "5000pts"],
    )
    def test_gaussian_scaling(self, benchmark, n_samples):
        """Benchmark Gaussian generation at various dataset sizes."""
        params = GaussianParams(n_classes=2, n_samples_per_class=n_samples // 2, seed=42)
        result = benchmark(GaussianGenerator.generate, params)
        assert result["X_full"].shape[0] == n_samples


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-Class Benchmarks
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.performance
class TestMultiClassScaling:
    """Benchmark generation throughput as class count increases."""

    @pytest.mark.parametrize(
        "n_spirals",
        [2, 3, 5, 8],
        ids=["2class", "3class", "5class", "8class"],
    )
    def test_spiral_class_scaling(self, benchmark, n_spirals):
        """Benchmark spiral generation with varying class counts."""
        params = SpiralParams(n_spirals=n_spirals, n_points_per_spiral=200, seed=42)
        result = benchmark(SpiralGenerator.generate, params)
        assert result["X_full"].shape[0] == n_spirals * 200
        assert result["y_full"].shape[1] == n_spirals

    @pytest.mark.parametrize(
        "n_classes",
        [2, 3, 5, 8],
        ids=["2class", "3class", "5class", "8class"],
    )
    def test_gaussian_class_scaling(self, benchmark, n_classes):
        """Benchmark Gaussian generation with varying class counts."""
        params = GaussianParams(n_classes=n_classes, n_samples_per_class=100, seed=42)
        result = benchmark(GaussianGenerator.generate, params)
        assert result["X_full"].shape[0] == n_classes * 100
