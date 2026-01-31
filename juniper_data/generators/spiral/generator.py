"""Core NumPy-only spiral dataset generator.

This module provides the SpiralGenerator class for generating multi-spiral
classification datasets using only NumPy operations.
"""

from typing import Dict, Literal, Tuple

import numpy as np

from juniper_data.core.split import shuffle_and_split

from .params import SpiralParams

VERSION = "1.0.0"


class SpiralGenerator:
    """NumPy-only generator for multi-spiral classification datasets.

    All methods are static to ensure the generator is stateless and side-effect free.
    """

    @staticmethod
    def generate(params: SpiralParams) -> Dict[str, np.ndarray]:
        """Generate a complete spiral dataset with train/test splits.

        Main public API for spiral dataset generation.

        Args:
            params: SpiralParams instance defining generation configuration.

        Returns:
            Dictionary containing:
                - X_train: Training features (n_train, 2)
                - y_train: Training labels (n_train, n_spirals)
                - X_test: Test features (n_test, 2)
                - y_test: Test labels (n_test, n_spirals)
                - X_full: Full dataset features (total_points, 2)
                - y_full: Full dataset labels (total_points, n_spirals)
        """
        rng = np.random.default_rng(params.seed)

        X, y = SpiralGenerator._generate_raw(params, rng)

        split_result = shuffle_and_split(
            X=X,
            y=y,
            train_ratio=params.train_ratio,
            test_ratio=params.test_ratio,
            seed=params.seed,
            shuffle=params.shuffle,
        )

        return {
            "X_train": split_result["X_train"],
            "y_train": split_result["y_train"],
            "X_test": split_result["X_test"],
            "y_test": split_result["y_test"],
            "X_full": X,
            "y_full": y,
        }

    @staticmethod
    def _generate_raw(params: SpiralParams, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """Generate raw spiral coordinates and labels.

        Args:
            params: SpiralParams instance defining generation configuration.
            rng: NumPy random generator for reproducibility.

        Returns:
            Tuple of (X, y) where:
                - X: Feature array of shape (total_points, 2)
                - y: One-hot label array of shape (total_points, n_spirals)
        """
        all_coords = []

        for i in range(params.n_spirals):
            angle_offset = 2 * np.pi * i / params.n_spirals
            coords = SpiralGenerator._generate_spiral_coordinates(
                n_points=params.n_points_per_spiral,
                radius=params.radius,
                n_rotations=params.n_rotations,
                angle_offset=angle_offset,
                clockwise=params.clockwise,
                noise=params.noise,
                rng=rng,
                algorithm=params.algorithm,
                origin=params.origin,
            )
            all_coords.append(coords)

        X = np.vstack(all_coords).astype(np.float32)
        y = SpiralGenerator._create_one_hot_labels(
            n_spirals=params.n_spirals,
            n_points_per_spiral=params.n_points_per_spiral,
        )

        return X, y

    @staticmethod
    def _generate_spiral_coordinates(
        n_points: int,
        radius: float,
        n_rotations: float,
        angle_offset: float,
        clockwise: bool,
        noise: float,
        rng: np.random.Generator,
        algorithm: Literal["modern", "legacy_cascor"] = "modern",
        origin: Tuple[float, float] = (0.0, 0.0),
    ) -> np.ndarray:
        """Generate coordinates for a single spiral arm.

        Args:
            n_points: Number of points to generate.
            radius: Maximum radius of the spiral.
            n_rotations: Number of full rotations.
            angle_offset: Angular offset for this spiral arm.
            clockwise: Whether spiral rotates clockwise.
            noise: Noise level to apply.
            rng: NumPy random generator.
            algorithm: Generation algorithm ('modern' or 'legacy_cascor').
            origin: Origin point (x, y) for spiral center.

        Returns:
            Array of shape (n_points, 2) containing x, y coordinates.
        """
        direction = 1 if clockwise else -1

        if algorithm == "legacy_cascor":
            distance = np.sqrt(rng.random(n_points)) * radius
            theta = direction * (distance + angle_offset)
            x = np.cos(theta) * distance + SpiralGenerator._make_noise_uniform(n_points, noise, rng)
            y = np.sin(theta) * distance + SpiralGenerator._make_noise_uniform(n_points, noise, rng)
        else:
            radii = np.linspace(0, radius, n_points)
            theta = np.linspace(0, 2 * np.pi * n_rotations, n_points) + angle_offset
            x = direction * radii * np.cos(theta) + SpiralGenerator._make_noise(n_points, noise, rng)
            y = direction * radii * np.sin(theta) + SpiralGenerator._make_noise(n_points, noise, rng)

        x += origin[0]
        y += origin[1]

        return np.column_stack([x, y]).astype(np.float32)

    @staticmethod
    def _make_noise(n_points: int, noise: float, rng: np.random.Generator) -> np.ndarray:
        """Generate random noise array using normal distribution.

        Args:
            n_points: Number of noise values to generate.
            noise: Noise scale factor.
            rng: NumPy random generator.

        Returns:
            Array of shape (n_points,) containing scaled random noise.
        """
        return rng.standard_normal(n_points) * noise

    @staticmethod
    def _make_noise_uniform(n_points: int, noise: float, rng: np.random.Generator) -> np.ndarray:
        """Generate uniform random noise in [0, noise).

        Args:
            n_points: Number of noise values to generate.
            noise: Noise scale factor.
            rng: NumPy random generator.

        Returns:
            Array of shape (n_points,) containing uniform random noise.
        """
        return rng.random(n_points) * noise

    @staticmethod
    def _create_one_hot_labels(n_spirals: int, n_points_per_spiral: int) -> np.ndarray:
        """Create one-hot encoded labels for spiral classes.

        Args:
            n_spirals: Number of spiral classes.
            n_points_per_spiral: Number of points per spiral class.

        Returns:
            Array of shape (total_points, n_spirals) with one-hot encoding.
        """
        total_points = n_spirals * n_points_per_spiral
        y = np.zeros((total_points, n_spirals), dtype=np.float32)

        for i in range(n_spirals):
            start_idx = i * n_points_per_spiral
            end_idx = (i + 1) * n_points_per_spiral
            y[start_idx:end_idx, i] = 1.0

        return y


def get_schema() -> dict:
    """Return JSON schema describing the generator parameters.

    Useful for API documentation and validation.

    Returns:
        JSON schema dictionary for SpiralParams.
    """
    return SpiralParams.model_json_schema()
