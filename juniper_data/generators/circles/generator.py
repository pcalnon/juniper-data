"""Core NumPy-only concentric circles dataset generator.

This module provides the CirclesGenerator class for generating
concentric circles classification datasets using only NumPy operations.
"""

import numpy as np

from juniper_data.core.partition_params import rescale_generator_params
from juniper_data.core.split import partition_and_assemble, resolve_counts_for_params

from .params import CirclesParams

VERSION = "3.0.0"


class CirclesGenerator:
    """NumPy-only generator for concentric circles classification datasets.

    Generates a binary classification dataset with points distributed on
    two concentric circles. The outer circle is class 0, and the inner
    circle is class 1.

    All methods are static to ensure the generator is stateless and side-effect free.
    """

    @staticmethod
    def generate(params: CirclesParams) -> dict[str, np.ndarray]:
        """Generate a complete concentric circles dataset with train/test splits.

        Args:
            params: CirclesParams instance defining generation configuration.

        Returns:
            Dictionary containing:
                - X_train: Training features (n_train, 2)
                - y_train: Training labels (n_train, 2)
                - X_val: Validation features (n_val, 2)
                - y_val: Validation labels (n_val, 2)
                - X_test: Test features (n_test, 2)
                - y_test: Test labels (n_test, 2)
        """
        rng = np.random.default_rng(params.seed)

        counts = resolve_counts_for_params(params, params.n_samples)
        # Additive sizing needs more raw rows than the size knob names,
        # because that knob now denotes the TRAIN count alone.
        gen_params = rescale_generator_params(params, n_samples=counts["n_raw_required"])

        X, y = CirclesGenerator._generate_raw(gen_params, rng)

        return partition_and_assemble(X, y, counts, params.seed, params.shuffle)

    @staticmethod
    def _generate_raw(params: CirclesParams, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Generate raw concentric circles coordinates and labels.

        Args:
            params: CirclesParams instance defining generation configuration.
            rng: NumPy random generator for reproducibility.

        Returns:
            Tuple of (X, y) where:
                - X: Feature array of shape (n_samples, 2)
                - y: One-hot label array of shape (n_samples, 2)
        """
        n_inner = int(params.n_samples * params.inner_ratio)
        n_outer = params.n_samples - n_inner

        inner_radius = params.outer_radius * params.factor

        outer_angles = rng.uniform(0, 2 * np.pi, n_outer)
        outer_x = params.outer_radius * np.cos(outer_angles)
        outer_y = params.outer_radius * np.sin(outer_angles)
        outer_points = np.column_stack([outer_x, outer_y])

        inner_angles = rng.uniform(0, 2 * np.pi, n_inner)
        inner_x = inner_radius * np.cos(inner_angles)
        inner_y = inner_radius * np.sin(inner_angles)
        inner_points = np.column_stack([inner_x, inner_y])

        X = np.vstack([outer_points, inner_points])

        if params.noise > 0:
            X += rng.standard_normal(X.shape) * params.noise

        X = X.astype(np.float32)

        y = np.zeros((params.n_samples, 2), dtype=np.float32)
        y[:n_outer, 0] = 1.0
        y[n_outer:, 1] = 1.0

        return X, y


def get_schema() -> dict:
    """Return JSON schema describing the generator parameters.

    Returns:
        JSON schema dictionary for CirclesParams.
    """
    return CirclesParams.model_json_schema()
