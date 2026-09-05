"""Core NumPy-only XOR dataset generator.

This module provides the XorGenerator class for generating XOR
classification datasets using only NumPy operations.
"""

import numpy as np

from juniper_data.core.split import partition_and_assemble, per_unit_count, resolve_counts_for_params

from .params import XorParams

VERSION = "2.0.0"

#: XOR is four quadrants by construction; the size knob is per-quadrant.
_N_QUADRANTS = 4


class XorGenerator:
    """NumPy-only generator for XOR classification datasets.

    The XOR dataset consists of 4 quadrants around the origin:
    - Quadrant 1 (++): x > 0, y > 0 -> Class 0
    - Quadrant 2 (-+): x < 0, y > 0 -> Class 1
    - Quadrant 3 (--): x < 0, y < 0 -> Class 0
    - Quadrant 4 (+-): x > 0, y < 0 -> Class 1

    All methods are static to ensure the generator is stateless and side-effect free.
    """

    @staticmethod
    def generate(params: XorParams) -> dict[str, np.ndarray]:
        """Generate a complete XOR dataset with train/test splits.

        Args:
            params: XorParams instance defining generation configuration.

        Returns:
            Dictionary containing:
                - X_train: Training features (n_train, 2)
                - y_train: Training labels (n_train, 2)
                - X_val: Validation features (n_val, 2)
                - y_val: Validation labels (n_val, 2)
                - X_test: Test features (n_test, 2)
                - y_test: Test labels (n_test, 2)
                - X_full: Full dataset features (total_points, 2)
                - y_full: Full dataset labels (total_points, 2)
        """
        rng = np.random.default_rng(params.seed)

        counts = resolve_counts_for_params(params, _N_QUADRANTS * params.n_points_per_quadrant)
        # Additive sizing needs more raw rows than the size knob names,
        # because that knob now denotes the TRAIN count alone.
        gen_params = params.model_copy(update={"n_points_per_quadrant": per_unit_count(counts["n_raw_required"], _N_QUADRANTS)})

        X, y = XorGenerator._generate_raw(gen_params, rng)

        return partition_and_assemble(X, y, counts, params.seed, params.shuffle)

    @staticmethod
    def _generate_raw(params: XorParams, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Generate raw XOR coordinates and labels.

        Args:
            params: XorParams instance defining generation configuration.
            rng: NumPy random generator for reproducibility.

        Returns:
            Tuple of (X, y) where:
                - X: Feature array of shape (total_points, 2)
                - y: One-hot label array of shape (total_points, 2)
        """
        n = params.n_points_per_quadrant

        q1 = XorGenerator._generate_quadrant(
            n_points=n,
            x_min=params.margin,
            x_max=params.x_range,
            y_min=params.margin,
            y_max=params.y_range,
            rng=rng,
        )
        q2 = XorGenerator._generate_quadrant(
            n_points=n,
            x_min=-params.x_range,
            x_max=-params.margin,
            y_min=params.margin,
            y_max=params.y_range,
            rng=rng,
        )
        q3 = XorGenerator._generate_quadrant(
            n_points=n,
            x_min=-params.x_range,
            x_max=-params.margin,
            y_min=-params.y_range,
            y_max=-params.margin,
            rng=rng,
        )
        q4 = XorGenerator._generate_quadrant(
            n_points=n,
            x_min=params.margin,
            x_max=params.x_range,
            y_min=-params.y_range,
            y_max=-params.margin,
            rng=rng,
        )

        X = np.vstack([q1, q2, q3, q4])

        if params.noise > 0:
            X += rng.standard_normal(X.shape) * params.noise

        X = X.astype(np.float32)

        y = np.zeros((4 * n, 2), dtype=np.float32)
        y[0 * n : 1 * n, 0] = 1.0
        y[1 * n : 2 * n, 1] = 1.0
        y[2 * n : 3 * n, 0] = 1.0
        y[3 * n : 4 * n, 1] = 1.0

        return X, y

    @staticmethod
    def _generate_quadrant(
        n_points: int,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate points uniformly distributed in a rectangular region.

        Args:
            n_points: Number of points to generate.
            x_min: Minimum x value.
            x_max: Maximum x value.
            y_min: Minimum y value.
            y_max: Maximum y value.
            rng: NumPy random generator.

        Returns:
            Array of shape (n_points, 2) containing x, y coordinates.
        """
        x = rng.uniform(x_min, x_max, n_points)
        y = rng.uniform(y_min, y_max, n_points)
        return np.column_stack([x, y])


def get_schema() -> dict:
    """Return JSON schema describing the generator parameters.

    Returns:
        JSON schema dictionary for XorParams.
    """
    return XorParams.model_json_schema()
