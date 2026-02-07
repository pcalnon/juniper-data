"""Core NumPy-only checkerboard dataset generator.

This module provides the CheckerboardGenerator class for generating
checkerboard classification datasets using only NumPy operations.
"""

import numpy as np

from juniper_data.core.split import shuffle_and_split

from .params import CheckerboardParams

VERSION = "1.0.0"


class CheckerboardGenerator:
    """NumPy-only generator for checkerboard classification datasets.

    Generates a 2D checkerboard pattern where alternating squares
    belong to different classes. Points are uniformly distributed
    across the grid.

    All methods are static to ensure the generator is stateless and side-effect free.
    """

    @staticmethod
    def generate(params: CheckerboardParams) -> dict[str, np.ndarray]:
        """Generate a complete checkerboard dataset with train/test splits.

        Args:
            params: CheckerboardParams instance defining generation configuration.

        Returns:
            Dictionary containing:
                - X_train: Training features (n_train, 2)
                - y_train: Training labels (n_train, 2)
                - X_test: Test features (n_test, 2)
                - y_test: Test labels (n_test, 2)
                - X_full: Full dataset features (n_samples, 2)
                - y_full: Full dataset labels (n_samples, 2)
        """
        rng = np.random.default_rng(params.seed)

        X, y = CheckerboardGenerator._generate_raw(params, rng)

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
    def _generate_raw(
        params: CheckerboardParams, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate raw checkerboard coordinates and labels.

        Args:
            params: CheckerboardParams instance defining generation configuration.
            rng: NumPy random generator for reproducibility.

        Returns:
            Tuple of (X, y) where:
                - X: Feature array of shape (n_samples, 2)
                - y: One-hot label array of shape (n_samples, 2)
        """
        x_min, x_max = params.x_range
        y_min, y_max = params.y_range

        x = rng.uniform(x_min, x_max, params.n_samples)
        y_coord = rng.uniform(y_min, y_max, params.n_samples)

        X = np.column_stack([x, y_coord])

        if params.noise > 0:
            X += rng.standard_normal(X.shape) * params.noise

        X = X.astype(np.float32)

        x_step = (x_max - x_min) / params.n_squares
        y_step = (y_max - y_min) / params.n_squares

        x_idx = np.floor((x - x_min) / x_step).astype(int)
        y_idx = np.floor((y_coord - y_min) / y_step).astype(int)

        x_idx = np.clip(x_idx, 0, params.n_squares - 1)
        y_idx = np.clip(y_idx, 0, params.n_squares - 1)

        labels = (x_idx + y_idx) % 2

        y = np.zeros((params.n_samples, 2), dtype=np.float32)
        y[labels == 0, 0] = 1.0
        y[labels == 1, 1] = 1.0

        return X, y


def get_schema() -> dict:
    """Return JSON schema describing the generator parameters.

    Returns:
        JSON schema dictionary for CheckerboardParams.
    """
    return CheckerboardParams.model_json_schema()
