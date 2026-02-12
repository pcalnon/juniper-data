"""Core NumPy-only Gaussian blobs dataset generator.

This module provides the GaussianGenerator class for generating
mixture-of-Gaussians classification datasets using only NumPy operations.
"""

import numpy as np

from juniper_data.core.split import shuffle_and_split

from .params import GaussianParams

VERSION = "1.0.0"


class GaussianGenerator:
    """NumPy-only generator for Gaussian blobs classification datasets.

    Generates a mixture-of-Gaussians dataset with configurable centers,
    standard deviations, and noise levels. Each class is sampled from
    a multivariate Gaussian distribution.

    All methods are static to ensure the generator is stateless and side-effect free.
    """

    @staticmethod
    def generate(params: GaussianParams) -> dict[str, np.ndarray]:
        """Generate a complete Gaussian blobs dataset with train/test splits.

        Args:
            params: GaussianParams instance defining generation configuration.

        Returns:
            Dictionary containing:
                - X_train: Training features (n_train, n_features)
                - y_train: Training labels (n_train, n_classes)
                - X_test: Test features (n_test, n_features)
                - y_test: Test labels (n_test, n_classes)
                - X_full: Full dataset features (total_points, n_features)
                - y_full: Full dataset labels (total_points, n_classes)
        """
        rng = np.random.default_rng(params.seed)

        X, y = GaussianGenerator._generate_raw(params, rng)

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
    def _generate_raw(params: GaussianParams, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Generate raw Gaussian blobs coordinates and labels.

        Args:
            params: GaussianParams instance defining generation configuration.
            rng: NumPy random generator for reproducibility.

        Returns:
            Tuple of (X, y) where:
                - X: Feature array of shape (total_points, n_features)
                - y: One-hot label array of shape (total_points, n_classes)
        """
        centers = GaussianGenerator._get_centers(params, rng)
        stds = GaussianGenerator._get_stds(params)

        total_points = params.n_classes * params.n_samples_per_class
        X = np.zeros((total_points, params.n_features), dtype=np.float32)
        y = np.zeros((total_points, params.n_classes), dtype=np.float32)

        for i in range(params.n_classes):
            start_idx = i * params.n_samples_per_class
            end_idx = start_idx + params.n_samples_per_class

            class_std = stds[i] if len(stds) > i else stds[0]
            samples = rng.standard_normal((params.n_samples_per_class, params.n_features))
            X[start_idx:end_idx] = samples * class_std + centers[i]

            y[start_idx:end_idx, i] = 1.0

        if params.noise > 0:
            X += rng.standard_normal(X.shape).astype(np.float32) * params.noise

        return X, y

    @staticmethod
    def _get_centers(params: GaussianParams, rng: np.random.Generator) -> np.ndarray:
        """Get or generate class centers.

        Args:
            params: GaussianParams instance.
            rng: NumPy random generator (unused if centers provided).

        Returns:
            Array of shape (n_classes, n_features) containing class centers.
        """
        if params.centers is not None:
            centers = np.array(params.centers, dtype=np.float32)
            if centers.shape[0] != params.n_classes:
                raise ValueError(f"Number of centers ({centers.shape[0]}) must match n_classes ({params.n_classes})")
            if centers.shape[1] != params.n_features:
                raise ValueError(f"Center dimensions ({centers.shape[1]}) must match n_features ({params.n_features})")
            return centers

        centers = np.zeros((params.n_classes, params.n_features), dtype=np.float32)
        angles = np.linspace(0, 2 * np.pi, params.n_classes, endpoint=False)

        for i, angle in enumerate(angles):
            centers[i, 0] = params.center_radius * np.cos(angle)
            if params.n_features > 1:
                centers[i, 1] = params.center_radius * np.sin(angle)

        return centers

    @staticmethod
    def _get_stds(params: GaussianParams) -> list[float]:
        """Get standard deviations for each class.

        Args:
            params: GaussianParams instance.

        Returns:
            List of standard deviations, one per class.
        """
        if isinstance(params.class_std, list):
            return params.class_std
        return [params.class_std] * params.n_classes


def get_schema() -> dict:
    """Return JSON schema describing the generator parameters.

    Returns:
        JSON schema dictionary for GaussianParams.
    """
    return GaussianParams.model_json_schema()
