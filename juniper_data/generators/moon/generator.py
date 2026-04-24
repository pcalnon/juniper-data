"""Core NumPy-only two-moons dataset generator.

Provides the ``MoonGenerator`` class for generating the classic
"two-moons" binary-classification dataset using only NumPy. Introduced
to satisfy XREPO-01b / DC-02 (client referenced a server-side moon
generator that did not exist).
"""

import numpy as np

from juniper_data.core.split import shuffle_and_split

from .params import MoonParams

VERSION = "1.0.0"


class MoonGenerator:
    """NumPy-only generator for two-moons classification datasets.

    Each half-moon (upper and lower) is a distinct class. The lower
    moon is shifted right by +1.0 on x and down by -0.5 on y so the
    two moons interleave — a standard nonlinear benchmark.

    All methods are static; the generator is stateless and side-effect
    free.
    """

    LOWER_X_OFFSET: float = 1.0
    LOWER_Y_OFFSET: float = 0.5

    @staticmethod
    def generate(params: MoonParams) -> dict[str, np.ndarray]:
        """Generate a complete two-moons dataset with train/test splits.

        Args:
            params: ``MoonParams`` instance defining the generation config.

        Returns:
            Dictionary with keys ``X_train``, ``y_train``, ``X_test``,
            ``y_test``, ``X_full``, ``y_full``. All arrays are float32;
            labels are one-hot encoded with shape ``(n_samples, 2)``.
        """
        rng = np.random.default_rng(params.seed)

        X, y = MoonGenerator._generate_raw(params, rng)

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
    def _generate_raw(params: MoonParams, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Generate raw two-moons coordinates and one-hot labels.

        Args:
            params: Generation parameters.
            rng: NumPy random generator for reproducibility.

        Returns:
            Tuple ``(X, y)`` where ``X`` has shape ``(n_samples, 2)``
            and ``y`` has shape ``(n_samples, 2)`` (one-hot).
        """
        n_upper = params.n_samples // 2
        n_lower = params.n_samples - n_upper

        upper_angles = np.linspace(0.0, np.pi, n_upper)
        upper_x = np.cos(upper_angles)
        upper_y = np.sin(upper_angles)
        upper_points = np.column_stack([upper_x, upper_y])

        lower_angles = np.linspace(0.0, np.pi, n_lower)
        lower_x = MoonGenerator.LOWER_X_OFFSET - np.cos(lower_angles)
        lower_y = MoonGenerator.LOWER_Y_OFFSET - np.sin(lower_angles)
        lower_points = np.column_stack([lower_x, lower_y])

        X = np.vstack([upper_points, lower_points])

        if params.noise > 0:
            X += rng.standard_normal(X.shape) * params.noise

        X = X.astype(np.float32)

        y = np.zeros((params.n_samples, 2), dtype=np.float32)
        y[:n_upper, 0] = 1.0
        y[n_upper:, 1] = 1.0

        return X, y


def get_schema() -> dict:
    """Return JSON schema describing ``MoonParams``."""
    return MoonParams.model_json_schema()
