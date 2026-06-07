"""Split and shuffle utilities for dataset partitioning.

This module provides pure NumPy utilities for shuffling and splitting datasets
into train/test sets with reproducible random number generation.
"""

import numpy as np


def shuffle_data(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle X and y arrays together using the same permutation.

    Args:
        X: Feature array of shape (n_samples, ...).
        y: Label array of shape (n_samples, ...).
        rng: NumPy random generator for reproducibility.

    Returns:
        Tuple of shuffled (X, y) arrays with the same permutation applied.

    Raises:
        ValueError: If X and y have different number of samples.
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y must have the same number of samples. Got X.shape[0]={X.shape[0]}, y.shape[0]={y.shape[0]}")

    permutation = rng.permutation(X.shape[0])
    return X[permutation], y[permutation]


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float,
    test_ratio: float,
) -> dict[str, np.ndarray]:
    """Split arrays into train and test sets based on ratios.

    Args:
        X: Feature array of shape (n_samples, ...).
        y: Label array of shape (n_samples, ...).
        train_ratio: Fraction of data for training (0.0 to 1.0).
        test_ratio: Fraction of data for testing (0.0 to 1.0).

    Returns:
        Dictionary with keys "X_train", "y_train", "X_test", "y_test".

    Raises:
        ValueError: If ratios are invalid or X and y have different sample counts.
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y must have the same number of samples. Got X.shape[0]={X.shape[0]}, y.shape[0]={y.shape[0]}")

    if not (0.0 <= train_ratio <= 1.0):
        raise ValueError(f"train_ratio must be between 0 and 1. Got {train_ratio}")

    if not (0.0 <= test_ratio <= 1.0):
        raise ValueError(f"test_ratio must be between 0 and 1. Got {test_ratio}")

    if train_ratio + test_ratio > 1.0:
        raise ValueError(f"train_ratio + test_ratio must not exceed 1.0. Got {train_ratio} + {test_ratio} = {train_ratio + test_ratio}")

    n_samples = X.shape[0]
    n_train = int(np.round(n_samples * train_ratio))
    n_test = int(np.round(n_samples * test_ratio))

    if n_train + n_test > n_samples:
        n_test = n_samples - n_train

    return {
        "X_train": X[:n_train],
        "y_train": y[:n_train],
        "X_test": X[n_train : n_train + n_test],
        "y_test": y[n_train : n_train + n_test],
    }


def shuffle_and_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float,
    test_ratio: float,
    seed: int | None = None,
    shuffle: bool = True,
) -> dict[str, np.ndarray]:
    """Optionally shuffle and then split data into train/test sets.

    High-level function that combines shuffling and splitting operations.
    Uses np.random.Generator for reproducible randomness.

    Args:
        X: Feature array of shape (n_samples, ...).
        y: Label array of shape (n_samples, ...).
        train_ratio: Fraction of data for training (0.0 to 1.0).
        test_ratio: Fraction of data for testing (0.0 to 1.0).
        seed: Random seed for reproducibility. If None, uses non-deterministic seed.
        shuffle: Whether to shuffle data before splitting. Defaults to True.

    Returns:
        Dictionary with keys "X_train", "y_train", "X_test", "y_test".

    Raises:
        ValueError: If ratios are invalid or X and y have different sample counts.
    """
    if shuffle:
        rng = np.random.default_rng(seed)
        X, y = shuffle_data(X, y, rng)

    return split_data(X, y, train_ratio, test_ratio)


def temporal_split_index(n_samples: int, train_ratio: float) -> int:
    """Row index of the chronological (non-shuffled) train/test boundary.

    Rows ``[0, idx)`` are train (earliest dates) and ``[idx, n)`` are test
    (latest). This is the date-ordered split the equities time-series generators
    use; promoting it here lets any time-series generator -- and the sequence
    windower, which maps the returned index to its target-date cut -- reuse one
    tested boundary instead of re-deriving it. For ``n_samples >= 2`` the index
    is clamped to ``[1, n_samples - 1]`` so both splits can be non-empty.

    Args:
        n_samples: total number of date-ordered rows.
        train_ratio: fraction of the earliest rows used for training, in ``(0, 1]``.

    Returns:
        The boundary row index.

    Raises:
        ValueError: if ``train_ratio`` is not in ``(0, 1]``.

    Note:
        A walk-forward (multi-fold rolling) variant is a planned extension
        (WS-1 / juniper-data#168 scope C) and is intentionally not implemented yet.
    """
    if not (0.0 < train_ratio <= 1.0):
        raise ValueError(f"train_ratio must be in (0, 1], got {train_ratio}")
    idx = int(round(n_samples * train_ratio))
    if n_samples >= 2:
        return min(max(idx, 1), n_samples - 1)
    return max(idx, 0)
