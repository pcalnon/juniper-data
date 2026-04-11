"""Default constants for Gaussian blobs dataset generation.

Mirrors the structure of ``generators/spiral/defaults.py``: per-field
defaults plus validation bounds for the Pydantic ``GaussianParams`` model.
"""

# Gaussian Blobs Defaults
GAUSSIAN_DEFAULT_N_CLASSES: int = 2
GAUSSIAN_DEFAULT_N_SAMPLES_PER_CLASS: int = 50
GAUSSIAN_DEFAULT_N_FEATURES: int = 2
GAUSSIAN_DEFAULT_CLASS_STD: float = 1.0
GAUSSIAN_DEFAULT_CENTER_RADIUS: float = 3.0
GAUSSIAN_DEFAULT_NOISE: float = 0.0

# Dataset Splitting Defaults
GAUSSIAN_DEFAULT_TRAIN_RATIO: float = 0.8
GAUSSIAN_DEFAULT_TEST_RATIO: float = 0.2

# Validation Bounds
MIN_N_CLASSES: int = 2
MAX_N_CLASSES: int = 10
MIN_N_SAMPLES_PER_CLASS: int = 1
MIN_N_FEATURES: int = 1
MIN_CENTER_RADIUS: float = 0.0  # ``gt=0`` — exclusive
MIN_NOISE: float = 0.0
