"""Default constants for XOR dataset generation.

Mirrors the structure of ``generators/spiral/defaults.py``: per-field
defaults plus validation bounds for the Pydantic ``XorParams`` model.
"""

# XOR Geometry Defaults
XOR_DEFAULT_N_POINTS_PER_QUADRANT: int = 50
XOR_DEFAULT_X_RANGE: float = 1.0
XOR_DEFAULT_Y_RANGE: float = 1.0
XOR_DEFAULT_MARGIN: float = 0.1

# Noise & Randomness Defaults
XOR_DEFAULT_NOISE: float = 0.0

# Dataset Splitting Defaults
XOR_DEFAULT_TRAIN_RATIO: float = 0.8
XOR_DEFAULT_TEST_RATIO: float = 0.2

# Validation Bounds
MIN_N_POINTS_PER_QUADRANT: int = 1
MIN_X_RANGE: float = 0.0  # ``gt=0`` — exclusive
MIN_Y_RANGE: float = 0.0  # ``gt=0`` — exclusive
MIN_MARGIN: float = 0.0
MIN_NOISE: float = 0.0
