"""Default constants for checkerboard dataset generation.

Mirrors the structure of ``generators/spiral/defaults.py``: per-field
defaults plus validation bounds for the Pydantic ``CheckerboardParams`` model.
"""

# Checkerboard Geometry Defaults
CHECKERBOARD_DEFAULT_N_SAMPLES: int = 200
CHECKERBOARD_DEFAULT_N_SQUARES: int = 4
CHECKERBOARD_DEFAULT_X_RANGE: tuple[float, float] = (0.0, 1.0)
CHECKERBOARD_DEFAULT_Y_RANGE: tuple[float, float] = (0.0, 1.0)

# Noise Defaults
CHECKERBOARD_DEFAULT_NOISE: float = 0.0

# Dataset Splitting Defaults
CHECKERBOARD_DEFAULT_TRAIN_RATIO: float = 0.8
CHECKERBOARD_DEFAULT_TEST_RATIO: float = 0.2

# Validation Bounds
MIN_N_SAMPLES: int = 2
MIN_N_SQUARES: int = 2
MAX_N_SQUARES: int = 16
MIN_NOISE: float = 0.0
