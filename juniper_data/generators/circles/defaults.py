"""Default constants for concentric circles dataset generation.

Mirrors the structure of ``generators/spiral/defaults.py``: per-field
defaults plus validation bounds for the Pydantic ``CirclesParams`` model.
"""

# Circles Geometry Defaults
CIRCLES_DEFAULT_N_SAMPLES: int = 100
CIRCLES_DEFAULT_OUTER_RADIUS: float = 1.0
CIRCLES_DEFAULT_FACTOR: float = 0.5
CIRCLES_DEFAULT_INNER_RATIO: float = 0.5

# Noise Defaults
CIRCLES_DEFAULT_NOISE: float = 0.0

# Dataset Splitting Defaults
CIRCLES_DEFAULT_TRAIN_RATIO: float = 0.8
CIRCLES_DEFAULT_TEST_RATIO: float = 0.2

# Validation Bounds
MIN_N_SAMPLES: int = 2
MIN_OUTER_RADIUS: float = 0.0  # ``gt=0`` — exclusive
MIN_FACTOR: float = 0.0  # ``gt=0`` — exclusive
MAX_FACTOR: float = 1.0  # ``lt=1`` — exclusive
MIN_NOISE: float = 0.0
MIN_INNER_RATIO: float = 0.0  # ``gt=0`` — exclusive
MAX_INNER_RATIO: float = 1.0
