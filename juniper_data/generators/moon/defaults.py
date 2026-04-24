"""Default constants for two-moons dataset generation.

Mirrors ``generators/circles/defaults.py``: per-field defaults plus
validation bounds for the Pydantic ``MoonParams`` model.
"""

# Moon Geometry Defaults
MOON_DEFAULT_N_SAMPLES: int = 200

# Noise Defaults
MOON_DEFAULT_NOISE: float = 0.1

# Dataset Splitting Defaults
MOON_DEFAULT_TRAIN_RATIO: float = 0.8
MOON_DEFAULT_TEST_RATIO: float = 0.2

# Validation Bounds
MIN_N_SAMPLES: int = 2
MIN_NOISE: float = 0.0
