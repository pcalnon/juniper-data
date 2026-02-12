"""Default constants for spiral dataset generation.

This module defines all default constants and validation bounds for the spiral
dataset generator, migrated from JuniperCascor constants_problem.py.
"""

# Spiral Geometry Defaults
SPIRAL_DEFAULT_N_SPIRALS: int = 2
SPIRAL_DEFAULT_N_POINTS: int = 97
SPIRAL_DEFAULT_N_ROTATIONS: float = 3.0
SPIRAL_DEFAULT_CLOCKWISE: bool = True
SPIRAL_DEFAULT_DISTRIBUTION: float = 0.80
SPIRAL_DEFAULT_ORIGIN: tuple[float, float] = (0.0, 0.0)
SPIRAL_DEFAULT_RADIUS: float = 10.0

# Noise & Randomness Defaults
SPIRAL_DEFAULT_NOISE: float = 0.25
SPIRAL_DEFAULT_RANDOM_VALUE_SCALE: float = 0.1
SPIRAL_DEFAULT_SEED: int = 42

# Dataset Splitting Defaults
SPIRAL_DEFAULT_TRAIN_RATIO: float = 0.8
SPIRAL_DEFAULT_TEST_RATIO: float = 0.2

# Validation Bounds - Spirals
MIN_SPIRALS: int = 2
MAX_SPIRALS: int = 10

# Validation Bounds - Points
MIN_POINTS: int = 10
MAX_POINTS: int = 10000

# Validation Bounds - Rotations
MIN_ROTATIONS: float = 0.5
MAX_ROTATIONS: float = 10.0

# Validation Bounds - Noise
MIN_NOISE: float = 0.0
MAX_NOISE: float = 2.0
