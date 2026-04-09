"""Default constants for ARC-AGI dataset generator.

Mirrors the structure of ``generators/spiral/defaults.py``: per-field
defaults plus validation bounds for the Pydantic ``ArcAgiParams`` model.
"""

# Source Defaults
ARC_AGI_DEFAULT_SOURCE: str = "huggingface"
ARC_AGI_SOURCE_CHOICES: tuple[str, ...] = ("huggingface", "local")
ARC_AGI_DEFAULT_SUBSET: str = "training"
ARC_AGI_SUBSET_CHOICES: tuple[str, ...] = ("training", "evaluation", "all")

# Padding Defaults
ARC_AGI_DEFAULT_PAD_TO: int = 30
ARC_AGI_DEFAULT_PAD_VALUE: int = -1

# Sample Selection Defaults
ARC_AGI_DEFAULT_INCLUDE_TEST: bool = True
ARC_AGI_DEFAULT_FLATTEN_PAIRS: bool = True

# Dataset Splitting Defaults
ARC_AGI_DEFAULT_TRAIN_RATIO: float = 0.8
ARC_AGI_DEFAULT_TEST_RATIO: float = 0.2

# Validation Bounds
MIN_PAD_TO: int = 1
MAX_PAD_TO: int = 50
MIN_PAD_VALUE: int = -1
MAX_PAD_VALUE: int = 9
