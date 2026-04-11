"""Default constants for MNIST dataset generator.

Mirrors the structure of ``generators/spiral/defaults.py``: per-field
defaults for the Pydantic ``MnistParams`` model.
"""

# MNIST Source Defaults
MNIST_DEFAULT_DATASET: str = "mnist"
MNIST_DATASET_CHOICES: tuple[str, ...] = ("mnist", "fashion_mnist")

# MNIST Preprocessing Defaults
MNIST_DEFAULT_FLATTEN: bool = True
MNIST_DEFAULT_NORMALIZE: bool = True
MNIST_DEFAULT_ONE_HOT_LABELS: bool = True

# Dataset Splitting Defaults
MNIST_DEFAULT_TRAIN_RATIO: float = 0.8
MNIST_DEFAULT_TEST_RATIO: float = 0.2

# MNIST Image Shape (used by ``flatten``)
MNIST_IMAGE_HEIGHT: int = 28
MNIST_IMAGE_WIDTH: int = 28
MNIST_FLATTENED_LENGTH: int = 784  # 28 * 28
MNIST_NUM_CLASSES: int = 10
