#!/usr/bin/env python
"""
Golden Dataset Generator for JuniperData Parity Testing

This script generates golden reference datasets from the existing JuniperCascor
SpiralProblem implementation for use in validating the new JuniperData implementation.

Run this script from the JuniperCascor environment to generate the golden datasets.

Usage:
    cd /home/pcalnon/Development/python/Juniper/JuniperCascor/juniper_cascor/src
    python /home/pcalnon/Development/python/Juniper/JuniperData/tests/fixtures/generate_golden_datasets.py
"""

import json
import sys
from pathlib import Path

import numpy as np

# Append JuniperCascor source directory for local script execution
sys.path.insert(0, "/home/pcalnon/Development/python/Juniper/JuniperCascor/juniper_cascor/src")

from spiral_problem.spiral_problem import SpiralProblem  # noqa: E402

GOLDEN_DATASETS_DIR = Path("/home/pcalnon/Development/python/Juniper/JuniperData/tests/fixtures/golden_datasets")

DATASET_CONFIGS = [
    {
        "name": "2_spiral",
        "n_spirals": 2,
        "n_points": 100,
        "noise": 0.1,
        "seed": 42,
        "train_ratio": 0.8,
        "test_ratio": 0.2,
    },
    {
        "name": "3_spiral",
        "n_spirals": 3,
        "n_points": 50,
        "noise": 0.05,
        "seed": 42,
        "train_ratio": 0.8,
        "test_ratio": 0.2,
    },
]


def generate_golden_dataset(config: dict) -> dict:
    """Generate a golden dataset with the specified configuration."""
    np.random.seed(config["seed"])

    import torch  # noqa: E402

    torch.manual_seed(config["seed"])

    problem = SpiralProblem(
        _SpiralProblem__n_spirals=config["n_spirals"],
        _SpiralProblem__n_points=config["n_points"],
        _SpiralProblem__noise=config["noise"],
        _SpiralProblem__random_seed=config["seed"],
        _SpiralProblem__train_ratio=config["train_ratio"],
        _SpiralProblem__test_ratio=config["test_ratio"],
    )

    (X_train, y_train), (X_test, y_test), (X_full, y_full) = problem.generate_n_spiral_dataset(
        n_spirals=config["n_spirals"],
        n_points=config["n_points"],
        noise_level=config["noise"],
        train_ratio=config["train_ratio"],
        test_ratio=config["test_ratio"],
    )

    X_train_np = X_train.numpy()
    y_train_np = y_train.numpy()
    X_test_np = X_test.numpy()
    y_test_np = y_test.numpy()

    metadata = {
        "config": config,
        "shapes": {
            "X_train": list(X_train_np.shape),
            "y_train": list(y_train_np.shape),
            "X_test": list(X_test_np.shape),
            "y_test": list(y_test_np.shape),
        },
        "dtypes": {
            "X_train": str(X_train_np.dtype),
            "y_train": str(y_train_np.dtype),
            "X_test": str(X_test_np.dtype),
            "y_test": str(y_test_np.dtype),
        },
        "class_distribution": {
            "train": compute_class_distribution(y_train_np),
            "test": compute_class_distribution(y_test_np),
        },
        "value_ranges": {
            "X_train": {"min": float(X_train_np.min()), "max": float(X_train_np.max())},
            "X_test": {"min": float(X_test_np.min()), "max": float(X_test_np.max())},
        },
    }

    return {
        "X_train": X_train_np,
        "y_train": y_train_np,
        "X_test": X_test_np,
        "y_test": y_test_np,
        "metadata": metadata,
    }


def compute_class_distribution(y: np.ndarray) -> dict:
    """Compute class distribution from one-hot encoded labels."""
    class_indices = np.argmax(y, axis=1)
    unique, counts = np.unique(class_indices, return_counts=True)
    return {f"class_{int(c)}": int(cnt) for c, cnt in zip(unique, counts)}


def save_golden_dataset(data: dict, name: str) -> None:
    """Save golden dataset as NPZ file with metadata JSON."""
    GOLDEN_DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    npz_path = GOLDEN_DATASETS_DIR / f"{name}.npz"
    np.savez(
        npz_path,
        X_train=data["X_train"],
        y_train=data["y_train"],
        X_test=data["X_test"],
        y_test=data["y_test"],
    )
    print(f"Saved: {npz_path}")

    metadata_path = GOLDEN_DATASETS_DIR / f"{name}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(data["metadata"], f, indent=2)
    print(f"Saved: {metadata_path}")


def print_dataset_info(data: dict, name: str) -> None:
    """Print dataset information for verification."""
    meta = data["metadata"]
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"{'='*60}")
    print("Configuration:")
    for key, value in meta["config"].items():
        print(f"  {key}: {value}")
    print("\nShapes:")
    for key, shape in meta["shapes"].items():
        print(f"  {key}: {shape}")
    print("\nDtypes:")
    for key, dtype in meta["dtypes"].items():
        print(f"  {key}: {dtype}")
    print("\nClass Distribution:")
    for split, dist in meta["class_distribution"].items():
        print(f"  {split}: {dist}")
    print("\nValue Ranges:")
    for key, ranges in meta["value_ranges"].items():
        print(f"  {key}: min={ranges['min']:.6f}, max={ranges['max']:.6f}")


def main():
    """Generate all golden datasets."""
    print("Generating Golden Reference Datasets")
    print("=" * 60)

    for config in DATASET_CONFIGS:
        print(f"\nGenerating {config['name']} dataset...")
        data = generate_golden_dataset(config)
        print_dataset_info(data, config["name"])
        save_golden_dataset(data, config["name"])

    print("\n" + "=" * 60)
    print("Golden dataset generation complete!")
    print(f"Output directory: {GOLDEN_DATASETS_DIR}")


if __name__ == "__main__":
    main()
