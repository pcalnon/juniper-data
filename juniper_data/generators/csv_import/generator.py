"""CSV/JSON import generator for custom datasets.

This module provides the CsvImportGenerator class for loading
datasets from CSV and JSON files.
"""

import csv
import json
from pathlib import Path

import numpy as np

from juniper_data.core.split import shuffle_and_split

from .params import CsvImportParams

VERSION = "1.0.0"


class CsvImportGenerator:
    """Generator for importing datasets from CSV/JSON files.

    Loads data from local files and converts them to the
    JuniperData format with train/test splits.

    All methods are static to ensure the generator is stateless and side-effect free.
    """

    @staticmethod
    def generate(params: CsvImportParams) -> dict[str, np.ndarray]:
        """Generate a dataset from a CSV/JSON file with train/test splits.

        Args:
            params: CsvImportParams instance defining import configuration.

        Returns:
            Dictionary containing:
                - X_train: Training features
                - y_train: Training labels
                - X_test: Test features
                - y_test: Test labels
                - X_full: Full dataset features
                - y_full: Full dataset labels

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is unsupported.
        """
        X, y = CsvImportGenerator._load_and_preprocess(params)

        split_result = shuffle_and_split(
            X=X,
            y=y,
            train_ratio=params.train_ratio,
            test_ratio=params.test_ratio,
            seed=params.seed,
            shuffle=params.shuffle,
        )

        return {
            "X_train": split_result["X_train"],
            "y_train": split_result["y_train"],
            "X_test": split_result["X_test"],
            "y_test": split_result["y_test"],
            "X_full": X,
            "y_full": y,
        }

    @staticmethod
    def _load_and_preprocess(params: CsvImportParams) -> tuple[np.ndarray, np.ndarray]:
        """Load data from file and preprocess.

        Args:
            params: CsvImportParams instance.

        Returns:
            Tuple of (X, y) arrays.
        """
        path = Path(params.file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {params.file_path}")

        file_format = params.file_format
        if file_format == "auto":
            suffix = path.suffix.lower()
            if suffix == ".csv":
                file_format = "csv"
            elif suffix in (".json", ".jsonl"):
                file_format = "json"
            else:
                raise ValueError(f"Cannot auto-detect format for extension: {suffix}")

        if file_format == "csv":
            data = CsvImportGenerator._load_csv(path, params)
        else:
            data = CsvImportGenerator._load_json(path, params)

        return CsvImportGenerator._convert_to_arrays(data, params)

    @staticmethod
    def _load_csv(path: Path, params: CsvImportParams) -> list[dict]:
        """Load data from CSV file."""
        data = []
        with open(path, newline="", encoding="utf-8") as f:
            if params.header:
                reader = csv.DictReader(f, delimiter=params.delimiter)
            else:
                first_row = next(csv.reader(f, delimiter=params.delimiter))
                f.seek(0)
                fieldnames = [f"col_{i}" for i in range(len(first_row))]
                reader = csv.DictReader(f, fieldnames=fieldnames, delimiter=params.delimiter)

            for row in reader:
                data.append(row)

        return data

    @staticmethod
    def _load_json(path: Path, params: CsvImportParams) -> list[dict]:
        """Load data from JSON file."""
        with open(path, encoding="utf-8") as f:
            content = f.read().strip()

            if content.startswith("["):
                data = json.loads(content)
            else:
                data = [json.loads(line) for line in content.split("\n") if line.strip()]

        return data

    @staticmethod
    def _convert_to_arrays(data: list[dict], params: CsvImportParams) -> tuple[np.ndarray, np.ndarray]:
        """Convert loaded data to numpy arrays."""
        if not data:
            raise ValueError("No data found in file")

        all_columns = list(data[0].keys())

        if params.feature_columns is not None:
            feature_cols = params.feature_columns
        else:
            feature_cols = [c for c in all_columns if c != params.label_column]

        features = []
        labels = []

        for row in data:
            feature_row = []
            for col in feature_cols:
                val = row.get(col, 0)
                try:
                    feature_row.append(float(val))
                except (ValueError, TypeError):
                    feature_row.append(0.0)
            features.append(feature_row)

            label_val = row.get(params.label_column)
            labels.append(label_val)

        X = np.array(features, dtype=np.float32)

        if params.normalize_features:
            X_min = X.min(axis=0, keepdims=True)
            X_max = X.max(axis=0, keepdims=True)
            X_range = X_max - X_min
            X_range[X_range == 0] = 1
            X = (X - X_min) / X_range

        unique_labels = sorted([str(lbl) for lbl in set(labels)])
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        n_classes = len(unique_labels)

        label_indices = np.array([label_to_idx[str(lbl)] for lbl in labels])

        if params.one_hot_labels:
            y = np.zeros((len(labels), n_classes), dtype=np.float32)
            y[np.arange(len(labels)), label_indices] = 1.0
        else:
            y = label_indices.astype(np.float32).reshape(-1, 1)

        return X, y


def get_schema() -> dict:
    """Return JSON schema describing the generator parameters.

    Returns:
        JSON schema dictionary for CsvImportParams.
    """
    return CsvImportParams.model_json_schema()
