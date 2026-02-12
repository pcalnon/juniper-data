"""ARC-AGI dataset generator.

This module provides the ArcAgiGenerator class for loading ARC-AGI
(Abstraction and Reasoning Corpus) tasks from Hugging Face or local files.
"""

import json
from pathlib import Path

import numpy as np

from juniper_data.core.split import shuffle_and_split

from .params import ArcAgiParams

VERSION = "1.0.0"

try:
    from datasets import load_dataset as hf_load_dataset

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    hf_load_dataset = None  # type: ignore[assignment]


class ArcAgiGenerator:
    """Generator for ARC-AGI reasoning tasks.

    Loads ARC tasks and converts them to padded numpy arrays suitable
    for machine learning. Each task contains input/output grid pairs
    demonstrating a transformation pattern.

    Grid values are integers 0-9 (colors), with -1 used for padding.

    Requires the `datasets` package for HuggingFace source:
    pip install datasets

    All methods are static to ensure the generator is stateless and side-effect free.
    """

    @staticmethod
    def generate(params: ArcAgiParams) -> dict[str, np.ndarray]:
        """Generate an ARC-AGI dataset with train/test splits.

        Args:
            params: ArcAgiParams instance defining loading configuration.

        Returns:
            Dictionary containing:
                - X_train: Training input grids
                - y_train: Training output grids
                - X_test: Test input grids
                - y_test: Test output grids
                - X_full: All input grids
                - y_full: All output grids
                - task_ids: Task identifiers for each sample

        Raises:
            ImportError: If datasets package is not installed (HF source).
            FileNotFoundError: If local path does not exist.
        """
        if params.source == "huggingface":
            tasks = ArcAgiGenerator._load_from_huggingface(params)
        else:
            tasks = ArcAgiGenerator._load_from_local(params)

        X, y, task_ids = ArcAgiGenerator._convert_tasks_to_arrays(tasks, params)

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
            "task_ids": task_ids,
        }

    @staticmethod
    def _load_from_huggingface(params: ArcAgiParams) -> list[dict]:
        """Load ARC tasks from Hugging Face Hub."""
        if not HF_AVAILABLE:
            raise ImportError("Hugging Face datasets package not installed. " "Install with: pip install datasets")

        assert hf_load_dataset is not None

        try:
            ds = hf_load_dataset("fchollet/arc-agi", split="train")
        except Exception:
            ds = hf_load_dataset("multimodal-reasoning-lab/ARC-AGI", split="train")

        tasks: list[dict] = []
        for item in ds:
            task = {
                "task_id": item.get("task_id", f"task_{len(tasks)}"),
                "train": item.get("train", []),
                "test": item.get("test", []),
            }
            tasks.append(task)

        if params.n_tasks is not None:
            if params.seed is not None:
                rng = np.random.default_rng(params.seed)
                indices = rng.choice(len(tasks), min(params.n_tasks, len(tasks)), replace=False)
                tasks = [tasks[i] for i in indices]
            else:
                tasks = tasks[: params.n_tasks]

        return tasks

    @staticmethod
    def _load_from_local(params: ArcAgiParams) -> list[dict]:
        """Load ARC tasks from local JSON files."""
        if params.local_path is None:
            raise ValueError("local_path is required when source='local'")

        base_path = Path(params.local_path)
        if not base_path.exists():
            raise FileNotFoundError(f"Path not found: {params.local_path}")

        tasks = []

        if params.subset in ("training", "all"):
            training_path = base_path / "training"
            if training_path.exists():
                tasks.extend(ArcAgiGenerator._load_json_dir(training_path))

        if params.subset in ("evaluation", "all"):
            eval_path = base_path / "evaluation"
            if eval_path.exists():
                tasks.extend(ArcAgiGenerator._load_json_dir(eval_path))

        if params.n_tasks is not None:
            if params.seed is not None:
                rng = np.random.default_rng(params.seed)
                indices = rng.choice(len(tasks), min(params.n_tasks, len(tasks)), replace=False)
                tasks = [tasks[i] for i in indices]
            else:
                tasks = tasks[: params.n_tasks]

        return tasks

    @staticmethod
    def _load_json_dir(dir_path: Path) -> list[dict]:
        """Load all JSON task files from a directory."""
        tasks = []
        for json_file in sorted(dir_path.glob("*.json")):
            with open(json_file, encoding="utf-8") as f:
                task_data = json.load(f)
                task_data["task_id"] = json_file.stem
                tasks.append(task_data)
        return tasks

    @staticmethod
    def _convert_tasks_to_arrays(tasks: list[dict], params: ArcAgiParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert ARC tasks to padded numpy arrays."""
        inputs = []
        outputs = []
        task_ids = []

        for task in tasks:
            task_id = task.get("task_id", "unknown")

            for pair in task.get("train", []):
                input_grid = ArcAgiGenerator._pad_grid(pair["input"], params.pad_to, params.pad_value)
                output_grid = ArcAgiGenerator._pad_grid(pair["output"], params.pad_to, params.pad_value)
                inputs.append(input_grid)
                outputs.append(output_grid)
                task_ids.append(task_id)

            if params.include_test:
                for pair in task.get("test", []):
                    input_grid = ArcAgiGenerator._pad_grid(pair["input"], params.pad_to, params.pad_value)
                    output_grid = ArcAgiGenerator._pad_grid(
                        pair.get("output", [[params.pad_value]]),
                        params.pad_to,
                        params.pad_value,
                    )
                    inputs.append(input_grid)
                    outputs.append(output_grid)
                    task_ids.append(task_id)

        if not inputs:
            X_arr = np.zeros((0, params.pad_to * params.pad_to), dtype=np.float32)
            y_arr = np.zeros((0, params.pad_to * params.pad_to), dtype=np.float32)
            ids = np.array([], dtype=object)
            return X_arr, y_arr, ids

        X_stacked = np.stack(inputs)
        y_stacked = np.stack(outputs)

        if params.flatten_pairs:
            X_arr = X_stacked.reshape(len(X_stacked), -1).astype(np.float32)
            y_arr = y_stacked.reshape(len(y_stacked), -1).astype(np.float32)
        else:
            X_arr = X_stacked.astype(np.float32)
            y_arr = y_stacked.astype(np.float32)

        return X_arr, y_arr, np.array(task_ids, dtype=object)

    @staticmethod
    def _pad_grid(grid: list[list[int]], pad_to: int, pad_value: int) -> np.ndarray:
        """Pad a grid to the specified size."""
        arr = np.array(grid, dtype=np.int16)
        h, w = arr.shape

        padded = np.full((pad_to, pad_to), pad_value, dtype=np.int16)
        padded[:h, :w] = arr

        return padded


def get_schema() -> dict:
    """Return JSON schema describing the generator parameters.

    Returns:
        JSON schema dictionary for ArcAgiParams.
    """
    return ArcAgiParams.model_json_schema()
