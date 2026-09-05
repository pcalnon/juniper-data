"""ARC-AGI dataset generator.

This module provides the ArcAgiGenerator class for loading ARC-AGI
(Abstraction and Reasoning Corpus) tasks from Hugging Face or local files.
"""

import json
import logging
from pathlib import Path

import numpy as np

from juniper_data.core.constants import CHARSET_UTF8
from juniper_data.core.split import partition_and_assemble, resolve_counts_for_params

from .params import ArcAgiParams

VERSION = "2.0.0"

logger = logging.getLogger(__name__)

try:
    from datasets import load_dataset as hf_load_dataset

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    hf_load_dataset = None  # type: ignore[assignment]


# Hub source for ARC task JSON.
#
# The original primary, ``fchollet/arc-agi``, does not exist on the Hub (its
# ``stat`` reports missing, not a transient error) -- and, on the evidence, never
# did for this code: it was hardcoded as a literal when the generator was
# introduced and never changed, while every test patches ``hf_load_dataset``, so
# no test ever resolved it live. Say "does not exist and was never verified",
# not "no longer exists": the second asserts a removal nobody observed. The
# former fallback,
# ``multimodal-reasoning-lab/ARC-AGI``, is a different kind of dataset entirely --
# reasoning traces and images, with no ARC task columns. This repo carries the
# canonical schema: rows of ``train`` / ``test`` lists of ``{"input", "output"}``
# integer grids, 400 tasks in the ``training`` split, ~348 KB.
#
# NOTE the split name. This dataset's splits are ``training`` / ``evaluation`` /
# ``trial`` -- there is no ``train``. The previous code requested ``split="train"``,
# which would fail here too; the constant keeps the pairing visible so the two
# cannot drift apart.
HF_DATASET_REPO = "lordspline/arc-agi"
HF_DATASET_SPLIT = "training"


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
        tasks = ArcAgiGenerator._load_from_huggingface(params) if params.source == "huggingface" else ArcAgiGenerator._load_from_local(params)

        X, y, task_ids = ArcAgiGenerator._convert_tasks_to_arrays(tasks, params)

        # A generator that produces nothing must say so, not return an empty dataset.
        #
        # Backstop for the whole class, independent of WHY the arrays came back empty:
        # a dead source, a schema change, an over-aggressive filter, or a future
        # source whose rows parse but contain no grid pairs. Without it, a
        # zero-sample result is indistinguishable from a real one to everything
        # downstream -- juniper-data has no empty-dataset check in its API or core
        # layers, so the artifact is persisted, content-addressed, and served to a
        # trainer. Silent-empty is strictly worse than a loud failure: the caller
        # trains on nothing and reads the result as a modelling outcome.
        if X.shape[0] == 0:
            raise RuntimeError(f"ARC-AGI generation produced 0 samples from {len(tasks)} task(s). The source loaded but yielded no usable input/output grid pairs -- this is a source or schema problem, not an empty request.")

        # Carve only: ARC-AGI reads a fixed corpus of tasks, so additive sizing
        # has no way to produce the extra rows it would promise.
        counts = resolve_counts_for_params(params, X.shape[0])

        split_result = partition_and_assemble(X, y, counts, params.seed, params.shuffle)
        split_result["task_ids"] = task_ids
        return split_result

    @staticmethod
    def _load_from_huggingface(params: ArcAgiParams) -> list[dict]:
        """Load ARC tasks from Hugging Face Hub."""
        if not HF_AVAILABLE:
            raise ImportError("Hugging Face datasets package not installed. Install with: pip install datasets")

        # assert hf_load_dataset is not None

        # ERR-13: narrow except to expected network/auth failures so genuine
        # programming errors (TypeError, ValueError, AttributeError, ...) from
        # the ``datasets`` library propagate instead of being silently swapped
        # out for the fallback dataset. ``requests.exceptions.ConnectionError``
        # already inherits from the builtin ``ConnectionError`` and
        # ``requests.exceptions.HTTPError`` from ``OSError`` (via ``IOError``),
        # which transitively covers ``huggingface_hub.errors.HfHubHTTPError``
        # and friends (gated/private/missing repo errors, rate-limits, etc.).
        try:
            ds = hf_load_dataset(HF_DATASET_REPO, split=HF_DATASET_SPLIT)  # nosec B615
        except (ConnectionError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"ARC-AGI dataset {HF_DATASET_REPO!r} (split {HF_DATASET_SPLIT!r}) is unavailable: {type(exc).__name__}: {exc}. Use source='local' with local_path pointing at ARC task JSON if the Hub is unreachable.") from exc

        # Fail loudly on a schema mismatch instead of silently yielding nothing.
        #
        # This guard exists because its absence shipped a real, silent failure. The
        # previous primary, ``fchollet/arc-agi``, does not exist on the Hub (and was
        # never verified to -- see the source note at the top of this module); the
        # fallback, ``multimodal-reasoning-lab/ARC-AGI``, is a multimodal
        # reasoning-trace dataset whose columns are ``Question`` / ``Text Reasoning
        # Trace`` / ``Final Answer`` plus 46 image columns -- no ``train``, no
        # ``test``. The parse below used ``item.get("train", [])``, so all 2000 rows
        # produced empty tasks and ``generate`` returned ``X_full`` with shape
        # ``(0, 900)``: a syntactically valid, entirely empty dataset, built out of
        # 17 232 decoded images that were then discarded. Nothing downstream rejects
        # a zero-sample dataset, so it would have been persisted, content-addressed,
        # and served to a trainer as if real.
        #
        # 17 232, not 92 000. 2000 rows x 46 image columns is 92 000 *cells*, but
        # only 18.7% are populated; a null cell decodes to ``None`` at no cost. The
        # larger figure is arithmetic presented as a count of work done -- which is
        # the same error class as the runtime it was quoted to explain (below).
        #
        # ``.get(key, default)`` is what made it silent. The columns are checked
        # once, against the dataset's declared features, before any row is parsed.
        columns = set(getattr(ds, "column_names", None) or [])
        missing = {"train", "test"} - columns
        if columns and missing:
            raise RuntimeError(f"ARC-AGI dataset {HF_DATASET_REPO!r} (split {HF_DATASET_SPLIT!r}) does not have the expected ARC task schema: missing column(s) {sorted(missing)}; found {sorted(columns)}. An ARC task row must carry 'train' and 'test' lists of {{'input', 'output'}} grids.")

        tasks: list[dict] = []
        for item in ds:
            task = {
                "task_id": item.get("task_id", f"task_{len(tasks)}"),
                "train": item.get("train", []),
                "test": item.get("test", []),
            }
            tasks.append(task)

        if params.n_tasks is not None:
            if params.seed is None:
                tasks = tasks[: params.n_tasks]

            else:
                rng = np.random.default_rng(params.seed)
                indices = rng.choice(len(tasks), min(params.n_tasks, len(tasks)), replace=False)
                tasks = [tasks[i] for i in indices]
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
            if params.seed is None:
                tasks = tasks[: params.n_tasks]

            else:
                rng = np.random.default_rng(params.seed)
                indices = rng.choice(len(tasks), min(params.n_tasks, len(tasks)), replace=False)
                tasks = [tasks[i] for i in indices]
        return tasks

    @staticmethod
    def _load_json_dir(dir_path: Path) -> list[dict]:
        """Load all JSON task files from a directory."""
        tasks = []
        for json_file in sorted(dir_path.glob("*.json")):
            with open(json_file, encoding=CHARSET_UTF8) as f:
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
