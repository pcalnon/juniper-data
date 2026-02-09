"""Unit tests for the ARC-AGI dataset generator."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from juniper_data.generators.arc_agi.params import ArcAgiParams


def _make_sample_tasks(n_tasks=3, train_pairs=2, test_pairs=1):
    """Create sample ARC tasks for testing."""
    tasks = []
    for i in range(n_tasks):
        task = {
            "task_id": f"task_{i}",
            "train": [
                {
                    "input": [[j, j + 1], [j + 2, j + 3]],
                    "output": [[j + 3, j + 2], [j + 1, j]],
                }
                for j in range(train_pairs)
            ],
            "test": [
                {
                    "input": [[1, 2], [3, 4]],
                    "output": [[4, 3], [2, 1]],
                }
                for _ in range(test_pairs)
            ],
        }
        tasks.append(task)
    return tasks


def _make_mock_hf_dataset(tasks):
    """Create a mock HuggingFace dataset from task list."""
    mock_ds = MagicMock()
    mock_ds.__len__ = MagicMock(return_value=len(tasks))
    mock_ds.__iter__ = MagicMock(return_value=iter(tasks))

    return mock_ds


@pytest.fixture
def mock_hf_load():
    """Patch HF_AVAILABLE and hf_load_dataset for the arc_agi generator module."""
    mock_load = MagicMock()

    with patch("juniper_data.generators.arc_agi.generator.HF_AVAILABLE", True):
        with patch("juniper_data.generators.arc_agi.generator.hf_load_dataset", mock_load):
            yield mock_load


@pytest.mark.unit
@pytest.mark.generators
class TestArcAgiParams:
    """Tests for ArcAgiParams validation."""

    def test_default_params(self) -> None:
        """Default parameters are valid."""
        params = ArcAgiParams()
        assert params.source == "huggingface"
        assert params.local_path is None
        assert params.subset == "training"
        assert params.n_tasks is None
        assert params.pad_to == 30
        assert params.pad_value == -1
        assert params.include_test is True
        assert params.flatten_pairs is True
        assert params.train_ratio == 0.8

    def test_custom_params(self) -> None:
        """Custom parameters are accepted."""
        params = ArcAgiParams(
            source="local",
            local_path="/data/arc",
            subset="evaluation",
            n_tasks=10,
            pad_to=15,
            pad_value=0,
            include_test=False,
            flatten_pairs=False,
            seed=42,
        )
        assert params.source == "local"
        assert params.local_path == "/data/arc"
        assert params.n_tasks == 10

    def test_invalid_pad_to(self) -> None:
        """pad_to must be >= 1 and <= 50."""
        with pytest.raises(ValueError):
            ArcAgiParams(pad_to=0)
        with pytest.raises(ValueError):
            ArcAgiParams(pad_to=51)

    def test_invalid_n_tasks(self) -> None:
        """n_tasks must be >= 1."""
        with pytest.raises(ValueError):
            ArcAgiParams(n_tasks=0)

    def test_invalid_train_ratio(self) -> None:
        """train_ratio must be in (0, 1]."""
        with pytest.raises(ValueError):
            ArcAgiParams(train_ratio=0)


@pytest.mark.unit
@pytest.mark.generators
class TestArcAgiGeneratorHuggingFace:
    """Tests for ARC-AGI generation from HuggingFace source."""

    def test_generate_from_hf(self, mock_hf_load) -> None:
        """Generate produces correct output structure from HuggingFace."""
        tasks = _make_sample_tasks(n_tasks=2, train_pairs=2, test_pairs=1)
        mock_ds = _make_mock_hf_dataset(tasks)
        mock_hf_load.return_value = mock_ds

        from juniper_data.generators.arc_agi.generator import ArcAgiGenerator

        params = ArcAgiParams(source="huggingface", seed=42)
        result = ArcAgiGenerator.generate(params)

        assert "X_train" in result
        assert "y_train" in result
        assert "X_test" in result
        assert "y_test" in result
        assert "X_full" in result
        assert "y_full" in result
        assert "task_ids" in result

    def test_generate_correct_shapes_flattened(self, mock_hf_load) -> None:
        """Flattened output has correct shape (n_pairs, pad_to*pad_to)."""
        tasks = _make_sample_tasks(n_tasks=2, train_pairs=2, test_pairs=1)
        mock_ds = _make_mock_hf_dataset(tasks)
        mock_hf_load.return_value = mock_ds

        from juniper_data.generators.arc_agi.generator import ArcAgiGenerator

        params = ArcAgiParams(pad_to=5, flatten_pairs=True, seed=42)
        result = ArcAgiGenerator.generate(params)

        n_total = 2 * 2 + 2 * 1  # 2 tasks * 2 train + 2 tasks * 1 test
        assert result["X_full"].shape == (n_total, 25)
        assert result["y_full"].shape == (n_total, 25)

    def test_generate_correct_shapes_not_flattened(self, mock_hf_load) -> None:
        """Non-flattened output has correct shape (n_pairs, pad_to, pad_to)."""
        tasks = _make_sample_tasks(n_tasks=2, train_pairs=2, test_pairs=1)
        mock_ds = _make_mock_hf_dataset(tasks)
        mock_hf_load.return_value = mock_ds

        from juniper_data.generators.arc_agi.generator import ArcAgiGenerator

        params = ArcAgiParams(pad_to=5, flatten_pairs=False, seed=42)
        result = ArcAgiGenerator.generate(params)

        n_total = 2 * 2 + 2 * 1
        assert result["X_full"].shape == (n_total, 5, 5)

    def test_generate_without_test_pairs(self, mock_hf_load) -> None:
        """Generate without test pairs only uses train pairs."""
        tasks = _make_sample_tasks(n_tasks=2, train_pairs=3, test_pairs=2)
        mock_ds = _make_mock_hf_dataset(tasks)
        mock_hf_load.return_value = mock_ds

        from juniper_data.generators.arc_agi.generator import ArcAgiGenerator

        params = ArcAgiParams(include_test=False, pad_to=5, seed=42)
        result = ArcAgiGenerator.generate(params)

        n_total = 2 * 3  # only train pairs
        assert result["X_full"].shape[0] == n_total

    def test_generate_with_n_tasks_seed(self, mock_hf_load) -> None:
        """n_tasks with seed selects random subset."""
        tasks = _make_sample_tasks(n_tasks=10, train_pairs=1, test_pairs=0)
        mock_ds = _make_mock_hf_dataset(tasks)
        mock_hf_load.return_value = mock_ds

        from juniper_data.generators.arc_agi.generator import ArcAgiGenerator

        params = ArcAgiParams(n_tasks=3, seed=42, include_test=False, pad_to=5)
        result = ArcAgiGenerator.generate(params)

        assert result["X_full"].shape[0] == 3

    def test_generate_with_n_tasks_no_seed(self, mock_hf_load) -> None:
        """n_tasks without seed takes first N tasks."""
        tasks = _make_sample_tasks(n_tasks=10, train_pairs=1, test_pairs=0)
        mock_ds = _make_mock_hf_dataset(tasks)
        mock_hf_load.return_value = mock_ds

        from juniper_data.generators.arc_agi.generator import ArcAgiGenerator

        params = ArcAgiParams(n_tasks=3, seed=None, include_test=False, pad_to=5)
        result = ArcAgiGenerator.generate(params)

        assert result["X_full"].shape[0] == 3

    def test_generate_hf_fallback_dataset(self, mock_hf_load) -> None:
        """HuggingFace loading tries fallback dataset on failure."""
        tasks = _make_sample_tasks(n_tasks=2, train_pairs=1, test_pairs=0)
        mock_ds = _make_mock_hf_dataset(tasks)

        call_count = [0]

        def side_effect(name, split=None):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Dataset not found")
            return mock_ds

        mock_hf_load.side_effect = side_effect

        from juniper_data.generators.arc_agi.generator import ArcAgiGenerator

        params = ArcAgiParams(include_test=False, pad_to=5, seed=42)
        result = ArcAgiGenerator.generate(params)

        assert mock_hf_load.call_count == 2
        assert result["X_full"].shape[0] == 2

    def test_generate_raises_without_datasets(self) -> None:
        """Raises ImportError when datasets not installed."""
        with patch("juniper_data.generators.arc_agi.generator.HF_AVAILABLE", False):
            from juniper_data.generators.arc_agi.generator import ArcAgiGenerator

            params = ArcAgiParams(source="huggingface")
            with pytest.raises(ImportError, match="Hugging Face datasets package not installed"):
                ArcAgiGenerator.generate(params)

    def test_generate_hf_missing_task_id(self, mock_hf_load) -> None:
        """Handle HF items without task_id."""
        tasks = [{"train": [{"input": [[1]], "output": [[2]]}], "test": []}]
        mock_ds = _make_mock_hf_dataset(tasks)
        mock_hf_load.return_value = mock_ds

        from juniper_data.generators.arc_agi.generator import ArcAgiGenerator

        params = ArcAgiParams(include_test=False, pad_to=5, seed=42)
        result = ArcAgiGenerator.generate(params)

        assert "task_0" in str(result["task_ids"][0])


@pytest.mark.unit
@pytest.mark.generators
class TestArcAgiGeneratorLocal:
    """Tests for ARC-AGI generation from local files."""

    def test_generate_from_local(self, tmp_path) -> None:
        """Generate from local JSON files."""
        from juniper_data.generators.arc_agi.generator import ArcAgiGenerator

        training_dir = tmp_path / "training"
        training_dir.mkdir()

        task = {
            "train": [
                {"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]},
            ],
            "test": [
                {"input": [[5, 6], [7, 8]], "output": [[8, 7], [6, 5]]},
            ],
        }
        (training_dir / "task1.json").write_text(json.dumps(task))
        (training_dir / "task2.json").write_text(json.dumps(task))

        params = ArcAgiParams(source="local", local_path=str(tmp_path), subset="training", pad_to=5, seed=42)
        result = ArcAgiGenerator.generate(params)

        assert result["X_full"].shape[0] == 4  # 2 tasks * (1 train + 1 test)
        assert result["task_ids"].shape[0] == 4

    def test_generate_local_evaluation_subset(self, tmp_path) -> None:
        """Generate from evaluation subset."""
        from juniper_data.generators.arc_agi.generator import ArcAgiGenerator

        eval_dir = tmp_path / "evaluation"
        eval_dir.mkdir()

        task = {"train": [{"input": [[1]], "output": [[2]]}], "test": []}
        (eval_dir / "eval1.json").write_text(json.dumps(task))

        params = ArcAgiParams(source="local", local_path=str(tmp_path), subset="evaluation", include_test=False, pad_to=5, seed=42)
        result = ArcAgiGenerator.generate(params)

        assert result["X_full"].shape[0] == 1

    def test_generate_local_all_subsets(self, tmp_path) -> None:
        """Generate from all subsets."""
        from juniper_data.generators.arc_agi.generator import ArcAgiGenerator

        training_dir = tmp_path / "training"
        training_dir.mkdir()
        eval_dir = tmp_path / "evaluation"
        eval_dir.mkdir()

        task = {"train": [{"input": [[1]], "output": [[2]]}], "test": []}
        (training_dir / "t1.json").write_text(json.dumps(task))
        (eval_dir / "e1.json").write_text(json.dumps(task))

        params = ArcAgiParams(source="local", local_path=str(tmp_path), subset="all", include_test=False, pad_to=5, seed=42)
        result = ArcAgiGenerator.generate(params)

        assert result["X_full"].shape[0] == 2

    def test_generate_local_missing_path(self) -> None:
        """Raises ValueError when local_path is None."""
        from juniper_data.generators.arc_agi.generator import ArcAgiGenerator

        params = ArcAgiParams(source="local", local_path=None)
        with pytest.raises(ValueError, match="local_path is required"):
            ArcAgiGenerator.generate(params)

    def test_generate_local_nonexistent_path(self) -> None:
        """Raises FileNotFoundError when path doesn't exist."""
        from juniper_data.generators.arc_agi.generator import ArcAgiGenerator

        params = ArcAgiParams(source="local", local_path="/nonexistent/path")
        with pytest.raises(FileNotFoundError, match="Path not found"):
            ArcAgiGenerator.generate(params)

    def test_generate_local_with_n_tasks_seed(self, tmp_path) -> None:
        """n_tasks with seed selects random subset from local files."""
        from juniper_data.generators.arc_agi.generator import ArcAgiGenerator

        training_dir = tmp_path / "training"
        training_dir.mkdir()

        task = {"train": [{"input": [[1]], "output": [[2]]}], "test": []}
        for i in range(10):
            (training_dir / f"task_{i}.json").write_text(json.dumps(task))

        params = ArcAgiParams(source="local", local_path=str(tmp_path), n_tasks=3, seed=42, include_test=False, pad_to=5)
        result = ArcAgiGenerator.generate(params)

        assert result["X_full"].shape[0] == 3

    def test_generate_local_with_n_tasks_no_seed(self, tmp_path) -> None:
        """n_tasks without seed takes first N local tasks."""
        from juniper_data.generators.arc_agi.generator import ArcAgiGenerator

        training_dir = tmp_path / "training"
        training_dir.mkdir()

        task = {"train": [{"input": [[1]], "output": [[2]]}], "test": []}
        for i in range(10):
            (training_dir / f"task_{i:02d}.json").write_text(json.dumps(task))

        params = ArcAgiParams(source="local", local_path=str(tmp_path), n_tasks=3, seed=None, include_test=False, pad_to=5)
        result = ArcAgiGenerator.generate(params)

        assert result["X_full"].shape[0] == 3

    def test_generate_local_missing_subdirs(self, tmp_path) -> None:
        """Handle missing training/evaluation subdirectories gracefully."""
        from juniper_data.generators.arc_agi.generator import ArcAgiGenerator

        params = ArcAgiParams(source="local", local_path=str(tmp_path), subset="training", include_test=False, pad_to=5, seed=42)
        result = ArcAgiGenerator.generate(params)

        assert result["X_full"].shape[0] == 0


@pytest.mark.unit
@pytest.mark.generators
class TestArcAgiGeneratorPadGrid:
    """Tests for _pad_grid helper."""

    def test_pad_grid_smaller_than_target(self) -> None:
        """Pad a small grid to target size."""
        from juniper_data.generators.arc_agi.generator import ArcAgiGenerator

        grid = [[1, 2], [3, 4]]
        result = ArcAgiGenerator._pad_grid(grid, pad_to=5, pad_value=-1)

        assert result.shape == (5, 5)
        assert result[0, 0] == 1
        assert result[0, 2] == -1
        assert result[2, 0] == -1

    def test_pad_grid_exact_size(self) -> None:
        """Grid exactly matching target size is unchanged."""
        from juniper_data.generators.arc_agi.generator import ArcAgiGenerator

        grid = [[1, 2], [3, 4]]
        result = ArcAgiGenerator._pad_grid(grid, pad_to=2, pad_value=0)

        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result, [[1, 2], [3, 4]])


@pytest.mark.unit
@pytest.mark.generators
class TestArcAgiGeneratorConvertTasks:
    """Tests for _convert_tasks_to_arrays helper."""

    def test_convert_empty_tasks(self) -> None:
        """Empty task list returns zero-sized arrays."""
        from juniper_data.generators.arc_agi.generator import ArcAgiGenerator

        params = ArcAgiParams(pad_to=5, include_test=False)
        X, y, ids = ArcAgiGenerator._convert_tasks_to_arrays([], params)

        assert X.shape == (0, 25)
        assert y.shape == (0, 25)
        assert len(ids) == 0

    def test_convert_tasks_with_test_missing_output(self) -> None:
        """Test pairs with missing output get padded."""
        from juniper_data.generators.arc_agi.generator import ArcAgiGenerator

        tasks = [
            {
                "task_id": "test-task",
                "train": [{"input": [[1]], "output": [[2]]}],
                "test": [{"input": [[3]]}],
            }
        ]

        params = ArcAgiParams(pad_to=5, include_test=True, flatten_pairs=True)
        X, y, ids = ArcAgiGenerator._convert_tasks_to_arrays(tasks, params)

        assert X.shape[0] == 2  # 1 train + 1 test
        assert y.shape[0] == 2

    def test_convert_tasks_unknown_task_id(self) -> None:
        """Tasks without task_id get 'unknown'."""
        from juniper_data.generators.arc_agi.generator import ArcAgiGenerator

        tasks = [{"train": [{"input": [[1]], "output": [[2]]}], "test": []}]

        params = ArcAgiParams(pad_to=5, include_test=False, flatten_pairs=True)
        X, y, ids = ArcAgiGenerator._convert_tasks_to_arrays(tasks, params)

        assert ids[0] == "unknown"


@pytest.mark.unit
@pytest.mark.generators
class TestArcAgiGetSchema:
    """Tests for get_schema function."""

    def test_get_schema_returns_dict(self) -> None:
        """get_schema returns a dictionary."""
        from juniper_data.generators.arc_agi.generator import get_schema

        schema = get_schema()
        assert isinstance(schema, dict)

    def test_get_schema_has_properties(self) -> None:
        """Schema has expected properties."""
        from juniper_data.generators.arc_agi.generator import get_schema

        schema = get_schema()
        assert "properties" in schema
        assert "source" in schema["properties"]
        assert "pad_to" in schema["properties"]
        assert "n_tasks" in schema["properties"]


@pytest.mark.unit
@pytest.mark.generators
class TestArcAgiVersion:
    """Tests for version constant."""

    def test_version_format(self) -> None:
        """Version follows semver format."""
        from juniper_data.generators.arc_agi.generator import VERSION

        parts = VERSION.split(".")
        assert len(parts) == 3
        for part in parts:
            assert part.isdigit()


@pytest.mark.unit
@pytest.mark.generators
class TestArcAgiImports:
    """Tests for __init__.py imports."""

    def test_init_exports(self) -> None:
        """__init__.py exports expected symbols."""
        from juniper_data.generators.arc_agi import ArcAgiGenerator, ArcAgiParams, VERSION, get_schema

        assert ArcAgiGenerator is not None
        assert ArcAgiParams is not None
        assert VERSION is not None
        assert get_schema is not None
