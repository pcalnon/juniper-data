"""Unit tests for the MNIST dataset generator."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from juniper_data.generators.mnist.params import MnistParams


def _make_mock_hf_dataset(n_samples=20, n_classes=10):
    """Create a mock HuggingFace dataset for MNIST-like data."""
    mock_ds = MagicMock()
    mock_ds.__len__ = MagicMock(return_value=n_samples)

    labels = [i % n_classes for i in range(n_samples)]

    images = []
    for _ in range(n_samples):
        mock_img = MagicMock()
        mock_img.convert.return_value = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
        images.append(mock_img)

    def ds_iter():
        for i in range(n_samples):
            yield {"image": images[i], "label": labels[i]}

    mock_ds.__iter__ = MagicMock(side_effect=ds_iter)
    mock_ds.__getitem__ = MagicMock(side_effect=lambda key: labels if key == "label" else images)
    mock_ds.shuffle.return_value = mock_ds
    mock_ds.select.return_value = mock_ds

    return mock_ds, labels, images


@pytest.fixture
def mock_hf_load():
    """Patch HF_AVAILABLE and hf_load_dataset for the mnist generator module."""
    mock_load = MagicMock()

    with patch("juniper_data.generators.mnist.generator.HF_AVAILABLE", True):
        with patch("juniper_data.generators.mnist.generator.hf_load_dataset", mock_load):
            yield mock_load


@pytest.mark.unit
@pytest.mark.generators
class TestMnistParams:
    """Tests for MnistParams validation."""

    def test_default_params(self) -> None:
        """Default parameters are valid."""
        params = MnistParams()
        assert params.dataset == "mnist"
        assert params.n_samples is None
        assert params.flatten is True
        assert params.normalize is True
        assert params.one_hot_labels is True
        assert params.train_ratio == 0.8
        assert params.test_ratio == 0.2

    def test_fashion_mnist(self) -> None:
        """Fashion-MNIST variant is accepted."""
        params = MnistParams(dataset="fashion_mnist")
        assert params.dataset == "fashion_mnist"

    def test_custom_params(self) -> None:
        """Custom parameters are accepted."""
        params = MnistParams(
            n_samples=100,
            flatten=False,
            normalize=False,
            one_hot_labels=False,
            seed=42,
            train_ratio=0.7,
            test_ratio=0.3,
        )
        assert params.n_samples == 100
        assert params.flatten is False
        assert params.seed == 42

    def test_invalid_n_samples(self) -> None:
        """n_samples must be >= 1."""
        with pytest.raises(ValueError):
            MnistParams(n_samples=0)

    def test_invalid_train_ratio(self) -> None:
        """train_ratio must be in (0, 1]."""
        with pytest.raises(ValueError):
            MnistParams(train_ratio=0)

    def test_invalid_dataset_name(self) -> None:
        """Invalid dataset name is rejected."""
        with pytest.raises(ValueError):
            MnistParams(dataset="cifar10")  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.generators
class TestMnistGenerator:
    """Tests for MnistGenerator functionality."""

    def test_generate_correct_shapes(self, mock_hf_load) -> None:
        """Generated arrays have correct shapes."""
        mock_ds, _, _ = _make_mock_hf_dataset(n_samples=20)
        mock_hf_load.return_value = mock_ds

        from juniper_data.generators.mnist.generator import MnistGenerator

        params = MnistParams(seed=42)
        result = MnistGenerator.generate(params)

        n_total = 20
        n_train = int(n_total * 0.8)
        n_test = n_total - n_train

        assert result["X_train"].shape[0] == n_train
        assert result["X_test"].shape[0] == n_test
        assert result["X_full"].shape[0] == n_total
        assert result["y_full"].shape[0] == n_total

    def test_generate_flattened(self, mock_hf_load) -> None:
        """Flatten option produces 1D features (784 for 28x28)."""
        mock_ds, _, _ = _make_mock_hf_dataset(n_samples=10)
        mock_hf_load.return_value = mock_ds

        from juniper_data.generators.mnist.generator import MnistGenerator

        params = MnistParams(flatten=True, seed=42)
        result = MnistGenerator.generate(params)

        assert result["X_full"].shape[1] == 784

    def test_generate_not_flattened(self, mock_hf_load) -> None:
        """Non-flatten produces 2D features (28x28)."""
        mock_ds, _, _ = _make_mock_hf_dataset(n_samples=10)
        mock_hf_load.return_value = mock_ds

        from juniper_data.generators.mnist.generator import MnistGenerator

        params = MnistParams(flatten=False, seed=42)
        result = MnistGenerator.generate(params)

        assert result["X_full"].shape[1:] == (28, 28)

    def test_generate_normalized(self, mock_hf_load) -> None:
        """Normalized values are in [0, 1]."""
        mock_ds, _, _ = _make_mock_hf_dataset(n_samples=10)
        mock_hf_load.return_value = mock_ds

        from juniper_data.generators.mnist.generator import MnistGenerator

        params = MnistParams(normalize=True, seed=42)
        result = MnistGenerator.generate(params)

        assert result["X_full"].max() <= 1.0
        assert result["X_full"].min() >= 0.0

    def test_generate_not_normalized(self, mock_hf_load) -> None:
        """Non-normalized values can exceed 1.0."""
        mock_ds, _, _ = _make_mock_hf_dataset(n_samples=10)
        mock_hf_load.return_value = mock_ds

        from juniper_data.generators.mnist.generator import MnistGenerator

        params = MnistParams(normalize=False, seed=42)
        result = MnistGenerator.generate(params)

        assert result["X_full"].dtype == np.float32

    def test_generate_one_hot_labels(self, mock_hf_load) -> None:
        """One-hot encoding produces correct label shape."""
        mock_ds, _, _ = _make_mock_hf_dataset(n_samples=20, n_classes=10)
        mock_hf_load.return_value = mock_ds

        from juniper_data.generators.mnist.generator import MnistGenerator

        params = MnistParams(one_hot_labels=True, seed=42)
        result = MnistGenerator.generate(params)

        assert result["y_full"].shape[1] == 10
        row_sums = result["y_full"].sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(20))

    def test_generate_integer_labels(self, mock_hf_load) -> None:
        """Non-one-hot produces integer labels."""
        mock_ds, _, _ = _make_mock_hf_dataset(n_samples=10, n_classes=10)
        mock_hf_load.return_value = mock_ds

        from juniper_data.generators.mnist.generator import MnistGenerator

        params = MnistParams(one_hot_labels=False, seed=42)
        result = MnistGenerator.generate(params)

        assert result["y_full"].shape[1] == 1

    def test_generate_with_seed_shuffle(self, mock_hf_load) -> None:
        """Seed triggers dataset shuffle."""
        mock_ds, _, _ = _make_mock_hf_dataset(n_samples=10)
        mock_hf_load.return_value = mock_ds

        from juniper_data.generators.mnist.generator import MnistGenerator

        params = MnistParams(seed=42)
        MnistGenerator.generate(params)

        mock_ds.shuffle.assert_called_once_with(seed=42)

    def test_generate_with_n_samples(self, mock_hf_load) -> None:
        """n_samples limits the dataset."""
        mock_ds, _, _ = _make_mock_hf_dataset(n_samples=100)
        mock_hf_load.return_value = mock_ds

        from juniper_data.generators.mnist.generator import MnistGenerator

        params = MnistParams(n_samples=50, seed=42)
        MnistGenerator.generate(params)

        mock_ds.select.assert_called_once()

    def test_generate_no_seed_no_shuffle(self, mock_hf_load) -> None:
        """Without seed, no shuffle is called."""
        mock_ds, _, _ = _make_mock_hf_dataset(n_samples=10)
        mock_hf_load.return_value = mock_ds

        from juniper_data.generators.mnist.generator import MnistGenerator

        params = MnistParams(seed=None, n_samples=None)
        MnistGenerator.generate(params)

        mock_ds.shuffle.assert_not_called()
        mock_ds.select.assert_not_called()

    def test_generate_image_without_convert(self, mock_hf_load) -> None:
        """Handle images that don't have a convert method."""
        mock_ds = MagicMock()
        mock_ds.__len__ = MagicMock(return_value=5)

        raw_images = [np.random.randint(0, 255, (28, 28), dtype=np.uint8) for _ in range(5)]
        labels = [0, 1, 2, 3, 4]

        def ds_iter():
            for i in range(5):
                yield {"image": raw_images[i], "label": labels[i]}

        mock_ds.__iter__ = MagicMock(side_effect=ds_iter)
        mock_ds.__getitem__ = MagicMock(side_effect=lambda key: labels if key == "label" else raw_images)
        mock_ds.shuffle.return_value = mock_ds
        mock_ds.select.return_value = mock_ds
        mock_hf_load.return_value = mock_ds

        from juniper_data.generators.mnist.generator import MnistGenerator

        params = MnistParams(seed=42)
        result = MnistGenerator.generate(params)
        assert result["X_full"].shape[0] == 5

    def test_generate_raises_without_datasets(self) -> None:
        """Raises ImportError when datasets not installed."""
        with patch("juniper_data.generators.mnist.generator.HF_AVAILABLE", False):
            from juniper_data.generators.mnist.generator import MnistGenerator

            params = MnistParams()
            with pytest.raises(ImportError, match="Hugging Face datasets package not installed"):
                MnistGenerator.generate(params)

    def test_generate_correct_dtypes(self, mock_hf_load) -> None:
        """All arrays are float32."""
        mock_ds, _, _ = _make_mock_hf_dataset(n_samples=10)
        mock_hf_load.return_value = mock_ds

        from juniper_data.generators.mnist.generator import MnistGenerator

        params = MnistParams(seed=42)
        result = MnistGenerator.generate(params)

        for key in ["X_train", "y_train", "X_test", "y_test", "X_full", "y_full"]:
            assert result[key].dtype == np.float32


@pytest.mark.unit
@pytest.mark.generators
class TestMnistGetSchema:
    """Tests for get_schema function."""

    def test_get_schema_returns_dict(self) -> None:
        """get_schema returns a dictionary."""
        from juniper_data.generators.mnist.generator import get_schema

        schema = get_schema()
        assert isinstance(schema, dict)

    def test_get_schema_has_properties(self) -> None:
        """Schema has expected properties."""
        from juniper_data.generators.mnist.generator import get_schema

        schema = get_schema()
        assert "properties" in schema
        assert "dataset" in schema["properties"]
        assert "n_samples" in schema["properties"]
        assert "flatten" in schema["properties"]


@pytest.mark.unit
@pytest.mark.generators
class TestMnistVersion:
    """Tests for version constant."""

    def test_version_format(self) -> None:
        """Version follows semver format."""
        from juniper_data.generators.mnist.generator import VERSION

        parts = VERSION.split(".")
        assert len(parts) == 3
        for part in parts:
            assert part.isdigit()


@pytest.mark.unit
@pytest.mark.generators
class TestMnistImports:
    """Tests for __init__.py imports."""

    def test_init_exports(self) -> None:
        """__init__.py exports expected symbols."""
        from juniper_data.generators.mnist import VERSION, MnistGenerator, MnistParams, get_schema

        assert MnistGenerator is not None
        assert MnistParams is not None
        assert VERSION is not None
        assert get_schema is not None

    def test_module_level_hf_available_true(self) -> None:
        """Module-level HF_AVAILABLE is True when datasets is importable."""
        import importlib
        import sys
        from types import ModuleType
        from unittest.mock import MagicMock

        # Inject a fake 'datasets' module so the try-branch succeeds
        fake_datasets = ModuleType("datasets")
        fake_datasets.load_dataset = MagicMock()  # type: ignore[attr-defined]
        sys.modules["datasets"] = fake_datasets

        mod_name = "juniper_data.generators.mnist.generator"
        sys.modules.pop(mod_name, None)
        try:
            mod = importlib.import_module(mod_name)
            assert mod.HF_AVAILABLE is True
        finally:
            # Restore original state
            sys.modules.pop("datasets", None)
            sys.modules.pop(mod_name, None)
            importlib.import_module(mod_name)
