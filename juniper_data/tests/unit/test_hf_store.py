"""Unit tests for HuggingFaceDatasetStore."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from juniper_data.core.models import DatasetMeta
from juniper_data.storage.memory import InMemoryDatasetStore
from juniper_data.tests.partitions import whole


@pytest.fixture
def sample_meta() -> DatasetMeta:
    """Create sample metadata."""
    return DatasetMeta(
        dataset_id="test-dataset",
        generator="test",
        generator_version="1.0.0",
        params={"seed": 42},
        n_samples=100,
        n_features=2,
        n_classes=2,
        n_train=80,
        n_test=20,
        class_distribution={"0": 50, "1": 50},
        created_at=datetime.now(UTC),
    )


@pytest.fixture
def sample_arrays() -> dict[str, np.ndarray]:
    """Create sample arrays."""
    rng = np.random.default_rng(42)
    return {
        "X_train": rng.standard_normal((80, 2)).astype(np.float32),
        "y_train": rng.standard_normal((80, 2)).astype(np.float32),
        "X_test": rng.standard_normal((20, 2)).astype(np.float32),
        "y_test": rng.standard_normal((20, 2)).astype(np.float32),
    }


def _make_mock_hf_dataset(n_samples=20, n_classes=3, feature_type="tabular"):
    """Create a mock HuggingFace dataset object."""
    mock_ds = MagicMock()
    mock_ds.column_names = ["feature1", "feature2", "label"] if feature_type == "tabular" else ["image", "label"]
    mock_ds.__len__ = MagicMock(return_value=n_samples)

    labels = list(range(n_classes)) * (n_samples // n_classes) + list(range(n_samples % n_classes))
    labels = labels[:n_samples]
    mock_ds.__getitem__ = MagicMock()

    if feature_type == "tabular":

        def getitem(key):
            if key == "feature1":
                return list(range(n_samples))
            elif key == "feature2":
                return list(range(n_samples, 2 * n_samples))
            elif key == "label":
                return labels
            return []

        mock_ds.__getitem__.side_effect = getitem
    else:
        mock_images = []
        for _ in range(n_samples):
            mock_img = MagicMock()
            mock_img.convert.return_value = MagicMock()
            np_arr = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
            mock_img.convert.return_value = np_arr
            mock_images.append({"image": mock_img, "label": labels[_ % len(labels)]})
        mock_ds.__iter__ = MagicMock(return_value=iter(mock_images))
        mock_ds.__getitem__.side_effect = lambda key: labels if key == "label" else [m["image"] for m in mock_images] if key == "image" else []

    mock_ds.shuffle.return_value = mock_ds
    mock_ds.select.return_value = mock_ds

    return mock_ds, labels


@pytest.fixture
def mock_hf_module():
    """Create a mock HF datasets module and patch it into hf_store."""
    mock_load = MagicMock()

    with patch("juniper_data.storage.hf_store.HF_AVAILABLE", True):
        with patch("juniper_data.storage.hf_store.hf_load_dataset", mock_load):
            yield mock_load


@pytest.mark.unit
@pytest.mark.storage
class TestHuggingFaceDatasetStoreInit:
    """Tests for HuggingFaceDatasetStore initialization."""

    def test_init_default(self, mock_hf_module) -> None:
        """Initialize with default parameters."""
        from juniper_data.storage.hf_store import HuggingFaceDatasetStore

        store = HuggingFaceDatasetStore()
        assert isinstance(store._cache_store, InMemoryDatasetStore)
        assert store._cache_dir is None

    def test_init_custom_cache_store(self, mock_hf_module) -> None:
        """Initialize with custom cache store."""
        from juniper_data.storage.hf_store import HuggingFaceDatasetStore

        custom_cache = InMemoryDatasetStore()
        store = HuggingFaceDatasetStore(cache_store=custom_cache)
        assert store._cache_store is custom_cache

    def test_init_with_cache_dir(self, mock_hf_module) -> None:
        """Initialize with cache directory."""
        from juniper_data.storage.hf_store import HuggingFaceDatasetStore

        store = HuggingFaceDatasetStore(cache_dir="/tmp/hf_cache")
        assert store._cache_dir == "/tmp/hf_cache"

    def test_init_raises_without_datasets(self) -> None:
        """Raises ImportError when datasets package is not available."""
        with patch("juniper_data.storage.hf_store.HF_AVAILABLE", False):
            from juniper_data.storage.hf_store import HuggingFaceDatasetStore

            with pytest.raises(ImportError, match="Hugging Face datasets package not installed"):
                HuggingFaceDatasetStore()


@pytest.mark.unit
@pytest.mark.storage
class TestHuggingFaceDatasetStoreLoadDataset:
    """Tests for load_hf_dataset operation."""

    def test_load_tabular_dataset(self, mock_hf_module) -> None:
        """Load a tabular dataset from HuggingFace."""
        from juniper_data.storage.hf_store import HuggingFaceDatasetStore

        mock_ds, labels = _make_mock_hf_dataset(n_samples=20, n_classes=3, feature_type="tabular")
        mock_hf_module.return_value = mock_ds

        store = HuggingFaceDatasetStore()
        dataset_id, meta, arrays = store.load_hf_dataset("test-dataset", feature_columns=["feature1", "feature2"], label_column="label")

        assert "hf-test-dataset" in dataset_id
        assert meta.generator == "huggingface"
        # juniper-data#411: this line asserted `"X_full" in arrays` -- the fixture encoded
        # the defect. The store now emits exactly the decision-11 contract.
        assert set(arrays) == {"X_train", "y_train", "X_val", "y_val", "X_test", "y_test"}
        assert whole(arrays, "X").dtype == np.float32

    def test_load_emits_three_partitions_at_the_decision_11_version(self, mock_hf_module) -> None:
        """0.8 / 0.1 / 0.1 carve over 20 rows, stamped 3.0.0, meta counts match arrays."""
        from juniper_data.storage.hf_store import HuggingFaceDatasetStore

        mock_ds, _ = _make_mock_hf_dataset(n_samples=20, n_classes=2, feature_type="tabular")
        mock_hf_module.return_value = mock_ds

        store = HuggingFaceDatasetStore()
        dataset_id, meta, arrays = store.load_hf_dataset("test-dataset", seed=7, normalize=False, feature_columns=["feature1", "feature2"])

        assert meta.generator_version == "3.0.0"
        assert "-huggingface-3.0.0-" in dataset_id
        assert (meta.n_train, meta.n_val, meta.n_test) == (16, 2, 2)
        assert [arrays[f"X_{p}"].shape[0] for p in ("train", "val", "test")] == [16, 2, 2]
        assert meta.n_samples == 20
        assert sum(meta.class_distribution.values()) == 20
        # Contiguous and disjoint: the three blocks reassemble the loaded rows in order.
        np.testing.assert_array_equal(whole(arrays, "X")[:, 0], np.arange(20, dtype=np.float32))

    def test_ratios_are_honoured_and_unused_rows_are_left_out(self, mock_hf_module) -> None:
        """A ratio sum below 1 leaves the tail out of every partition and out of the meta."""
        from juniper_data.storage.hf_store import HuggingFaceDatasetStore

        mock_ds, _ = _make_mock_hf_dataset(n_samples=20, n_classes=2, feature_type="tabular")
        mock_hf_module.return_value = mock_ds

        store = HuggingFaceDatasetStore()
        _, meta, arrays = store.load_hf_dataset("test-dataset", seed=7, feature_columns=["feature1", "feature2"], train_ratio=0.5, val_ratio=0.2, test_ratio=0.1)

        assert (meta.n_train, meta.n_val, meta.n_test) == (10, 4, 2)
        assert meta.n_samples == 16
        assert sum(meta.class_distribution.values()) == 16
        assert whole(arrays, "X").shape[0] == 16

    @pytest.mark.parametrize(
        ("ratios", "message"),
        [
            ({"train_ratio": 0.9}, "must be <= 1.0"),
            ({"train_ratio": 0.0}, "train_ratio must be greater than 0"),
            ({"val_ratio": -0.1}, "val_ratio must be between 0 and 1"),
        ],
        ids=["oversubscribed", "no-train", "negative-val"],
    )
    def test_invalid_ratios_raise(self, mock_hf_module, ratios, message) -> None:
        """Invalid ratios fail loudly instead of silently trimming a partition away."""
        from juniper_data.storage.hf_store import HuggingFaceDatasetStore

        mock_ds, _ = _make_mock_hf_dataset(n_samples=20, n_classes=2, feature_type="tabular")
        mock_hf_module.return_value = mock_ds

        store = HuggingFaceDatasetStore()
        with pytest.raises(ValueError, match=message):
            store.load_hf_dataset("test-dataset", seed=7, feature_columns=["feature1", "feature2"], **ratios)

    def test_dataset_id_carries_version_and_params(self, mock_hf_module) -> None:
        """Same request -> same ID; a different partitioning -> a different ID.

        The old ID was `hf-<name>-<rows>`, so two partitionings of one dataset collided
        and a cached two-way artifact answered a three-way request under the same ID.
        """
        from juniper_data.storage.hf_store import HuggingFaceDatasetStore

        mock_ds, _ = _make_mock_hf_dataset(n_samples=20, n_classes=2, feature_type="tabular")
        mock_hf_module.return_value = mock_ds

        store = HuggingFaceDatasetStore()
        first, _, _ = store.load_hf_dataset("test-dataset", seed=7, feature_columns=["feature1", "feature2"])
        again, _, _ = store.load_hf_dataset("test-dataset", seed=7, feature_columns=["feature1", "feature2"])
        other, _, _ = store.load_hf_dataset("test-dataset", seed=7, feature_columns=["feature1", "feature2"], train_ratio=0.7, val_ratio=0.2)

        assert first == again
        assert first != other

    def test_unseeded_loads_reuse_one_id_and_one_cache_entry(self, mock_hf_module) -> None:
        """No seed means no shuffle here, so the load is repeatable and must not mint a new ID.

        generate_dataset_id's BUG-JD-04 nonce assumes 'no seed = fresh random draw', true for
        a generator and false for these stores. Without the marker every identical call
        added a full copy to the never-evicting default cache store.
        """
        from juniper_data.storage.hf_store import HuggingFaceDatasetStore

        mock_ds, _ = _make_mock_hf_dataset(n_samples=20, n_classes=2, feature_type="tabular")
        mock_hf_module.return_value = mock_ds

        store = HuggingFaceDatasetStore()
        ids = {store.load_hf_dataset("test-dataset", feature_columns=["feature1", "feature2"])[0] for _ in range(3)}

        assert len(ids) == 1
        assert store._cache_store.list_datasets() == list(ids)
        _, meta, _ = store.load_hf_dataset("test-dataset", feature_columns=["feature1", "feature2"])
        assert meta.params["seed"] is None, "the recorded params keep the real None; only the hash uses the marker"

    def test_numpy_seed_is_accepted_and_hashes_like_an_int(self, mock_hf_module) -> None:
        """rng.integers() returns np.int64, which json.dumps cannot serialise."""
        from juniper_data.storage.hf_store import HuggingFaceDatasetStore

        mock_ds, _ = _make_mock_hf_dataset(n_samples=20, n_classes=2, feature_type="tabular")
        mock_hf_module.return_value = mock_ds

        store = HuggingFaceDatasetStore()
        as_numpy, _, _ = store.load_hf_dataset("test-dataset", seed=np.int64(7), feature_columns=["feature1", "feature2"])
        as_int, _, _ = store.load_hf_dataset("test-dataset", seed=7, feature_columns=["feature1", "feature2"])

        assert as_numpy == as_int

    def test_invalid_ratios_fail_before_the_download(self, mock_hf_module) -> None:
        """A bad request must not cost a Hub fetch."""
        from juniper_data.storage.hf_store import HuggingFaceDatasetStore

        store = HuggingFaceDatasetStore()
        with pytest.raises(ValueError, match="must be <= 1.0"):
            store.load_hf_dataset("test-dataset", train_ratio=0.9)

        mock_hf_module.assert_not_called()

    def test_load_with_config_name(self, mock_hf_module) -> None:
        """Load with config name included in dataset_id."""
        from juniper_data.storage.hf_store import HuggingFaceDatasetStore

        mock_ds, _ = _make_mock_hf_dataset(n_samples=10, n_classes=2, feature_type="tabular")
        mock_hf_module.return_value = mock_ds

        store = HuggingFaceDatasetStore()
        dataset_id, meta, arrays = store.load_hf_dataset("test-dataset", config_name="v2", feature_columns=["feature1", "feature2"])

        assert "-v2-" in dataset_id

    def test_load_with_seed(self, mock_hf_module) -> None:
        """Load with seed triggers shuffle."""
        from juniper_data.storage.hf_store import HuggingFaceDatasetStore

        mock_ds, _ = _make_mock_hf_dataset(n_samples=10, n_classes=2, feature_type="tabular")
        mock_hf_module.return_value = mock_ds

        store = HuggingFaceDatasetStore()
        store.load_hf_dataset("test-dataset", seed=42, feature_columns=["feature1", "feature2"])

        mock_ds.shuffle.assert_called_once_with(seed=42)

    def test_load_with_n_samples(self, mock_hf_module) -> None:
        """Load with n_samples limits data."""
        from juniper_data.storage.hf_store import HuggingFaceDatasetStore

        mock_ds, _ = _make_mock_hf_dataset(n_samples=20, n_classes=2, feature_type="tabular")
        mock_hf_module.return_value = mock_ds

        store = HuggingFaceDatasetStore()
        store.load_hf_dataset("test-dataset", n_samples=5, feature_columns=["feature1", "feature2"])

        mock_ds.select.assert_called_once()

    def test_load_without_one_hot(self, mock_hf_module) -> None:
        """Load without one-hot encoding produces integer labels."""
        from juniper_data.storage.hf_store import HuggingFaceDatasetStore

        mock_ds, _ = _make_mock_hf_dataset(n_samples=10, n_classes=2, feature_type="tabular")
        mock_hf_module.return_value = mock_ds

        store = HuggingFaceDatasetStore()
        _, meta, arrays = store.load_hf_dataset("test-dataset", one_hot_labels=False, feature_columns=["feature1", "feature2"])

        assert whole(arrays, "y").shape[1] == 1

    def test_load_with_normalization(self, mock_hf_module) -> None:
        """Load with normalization scales features."""
        from juniper_data.storage.hf_store import HuggingFaceDatasetStore

        mock_ds, _ = _make_mock_hf_dataset(n_samples=10, n_classes=2, feature_type="tabular")
        mock_hf_module.return_value = mock_ds

        store = HuggingFaceDatasetStore()
        _, _, arrays = store.load_hf_dataset("test-dataset", normalize=True, feature_columns=["feature1", "feature2"])

        assert whole(arrays, "X").max() <= 1.0

    def test_load_saves_to_cache(self, mock_hf_module) -> None:
        """Load saves the result to cache store."""
        from juniper_data.storage.hf_store import HuggingFaceDatasetStore

        mock_ds, _ = _make_mock_hf_dataset(n_samples=10, n_classes=2, feature_type="tabular")
        mock_hf_module.return_value = mock_ds

        store = HuggingFaceDatasetStore()
        dataset_id, _, _ = store.load_hf_dataset("test-dataset", feature_columns=["feature1", "feature2"])

        assert store._cache_store.exists(dataset_id)


@pytest.mark.unit
@pytest.mark.storage
class TestHuggingFaceDatasetStoreExtractImages:
    """Tests for _extract_images operation."""

    def test_extract_images_pil_like(self, mock_hf_module) -> None:
        """Extract images from PIL-like objects."""
        from juniper_data.storage.hf_store import HuggingFaceDatasetStore

        store = HuggingFaceDatasetStore()

        mock_img = MagicMock()
        mock_img.convert.return_value = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
        mock_ds = [{"image": mock_img}, {"image": mock_img}]

        result = store._extract_images(mock_ds, "image", flatten=True, normalize=True)
        assert result.shape == (2, 784)
        assert result.max() <= 1.0

    def test_extract_images_numpy_like(self, mock_hf_module) -> None:
        """Extract images from numpy-like objects."""
        from juniper_data.storage.hf_store import HuggingFaceDatasetStore

        store = HuggingFaceDatasetStore()

        mock_tensor = MagicMock(spec=["numpy"])
        mock_tensor.numpy.return_value = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
        mock_ds = [{"image": mock_tensor}, {"image": mock_tensor}]

        result = store._extract_images(mock_ds, "image", flatten=False, normalize=False)
        assert result.shape == (2, 28, 28)
        assert result.dtype == np.float32

    def test_extract_images_raw_array(self, mock_hf_module) -> None:
        """Extract images from raw array-like objects."""
        from juniper_data.storage.hf_store import HuggingFaceDatasetStore

        store = HuggingFaceDatasetStore()

        raw_data = [[1, 2], [3, 4]]
        mock_ds = [{"image": raw_data}, {"image": raw_data}]

        result = store._extract_images(mock_ds, "image", flatten=True, normalize=True)
        assert result.shape == (2, 4)


@pytest.mark.unit
@pytest.mark.storage
class TestHuggingFaceDatasetStoreExtractFeaturesLabels:
    """Tests for _extract_features_labels operation."""

    def test_auto_detect_feature_columns(self, mock_hf_module) -> None:
        """Auto-detect feature columns excluding label and id columns."""
        from juniper_data.storage.hf_store import HuggingFaceDatasetStore

        store = HuggingFaceDatasetStore()

        mock_ds = MagicMock()
        mock_ds.column_names = ["feature1", "feature2", "label", "idx"]

        mock_ds.__getitem__.side_effect = lambda key: [1.0, 2.0] if key in ("feature1", "feature2") else [0, 1]

        X, y, n_classes = store._extract_features_labels(mock_ds, feature_columns=None, label_column="label", flatten=True, normalize=False, one_hot_labels=True)
        assert X.dtype == np.float32

    def test_single_image_feature(self, mock_hf_module) -> None:
        """Handle single image column."""
        from juniper_data.storage.hf_store import HuggingFaceDatasetStore

        store = HuggingFaceDatasetStore()

        mock_img = MagicMock()
        mock_img.convert.return_value = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
        mock_ds = MagicMock()
        mock_ds.column_names = ["image", "label"]
        mock_ds.__iter__ = MagicMock(return_value=iter([{"image": mock_img}, {"image": mock_img}]))
        mock_ds.__getitem__.side_effect = lambda key: [0, 1] if key == "label" else [mock_img, mock_img]

        X, y, n_classes = store._extract_features_labels(mock_ds, feature_columns=["image"], label_column="label", flatten=True, normalize=True, one_hot_labels=True)
        assert X.shape[0] == 2
        assert n_classes == 2

    def test_tensor_feature_columns(self, mock_hf_module) -> None:
        """Handle feature columns with tensor-like values."""
        from juniper_data.storage.hf_store import HuggingFaceDatasetStore

        store = HuggingFaceDatasetStore()

        mock_tensor = MagicMock()
        mock_tensor.numpy.return_value = np.array(1.0)

        mock_ds = MagicMock()
        mock_ds.column_names = ["feat", "label"]

        mock_ds.__getitem__.side_effect = lambda key: [mock_tensor, mock_tensor] if key == "feat" else [0, 1]  # type: ignore[list-item]

        X, y, n_classes = store._extract_features_labels(mock_ds, feature_columns=["feat"], label_column="label", flatten=True, normalize=False, one_hot_labels=False)
        assert y.shape[1] == 1


@pytest.mark.unit
@pytest.mark.storage
class TestHuggingFaceDatasetStoreDelegation:
    """Tests for delegated cache store operations."""

    def test_save_delegates(self, mock_hf_module, sample_meta, sample_arrays) -> None:
        """save delegates to cache store."""
        from juniper_data.storage.hf_store import HuggingFaceDatasetStore

        store = HuggingFaceDatasetStore()
        store.save("test-1", sample_meta, sample_arrays)
        assert store._cache_store.exists("test-1")

    def test_get_meta_delegates(self, mock_hf_module, sample_meta, sample_arrays) -> None:
        """get_meta delegates to cache store."""
        from juniper_data.storage.hf_store import HuggingFaceDatasetStore

        store = HuggingFaceDatasetStore()
        store._cache_store.save("test-1", sample_meta, sample_arrays)

        result = store.get_meta("test-1")
        assert result is not None
        assert result.dataset_id == sample_meta.dataset_id

    def test_get_artifact_bytes_delegates(self, mock_hf_module, sample_meta, sample_arrays) -> None:
        """get_artifact_bytes delegates to cache store."""
        from juniper_data.storage.hf_store import HuggingFaceDatasetStore

        store = HuggingFaceDatasetStore()
        store._cache_store.save("test-1", sample_meta, sample_arrays)

        result = store.get_artifact_bytes("test-1")
        assert result is not None

    def test_exists_delegates(self, mock_hf_module) -> None:
        """exists delegates to cache store."""
        from juniper_data.storage.hf_store import HuggingFaceDatasetStore

        store = HuggingFaceDatasetStore()
        assert store.exists("nonexistent") is False

    def test_delete_delegates(self, mock_hf_module, sample_meta, sample_arrays) -> None:
        """delete delegates to cache store."""
        from juniper_data.storage.hf_store import HuggingFaceDatasetStore

        store = HuggingFaceDatasetStore()
        store._cache_store.save("test-1", sample_meta, sample_arrays)
        assert store.delete("test-1") is True
        assert not store.exists("test-1")

    def test_list_datasets_delegates(self, mock_hf_module, sample_meta, sample_arrays) -> None:
        """list_datasets delegates to cache store."""
        from juniper_data.storage.hf_store import HuggingFaceDatasetStore

        store = HuggingFaceDatasetStore()
        store._cache_store.save("test-1", sample_meta, sample_arrays)

        result = store.list_datasets()
        assert "test-1" in result

    def test_update_meta_delegates(self, mock_hf_module, sample_meta, sample_arrays) -> None:
        """update_meta delegates to cache store."""
        from juniper_data.storage.hf_store import HuggingFaceDatasetStore

        store = HuggingFaceDatasetStore()
        store._cache_store.save("test-1", sample_meta, sample_arrays)

        result = store.update_meta("test-1", sample_meta)
        assert result is True

    def test_list_all_metadata_delegates(self, mock_hf_module, sample_meta, sample_arrays) -> None:
        """list_all_metadata delegates to cache store."""
        from juniper_data.storage.hf_store import HuggingFaceDatasetStore

        store = HuggingFaceDatasetStore()
        store._cache_store.save("test-1", sample_meta, sample_arrays)

        result = store.list_all_metadata()
        assert len(result) == 1
