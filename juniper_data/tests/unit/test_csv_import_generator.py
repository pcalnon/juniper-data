"""Unit tests for the CSV/JSON import generator."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from juniper_data.generators.csv_import import (
    VERSION,
    CsvImportGenerator,
    CsvImportParams,
    get_schema,
)


@pytest.fixture
def sample_csv_file() -> Path:
    """Create a sample CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("feature1,feature2,label\n")
        f.write("1.0,2.0,A\n")
        f.write("3.0,4.0,B\n")
        f.write("5.0,6.0,A\n")
        f.write("7.0,8.0,B\n")
        return Path(f.name)


@pytest.fixture
def sample_json_file() -> Path:
    """Create a sample JSON file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write('[{"feature1": 1.0, "feature2": 2.0, "label": "A"},')
        f.write('{"feature1": 3.0, "feature2": 4.0, "label": "B"},')
        f.write('{"feature1": 5.0, "feature2": 6.0, "label": "A"},')
        f.write('{"feature1": 7.0, "feature2": 8.0, "label": "B"}]')
        return Path(f.name)


@pytest.fixture
def sample_jsonl_file() -> Path:
    """Create a sample JSONL file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"feature1": 1.0, "feature2": 2.0, "label": 0}\n')
        f.write('{"feature1": 3.0, "feature2": 4.0, "label": 1}\n')
        f.write('{"feature1": 5.0, "feature2": 6.0, "label": 0}\n')
        f.write('{"feature1": 7.0, "feature2": 8.0, "label": 1}\n')
        return Path(f.name)


class TestCsvImportParams:
    """Tests for CsvImportParams validation."""

    def test_valid_params(self) -> None:
        """Valid parameters should be accepted."""
        params = CsvImportParams(
            file_path="/path/to/file.csv",
            feature_columns=["col1", "col2"],
            label_column="target",
        )
        assert params.file_path == "/path/to/file.csv"
        assert params.feature_columns == ["col1", "col2"]
        assert params.label_column == "target"

    def test_default_values(self) -> None:
        """Default values should be set correctly."""
        params = CsvImportParams(file_path="/path/to/file.csv")
        assert params.file_format == "auto"
        assert params.feature_columns is None
        assert params.label_column == "label"
        assert params.delimiter == ","
        assert params.header is True
        assert params.one_hot_labels is True
        assert params.normalize_features is False


class TestCsvImportGenerator:
    """Tests for CsvImportGenerator."""

    def test_load_csv_file(self, sample_csv_file: Path) -> None:
        """Should load data from CSV file."""
        params = CsvImportParams(
            file_path=str(sample_csv_file),
            seed=42,
        )
        result = CsvImportGenerator.generate(params)

        assert result["X_full"].shape == (4, 2)
        assert result["y_full"].shape == (4, 2)

    def test_load_json_file(self, sample_json_file: Path) -> None:
        """Should load data from JSON file."""
        params = CsvImportParams(
            file_path=str(sample_json_file),
            seed=42,
        )
        result = CsvImportGenerator.generate(params)

        assert result["X_full"].shape == (4, 2)
        assert result["y_full"].shape == (4, 2)

    def test_load_jsonl_file(self, sample_jsonl_file: Path) -> None:
        """Should load data from JSONL file."""
        params = CsvImportParams(
            file_path=str(sample_jsonl_file),
            seed=42,
        )
        result = CsvImportGenerator.generate(params)

        assert result["X_full"].shape == (4, 2)
        assert result["y_full"].shape == (4, 2)

    def test_feature_values(self, sample_csv_file: Path) -> None:
        """Feature values should be correctly parsed."""
        params = CsvImportParams(
            file_path=str(sample_csv_file),
            shuffle=False,
        )
        result = CsvImportGenerator.generate(params)

        expected_X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        np.testing.assert_array_equal(result["X_full"], expected_X)

    def test_one_hot_labels(self, sample_csv_file: Path) -> None:
        """Labels should be one-hot encoded."""
        params = CsvImportParams(
            file_path=str(sample_csv_file),
            one_hot_labels=True,
            shuffle=False,
        )
        result = CsvImportGenerator.generate(params)

        row_sums = result["y_full"].sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(4))

    def test_non_one_hot_labels(self, sample_csv_file: Path) -> None:
        """Labels should be indices when one_hot=False."""
        params = CsvImportParams(
            file_path=str(sample_csv_file),
            one_hot_labels=False,
            shuffle=False,
        )
        result = CsvImportGenerator.generate(params)

        assert result["y_full"].shape == (4, 1)

    def test_normalize_features(self, sample_csv_file: Path) -> None:
        """Features should be normalized to [0, 1]."""
        params = CsvImportParams(
            file_path=str(sample_csv_file),
            normalize_features=True,
            shuffle=False,
        )
        result = CsvImportGenerator.generate(params)

        assert result["X_full"].min() >= 0.0
        assert result["X_full"].max() <= 1.0

    def test_file_not_found(self) -> None:
        """Should raise FileNotFoundError for missing file."""
        params = CsvImportParams(file_path="/nonexistent/path/file.csv")

        with pytest.raises(FileNotFoundError):
            CsvImportGenerator.generate(params)

    def test_train_test_split(self, sample_csv_file: Path) -> None:
        """Train/test split should work correctly."""
        params = CsvImportParams(
            file_path=str(sample_csv_file),
            train_ratio=0.5,
            test_ratio=0.5,
            seed=42,
        )
        result = CsvImportGenerator.generate(params)

        assert len(result["X_train"]) == 2
        assert len(result["X_test"]) == 2

    def test_auto_detect_unsupported_extension(self) -> None:
        """Unsupported file extension should raise ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write("<data></data>")
            xml_path = f.name

        params = CsvImportParams(file_path=xml_path)
        with pytest.raises(ValueError, match="Cannot auto-detect format"):
            CsvImportGenerator.generate(params)

    def test_csv_without_header(self) -> None:
        """Should load headerless CSV with auto-generated column names."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("1.0,2.0,A\n")
            f.write("3.0,4.0,B\n")
            f.write("5.0,6.0,A\n")
            f.write("7.0,8.0,B\n")
            csv_path = f.name

        params = CsvImportParams(
            file_path=csv_path,
            header=False,
            label_column="col_2",
            seed=42,
        )
        result = CsvImportGenerator.generate(params)

        assert result["X_full"].shape == (4, 2)
        assert result["y_full"].shape == (4, 2)

    def test_json_jsonl_format(self) -> None:
        """Should load JSONL (non-array) format via the else branch."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"feature1": 1.0, "feature2": 2.0, "label": "A"}\n')
            f.write('{"feature1": 3.0, "feature2": 4.0, "label": "B"}\n')
            jsonl_path = f.name

        params = CsvImportParams(file_path=jsonl_path, seed=42)
        result = CsvImportGenerator.generate(params)

        assert result["X_full"].shape == (2, 2)

    def test_convert_to_arrays_empty_data(self) -> None:
        """Empty file should raise ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("feature1,feature2,label\n")
            csv_path = f.name

        params = CsvImportParams(file_path=csv_path, seed=42)
        with pytest.raises(ValueError, match="No data found"):
            CsvImportGenerator.generate(params)

    def test_feature_columns_explicit(self) -> None:
        """Explicit feature_columns should select only those columns."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("a,b,c,label\n")
            f.write("1.0,2.0,3.0,A\n")
            f.write("4.0,5.0,6.0,B\n")
            csv_path = f.name

        params = CsvImportParams(
            file_path=csv_path,
            feature_columns=["a", "c"],
            seed=42,
        )
        result = CsvImportGenerator.generate(params)

        assert result["X_full"].shape == (2, 2)

    def test_non_numeric_feature_values(self) -> None:
        """Non-numeric feature values should be replaced with 0.0."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("feature1,feature2,label\n")
            f.write("1.0,hello,A\n")
            f.write("3.0,world,B\n")
            csv_path = f.name

        params = CsvImportParams(
            file_path=csv_path,
            shuffle=False,
            seed=42,
        )
        result = CsvImportGenerator.generate(params)

        assert result["X_full"][0, 1] == 0.0
        assert result["X_full"][1, 1] == 0.0

    def test_normalize_with_constant_feature(self) -> None:
        """Normalization with a constant feature column should not produce NaN."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("feature1,feature2,label\n")
            f.write("5.0,1.0,A\n")
            f.write("5.0,2.0,B\n")
            f.write("5.0,3.0,A\n")
            csv_path = f.name

        params = CsvImportParams(
            file_path=csv_path,
            normalize_features=True,
            shuffle=False,
            seed=42,
        )
        result = CsvImportGenerator.generate(params)

        assert not np.any(np.isnan(result["X_full"]))
        assert result["X_full"][:, 0].min() == 0.0
        assert result["X_full"][:, 0].max() == 0.0

    def test_explicit_csv_format(self, sample_csv_file: Path) -> None:
        """Explicit file_format='csv' should bypass auto-detect."""
        params = CsvImportParams(
            file_path=str(sample_csv_file),
            file_format="csv",
            seed=42,
        )
        result = CsvImportGenerator.generate(params)

        assert result["X_full"].shape == (4, 2)

    def test_explicit_json_format(self, sample_json_file: Path) -> None:
        """Explicit file_format='json' should bypass auto-detect."""
        params = CsvImportParams(
            file_path=str(sample_json_file),
            file_format="json",
            seed=42,
        )
        result = CsvImportGenerator.generate(params)

        assert result["X_full"].shape == (4, 2)


class TestGetSchema:
    """Tests for get_schema function."""

    def test_returns_dict(self) -> None:
        """get_schema should return a dictionary."""
        schema = get_schema()
        assert isinstance(schema, dict)

    def test_schema_has_properties(self) -> None:
        """Schema should have properties key."""
        schema = get_schema()
        assert "properties" in schema


class TestVersion:
    """Tests for VERSION constant."""

    def test_version_format(self) -> None:
        """VERSION should be a valid semver string."""
        parts = VERSION.split(".")
        assert len(parts) == 3
        assert all(part.isdigit() for part in parts)
