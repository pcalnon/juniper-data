"""Unit tests for the CSV/JSON import generator."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from juniper_data.generators.csv_import import (
    VERSION,
    CsvImportGenerator,
    CsvImportParams,
    get_schema,
)


@pytest.fixture
def import_dir(tmp_path: Path):
    """Patch get_settings to use a temporary import directory."""
    from unittest.mock import MagicMock

    mock_settings = MagicMock()
    mock_settings.import_dir = str(tmp_path)
    with patch("juniper_data.generators.csv_import.generator.get_settings", return_value=mock_settings):
        yield tmp_path


@pytest.fixture
def sample_csv_file(import_dir: Path) -> Path:
    """Create a sample CSV file for testing."""
    csv_file = import_dir / "sample.csv"
    csv_file.write_text("feature1,feature2,label\n1.0,2.0,A\n3.0,4.0,B\n5.0,6.0,A\n7.0,8.0,B\n")
    return csv_file


@pytest.fixture
def sample_json_file(import_dir: Path) -> Path:
    """Create a sample JSON file for testing."""
    json_file = import_dir / "sample.json"
    json_file.write_text('[{"feature1": 1.0, "feature2": 2.0, "label": "A"},{"feature1": 3.0, "feature2": 4.0, "label": "B"},{"feature1": 5.0, "feature2": 6.0, "label": "A"},{"feature1": 7.0, "feature2": 8.0, "label": "B"}]')
    return json_file


@pytest.fixture
def sample_jsonl_file(import_dir: Path) -> Path:
    """Create a sample JSONL file for testing."""
    jsonl_file = import_dir / "sample.jsonl"
    jsonl_file.write_text('{"feature1": 1.0, "feature2": 2.0, "label": 0}\n{"feature1": 3.0, "feature2": 4.0, "label": 1}\n{"feature1": 5.0, "feature2": 6.0, "label": 0}\n{"feature1": 7.0, "feature2": 8.0, "label": 1}\n')
    return jsonl_file


class TestCsvImportParams:
    """Tests for CsvImportParams validation."""

    def test_valid_params(self) -> None:
        """Valid parameters should be accepted."""
        params = CsvImportParams(
            file_path="data/file.csv",
            feature_columns=["col1", "col2"],
            label_column="target",
        )
        assert params.file_path == "data/file.csv"
        assert params.feature_columns == ["col1", "col2"]
        assert params.label_column == "target"

    def test_default_values(self) -> None:
        """Default values should be set correctly."""
        params = CsvImportParams(file_path="data/file.csv")
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
            file_path=sample_csv_file.name,
            seed=42,
        )
        result = CsvImportGenerator.generate(params)

        assert result["X_full"].shape == (4, 2)
        assert result["y_full"].shape == (4, 2)

    def test_load_json_file(self, sample_json_file: Path) -> None:
        """Should load data from JSON file."""
        params = CsvImportParams(
            file_path=sample_json_file.name,
            seed=42,
        )
        result = CsvImportGenerator.generate(params)

        assert result["X_full"].shape == (4, 2)
        assert result["y_full"].shape == (4, 2)

    def test_load_jsonl_file(self, sample_jsonl_file: Path) -> None:
        """Should load data from JSONL file."""
        params = CsvImportParams(
            file_path=sample_jsonl_file.name,
            seed=42,
        )
        result = CsvImportGenerator.generate(params)

        assert result["X_full"].shape == (4, 2)
        assert result["y_full"].shape == (4, 2)

    def test_feature_values(self, sample_csv_file: Path) -> None:
        """Feature values should be correctly parsed."""
        params = CsvImportParams(
            file_path=sample_csv_file.name,
            shuffle=False,
        )
        result = CsvImportGenerator.generate(params)

        expected_X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        np.testing.assert_array_equal(result["X_full"], expected_X)

    def test_one_hot_labels(self, sample_csv_file: Path) -> None:
        """Labels should be one-hot encoded."""
        params = CsvImportParams(
            file_path=sample_csv_file.name,
            one_hot_labels=True,
            shuffle=False,
        )
        result = CsvImportGenerator.generate(params)

        row_sums = result["y_full"].sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(4))

    def test_non_one_hot_labels(self, sample_csv_file: Path) -> None:
        """Labels should be indices when one_hot=False."""
        params = CsvImportParams(
            file_path=sample_csv_file.name,
            one_hot_labels=False,
            shuffle=False,
        )
        result = CsvImportGenerator.generate(params)

        assert result["y_full"].shape == (4, 1)

    def test_normalize_features(self, sample_csv_file: Path) -> None:
        """Features should be normalized to [0, 1]."""
        params = CsvImportParams(
            file_path=sample_csv_file.name,
            normalize_features=True,
            shuffle=False,
        )
        result = CsvImportGenerator.generate(params)

        assert result["X_full"].min() >= 0.0
        assert result["X_full"].max() <= 1.0

    def test_file_not_found(self, import_dir: Path) -> None:
        """Should raise FileNotFoundError for missing file."""
        params = CsvImportParams(file_path="nonexistent_file.csv")

        with pytest.raises(FileNotFoundError):
            CsvImportGenerator.generate(params)

    def test_train_test_split(self, sample_csv_file: Path) -> None:
        """Train/test split should work correctly."""
        params = CsvImportParams(
            file_path=sample_csv_file.name,
            train_ratio=0.5,
            test_ratio=0.5,
            seed=42,
        )
        result = CsvImportGenerator.generate(params)

        assert len(result["X_train"]) == 2
        assert len(result["X_test"]) == 2

    def test_auto_detect_unsupported_extension(self, import_dir: Path) -> None:
        """Unsupported file extension should raise ValueError."""
        xml_file = import_dir / "test.xml"
        xml_file.write_text("<data></data>")

        params = CsvImportParams(file_path="test.xml")
        with pytest.raises(ValueError, match="Cannot auto-detect format"):
            CsvImportGenerator.generate(params)

    def test_csv_without_header(self, import_dir: Path) -> None:
        """Should load headerless CSV with auto-generated column names."""
        csv_file = import_dir / "no_header.csv"
        csv_file.write_text("1.0,2.0,A\n3.0,4.0,B\n5.0,6.0,A\n7.0,8.0,B\n")

        params = CsvImportParams(
            file_path="no_header.csv",
            header=False,
            label_column="col_2",
            seed=42,
        )
        result = CsvImportGenerator.generate(params)

        assert result["X_full"].shape == (4, 2)
        assert result["y_full"].shape == (4, 2)

    def test_json_jsonl_format(self, import_dir: Path) -> None:
        """Should load JSONL (non-array) format via the else branch."""
        jsonl_file = import_dir / "test.json"
        jsonl_file.write_text('{"feature1": 1.0, "feature2": 2.0, "label": "A"}\n{"feature1": 3.0, "feature2": 4.0, "label": "B"}\n')

        params = CsvImportParams(file_path="test.json", seed=42)
        result = CsvImportGenerator.generate(params)

        assert result["X_full"].shape == (2, 2)

    def test_convert_to_arrays_empty_data(self, import_dir: Path) -> None:
        """Empty file should raise ValueError."""
        csv_file = import_dir / "empty.csv"
        csv_file.write_text("feature1,feature2,label\n")

        params = CsvImportParams(file_path="empty.csv", seed=42)
        with pytest.raises(ValueError, match="No data found"):
            CsvImportGenerator.generate(params)

    def test_feature_columns_explicit(self, import_dir: Path) -> None:
        """Explicit feature_columns should select only those columns."""
        csv_file = import_dir / "multi_col.csv"
        csv_file.write_text("a,b,c,label\n1.0,2.0,3.0,A\n4.0,5.0,6.0,B\n")

        params = CsvImportParams(
            file_path="multi_col.csv",
            feature_columns=["a", "c"],
            seed=42,
        )
        result = CsvImportGenerator.generate(params)

        assert result["X_full"].shape == (2, 2)

    def test_non_numeric_feature_values(self, import_dir: Path) -> None:
        """Non-numeric feature values should be replaced with 0.0."""
        csv_file = import_dir / "non_numeric.csv"
        csv_file.write_text("feature1,feature2,label\n1.0,hello,A\n3.0,world,B\n")

        params = CsvImportParams(
            file_path="non_numeric.csv",
            shuffle=False,
            seed=42,
        )
        result = CsvImportGenerator.generate(params)

        assert result["X_full"][0, 1] == 0.0
        assert result["X_full"][1, 1] == 0.0

    def test_empty_csv_without_header_raises(self, import_dir: Path) -> None:
        """Empty CSV with header=False should raise ValueError."""
        csv_file = import_dir / "truly_empty.csv"
        csv_file.write_text("")

        params = CsvImportParams(
            file_path="truly_empty.csv",
            header=False,
            seed=42,
        )
        with pytest.raises(ValueError, match="CSV file is empty"):
            CsvImportGenerator.generate(params)

    def test_normalize_with_constant_feature(self, import_dir: Path) -> None:
        """Normalization with a constant feature column should not produce NaN."""
        csv_file = import_dir / "constant.csv"
        csv_file.write_text("feature1,feature2,label\n5.0,1.0,A\n5.0,2.0,B\n5.0,3.0,A\n")

        params = CsvImportParams(
            file_path="constant.csv",
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
            file_path=sample_csv_file.name,
            file_format="csv",
            seed=42,
        )
        result = CsvImportGenerator.generate(params)

        assert result["X_full"].shape == (4, 2)

    def test_explicit_json_format(self, sample_json_file: Path) -> None:
        """Explicit file_format='json' should bypass auto-detect."""
        params = CsvImportParams(
            file_path=sample_json_file.name,
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
