"""Unit tests for the CSV/JSON import generator."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from juniper_data.core.limits import (
    CSV_IMPORT_DEFAULT_ALLOW_TRUNCATION,
    CSV_IMPORT_DEFAULT_MAX_BYTES,
    TRUNCATION_META_KEY,
    InputTooLargeError,
)
from juniper_data.core.meta import pop_truncation_meta
from juniper_data.generators.csv_import import (
    VERSION,
    CsvImportGenerator,
    CsvImportParams,
    get_schema,
)

pytestmark = [pytest.mark.unit, pytest.mark.generators]

# Small enough that the fixtures below overflow it by a wide margin, so a test
# never depends on a source landing near the boundary by luck.
TINY_CAP_BYTES = 120


@pytest.fixture
def import_dir(tmp_path: Path):
    """Patch get_settings to use a temporary import directory.

    The two APD-DATA-018 bound settings are given REAL values, not left as
    MagicMock attributes. A bare MagicMock raises ``TypeError: '<=' not
    supported between instances of 'int' and 'MagicMock'`` the moment the
    generator compares a file size against the cap -- which is the good
    outcome, but only because MagicMock happens not to fake ordering. Had it
    returned a truthy mock instead, every test here would have taken the
    under-cap branch and passed while exercising nothing.
    """
    from unittest.mock import MagicMock

    mock_settings = MagicMock()
    mock_settings.import_dir = str(tmp_path)
    mock_settings.csv_import_max_bytes = CSV_IMPORT_DEFAULT_MAX_BYTES
    mock_settings.csv_import_allow_truncation = CSV_IMPORT_DEFAULT_ALLOW_TRUNCATION
    with patch("juniper_data.generators.csv_import.generator.get_settings", return_value=mock_settings):
        yield tmp_path


@pytest.fixture
def bounded_import_dir(tmp_path: Path):
    """Like ``import_dir`` but with a deliberately tiny cap, truncation off.

    Returns ``(tmp_path, mock_settings)`` so a test can flip the deployment-wide
    opt-in without rebuilding the patch.
    """
    from unittest.mock import MagicMock

    mock_settings = MagicMock()
    mock_settings.import_dir = str(tmp_path)
    mock_settings.csv_import_max_bytes = TINY_CAP_BYTES
    mock_settings.csv_import_allow_truncation = False
    with patch("juniper_data.generators.csv_import.generator.get_settings", return_value=mock_settings):
        yield tmp_path, mock_settings


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
        """The FITTED partition is normalized to [0, 1] (juniper-data#314).

        This previously asserted the bound on ``X_full``, which held only because the
        statistics were fit over every row -- test rows included -- and then applied to
        train. Under a train-only fit the bound belongs to ``X_train``; rows outside the
        training range legitimately fall outside [0, 1], and that is the point.
        """
        params = CsvImportParams(
            file_path=sample_csv_file.name,
            normalize_features=True,
            shuffle=False,
        )
        result = CsvImportGenerator.generate(params)

        assert result["X_train"].min() >= 0.0
        assert result["X_train"].max() <= 1.0 + 1e-6

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
            # 0.5 + 0.5 accounts for every row; state the validation share as 0
            # rather than leaving the 0.1 default to over-subscribe at 1.1.
            val_ratio=0.0,
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


class TestInputByteCap:
    """APD-DATA-018: the csv_import input bound, its refusal, and its annotation.

    The owner's decision (2026-09-04): cap in **bytes**; over-cap **truncates**
    rather than rejecting outright, but only after the caller has explicitly
    opted in, and the resulting dataset carries a **permanent** annotation.
    Every arm below pins one half of that; the refusal arm and the annotation
    arm are the two that make truncation safe rather than silent.
    """

    @staticmethod
    def _wide_csv(directory: Path, name: str = "wide.csv", rows: int = 40) -> Path:
        path = directory / name
        lines = ["feature1,feature2,label"]
        lines.extend(f"{i}.0,{i + 1}.0,{'A' if i % 2 else 'B'}" for i in range(rows))
        path.write_text("\n".join(lines) + "\n")
        return path

    def test_under_cap_reports_no_truncation(self, import_dir: Path) -> None:
        """A source inside the cap must not carry the reserved channel key at all.

        Absence, not ``{"truncated": False}``: a consumer must never have to
        distinguish "complete" from "the generator forgot to report".
        """
        self._wide_csv(import_dir)
        result = CsvImportGenerator.generate(CsvImportParams(file_path="wide.csv"))
        assert TRUNCATION_META_KEY not in result

    def test_over_cap_without_opt_in_is_refused(self, bounded_import_dir) -> None:
        """The default is refusal. Truncated data never reaches a caller who did not ask."""
        directory, _ = bounded_import_dir
        self._wide_csv(directory)
        with pytest.raises(InputTooLargeError) as excinfo:
            CsvImportGenerator.generate(CsvImportParams(file_path="wide.csv"))
        # The message must be actionable: it names the remedy and both numbers.
        assert "allow_truncation" in str(excinfo.value)
        assert excinfo.value.cap == TINY_CAP_BYTES
        assert excinfo.value.actual > TINY_CAP_BYTES
        assert excinfo.value.unit == "bytes"

    def test_refusal_is_a_value_error(self, bounded_import_dir) -> None:
        """Subclassing ValueError is load-bearing, not incidental.

        The route maps this type to 422; a call path that forgets to catch it
        still lands on the app-level ValueError handler's 400 rather than
        reporting a caller error as a 500.
        """
        directory, _ = bounded_import_dir
        self._wide_csv(directory)
        with pytest.raises(ValueError):
            CsvImportGenerator.generate(CsvImportParams(file_path="wide.csv"))

    def test_request_opt_in_truncates_and_annotates(self, bounded_import_dir) -> None:
        """Per-request opt-in produces a partial dataset AND its permanent record."""
        directory, _ = bounded_import_dir
        path = self._wide_csv(directory)
        result = CsvImportGenerator.generate(CsvImportParams(file_path="wide.csv", allow_truncation=True))

        annotation = result[TRUNCATION_META_KEY]
        assert annotation["truncated"] is True
        assert annotation["reason"] == "source_exceeded_byte_cap"
        assert annotation["unit"] == "bytes"
        assert annotation["cap"] == TINY_CAP_BYTES
        assert annotation["requested"] == path.stat().st_size
        assert annotation["imported"] <= TINY_CAP_BYTES
        assert 0 < annotation["records_imported"] < 40
        # The arrays must agree with the annotation -- an annotation that does
        # not match what was actually imported is worse than none.
        assert result["X_full"].shape[0] == annotation["records_imported"]

    def test_deployment_opt_in_truncates(self, bounded_import_dir) -> None:
        """The env-var / .env surface works without any request parameter."""
        directory, mock_settings = bounded_import_dir
        self._wide_csv(directory)
        mock_settings.csv_import_allow_truncation = True
        result = CsvImportGenerator.generate(CsvImportParams(file_path="wide.csv"))
        assert result[TRUNCATION_META_KEY]["truncated"] is True

    def test_explicit_request_cap_overrides_deployment_default(self, import_dir: Path) -> None:
        """An explicitly-set max_bytes wins over the deployment default.

        ``import_dir`` carries the real 128 MiB default, so the source is far
        under it; the request tightens the cap and must be obeyed.
        """
        self._wide_csv(import_dir)
        with pytest.raises(InputTooLargeError):
            CsvImportGenerator.generate(CsvImportParams(file_path="wide.csv", max_bytes=TINY_CAP_BYTES))

    def test_truncation_stops_at_a_record_boundary(self, bounded_import_dir) -> None:
        """No half-row survives the cut.

        The cap is a byte offset and lands mid-line essentially always, so the
        parser must never see a partial record. Every imported row must have
        both features populated.
        """
        directory, _ = bounded_import_dir
        self._wide_csv(directory)
        result = CsvImportGenerator.generate(CsvImportParams(file_path="wide.csv", allow_truncation=True))
        assert not np.isnan(result["X_full"]).any()
        assert result["X_full"].shape[1] == 2

    @pytest.mark.parametrize(
        ("label", "text"),
        [
            ("cut inside a quoted field, before the final comma", 'feature1,feature2,label\n1.0,2.0,A\n3.0,"q\n'),
            ("quoted field with an embedded newline spanning the cut", 'feature1,feature2,label\n1.0,2.0,A\n"a\nb",4.0\n'),
            ("short final row that still ends in a newline", "feature1,feature2,label\n1.0,2.0,A\n3.0,4.0\n"),
        ],
    )
    def test_partial_final_row_is_dropped(self, import_dir: Path, label: str, text: str) -> None:
        """A record left incomplete by the cut must not become a data row.

        Trimming to the last newline is NOT sufficient on its own, which is the
        whole reason the drop exists: a newline inside a quoted field is a legal
        CSV byte, so the trim can land mid-record and still end on a newline.
        ``DictReader`` then reports the absent trailing columns as ``None`` --
        distinct from an empty field, which is ``""``.

        Exercised at the parser rather than through a byte cap because the
        arithmetic needed to make a cap land inside a quoted field is incidental
        to the behaviour being pinned. Each case here was observed live before
        being written down; the first draft of this test used a shape where all
        three fields happen to survive, and a mutation run caught that it
        proved nothing.
        """
        params = CsvImportParams(file_path="unused.csv")
        kept = CsvImportGenerator._parse_csv_text(text, params, drop_trailing_partial=True)
        dropped = CsvImportGenerator._parse_csv_text(text, params, drop_trailing_partial=False)

        assert len(kept) == len(dropped) - 1, label
        assert all(value is not None for row in kept for value in row.values()), label
        assert any(value is None for value in dropped[-1].values()), label

    def test_truncated_json_array_keeps_only_complete_elements(self, bounded_import_dir) -> None:
        """A byte cap cannot cut a JSON array at a valid point; decode what fits."""
        directory, _ = bounded_import_dir
        path = directory / "wide.json"
        elements = [f'{{"feature1": {i}.0, "feature2": {i + 1}.0, "label": "A"}}' for i in range(40)]
        path.write_text("[\n" + ",\n".join(elements) + "\n]\n")

        result = CsvImportGenerator.generate(CsvImportParams(file_path="wide.json", allow_truncation=True))
        annotation = result[TRUNCATION_META_KEY]
        assert 0 < annotation["records_imported"] < 40
        assert result["X_full"].shape[0] == annotation["records_imported"]

    def test_truncated_jsonl_drops_the_partial_final_line(self, bounded_import_dir) -> None:
        """JSONL truncation drops only the last line, and only when unparseable."""
        directory, _ = bounded_import_dir
        path = directory / "wide.jsonl"
        path.write_text("\n".join(f'{{"feature1": {i}.0, "feature2": {i + 1}.0, "label": 0}}' for i in range(40)) + "\n")

        result = CsvImportGenerator.generate(CsvImportParams(file_path="wide.jsonl", allow_truncation=True))
        assert 0 < result[TRUNCATION_META_KEY]["records_imported"] < 40

    def test_malformed_line_that_is_not_the_last_still_raises(self, bounded_import_dir) -> None:
        """Tolerating the cap's casualty must not tolerate a corrupt source.

        Without this arm the truncation path would silently import a corrupt
        file as a short one -- the same silent-partial class the annotation
        exists to prevent, reintroduced through the error handling.
        """
        directory, _ = bounded_import_dir
        path = directory / "corrupt.jsonl"
        good = '{"feature1": 1.0, "feature2": 2.0, "label": 0}'
        path.write_text(f"{good}\nNOT JSON AT ALL\n{good}\n{good}\n{good}\n{good}\n")

        with pytest.raises(json.JSONDecodeError):
            CsvImportGenerator.generate(CsvImportParams(file_path="corrupt.jsonl", allow_truncation=True))

    def test_pop_truncation_meta_leaves_arrays_only(self, bounded_import_dir) -> None:
        """The route's pop must strip the non-array key before checksumming.

        If it did not, the reserved key would reach ``compute_checksum`` and the
        NPZ writer, which accept arrays only.
        """
        directory, _ = bounded_import_dir
        self._wide_csv(directory)
        result = CsvImportGenerator.generate(CsvImportParams(file_path="wide.csv", allow_truncation=True))

        annotation = pop_truncation_meta(result)
        assert annotation is not None and annotation["truncated"] is True
        assert TRUNCATION_META_KEY not in result
        assert all(isinstance(value, np.ndarray) for value in result.values())

    def test_pop_truncation_meta_returns_none_when_absent(self) -> None:
        """None, not {} -- so ``meta.truncation`` alone answers "is this partial"."""
        assert pop_truncation_meta({"X_full": np.zeros((1, 1), dtype=np.float32)}) is None

    # ------------------------------------------------------------------
    # The bound must not be defeatable by the party it exists to bound.
    # Both arms below pin a finding raised in review on juniper-data#326.
    # ------------------------------------------------------------------

    def test_request_cannot_RAISE_the_deployment_cap(self, bounded_import_dir) -> None:
        """A request may only lower the cap. Otherwise the DoS bound is caller-controlled.

        The first draft let an explicitly-supplied ``max_bytes`` win outright,
        so ``max_bytes: 10000000000`` skipped the cap entirely -- and a
        generated client that serialises schema defaults would have silently
        raised a *lower* operator ceiling on every request. The existing
        override arm only covers tightening, which is why this one exists.
        """
        directory, _ = bounded_import_dir
        self._wide_csv(directory)
        with pytest.raises(InputTooLargeError) as excinfo:
            CsvImportGenerator.generate(CsvImportParams(file_path="wide.csv", max_bytes=10_000_000_000))
        # The operator's ceiling is what governs, not the caller's request.
        assert excinfo.value.cap == TINY_CAP_BYTES

    def test_a_lying_stat_does_not_bypass_the_cap(self, bounded_import_dir) -> None:
        """The READ enforces the bound; ``stat`` is only a cheap pre-check.

        A FIFO reports ``st_size == 0``, and a file in a shared ``import_dir``
        can grow between the stat and the open. Either takes the under-cap
        branch on stat's word alone, so the ingestion path must re-check what it
        actually read rather than trusting the number it was told.
        """
        directory, _ = bounded_import_dir
        path = self._wide_csv(directory)
        real_stat = Path.stat

        class _FakeStat:
            """Reports st_size == 0 the way a FIFO does, delegating everything else."""

            def __init__(self, real) -> None:  # noqa: ANN001
                self._real = real
                self.st_size = 0

            def __getattr__(self, item):  # noqa: ANN001, ANN204
                return getattr(self._real, item)

        def lying_stat(self, *args, **kwargs):  # noqa: ANN001, ANN202
            result = real_stat(self, *args, **kwargs)
            return _FakeStat(result) if self.name == path.name else result

        with patch.object(Path, "stat", lying_stat), pytest.raises(InputTooLargeError):
            CsvImportGenerator.generate(CsvImportParams(file_path="wide.csv"))

    def test_non_positive_cap_is_refused_rather_than_inverting_read(self) -> None:
        """``read(n)`` with n < 0 means "read everything" -- the cap's exact opposite.

        Reachable only through operator misconfiguration, which is why
        ``Settings.csv_import_max_bytes`` carries ``gt=0``; this is the second
        line of that defence, at the one place bytes are actually ingested.
        """
        with pytest.raises(ValueError, match="must be positive"):
            CsvImportGenerator._read_capped_bytes(Path("/dev/null"), -1)
        with pytest.raises(ValueError, match="must be positive"):
            CsvImportGenerator._read_capped_bytes(Path("/dev/null"), 0)

    def test_settings_reject_a_non_positive_cap(self) -> None:
        """A mistyped env var must fail deployment loudly, not silently unbound the read."""
        from pydantic import ValidationError

        from juniper_data.api.settings import Settings

        with pytest.raises(ValidationError):
            Settings(csv_import_max_bytes=-1)
        with pytest.raises(ValidationError):
            Settings(csv_import_max_bytes=0)

    def test_bind_deployment_defaults_puts_effective_policy_in_dump(self, bounded_import_dir) -> None:
        """The cache key must follow the resolved cap and opt-in, not Field defaults.

        After the request-cannot-raise-the-cap clamp, omit-max_bytes and an
        explicit 128 MiB schema default resolve to the SAME effective cap
        (the deployment ceiling). Binding must record that ceiling, not leave
        the dump at 128 MiB -- otherwise a later restart that raises the cap
        reuses the truncated artifact. Global allow_truncation must appear in
        the dump for the same reason.
        """
        _directory, mock_settings = bounded_import_dir
        omitted = CsvImportGenerator.bind_deployment_defaults(CsvImportParams(file_path="wide.csv", allow_truncation=True))
        explicit_default = CsvImportGenerator.bind_deployment_defaults(CsvImportParams(file_path="wide.csv", allow_truncation=True, max_bytes=CSV_IMPORT_DEFAULT_MAX_BYTES))
        assert omitted.max_bytes == TINY_CAP_BYTES
        assert explicit_default.max_bytes == TINY_CAP_BYTES
        assert omitted.model_dump()["max_bytes"] == explicit_default.model_dump()["max_bytes"]

        mock_settings.csv_import_max_bytes = CSV_IMPORT_DEFAULT_MAX_BYTES
        omitted_wide = CsvImportGenerator.bind_deployment_defaults(CsvImportParams(file_path="wide.csv", allow_truncation=True))
        assert omitted_wide.max_bytes == CSV_IMPORT_DEFAULT_MAX_BYTES
        assert omitted.model_dump()["max_bytes"] != omitted_wide.model_dump()["max_bytes"]

        mock_settings.csv_import_max_bytes = TINY_CAP_BYTES
        mock_settings.csv_import_allow_truncation = True
        inherited = CsvImportGenerator.bind_deployment_defaults(CsvImportParams(file_path="wide.csv"))
        assert inherited.allow_truncation is True
        assert inherited.model_dump()["allow_truncation"] is True

    def test_truncated_minified_json_array_keeps_complete_elements(self, bounded_import_dir) -> None:
        """json.dumps / JSON.stringify emit one line. Newline-trimming that
        prefix yields empty text and "No data found" -- the decoder must see
        the byte prefix itself.
        """
        directory, _ = bounded_import_dir
        path = directory / "wide.json"
        elements = [{"feature1": float(i), "feature2": float(i + 1), "label": "A"} for i in range(40)]
        path.write_text(json.dumps(elements, separators=(",", ":")))
        assert "\n" not in path.read_text()

        result = CsvImportGenerator.generate(CsvImportParams(file_path="wide.json", allow_truncation=True))
        annotation = result[TRUNCATION_META_KEY]
        assert 0 < annotation["records_imported"] < 40
        assert result["X_full"].shape[0] == annotation["records_imported"]

    def test_unclosed_quote_drops_last_row_even_when_all_fields_are_present(self) -> None:
        """A 2-column file whose unclosed quote swallows later lines has no None.

        The short-row guard therefore cannot see the damage; the quote scan must.
        """
        text = 'id,value\n1,a\n2,"multi\n3,b\n'
        params = CsvImportParams(file_path="unused.csv")
        kept = CsvImportGenerator._parse_csv_text(text, params, drop_trailing_partial=True)
        dropped = CsvImportGenerator._parse_csv_text(text, params, drop_trailing_partial=False)

        assert len(kept) == len(dropped) - 1
        assert kept == [{"id": "1", "value": "a"}]
        assert all(value is not None for row in kept for value in row.values())
        assert all(value is not None for value in dropped[-1].values())

    # DROPPED ON HARVEST: `test_explicit_schema_default_max_bytes_overrides_tighter_deployment`.
    #
    # It asserted that an explicit `max_bytes` equal to the 128 MiB schema default must
    # WIN over a tighter deployment ceiling. `_resolve_bounds` deliberately decides the
    # other way -- "a request may only LOWER the cap, never raise it" -- precisely because
    # a generated client that serialises Field defaults would otherwise send the schema's
    # own value on every request and silently override a lower operator ceiling. A bound
    # the bounded party can raise is not a bound.
    #
    # The surviving half of that scenario is pinned the correct way round by
    # `test_csv_import_cap_enforcement.py::test_explicit_schema_default_is_clamped_to_the_deployment_ceiling`.

    def test_no_newline_inside_cap_fails_closed(self, bounded_import_dir) -> None:
        """A cut that contains not one whole record must not import a half-row.

        ``_read_capped_text`` discards everything after the last newline; a
        source with no newline inside the cap yields empty text, which
        ``_convert_to_arrays`` refuses. Returning the unterminated bytes
        instead would let DictReader materialise a short row as data.
        """
        directory, _ = bounded_import_dir
        (directory / "oneline.csv").write_text("feature1,feature2,label," + ("x" * 200), encoding="utf-8")
        with pytest.raises(ValueError, match="No data found"):
            CsvImportGenerator.generate(CsvImportParams(file_path="oneline.csv", allow_truncation=True))

    def test_request_cannot_opt_out_of_deployment_allow_truncation(self, bounded_import_dir) -> None:
        """The OR is asymmetric: a request cannot override a deployment-wide opt-in.

        The deployment operator is the more privileged party. If a caller
        could pass ``allow_truncation=false`` and undo ``JUNIPER_DATA_CSV_IMPORT_ALLOW_TRUNCATION``,
        the env-var surface the owner required for CLI callers would be a
        suggestion rather than a bound.
        """
        directory, mock_settings = bounded_import_dir
        self._wide_csv(directory)
        mock_settings.csv_import_allow_truncation = True
        result = CsvImportGenerator.generate(CsvImportParams(file_path="wide.csv", allow_truncation=False))
        assert result[TRUNCATION_META_KEY]["truncated"] is True

    def test_source_exactly_at_cap_is_complete(self, import_dir: Path) -> None:
        """A source whose size equals the cap is complete, not over it.

        ``bytes_total <= cap_bytes`` is the contract. Mutating it to ``<``
        would refuse a file that fits, which is an off-by-one that turns a
        valid import into a 422 -- or, with opt-in, a false truncation
        annotation on a complete dataset.
        """
        path = self._wide_csv(import_dir, rows=4)
        cap = path.stat().st_size
        result = CsvImportGenerator.generate(CsvImportParams(file_path="wide.csv", max_bytes=cap, shuffle=False))
        assert TRUNCATION_META_KEY not in result
        assert result["X_full"].shape[0] == 4

    def test_utf8_split_character_does_not_raise(self, import_dir: Path) -> None:
        """A cap that lands inside a multi-byte UTF-8 sequence must not 500.

        The cap is a byte offset and UTF-8 is variable-width, so the final
        character is routinely cut in half. ``errors='ignore'`` drops that
        partial sequence; a strict decode would raise ``UnicodeDecodeError``
        and the route's bare re-raise would surface it as a 500 -- a caller
        error reported as a server fault. The complete records before the
        cut must still import.
        """
        complete = "feature1,feature2,label\n1.0,2.0,A\n"
        remainder = "€,2.0,B\n"
        raw = complete.encode("utf-8") + remainder.encode("utf-8")
        (import_dir / "split.csv").write_bytes(raw)
        cap = len(complete.encode("utf-8")) + 1  # first byte of U+20AC only
        assert cap < len(raw)

        result = CsvImportGenerator.generate(CsvImportParams(file_path="split.csv", max_bytes=cap, allow_truncation=True, shuffle=False))
        assert result["X_full"].shape[0] == 1
        assert result[TRUNCATION_META_KEY]["truncated"] is True
