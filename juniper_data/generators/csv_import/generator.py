"""CSV/JSON import generator for custom datasets.

This module provides the CsvImportGenerator class for loading
datasets from CSV and JSON files.
"""

import csv
import io
import json
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from juniper_data.api.settings import get_settings
from juniper_data.core.constants import CHARSET_UTF8
from juniper_data.core.limits import TRUNCATION_META_KEY, InputTooLargeError, build_truncation_meta
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
    def generate(params: CsvImportParams) -> dict[str, Any]:
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
                - truncation: reserved non-array channel key, present ONLY when
                  the source exceeded its byte cap and a partial import was
                  authorised. The route pops it into ``DatasetMeta`` before
                  checksumming, so the persisted arrays stay array-only.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is unsupported.
            InputTooLargeError: If the source exceeds its byte cap and neither
                the request nor the deployment allowed truncation. Subclasses
                ValueError; the route maps it to 422.
        """
        X, y, truncation = CsvImportGenerator._load_and_preprocess(params)

        split_result = shuffle_and_split(
            X=X,
            y=y,
            train_ratio=params.train_ratio,
            test_ratio=params.test_ratio,
            seed=params.seed,
            shuffle=params.shuffle,
        )

        result: dict[str, Any] = {
            "X_train": split_result["X_train"],
            "y_train": split_result["y_train"],
            "X_test": split_result["X_test"],
            "y_test": split_result["y_test"],
            "X_full": X,
            "y_full": y,
        }

        # APD-DATA-018: hand the route the permanent truncation annotation over
        # the reserved channel key, the same way _synthetic.py hands over
        # "scaling". The route pops it BEFORE checksum + NPZ persist, so the
        # stored arrays stay array-only. Absent entirely when nothing was cut --
        # a caller must never have to distinguish "not truncated" from "the
        # generator forgot to say".
        if truncation is not None:
            result[TRUNCATION_META_KEY] = truncation

        return result

    @staticmethod
    def _resolve_bounds(params: CsvImportParams) -> tuple[int, bool]:
        """Resolve the effective byte cap and truncation opt-in for this request.

        Precedence is *request over deployment*, and it is asymmetric on purpose:

        * ``max_bytes`` -- an explicitly-supplied request value wins; otherwise
          the deployment default applies. ``model_fields_set`` is what
          distinguishes "the caller asked for the schema default" from "the
          caller said nothing", which a plain equality check against the default
          cannot do.
        * ``allow_truncation`` -- a logical OR. Either the caller opts in for
          this request, or the deployment has opted in for every request. There
          is no way to opt *out* of a deployment-wide opt-in, because the
          deployment operator is the more privileged party.

        Returns:
            ``(cap_bytes, allow_truncation)``.
        """
        settings = get_settings()
        cap = params.max_bytes if "max_bytes" in params.model_fields_set else settings.csv_import_max_bytes
        allow = bool(params.allow_truncation or settings.csv_import_allow_truncation)
        return cap, allow

    @staticmethod
    def _load_and_preprocess(params: CsvImportParams) -> tuple[np.ndarray, np.ndarray, dict[str, Any] | None]:
        """Load data from file and preprocess.

        Args:
            params: CsvImportParams instance.

        Returns:
            Tuple of ``(X, y, truncation)``; ``truncation`` is None unless the
            source exceeded its cap and a partial import was authorised.

        Raises:
            InputTooLargeError: source over the cap without an opt-in.
        """

        settings = get_settings()
        import_base = Path(settings.import_dir).resolve()
        resolved = (import_base / params.file_path).resolve()
        if not resolved.is_relative_to(import_base):
            raise ValueError(f"Path traversal detected: {params.file_path} resolves outside import directory")
        path = resolved

        if not path.exists():
            raise FileNotFoundError(f"File not found: {params.file_path}")

        file_format = params.file_format
        if file_format == "auto":
            suffix = path.suffix.lower()
            if suffix == ".csv":
                file_format = "csv"
            elif suffix in {".json", ".jsonl"}:
                file_format = "json"
            else:
                raise ValueError(f"Cannot auto-detect format for extension: {suffix}")

        # APD-DATA-018: decide the bound BEFORE reading a byte. stat() is O(1);
        # discovering the size by reading the file is precisely the cost the cap
        # exists to avoid.
        cap_bytes, allow_truncation = CsvImportGenerator._resolve_bounds(params)
        bytes_total = path.stat().st_size

        if bytes_total <= cap_bytes:
            data = CsvImportGenerator._load_csv(path, params) if file_format == "csv" else CsvImportGenerator._load_json(path, params)
            X, y = CsvImportGenerator._convert_to_arrays(data, params)
            return X, y, None

        if not allow_truncation:
            raise InputTooLargeError(source=params.file_path, bytes_total=bytes_total, cap_bytes=cap_bytes)

        chunk = CsvImportGenerator._read_capped_text(path, cap_bytes)
        data = CsvImportGenerator._parse_csv_text(chunk, params, drop_trailing_partial=True) if file_format == "csv" else CsvImportGenerator._parse_json_text(chunk, tolerate_truncated=True)

        X, y = CsvImportGenerator._convert_to_arrays(data, params)
        truncation = build_truncation_meta(
            bytes_read=len(chunk.encode(CHARSET_UTF8)),
            bytes_total=bytes_total,
            cap_bytes=cap_bytes,
            records_imported=len(data),
        )
        return X, y, truncation

    @staticmethod
    def _read_capped_text(path: Path, cap_bytes: int) -> str:
        """Read at most ``cap_bytes`` and return text trimmed to a line boundary.

        Two distinct hazards are handled here, and both are silent if missed:

        * **A split multi-byte character.** The cap is a byte offset and UTF-8 is
          variable-width, so the final character may be cut in half.
          ``errors="ignore"`` drops that partial sequence rather than raising.
        * **A split record.** Cutting at an arbitrary byte almost always lands
          mid-line, so everything after the final newline is discarded. A source
          with no newline inside the cap yields empty text, which
          ``_convert_to_arrays`` turns into the existing "No data found in file"
          error -- the correct outcome, since not one whole record fits.
        """
        with open(path, "rb") as handle:
            raw = handle.read(cap_bytes)
        text = raw.decode(CHARSET_UTF8, errors="ignore")
        cut = text.rfind("\n")
        return "" if cut == -1 else text[: cut + 1]

    @staticmethod
    def _load_csv(path: Path, params: CsvImportParams) -> list[dict]:
        """Load data from CSV file (whole file; the source is within its cap)."""
        with open(path, newline="", encoding=CHARSET_UTF8) as f:
            return CsvImportGenerator._parse_csv_stream(f, params, drop_trailing_partial=False)

    @staticmethod
    def _parse_csv_text(text: str, params: CsvImportParams, *, drop_trailing_partial: bool) -> list[dict]:
        """Parse CSV held in memory (the capped path)."""
        return CsvImportGenerator._parse_csv_stream(io.StringIO(text, newline=""), params, drop_trailing_partial=drop_trailing_partial)

    @staticmethod
    def _parse_csv_stream(stream: TextIO, params: CsvImportParams, *, drop_trailing_partial: bool) -> list[dict]:
        """Parse CSV rows from a text stream.

        Both the whole-file and the capped path go through here so the two
        cannot drift; the capped path differs only by ``drop_trailing_partial``.

        Args:
            stream: seekable text stream positioned at the start.
            params: import configuration.
            drop_trailing_partial: discard a final short row. Only the capped
                path sets this, and it guards a hazard newline-trimming alone
                does not cover: a newline **inside a quoted field** is a legal
                CSV byte, so the trim can still land mid-record and leave the
                last row missing its trailing columns. ``DictReader`` reports
                absent columns as ``None`` (an empty field is ``""``), which
                makes the two distinguishable.

        Returns:
            List of row dicts.
        """
        data: list[dict] = []
        if params.header:
            reader = csv.DictReader(stream, delimiter=params.delimiter)
        else:
            csv_reader = csv.reader(stream, delimiter=params.delimiter)
            try:
                first_row = next(csv_reader)
            except StopIteration as e:
                raise ValueError("CSV file is empty or contains only a header row") from e
            stream.seek(0)
            fieldnames = [f"col_{i}" for i in range(len(first_row))]
            reader = csv.DictReader(stream, fieldnames=fieldnames, delimiter=params.delimiter)

        data.extend(iter(reader))

        if drop_trailing_partial and data and any(value is None for value in data[-1].values()):
            data.pop()

        return data

    @staticmethod
    def _load_json(path: Path, params: CsvImportParams) -> list[dict]:
        """Load data from JSON file (whole file; the source is within its cap)."""
        with open(path, encoding=CHARSET_UTF8) as f:
            return CsvImportGenerator._parse_json_text(f.read(), tolerate_truncated=False)

    @staticmethod
    def _parse_json_text(content: str, *, tolerate_truncated: bool) -> list[dict]:
        """Parse a JSON array or JSONL document.

        Args:
            content: the document text.
            tolerate_truncated: the text may end mid-document (the capped path).
                A byte cap cannot cut a JSON **array** at a valid point -- the
                closing ``]`` is gone and the final element is very likely
                half-written -- so rather than failing, the array branch decodes
                as many complete top-level values as it can with ``raw_decode``
                and stops at the first incomplete one. The JSONL branch drops a
                final unparseable line for the same reason.

        Returns:
            List of record dicts.
        """
        content = content.strip()
        if not content:
            return []

        if content.startswith("["):
            if not tolerate_truncated:
                return json.loads(content)
            return CsvImportGenerator._decode_partial_json_array(content)

        records: list[dict] = []
        lines = [line for line in content.split("\n") if line.strip()]
        for index, line in enumerate(lines):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Only the FINAL line may be a casualty of the cap. A malformed
                # line anywhere else is a real defect in the source and must
                # still raise, or a corrupt file would import as a short one.
                if tolerate_truncated and index == len(lines) - 1:
                    break
                raise
        return records

    @staticmethod
    def _decode_partial_json_array(content: str) -> list[dict]:
        """Decode as many complete elements of a truncated JSON array as exist."""
        decoder = json.JSONDecoder()
        records: list[dict] = []
        index = content.index("[") + 1
        length = len(content)
        while True:
            while index < length and content[index] in ", \t\r\n":
                index += 1
            if index >= length or content[index] == "]":
                break
            try:
                value, index = decoder.raw_decode(content, index)
            except json.JSONDecodeError:
                break  # the element straddling the cap
            records.append(value)
        return records

    @staticmethod
    def _convert_to_arrays(data: list[dict], params: CsvImportParams) -> tuple[np.ndarray, np.ndarray]:
        """Convert loaded data to numpy arrays."""
        if not data:
            raise ValueError("No data found in file")

        all_columns = list(data[0].keys())

        feature_cols = params.feature_columns if params.feature_columns is not None else [c for c in all_columns if c != params.label_column]

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
