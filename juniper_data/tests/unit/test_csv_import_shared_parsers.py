#!/usr/bin/env python
"""Shared-parser identity for csv_import's whole-file and capped paths (APD-DATA-018).

``_load_csv`` and ``_load_json`` shrank because their bodies moved into
``_parse_csv_stream`` / ``_parse_json_text``. That sharing is the point: the
uncapped path and the byte-capped path must parse identically except for the
deliberate ``drop_trailing_partial`` / ``tolerate_truncated`` flags. Two
hand-maintained copies would be the fork-drift class this ecosystem keeps
filing defects about.

#326 already pins refusal, annotation, and the drop of a *short* final row.
#328 / #330 own equality-at-cap, UTF-8 cuts, minified JSON, and cache identity.
This file does not repeat those. It pins the remaining two holes that a
plausible un-extract would reopen without those suites going red:

1. An empty field is ``""``, not ``None``. The drop guard must not treat
   falsy as missing, or a legitimate last row that ends in an empty column
   is discarded on every authorised truncation.
2. Whole-file loaders and the in-memory parsers agree on the same bytes
   (LF, CRLF, headerless, JSON array, JSONL), and a *complete* JSON array
   through ``tolerate_truncated=True`` keeps every element.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

# Load-bearing side effect, NOT an unused import. ``juniper_data.generators.csv_import``
# cannot be imported on its own -- it hits a circular import through
# ``api.routes.generators`` (juniper-data#316). Importing the routes module first
# closes the cycle so this file is runnable without relying on collection order.
#
# Written as an explicit ``import_module`` call rather than a bare import with ``# noqa: F401``
# because the noqa satisfied ruff but not CodeQL, which correctly flagged an unused NAME.
importlib.import_module("juniper_data.api.routes.generators")

from juniper_data.generators.csv_import import CsvImportGenerator, CsvImportParams  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.generators]

_CSV_WITH_HEADER = "feature1,feature2,label\n1.0,2.0,A\n3.0,4.0,B\n"
_CSV_HEADERLESS = "1.0,2.0,A\n3.0,4.0,B\n"
_JSON_ARRAY = '[{"feature1": 1.0, "feature2": 2.0, "label": "A"}, {"feature1": 3.0, "feature2": 4.0, "label": "B"}]\n'
_JSONL = '{"feature1": 1.0, "feature2": 2.0, "label": 0}\n{"feature1": 3.0, "feature2": 4.0, "label": 1}\n'


def _params(*, header: bool = True) -> CsvImportParams:
    return CsvImportParams(file_path="unused.csv", header=header)


class TestEmptyFieldIsNotAPartialRow:
    """The drop guard keys on ``None`` (absent column), not on emptiness."""

    def test_empty_unquoted_field_on_the_last_row_is_kept(self) -> None:
        """``3.0,,B`` is a complete record. ``feature2`` is ``""``, not ``None``.

        Mutation: ``any(value is None ...)`` → ``any(not value ...)`` drops this
        row, and the existing short-row suite still passes because every case
        there has a genuine missing column.
        """
        text = "feature1,feature2,label\n1.0,2.0,A\n3.0,,B\n"
        params = _params()
        kept = CsvImportGenerator._parse_csv_text(text, params, drop_trailing_partial=True)
        raw = CsvImportGenerator._parse_csv_text(text, params, drop_trailing_partial=False)

        assert kept == raw
        assert len(kept) == 2
        assert kept[-1]["feature2"] == ""
        assert all(value is not None for row in kept for value in row.values())

    def test_quoted_empty_field_on_the_last_row_is_kept(self) -> None:
        """``\"\"`` is also a present empty field, not a missing column."""
        text = 'feature1,feature2,label\n1.0,2.0,A\n3.0,"",B\n'
        params = _params()
        kept = CsvImportGenerator._parse_csv_text(text, params, drop_trailing_partial=True)

        assert len(kept) == 2
        assert kept[-1]["feature2"] == ""


class TestWholeFileAndInMemoryParsersAgree:
    """The extract-method waiver's load-bearing claim: one parser, two entry points."""

    def test_load_csv_matches_parse_csv_text_for_lf(self, tmp_path: Path) -> None:
        path = tmp_path / "rows.csv"
        path.write_text(_CSV_WITH_HEADER)
        params = _params()

        from_file = CsvImportGenerator._load_csv(path, params)
        from_text = CsvImportGenerator._parse_csv_text(path.read_text(), params, drop_trailing_partial=False)

        assert from_file == from_text
        assert len(from_file) == 2

    def test_load_csv_matches_parse_csv_text_for_crlf(self, tmp_path: Path) -> None:
        """``newline=\"\"`` on both open() and StringIO is what keeps CRLF from drifting."""
        path = tmp_path / "rows.csv"
        path.write_bytes(_CSV_WITH_HEADER.replace("\n", "\r\n").encode("utf-8"))
        params = _params()

        from_file = CsvImportGenerator._load_csv(path, params)
        from_text = CsvImportGenerator._parse_csv_text(_CSV_WITH_HEADER.replace("\n", "\r\n"), params, drop_trailing_partial=False)

        assert from_file == from_text
        assert [row["label"] for row in from_file] == ["A", "B"]

    def test_load_csv_matches_parse_csv_text_without_a_header(self, tmp_path: Path) -> None:
        """header=False seeks the stream to invent fieldnames; both entry points must seek the same way."""
        path = tmp_path / "rows.csv"
        path.write_text(_CSV_HEADERLESS)
        params = _params(header=False)

        from_file = CsvImportGenerator._load_csv(path, params)
        from_text = CsvImportGenerator._parse_csv_text(path.read_text(), params, drop_trailing_partial=False)

        assert from_file == from_text
        assert list(from_file[0].keys()) == ["col_0", "col_1", "col_2"]

    def test_load_json_matches_parse_json_text_for_an_array(self, tmp_path: Path) -> None:
        path = tmp_path / "rows.json"
        path.write_text(_JSON_ARRAY)

        from_file = CsvImportGenerator._load_json(path, _params())
        from_text = CsvImportGenerator._parse_json_text(path.read_text(), tolerate_truncated=False)

        assert from_file == from_text
        assert len(from_file) == 2

    def test_load_json_matches_parse_json_text_for_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "rows.jsonl"
        path.write_text(_JSONL)

        from_file = CsvImportGenerator._load_json(path, _params())
        from_text = CsvImportGenerator._parse_json_text(path.read_text(), tolerate_truncated=False)

        assert from_file == from_text
        assert len(from_file) == 2


class TestCompleteJsonSurvivesThePartialDecoder:
    """``tolerate_truncated`` must not drop elements of a finished document."""

    def test_complete_array_keeps_every_element_when_truncation_is_tolerated(self) -> None:
        """The capped path uses this flag even when the prefix happens to be valid JSON.

        Mutation: ``_decode_partial_json_array`` breaking on a complete ``]``
        (or stopping one element early) would make the capped path silently
        shorter than the whole-file path for the same bytes.
        """
        strict = CsvImportGenerator._parse_json_text(_JSON_ARRAY, tolerate_truncated=False)
        tolerant = CsvImportGenerator._parse_json_text(_JSON_ARRAY, tolerate_truncated=True)

        assert tolerant == strict
        assert len(tolerant) == 2
