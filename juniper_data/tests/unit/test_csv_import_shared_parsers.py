#!/usr/bin/env python
"""Shared-parser identity for csv_import's whole-file and capped paths (APD-DATA-018).

``generate()``'s uncapped branch and its byte-capped branch must parse identically
except for the deliberate ``drop_trailing_partial`` / ``tolerate_truncated`` flags.
Two hand-maintained copies would be the fork-drift class this ecosystem keeps
filing defects about.

Harvest note: this suite was authored against an intermediate design in which the
whole-file path was still wrapped in ``_load_csv`` / ``_load_json``. What landed
went further -- ``generate()`` calls ``_parse_csv_text`` / ``_parse_json_text``
directly and the wrappers were removed -- so ``TestWholeFileAndInMemoryParsersAgree``
is aimed at ``generate()`` instead of at two names that no longer exist. The
property is the same one; only the entry point moved.

#326 already pins refusal, annotation, and the drop of a *short* final row.
#328 / #330 own equality-at-cap, UTF-8 cuts, minified JSON, and cache identity.
This file does not repeat those. It pins the remaining two holes that a
plausible un-extract would reopen without those suites going red:

1. An empty field is ``""``, not ``None``. The drop guard must not treat
   falsy as missing, or a legitimate last row that ends in an empty column
   is discarded on every authorised truncation.
2. The whole-file entry point and the in-memory parsers agree on the same bytes
   (LF, CRLF, headerless, JSON array, JSONL), and a *complete* JSON array
   through ``tolerate_truncated=True`` keeps every element.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
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
    """The extract-method waiver's load-bearing claim: one parser, two entry points.

    RETARGETED ON HARVEST. This suite was written against a branch where the whole-file
    entry points were still named ``_load_csv`` / ``_load_json`` and merely delegated to
    the text parsers. The extraction went one step further before it landed: those two
    wrappers no longer exist, and ``generate()`` calls ``_parse_csv_text`` /
    ``_parse_json_text`` directly (``generators/csv_import/generator.py``, the
    ``not over_cap`` branch). Asserting on the old names would be red forever, and
    deleting the class would drop a live property, so it is aimed at the seam that
    actually remains:

        read bytes -> strict UTF-8 decode -> parse        (what ``generate()`` does)
                    ==
        parse the same text directly                      (what the capped path does)

    The comparison text is taken from ``read_bytes().decode()``, never ``read_text()``:
    universal-newline translation would silently repair a CRLF divergence, and CRLF
    fidelity is one of the things this class exists to pin.
    """

    @staticmethod
    def _pairs(X, y) -> list:
        """``(feature-tuple, label)`` pairs, order-normalised.

        ``generate()`` SHUFFLES on its way through the split, so ``X_full`` comes back
        permuted relative to source order while the in-memory parser does not. Comparing
        the arrays directly would fail on a correct parse. The pairing is kept -- sorting
        row-wise on each column independently would compare a matrix that never existed.
        """
        return sorted((tuple(row), str(label)) for row, label in zip(X.tolist(), y.ravel().tolist()))

    @classmethod
    def _generated(cls, directory: Path, name: str, params: CsvImportParams) -> list:
        settings = MagicMock()
        settings.import_dir = str(directory)
        settings.csv_import_max_bytes = 10_000_000
        settings.csv_import_allow_truncation = False
        with patch("juniper_data.generators.csv_import.generator.get_settings", return_value=settings):
            result = CsvImportGenerator.generate(params.model_copy(update={"file_path": name}))
        return cls._pairs(result["X_full"], result["y_full"])

    @classmethod
    def _from_text(cls, path: Path, params: CsvImportParams, *, csv_format: bool) -> list:
        text = path.read_bytes().decode("utf-8")
        rows = (
            CsvImportGenerator._parse_csv_text(text, params, drop_trailing_partial=False)
            if csv_format
            else CsvImportGenerator._parse_json_text(text, tolerate_truncated=False)
        )
        return cls._pairs(*CsvImportGenerator._convert_to_arrays(rows, params))

    def test_generate_matches_parse_csv_text_for_lf(self, tmp_path: Path) -> None:
        path = tmp_path / "rows.csv"
        path.write_bytes(_CSV_WITH_HEADER.encode("utf-8"))
        params = _params()

        generated = self._generated(tmp_path, "rows.csv", params)
        from_text = self._from_text(path, params, csv_format=True)

        assert generated == from_text
        assert len(generated) == 2

    def test_generate_matches_parse_csv_text_for_crlf(self, tmp_path: Path) -> None:
        """``newline=""`` in the parser is what keeps CRLF from drifting.

        Without it ``csv`` sees a stray ``\\r`` in the last field of every row, so the
        two sides disagree on the VALUES while still agreeing on the row count -- which
        is why this compares the pairs rather than ``len``.
        """
        path = tmp_path / "rows.csv"
        path.write_bytes(_CSV_WITH_HEADER.replace("\n", "\r\n").encode("utf-8"))
        params = _params()

        generated = self._generated(tmp_path, "rows.csv", params)
        from_text = self._from_text(path, params, csv_format=True)

        assert generated == from_text
        # Assert on the PARSED FIELD, not on the encoded label array: `_convert_to_arrays`
        # label-encodes, and "A\r"/"B\r" are still two distinct classes that encode to the
        # same 0.0/1.0 as "A"/"B". The stray carriage return would be invisible there --
        # a check whose unit is coarser than the thing it is meant to see.
        rows = CsvImportGenerator._parse_csv_text(path.read_bytes().decode("utf-8"), params, drop_trailing_partial=False)
        assert [row["label"] for row in rows] == ["A", "B"]

    def test_generate_matches_parse_csv_text_without_a_header(self, tmp_path: Path) -> None:
        """header=False invents ``col_N`` fieldnames; both entry points must invent the same ones."""
        path = tmp_path / "rows.csv"
        path.write_bytes(_CSV_HEADERLESS.encode("utf-8"))
        params = _params(header=False)

        generated = self._generated(tmp_path, "rows.csv", params)
        from_text = self._from_text(path, params, csv_format=True)

        assert generated == from_text
        assert list(CsvImportGenerator._parse_csv_text(_CSV_HEADERLESS, params, drop_trailing_partial=False)[0].keys()) == ["col_0", "col_1", "col_2"]

    def test_generate_matches_parse_json_text_for_an_array(self, tmp_path: Path) -> None:
        path = tmp_path / "rows.json"
        path.write_bytes(_JSON_ARRAY.encode("utf-8"))
        params = _params()

        generated = self._generated(tmp_path, "rows.json", params)
        from_text = self._from_text(path, params, csv_format=False)

        assert generated == from_text
        assert len(generated) == 2

    def test_generate_matches_parse_json_text_for_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "rows.jsonl"
        path.write_bytes(_JSONL.encode("utf-8"))
        params = _params()

        generated = self._generated(tmp_path, "rows.jsonl", params)
        from_text = self._from_text(path, params, csv_format=False)

        assert generated == from_text
        assert len(generated) == 2


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
