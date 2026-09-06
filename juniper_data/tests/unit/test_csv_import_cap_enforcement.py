#!/usr/bin/env python
"""Read-side enforcement of the csv_import byte cap (APD-DATA-018 / #326 5a8ae63).

#326 already pins that a request cannot *raise* the deployment cap and that a
lying ``stat`` still *refuses* when truncation is off. #328 owns exact-at-cap,
OR opt-in, UTF-8 cuts, persist, and env knobs. #330 owns minified JSON, the
2-column quote case, and cache identity. #331 owns shared-parser identity.

This file does not repeat those. It pins the remaining two holes the 5a8ae63
fix introduced, which a plausible revert would reopen without those suites
going red:

1. ``_read_capped_bytes`` is the ingest bound. An unbounded ``read()`` that
   then compares ``len`` still refuses, so every generate-level refusal test
   stays green while the DoS bound is gone.
2. A FIFO (``st_size == 0``) with *opt-in* must still truncate and must
   annotate ``bytes_total`` from what was actually observed, not the lying
   zero -- otherwise a persisted descriptor looks like an empty complete
   source.
3. A generated client that serialises schema defaults sends
   ``max_bytes=134217728`` on every request (``model_fields_set``). That
   value must not raise a *tighter* operator ceiling. The existing raise
   arm uses ``10**10``, which a ``if requested > 10**9`` clamp would still
   catch; the schema-default is the footgun the commit names.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Load-bearing side effect, NOT an unused import. ``juniper_data.generators.csv_import``
# cannot be imported on its own -- it hits a circular import through
# ``api.routes.generators`` (juniper-data#316). Importing the routes module first
# closes the cycle so this file is runnable without relying on collection order.
#
# Written as an explicit ``import_module`` call rather than a bare import marked unused
# because a ruff unused-import suppression satisfied ruff but not CodeQL.
importlib.import_module("juniper_data.api.routes.generators")

from juniper_data.core.limits import (  # noqa: E402
    CSV_IMPORT_DEFAULT_MAX_BYTES,
    TRUNCATION_META_KEY,
    UNIT_BYTES,
    InputTooLargeError,
)
from juniper_data.generators.csv_import import CsvImportGenerator, CsvImportParams  # noqa: E402
from juniper_data.tests.partitions import whole  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.generators]

# Small enough that the fixtures below overflow it by a wide margin, so a test
# never depends on a source landing near the boundary by luck.
TINY_CAP_BYTES = 120


@pytest.fixture
def bounded_import_dir(tmp_path: Path):
    """Temporary import dir with a tiny deployment cap and truncation off."""
    mock_settings = MagicMock()
    mock_settings.import_dir = str(tmp_path)
    mock_settings.csv_import_max_bytes = TINY_CAP_BYTES
    mock_settings.csv_import_allow_truncation = False
    with patch("juniper_data.generators.csv_import.generator.get_settings", return_value=mock_settings):
        yield tmp_path, mock_settings


def _wide_csv(directory: Path, name: str = "wide.csv", rows: int = 40) -> Path:
    path = directory / name
    lines = ["feature1,feature2,label"]
    lines.extend(f"{i}.0,{i + 1}.0,{'A' if i % 2 else 'B'}" for i in range(rows))
    path.write_text("\n".join(lines) + "\n")
    return path


def _fifo_stat(path: Path):
    """Patch ``Path.stat`` so *path* reports ``st_size == 0`` the way a FIFO does.

    Must wrap the real result rather than building ``{\"st_size\": 0, **attrs}``:
    the real ``st_size`` would overwrite the zero and the FIFO case would never
    be simulated (the M8-survived-because-the-test-was-wrong class).
    """
    real_stat = Path.stat

    class _FakeStat:
        def __init__(self, real) -> None:  # noqa: ANN001
            self._real = real
            self.st_size = 0

        def __getattr__(self, item):  # noqa: ANN001, ANN204
            return getattr(self._real, item)

    def lying_stat(self, *args, **kwargs):  # noqa: ANN001, ANN202
        result = real_stat(self, *args, **kwargs)
        return _FakeStat(result) if self.name == path.name else result

    return patch.object(Path, "stat", lying_stat)


class TestReadCappedBytesIsTheIngestBound:
    """The helper, not a later ``len`` check, is what stops unbounded ingestion."""

    def test_returns_at_most_limit_bytes_from_a_larger_file(self, tmp_path: Path) -> None:
        """Mutation: ``handle.read(limit)`` → ``handle.read()`` returns the whole file.

        Every generate-level refusal test still passes under that mutation,
        because they compare ``len(raw)`` after the read. This is the arm that
        actually bounds peak RSS / wall-clock.
        """
        path = tmp_path / "blob.bin"
        path.write_bytes(b"X" * 1000)

        got = CsvImportGenerator._read_capped_bytes(path, 50)

        assert len(got) == 50
        assert got == b"X" * 50

    def test_returns_the_whole_file_when_smaller_than_limit(self, tmp_path: Path) -> None:
        """An under-cap source must not be trimmed by the reader itself."""
        path = tmp_path / "blob.bin"
        payload = b"hello-world"
        path.write_bytes(payload)

        got = CsvImportGenerator._read_capped_bytes(path, 50)

        assert got == payload


class TestLyingStatWithOptInAnnotatesObservedSize:
    """The FIFO / grew-after-stat path with truncation *authorised*."""

    def test_under_reported_stat_does_not_persist_bytes_total_zero(self, bounded_import_dir) -> None:
        """``bytes_total = max(stat, observed)`` is load-bearing on the opt-in path.

        The existing lying-stat arm only checks refusal. With opt-in, a FIFO
        still truncates, and the annotation must not record a zero requested
        size (a complete-looking empty source). The observed read is ``cap + 1``,
        which is a true lower bound either way.

        Mutation: ``max(stat_bytes, len(raw))`` → ``stat_bytes`` stores 0.

        Key names: this suite was harvested from a branch predating the unified
        descriptor. ``build_truncation_meta`` now emits ``cap`` / ``requested`` /
        ``imported`` with a ``unit`` that says how to read them, so one consumer
        shape serves both generators. The PROPERTY is unchanged; only the names
        moved, so the assertions are retargeted rather than dropped.
        """
        directory, _ = bounded_import_dir
        path = _wide_csv(directory)

        with _fifo_stat(path):
            result = CsvImportGenerator.generate(CsvImportParams(file_path="wide.csv", allow_truncation=True))

        annotation = result[TRUNCATION_META_KEY]
        assert annotation["truncated"] is True
        assert annotation["unit"] == UNIT_BYTES
        assert annotation["requested"] >= TINY_CAP_BYTES + 1
        assert annotation["requested"] != 0
        assert annotation["cap"] == TINY_CAP_BYTES
        assert 0 < annotation["records_imported"] < 40
        assert whole(result, "X").shape[0] == annotation["records_imported"]


class TestSchemaDefaultMaxBytesCannotRaiseTheCeiling:
    """Generated clients that serialise Field defaults must not undo a tighter cap."""

    def test_explicit_schema_default_is_clamped_to_the_deployment_ceiling(self, bounded_import_dir) -> None:
        """``max_bytes=134217728`` in ``model_fields_set`` is the OpenAPI-client case.

        The 5a8ae63 raise arm uses ``10_000_000_000``. That does not pin the
        subtler half: a client that sends the *schema default* looks like an
        explicit request and used to override a lower operator ceiling.
        """
        directory, _ = bounded_import_dir
        _wide_csv(directory)
        params = CsvImportParams(file_path="wide.csv", max_bytes=CSV_IMPORT_DEFAULT_MAX_BYTES)

        assert "max_bytes" in params.model_fields_set
        assert params.max_bytes == CSV_IMPORT_DEFAULT_MAX_BYTES
        assert params.max_bytes > TINY_CAP_BYTES

        with pytest.raises(InputTooLargeError) as excinfo:
            CsvImportGenerator.generate(params)

        # `.cap` / `.actual` / `.unit` replaced `.cap_bytes` / `.bytes_total` when the
        # error was unified across generators; the assertion is the same one.
        assert excinfo.value.unit == UNIT_BYTES
        assert excinfo.value.cap == TINY_CAP_BYTES
        assert excinfo.value.actual > TINY_CAP_BYTES
