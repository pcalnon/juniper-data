"""Pin the pre-push coverage summary in ``scripts/check_module_coverage.py``.

The per-module 85% line only warns. Aggregate coverage below 80% is what fails
the script. juniper-data#502: a clean aggregate used to print
``PASS: All N modules >= 85%`` even when modules had just been warned, so a
pre-push run read as a full pass. A pass with warnings now says how many
modules are under the advisory line and that the line is not enforced.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "check_module_coverage.py"
_spec = importlib.util.spec_from_file_location("check_module_coverage_under_test", _SCRIPT)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def _report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, aggregate: float, files: dict[str, float]) -> None:
    path = tmp_path / "coverage.json"
    path.write_text(
        json.dumps(
            {
                "totals": {"percent_covered": aggregate},
                "files": {name: {"summary": {"percent_covered": pct}} for name, pct in files.items()},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(_mod, "COVERAGE_JSON", path)
    monkeypatch.setattr(_mod, "AGGREGATE_FAIL_UNDER", 80.0)
    monkeypatch.setattr(_mod, "MODULE_FAIL_UNDER", 85.0)


def test_modules_under_the_advisory_line_do_not_read_as_a_full_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """The bug: exit 0 and ``PASS: All`` after printing the warning."""
    _report(tmp_path, monkeypatch, 90.0, {"juniper_data/low.py": 84.99, "juniper_data/ok.py": 85.0})
    assert _mod.check_coverage() == 0
    out = capsys.readouterr().out
    assert "WARN: 1 module(s) below 85.0% coverage:" in out
    assert "juniper_data/low.py: 84.99%" in out
    assert "PASS: All" not in out
    assert "PASS: aggregate >= 80.0%; 1 of 2 modules below 85.0% (warned above, not enforced)" in out


def test_every_module_at_the_threshold_is_a_full_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """80.0 aggregate and 85.0 per module are the passing sides of both ``<`` checks."""
    _report(tmp_path, monkeypatch, 80.0, {"juniper_data/a.py": 85.0})
    assert _mod.check_coverage() == 0
    out = capsys.readouterr().out
    assert "WARN" not in out
    assert "FAIL" not in out
    assert "PASS: All 1 modules >= 85.0%, aggregate >= 80.0%" in out


def test_an_aggregate_miss_fails_and_prints_no_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """A warned module must not add a PASS line onto a failed aggregate."""
    _report(tmp_path, monkeypatch, 79.99, {"juniper_data/low.py": 10.0, "juniper_data/ok.py": 99.0})
    assert _mod.check_coverage() == 1
    out = capsys.readouterr().out
    assert "FAIL: Aggregate coverage 79.99% < 80.0%" in out
    assert "WARN: 1 module(s) below 85.0% coverage:" in out
    assert "PASS" not in out


def test_a_test_file_in_the_metrics_is_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _report(
        tmp_path,
        monkeypatch,
        100.0,
        {"juniper_data/tests/unit/test_x.py": 100.0, "juniper_data/a.py": 90.0},
    )
    with pytest.raises(SystemExit) as exc:
        _mod.check_coverage()
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "1 test file(s) found in coverage metrics:" in out
    assert "juniper_data/tests/unit/test_x.py" in out
    assert "PASS" not in out


def test_a_missing_report_is_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(_mod, "COVERAGE_JSON", tmp_path / "missing.json")
    with pytest.raises(SystemExit) as exc:
        _mod.check_coverage()
    assert exc.value.code == 1
    assert "coverage.json not found" in capsys.readouterr().out
