"""Rehearse ``util/run_coverage.bash``, the body of ``make coverage``.

CI's unit lane has two coverage gates: the aggregate ``coverage report --fail-under``
and ``juniper-coverage-gap-map --enforce`` (90% per file, 95% pooled per sub-module).
Until juniper-data#503 the local script ran only the aggregate gate and then said so,
so a file under 90% failed CI while ``make coverage`` stayed green. The script now
refuses to start when ``juniper-coverage-gap-map`` is absent — the tool is not a
project dependency — and otherwise runs the advisory module listing before the
per-file gate, so that listing still prints when the gate fails.

The tests copy the script into a temp repo and stub ``python`` and the gap-map
binary. A copy of the command lines would go stale the moment the script was edited.
"""

from __future__ import annotations

import os
import shlex
import subprocess  # nosec B404 - runs the repo's own coverage script (fixed argv)
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "util" / "run_coverage.bash"
_WORKFLOW = _REPO / ".github" / "workflows" / "ci.yml"
_PIN = "juniper-ci-tools>=0.9.0,<0.10.0"
_GAPMAP = "juniper-coverage-gap-map --coverage-json reports/coverage.json --enforce"
_PYTEST = "-m pytest -m unit and not slow juniper_data/tests/unit --cov=juniper_data --cov-report=term-missing --cov-report=json:reports/coverage.json"


def _install(
    tmp_path: Path,
    *,
    with_gapmap: bool,
    pytest_status: int = 0,
    report_status: int = 0,
    module_status: int = 0,
    gapmap_status: int = 0,
) -> Path:
    log = tmp_path / "calls.log"
    bindir = tmp_path / "bin"
    bindir.mkdir()
    quoted_log = shlex.quote(str(log))
    python = bindir / "python"
    python.write_text(
        "\n".join(
            [
                "#!/bin/sh",
                f"printf '%s\\n' \"$*\" >> {quoted_log}",
                'case "$1 $2" in',
                f'  "-m pytest") exit {int(pytest_status)} ;;',
                f'  "-m coverage") exit {int(report_status)} ;;',
                "esac",
                'case "$*" in',
                f"  *check_module_coverage.py*) exit {int(module_status)} ;;",
                "esac",
                'echo "unexpected python argv: $*" >&2',
                "exit 99",
                "",
            ]
        ),
        encoding="utf-8",
    )
    python.chmod(0o755)
    if with_gapmap:
        gapmap = bindir / "juniper-coverage-gap-map"
        gapmap.write_text(
            "\n".join(
                [
                    "#!/bin/sh",
                    f"printf '%s\\n' \"gapmap $*\" >> {quoted_log}",
                    f"exit {int(gapmap_status)}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        gapmap.chmod(0o755)
    util = tmp_path / "util"
    util.mkdir()
    script = util / "run_coverage.bash"
    script.write_bytes(_SCRIPT.read_bytes())
    script.chmod(0o755)
    return script


def _run(
    tmp_path: Path,
    *,
    with_gapmap: bool,
    fail_under: str | None = None,
    pytest_status: int = 0,
    report_status: int = 0,
    module_status: int = 0,
    gapmap_status: int = 0,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    script = _install(
        tmp_path,
        with_gapmap=with_gapmap,
        pytest_status=pytest_status,
        report_status=report_status,
        module_status=module_status,
        gapmap_status=gapmap_status,
    )
    env = {
        "PATH": f"{tmp_path / 'bin'}{os.pathsep}/usr/bin:/bin",
        "HOME": str(tmp_path),
        "LANG": "C",
    }
    if fail_under is not None:
        env["COVERAGE_FAIL_UNDER"] = fail_under
    proc = subprocess.run(  # nosec B603 B607 - coverage script, fixed argv, stubs written above
        ["bash", str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    log = tmp_path / "calls.log"
    lines = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return proc, lines


def test_a_missing_per_file_tool_exits_before_the_tests(tmp_path: Path) -> None:
    """The full unit lane must not run when the run cannot finish CI's second gate."""
    proc, lines = _run(tmp_path, with_gapmap=False)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert lines == []
    assert not (tmp_path / "reports").exists()
    assert proc.stdout == ""
    assert "juniper-coverage-gap-map not found" in proc.stderr
    assert _PIN in proc.stderr


def test_both_gates_run_after_the_unit_lane(tmp_path: Path) -> None:
    proc, lines = _run(tmp_path, with_gapmap=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert lines == [
        _PYTEST,
        "-m coverage report --fail-under=80",
        "scripts/check_module_coverage.py",
        f"gapmap {_GAPMAP.removeprefix('juniper-coverage-gap-map ')}",
    ]


def test_the_aggregate_threshold_comes_from_the_environment(tmp_path: Path) -> None:
    proc, lines = _run(tmp_path, with_gapmap=True, fail_under="90")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert lines[1] == "-m coverage report --fail-under=90"


@pytest.mark.parametrize(
    ("pytest_status", "report_status", "module_status", "calls"),
    [
        (1, 0, 0, 1),
        (0, 1, 0, 2),
        (0, 0, 1, 3),
    ],
)
def test_an_earlier_failure_does_not_reach_the_per_file_gate(tmp_path: Path, pytest_status: int, report_status: int, module_status: int, calls: int) -> None:
    """``set -e``: a red unit lane or aggregate gate must not be followed by a green per-file line."""
    proc, lines = _run(
        tmp_path,
        with_gapmap=True,
        pytest_status=pytest_status,
        report_status=report_status,
        module_status=module_status,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert len(lines) == calls
    assert not any(line.startswith("gapmap ") for line in lines)


def test_a_failing_per_file_gate_fails_the_run_after_the_advisory_listing(tmp_path: Path) -> None:
    """The module listing is not CI's gate. It runs first so a failing ``--enforce`` still shows it."""
    proc, lines = _run(tmp_path, with_gapmap=True, gapmap_status=1)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "check_module_coverage.py" in lines[-2]
    assert lines[-1] == f"gapmap {_GAPMAP.removeprefix('juniper-coverage-gap-map ')}"


def test_the_tool_pin_and_the_json_the_gate_reads_match_ci() -> None:
    script = _SCRIPT.read_text(encoding="utf-8")
    workflow = _WORKFLOW.read_text(encoding="utf-8")
    assert _PIN in script
    assert f'pip install "{_PIN}"' in workflow
    assert _GAPMAP in script
    assert _GAPMAP in workflow
    assert "--cov-report=json:reports/coverage.json" in script
    assert "--cov-report=json:reports/coverage.json" in workflow


def test_make_coverage_runs_this_script_and_nothing_else() -> None:
    text = (_REPO / "Makefile").read_text(encoding="utf-8")
    recipes: list[str] = []
    capture = False
    for line in text.splitlines():
        if line.startswith("coverage:"):
            capture = True
            continue
        if not capture:
            continue
        if line.startswith("\t"):
            recipes.append(line)
        elif line.strip() and not line.startswith("#"):
            break
    assert recipes == ["\t@bash util/run_coverage.bash"]
