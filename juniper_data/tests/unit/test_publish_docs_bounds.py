"""Pin the publish docs' numbers to ``publish.yml``: the bounds the docs state are the bounds the steps run.

``docs/ci_cd/CICD_MANUAL.md``, ``docs/ci_cd/CICD_REFERENCE.md``, ``docs/REFERENCE.md`` and
``docs/DEVELOPER_CHEATSHEET.md`` give an operator the numbers that decide whether a failed check was index lag: how
often and how far apart the JSON checks poll, how long each request may take, how many times the install check tries,
how far apart, and how long it waits first. Each is read here from the workflow and required in the docs, so a change
to either side alone fails. A module of its own, as juniper-cascor#777's ``src/tests/unit/test_publish_docs_bounds.py``
is (this is its port): ``test_publish_one_artifact.py`` reads the workflows alone.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

# CI collects ``-m "unit and not slow"`` with ``--strict-markers`` and does not auto-mark, so an unmarked test is
# collected and then silently DESELECTED. Module-level, so it covers tests added later.
pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[3]
WORKFLOW = REPO / ".github" / "workflows" / "publish.yml"
DOCS = REPO / "docs"
#: The JSON check every copy of the template runs; its first copy, in the TestPyPI job, is read here.
INDEX_CHECK = "Verify TestPyPI serves exactly these files"
INSTALL = "verify-testpypi-install"
INSTALL_STEP = "Verify TestPyPI install"
MANUAL = "ci_cd/CICD_MANUAL.md"
CICD_REFERENCE = "ci_cd/CICD_REFERENCE.md"
REFERENCE = "REFERENCE.md"
CHEATSHEET = "DEVELOPER_CHEATSHEET.md"


def _jobs() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]


def _run(job: dict, name: str) -> str:
    found = [s for s in job.get("steps") or [] if s.get("name") == name]
    assert len(found) == 1, f"expected exactly one step named {name!r}"
    return str(found[0]["run"])


def _doc(name: str) -> str:
    """The doc's text with each run of whitespace made one space: a rewrapped paragraph states the same bounds."""
    return " ".join((DOCS / name).read_text(encoding="utf-8").split())


def _retry() -> tuple[int, list[int], list[int]]:
    """The install check's (tries, the waits inside its loop, the waits before it), read from its commands."""
    run = _run(_jobs()[INSTALL], INSTALL_STEP)
    commands = [line.strip() for line in run.splitlines() if line.strip() and not line.strip().startswith("#")]
    loops = [i for i, c in enumerate(commands) if re.fullmatch(r"for attempt in [0-9 ]+; do", c)]
    assert len(loops) == 1, "expected one retry loop"
    start, end = loops[0], commands.index("done", loops[0])
    tries = len(commands[start].split()[3:-1])
    waits = [int(c.split()[1]) for c in commands[start:end] if re.fullmatch(r"sleep [0-9]+", c)]
    before = [int(c.split()[1]) for c in commands[:start] if re.fullmatch(r"sleep [0-9]+", c)]
    return tries, waits, before


class TestPublishDocsStateTheBounds:
    def test_the_json_checks_bounds(self):
        run = _run(_jobs()["testpypi"], INDEX_CHECK)
        attempts_match = re.search(r'"HASH_CHECK_ATTEMPTS", "(\d+)"', run)
        interval_match = re.search(r'"HASH_CHECK_INTERVAL", "(\d+)"', run)
        timeout_match = re.search(r"urllib\.request\.urlopen\(request, timeout=(\d+)\)", run)
        assert attempts_match and interval_match and timeout_match
        attempts, interval, timeout = int(attempts_match.group(1)), int(interval_match.group(1)), int(timeout_match.group(1))
        manual = _doc(MANUAL)
        # The poll, and the worst case: every sleep, and every request running to its timeout.
        assert f"polled up to {attempts} times {interval} seconds apart" in manual
        assert f"A JSON check that gives up after {attempts} attempts is index lag ({attempts - 1} sleeps of {interval} seconds, plus up to {attempts} request timeouts of {timeout} seconds)" in manual
        assert f"PyPI's JSON did not list the files within its {attempts} attempts" in manual
        assert f"A JSON check that gives up after {attempts} attempts is index lag" in _doc(CHEATSHEET)

    def test_the_install_checks_retry(self):
        tries, waits, before = _retry()
        # One loop: its tries, one wait between them, and one wait before the first.
        assert len(waits) == 1
        assert len(before) == 1
        wait, first = waits[0], before[0]
        manual = _doc(MANUAL)
        assert f"Wait {first} seconds for index sync" in manual
        assert f"up to {tries} tries {wait} seconds apart" in manual
        assert f"it sleeps {first} seconds, then tries the install up to {tries} times" in manual
        assert f"up to {tries} tries, {wait} s apart" in _doc(CICD_REFERENCE)
        reference = _doc(REFERENCE)
        assert f"sleep {first} for attempt in {' '.join(str(n) for n in range(1, tries + 1))}; do" in reference
        assert f"fi sleep {wait} done" in reference
        # The error the check stops with after its last try, as the step prints it and as the docs quote it.
        error = f"is not installable from TestPyPI after {tries} tries (index lag?)"
        assert error in _run(_jobs()[INSTALL], INSTALL_STEP)
        for name in (MANUAL, REFERENCE):
            assert error in _doc(name), name
