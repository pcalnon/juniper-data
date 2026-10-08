"""Drive the AGENTS.md date-check shell, not a reimplementation of it.

``.github/workflows/agents-md-touch-up.yml`` used to rewrite ``**Last Updated**:``
and push the commit back. That push was unsigned, so branch protection rejected
it, and the ``[skip ci]`` tag left the PR with no required check. The job now
only verifies. These tests extract that ``run:`` step and execute it.

The clock is a ``date`` stub, so nothing here depends on the runner's UTC date.
The child environment is minimal (``PATH``, ``HOME``, ``LANG``, and ``BASE_SHA``
when the case supplies one). Scratch git repos live in a temporary directory;
the test source stays in the tree.
"""

from __future__ import annotations

import os
import subprocess  # nosec B404 - runs the workflow's own extracted shell with fixed argv
import tempfile
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[3]
WORKFLOW = REPO / ".github" / "workflows" / "agents-md-touch-up.yml"
STEP_NAME = "Verify AGENTS.md `**Last Updated**:` was bumped in this PR"
TODAY = "2026-10-06"
REAL_DATE = "/usr/bin/date"

DATE_STUB = """#!/bin/sh
if [ "${{1-}}" = "-u" ] && [ "${{2-}}" = "+%Y-%m-%d" ]; then
  printf '%s\\n' '{today}'
  exit 0
fi
exec {real_date} "$@"
"""


def _workflow() -> dict:
    data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    # PyYAML (YAML 1.1) reads a bare `on:` key as boolean True.
    data["on"] = data.pop(True, data.get("on"))
    return data


def _script() -> str:
    steps = _workflow()["jobs"]["verify-date"]["steps"]
    matches = [step for step in steps if step.get("name") == STEP_NAME]
    assert len(matches) == 1, f"expected one {STEP_NAME!r} step, found {len(matches)}"
    script = matches[0].get("run")
    assert isinstance(script, str) and script.strip(), "the date check has no run script"
    return script


def _agents(updated: str | None, body: str = "alpha\n") -> str:
    """Header block in the shape the workflow greps, plus a body line."""
    if updated is None:
        return f"# Guide\n\n{body}"
    return f"# Guide\n\n**Last Updated**: {updated}\n\n{body}"


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(  # nosec B603 B607 - fixed git argv inside a temp fixture
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        env=_child_env(),
    )
    if proc.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {proc.stderr}")
    return proc.stdout.strip()


def _init_repo(repo: Path) -> None:
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "commit.gpgsign", "false")


def _commit(repo: Path, text: str, message: str) -> str:
    (repo / "AGENTS.md").write_text(text, encoding="utf-8")
    _git(repo, "add", "AGENTS.md")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _child_env(**overrides: str) -> dict[str, str]:
    """Minimal environment. Copying ``os.environ`` would leak into failure tracebacks."""
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),  # nosec B108 - git needs a HOME; the fixture is the scratch dir
        "LANG": "C",
    }
    env.update(overrides)
    return env


def _run(
    repo: Path,
    *,
    base_sha: str | None,
    today: str = TODAY,
) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as scratch:
        scratch_path = Path(scratch)
        script_path = scratch_path / "verify.sh"
        script_path.write_text(_script(), encoding="utf-8")
        stub_bin = scratch_path / "bin"
        stub_bin.mkdir()
        date_stub = stub_bin / "date"
        date_stub.write_text(DATE_STUB.format(today=today, real_date=REAL_DATE), encoding="utf-8")
        date_stub.chmod(0o755)
        env = _child_env(PATH=str(stub_bin) + os.pathsep + os.environ.get("PATH", "/usr/bin:/bin"))
        if base_sha is not None:
            env["BASE_SHA"] = base_sha
        return subprocess.run(  # nosec B603 B607 - workflow shell, fixed argv, no shell=True
            ["bash", "--noprofile", "--norc", str(script_path)],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )


def _plain_dir(text: str) -> Path:
    """A directory that is not a git repo, so a stray ``git diff`` cannot succeed."""
    path = Path(tempfile.mkdtemp(prefix="agents-md-date-"))
    (path / "AGENTS.md").write_text(text, encoding="utf-8")
    return path


class TestWorkflowContract:
    """The job verifies. It does not commit, and it does not need write access."""

    def test_pull_request_paths_and_read_only_permissions(self) -> None:
        workflow = _workflow()
        assert workflow["on"]["pull_request"]["paths"] == ["AGENTS.md"]
        assert workflow["permissions"]["contents"] == "read"
        script = _script()
        assert 'git diff "${BASE_SHA}...HEAD"' in script
        assert "git commit" not in script
        assert "git push" not in script


class TestDateCheck:
    """Each case is one branch of the extracted shell."""

    def test_missing_field_warns_and_passes(self) -> None:
        repo = _plain_dir("# Guide\n\nno header field\n")
        proc = _run(repo, base_sha=None)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "no '**Last Updated**:' field" in proc.stdout

    @pytest.mark.parametrize(
        "value",
        ["2026-9-23", "2026/10/06", "", "September 23, 2026"],
        ids=["unpadded", "slashes", "empty", "prose"],
    )
    def test_non_iso_value_fails(self, value: str) -> None:
        repo = _plain_dir(_agents(value))
        proc = _run(repo, base_sha=None)
        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert "not a YYYY-MM-DD date" in proc.stdout
        # ``tr`` deletes whitespace before the pattern match, so the error names that form.
        shown = "''" if value == "" else value.replace(" ", "")
        assert shown in proc.stdout

    def test_future_date_fails_even_when_the_line_was_added(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            _init_repo(repo)
            base = _commit(repo, _agents("2020-01-01"), "base")
            _commit(repo, _agents("2026-10-07"), "bump into the future")
            proc = _run(repo, base_sha=base)
        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert "is in the future" in proc.stdout
        assert "2026-10-07" in proc.stdout
        assert "was bumped" not in proc.stdout

    def test_today_passes_when_the_line_is_absent_from_the_diff(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            _init_repo(repo)
            base = _commit(repo, _agents(TODAY, "alpha\n"), "header already today")
            _commit(repo, _agents(TODAY, "beta\n"), "body only")
            proc = _run(repo, base_sha=base)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "already today's UTC date" in proc.stdout
        assert "was bumped" not in proc.stdout

    def test_today_passes_without_a_resolvable_base(self) -> None:
        repo = _plain_dir(_agents(TODAY))
        proc = _run(repo, base_sha=None)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "already today's UTC date" in proc.stdout

    def test_a_bump_to_a_past_date_passes(self) -> None:
        """A PR opened Monday does not have to equal the Thursday CI re-run's date."""
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            _init_repo(repo)
            base = _commit(repo, _agents("2020-01-01"), "old header")
            _commit(repo, _agents("2026-09-23"), "bump, not to today")
            proc = _run(repo, base_sha=base)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "was bumped in this PR" in proc.stdout
        assert "2026-09-23" in proc.stdout

    def test_unchanged_header_fails_even_if_prose_mentions_the_marker(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            _init_repo(repo)
            base = _commit(repo, _agents("2026-09-23", "alpha\n"), "base")
            _commit(
                repo,
                _agents("2026-09-23", "See the **Last Updated**: field before editing.\n"),
                "prose only",
            )
            proc = _run(repo, base_sha=base)
        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert "does not bump" in proc.stdout
        assert f"**Last Updated**: {TODAY}" in proc.stdout

    def test_three_dot_ignores_a_header_the_base_tip_moved(self) -> None:
        """``base.sha`` is the base tip, not the merge base.

        Main bumped the header after the PR branched. The PR changed the body
        and kept its old header. A two-dot diff against that tip shows the old
        header as an added line and would pass. The three-dot diff is the PR's
        own commits, which never touched the header, so the check fails.
        """
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            _init_repo(repo)
            _commit(repo, _agents("2020-01-01", "alpha\n"), "fork point")
            _git(repo, "checkout", "-b", "pr")
            _git(repo, "checkout", "main")
            base_tip = _commit(repo, _agents("2026-09-01", "alpha\n"), "main bumps the header")
            _git(repo, "checkout", "pr")
            _commit(repo, _agents("2020-01-01", "beta\n"), "pr changes the body only")
            proc = _run(repo, base_sha=base_tip)
        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert "does not bump" in proc.stdout
        assert "was bumped" not in proc.stdout

    def test_surrounding_whitespace_still_parses_as_today(self) -> None:
        repo = _plain_dir("# Guide\n\n**Last Updated**:   2026-10-06   \n\nbody\n")
        proc = _run(repo, base_sha=None)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "already today's UTC date" in proc.stdout

    def test_only_the_first_header_line_is_the_value(self) -> None:
        text = "# Guide\n\n**Last Updated**: 2026-10-06\n\n**Last Updated**: not-a-date\n\nbody\n"
        repo = _plain_dir(text)
        proc = _run(repo, base_sha=None)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "already today's UTC date" in proc.stdout
