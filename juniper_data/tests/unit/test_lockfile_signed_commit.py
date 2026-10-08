"""Pin the lockfile regen's signed commit and the freshness check's pin comparison.

A local ``git commit`` on the runner is unsigned, and this repo rejects unsigned
history, so ``lockfile-update.yml`` must land the regen through
``createCommitOnBranch`` with ``expectedHeadOid``. A quiet diff must not call
the API. A GraphQL error, or a failed ``gh``, must not be reported as a signed
commit. Dependabot with no PAT is a green skip; every other actor without it
fails closed. The freshness check compares resolved pins under ``--constraint``
and must not treat a comment, a blank line, or a reordered header as drift.

These tests extract the workflow's own shell. ``uv`` and ``gh`` are PATH stubs.
Nothing is published and no token is a credential. Both steps hardcode scratch
files under ``/tmp/``; each test rewrites that prefix to its own directory, so
two runs never share a file and none is left behind.
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess  # nosec B404 - runs the workflow's own extracted shell hermetically (fixed argv)
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[3]
UPDATE_WORKFLOW = REPO / ".github" / "workflows" / "lockfile-update.yml"
CI_WORKFLOW = REPO / ".github" / "workflows" / "ci.yml"
ACTOR_EXPR = "${{ github.actor }}"
PAT_PRESENCE = "${{ secrets.CROSS_REPO_DISPATCH_TOKEN != '' }}"
PAT_SECRET = "${{ secrets.CROSS_REPO_DISPATCH_TOKEN }}"  # nosec B105 - workflow expression, not a credential
PROCEED_IF = "steps.gate.outputs.proceed == 'true'"
EXTRAS = ["api", "observability", "mnist", "equities"]
HEADLINE = "[dependabot skip] Update requirements.lock"
TOKEN_SENTINEL = "pat-sentinel-not-a-credential"
RUNNER_TMP = "/tmp/"  # nosec B108 - the prefix both steps hardcode; each test rewrites it to its own directory


def _relocated(script: str, scratch: Path) -> str:
    """The step with its ``/tmp/`` scratch files moved into ``scratch``."""
    assert RUNNER_TMP in script, "the step no longer writes under /tmp/; drop the relocation"
    return script.replace(RUNNER_TMP, f"{scratch}/")


def _base64_has_gnu_w() -> bool:
    """Whether this runner's ``base64`` takes GNU ``-w0``, which the commit step uses for the payload."""
    try:
        proc = subprocess.run(["base64", "-w0"], input="x", capture_output=True, text=True, check=False, timeout=10)  # nosec B603 B607 - capability probe, fixed argv
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0 and proc.stdout == "eA=="


# The commit step runs on ubuntu-latest and encodes the lockfile with GNU ``base64 -w0``. BSD/macOS
# base64 has no ``-w``, so on CI's required macOS unit-test leg the payload is empty: that measures
# the runner's userland, not the workflow (it is how this suite's first CI run failed there).
# Never skipped on Linux: there a failing probe is a real problem, so the case runs and says so.
needs_gnu_base64 = pytest.mark.skipif(not sys.platform.startswith("linux") and not _base64_has_gnu_w(), reason="the commit step runs on ubuntu-latest and encodes the lockfile with GNU `base64 -w0`; this runner's base64 has no -w")


@pytest.fixture
def scratch(tmp_path: Path) -> Path:
    """Stands in for the runner's ``/tmp/``."""
    path = tmp_path / "runner-tmp"
    path.mkdir()
    return path


def _child_env(**overrides: str) -> dict[str, str]:
    """Minimal child environment. The real process environment is not copied."""
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),  # nosec B108 - git needs a HOME; fallback only
        "LANG": "C",
    }
    env.update(overrides)
    return env


def _workflow(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    # PyYAML (YAML 1.1) reads the bare `on:` key as boolean True.
    data["on"] = data.pop(True, data.get("on"))
    return data


def _update() -> dict[str, Any]:
    return _workflow(UPDATE_WORKFLOW)


def _update_step(name: str) -> dict[str, Any]:
    steps = _update()["jobs"]["update-lockfile"]["steps"]
    matches = [step for step in steps if step.get("name") == name]
    assert len(matches) == 1, f"expected one step named {name!r}, found {len(matches)}"
    return matches[0]


def _freshness_step() -> dict[str, Any]:
    job = _workflow(CI_WORKFLOW)["jobs"]["lockfile-check"]
    matches = [step for step in job["steps"] if step.get("name") == "Check lockfile freshness"]
    assert len(matches) == 1
    return matches[0]


def _run_script(script: str, *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    script_path = cwd / ".lockfile-step.sh"
    script_path.write_text(script, encoding="utf-8")
    try:
        return subprocess.run(  # nosec B603 B607 - workflow shell, fixed argv
            ["bash", str(script_path)],
            cwd=cwd,
            capture_output=True,
            text=True,
            env=env,
            check=False,
            timeout=30,
        )
    finally:
        script_path.unlink(missing_ok=True)


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(  # nosec B603 B607 - fixed git argv in a temp fixture
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        env=_child_env(GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t"),
    )
    if proc.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {proc.stderr}")
    return proc.stdout.strip()


def _init_repo(root: Path, lock_text: str) -> Path:
    repo = root / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "dependabot/pip/numpy-1.2.3")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "requirements.lock").write_text(lock_text, encoding="utf-8")
    _git(repo, "add", "requirements.lock")
    _git(repo, "commit", "-m", "lock")
    return repo


def _gate_script() -> str:
    raw = _update_step("Gate on CROSS_REPO_DISPATCH_TOKEN availability")["run"]
    assert ACTOR_EXPR in raw, "the actor must stay an Actions expression so the shell never sees a literal"
    return raw.replace(ACTOR_EXPR, "$LOCKFILE_ACTOR")


def _run_gate(have_pat: str, actor: str) -> tuple[subprocess.CompletedProcess[str], str]:
    with tempfile.TemporaryDirectory() as td:
        cwd = Path(td)
        output = cwd / "github_output"
        output.write_text("", encoding="utf-8")
        proc = _run_script(
            _gate_script(),
            cwd=cwd,
            env=_child_env(HAVE_PAT=have_pat, LOCKFILE_ACTOR=actor, GITHUB_OUTPUT=str(output)),
        )
        return proc, output.read_text(encoding="utf-8")


def _code(run: str) -> str:
    """Drop comment lines so a mention in a comment is not an executed command."""
    return "\n".join(line for line in run.splitlines() if not line.strip().startswith("#"))


def _output_map(text: str) -> dict[str, str]:
    mapped: dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            mapped[key] = value
    return mapped


class TestPatGate:
    """The shell branches on token presence. The token value itself is not in the step."""

    @pytest.mark.parametrize(
        ("have_pat", "actor", "returncode", "proceed"),
        [
            ("true", "dependabot[bot]", 0, "true"),
            ("true", "pcalnon", 0, "true"),
            ("false", "dependabot[bot]", 0, "false"),
            ("", "dependabot[bot]", 0, "false"),
            ("True", "dependabot[bot]", 0, "false"),
            ("false", "pcalnon", 1, None),
            ("", "pcalnon", 1, None),
            ("true ", "pcalnon", 1, None),
            ("false", "Dependabot[bot]", 1, None),
            ("false", "dependabot", 1, None),
            ("false", "dependabot[bot] ", 1, None),
        ],
    )
    def test_only_an_exact_true_proceeds_and_only_dependabot_may_skip(self, have_pat: str, actor: str, returncode: int, proceed: str | None) -> None:
        proc, written = _run_gate(have_pat, actor)
        combined = proc.stdout + proc.stderr
        assert proc.returncode == returncode, combined
        mapped = _output_map(written)
        if proceed is None:
            assert "proceed" not in mapped, written
            assert "secret misconfiguration" in combined
            assert "Dependabot secret store" not in combined
        else:
            assert mapped.get("proceed") == proceed, written
            assert "secret misconfiguration" not in combined
        if proceed == "false":
            assert "Dependabot secret store" in combined
            assert "Lockfile Freshness" in combined
        if proceed == "true":
            assert "Dependabot secret store" not in combined

    def test_the_shell_sees_presence_not_the_secret(self) -> None:
        step = _update_step("Gate on CROSS_REPO_DISPATCH_TOKEN availability")
        assert step["env"]["HAVE_PAT"] == PAT_PRESENCE
        # The notice names the secret so an operator can register it. The shell must not expand the value.
        assert "secrets.CROSS_REPO_DISPATCH_TOKEN" not in step["run"]
        assert step.get("if") is None


class TestSignedCommit:
    """The regen commit is a compare-and-swap through the signing API."""

    def _commit_script(self) -> str:
        return _update_step("Commit updated lockfile (GitHub-signed, via API)")["run"]

    def _run_commit(
        self,
        repo: Path,
        scratch: Path,
        *,
        result_body: str = "",
        gh_rc: int = 0,
        extra_env: dict[str, str] | None = None,
        omit: tuple[str, ...] = (),
    ) -> tuple[subprocess.CompletedProcess[str], str | None]:
        bin_dir = scratch.parent / "bin"
        bin_dir.mkdir()
        argv_file = scratch.parent / "gh-argv.txt"
        stub = bin_dir / "gh"
        stub.write_text(
            '#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$GH_ARGV_FILE"\nif [[ "${GH_RC}" != "0" ]]; then\n  echo "gh failed" >&2\n  exit "$GH_RC"\nfi\nprintf \'%s\\n\' "$GH_RESULT_BODY"\n',
            encoding="utf-8",
        )
        stub.chmod(0o755)
        env = _child_env(
            PATH=str(bin_dir) + os.pathsep + os.environ.get("PATH", "/usr/bin:/bin"),
            GH_TOKEN=TOKEN_SENTINEL,
            GITHUB_REPOSITORY="pcalnon/juniper-data",
            BRANCH="dependabot/pip/numpy-1.2.3",
            GH_ARGV_FILE=str(argv_file),
            GH_RC=str(gh_rc),
            GH_RESULT_BODY=result_body,
        )
        if extra_env:
            env.update(extra_env)
        for key in omit:
            env.pop(key, None)
        proc = _run_script(_relocated(self._commit_script(), scratch), cwd=repo, env=env)
        argv_text = argv_file.read_text(encoding="utf-8") if argv_file.exists() else None
        return proc, argv_text

    def test_a_quiet_diff_does_not_call_the_api(self, tmp_path: Path, scratch: Path) -> None:
        repo = _init_repo(tmp_path, "numpy==1.24.0\n")
        proc, argv_text = self._run_commit(repo, scratch, result_body="should-not-be-read")
        combined = proc.stdout + proc.stderr
        assert proc.returncode == 0, combined
        assert "no commit needed" in combined
        assert "Signed lockfile commit" not in combined
        assert argv_text is None
        assert not (scratch / "lockfile-commit.json").exists()

    @needs_gnu_base64
    def test_a_changed_lockfile_is_the_compare_and_swap_payload(self, tmp_path: Path, scratch: Path) -> None:
        repo = _init_repo(tmp_path, "numpy==1.24.0\n")
        updated = "numpy==1.25.0\n# PINNED-BY-TEST\n"
        (repo / "requirements.lock").write_text(updated, encoding="utf-8")
        head = _git(repo, "rev-parse", "HEAD")
        body = '{"data":{"createCommitOnBranch":{"commit":{"oid":"abc","url":"https://example.test/commit/abc"}}}}'
        proc, argv_text = self._run_commit(repo, scratch, result_body=body)
        combined = proc.stdout + proc.stderr
        assert proc.returncode == 0, combined
        assert "Signed lockfile commit: https://example.test/commit/abc" in combined
        assert TOKEN_SENTINEL not in combined
        assert argv_text is not None
        recorded = argv_text.splitlines()
        assert recorded[:3] == ["api", "graphql", "--input"]
        commit_json = scratch / "lockfile-commit.json"
        assert str(commit_json) in recorded
        assert TOKEN_SENTINEL not in recorded
        payload = json.loads(commit_json.read_text(encoding="utf-8"))
        change = payload["variables"]["input"]
        assert change["expectedHeadOid"] == head
        assert change["message"]["headline"] == HEADLINE
        assert change["branch"] == {"repositoryNameWithOwner": "pcalnon/juniper-data", "branchName": "dependabot/pip/numpy-1.2.3"}
        addition = change["fileChanges"]["additions"]
        assert len(addition) == 1
        assert addition[0]["path"] == "requirements.lock"
        assert base64.b64decode(addition[0]["contents"]) == updated.encode("utf-8")
        assert TOKEN_SENTINEL not in commit_json.read_text(encoding="utf-8")
        assert "createCommitOnBranch" in payload["query"]

    def test_graphql_errors_are_not_a_signed_commit(self, tmp_path: Path, scratch: Path) -> None:
        repo = _init_repo(tmp_path, "numpy==1.24.0\n")
        (repo / "requirements.lock").write_text("numpy==9.9.9\n", encoding="utf-8")
        body = '{"errors":[{"message":"expectedHeadOid mismatch"}]}'
        proc, _argv = self._run_commit(repo, scratch, result_body=body)
        combined = proc.stdout + proc.stderr
        assert proc.returncode == 1, combined
        assert "lockfile NOT committed" in combined
        assert "expectedHeadOid mismatch" in combined
        assert "Signed lockfile commit" not in combined

    def test_a_failing_gh_is_not_a_signed_commit(self, tmp_path: Path, scratch: Path) -> None:
        repo = _init_repo(tmp_path, "numpy==1.24.0\n")
        (repo / "requirements.lock").write_text("numpy==9.9.9\n", encoding="utf-8")
        proc, argv_text = self._run_commit(repo, scratch, gh_rc=7, result_body='{"data":{"createCommitOnBranch":{"commit":{"url":"https://example.test/nope"}}}}')
        combined = proc.stdout + proc.stderr
        assert proc.returncode != 0, combined
        assert argv_text is not None
        assert "api" in argv_text
        assert "Signed lockfile commit" not in combined

    @pytest.mark.parametrize("missing", ["GITHUB_REPOSITORY", "BRANCH"])
    def test_a_missing_target_does_not_call_the_api(self, tmp_path: Path, scratch: Path, missing: str) -> None:
        repo = _init_repo(tmp_path, "numpy==1.24.0\n")
        (repo / "requirements.lock").write_text("numpy==9.9.9\n", encoding="utf-8")
        proc, argv_text = self._run_commit(repo, scratch, omit=(missing,))
        assert proc.returncode != 0, proc.stdout + proc.stderr
        assert argv_text is None
        assert not (scratch / "lockfile-commit.json").exists()


class TestFreshnessPins:
    """The checker fails on pin drift and passes when only the header moved."""

    def _run(self, root: Path, scratch: Path, lock_text: str, check_text: str, *, uv_rc: int = 0) -> tuple[subprocess.CompletedProcess[str], list[str]]:
        (root / "requirements.lock").write_text(lock_text, encoding="utf-8")
        fixture = root / "check-fixture.txt"
        fixture.write_text(check_text, encoding="utf-8")
        bin_dir = root / "bin"
        bin_dir.mkdir()
        argv_file = root / "uv-argv.txt"
        stub = bin_dir / "uv"
        stub.write_text(
            '#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$UV_ARGV_FILE"\nif [[ "${UV_RC}" != "0" ]]; then\n  echo "uv failed" >&2\n  exit "$UV_RC"\nfi\nout=""\nprev=""\nfor a in "$@"; do\n  if [[ "$prev" == "-o" ]]; then out="$a"; fi\n  prev="$a"\ndone\ncp "$UV_FIXTURE" "$out"\n',
            encoding="utf-8",
        )
        stub.chmod(0o755)
        env = _child_env(
            PATH=str(bin_dir) + os.pathsep + os.environ.get("PATH", "/usr/bin:/bin"),
            UV_ARGV_FILE=str(argv_file),
            UV_RC=str(uv_rc),
            UV_FIXTURE=str(fixture),
        )
        proc = _run_script(_relocated(_freshness_step()["run"], scratch), cwd=root, env=env)
        argv = argv_file.read_text(encoding="utf-8").splitlines() if argv_file.exists() else []
        return proc, argv

    def test_comments_blanks_and_order_are_not_drift(self, tmp_path: Path, scratch: Path) -> None:
        lock_text = "# autogenerated\n# -o requirements.lock\n\nzlib==1.0\nnumpy==1.24.0\n"
        check_text = "# different header because -o /tmp/requirements.lock.check\n# -c requirements.lock\nnumpy==1.24.0\nzlib==1.0\n\n"
        proc, argv = self._run(tmp_path, scratch, lock_text, check_text)
        combined = proc.stdout + proc.stderr
        assert proc.returncode == 0, combined
        assert "satisfies pyproject.toml" in combined
        assert self._extras(argv) == EXTRAS
        assert "--upgrade" not in argv
        assert argv[argv.index("--constraint") + 1] == "requirements.lock"
        assert argv[argv.index("-o") + 1] == str(scratch / "requirements.lock.check")

    def test_a_moved_pin_fails_and_names_the_upgrade_refresh(self, tmp_path: Path, scratch: Path) -> None:
        proc, _argv = self._run(tmp_path, scratch, "numpy==1.24.0\n", "numpy==1.25.0\n")
        combined = proc.stdout + proc.stderr
        assert proc.returncode == 1, combined
        assert "no longer satisfies pyproject.toml" in combined
        assert "satisfies pyproject.toml ✓" not in combined
        for extra in EXTRAS:
            assert f"--extra {extra}" in combined
        assert "--upgrade -o requirements.lock" in combined

    def test_a_failing_resolve_is_not_a_pass(self, tmp_path: Path, scratch: Path) -> None:
        proc, argv = self._run(tmp_path, scratch, "numpy==1.24.0\n", "numpy==1.24.0\n", uv_rc=2)
        combined = proc.stdout + proc.stderr
        assert proc.returncode != 0, combined
        assert argv[:2] == ["pip", "compile"]
        assert "satisfies pyproject.toml ✓" not in combined

    @staticmethod
    def _extras(argv: list[str]) -> list[str]:
        return [argv[index + 1] for index, arg in enumerate(argv) if arg == "--extra"]


class TestLockfileWorkflowContract:
    """Wiring that decides who regenerates, and that the commit cannot fall back to git."""

    def test_the_job_skips_forks_and_release_heads_and_non_dependabot_pushes(self) -> None:
        job_if = _update()["jobs"]["update-lockfile"]["if"]
        assert "github.actor == 'dependabot[bot]'" in job_if
        assert "head.repo.full_name == github.repository" in job_if
        assert "!startsWith(github.head_ref, 'release/')" in job_if

    def test_push_is_only_dependabot_pip_and_pull_request_is_only_pyproject(self) -> None:
        triggers = _update()["on"]
        assert triggers["push"]["branches"] == ["dependabot/pip/**"]
        assert triggers["pull_request"]["paths"] == ["pyproject.toml"]

    def test_checkout_and_the_commit_use_the_dispatch_pat(self) -> None:
        checkout = _update_step("Checkout Code")
        commit = _update_step("Commit updated lockfile (GitHub-signed, via API)")
        assert checkout["with"]["token"] == PAT_SECRET
        assert checkout["with"]["ref"] == "${{ github.head_ref || github.ref }}"
        assert "github.token" not in str(checkout["with"])
        assert commit["env"]["GH_TOKEN"] == PAT_SECRET
        assert commit["env"]["BRANCH"] == "${{ github.head_ref || github.ref_name }}"
        assert "github.token" not in str(commit["env"])

    def test_every_mutating_step_waits_for_the_gate(self) -> None:
        names = [
            "Checkout Code",
            "Set up Python 3.14",
            "Install uv",
            "Regenerate requirements.lock",
            "Commit updated lockfile (GitHub-signed, via API)",
        ]
        for name in names:
            assert _update_step(name)["if"] == PROCEED_IF, name

    def test_regen_upgrades_and_the_checker_constrains_the_same_extras(self) -> None:
        regen = _code(_update_step("Regenerate requirements.lock")["run"])
        fresh = _code(_freshness_step()["run"])
        for extra in EXTRAS:
            assert f"--extra {extra}" in regen
            assert f"--extra {extra}" in fresh
        assert "--upgrade" in regen
        assert "--constraint" not in regen
        assert "--constraint requirements.lock" in fresh
        assert "-o requirements.lock" in regen
        assert "-o /tmp/requirements.lock.check" in fresh

    def test_the_commit_step_cannot_fall_back_to_a_local_git_commit(self) -> None:
        run = _code(_update_step("Commit updated lockfile (GitHub-signed, via API)")["run"])
        assert "createCommitOnBranch" in run
        assert "expectedHeadOid" in run
        assert HEADLINE in run
        assert "git commit" not in run
        assert "git push" not in run

    def test_both_jobs_pin_the_same_uv(self) -> None:
        update_install = _update_step("Install uv")["run"]
        ci_job = _workflow(CI_WORKFLOW)["jobs"]["lockfile-check"]
        ci_install = next(step["run"] for step in ci_job["steps"] if step.get("name") == "Install uv")
        # An exact pin, without restating the version: bumping uv in both files must not fail this.
        assert re.fullmatch(r"pip install uv==\d+\.\d+\.\d+", update_install.strip()), update_install
        assert update_install.strip() == ci_install.strip()
