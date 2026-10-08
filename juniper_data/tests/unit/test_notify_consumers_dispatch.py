"""Pin ``notify-consumers.yml``: a 204 is not delivery, and a bad version never dispatches.

#426 fires ``repository_dispatch`` of type ``juniper-data-published`` after a PyPI release.
#431 closed the residual gap on juniper-recurrence#178: GitHub returns 204 whether or not any
workflow listens, so the job then waits for a run whose default title is that event type.
A renamed listener, a ``run-name:``, or a listing that is not a run list must not look like
success -- and a listing GitHub never returned must not be blamed on the consumer.

These tests extract the workflow's own shell (not a reimplementation) and drive it with a
stub ``curl`` and a no-op ``sleep``. The real confirmation window is about two minutes;
nothing here waits.
"""

from __future__ import annotations

import json
import os
import subprocess  # nosec B404 - runs the workflow's OWN extracted shell hermetically (fixed argv)
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml

# CI collects ``-m "unit and not slow"`` with ``--strict-markers`` and does not auto-mark.
pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[3]
WORKFLOW = REPO / ".github" / "workflows" / "notify-consumers.yml"
PUBLISH = REPO / ".github" / "workflows" / "publish.yml"

EVENT = "juniper-data-published"
REPO_NAME = "juniper-recurrence"
TOKEN = "test-dispatch-token"
SHA = "0123456789abcdef0123456789abcdef01234567"
RUN_URL = "https://example.test/runs/7"

_CURL_STUB = """\
#!/bin/bash
set -euo pipefail
state="${CURL_STATE:?}"
mkdir -p "$state"
printf '%s\\n' "$*" >> "$state/invocations"
url=""
data=""
prev=""
for arg in "$@"; do
  case "$prev" in
    -d|--data|--data-raw) data="$arg" ;;
  esac
  case "$arg" in
    https://*) url="$arg" ;;
  esac
  prev="$arg"
done
printf '%s\\n' "$url" >> "$state/urls"
if [[ -n "$data" ]]; then
  printf '%s\\n' "$data" >> "$state/bodies"
fi
if [[ "$url" == */dispatches ]]; then
  code=0
  if [[ -f "$state/dispatch.exit" ]]; then code="$(cat "$state/dispatch.exit")"; fi
  if [[ -f "$state/dispatch.body" ]]; then cat "$state/dispatch.body"; fi
  exit "$code"
fi
if [[ "$url" == */actions/runs ]]; then
  n=0
  if [[ -f "$state/attempt" ]]; then n="$(cat "$state/attempt")"; fi
  n=$((n + 1))
  printf '%s' "$n" > "$state/attempt"
  mkdir -p "$state/runs"
  body="$state/runs/$n.body"
  codef="$state/runs/$n.exit"
  if [[ ! -f "$body" && -f "$state/runs/default.body" ]]; then body="$state/runs/default.body"; fi
  if [[ ! -f "$codef" && -f "$state/runs/default.exit" ]]; then codef="$state/runs/default.exit"; fi
  code=0
  if [[ -f "$codef" ]]; then code="$(cat "$codef")"; fi
  if [[ -f "$body" ]]; then cat "$body"; fi
  exit "$code"
fi
echo "curl stub: unexpected url: ${url:-<none>}" >&2
exit 97
"""

_SLEEP_STUB = """\
#!/bin/bash
printf '%s\\n' "$*" >> "${CURL_STATE:?}/sleeps"
exit 0
"""


def _load(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    # PyYAML (YAML 1.1) reads a bare ``on:`` key as boolean True.
    if True in data:
        data["on"] = data.pop(True)
    return data


def _notify() -> dict[str, Any]:
    return _load(WORKFLOW)


def _step(name: str) -> dict[str, Any]:
    matches = [s for s in _notify()["jobs"]["dispatch"]["steps"] if s.get("name") == name]
    assert len(matches) == 1, f"expected exactly one step named {name!r}, found {len(matches)}"
    assert "run" in matches[0], f"{name!r} has no run script"
    return matches[0]


def _child_env(**overrides: str) -> dict[str, str]:
    """A minimal child environment. The parent environment is not copied."""
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),  # nosec B108 - git-less fallback; the shell needs a HOME
        "LANG": "C",
        "LC_ALL": "C",
    }
    env.update(overrides)
    return env


def _prepare(tmp: Path, script: str) -> tuple[Path, Path]:
    state = tmp / "state"
    state.mkdir()
    (state / "runs").mkdir()
    bindir = tmp / "bin"
    bindir.mkdir()
    curl = bindir / "curl"
    curl.write_text(_CURL_STUB, encoding="utf-8")
    curl.chmod(0o755)
    sleep = bindir / "sleep"
    sleep.write_text(_SLEEP_STUB, encoding="utf-8")
    sleep.chmod(0o755)
    script_path = tmp / "step.sh"
    script_path.write_text(script, encoding="utf-8")
    return state, script_path


@dataclass
class _Step:
    """One run of an extracted workflow step, plus the files it wrote."""

    returncode: int
    stdout: str
    stderr: str
    state: Path
    summary: str
    github_output: str


def _execute(tmp: Path, state: Path, script_path: Path, **env: str) -> _Step:
    output = tmp / "github_output"
    summary = tmp / "step_summary"
    output.write_text("", encoding="utf-8")
    summary.write_text("", encoding="utf-8")
    path = f"{tmp / 'bin'}:{os.environ.get('PATH', '/usr/bin:/bin')}"
    proc = subprocess.run(  # nosec B603 - bash plus this repo's extracted workflow step, fixed argv
        ["bash", str(script_path)],
        capture_output=True,
        text=True,
        timeout=20,
        env=_child_env(
            PATH=path,
            CURL_STATE=str(state),
            GITHUB_OUTPUT=str(output),
            GITHUB_STEP_SUMMARY=str(summary),
            **env,
        ),
        check=False,
    )
    return _Step(proc.returncode, proc.stdout, proc.stderr, state, summary.read_text(encoding="utf-8"), output.read_text(encoding="utf-8"))


def _run(tmp: Path, script: str, *, dispatch_exit: str | None = None, dispatch_body: str | None = None, **env: str) -> _Step:
    state, script_path = _prepare(tmp, script)
    if dispatch_exit is not None:
        (state / "dispatch.exit").write_text(dispatch_exit, encoding="utf-8")
    if dispatch_body is not None:
        (state / "dispatch.body").write_text(dispatch_body, encoding="utf-8")
    return _execute(tmp, state, script_path, **env)


def _invocations(proc: _Step) -> str:
    path = proc.state / "invocations"
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def _sleeps(proc: _Step) -> list[str]:
    path = proc.state / "sleeps"
    if not path.is_file():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


def _date_has_gnu_d() -> bool:
    """Whether this runner's ``date`` takes GNU ``-d``, which the Dispatch step uses for ``since``."""
    try:
        proc = subprocess.run(["date", "-u", "-d", "30 seconds ago", "+%s"], capture_output=True, text=True, check=False, timeout=10)  # nosec B603 B607 - capability probe, fixed argv
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0 and proc.stdout.strip().isdigit()


# The Dispatch step runs on ubuntu-latest. Under BSD/macOS ``date`` (CI's required macOS unit-test
# leg) ``date -d`` fails, so the step dies before it reaches ``curl`` and these cases test nothing.
# Never skipped on Linux: there a failing probe is a real problem, so the cases run and say so.
needs_gnu_date = pytest.mark.skipif(not sys.platform.startswith("linux") and not _date_has_gnu_d(), reason="the Dispatch step runs on ubuntu-latest and computes `since` with GNU `date -d`; this runner's date has no -d")


def _listing(*titles: str, urls: list[str] | None = None) -> str:
    runs = []
    for index, title in enumerate(titles):
        url = RUN_URL if urls is None else urls[index]
        runs.append({"name": "CI", "display_title": title, "html_url": url})
    return json.dumps({"workflow_runs": runs})


def _run_confirm(
    tmp: Path,
    *,
    body: str | None,
    exit_code: int | None,
    bodies: dict[int, tuple[str, int]] | None,
    since: str,
) -> _Step:
    """Drive the confirmation step.

    ``body`` / ``exit_code`` apply to every attempt. ``bodies`` overrides individual
    attempts (1-based) with ``(stdout, exit code)``.
    """
    state, script_path = _prepare(tmp, _step("Confirm the consumer started a run")["run"])
    if body is not None or exit_code is not None:
        if body is not None:
            (state / "runs" / "default.body").write_text(body, encoding="utf-8")
        if exit_code is not None:
            (state / "runs" / "default.exit").write_text(str(exit_code), encoding="utf-8")
    for attempt, (attempt_body, attempt_code) in (bodies or {}).items():
        (state / "runs" / f"{attempt}.body").write_text(attempt_body, encoding="utf-8")
        (state / "runs" / f"{attempt}.exit").write_text(str(attempt_code), encoding="utf-8")
    return _execute(tmp, state, script_path, TOKEN=TOKEN, REPO=REPO_NAME, SINCE=since, EVENT_TYPE=EVENT)


class TestDispatchStep:
    """The POST itself: version grammar, the token, and a non-2xx curl."""

    @needs_gnu_date
    def test_a_v_prefixed_release_dispatches_the_bare_version_and_not_the_token(self, tmp_path: Path) -> None:
        proc = _run(
            tmp_path,
            _step("Dispatch")["run"],
            TOKEN=TOKEN,
            RAW_VERSION="v0.16.0",
            REPO=REPO_NAME,
            SOURCE_SHA=SHA,
        )
        assert proc.returncode == 0, proc.stderr
        payload = json.loads((proc.state / "bodies").read_text(encoding="utf-8"))
        assert payload == {
            "event_type": EVENT,
            "client_payload": {"source": "juniper-data", "version": "0.16.0", "sha": SHA},
        }
        assert TOKEN not in json.dumps(payload)
        assert TOKEN not in proc.stdout
        assert TOKEN not in proc.summary
        invocation = _invocations(proc)
        assert "--fail-with-body" in invocation
        assert f"https://api.github.com/repos/pcalnon/{REPO_NAME}/dispatches" in invocation
        assert f"Bearer {TOKEN}" in invocation
        assert "dispatched" in proc.summary
        since_line = next(line for line in proc.github_output.splitlines() if line.startswith("since="))
        stamp = datetime.strptime(since_line.removeprefix("since="), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        delta = datetime.now(UTC) - stamp
        assert timedelta(seconds=5) <= delta <= timedelta(seconds=120)

    @pytest.mark.parametrize("raw", ["0.16", "v0.16", "vv0.16.0", "V0.16.0", "0.16.0-rc1", "0.16.0 ", "latest", "", "1.2.3;touch"])
    def test_a_version_that_is_not_x_y_z_never_calls_curl(self, tmp_path: Path, raw: str) -> None:
        proc = _run(
            tmp_path,
            _step("Dispatch")["run"],
            TOKEN=TOKEN,
            RAW_VERSION=raw,
            REPO=REPO_NAME,
            SOURCE_SHA=SHA,
        )
        assert proc.returncode == 1
        assert "version must be X.Y.Z" in proc.stdout
        assert _invocations(proc) == ""
        assert proc.summary == ""

    def test_an_empty_token_never_calls_curl(self, tmp_path: Path) -> None:
        proc = _run(
            tmp_path,
            _step("Dispatch")["run"],
            TOKEN="",
            RAW_VERSION="0.16.0",
            REPO=REPO_NAME,
            SOURCE_SHA=SHA,
        )
        assert proc.returncode == 1
        assert "CROSS_REPO_DISPATCH_TOKEN is not available" in proc.stdout
        assert _invocations(proc) == ""

    @needs_gnu_date
    def test_a_failing_curl_fails_the_step(self, tmp_path: Path) -> None:
        """A 403 from an under-scoped token used to look like success when ``--fail`` was omitted."""
        proc = _run(
            tmp_path,
            _step("Dispatch")["run"],
            dispatch_exit="22",
            dispatch_body="Forbidden",
            TOKEN=TOKEN,
            RAW_VERSION="0.16.0",
            REPO=REPO_NAME,
            SOURCE_SHA=SHA,
        )
        assert proc.returncode != 0
        # The step must have reached the POST. A step that died earlier also exits non-zero.
        assert f"https://api.github.com/repos/pcalnon/{REPO_NAME}/dispatches" in _invocations(proc)
        assert "dispatched" not in proc.summary
        assert TOKEN not in proc.stdout
        assert TOKEN not in proc.stderr


class TestConfirmStep:
    """The wait after the 204. Title match is exact; a bad listing is a different error."""

    def test_an_exact_title_match_is_delivery(self, tmp_path: Path) -> None:
        proc = _run_confirm(tmp_path, body=_listing(EVENT), exit_code=0, bodies=None, since="2026-10-05T20:00:00Z")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert RUN_URL in proc.stdout
        assert RUN_URL in proc.summary
        assert _sleeps(proc) == []
        invocation = _invocations(proc)
        assert f"https://api.github.com/repos/pcalnon/{REPO_NAME}/actions/runs" in invocation
        assert "event=repository_dispatch" in invocation
        assert "created=>=2026-10-05T20:00:00Z" in invocation
        assert "per_page=20" in invocation
        assert TOKEN not in proc.stdout
        assert TOKEN not in proc.summary

    def test_a_later_row_with_the_event_title_counts(self, tmp_path: Path) -> None:
        """The first row is an ordinary CI run. Selecting before filtering would miss the dispatch."""
        body = json.dumps(
            {
                "workflow_runs": [
                    {"name": "CI", "display_title": "CI", "html_url": "https://example.test/runs/1"},
                    {"name": "CI", "display_title": EVENT, "html_url": RUN_URL},
                ]
            }
        )
        proc = _run_confirm(tmp_path, body=body, exit_code=0, bodies=None, since="2026-10-05T20:00:00Z")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert RUN_URL in proc.stdout
        assert "https://example.test/runs/1" not in proc.stdout

    @pytest.mark.parametrize(
        "title",
        [
            "CI",  # a run-name replaces the default title, which is the event type
            f"{EVENT}-extra",  # a prefix is not the event
            "Juniper-data-published",  # the match is case-sensitive
        ],
    )
    def test_a_title_that_is_not_the_event_is_not_delivery(self, tmp_path: Path, title: str) -> None:
        proc = _run_confirm(tmp_path, body=_listing(title), exit_code=0, bodies=None, since="2026-10-05T20:00:00Z")
        assert proc.returncode == 1
        assert "started no juniper-data-published run" in proc.stdout
        assert "could not be listed" not in proc.stdout
        assert _sleeps(proc) == ["10"] * 11

    def test_an_empty_run_list_is_the_consumer_not_starting(self, tmp_path: Path) -> None:
        proc = _run_confirm(tmp_path, body=_listing(), exit_code=0, bodies=None, since="2026-10-05T20:00:00Z")
        assert proc.returncode == 1
        assert "started no juniper-data-published run" in proc.stdout
        assert "could not be listed" not in proc.stdout
        assert "listings failed" not in proc.stdout

    @pytest.mark.parametrize("body", ["{}", '{"message":"Not Found"}', "", "not-json"])
    def test_a_body_that_is_not_a_run_listing_is_not_blamed_on_the_consumer(self, tmp_path: Path, body: str) -> None:
        proc = _run_confirm(tmp_path, body=body, exit_code=0, bodies=None, since="2026-10-05T20:00:00Z")
        assert proc.returncode == 1
        assert "could not be listed" in proc.stdout
        assert "started no juniper-data-published run" not in proc.stdout
        assert "not a run listing" in proc.stdout
        assert TOKEN not in proc.stdout

    def test_a_curl_failure_on_every_attempt_is_an_unknown_not_a_missing_listener(self, tmp_path: Path) -> None:
        proc = _run_confirm(tmp_path, body="HTTP 403", exit_code=22, bodies=None, since="2026-10-05T20:00:00Z")
        assert proc.returncode == 1
        assert "could not be listed" in proc.stdout
        assert "curl exit 22" in proc.stdout
        assert "started no juniper-data-published run" not in proc.stdout
        assert TOKEN not in proc.stdout

    def test_a_listing_that_fails_after_an_empty_one_says_the_window_was_partial(self, tmp_path: Path) -> None:
        bodies = {1: (_listing(), 0)}
        for attempt in range(2, 13):
            bodies[attempt] = ("HTTP 502", 22)
        proc = _run_confirm(tmp_path, body=None, exit_code=None, bodies=bodies, since="2026-10-05T20:00:00Z")
        assert proc.returncode == 1
        assert "started no juniper-data-published run" in proc.stdout
        assert "11 of 12 listings failed" in proc.stdout
        assert "could not be listed in any of" not in proc.stdout

    def test_a_match_on_a_later_attempt_still_counts(self, tmp_path: Path) -> None:
        bodies = {
            1: (_listing(), 0),
            2: (_listing(), 0),
            3: (_listing(EVENT), 0),
        }
        proc = _run_confirm(tmp_path, body=None, exit_code=None, bodies=bodies, since="2026-10-05T20:00:00Z")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert RUN_URL in proc.stdout
        assert _sleeps(proc) == ["10", "10"]


class TestWorkflowWiring:
    """The caller must not dispatch before PyPI, and the confirm step must be able to fail the job."""

    def test_publish_calls_the_workflow_only_after_pypi(self) -> None:
        job = _load(PUBLISH)["jobs"]["notify-consumers"]
        assert job["needs"] == "pypi"
        assert job["uses"] == "./.github/workflows/notify-consumers.yml"
        assert job["with"]["version"] == "${{ github.event.release.tag_name }}"
        assert job["secrets"]["CROSS_REPO_DISPATCH_TOKEN"] == "${{ secrets.CROSS_REPO_DISPATCH_TOKEN }}"
        assert job["permissions"] == {}

    def test_the_reusable_workflow_stays_fail_closed(self) -> None:
        doc = _notify()
        assert doc["permissions"] == {}
        job = doc["jobs"]["dispatch"]
        assert job["strategy"]["matrix"]["repo"] == [REPO_NAME]
        assert "continue-on-error" not in job
        confirm = _step("Confirm the consumer started a run")
        assert "continue-on-error" not in confirm
        script = confirm["run"]
        assert "attempts=12" in script
        assert "sleep 10" in script
        assert "display_title" in script
        assert "GITHUB_TOKEN" not in script
        dispatch = _step("Dispatch")["run"]
        assert "--fail-with-body" in dispatch
        assert "GITHUB_TOKEN" not in dispatch
