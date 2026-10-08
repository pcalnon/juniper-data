#!/usr/bin/env python3
"""Execute the main-verify failure-notify shells.

The catch-up rehearsal never runs the tracker upsert or the Slack post. The
tracker is one open issue per red streak, matched on the workflow's own
authorship plus the exact title. A pull request, a longer title, a case
change, or a later duplicate must not capture it. Opening the issue is the
point of the job: a failed create fails the step, and a failed label does not.
The Slack text must not carry the webhook.

Project: juniper-data
"""

from __future__ import annotations

import json
import os
import subprocess  # nosec B404 - runs the workflow's own extracted shell (fixed argv)
import sys
import tempfile
import unittest
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

WORKFLOW_NAME = "main-verify.yml"
UPSERT_STEP = "Upsert tracking issue (stable title, one per red streak)"
SLACK_STEP = "Slack notification (non-blocking)"
TITLE = "main-verify: post-merge verification failing"
SHA = "abc123def4567890aaaa"
TOKEN = "ghp_SENTINEL_NOT_A_REAL_TOKEN"  # nosec B105 - fixture sentinel, asserted absent from output
WEBHOOK = "https://hooks.example.test/services/T000/B000/SECRETVALUE"
REPO = "pcalnon/juniper-data"

_GH_STUB = r"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\0' "$@" >> "$GH_LOG"
printf '\n' >> "$GH_LOG"
cmd="${1:-}"
shift || true
if [ "$cmd" = "api" ]; then
  url=""
  jq_filter=""
  prev=""
  for a in "$@"; do
    if [ "$prev" = "--jq" ]; then
      jq_filter="$a"
    fi
    case "$a" in
      repos/*) url="$a" ;;
    esac
    prev="$a"
  done
  printf '%s\n' "$url" >> "$GH_URLS"
  if [ "${GH_API_RC:-0}" != "0" ]; then
    echo "api failed" >&2
    exit "${GH_API_RC}"
  fi
  if [ -n "$jq_filter" ]; then
    jq -r "$jq_filter" "$GH_ISSUES_JSON"
  else
    cat "$GH_ISSUES_JSON"
  fi
  exit 0
fi
if [ "$cmd" = "label" ]; then
  exit "${GH_LABEL_RC:-0}"
fi
if [ "$cmd" = "issue" ]; then
  sub="${1:-}"
  if [ "$sub" = "comment" ]; then
    exit "${GH_COMMENT_RC:-0}"
  fi
  if [ "$sub" = "create" ]; then
    if [ "${GH_CREATE_RC:-0}" != "0" ]; then
      echo "create failed" >&2
      exit "${GH_CREATE_RC}"
    fi
    printf '%s\n' "https://github.com/pcalnon/juniper-data/issues/${GH_NEW_NUMBER:-77}"
    exit 0
  fi
  if [ "$sub" = "edit" ]; then
    exit "${GH_EDIT_RC:-0}"
  fi
fi
echo "unexpected gh invocation: $cmd" >&2
exit 99
"""

_CURL_STUB = r"""#!/usr/bin/env bash
printf '%s\0' "$@" >> "$CURL_LOG"
printf '\n' >> "$CURL_LOG"
exit "${CURL_RC:-0}"
"""


def _child_env(**overrides: str) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),  # nosec B108 - bash fallback only
        "LANG": "C",
    }
    env.update(overrides)
    return env


def _repo_root() -> Path:
    cur = Path(__file__).resolve().parent
    for _ in range(8):
        if (cur / ".github" / "workflows").is_dir():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise AssertionError(f"could not locate repo root from {Path(__file__)}")


def _workflow() -> dict:
    wf = _repo_root() / ".github" / "workflows" / WORKFLOW_NAME
    if not wf.is_file():
        raise AssertionError(f"{WORKFLOW_NAME} missing at {wf}")
    return yaml.safe_load(wf.read_text(encoding="utf-8"))


def _step(job: str, name: str) -> dict:
    steps = _workflow().get("jobs", {}).get(job, {}).get("steps", [])
    step = next((s for s in steps if s.get("name") == name), None)
    if step is None or "run" not in step:
        raise AssertionError(f"{name!r} run step missing from {job}")
    return step


def _invocations(log: Path) -> list[list[str]]:
    if not log.is_file():
        return []
    rows: list[list[str]] = []
    for row in log.read_bytes().split(b"\n"):
        if not row:
            continue
        rows.append([part.decode() for part in row.split(b"\0") if part])
    return rows


def _commands(invocations: list[list[str]], *prefix: str) -> list[list[str]]:
    return [inv for inv in invocations if inv[: len(prefix)] == list(prefix)]


class NotifyUpsertRehearsalTest(unittest.TestCase):
    """Drive the real upsert shell against a stub ``gh`` and real ``jq``."""

    script: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.script = _step("notify", UPSERT_STEP)["run"]
        notify = _workflow()["jobs"]["notify"]
        if notify.get("if") != "failure()":
            raise AssertionError("notify must stay on failure() so a green main does not file a tracker")

    def _run(
        self,
        issues: list[dict],
        *,
        api_rc: int = 0,
        comment_rc: int = 0,
        create_rc: int = 0,
        label_rc: int = 0,
        edit_rc: int = 0,
    ) -> tuple[subprocess.CompletedProcess[str], list[list[str]], list[str], Path]:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            work = root / "work"
            work.mkdir()
            bin_dir = root / "bin"
            bin_dir.mkdir()
            gh = bin_dir / "gh"
            gh.write_text(_GH_STUB, encoding="utf-8")
            gh.chmod(0o755)
            log = root / "gh.log"
            urls = root / "urls.txt"
            issues_path = root / "issues.json"
            issues_path.write_text(json.dumps(issues), encoding="utf-8")
            env = _child_env(
                GH_TOKEN=TOKEN,
                REPO=REPO,
                TITLE=TITLE,
                SHA=SHA,
                RUN_URL=f"https://github.com/{REPO}/actions/runs/99",
                SYMBOL_RESULT="failure",
                GH_LOG=str(log),
                GH_URLS=str(urls),
                GH_ISSUES_JSON=str(issues_path),
                GH_API_RC=str(api_rc),
                GH_COMMENT_RC=str(comment_rc),
                GH_CREATE_RC=str(create_rc),
                GH_LABEL_RC=str(label_rc),
                GH_EDIT_RC=str(edit_rc),
            )
            env["PATH"] = str(bin_dir) + os.pathsep + env["PATH"]
            script = work / "upsert.sh"
            script.write_text(self.script, encoding="utf-8")
            proc = subprocess.run(  # nosec B603 B607 - extracted workflow shell, fixed argv
                ["bash", str(script)],
                cwd=work,
                capture_output=True,
                text=True,
                env=env,
                check=False,
                timeout=15,
            )
            url_lines = urls.read_text(encoding="utf-8").splitlines() if urls.is_file() else []
            # Keep the body files readable after the temp dir is removed.
            bodies = {}
            for name in ("issue-body.md", "issue-comment.md"):
                path = work / name
                if path.is_file():
                    bodies[name] = path.read_text(encoding="utf-8")
            proc.bodies = bodies  # type: ignore[attr-defined]
            return proc, _invocations(log), url_lines, work

    def _assert_no_token(self, proc: subprocess.CompletedProcess[str]) -> None:
        blob = proc.stdout + proc.stderr + "".join(proc.bodies.values())  # type: ignore[attr-defined]
        self.assertNotIn(TOKEN, blob)

    def test_notify_job_runs_only_on_failure(self) -> None:
        self.assertEqual(_workflow()["jobs"]["notify"]["if"], "failure()")

    def test_first_exact_issue_is_commented_and_decoys_are_not(self) -> None:
        issues = [
            {"number": 11, "title": TITLE, "pull_request": {"url": "https://example.test/pull/11"}},
            {"number": 12, "title": TITLE + " today"},
            {"number": 13, "title": "Main-verify: post-merge verification failing"},
            {"number": 14, "title": "main-verify: post-merge"},
            {"number": 15, "title": TITLE},
            {"number": 16, "title": TITLE},
        ]
        proc, invocations, urls, _work = self._run(issues)
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        comments = _commands(invocations, "issue", "comment")
        self.assertEqual([inv[2] for inv in comments], ["15"])
        self.assertEqual(_commands(invocations, "issue", "create"), [])
        self.assertIn(SHA, proc.bodies["issue-comment.md"])  # type: ignore[attr-defined]
        self.assertTrue(urls)
        self.assertIn("creator=github-actions%5Bbot%5D", urls[0])
        self.assertIn("state=open", urls[0])
        self.assertIn("per_page=100", urls[0])
        self._assert_no_token(proc)

    def test_no_match_opens_one_issue_under_the_stable_title(self) -> None:
        proc, invocations, _urls, _work = self._run([])
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        creates = _commands(invocations, "issue", "create")
        self.assertEqual(len(creates), 1)
        title_at = creates[0].index("--title")
        self.assertEqual(creates[0][title_at + 1], TITLE)
        self.assertNotIn(SHA, creates[0][title_at + 1])
        body = proc.bodies["issue-body.md"]  # type: ignore[attr-defined]
        self.assertIn(SHA, body)
        self.assertNotIn(SHA, TITLE)
        edits = _commands(invocations, "issue", "edit")
        self.assertEqual([inv[2] for inv in edits], ["77"])
        self.assertIn("--add-label", edits[0])
        self.assertIn("main-verify", edits[0])
        self._assert_no_token(proc)

    def test_label_failures_still_exit_zero(self) -> None:
        proc, invocations, _urls, _work = self._run([], label_rc=1, edit_rc=1)
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        self.assertEqual(len(_commands(invocations, "issue", "create")), 1)
        self.assertEqual(len(_commands(invocations, "issue", "edit")), 1)
        self.assertEqual(len(_commands(invocations, "label", "create")), 1)

    def test_issue_create_failure_exits_and_does_not_edit(self) -> None:
        proc, invocations, _urls, _work = self._run([], create_rc=1)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("failed to open", proc.stdout + proc.stderr)
        self.assertEqual(_commands(invocations, "issue", "edit"), [])
        self.assertEqual(len(_commands(invocations, "issue", "create")), 1)

    def test_comment_failure_does_not_open_another_issue(self) -> None:
        proc, invocations, _urls, _work = self._run([{"number": 15, "title": TITLE}], comment_rc=1)
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(_commands(invocations, "issue", "create"), [])
        self.assertEqual([inv[2] for inv in _commands(invocations, "issue", "comment")], ["15"])

    def test_list_failure_still_opens_and_the_query_is_scoped(self) -> None:
        proc, invocations, urls, _work = self._run([{"number": 15, "title": TITLE}], api_rc=1)
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        self.assertEqual(len(_commands(invocations, "issue", "create")), 1)
        self.assertEqual(_commands(invocations, "issue", "comment"), [])
        self.assertEqual(len(urls), 1)
        self.assertIn("creator=github-actions%5Bbot%5D", urls[0])
        self.assertIn("state=open", urls[0])
        self.assertIn("per_page=100", urls[0])
        self.assertNotIn("creator=github-actions[bot]", urls[0])


class SlackNotifyRehearsalTest(unittest.TestCase):
    """The webhook is a secret. The payload is not a place it may appear."""

    step: dict
    script: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.step = _step("notify", SLACK_STEP)
        cls.script = cls.step["run"]

    def test_slack_step_is_non_blocking(self) -> None:
        self.assertIs(self.step.get("continue-on-error"), True)

    def _run_slack(self, webhook: str | None) -> tuple[subprocess.CompletedProcess[str], list[list[str]], str]:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            work = root / "work"
            work.mkdir()
            bin_dir = root / "bin"
            bin_dir.mkdir()
            curl = bin_dir / "curl"
            curl.write_text(_CURL_STUB, encoding="utf-8")
            curl.chmod(0o755)
            # The step calls a bare ``python``. Point it at this interpreter rather than a fixed
            # system path, which not every runner or container has.
            python = bin_dir / "python"
            python.symlink_to(sys.executable)
            log = root / "curl.log"
            env = _child_env(
                CURL_LOG=str(log),
                RUN_URL=f"https://github.com/{REPO}/actions/runs/99",
                SHA=SHA,
                SYMBOL_RESULT="failure",
            )
            if webhook is not None:
                env["SLACK_WEBHOOK_URL"] = webhook
            env["PATH"] = str(bin_dir) + os.pathsep + env["PATH"]
            script = work / "slack.sh"
            script.write_text(self.script, encoding="utf-8")
            proc = subprocess.run(  # nosec B603 B607 - extracted workflow shell, fixed argv
                ["bash", str(script)],
                cwd=work,
                capture_output=True,
                text=True,
                env=env,
                check=False,
                timeout=15,
            )
            payload = ""
            payload_path = work / "slack-payload.json"
            if payload_path.is_file():
                payload = payload_path.read_text(encoding="utf-8")
            return proc, _invocations(log), payload

    def test_missing_webhook_skips_without_calling_curl(self) -> None:
        proc, invocations, payload = self._run_slack(None)
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        self.assertIn("skipping Slack", proc.stdout)
        self.assertEqual(invocations, [])
        self.assertEqual(payload, "")

    def test_slack_payload_omits_the_webhook(self) -> None:
        proc, invocations, payload = self._run_slack(WEBHOOK)
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        self.assertNotIn(WEBHOOK, payload)
        self.assertNotIn("SECRETVALUE", payload)
        body = json.loads(payload)
        self.assertIn(SHA[:12], body["text"])
        self.assertEqual(len(invocations), 1)
        self.assertIn(WEBHOOK, invocations[0])
        self.assertNotIn(WEBHOOK, proc.stdout + proc.stderr)
