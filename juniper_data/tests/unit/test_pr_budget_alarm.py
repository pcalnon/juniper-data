"""Execute the open-PR budget alarm's own shell.

``.github/workflows/pr-budget-alarm.yml`` is report-only: a breach stays green,
a failed ``gh pr list`` stays green and must not be reported as an empty queue,
and the Slack notice carries counts plus the run URL. The webhook itself stays
out of that notice. An empty or unset ``PR_BUDGET_WARN`` / ``PR_BUDGET_ALARM``
falls back to 15 / 30. ``cursor/`` is a prefix match, so ``Cursor/``, ``cursor-foo``,
and ``feature/cursor/x`` are not fleet PRs.

The cursor count cannot raise a level the total would not — both share one
threshold and a cursor PR is also an open PR. What it can do is mis-report the
fleet column while the level stays OK. These tests pin that column.

Nothing here reimplements the shell. The workflow YAML is the source, and a
missing step fails the test rather than skipping it.
"""

from __future__ import annotations

import json
import os
import subprocess  # nosec B404 - runs the workflow's own extracted shell with fixed argv
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[3]
WORKFLOW = REPO / ".github" / "workflows" / "pr-budget-alarm.yml"

_WEBHOOK = "https://hooks.example.test/services/T000/B000/fake"


def _workflow() -> dict[str, Any]:
    data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    # PyYAML (YAML 1.1) reads the bare `on:` key as boolean True.
    if True in data and "on" not in data:
        data["on"] = data.pop(True)
    return data


def _steps() -> list[dict[str, Any]]:
    return list(_workflow()["jobs"]["budget-alarm"]["steps"])


def _one(matches: list[dict[str, Any]], label: str) -> dict[str, Any]:
    assert len(matches) == 1, f"expected one step for {label}, found {len(matches)}"
    assert "run" in matches[0]
    return matches[0]


def _count_step() -> dict[str, Any]:
    return _one([step for step in _steps() if step.get("id") == "count"], "id=count")


def _slack_step() -> dict[str, Any]:
    return _one([step for step in _steps() if str(step.get("name", "")).startswith("Slack notification")], "Slack notification")


def _child_env(path: str, **overrides: str) -> dict[str, str]:
    """Minimal child environment. The parent environment is not copied."""
    env = {
        "PATH": path,
        "HOME": os.environ.get("HOME", "/tmp"),  # nosec B108 - git/bash fallback only; no secrets
        "LANG": "C",
    }
    env.update(overrides)
    return env


def _write_executable(directory: Path, name: str, body: str) -> None:
    path = directory / name
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _outputs(text: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in text.splitlines():
        key, _, value = line.partition("=")
        if key:
            parsed[key] = value
    return parsed


def _run_shell(script: str, env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # nosec B603 B607 - fixed bash argv; script is the workflow's own step
        ["bash", "-c", script],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _prs(*names: str) -> str:
    return json.dumps([{"number": index + 1, "headRefName": name} for index, name in enumerate(names)])


class _CountResult:
    def __init__(self, proc: subprocess.CompletedProcess[str], outputs: dict[str, str], summary: str) -> None:
        self.proc = proc
        self.outputs = outputs
        self.summary = summary


def _count(
    tmp_path: Path,
    prs_json: str,
    *,
    gh_rc: int = 0,
    gh_err: str = "",
    warn: str | None = None,
    alarm: str | None = None,
) -> _CountResult:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    _write_executable(
        bin_dir,
        "gh",
        '#!/bin/sh\nif [ "${GH_RC}" != "0" ]; then\n  printf "%s" "${GH_ERR}" >&2\n  exit "${GH_RC}"\nfi\nprintf "%s" "${GH_STDOUT}"\n',
    )
    output = tmp_path / "output.txt"
    summary = tmp_path / "summary.md"
    output.write_text("", encoding="utf-8")
    summary.write_text("", encoding="utf-8")
    env = _child_env(
        f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '/usr/bin:/bin')}",
        GH_REPO="pcalnon/juniper-data",
        GH_RC=str(gh_rc),
        GH_ERR=gh_err,
        GH_STDOUT=prs_json,
        GITHUB_OUTPUT=str(output),
        GITHUB_STEP_SUMMARY=str(summary),
    )
    if warn is not None:
        env["PR_BUDGET_WARN"] = warn
    if alarm is not None:
        env["PR_BUDGET_ALARM"] = alarm
    proc = _run_shell(_count_step()["run"], env, tmp_path)
    return _CountResult(proc, _outputs(output.read_text(encoding="utf-8")), summary.read_text(encoding="utf-8"))


class TestBudgetThresholds:
    def test_unset_thresholds_are_15_and_30(self, tmp_path: Path) -> None:
        result = _count(tmp_path, "[]")
        assert result.proc.returncode == 0, result.proc.stderr
        assert result.outputs["level"] == "OK"
        assert result.outputs["warn"] == "15"
        assert result.outputs["alarm"] == "30"
        assert result.outputs["total"] == "0"
        assert result.outputs["cursor"] == "0"

    def test_empty_thresholds_are_15_and_30(self, tmp_path: Path) -> None:
        """`${VAR:-15}` treats an empty repo variable the same as an unset one."""
        result = _count(tmp_path, "[]", warn="", alarm="")
        assert result.proc.returncode == 0, result.proc.stderr
        assert result.outputs["warn"] == "15"
        assert result.outputs["alarm"] == "30"
        assert result.outputs["level"] == "OK"

    @pytest.mark.parametrize(
        ("total", "level"),
        [(14, "OK"), (15, "WARN"), (29, "WARN"), (30, "ALARM"), (31, "ALARM")],
    )
    def test_total_crosses_warn_then_alarm_on_the_boundary(self, tmp_path: Path, total: int, level: str) -> None:
        names = [f"feature/{index}" for index in range(total)]
        result = _count(tmp_path, _prs(*names))
        assert result.proc.returncode == 0, result.proc.stderr
        assert result.outputs["level"] == level
        assert result.outputs["total"] == str(total)
        assert f"| Status | **{level}** |" in result.summary

    def test_a_breach_stays_green(self, tmp_path: Path) -> None:
        result = _count(tmp_path, _prs(*(f"feature/{index}" for index in range(30))))
        assert result.proc.returncode == 0
        assert result.outputs["level"] == "ALARM"
        assert "this run stays green" in result.summary

    def test_a_lowered_ceiling_uses_the_repo_variable(self, tmp_path: Path) -> None:
        result = _count(tmp_path, _prs("feature/a", "feature/b"), warn="2", alarm="4")
        assert result.outputs["level"] == "WARN"
        assert result.outputs["warn"] == "2"
        result = _count(tmp_path, _prs("a", "b", "c", "d"), warn="2", alarm="4")
        assert result.outputs["level"] == "ALARM"


class TestCursorPrefix:
    def test_only_a_cursor_slash_prefix_counts(self, tmp_path: Path) -> None:
        result = _count(
            tmp_path,
            _prs(
                "cursor/foo",
                "cursor/",
                "Cursor/foo",
                "cursor-foo",
                "feature/cursor/x",
                "main",
            ),
        )
        assert result.proc.returncode == 0, result.proc.stderr
        assert result.outputs["total"] == "6"
        assert result.outputs["cursor"] == "2"
        assert result.outputs["level"] == "OK"
        assert "| Open `cursor/` PRs | 2 |" in result.summary


class TestQueryFailureIsNotAnEmptyQueue:
    def test_gh_failure_stays_green_and_writes_no_zero_count(self, tmp_path: Path) -> None:
        result = _count(tmp_path, "", gh_rc=1, gh_err="api down\nretry later\n")
        assert result.proc.returncode == 0, result.proc.stderr
        assert result.outputs == {"level": "OK"}
        assert "total=" not in (tmp_path / "output.txt").read_text(encoding="utf-8")
        assert "Could not query open PRs" in result.summary
        assert "Open PRs (total)" not in result.summary
        assert "::warning title=pr-budget-alarm::Could not list open PRs: api down retry later " in result.proc.stdout

    def test_non_json_stdout_is_not_a_quiet_zero(self, tmp_path: Path) -> None:
        """``gh`` exiting 0 with a non-JSON body fails ``jq`` and publishes no count.

        Two residuals stay unpinned. ``jq`` 1.7 exits 0 on an empty body and on
        ``null`` (``length`` is 0), and this shell then reports OK. ``gh pr list
        --json`` emits an array, so those shapes are not the failure path above.
        """
        result = _count(tmp_path, "not-json")
        published = (tmp_path / "output.txt").read_text(encoding="utf-8")
        assert result.proc.returncode != 0
        assert "total=0" not in published
        assert result.outputs.get("level") != "OK" or "total" not in result.outputs


class TestSlackNotice:
    def _slack(
        self,
        tmp_path: Path,
        *,
        webhook: str | None,
        curl_rc: int = 0,
        level: str = "ALARM",
    ) -> tuple[subprocess.CompletedProcess[str], str]:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        args_file = tmp_path / "curl-args"
        _write_executable(
            bin_dir,
            "curl",
            "#!/usr/bin/env python3\nimport os, sys\nopen(os.environ['CURL_ARGS'], 'w', encoding='utf-8').write('\\0'.join(sys.argv[1:]))\nraise SystemExit(int(os.environ.get('CURL_RC', '0')))\n",
        )
        env = _child_env(
            f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '/usr/bin:/bin')}",
            CURL_ARGS=str(args_file),
            CURL_RC=str(curl_rc),
            RUN_URL="https://github.com/pcalnon/juniper-data/actions/runs/99",
            LEVEL=level,
            TOTAL="30",
            CURSOR="12",
            WARN="15",
            ALARM="30",
        )
        if webhook is not None:
            env["SLACK_WEBHOOK_URL"] = webhook
        proc = _run_shell(_slack_step()["run"], env, tmp_path)
        captured = args_file.read_text(encoding="utf-8") if args_file.is_file() else ""
        return proc, captured

    def test_a_missing_webhook_annotates_and_does_not_post(self, tmp_path: Path) -> None:
        proc, captured = self._slack(tmp_path, webhook=None)
        assert proc.returncode == 0, proc.stderr
        assert captured == ""
        assert "SLACK_WEBHOOK_URL is not set" in proc.stdout
        assert "30 open PR(s), 12 on cursor/ branches" in proc.stdout
        assert _WEBHOOK not in proc.stdout

    def test_an_empty_webhook_is_the_same_as_a_missing_one(self, tmp_path: Path) -> None:
        proc, captured = self._slack(tmp_path, webhook="")
        assert proc.returncode == 0, proc.stderr
        assert captured == ""

    def test_the_payload_has_the_counts_and_not_the_webhook(self, tmp_path: Path) -> None:
        proc, captured = self._slack(tmp_path, webhook=_WEBHOOK)
        assert proc.returncode == 0, proc.stderr
        assert _WEBHOOK not in proc.stdout
        assert _WEBHOOK not in proc.stderr
        args = captured.split("\0")
        assert args[0:3] == ["-fsS", "-X", "POST"]
        assert "Content-Type: application/json" in args
        payload = json.loads(args[args.index("-d") + 1])
        text = payload["text"]
        assert text.startswith("PR budget ALARM: 30 open PR(s), 12 on cursor/ branches")
        assert "warn=15" in text and "alarm=30" in text
        assert text.endswith("Run: https://github.com/pcalnon/juniper-data/actions/runs/99")
        assert _WEBHOOK not in text
        assert args[-1] == _WEBHOOK

    def test_a_failed_post_fails_the_shell(self, tmp_path: Path) -> None:
        """The workflow's ``continue-on-error`` is what keeps the run green, not the shell."""
        proc, _captured = self._slack(tmp_path, webhook=_WEBHOOK, curl_rc=22)
        assert proc.returncode != 0


class TestBudgetWorkflowWiring:
    def test_the_alarm_is_schedule_and_dispatch_only(self) -> None:
        triggers = _workflow()["on"]
        assert "pull_request" not in triggers
        assert triggers["schedule"] == [{"cron": "0 14 * * *"}]
        assert "workflow_dispatch" in triggers

    def test_permissions_stay_read_only(self) -> None:
        assert _workflow()["permissions"] == {"contents": "read", "pull-requests": "read"}

    def test_a_breach_still_notifies_and_a_post_failure_does_not_fail_the_job(self) -> None:
        step = _slack_step()
        assert step["if"] == "steps.count.outputs.level != 'OK'"
        assert step["continue-on-error"] is True
        count = _count_step()
        assert "continue-on-error" not in count

    def test_the_count_step_never_mentions_the_webhook(self) -> None:
        assert "SLACK_WEBHOOK_URL" not in _count_step()["run"]
