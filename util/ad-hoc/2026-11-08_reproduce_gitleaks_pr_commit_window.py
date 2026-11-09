#!/usr/bin/env python3
"""Reproduce gitleaks-action's pull-request commit window, and show ci.yml's replacement step scans the whole range.

Project: juniper-data
Sub-Project: ad-hoc tooling
Author: Paul Calnon
Created: 2026-11-08
Status: ad-hoc — investigation (the reproduction for the juniper-data defect in §4 of juniper-ml's notes/JUNIPER_2026-11-06_JUNIPER-ECOSYSTEM_CURSOR-FLOOD-4-DISPOSITION.md)
Retire when: gitleaks-action's pull_request path covers a pull request's whole range upstream, or ci.yml stops scanning pull requests with its own step.
Related: .github/workflows/ci.yml ("Run Gitleaks over the pull request"), juniper_data/tests/unit/test_gitleaks_config.py

gitleaks-action v3.0.0 (e0c47f4f8be36e29cdc102c57e68cb5cbf0e8d1e) lists a pull request's commits with one
unpaginated ``GET /repos/{owner}/{repo}/pulls/{n}/commits`` (30 per page, oldest first) and scans
``--no-merges --first-parent <page[0]>^..<page[-1]>`` (src/gitleaks.js). This script builds throwaway branches in a
SCRATCH CLONE that has no remote, plants GitHub-PAT-shaped tokens assembled at runtime (never a credential), and runs:

    linear-35          35 commits, a token in commit 1 and another in commit 35
      action-window      the action's own argv over the first API page (commits 1-30): finds 1, misses 35
      newest-30-window   the same argv over the newest 30 commits (6-35): finds 35, misses 1
      fixed-step         ci.yml's "Run Gitleaks over the pull request" shell, read from the workflow: finds both
    merge-side-branch  five commits; the token is on a side branch merged into the pull request's branch
      action-window      all five fit on one page, and --first-parent still drops the side-branch commit
      fixed-step         finds it
    clean-35           a token in the BASE commit, then 35 clean commits
      fixed-step         exits 0: the base is outside base..head, and a clean range passes

The API page is emulated as ``git rev-list --reverse --date-order base..head`` (commit dates are distinct and
ascending), matching the order the endpoint returned for juniper-ml#1797. gitleaks redacts every log and report
(``--redact``); the script fails if any token appears in a file it wrote. ``--cleanup`` deletes the scratch branches
and files, expires the reflog and prunes, then proves no planted blob is left in the clone.

Usage:
    python3 util/ad-hoc/2026-11-08_reproduce_gitleaks_pr_commit_window.py --gitleaks <bin> --clone <scratch clone> --logs <dir> [--cleanup]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import string
import subprocess  # nosec B404 - fixed argv: git, gitleaks and bash, on a scratch clone
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "ci.yml"
STEP_NAME = "Run Gitleaks over the pull request (Secrets Detection)"
#: gitleaks-action v3.0.0's Scan() argv (src/gitleaks.js), in its order; --report-path and --log-opts are added per run.
ACTION_FLAGS = ("detect", "--redact", "-v", "--exit-code=2", "--report-format=sarif")
ACTION_LOG_LEVEL = "--log-level=debug"
#: GitHub's default per_page for the endpoint, and the action passes none.
PAGE = 30
EPOCH = 1_762_560_000
BRANCH_PREFIX = "scratch/pr-window-"
SCRATCH_DIR = "scratch-pr-window"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
SCANNED = re.compile(r"(\d+) commits scanned")
ALPHABET = string.ascii_letters + string.digits


def fabricated_token(step: int) -> str:
    """A GitHub-PAT-SHAPED value from a fixed-step walk over the alphabet: deterministic, never a credential."""
    return "gh" + "p_" + "".join(ALPHABET[(11 + i * step) % len(ALPHABET)] for i in range(36))


class Scratch:
    """git in the scratch clone, isolated from the user's and the system's git config, with dated commits."""

    def __init__(self, clone: Path, home: Path) -> None:
        self.clone = clone
        self.home = home
        self.tick = 0

    def env(self, **extra: str) -> dict[str, str]:
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(self.home),
            "LANG": "C",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "scratch",
            "GIT_AUTHOR_EMAIL": "scratch@example.invalid",
            "GIT_COMMITTER_NAME": "scratch",
            "GIT_COMMITTER_EMAIL": "scratch@example.invalid",
        }
        env.update(extra)
        return env

    def run_git(self, *args: str, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", "-c", "commit.gpgsign=false", "-c", "core.hooksPath=" + os.devnull, *args], cwd=self.clone, env=self.env(), input=stdin, capture_output=True, text=True, check=False)  # nosec B603 B607

    def git(self, *args: str) -> str:
        proc = self.run_git(*args)
        if proc.returncode != 0:
            raise SystemExit(f"git {' '.join(args)} failed ({proc.returncode}): {proc.stderr.strip()}")
        return proc.stdout.strip()

    def _dated(self) -> dict[str, str]:
        self.tick += 1
        stamp = f"{EPOCH + 60 * self.tick} +0000"
        return {"GIT_AUTHOR_DATE": stamp, "GIT_COMMITTER_DATE": stamp}

    def commit(self, rel: str, text: str, message: str) -> str:
        path = self.clone / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        self.git("add", "--", rel)
        proc = subprocess.run(["git", "-c", "commit.gpgsign=false", "-c", "core.hooksPath=" + os.devnull, "commit", "--no-verify", "-q", "-m", message], cwd=self.clone, env=self.env(**self._dated()), capture_output=True, text=True, check=False)  # nosec B603 B607
        if proc.returncode != 0:
            raise SystemExit(f"commit {message!r} failed: {proc.stderr.strip()}")
        return self.git("rev-parse", "HEAD")

    def merge(self, branch: str, message: str) -> str:
        proc = subprocess.run(["git", "-c", "commit.gpgsign=false", "-c", "core.hooksPath=" + os.devnull, "merge", "--no-ff", "--no-verify", "-q", "-m", message, branch], cwd=self.clone, env=self.env(**self._dated()), capture_output=True, text=True, check=False)  # nosec B603 B607
        if proc.returncode != 0:
            raise SystemExit(f"merge of {branch} failed: {proc.stderr.strip()}")
        return self.git("rev-parse", "HEAD")

    def api_page(self, base: str, head: str) -> list[str]:
        """The first page the action's unpaginated request returns: the oldest PAGE commits of base..head."""
        return self.git("rev-list", "--reverse", "--date-order", f"{base}..{head}").split()[:PAGE]


def leak_line(token: str) -> str:
    return "GITHUB_" + "TOKEN=" + token + "\n"


def build_linear(sc: Scratch, base: str, name: str, count: int, leaks: dict[int, str]) -> list[str]:
    sc.git("checkout", "-q", "-B", BRANCH_PREFIX + name, base)
    shas = []
    for i in range(1, count + 1):
        if i in leaks:
            shas.append(sc.commit(f"{SCRATCH_DIR}/leak-in-commit-{i:02d}.txt", leak_line(leaks[i]), f"scratch pr-window {name}: commit {i:02d} of {count} (planted token)"))
        else:
            shas.append(sc.commit(f"{SCRATCH_DIR}/filler-{i:02d}.txt", f"{name} filler {i:02d}\n", f"scratch pr-window {name}: commit {i:02d} of {count}"))
    return shas


def build_merge(sc: Scratch, base: str, token: str) -> dict[str, str]:
    main_line, side = BRANCH_PREFIX + "merge-side-branch", BRANCH_PREFIX + "merge-side-branch-topic"
    sc.git("checkout", "-q", "-B", main_line, base)
    p1 = sc.commit(f"{SCRATCH_DIR}/merge-p1.txt", "p1\n", "scratch pr-window merge: p1")
    sc.git("checkout", "-q", "-B", side, p1)
    s1 = sc.commit(f"{SCRATCH_DIR}/leak-on-side-branch.txt", leak_line(token), "scratch pr-window merge: s1 on the side branch (planted token)")
    sc.git("checkout", "-q", main_line)
    p2 = sc.commit(f"{SCRATCH_DIR}/merge-p2.txt", "p2\n", "scratch pr-window merge: p2")
    m = sc.merge(side, "scratch pr-window merge: merge the side branch")
    p3 = sc.commit(f"{SCRATCH_DIR}/merge-p3.txt", "p3\n", "scratch pr-window merge: p3 (head)")
    return {"p1": p1, "s1": s1, "p2": p2, "m": m, "p3": p3}


def findings(report: Path) -> list[dict[str, str]]:
    if not report.is_file():
        return []
    sarif = json.loads(report.read_text(encoding="utf-8"))
    out = []
    for res in sarif["runs"][0].get("results", []):
        loc = res["locations"][0]["physicalLocation"]
        out.append({"commit": res["partialFingerprints"]["commitSha"], "file": loc["artifactLocation"]["uri"], "rule": res["ruleId"], "line": loc["region"]["startLine"]})
    return out


def run_action_window(sc: Scratch, gitleaks: Path, logs: Path, label: str, first: str, last: str) -> dict:
    report = logs / f"{label}.sarif.json"
    argv = [str(gitleaks), *ACTION_FLAGS, f"--report-path={report}", ACTION_LOG_LEVEL, f"--log-opts=--no-merges --first-parent {first}^..{last}"]
    proc = subprocess.run(argv, cwd=sc.clone, env=sc.env(), capture_output=True, text=True, check=False)  # nosec B603
    return _record(logs, label, argv, proc, report)


def run_fixed_step(sc: Scratch, gitleaks: Path, logs: Path, label: str, base: str, head: str, pr_commits: int) -> dict:
    steps = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]["security"]["steps"]
    step = next(s for s in steps if s.get("name") == STEP_NAME)
    if "shell" in step:
        raise SystemExit(f"{STEP_NAME!r} names a shell; this rehearsal assumes the default `bash -e`")
    summary = logs / f"{label}.step_summary.txt"
    # GitHub sets RUNNER_TEMP for every step; the scan writes its git filter there.
    runner_temp = logs / f"{label}.runner-temp"
    runner_temp.mkdir(exist_ok=True)
    env = sc.env(PATH=f"{gitleaks.parent}{os.pathsep}{os.environ.get('PATH', '/usr/bin:/bin')}", BASE_SHA=base, HEAD_SHA=head, PR_COMMITS=str(pr_commits), GITHUB_STEP_SUMMARY=str(summary), RUNNER_TEMP=str(runner_temp))
    reports_dir = sc.clone / "reports"
    if reports_dir.exists():
        shutil.rmtree(reports_dir)
    proc = subprocess.run(["bash", "-e", "-c", str(step["run"])], cwd=sc.clone, env=env, capture_output=True, text=True, check=False)  # nosec B603 B607
    report = logs / f"{label}.sarif.json"
    produced = sc.clone / "reports" / "security" / "gitleaks.sarif"
    if produced.is_file():
        shutil.move(str(produced), str(report))
    shutil.rmtree(reports_dir, ignore_errors=True)
    return _record(logs, label, ["bash", "-e", "-c", f"<ci.yml: {STEP_NAME}>"], proc, report)


def _record(logs: Path, label: str, argv: list[str], proc: subprocess.CompletedProcess[str], report: Path) -> dict:
    text = ANSI.sub("", (proc.stdout or "") + (proc.stderr or ""))
    (logs / f"{label}.log").write_text("$ " + " ".join(argv) + "\n" + text + f"\n[exit {proc.returncode}]\n", encoding="utf-8")
    scanned = SCANNED.search(text)
    executed = next((line.split("executing: ", 1)[1] for line in text.splitlines() if "executing: " in line), None)
    return {"label": label, "exit": proc.returncode, "commits_scanned": int(scanned.group(1)) if scanned else None, "git_command": executed, "findings": findings(report), "log": str(logs / f"{label}.log")}


def check(result: dict, names: dict[str, str], exit_code: int, found: set[str], step_line: str | None = None) -> None:
    files = {Path(f["file"]).name for f in result["findings"]}
    result["finding_commits"] = sorted({names.get(f["commit"], f["commit"]) for f in result["findings"]})
    result["finding_files"] = sorted(files)
    result["expected"] = {"exit": exit_code, "finding_files": sorted(found), **({"step_line": step_line} if step_line else {})}
    ok = result["exit"] == exit_code and files == found
    if step_line is not None:
        ok = ok and step_line in Path(result["log"]).read_text(encoding="utf-8")
    result["verdict"] = "AS EXPECTED" if ok else "UNEXPECTED"


def files_holding_a_token(logs: Path, tokens: list[str]) -> list[str]:
    """Every file this run wrote that holds a planted token in clear: gitleaks' --redact must leave none."""
    return sorted(p.name for p in logs.iterdir() if p.is_file() and any(t in p.read_text(encoding="utf-8", errors="replace") for t in tokens))


def scratch_branches(sc: Scratch) -> list[str]:
    return sc.git("for-each-ref", "--format=%(refname:short)", "refs/heads/" + BRANCH_PREFIX + "*").split()


def cleanup(sc: Scratch, base: str, planted: list[str]) -> dict:
    """Delete the scratch branches and files, expire the reflog and prune; report whether any planted blob survived."""
    sc.git("checkout", "-q", "--detach", base)
    for ref in scratch_branches(sc):
        sc.git("branch", "-q", "-D", ref)
    shutil.rmtree(sc.clone / SCRATCH_DIR, ignore_errors=True)
    shutil.rmtree(sc.clone / "reports", ignore_errors=True)
    sc.git("reflog", "expire", "--expire=now", "--all")
    sc.git("gc", "--prune=now", "--quiet")
    present = {}
    for token in planted:
        # hash-object without -w computes the blob id of a planted file's content and writes nothing.
        blob = sc.run_git("hash-object", "--stdin", stdin=leak_line(token)).stdout.strip()
        present[blob] = sc.run_git("cat-file", "-e", blob).returncode == 0
    return {"branches_left": scratch_branches(sc), "planted_blob_still_present": present}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gitleaks", type=Path, required=True, help="a gitleaks 8.24.3 binary")
    ap.add_argument("--clone", type=Path, required=True, help="a scratch clone of juniper-data with NO remote")
    ap.add_argument("--logs", type=Path, required=True, help="directory for the logs, reports and summary.json")
    # Not HEAD: a run leaves the clone on a scratch branch, and a second run would then build on the first one's head.
    ap.add_argument("--base", default="main", help="the base commit to branch from (default: the clone's main)")
    ap.add_argument("--cleanup", action="store_true", help="delete the scratch branches and planted files afterwards, and prune")
    args = ap.parse_args()
    clone, logs = args.clone.resolve(), args.logs.resolve()
    logs.mkdir(parents=True, exist_ok=True)
    sc = Scratch(clone, logs)
    if sc.git("remote"):
        raise SystemExit(f"{clone} has a remote; this script only runs in a clone that cannot be pushed")
    base = sc.git("rev-parse", "--verify", args.base + "^{commit}")
    tokens = {"A": fabricated_token(29), "B": fabricated_token(37), "C": fabricated_token(41), "D": fabricated_token(43)}
    names: dict[str, str] = {}
    results = []

    linear = build_linear(sc, base, "linear-35", 35, {1: tokens["A"], 35: tokens["B"]})
    names.update({sha: f"linear-35 commit {i:02d}" for i, sha in enumerate(linear, start=1)})
    page = sc.api_page(base, linear[-1])
    if page != linear[:PAGE]:
        raise SystemExit("the emulated first API page is not commits 1-30 of the linear branch")
    r = run_action_window(sc, args.gitleaks, logs, "linear-35.action-window", page[0], page[-1])
    check(r, names, 2, {"leak-in-commit-01.txt"})
    results.append(r)
    r = run_action_window(sc, args.gitleaks, logs, "linear-35.newest-30-window", linear[-PAGE], linear[-1])
    check(r, names, 2, {"leak-in-commit-35.txt"})
    results.append(r)
    r = run_fixed_step(sc, args.gitleaks, logs, "linear-35.fixed-step", base, linear[-1], 35)
    check(r, names, 1, {"leak-in-commit-01.txt", "leak-in-commit-35.txt"}, "Scanning 35 commit(s)")
    results.append(r)

    merge = build_merge(sc, base, tokens["C"])
    names.update({sha: f"merge-side-branch {k}" for k, sha in merge.items()})
    page = sc.api_page(base, merge["p3"])
    r = run_action_window(sc, args.gitleaks, logs, "merge-side-branch.action-window", page[0], page[-1])
    r["api_page"] = [names[s] for s in page]
    check(r, names, 0, set())
    results.append(r)
    r = run_fixed_step(sc, args.gitleaks, logs, "merge-side-branch.fixed-step", base, merge["p3"], 5)
    check(r, names, 1, {"leak-on-side-branch.txt"}, "Scanning 5 commit(s)")
    results.append(r)

    sc.git("checkout", "-q", "-B", BRANCH_PREFIX + "clean-35", base)
    pr_base = sc.commit(f"{SCRATCH_DIR}/leak-in-the-base.txt", leak_line(tokens["D"]), "scratch pr-window clean-35: the BASE commit (planted token, outside the range)")
    names[pr_base] = "clean-35 base"
    clean = build_linear(sc, pr_base, "clean-35", 35, {})
    r = run_fixed_step(sc, args.gitleaks, logs, "clean-35.fixed-step", pr_base, clean[-1], 35)
    check(r, names, 0, set(), "Scanning 35 commit(s)")
    results.append(r)

    leaked = files_holding_a_token(logs, list(tokens.values()))
    summary = {
        "gitleaks": str(args.gitleaks),
        "base": base,
        "workflow": str(WORKFLOW),
        "step": STEP_NAME,
        "results": results,
        "files_holding_an_unredacted_token": leaked,
    }
    if args.cleanup:
        summary["cleanup"] = cleanup(sc, base, list(tokens.values()))
    else:
        sc.git("checkout", "-q", "--detach", base)
    (logs / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    leaked = files_holding_a_token(logs, list(tokens.values()))  # again, now that summary.json exists
    for r in results:
        print(f"{r['verdict']:<11} {r['label']:<32} exit={r['exit']} commits_scanned={r['commits_scanned']} finding_files={r['finding_files']}")
    print(f"unredacted token in any written file: {leaked or 'none'}")
    if args.cleanup:
        print(f"cleanup: {summary['cleanup']}")
    bad = [r["label"] for r in results if r["verdict"] != "AS EXPECTED"]
    left = args.cleanup and (summary["cleanup"]["branches_left"] or any(summary["cleanup"]["planted_blob_still_present"].values()))
    return 1 if bad or leaked or left else 0


if __name__ == "__main__":
    sys.exit(main())
