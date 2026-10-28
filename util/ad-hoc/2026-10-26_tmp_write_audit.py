#!/usr/bin/env python3
"""
Run pytest with every write under /tmp or /var/tmp BLOCKED, and name the test that tried.

Project: juniper-data
Sub-Project: ad-hoc tooling
Author: Paul Calnon
Created: 2026-10-26
Status: ad-hoc — investigation
Retire when: every Juniper suite runs clean under it, or a permanent gate replaces it
Related: the 2026-10-26 /tmp remediation (owner directive: no file is ever written under /tmp);
         juniper-ml notes/JUNIPER_2026-10-26_JUNIPER-ECOSYSTEM_REPO-TMP-DIRECTORY-CONVENTION.md

Why
---
A ``/tmp/...`` literal in a test is not by itself a write. ``Settings(storage_path="/tmp/x")``
writes only if the code under test creates that directory, and reading the code answers that
for one call path at a time. Running the suite answers it for all of them.

This installs a ``sys.addaudithook`` before pytest starts. The hook RAISES ``PermissionError``
on ``os.mkdir``, an open for writing, the target of a rename / replace / link / symlink / copy /
move, a truncate, and a delete, whenever the path is under a prohibited root. A measurement run
therefore writes nothing there, and every attempt is recorded against the running test. A test
that fails here only because of the block is exactly a test that writes under /tmp.

Blind spot: an audit hook sees this interpreter only. A subprocess (bash, git, a second python)
is not covered, so every subprocess whose argv or cwd names a prohibited root is RECORDED (not
blocked) for a reader to check by hand.

Usage
-----
    TMPDIR=<repo>/tmp python3 util/ad-hoc/2026-10-26_tmp_write_audit.py [--report FILE] [--show-module PKG] -- <pytest args>

It refuses to start when ``tempfile.gettempdir()`` is itself under a prohibited root: pytest's
``tmp_path`` would then be blocked in every test and the report would be noise.

Exit: 3 when any write under a prohibited root was attempted; otherwise pytest's exit code.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile

PROHIBITED_ROOTS = ("/tmp", "/var/tmp", "/private/tmp", "/private/var/tmp")
# A prohibited root as a path token inside free text (an argv element, a shell command):
# not preceded by a path or word character, so ``<repo>/tmp`` and ``$TMPDIR/tmp`` do not match.
PROHIBITED_IN_TEXT = re.compile(r"(?<![\w.~/$])/(?:private/)?(?:var/)?tmp(?![\w.-])")
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
WRITE_MODE_CHARS = frozenset("wax+")
# audit event -> positions of the path arguments that the operation creates, changes or removes
PATH_ARGS = {
    "os.mkdir": (0,),
    "os.rename": (0, 1),
    "os.link": (1,),
    "os.symlink": (1,),
    "os.truncate": (0,),
    "os.remove": (0,),
    "os.rmdir": (0,),
    "shutil.copyfile": (1,),
    "shutil.copytree": (1,),
    "shutil.move": (1,),
    "shutil.rmtree": (0,),
}
SUBPROCESS_EVENTS = ("subprocess.Popen", "os.system", "os.posix_spawn")

ATTEMPTS: list[dict] = []
SUBPROCESS_NOTES: list[dict] = []
CURRENT_TEST = ["<startup>"]


def as_abs_path(obj: object) -> str | None:
    """An absolute path for a path-like audit argument; None for a file descriptor or anything else."""
    if obj is None or isinstance(obj, int):
        return None
    try:
        text = os.fsdecode(os.fspath(obj))  # type: ignore[arg-type]
    except TypeError:
        return None
    return os.path.abspath(text)


def is_prohibited(path: str) -> bool:
    return any(path == root or path.startswith(root + "/") for root in PROHIBITED_ROOTS)


def names_prohibited_root(value: object) -> bool:
    """True when a subprocess argv element names a prohibited root as a path.

    A token match, not a substring one: ``<repo>/tmp/x`` contains ``/tmp`` and is exactly
    where a temp file belongs, so a substring test reports every compliant subprocess.
    """
    if isinstance(value, (list, tuple)):
        return any(names_prohibited_root(item) for item in value)
    if isinstance(value, (str, bytes, os.PathLike)):
        return PROHIBITED_IN_TEXT.search(os.fsdecode(os.fspath(value))) is not None
    return False


def opens_for_write(args: tuple) -> bool:
    mode = args[1] if len(args) > 1 else None
    flags = args[2] if len(args) > 2 else None
    if isinstance(flags, int) and flags & WRITE_FLAGS:
        return True
    return isinstance(mode, str) and bool(WRITE_MODE_CHARS.intersection(mode))


def audit_hook(event: str, args: tuple) -> None:
    if event in SUBPROCESS_EVENTS:
        cwd = args[2] if event == "subprocess.Popen" and len(args) > 2 else None
        argv = args[1] if event in ("subprocess.Popen", "os.posix_spawn") and len(args) > 1 else args[:1]
        cwd_path = as_abs_path(cwd)
        if names_prohibited_root(argv) or (cwd_path is not None and is_prohibited(cwd_path)):
            SUBPROCESS_NOTES.append({"test": CURRENT_TEST[0], "event": event, "argv": repr(argv)[:400], "cwd": repr(cwd)})
        return
    if event == "open":
        if not opens_for_write(args):
            return
        positions: tuple = (0,)
    elif event in PATH_ARGS:
        positions = PATH_ARGS[event]
    else:
        return
    for position in positions:
        if position >= len(args):
            continue
        path = as_abs_path(args[position])
        if path is not None and is_prohibited(path):
            ATTEMPTS.append({"test": CURRENT_TEST[0], "event": event, "path": path})
            raise PermissionError(f"tmp-write-audit blocked {event} under a prohibited root: {path}")


class AuditReporter:
    """pytest plugin: tracks the running test and prints what the hook saw."""

    def __init__(self, show_modules: list[str]) -> None:
        self.show_modules = show_modules

    def pytest_report_header(self, config):
        return [
            f"tmp-write-audit: blocking writes under {', '.join(PROHIBITED_ROOTS)}",
            f"tmp-write-audit: tempfile.gettempdir() = {tempfile.gettempdir()}",
        ]

    def pytest_runtest_logstart(self, nodeid, location):
        CURRENT_TEST[0] = nodeid

    def pytest_runtest_logfinish(self, nodeid, location):
        CURRENT_TEST[0] = "<between tests>"

    def pytest_terminal_summary(self, terminalreporter):
        terminalreporter.section("tmp-write-audit")
        # Which checkout each package under test came from, read from sys.modules AFTER the run.
        # Never import here or earlier: importing before collection inserts the test rootdir pins
        # whatever the environment resolves first (an installed wheel, a primary's editable
        # install), which is exactly the mistake this line exists to catch.
        for name in self.show_modules:
            module = sys.modules.get(name)
            where = os.path.dirname(module.__file__ or "?") if module is not None else "NOT IMPORTED by this run"
            terminalreporter.write_line(f"{name} imported from {where}")
        terminalreporter.write_line(f"blocked write attempts: {len(ATTEMPTS)}")
        for row in ATTEMPTS:
            terminalreporter.write_line(f"  BLOCKED {row['event']:<16} {row['path']}  <- {row['test']}")
        terminalreporter.write_line(f"subprocesses naming a prohibited root (not blocked; read by hand): {len(SUBPROCESS_NOTES)}")
        for row in SUBPROCESS_NOTES:
            terminalreporter.write_line(f"  NOTE {row['event']} argv={row['argv']} cwd={row['cwd']}  <- {row['test']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", default=None, help="also write the findings as JSON to this file")
    parser.add_argument("--show-module", action="append", default=[], help="after the run, print where this package was imported from (repeatable; never imports it)")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER, help="arguments for pytest, after --")
    args = parser.parse_args(argv)
    pytest_args = args.pytest_args[1:] if args.pytest_args[:1] == ["--"] else args.pytest_args

    temp_root = os.path.realpath(tempfile.gettempdir())
    if is_prohibited(temp_root):
        print(f"refusing to run: tempfile.gettempdir() is {temp_root}. Set TMPDIR to the repo's tmp/ first.", file=sys.stderr)
        return 2
    if args.report and is_prohibited(os.path.abspath(args.report)):
        print(f"refusing to run: --report {args.report} is under a prohibited root.", file=sys.stderr)
        return 2

    import pytest

    sys.addaudithook(audit_hook)
    exit_code = int(pytest.main(pytest_args, plugins=[AuditReporter(args.show_module)]))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump({"pytest_exit": exit_code, "blocked": ATTEMPTS, "subprocess_notes": SUBPROCESS_NOTES}, handle, indent=2)
    return 3 if ATTEMPTS else exit_code


if __name__ == "__main__":
    sys.exit(main())
