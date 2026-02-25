#!/usr/bin/env python3
#####################################################################################################################################################################################################
# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     check_module_coverage.py
# Author:        Paul Calnon
# Version:       0.5.0
#
# Date Created:  2026-02-24
# Last Modified: 2026-02-24
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2026 Paul Calnon
#
# Description:
#    Enforces per-module and aggregate code coverage thresholds.
#    Reads coverage.py JSON output and verifies that:
#      - Each source module individually meets the per-module threshold (85%)
#      - Aggregate coverage meets the application threshold (95%)
#      - No test files are included in coverage metrics
#
# Usage:
#    # Check coverage from existing .coverage file (CI mode)
#    python scripts/check_module_coverage.py
#
#    # Run tests first, then check (pre-commit mode)
#    python scripts/check_module_coverage.py --run-tests
#
# References:
#    - RD-005: Reconcile Coverage Metrics
#    - RD-007: Improve Coverage for Low-Coverage Modules
#####################################################################################################################################################################################################

import json
import subprocess
import sys
from pathlib import Path

# Coverage thresholds
MODULE_FAIL_UNDER = 85.0
AGGREGATE_FAIL_UNDER = 95.0

# Coverage JSON output path
COVERAGE_JSON = Path("coverage.json")


def run_tests():
    """Run pytest with coverage and generate JSON report."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "juniper_data/tests/",
        "--cov=juniper_data",
        "--cov-report=json:coverage.json",
        "-q",
    ]
    result = subprocess.run(cmd)
    if result.returncode not in (0, 5):  # 5 = no tests collected
        print(f"ERROR: Tests failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def generate_json_from_coverage_data():
    """Generate coverage.json from an existing .coverage file."""
    result = subprocess.run(
        [sys.executable, "-m", "coverage", "json", "-o", "coverage.json"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("ERROR: No .coverage file found. Run tests first or use --run-tests")
        if result.stderr:
            print(result.stderr.strip())
        sys.exit(1)


def check_coverage():
    """Read coverage JSON and enforce per-module and aggregate thresholds."""
    if not COVERAGE_JSON.exists():
        print("ERROR: coverage.json not found")
        sys.exit(1)

    data = json.loads(COVERAGE_JSON.read_text())

    aggregate = data["totals"]["percent_covered"]

    # Check for test file leaks
    test_files = [f for f in data["files"] if "/tests/" in f]
    if test_files:
        print(f"ERROR: {len(test_files)} test file(s) found in coverage metrics:")
        for f in test_files:
            print(f"  {f}")
        sys.exit(1)

    # Check per-module coverage
    failures = []
    for filepath, file_data in sorted(data["files"].items()):
        pct = file_data["summary"]["percent_covered"]
        if pct < MODULE_FAIL_UNDER:
            failures.append((filepath, pct))

    # Report results
    exit_code = 0

    print(f"Aggregate coverage: {aggregate:.2f}% (threshold: {AGGREGATE_FAIL_UNDER}%)")

    if aggregate < AGGREGATE_FAIL_UNDER:
        print(f"FAIL: Aggregate coverage {aggregate:.2f}% < {AGGREGATE_FAIL_UNDER}%")
        exit_code = 1

    if failures:
        print(f"\nFAIL: {len(failures)} module(s) below {MODULE_FAIL_UNDER}% coverage:")
        for filepath, pct in failures:
            print(f"  {filepath}: {pct:.2f}%")
        exit_code = 1

    if exit_code == 0:
        print(f"PASS: All {len(data['files'])} modules >= {MODULE_FAIL_UNDER}%, aggregate >= {AGGREGATE_FAIL_UNDER}%")

    return exit_code


def main():
    if "--run-tests" in sys.argv:
        run_tests()
    else:
        generate_json_from_coverage_data()

    exit_code = check_coverage()

    # Cleanup
    COVERAGE_JSON.unlink(missing_ok=True)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
