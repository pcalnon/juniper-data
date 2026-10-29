#!/usr/bin/env bash
#####################################################################################################
# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     util/run_coverage.bash
# Author:        Paul Calnon
# Version:       0.1.0
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2026 Paul Calnon
#
# Description:
#    Reproduce CI's unit lane and its aggregate coverage gate locally: the same
#    `pytest -m "unit and not slow" juniper_data/tests/unit --cov=juniper_data` run,
#    then `coverage report --fail-under`, then scripts/check_module_coverage.py,
#    which lists modules below 85% as WARN without failing. It does NOT run CI's
#    per-file gate (juniper-coverage-gap-map --enforce: 90% statement coverage per
#    file, 95% pooled per sub-module); docs/REFERENCE.md § Coverage Gates has the
#    commands for that. Use plain pytest for a subset.
#
# Usage:
#    bash util/run_coverage.bash                          # unit lane + aggregate gate
#    make coverage                                        # equivalent wrapper
#    COVERAGE_FAIL_UNDER=90 bash util/run_coverage.bash   # override the aggregate gate
#
# References:
#    - https://pytest-cov.readthedocs.io/
#####################################################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

COVERAGE_FAIL_UNDER="${COVERAGE_FAIL_UNDER:-80}"

echo "==> Coverage (CI's unit lane + ${COVERAGE_FAIL_UNDER}% aggregate gate; not the per-file gate) — ${REPO_ROOT}"

# ── CI's unit-lane pytest and aggregate gate (keep in sync with .github/workflows/ci.yml) ──
python -m pytest -m "unit and not slow" juniper_data/tests/unit --cov=juniper_data --cov-report=term-missing
python -m coverage report --fail-under="${COVERAGE_FAIL_UNDER}"
python scripts/check_module_coverage.py
# ─────────────────────────────────────────────────────────────────────────────────────
