#!/usr/bin/env bash
#####################################################################################################
# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     util/run_coverage.bash
# Author:        Paul Calnon
# Version:       0.2.0
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2026 Paul Calnon
#
# Description:
#    Reproduce CI's unit lane and both of its coverage gates locally: the same
#    `pytest -m "unit and not slow" juniper_data/tests/unit --cov=juniper_data` run,
#    then the aggregate gate (`coverage report --fail-under`), then
#    scripts/check_module_coverage.py, which lists modules below 85% as WARN without
#    failing, then CI's per-file gate (juniper-coverage-gap-map --enforce: 90% statement
#    coverage per file, 95% pooled per sub-module). The per-file gate reads
#    reports/coverage.json, which the pytest run writes. It needs juniper-ci-tools,
#    which is not a project dependency, so without that tool this stops before the
#    tests. Use plain pytest for a subset.
#
# Usage:
#    bash util/run_coverage.bash                          # unit lane + both coverage gates
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

# CI's per-file gate needs juniper-ci-tools, which pyproject.toml does not install: check for it
# before the tests rather than fail after them.
if ! command -v juniper-coverage-gap-map >/dev/null 2>&1; then
    echo "ERROR: juniper-coverage-gap-map not found; CI's per-file gate needs it." >&2
    echo "       Install CI's pin: pip install \"juniper-ci-tools>=0.9.0,<0.10.0\"" >&2
    exit 2
fi

echo "==> Coverage (CI's unit lane + ${COVERAGE_FAIL_UNDER}% aggregate gate + per-file gate) — ${REPO_ROOT}"

# ── CI's unit-lane pytest and both coverage gates (keep in sync with .github/workflows/ci.yml) ──
# scripts/check_module_coverage.py is not CI's. It runs before the per-file gate so that its
# listing prints even when that gate fails.
mkdir -p reports
python -m pytest -m "unit and not slow" juniper_data/tests/unit --cov=juniper_data --cov-report=term-missing --cov-report=json:reports/coverage.json
python -m coverage report --fail-under="${COVERAGE_FAIL_UNDER}"
python scripts/check_module_coverage.py
juniper-coverage-gap-map --coverage-json reports/coverage.json --enforce
# ─────────────────────────────────────────────────────────────────────────────────────
