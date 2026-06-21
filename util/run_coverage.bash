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
#    Reproduce the CI coverage gates locally (full suite): aggregate
#    (--cov-fail-under) AND the per-module floor (scripts/check_module_coverage.py).
#    Mirrors .github/workflows/ci.yml so a developer can verify both gates before
#    pushing. Full suite by design; use plain pytest for a subset.
#
# Usage:
#    bash util/run_coverage.bash                          # full suite + both gates
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

echo "==> Coverage (reproduces CI gates: ${COVERAGE_FAIL_UNDER}% aggregate + per-module floor) — ${REPO_ROOT}"

# ── Reproduce the CI coverage sequence (keep in sync with .github/workflows/ci.yml) ──
python -m pytest -m "unit and not slow" juniper_data/tests/unit --cov=juniper_data --cov-report=term-missing
python -m coverage report --fail-under="${COVERAGE_FAIL_UNDER}"
python scripts/check_module_coverage.py
# ─────────────────────────────────────────────────────────────────────────────────────
