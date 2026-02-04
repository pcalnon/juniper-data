# JuniperData Test Suite & CI/CD Enhancement Development Plan

**Project**: JuniperData - Dataset Generation Service
**Plan Version**: 1.0
**Created**: 2026-02-03
**Author**: Claude (Opus 4.5)
**Status**: Draft - Pending Approval

---

## Executive Summary

This development plan consolidates findings from two independent audits of the JuniperData test suite and CI/CD pipeline. The analysis validates that the core testing infrastructure is sound (207 tests, 100% coverage, all passing), but identifies 21 actionable improvements across security, code quality, and CI/CD configuration.

### Current State Summary

| Metric | Value | Assessment |
|--------|-------|------------|
| Total Tests | 207 (183 unit, 24 integration) | **Excellent** |
| Code Coverage | 100% | **Excellent** |
| Tests Passing | 207/207 (100%) | **Excellent** |
| Critical Issues | 3 | **Requires Attention** |
| High Priority Issues | 5 | **Requires Attention** |
| Medium Priority Issues | 8 | **Moderate** |
| Low Priority Issues | 5 | **Minor** |

---

## Table of Contents

1. [Consolidated Issue Analysis](#1-consolidated-issue-analysis)
2. [Validation of Audit Findings](#2-validation-of-audit-findings)
3. [Prioritized Enhancement Plan](#3-prioritized-enhancement-plan)
4. [Implementation Phases](#4-implementation-phases)
5. [Effort Estimates](#5-effort-estimates)
6. [Risk Assessment](#6-risk-assessment)
7. [Success Criteria](#7-success-criteria)
8. [Appendices](#8-appendices)

---

## 1. Consolidated Issue Analysis

### 1.1 Critical Issues (Immediate Action Required)

These issues represent security risks or could result in false confidence in test results.

| ID | Issue | Location | Description | Risk |
|----|-------|----------|-------------|------|
| C-01 | Silent Test Pass | `test_main.py:32-38` | `try/except ImportError: pass` allows test to pass without executing assertions | Test reliability compromised |
| C-02 | Security Scans Non-Blocking | `ci.yml:313` | Bandit uses `\|\| true` which masks all security findings | Security vulnerabilities undetected |
| C-03 | Missing Dependabot | `.github/` | No `dependabot.yml` configuration exists | No automated security updates for dependencies |

### 1.2 High Priority Issues

These issues affect code quality enforcement and security posture.

| ID | Issue | Location | Description | Impact |
|----|-------|----------|-------------|--------|
| H-01 | Tests Excluded from Flake8 | `.pre-commit-config.yaml:126` | `exclude: ^juniper_data/tests/` bypasses linting for test code | 7 unused imports undetected; style inconsistencies allowed |
| H-02 | Tests Excluded from MyPy | `.pre-commit-config.yaml:139` | `files: ^juniper_data/(?!tests).*\.py$` bypasses type checking for tests | Type errors in tests undetected |
| H-03 | pip-audit Warnings Non-Fatal | `ci.yml:329` | Uses `\|\| echo "::warning::"` allowing vulnerable dependencies | Dependency vulnerabilities may ship to production |
| H-04 | Pytest Warning Suppression | `pyproject.toml:119` | `-p no:warnings` hides all deprecation warnings | Breaking changes from dependencies go unnoticed |
| H-05 | Excessive Flake8 Ignores | `.pre-commit-config.yaml:118` | Ignores E722 (bare except), F401 (unused imports), and others | Silent failures, dead code accumulation |

### 1.3 Medium Priority Issues

These issues affect maintainability and best practices compliance.

| ID | Issue | Location | Description | Impact |
|----|-------|----------|-------------|--------|
| M-01 | MyPy Type Errors | Production code | 4 type errors in numpy savez/savez_compressed calls | Type safety compromised |
| M-02 | No CodeQL Analysis | `.github/workflows/` | Deep semantic security analysis not configured | Potential vulnerabilities undetected |
| M-03 | No Coverage Upload | `ci.yml` | Coverage reports not uploaded to external service | No historical coverage tracking |
| M-04 | Slow Test Infrastructure | `ci.yml:141, 254` | Slow tests excluded but no separate job for them | Slow tests will be silently skipped in future |
| M-05 | Line Length 512 | `.pre-commit-config.yaml:92` | Excessively permissive line length | Code readability concerns |
| M-06 | No Performance Tests | `pyproject.toml:128` | `performance` marker defined but no tests use it | No regression detection for performance |
| M-07 | Continue on Collection Errors | `pyproject.toml:122` | `--continue-on-collection-errors` may hide import failures | Broken tests may go unnoticed |
| M-08 | SARIF Upload Continue on Error | `ci.yml:320` | `continue-on-error: true` for SARIF upload | GitHub Security integration failures ignored |

### 1.4 Low Priority Issues

These issues represent best practice refinements.

| ID | Issue | Location | Description | Impact |
|----|-------|----------|-------------|--------|
| L-01 | Action Version Pinning | `ci.yml` | Uses `@v4`, `@v5` instead of SHA pins | Minor supply chain risk |
| L-02 | No pyupgrade Hook | `.pre-commit-config.yaml` | Missing Python syntax modernization | Older syntax patterns may persist |
| L-03 | No shellcheck Hook | `.pre-commit-config.yaml` | No shell script linting | Shell script issues undetected |
| L-04 | Python 3.14 Not Fully Supported | `.pre-commit-config.yaml:33` | Black doesn't support py314 target | Minor inconsistency |
| L-05 | No Documentation Build Step | `ci.yml` | Documentation not validated in CI | Documentation drift possible |

---

## 2. Validation of Audit Findings

### 2.1 Verified Issues

The following issues from both audit reports were independently verified:

| Issue | Audit Report | Verification Method | Status |
|-------|--------------|---------------------|--------|
| Silent `pass` in test_main.py | Both | Code inspection (lines 32-38) | **CONFIRMED** |
| No dependabot.yml | Both | `find .github -name "dependabot*"` | **CONFIRMED** |
| 7 unused imports in tests | Second audit | `flake8 --select=F401` | **CONFIRMED** |
| 4 MyPy type errors | Second audit | `mypy juniper_data` | **CONFIRMED** |
| Bandit `\|\| true` | Both | Code inspection (ci.yml:313) | **CONFIRMED** |
| pip-audit warning handling | Both | Code inspection (ci.yml:329) | **CONFIRMED** |
| Tests excluded from flake8 | Both | Code inspection | **CONFIRMED** |
| Tests excluded from mypy | Both | Code inspection | **CONFIRMED** |
| Warning suppression | Both | pyproject.toml:119 | **CONFIRMED** |
| No slow tests exist | Both | `pytest -m slow --collect-only` | **CONFIRMED** |
| 100% coverage | Both | `pytest --cov` | **CONFIRMED** |

### 2.2 Analysis Assumptions Validated

| Assumption | Validation | Result |
|------------|------------|--------|
| Test suite is comprehensive | 207 tests across all components | **VALID** |
| Coverage threshold is enforced | `--cov-fail-under=80` in CI | **VALID** |
| Security scans run in CI | Gitleaks, Bandit, pip-audit configured | **VALID** |
| Pre-commit hooks are comprehensive | 16 hooks configured | **VALID** |
| Integration tests conditional | Only on PR/main/develop | **VALID** |

### 2.3 Best Practices Compliance Review

| Practice | Current State | Recommendation |
|----------|---------------|----------------|
| Security scanning | Configured but non-blocking | Make blocking for high/medium severity |
| Dependency management | No automated updates | Add Dependabot |
| Code coverage | 100% achieved, 80% threshold | Maintain current approach |
| Type checking | Partial (excludes tests) | Extend to tests with relaxed rules |
| Linting | Partial (excludes tests) | Extend to tests |
| Secret detection | Gitleaks configured | Adequate |
| Test organization | Clear unit/integration split | Adequate |

---

## 3. Prioritized Enhancement Plan

### 3.1 Priority Matrix

```
                    IMPACT
                    High    Medium    Low
               ┌─────────┬─────────┬─────────┐
        High   │  C-01   │  H-01   │  L-01   │
               │  C-02   │  H-02   │  L-02   │
URGENCY        │  C-03   │  H-05   │         │
               ├─────────┼─────────┼─────────┤
        Medium │  H-03   │  M-01   │  L-03   │
               │  H-04   │  M-02   │  L-04   │
               │         │  M-03   │  L-05   │
               ├─────────┼─────────┼─────────┤
        Low    │         │  M-04   │         │
               │         │  M-05   │         │
               │         │  M-06   │         │
               └─────────┴─────────┴─────────┘
```

### 3.2 Recommended Fix Order

1. **Phase 1 (Critical)**: C-01, C-02, C-03
2. **Phase 2 (High Priority)**: H-01, H-02, H-03, H-04, H-05
3. **Phase 3 (Medium Priority)**: M-01, M-02, M-03, M-04
4. **Phase 4 (Low Priority)**: Remaining items as time permits

---

## 4. Implementation Phases

### Phase 1: Critical Security & Reliability Fixes

**Objective**: Eliminate false confidence in test results and security posture.

#### Task 1.1: Fix Silent Test Pass (C-01)

**File**: `juniper_data/tests/unit/test_main.py`

**Current Code** (lines 32-38):
```python
try:
    importlib.reload(main_module)
    result = main_module.main()
    assert result == 1
    mock_print.assert_called()
except ImportError:
    pass  # <-- Silent pass
```

**Recommended Fix**:
```python
@pytest.mark.skipif(
    not importlib.util.find_spec("uvicorn"),
    reason="Test requires uvicorn to be installed to test import error handling"
)
def test_main_import_error_uvicorn_not_installed(self) -> None:
    """Test main returns 1 when uvicorn is not installed."""
    # ... existing mock setup ...

    importlib.reload(main_module)
    result = main_module.main()
    assert result == 1
    mock_print.assert_called()
```

**Alternative Fix** (if test must handle missing uvicorn):
```python
def test_main_import_error_uvicorn_not_installed(self) -> None:
    """Test main returns 1 when uvicorn is not installed."""
    # ... existing mock setup ...

    try:
        importlib.reload(main_module)
        result = main_module.main()
        assert result == 1
        mock_print.assert_called()
    except ImportError as e:
        pytest.skip(f"Cannot test uvicorn import error: {e}")
```

**Verification**: `pytest juniper_data/tests/unit/test_main.py::TestMain::test_main_import_error_uvicorn_not_installed -v`

---

#### Task 1.2: Make Security Scans Blocking (C-02)

**File**: `.github/workflows/ci.yml`

**Current Code** (line 313):
```yaml
bandit -r juniper_data -f sarif -o reports/security/bandit.sarif || true
```

**Recommended Fix**:
```yaml
# Run Bandit with exit-zero for SARIF generation, then run blocking check
- name: Run Bandit (SAST) -> SARIF
  run: |
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║       JuniperData - Bandit Security Scan                   ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    # Generate SARIF report (always succeeds for upload)
    bandit -r juniper_data -f sarif -o reports/security/bandit.sarif --exit-zero
    # Blocking check - fail on medium or higher severity
    bandit -r juniper_data -ll -ii --severity-level medium
```

**Also update pip-audit** (line 329):
```yaml
# Current
pip-audit -r reports/security/pip-freeze.txt || echo "::warning::Vulnerabilities found in dependencies"

# Recommended
pip-audit -r reports/security/pip-freeze.txt --strict
```

**Verification**: Trigger CI with known vulnerability to confirm blocking behavior.

---

#### Task 1.3: Add Dependabot Configuration (C-03)

**Create**: `.github/dependabot.yml`

```yaml
# Dependabot configuration for JuniperData
# Automatically creates PRs for dependency updates
version: 2

updates:
  # Python dependencies
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "09:00"
      timezone: "America/New_York"
    open-pull-requests-limit: 5
    labels:
      - "dependencies"
      - "security"
    commit-message:
      prefix: "deps"
    groups:
      # Group minor/patch updates together
      python-minor:
        patterns:
          - "*"
        update-types:
          - "minor"
          - "patch"

  # GitHub Actions
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
    open-pull-requests-limit: 3
    labels:
      - "dependencies"
      - "ci"
    commit-message:
      prefix: "ci"
```

**Verification**: Check GitHub Security tab for Dependabot alerts within 24 hours.

---

### Phase 2: Code Quality Enforcement

**Objective**: Extend linting and type checking to test code.

#### Task 2.1: Enable Flake8 on Tests (H-01)

**File**: `.pre-commit-config.yaml`

**Current Code** (lines 125-126):
```yaml
files: ^juniper_data/.*\.py$
exclude: ^juniper_data/tests/
```

**Recommended Fix**:
```yaml
files: ^juniper_data/.*\.py$
# Remove exclude to lint tests
# Use per-file ignores for test-specific patterns
args:
  - --max-line-length=512
  - --extend-ignore=E203,E265,E266,E501,W503,E402,E226,C409,C901,B008
  - --per-file-ignores=juniper_data/tests/*:B011,S101
  - --max-complexity=15
  - --select=B,C,E,F,W,T4,B9
```

**Pre-requisite**: Fix the 7 unused imports identified:
- `juniper_data/tests/fixtures/generate_golden_datasets.py:16` - remove `os`
- `juniper_data/tests/integration/test_storage_workflow.py:13` - remove `Dict`
- `juniper_data/tests/unit/test_api_app.py:4` - remove `AsyncMock`
- `juniper_data/tests/unit/test_api_routes.py:3` - remove `Dict`
- `juniper_data/tests/unit/test_api_routes.py:10` - remove `generators`
- `juniper_data/tests/unit/test_api_routes.py:164` - remove `io`
- `juniper_data/tests/unit/test_main.py:4` - remove `MagicMock`

**Verification**: `pre-commit run flake8 --all-files`

---

#### Task 2.2: Enable MyPy on Tests (H-02)

**File**: `.pre-commit-config.yaml`

**Current Code** (line 139):
```yaml
files: ^juniper_data/(?!tests).*\.py$
```

**Recommended Fix**:
```yaml
files: ^juniper_data/.*\.py$
args:
  - --ignore-missing-imports
  - --no-strict-optional
  - --allow-untyped-defs  # Relaxed for test code
```

**Also update** `pyproject.toml` (lines 177-183):
```toml
[tool.mypy]
python_version = "3.14"
warn_return_any = false
warn_unused_configs = true
ignore_missing_imports = true
# Remove test exclusion to enable type checking
# Allow untyped defs in tests
[[tool.mypy.overrides]]
module = "juniper_data.tests.*"
allow_untyped_defs = true
disallow_untyped_defs = false
```

**Verification**: `mypy juniper_data --ignore-missing-imports`

---

#### Task 2.3: Reduce Flake8 Ignore Rules (H-05)

**File**: `.pre-commit-config.yaml`

**Current Ignores** (line 118):
```yaml
--extend-ignore=E203,E265,E266,E501,W503,E722,E402,E226,C409,C901,B008,B904,B905,B907,F401
```

**Rules to Remove**:

| Rule | Description | Action | Rationale |
|------|-------------|--------|-----------|
| E722 | Bare `except:` | **REMOVE** | Security risk - catches all exceptions including KeyboardInterrupt |
| F401 | Unused imports | **REMOVE** | Dead code accumulation |
| B904 | `raise` without `from` in `except` | **KEEP** | Too noisy for this codebase |

**Recommended Fix**:
```yaml
--extend-ignore=E203,E265,E266,E501,W503,E402,E226,C409,C901,B008,B904,B905,B907
# Removed: E722, F401
```

**Pre-requisite**:
1. Fix unused imports (7 instances - see Task 2.1)
2. Audit codebase for bare `except:` clauses and fix them

**Verification**: `flake8 juniper_data --select=E722,F401`

---

#### Task 2.4: Remove Pytest Warning Suppression (H-04)

**File**: `pyproject.toml`

**Current Code** (line 119):
```toml
"-p", "no:warnings",
```

**Recommended Fix**:
```toml
# Remove "-p", "no:warnings", and add selective filtering:
filterwarnings = [
    "ignore::DeprecationWarning:httpx.*",
    "ignore::PendingDeprecationWarning:pydantic.*",
    # Add other known third-party warnings as discovered
]
```

**Verification**: `pytest --tb=short 2>&1 | grep -i warning`

---

#### Task 2.5: Make pip-audit Blocking (H-03)

Already covered in Task 1.2.

---

### Phase 3: Infrastructure Improvements

**Objective**: Add missing CI/CD capabilities and improve observability.

#### Task 3.1: Fix MyPy Type Errors (M-01)

**Files with errors**:
- `juniper_data/core/artifacts.py:18, 44`
- `juniper_data/storage/memory.py:65`
- `juniper_data/storage/local_fs.py:77`

**Error**: `Argument 2 to "savez" has incompatible type "**dict[str, ndarray[...]]"; expected "bool"`

**Root Cause**: NumPy stubs type `np.savez(file, *args, **kwds)` incorrectly.

**Recommended Fix**: Add type ignore comments with explanation:
```python
np.savez(file, **arrays)  # type: ignore[arg-type]  # numpy stubs incorrectly type **kwds
```

**Alternative**: Use explicit array names:
```python
np.savez(file, data=arrays["data"], labels=arrays["labels"])
```

**Verification**: `mypy juniper_data --ignore-missing-imports`

---

#### Task 3.2: Add CodeQL Analysis (M-02)

**Create**: `.github/workflows/codeql.yml`

```yaml
name: CodeQL

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 6 * * 1'  # Weekly on Monday

jobs:
  analyze:
    name: Analyze
    runs-on: ubuntu-latest
    permissions:
      actions: read
      contents: read
      security-events: write

    strategy:
      fail-fast: false
      matrix:
        language: ['python']

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Initialize CodeQL
        uses: github/codeql-action/init@v3
        with:
          languages: ${{ matrix.language }}
          queries: +security-and-quality

      - name: Autobuild
        uses: github/codeql-action/autobuild@v3

      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v3
        with:
          category: "/language:${{ matrix.language }}"
```

**Verification**: Check GitHub Security tab for CodeQL findings.

---

#### Task 3.3: Add Coverage Upload (M-03)

**File**: `.github/workflows/ci.yml`

**Add to unit-tests job** (after coverage artifacts upload):
```yaml
- name: Upload coverage to Codecov
  uses: codecov/codecov-action@v4
  if: matrix.python-version == '3.14'  # Only upload once
  with:
    token: ${{ secrets.CODECOV_TOKEN }}
    files: reports/coverage.xml
    flags: unittests
    name: juniper-data-coverage
    fail_ci_if_error: false
```

**Pre-requisite**:
1. Create Codecov account and link repository
2. Add `CODECOV_TOKEN` to repository secrets

**Verification**: Check Codecov dashboard after CI run.

---

#### Task 3.4: Add Slow Test Job (M-04)

**File**: `.github/workflows/ci.yml`

**Add new job**:
```yaml
# ═══════════════════════════════════════════════════════════════════════════
# Slow Tests: Run tests marked as slow (weekly or on-demand)
# ═══════════════════════════════════════════════════════════════════════════
slow-tests:
  name: Slow Tests
  runs-on: ubuntu-latest
  if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'

  steps:
    - name: Checkout Code
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: "3.14"
        cache: pip

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install ".[all]"

    - name: Run Slow Tests
      run: |
        echo "╔════════════════════════════════════════════════════════════╗"
        echo "║       JuniperData - Slow Tests                             ║"
        echo "╚════════════════════════════════════════════════════════════╝"
        python -m pytest -m "slow" --verbose --timeout=600
```

**Also add schedule trigger** (at top of file):
```yaml
on:
  push:
    # ... existing ...
  schedule:
    - cron: '0 3 * * 0'  # Weekly on Sunday at 3 AM UTC
  workflow_dispatch:
```

**Verification**: Manually trigger workflow dispatch and confirm job runs.

---

### Phase 4: Best Practice Refinements

**Objective**: Polish and future-proofing.

#### Task 4.1: Pin Action Versions to SHA (L-01)

**File**: `.github/workflows/ci.yml`

**Example changes**:
```yaml
# Current
uses: actions/checkout@v4

# Recommended (with version comment)
uses: actions/checkout@b4ffde65f46336ab88eb53be808477a3936bae11  # v4.1.1
```

**Actions to pin**:
- `actions/checkout@v4`
- `actions/setup-python@v5`
- `actions/cache@v4`
- `actions/upload-artifact@v4`
- `github/codeql-action/upload-sarif@v3`
- `gitleaks/gitleaks-action@v2`

**Verification**: CI continues to pass after pinning.

---

#### Task 4.2: Add pyupgrade Hook (L-02)

**File**: `.pre-commit-config.yaml`

**Add**:
```yaml
# Python Syntax Modernization - pyupgrade
- repo: https://github.com/asottile/pyupgrade
  rev: v3.15.0
  hooks:
    - id: pyupgrade
      name: Upgrade Python syntax
      args:
        - --py311-plus
      files: ^juniper_data/.*\.py$
```

**Verification**: `pre-commit run pyupgrade --all-files`

---

#### Task 4.3: Add shellcheck Hook (L-03)

**File**: `.pre-commit-config.yaml`

**Add**:
```yaml
# Shell Script Linting - shellcheck
- repo: https://github.com/shellcheck-py/shellcheck-py
  rev: v0.9.0.6
  hooks:
    - id: shellcheck
      name: Lint shell scripts
      types: [shell]
```

**Verification**: `pre-commit run shellcheck --all-files`

---

## 5. Effort Estimates

### 5.1 Phase Effort Summary

| Phase | Tasks | Estimated Effort | Expected Duration |
|-------|-------|------------------|-------------------|
| Phase 1 | 3 | 4-6 hours | 1 day |
| Phase 2 | 5 | 8-12 hours | 2-3 days |
| Phase 3 | 4 | 6-10 hours | 2 days |
| Phase 4 | 3 | 2-4 hours | 1 day |
| **Total** | **15** | **20-32 hours** | **6-7 days** |

### 5.2 Individual Task Estimates

| Task ID | Description | Effort | Complexity |
|---------|-------------|--------|------------|
| 1.1 | Fix silent test pass | 1 hour | Low |
| 1.2 | Make security scans blocking | 2 hours | Medium |
| 1.3 | Add Dependabot | 1 hour | Low |
| 2.1 | Enable flake8 on tests + fix imports | 3 hours | Medium |
| 2.2 | Enable mypy on tests | 2 hours | Medium |
| 2.3 | Reduce flake8 ignores | 2 hours | Medium |
| 2.4 | Remove warning suppression | 2 hours | Low |
| 2.5 | (Covered in 1.2) | - | - |
| 3.1 | Fix mypy type errors | 1 hour | Low |
| 3.2 | Add CodeQL | 2 hours | Low |
| 3.3 | Add coverage upload | 2 hours | Medium |
| 3.4 | Add slow test job | 2 hours | Low |
| 4.1 | Pin action versions | 1 hour | Low |
| 4.2 | Add pyupgrade | 30 min | Low |
| 4.3 | Add shellcheck | 30 min | Low |

---

## 6. Risk Assessment

### 6.1 Implementation Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Breaking existing CI | Medium | High | Test changes in feature branch first |
| Flake8/mypy on tests reveals many issues | Medium | Medium | Address incrementally; use per-file ignores initially |
| Dependabot creates too many PRs | Low | Low | Configure grouping and limits |
| CodeQL false positives | Medium | Low | Review findings before enabling blocking |
| pip-audit strict mode blocks valid releases | Low | Medium | Use `--ignore-vuln` for known false positives |

### 6.2 Risk of Inaction

| Risk | Probability | Impact | Consequence |
|------|-------------|--------|-------------|
| Security vulnerability in dependency | High | High | Unpatched vulnerability in production |
| Silent test failures | Medium | High | Bugs reach production undetected |
| Deprecated dependency breaks code | Medium | Medium | Emergency fixes required |
| Type errors cause runtime bugs | Low | Medium | Unexpected failures in production |

---

## 7. Success Criteria

### 7.1 Phase 1 Completion Criteria

- [ ] `test_main.py` passes without silent exception handling
- [ ] Bandit scan failures block CI for medium+ severity
- [ ] pip-audit failures block CI for any vulnerability
- [ ] Dependabot is enabled and creating PRs
- [ ] All existing tests still pass

### 7.2 Phase 2 Completion Criteria

- [ ] Flake8 runs on all Python files including tests
- [ ] MyPy runs on all Python files including tests (with relaxed rules for tests)
- [ ] No unused imports in codebase
- [ ] No bare `except:` clauses in codebase
- [ ] Pytest shows deprecation warnings (filtered as needed)
- [ ] Pre-commit passes on all files

### 7.3 Phase 3 Completion Criteria

- [ ] Zero mypy errors in production code
- [ ] CodeQL analysis running on PRs
- [ ] Coverage reports uploaded to Codecov
- [ ] Slow test job configured (even if no slow tests exist yet)
- [ ] Coverage badge visible in README

### 7.4 Phase 4 Completion Criteria

- [ ] All GitHub Actions pinned to SHA
- [ ] pyupgrade hook running
- [ ] shellcheck hook running
- [ ] All shell scripts pass shellcheck

---

## 8. Appendices

### 8.1 Files to Modify

| File | Phase(s) | Modification Type |
|------|----------|-------------------|
| `juniper_data/tests/unit/test_main.py` | 1 | Fix test logic |
| `.github/workflows/ci.yml` | 1, 3, 4 | Multiple enhancements |
| `.github/dependabot.yml` | 1 | Create new file |
| `.github/workflows/codeql.yml` | 3 | Create new file |
| `.pre-commit-config.yaml` | 2, 4 | Multiple enhancements |
| `pyproject.toml` | 2 | Update mypy/pytest config |
| `juniper_data/tests/fixtures/generate_golden_datasets.py` | 2 | Remove unused import |
| `juniper_data/tests/integration/test_storage_workflow.py` | 2 | Remove unused import |
| `juniper_data/tests/unit/test_api_app.py` | 2 | Remove unused import |
| `juniper_data/tests/unit/test_api_routes.py` | 2 | Remove unused imports |
| `juniper_data/core/artifacts.py` | 3 | Fix type annotations |
| `juniper_data/storage/memory.py` | 3 | Fix type annotations |
| `juniper_data/storage/local_fs.py` | 3 | Fix type annotations |

### 8.2 Commands for Verification

```bash
# Phase 1 Verification
pytest juniper_data/tests/unit/test_main.py -v
bandit -r juniper_data -ll -ii --severity-level medium
pip-audit --strict

# Phase 2 Verification
pre-commit run --all-files
flake8 juniper_data --select=E722,F401
mypy juniper_data

# Phase 3 Verification
mypy juniper_data --ignore-missing-imports
# CodeQL - check GitHub Security tab
# Codecov - check Codecov dashboard

# Phase 4 Verification
pre-commit run pyupgrade --all-files
pre-commit run shellcheck --all-files
```

### 8.3 Source Audit Reports

This development plan consolidates findings from:
1. `notes/TEST_SUITE_AUDIT_REPORT.md` (2026-02-03)
2. `notes/TEST_SUITE_CICD_AUDIT_REPORT_20260203_170839.md` (2026-02-03)

### 8.4 Additional Resources

- [GitHub Actions Best Practices](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions)
- [Dependabot Configuration](https://docs.github.com/en/code-security/dependabot/dependabot-version-updates/configuration-options-for-the-dependabot.yml-file)
- [CodeQL Documentation](https://codeql.github.com/docs/)
- [Codecov Documentation](https://docs.codecov.com/)
- [pre-commit Hooks](https://pre-commit.com/hooks.html)

---

**Document Status**: Ready for Review
**Next Action**: Obtain stakeholder approval before proceeding with Phase 1 implementation
**Prepared By**: Claude (Opus 4.5)
**Review Date**: TBD
