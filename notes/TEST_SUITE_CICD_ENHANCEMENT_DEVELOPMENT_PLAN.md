# JuniperData Test Suite & CI/CD Enhancement Development Plan

**Project**: JuniperData - Dataset Generation Service
**Version**: 1.0.0
**Created**: 2026-02-04
**Status**: DRAFT - Pending Review
**Consolidated From**: TEST_SUITE_CICD_ENHANCEMENT_DEVELOPMENT_PLAN_AMP.md, TEST_SUITE_CICD_ENHANCEMENT_DEVELOPMENT_PLAN_CLAUDE.md

---

## Executive Summary

This development plan consolidates findings from two independent audits of the JuniperData test suite and CI/CD pipeline. The analysis validates that the core testing infrastructure is sound (207 tests, 100% coverage, all passing), but identifies **24 actionable improvements** across security, code quality, and CI/CD configuration.

### Current State Summary

| Metric                 | Value                          | Assessment             |
| ---------------------- | ------------------------------ | ---------------------- |
| Total Tests            | 207 (183 unit, 24 integration) | **Excellent**          |
| Code Coverage          | 100%                           | **Excellent**          |
| Tests Passing          | 207/207 (100%)                 | **Excellent**          |
| Critical Issues        | 3                              | **Requires Attention** |
| High Priority Issues   | 6                              | **Requires Attention** |
| Medium Priority Issues | 10                             | **Moderate**           |
| Low Priority Issues    | 5                              | **Minor**              |

### Overall Risk Assessment

**Current State**: The test suite achieves 100% code coverage with all 207 tests passing. However, the CI/CD pipeline contains **critical security blind spots** where security scanners cannot fail the build, creating a false sense of security. Additionally, static analysis tools deliberately exclude test code, allowing quality issues to accumulate undetected.

**Recommended Action**: Immediate remediation of Critical issues is essential. The security scanning configuration currently defeats its stated purpose.

---

## Table of Contents

1. [Consolidated Issues Registry](#1-consolidated-issues-registry)
2. [Priority Classification](#2-priority-classification)
3. [Detailed Issue Analysis](#3-detailed-issue-analysis)
4. [Implementation Phases](#4-implementation-phases)
5. [Effort Estimates](#5-effort-estimates)
6. [Risk Assessment & Mitigation](#6-risk-assessment--mitigation)
7. [Success Criteria](#7-success-criteria)
8. [Appendix](#appendix)

---

## 1. Consolidated Issues Registry

### 1.1 Issues Validated from Audit Reports

| ID      | Issue                                                | Location                    | Severity     | Source     |
| ------- | ---------------------------------------------------- | --------------------------- | ------------ | ---------- |
| SEC-001 | Bandit failures suppressed with `\|\| true`          | ci.yml:313                  | **CRITICAL** | Both       |
| SEC-002 | pip-audit vulnerabilities don't fail build           | ci.yml:329                  | **CRITICAL** | Both       |
| SEC-003 | No Dependabot configuration                          | .github/                    | **CRITICAL** | Both       |
| TST-001 | Silent test pass via `except ImportError: pass`      | test_main.py:32-38          | **CRITICAL** | Both       |
| TST-002 | Tests excluded from Flake8 pre-commit                | .pre-commit-config.yaml:126 | **HIGH**     | Both       |
| TST-003 | Tests excluded from MyPy pre-commit                  | .pre-commit-config.yaml:139 | **HIGH**     | Both       |
| TST-004 | Pytest warnings globally suppressed                  | pyproject.toml:119          | **HIGH**     | Both       |
| CFG-001 | Excessive Flake8 ignore rules (E722, F401)           | .pre-commit-config.yaml:118 | **HIGH**     | Both       |
| SEC-004 | GitHub Actions not pinned to SHA                     | ci.yml:70,73,84,etc.        | **MEDIUM**   | AMP        |
| CFG-002 | 512-character line length (excessively permissive)   | Multiple files              | **MEDIUM**   | Both       |
| CFG-003 | MyPy type errors in production code (6 errors)       | 5 source files              | **MEDIUM**   | Both       |
| CFG-004 | Tests excluded from MyPy in pyproject.toml           | pyproject.toml:182          | **MEDIUM**   | AMP        |
| CFG-005 | Continue on collection errors in pytest              | pyproject.toml:122          | **MEDIUM**   | Claude     |
| INF-001 | No CodeQL analysis configured                        | .github/workflows/          | **MEDIUM**   | Both       |
| INF-002 | No coverage upload to external service               | ci.yml                      | **MEDIUM**   | Both       |
| INF-003 | No scheduled slow test job                           | ci.yml                      | **MEDIUM**   | Both       |
| INF-004 | SARIF upload uses continue-on-error                  | ci.yml:320                  | **MEDIUM**   | AMP        |
| INF-005 | No performance test infrastructure                   | pyproject.toml:128          | **MEDIUM**   | Claude     |
| L-01    | No pyupgrade hook for syntax modernization           | .pre-commit-config.yaml     | **LOW**      | Claude     |
| L-02    | No shellcheck hook for shell scripts                 | .pre-commit-config.yaml     | **LOW**      | Claude     |
| L-03    | Python 3.14 target not fully supported by Black      | .pre-commit-config.yaml:33  | **LOW**      | Claude     |
| L-04    | No documentation build step in CI                    | ci.yml                      | **LOW**      | Claude     |

### 1.2 Flake8 Violations in Test Code (Currently Undetected)

Running `flake8 juniper_data/tests` revealed **33+ issues** that are currently hidden due to test exclusion:

| Category                        | Count  | Impact                   |
| ------------------------------- | ------ | ------------------------ |
| F401 (unused imports)           | 7      | Dead code accumulation   |
| F541 (empty f-strings)          | 5      | Potential bugs           |
| E402 (import order)             | 1      | Style inconsistency      |
| SIM117 (nested with statements) | 21     | Reduced readability      |
| **Total**                       | **34** | Code quality degradation |

### 1.3 MyPy Type Errors in Production Code

| File                     | Line | Error                                      | Type       |
| ------------------------ | ---- | ------------------------------------------ | ---------- |
| `core/artifacts.py`      | 18   | `savez` type signature mismatch            | arg-type   |
| `core/artifacts.py`      | 44   | `savez` type signature mismatch            | arg-type   |
| `storage/memory.py`      | 65   | `savez_compressed` type signature mismatch | arg-type   |
| `storage/local_fs.py`    | 77   | `savez_compressed` type signature mismatch | arg-type   |
| `api/routes/datasets.py` | 19   | Incompatible None assignment               | assignment |
| `api/app.py`             | 40   | Implicit Optional not allowed              | assignment |

### 1.4 Unused Imports in Test Files

| File                                                      | Line | Unused Import |
| --------------------------------------------------------- | ---- | ------------- |
| `juniper_data/tests/fixtures/generate_golden_datasets.py` | 16   | `os`          |
| `juniper_data/tests/integration/test_storage_workflow.py` | 13   | `Dict`        |
| `juniper_data/tests/unit/test_api_app.py`                 | 4    | `AsyncMock`   |
| `juniper_data/tests/unit/test_api_routes.py`              | 3    | `Dict`        |
| `juniper_data/tests/unit/test_api_routes.py`              | 10   | `generators`  |
| `juniper_data/tests/unit/test_api_routes.py`              | 164  | `io`          |
| `juniper_data/tests/unit/test_main.py`                    | 4    | `MagicMock`   |

---

## 2. Priority Classification

### 2.1 Priority Definitions

| Priority | Label        | Definition                                                                                         | SLA               |
| -------- | ------------ | -------------------------------------------------------------------------------------------------- | ----------------- |
| P0       | **CRITICAL** | Security vulnerability, CI integrity compromised, or tests that silently pass without validation   | Immediate (< 24h) |
| P1       | **HIGH**     | Test quality degradation, missing security infrastructure, or significant code quality blind spots | 1 week            |
| P2       | **MEDIUM**   | Configuration improvements, type safety gaps, or reduced signal quality                            | 2 weeks           |
| P3       | **LOW**      | Style preferences, optional enhancements, or nice-to-have features                                 | 1 month           |

### 2.2 Issue Priority Matrix

```bash
┌─────────────────────────────────────────────────────────────────────────┐
│                        PRIORITY MATRIX                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  P0 CRITICAL (Immediate)                                                │
│  ├── SEC-001: Bandit || true suppression                                │
│  ├── SEC-002: pip-audit doesn't fail on vulnerabilities                 │
│  ├── SEC-003: Missing Dependabot configuration                          │
│  └── TST-001: Silent ImportError pass in test_main.py                   │
│                                                                         │
│  P1 HIGH (1 Week)                                                       │
│  ├── TST-002: Tests excluded from Flake8                                │
│  ├── TST-003: Tests excluded from MyPy                                  │
│  ├── TST-004: Pytest warnings suppressed                                │
│  └── CFG-001: Excessive Flake8 ignores (E722, F401)                     │
│                                                                         │
│  P2 MEDIUM (2 Weeks)                                                    │
│  ├── SEC-004: GitHub Actions not pinned to SHA                          │
│  ├── CFG-002: 512-char line length                                      │
│  ├── CFG-003: MyPy production code errors                               │
│  ├── CFG-004: Tests excluded from pyproject.toml MyPy                   │
│  ├── CFG-005: Continue on collection errors                             │
│  ├── INF-001: Missing CodeQL analysis                                   │
│  ├── INF-002: No coverage upload service                                │
│  ├── INF-003: No slow test job                                          │
│  ├── INF-004: SARIF upload continue-on-error                            │
│  └── INF-005: No performance test infrastructure                        │
│                                                                         │
│  P3 LOW (1 Month)                                                       │
│  ├── L-01: No pyupgrade hook                                            │
│  ├── L-02: No shellcheck hook                                           │
│  ├── L-03: Python 3.14 Black target                                     │
│  └── L-04: No documentation build step                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Issue Analysis

### 3.1 Critical Issues (P0)

#### SEC-001: Bandit Security Scan Suppression

**Location**: `.github/workflows/ci.yml` line 313

**Current State**:

```yaml
bandit -r juniper_data -f sarif -o reports/security/bandit.sarif || true
```

**Problem**: The `|| true` suffix guarantees the step exits with code 0 regardless of Bandit's findings. This means:

- High-severity security vulnerabilities are logged but don't fail CI
- The Quality Gate step checks `needs.security.result`, which will always be "success"
- Teams have a false sense of security ("all checks pass")

**Evidence**: Bandit currently finds 6 medium-severity B104 findings (hardcoded bind to all interfaces) which are expected test behaviors, but future real vulnerabilities would be equally ignored.

**Risk**: **SEVERE** - Defeats the entire purpose of security scanning in CI.

**Recommended Fix**:

```yaml
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

**Optional Alternative**: For graduated adoption, configure baseline exclusions for known-acceptable findings.

**Effort**: Small (< 1 hour)

---

#### SEC-002: pip-audit Vulnerability Detection Non-Blocking

**Location**: `.github/workflows/ci.yml` line 329

**Current State**:

```yaml
pip-audit -r reports/security/pip-freeze.txt || echo "::warning::Vulnerabilities found in dependencies"
```

**Problem**: Using `|| echo` converts any pip-audit failure (including critical vulnerabilities) into a warning annotation that doesn't fail the job.

**Risk**: **SEVERE** - Known vulnerable dependencies can enter production without any blocking check.

**Recommended Fix**:

```yaml
- name: Run pip-audit (Dependency Vulnerabilities)
  run: |
    pip install ".[all]"
    pip freeze > reports/security/pip-freeze.txt
    pip-audit -r reports/security/pip-freeze.txt --strict
```

**Optional Alternative**: For graduated adoption, use `--ignore-vuln` for known-acceptable vulnerabilities or `--only-high-severity` initially.

**Effort**: Small (< 1 hour, plus remediation time if vulnerabilities exist)

---

#### SEC-003: Missing Dependabot Configuration

**Location**: `.github/` directory

**Current State**: No `dependabot.yml` file exists.

**Problem**: Without Dependabot:

- Dependencies drift into vulnerable states silently
- Manual monitoring of security advisories required
- GitHub Actions versions may become outdated

**Recommended Fix**: Create `.github/dependabot.yml`:

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

**Optional Alternative** (simpler configuration):

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10
    groups:
      python-dependencies:
        patterns:
          - "*"
    labels:
      - "dependencies"
      - "python"

  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
    labels:
      - "dependencies"
      - "github-actions"
```

**Effort**: Small (< 30 minutes)

---

#### TST-001: Silent Test Pass via ImportError Exception Handling

**Location**: `juniper_data/tests/unit/test_main.py` lines 32-38

**Current State**:

```python
try:
    importlib.reload(main_module)
    result = main_module.main()
    assert result == 1
    mock_print.assert_called()
except ImportError:
    pass  # <-- Test passes silently if ImportError occurs
```

**Problem**: If an `ImportError` is raised during test execution, the test passes without executing any assertions. This converts "test scenario is broken" into "test passed."

**Risk**: **HIGH** - Can mask regressions or broken test infrastructure.

**Recommended Fix** (properly mock the import error):

```python
def test_main_import_error_uvicorn_not_installed(self) -> None:
    """Test main returns 1 when uvicorn is not installed."""
    import builtins

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "uvicorn":
            raise ImportError("No module named 'uvicorn'")
        return original_import(name, *args, **kwargs)

    with patch.object(sys, "argv", ["juniper_data"]), \
         patch("builtins.print") as mock_print, \
         patch.object(builtins, "__import__", side_effect=mock_import), \
         patch.dict(sys.modules, {"uvicorn": None}):

        import importlib
        from juniper_data import __main__ as main_module

        importlib.reload(main_module)
        result = main_module.main()
        assert result == 1
        mock_print.assert_called()
```

**Optional Alternative** (use pytest skip):

```python
@pytest.mark.skipif(
    importlib.util.find_spec("uvicorn") is None,
    reason="uvicorn not installed"
)
def test_main_import_error_uvicorn_not_installed(self) -> None:
    """Test main returns 1 when uvicorn is not installed."""
    # ... existing mock setup ...

    importlib.reload(main_module)
    result = main_module.main()
    assert result == 1
    mock_print.assert_called()
```

**Optional Alternative** (explicit skip on ImportError):

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

**Effort**: Small (< 1 hour)

---

### 3.2 High Priority Issues (P1)

#### TST-002: Tests Excluded from Flake8 Pre-commit

**Location**: `.pre-commit-config.yaml` line 126

**Current State**:

```yaml
files: ^juniper_data/.*\.py$
exclude: ^juniper_data/tests/
```

**Problem**: 34+ flake8 violations exist in test code that are invisible to developers. Issues include:

- 7 unused imports (F401) - dead code
- 5 empty f-strings (F541) - potential bugs
- 21 nested with statements (SIM117) - reduced readability

**Recommended Fix** (Graduated Approach):

**Phase 1**: Add a separate, relaxed hook for tests:

```yaml
- id: flake8
  name: Lint tests with Flake8 (relaxed)
  args:
    - --max-line-length=512
    - --extend-ignore=E203,E265,E266,E501,W503,E402,E226,C409,C901,SIM117
    - --select=F401,F541  # Start with unused imports and empty f-strings
  files: ^juniper_data/tests/.*\.py$
```

**Phase 2**: Fix existing violations, then merge hooks with full rules.

**Optional Alternative** (immediate full coverage with per-file ignores):

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

**Effort**: Medium (1-2 hours setup, 2-4 hours fixing violations)

---

#### TST-003: Tests Excluded from MyPy Pre-commit

**Location**: `.pre-commit-config.yaml` line 139

**Current State**:

```yaml
files: ^juniper_data/(?!tests).*\.py$
```

**Problem**: Type errors in test code go undetected, leading to:

- Runtime errors in tests that could be caught statically
- Inconsistent type usage between tests and production code
- Reduced confidence in test reliability

**Recommended Fix**:

```yaml
- id: mypy
  name: Type check with MyPy
  args:
    - --ignore-missing-imports
    - --no-strict-optional
    - --allow-untyped-defs  # Relaxed for tests
  files: ^juniper_data/.*\.py$
```

**Also update** `pyproject.toml`:

```toml
[tool.mypy]
python_version = "3.14"
warn_return_any = false
warn_unused_configs = true
ignore_missing_imports = true
# Allow untyped defs in tests
[[tool.mypy.overrides]]
module = "juniper_data.tests.*"
allow_untyped_defs = true
disallow_untyped_defs = false
```

**Effort**: Medium (1-2 hours, may require fixing type issues in tests)

---

#### TST-004: Pytest Warnings Globally Suppressed

**Location**: `pyproject.toml` line 119

**Current State**:

```toml
addopts = [
    "-p", "no:warnings",
    ...
]
```

**Problem**: All pytest warnings are suppressed, hiding:

- Deprecation warnings from dependencies
- Future-incompatible code patterns
- Potential test issues

**Recommended Fix**:

```toml
addopts = [
    "-ra",
    "-q",
    "--strict-markers",
    "--strict-config",
    "--tb=short",
]
filterwarnings = [
    "ignore::DeprecationWarning:uvicorn.*",
    "ignore::DeprecationWarning:httpx.*",
    "ignore::PendingDeprecationWarning:pydantic.*",
]
```

**Effort**: Small (< 1 hour, plus addressing any warnings that surface)

---

#### CFG-001: Excessive Flake8 Ignore Rules

**Location**: `.pre-commit-config.yaml` line 118

**Current State**:

```yaml
--extend-ignore=E203,E265,E266,E501,W503,E722,E402,E226,C409,C901,B008,B904,B905,B907,F401
```

**Problematic Rules**:

| Rule | Description            | Risk of Ignoring                            |
| ---- | ---------------------- | ------------------------------------------- |
| E722 | Bare `except:`         | Catches unexpected exceptions, masks errors |
| F401 | Unused imports         | Dead code accumulation                      |
| B008 | Mutable default args   | Common bug pattern                          |
| B904 | `raise` without `from` | Lost error context                          |

**Recommended Fix**: Gradually remove risky ignores:

1. **Immediate**: Remove `E722` (bare except)
2. **Week 1**: Remove `F401` (unused imports), run autoflake to clean
3. **Week 2**: Remove `B904` (raise without from)

```yaml
--extend-ignore=E203,E265,E266,E501,W503,E402,E226,C409,C901,B008,B905,B907
# Removed: E722, F401, B904
```

**Effort**: Medium (2-4 hours, depends on codebase violations)

---

### 3.3 Medium Priority Issues (P2)

#### SEC-004: GitHub Actions Not Pinned to SHA

**Location**: Throughout `ci.yml`

**Current State**:

```yaml
uses: actions/checkout@v4
uses: actions/setup-python@v5
uses: actions/cache@v4
```

**Problem**: Version tags can be retargeted, creating supply-chain risk. A compromised upstream action could inject malicious code.

**Recommended Fix**:

```yaml
uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683  # v4.2.2
uses: actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065  # v5.6.0
uses: actions/cache@5a3ec84eff668545956fd18022155c47e93e2684  # v4.2.3
```

**Effort**: Small (< 1 hour)

---

#### CFG-003: MyPy Type Errors in Production Code

**Location**: 5 source files with 6 errors

**Current Errors**:

1. `core/artifacts.py:18,44` - numpy `savez` signature issues
2. `storage/memory.py:65` - numpy `savez_compressed` signature
3. `storage/local_fs.py:77` - numpy `savez_compressed` signature
4. `api/routes/datasets.py:19` - None assignment to typed variable
5. `api/app.py:40` - Implicit Optional parameter

**Recommended Fix**: Address each error category:

**For numpy signature issues** (4 errors):

```python
# Add type: ignore comment with explanation
np.savez(file, **arrays)  # type: ignore[arg-type]  # numpy stubs incomplete
```

**For Optional parameter issues** (2 errors):

```python
# api/app.py:40 - Make Optional explicit
def create_app(settings: Optional[Settings] = None) -> FastAPI:
```

```python
# api/routes/datasets.py:19
_store: Optional[DatasetStore] = None
```

**Optional Alternative** (for numpy): Use explicit array names instead of `**kwargs`:

```python
np.savez(file, data=arrays["data"], labels=arrays["labels"])
```

**Effort**: Small-Medium (1-2 hours)

---

#### CFG-005: Continue on Collection Errors in Pytest

**Location**: `pyproject.toml` line 122

**Current State**:

```toml
"--continue-on-collection-errors",
```

**Problem**: This flag allows pytest to continue even when test collection fails due to import errors or syntax errors. This may hide broken tests.

**Recommended Fix**: Remove the flag and ensure all tests collect properly:

```toml
addopts = [
    "-ra",
    "-q",
    "--strict-markers",
    "--strict-config",
    "--tb=short",
]
```

**Effort**: Small (< 30 minutes, may require fixing collection errors)

---

#### INF-001: Missing CodeQL Analysis

**Location**: `.github/workflows/`

**Problem**: No CodeQL semantic analysis configured. CodeQL provides deeper security analysis than Bandit for:

- Injection vulnerabilities
- Data flow issues
- Complex security patterns

**Recommended Fix**: Create `.github/workflows/codeql.yml`:

```yaml
name: CodeQL Analysis

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

**Effort**: Small (< 1 hour)

---

#### INF-002: No Coverage Upload to External Service

**Problem**: Coverage reports are generated but not uploaded to services like Codecov or Coveralls for tracking trends.

**Recommended Fix**: Add Codecov upload step to `ci.yml`:

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

**Pre-requisites**:

1. Create Codecov account and link repository
2. Add `CODECOV_TOKEN` to repository secrets

**Effort**: Small (< 30 minutes)

---

#### INF-003: No Scheduled Slow Test Job

**Problem**: The `slow` marker exists but no tests use it, and no scheduled job runs slow tests.

**Recommended Fix**: Add slow test job to `ci.yml`:

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

**Also add schedule trigger**:

```yaml
on:
  push:
    # ... existing ...
  schedule:
    - cron: '0 3 * * 0'  # Weekly on Sunday at 3 AM UTC
  workflow_dispatch:
```

**Effort**: Small (< 1 hour)

---

#### INF-005: No Performance Test Infrastructure

**Location**: `pyproject.toml` line 128

**Problem**: The `performance` marker is defined but no tests use it and no infrastructure exists to run performance regression tests.

**Recommendation**: Document the performance test policy. Add infrastructure when performance tests are introduced. Consider using pytest-benchmark for future performance tests.

**Effort**: Small (documentation only for now)

---

### 3.4 Low Priority Issues (P3)

#### L-01: No pyupgrade Hook

**Problem**: Missing Python syntax modernization hook that automatically upgrades legacy Python syntax.

**Recommended Fix**: Add to `.pre-commit-config.yaml`:

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

**Effort**: Small (< 30 minutes)

---

#### L-02: No shellcheck Hook

**Problem**: No shell script linting configured.

**Recommended Fix**: Add to `.pre-commit-config.yaml`:

```yaml
# Shell Script Linting - shellcheck
- repo: https://github.com/shellcheck-py/shellcheck-py
  rev: v0.9.0.6
  hooks:
    - id: shellcheck
      name: Lint shell scripts
      types: [shell]
```

**Effort**: Small (< 30 minutes)

---

#### L-03: Python 3.14 Black Target

**Problem**: Black doesn't fully support py314 target yet.

**Recommendation**: Monitor Black releases and update when full support is available.

**Effort**: None (monitoring only)

---

#### L-04: No Documentation Build Step

**Problem**: Documentation is not validated in CI, allowing documentation drift.

**Recommendation**: Consider adding documentation build step if documentation grows.

**Effort**: Small (< 1 hour when needed)

---

## 4. Implementation Phases

### 4.1 Phase Overview

```bash
┌─────────────────────────────────────────────────────────────────────────┐
│                    DEVELOPMENT PHASES                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  PHASE 1: Security Remediation (Critical)                               │
│  ├── SEC-001: Fix Bandit || true                                        │
│  ├── SEC-002: Fix pip-audit warning handling                            │
│  ├── SEC-003: Add Dependabot configuration                              │
│  └── TST-001: Fix silent ImportError test                               │
│                                                                         │
│  PHASE 2: Code Quality Enforcement (High)                               │
│  ├── TST-002: Enable Flake8 on tests                                    │
│  ├── TST-003: Enable MyPy on tests                                      │
│  ├── TST-004: Remove pytest warning suppression                         │
│  └── CFG-001: Reduce Flake8 ignores                                     │
│                                                                         │
│  PHASE 3: Infrastructure Improvements (Medium)                          │
│  ├── SEC-004: Pin GitHub Actions to SHA                                 │
│  ├── CFG-003: Fix MyPy production errors                                │
│  ├── CFG-005: Remove continue-on-collection-errors                      │
│  ├── INF-001: Add CodeQL analysis                                       │
│  ├── INF-002: Add coverage upload                                       │
│  └── INF-003: Add slow test job                                         │
│                                                                         │
│  PHASE 4: Best Practice Refinements (Low)                               │
│  ├── L-01: Add pyupgrade hook                                           │
│  ├── L-02: Add shellcheck hook                                          │
│  └── Test code cleanup (34+ violations)                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Phase 1: Security Remediation (CRITICAL)

**Objective**: Ensure security scanners can actually fail the build.

**Priority**: P0 - Immediate

#### Tasks

| Task ID | Description                                               | Status |
| ------- | --------------------------------------------------------- | ------ |
| P1-T1   | Update Bandit command to use `--exit-zero` flag           | ☐      |
| P1-T2   | Add blocking Bandit check for medium+ severity            | ☐      |
| P1-T3   | Update pip-audit to use `--strict` flag                   | ☐      |
| P1-T4   | Create `.github/dependabot.yml`                           | ☐      |
| P1-T5   | Refactor test_main.py to remove silent exception handling | ☐      |
| P1-T6   | Run security scans to verify failures are detected        | ☐      |
| P1-T7   | Address any immediate security findings                   | ☐      |

#### Acceptance Criteria

- [ ] Bandit security findings cause CI to produce warnings (not silent success)
- [ ] pip-audit vulnerabilities cause CI to fail
- [ ] Dependabot PRs appear for outdated dependencies
- [ ] test_main.py no longer silently passes on ImportError
- [ ] All tests still pass (207/207)

#### Rollback Plan

If security scans reveal many issues, temporarily use `--exit-zero` for Bandit and `--ignore-vuln` for known issues while remediating.

---

### 4.3 Phase 2: Code Quality Enforcement

**Objective**: Extend static analysis to cover test code.

**Priority**: P1 - High

#### Tasks: 4.3, Phase 2

| Task ID | Description                                  | Status |
| ------- | -------------------------------------------- | ------ |
| P2-T1   | Add Flake8 hook for tests with relaxed rules | ☐      |
| P2-T2   | Fix F401 (unused imports) in tests (7)       | ☐      |
| P2-T3   | Fix F541 (empty f-strings) in tests (5)      | ☐      |
| P2-T4   | Update MyPy hook to include tests            | ☐      |
| P2-T5   | Remove `-p no:warnings` from pytest config   | ☐      |
| P2-T6   | Add selective filterwarnings configuration   | ☐      |
| P2-T7   | Remove E722 from Flake8 ignores              | ☐      |
| P2-T8   | Remove F401 from Flake8 ignores              | ☐      |
| P2-T9   | Run pre-commit on all files to verify        | ☐      |

#### Acceptance Criteria: 4.3, Phase 2

- [ ] Flake8 runs on test files in pre-commit
- [ ] No F401 or F541 violations in tests
- [ ] MyPy runs on test files in pre-commit
- [ ] E722 and F401 removed from global ignores
- [ ] Pytest shows relevant warnings (not all suppressed)
- [ ] Pre-commit passes on all files

---

### 4.4 Phase 3: Infrastructure Improvements

**Objective**: Add missing CI/CD capabilities and improve configuration.

**Priority**: P2 - Medium

#### Tasks: 4.4, Phase 2

| Task ID | Description                                      | Status |
| ------- | ------------------------------------------------ | ------ |
| P3-T1   | Pin all GitHub Actions to commit SHAs            | ☐      |
| P3-T2   | Fix 6 MyPy errors in production code             | ☐      |
| P3-T3   | Remove `--continue-on-collection-errors`         | ☐      |
| P3-T4   | Create CodeQL analysis workflow                  | ☐      |
| P3-T5   | Add Codecov/Coveralls integration                | ☐      |
| P3-T6   | Add slow test job and schedule trigger           | ☐      |
| P3-T7   | Update pyproject.toml MyPy exclude pattern       | ☐      |
| P3-T8   | Verify CodeQL scans complete successfully        | ☐      |

#### Acceptance Criteria: 4.4, Phase 2

- [ ] All GitHub Actions pinned to specific commit SHAs
- [ ] Zero mypy errors in production code
- [ ] CodeQL scans run on push to main/develop
- [ ] Coverage reports uploaded to Codecov
- [ ] Slow test job configured

---

### 4.5 Phase 4: Best Practice Refinements

**Objective**: Polish and future-proofing.

**Priority**: P3 - Low

#### Tasks: 4.5, Phase 4

| Task ID | Description                               | Status |
| ------- | ----------------------------------------- | ------ |
| P4-T1   | Add pyupgrade hook                        | ☐      |
| P4-T2   | Add shellcheck hook                       | ☐      |
| P4-T3   | Fix SIM117 violations (nested with) (21)  | ☐      |
| P4-T4   | Review and potentially reduce line length | ☐      |
| P4-T5   | Consider migration from flake8 to ruff    | ☐      |

#### Acceptance Criteria: 4.5, Phase 4

- [ ] pyupgrade hook running
- [ ] shellcheck hook running
- [ ] Test code readability improved (SIM117 fixed)

---

## 5. Effort Estimates

### 5.1 Phase Effort Summary

| Phase     | Issues | Estimated Effort | Priority |
| --------- | ------ | ---------------- | -------- |
| Phase 1   | 4      | 4-6 hours        | CRITICAL |
| Phase 2   | 4      | 8-12 hours       | HIGH     |
| Phase 3   | 6      | 6-10 hours       | MEDIUM   |
| Phase 4   | 4      | 3-5 hours        | LOW      |
| **Total** | **18** | **21-33 hours**  | -        |

### 5.2 Individual Task Estimates

| Issue ID | Description                          | Effort  | Complexity |
| -------- | ------------------------------------ | ------- | ---------- |
| SEC-001  | Fix Bandit `\|\|` true               | 1 hour  | Low        |
| SEC-002  | Fix pip-audit warning handling       | 1 hour  | Low        |
| SEC-003  | Add Dependabot configuration         | 30 min  | Low        |
| TST-001  | Fix silent ImportError test          | 1 hour  | Low        |
| TST-002  | Enable Flake8 on tests + fix imports | 3 hours | Medium     |
| TST-003  | Enable MyPy on tests                 | 2 hours | Medium     |
| TST-004  | Remove warning suppression           | 1 hour  | Low        |
| CFG-001  | Reduce Flake8 ignores                | 2 hours | Medium     |
| SEC-004  | Pin GitHub Actions to SHA            | 1 hour  | Low        |
| CFG-003  | Fix MyPy production errors           | 1 hour  | Low        |
| CFG-005  | Remove continue-on-collection-errors | 30 min  | Low        |
| INF-001  | Add CodeQL analysis                  | 1 hour  | Low        |
| INF-002  | Add coverage upload                  | 1 hour  | Medium     |
| INF-003  | Add slow test job                    | 1 hour  | Low        |
| L-01     | Add pyupgrade hook                   | 30 min  | Low        |
| L-02     | Add shellcheck hook                  | 30 min  | Low        |

---

## 6. Risk Assessment & Mitigation

### 6.1 Implementation Risks

| Risk                                       | Probability | Impact | Mitigation                                               |
| ------------------------------------------ | ----------- | ------ | -------------------------------------------------------- |
| Security scans reveal many vulnerabilities | Medium      | High   | Use graduated enforcement (`--only-high-severity` first) |
| Flake8 on tests creates excessive churn    | High        | Medium | Start with F401/F541 only, expand gradually              |
| MyPy on tests is too noisy                 | Medium      | Medium | Use `--allow-untyped-defs` for tests initially           |
| Warnings flood pytest output               | Medium      | Low    | Configure filterwarnings carefully                       |
| Breaking existing CI                       | Medium      | High   | Test changes in feature branch first                     |
| Dependabot creates too many PRs            | Low         | Low    | Configure grouping and limits                            |
| CodeQL false positives                     | Medium      | Low    | Review findings before enabling blocking                 |

### 6.2 Risk of Inaction

| Risk                                 | Probability | Impact | Consequence                           |
| ------------------------------------ | ----------- | ------ | ------------------------------------- |
| Security vulnerability in dependency | High        | High   | Unpatched vulnerability in production |
| Silent test failures                 | Medium      | High   | Bugs reach production undetected      |
| Deprecated dependency breaks code    | Medium      | Medium | Emergency fixes required              |
| Type errors cause runtime bugs       | Low         | Medium | Unexpected failures in production     |

### 6.3 Rollback Procedures

**Phase 1 Rollback**: Revert ci.yml changes if security scans reveal blocking issues that require extended remediation time.

**Phase 2 Rollback**: Remove test inclusion from hooks if violations are too numerous to address in allocated time. Create tech debt tracking issue.

**Phase 3 Rollback**: If CodeQL produces too many false positives, configure to warning-only mode while tuning.

---

## 7. Success Criteria

### 7.1 Phase-Specific Success Criteria

| Phase   | Criteria                             | Verification                             |
| ------- | ------------------------------------ | ---------------------------------------- |
| Phase 1 | Security scans can fail CI           | Trigger scan on known-vulnerable state   |
| Phase 2 | Test code covered by static analysis | `pre-commit run --all-files` passes      |
| Phase 3 | Infrastructure improvements active   | CodeQL, Codecov dashboards show data     |
| Phase 4 | Best practices hooks running         | `pre-commit run --all-files` passes      |

### 7.2 Overall Success Criteria

- [ ] All 207 tests continue to pass
- [ ] Coverage remains at ≥80%
- [ ] Security scans are blocking (not advisory)
- [ ] Static analysis covers 100% of Python code
- [ ] Pre-commit hooks complete in < 60 seconds
- [ ] No silent test passes possible
- [ ] Dependabot enabled and generating PRs

---

## Appendix

### A. Command Reference

```bash
# Run all tests
pytest juniper_data/tests/ -v

# Run tests with coverage
pytest juniper_data/tests/ --cov=juniper_data --cov-report=term-missing

# Run pre-commit on all files
pre-commit run --all-files

# Check MyPy manually
mypy juniper_data --ignore-missing-imports

# Check Flake8 on tests manually
flake8 juniper_data/tests --max-line-length=512

# Run Bandit manually (blocking)
bandit -r juniper_data -ll -ii --severity-level medium

# Run pip-audit manually (strict)
pip freeze > requirements.txt && pip-audit -r requirements.txt --strict

# Verify test collection
pytest --collect-only -q

# Check for unused imports
flake8 juniper_data --select=F401
```

### B. Files Modified by This Plan

| File                                                      | Phase(s) | Changes                                |
| --------------------------------------------------------- | -------- | -------------------------------------- |
| `.github/workflows/ci.yml`                                | 1, 3     | Security scan commands, action pinning |
| `.github/dependabot.yml`                                  | 1        | New file                               |
| `.github/workflows/codeql.yml`                            | 3        | New file                               |
| `.pre-commit-config.yaml`                                 | 2, 4     | Test inclusion, ignore reduction       |
| `pyproject.toml`                                          | 2, 3     | MyPy config, pytest warnings           |
| `juniper_data/tests/unit/test_main.py`                    | 1        | Remove silent exception handling       |
| `juniper_data/tests/fixtures/generate_golden_datasets.py` | 2        | Remove unused import                   |
| `juniper_data/tests/integration/test_storage_workflow.py` | 2        | Remove unused import                   |
| `juniper_data/tests/unit/test_api_app.py`                 | 2        | Remove unused import                   |
| `juniper_data/tests/unit/test_api_routes.py`              | 2        | Remove unused imports (3)              |
| `juniper_data/core/artifacts.py`                          | 3        | Type annotation fix                    |
| `juniper_data/storage/memory.py`                          | 3        | Type annotation fix                    |
| `juniper_data/storage/local_fs.py`                        | 3        | Type annotation fix                    |
| `juniper_data/api/app.py`                                 | 3        | Type annotation fix                    |
| `juniper_data/api/routes/datasets.py`                     | 3        | Type annotation fix                    |

### C. Reference Documentation

- [GitHub Actions Security Best Practices](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions)
- [Dependabot Configuration](https://docs.github.com/en/code-security/dependabot/dependabot-version-updates/configuration-options-for-the-dependabot.yml-file)
- [CodeQL for Python](https://codeql.github.com/docs/codeql-language-guides/codeql-for-python/)
- [Bandit Documentation](https://bandit.readthedocs.io/)
- [pip-audit Documentation](https://pypi.org/project/pip-audit/)
- [Pre-commit Hooks](https://pre-commit.com/)
- [Codecov Documentation](https://docs.codecov.com/)

### D. Audit Sources

This development plan consolidates findings from:

1. `TEST_SUITE_CICD_ENHANCEMENT_DEVELOPMENT_PLAN_AMP.md` - Audit by AI Code Review Agent (Amp)
2. `TEST_SUITE_CICD_ENHANCEMENT_DEVELOPMENT_PLAN_CLAUDE.md` - Audit by Claude (Opus 4.5)

All findings were independently validated against the source configuration files.

---

**Document Status**: Ready for Review
**Next Action**: Stakeholder approval and Phase 1 scheduling
**Review Cycle**: Weekly progress reviews recommended during implementation

---

**Consolidated by Claude (Opus 4.5) - 2026-02-04:**
