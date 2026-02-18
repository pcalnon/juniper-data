# JuniperData Test Suite & CI/CD Enhancement Development Plan

**Project**: JuniperData - Dataset Generation Service  
**Author**: AI Code Review Agent (Amp)  
**Version**: 1.0.0  
**Created**: 2026-02-03  
**Status**: DRAFT - Pending Review  

---

## Executive Summary

This development plan consolidates findings from two independent test suite and CI/CD audits performed on the JuniperData project. Through rigorous analysis and validation of the original audit conclusions, this plan provides a prioritized roadmap for enhancing test suite effectiveness, CI/CD reliability, and security posture.

### Key Findings Summary

| Category                 | Issues Found | Critical | High  | Medium | Low   |
| ------------------------ | ------------ | -------- | ----- | ------ | ----- |
| Security Scanning        | 4            | 2        | 1     | 1      | 0     |
| Test Suite Integrity     | 3            | 1        | 1     | 1      | 0     |
| Static Analysis Coverage | 4            | 0        | 2     | 2      | 0     |
| Configuration Quality    | 5            | 0        | 1     | 2      | 2     |
| Missing Infrastructure   | 4            | 0        | 1     | 2      | 1     |
| **Total**                | **20**       | **3**    | **6** | **8**  | **3** |

### Overall Risk Assessment

**Current State**: The test suite achieves 100% code coverage with all 207 tests passing. However, the CI/CD pipeline contains **critical security blind spots** where security scanners cannot fail the build, creating a false sense of security. Additionally, static analysis tools deliberately exclude test code, allowing quality issues to accumulate undetected.

**Recommended Action**: Immediate remediation of P0/Critical issues is essential. The security scanning configuration currently defeats its stated purpose.

---

## Table of Contents

1. [Consolidated Issues Registry](#1-consolidated-issues-registry)
2. [Priority Classification](#2-priority-classification)
3. [Detailed Issue Analysis](#3-detailed-issue-analysis)
4. [Development Plan](#4-development-plan)
5. [Implementation Phases](#5-implementation-phases)
6. [Risk Assessment & Mitigation](#6-risk-assessment--mitigation)
7. [Success Criteria](#7-success-criteria)
8. [Appendix](#appendix)

---

## 1. Consolidated Issues Registry

### 1.1 Issues Validated from Audit Reports

| ID      | Issue                                              | Location                    | Validated | Severity     |
| ------- | -------------------------------------------------- | --------------------------- | --------- | ------------ |
| SEC-001 | Bandit failures suppressed with `\|\| true`        | ci.yml:313                  | ✓         | **CRITICAL** |
| SEC-002 | pip-audit vulnerabilities don't fail build         | ci.yml:329                  | ✓         | **CRITICAL** |
| SEC-003 | No Dependabot configuration                        | .github/                    | ✓         | **HIGH**     |
| SEC-004 | GitHub Actions not pinned to SHA                   | ci.yml:70,73,84,etc.        | ✓         | **MEDIUM**   |
| TST-001 | Silent test pass via `except ImportError: pass`    | test_main.py:37-38          | ✓         | **CRITICAL** |
| TST-002 | Tests excluded from Flake8 pre-commit              | .pre-commit-config.yaml:126 | ✓         | **HIGH**     |
| TST-003 | Tests excluded from MyPy pre-commit                | .pre-commit-config.yaml:139 | ✓         | **HIGH**     |
| TST-004 | Pytest warnings globally suppressed                | pyproject.toml:119          | ✓         | **MEDIUM**   |
| CFG-001 | Excessive Flake8 ignore rules (E722, F401)         | .pre-commit-config.yaml:118 | ✓         | **MEDIUM**   |
| CFG-002 | 512-character line length (excessively permissive) | Multiple files              | ✓         | **LOW**      |
| CFG-003 | MyPy type errors in production code (6 errors)     | 5 source files              | ✓         | **MEDIUM**   |
| CFG-004 | Tests excluded from MyPy in pyproject.toml         | pyproject.toml:182          | ✓         | **MEDIUM**   |
| INF-001 | No CodeQL analysis configured                      | .github/workflows/          | ✓         | **MEDIUM**   |
| INF-002 | No coverage upload to external service             | ci.yml                      | ✓         | **LOW**      |
| INF-003 | No scheduled slow test job                         | ci.yml                      | ✓         | **LOW**      |
| INF-004 | SARIF upload uses continue-on-error                | ci.yml:320                  | ✓         | **LOW**      |

### 1.2 Flake8 Violations in Test Code (Currently Undetected)

Running `flake8 juniper_data/tests` revealed **33+ issues** that are currently hidden due to test exclusion:

| Category                        | Count  | Impact                   |
| ------------------------------- | ------ | ------------------------ |
| F401 (unused imports)           | 6      | Dead code accumulation   |
| F541 (empty f-strings)          | 5      | Potential bugs           |
| E402 (import order)             | 1      | Style inconsistency      |
| SIM117 (nested with statements) | 21     | Reduced readability      |
| **Total**                       | **33** | Code quality degradation |

### 1.3 MyPy Type Errors in Production Code

| File                     | Line | Error                                      | Type       |
| ------------------------ | ---- | ------------------------------------------ | ---------- |
| `core/artifacts.py`      | 18   | `savez` type signature mismatch            | arg-type   |
| `core/artifacts.py`      | 44   | `savez` type signature mismatch            | arg-type   |
| `storage/memory.py`      | 65   | `savez_compressed` type signature mismatch | arg-type   |
| `storage/local_fs.py`    | 77   | `savez_compressed` type signature mismatch | arg-type   |
| `api/routes/datasets.py` | 19   | Incompatible None assignment               | assignment |
| `api/app.py`             | 40   | Implicit Optional not allowed              | assignment |

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
│  └── TST-001: Silent ImportError pass in test_main.py                   │
│                                                                         │
│  P1 HIGH (1 Week)                                                       │
│  ├── SEC-003: Missing Dependabot configuration                          │
│  ├── TST-002: Tests excluded from Flake8                                │
│  └── TST-003: Tests excluded from MyPy                                  │
│                                                                         │
│  P2 MEDIUM (2 Weeks)                                                    │
│  ├── SEC-004: GitHub Actions not pinned to SHA                          │
│  ├── TST-004: Pytest warnings suppressed                                │
│  ├── CFG-001: Excessive Flake8 ignores                                  │
│  ├── CFG-003: MyPy production code errors                               │
│  ├── CFG-004: Tests excluded from pyproject.toml MyPy                   │
│  └── INF-001: Missing CodeQL analysis                                   │
│                                                                         │
│  P3 LOW (1 Month)                                                       │
│  ├── CFG-002: 512-char line length                                      │
│  ├── INF-002: No coverage upload service                                │
│  ├── INF-003: No slow test job                                          │
│  └── INF-004: SARIF upload continue-on-error                            │
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
    bandit -r juniper_data -f sarif -o reports/security/bandit.sarif --exit-zero
```

The `--exit-zero` flag will still exit 0 on findings (for SARIF generation), but will exit non-zero on actual Bandit errors. Alternatively, configure baseline exclusions for known-acceptable findings.

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

For graduated adoption, use `--ignore-vuln` for known-acceptable vulnerabilities or `--only-high-severity` initially.

**Effort**: Small (< 1 hour, plus remediation time if vulnerabilities exist)

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

**Recommended Fix**:

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

Alternatively, use `pytest.importorskip()` if the test should be skipped when uvicorn is unavailable:

```python
@pytest.mark.skipif(
    importlib.util.find_spec("uvicorn") is None,
    reason="uvicorn not installed"
)
def test_main_import_error_uvicorn_not_installed(self):
    ...
```

**Effort**: Small (< 1 hour)

---

### 3.2 High Priority Issues (P1)

#### SEC-003: Missing Dependabot Configuration

**Location**: `.github/` directory

**Current State**: No `dependabot.yml` file exists.

**Problem**: Without Dependabot:

- Dependencies drift into vulnerable states silently
- Manual monitoring of security advisories required
- GitHub Actions versions may become outdated

**Recommended Fix**: Create `.github/dependabot.yml`:

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

#### TST-002: Tests Excluded from Flake8 Pre-commit

**Location**: `.pre-commit-config.yaml` line 126

**Current State**:

```yaml
files: ^juniper_data/.*\.py$
exclude: ^juniper_data/tests/
```

**Problem**: 33+ flake8 violations exist in test code that are invisible to developers. Issues include:

- 6 unused imports (F401) - dead code
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

**Effort**: Medium (1-2 hours, may require fixing type issues in tests)

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
    "--continue-on-collection-errors",
    "--tb=short",
]
filterwarnings = [
    "ignore::DeprecationWarning:uvicorn.*",
    "ignore::PendingDeprecationWarning",
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

**Effort**: Medium (2-4 hours, depends on codebase violations)

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

**Effort**: Small-Medium (1-2 hours)

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
    - cron: '0 6 * * 1'

jobs:
  analyze:
    name: Analyze
    runs-on: ubuntu-latest
    permissions:
      security-events: write
    steps:
      - uses: actions/checkout@v4
      - uses: github/codeql-action/init@v3
        with:
          languages: python
      - uses: github/codeql-action/analyze@v3
```

**Effort**: Small (< 1 hour)

---

### 3.4 Low Priority Issues (P3)

#### CFG-002: 512-Character Line Length

**Location**: Multiple configuration files

**Assessment**: While unusually permissive, this is a team preference issue. No immediate action required unless it's actively harming code review.

**Effort**: Small (configuration change only)

---

#### INF-002: No Coverage Upload to External Service

**Problem**: Coverage reports are generated but not uploaded to services like Codecov or Coveralls for tracking trends.

**Recommended Fix**: Add Codecov upload step:

```yaml
- name: Upload coverage to Codecov
  uses: codecov/codecov-action@v4
  with:
    files: reports/coverage.xml
    fail_ci_if_error: false
```

**Effort**: Small (< 30 minutes)

---

#### INF-003: No Scheduled Slow Test Job

**Problem**: The `slow` marker exists but no tests use it, and no scheduled job runs slow tests.

**Recommendation**: Document the slow test policy. Add scheduled job when slow tests are introduced.

**Effort**: Small (documentation only for now)

---

## 4. Development Plan

### 4.1 Phase Overview

```bash
┌─────────────────────────────────────────────────────────────────────────┐
│                    DEVELOPMENT PHASES                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  PHASE 1: Security Remediation (Critical)          Duration: 1-2 days   │
│  ├── SEC-001: Fix Bandit || true                                        │
│  ├── SEC-002: Fix pip-audit warning handling                            │
│  └── TST-001: Fix silent ImportError test                               │
│                                                                         │
│  PHASE 2: Infrastructure Hardening                 Duration: 3-5 days   │
│  ├── SEC-003: Add Dependabot configuration                              │
│  ├── SEC-004: Pin GitHub Actions to SHA                                 │
│  └── INF-001: Add CodeQL analysis                                       │
│                                                                         │
│  PHASE 3: Static Analysis Coverage                Duration: 1 week      │
│  ├── TST-002: Enable Flake8 on tests                                    │
│  ├── TST-003: Enable MyPy on tests                                      │
│  ├── CFG-001: Reduce Flake8 ignores                                     │
│  └── CFG-003: Fix MyPy production errors                                │
│                                                                         │
│  PHASE 4: Configuration Quality                   Duration: 1 week      │
│  ├── TST-004: Enable pytest warnings                                    │
│  ├── CFG-004: Update pyproject.toml MyPy config                         │
│  └── Test code cleanup (33+ violations)                                 │
│                                                                         │
│  PHASE 5: Enhancements (Optional)                 Duration: Ongoing     │
│  ├── INF-002: Add coverage upload                                       │
│  ├── INF-003: Document slow test policy                                 │
│  └── CFG-002: Review line length policy                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Effort Estimates

| Phase     | Issues | Estimated Effort | Expected Duration |
| --------- | ------ | ---------------- | ----------------- |
| Phase 1   | 3      | 2-4 hours        | 1-2 days          |
| Phase 2   | 3      | 2-3 hours        | 3-5 days          |
| Phase 3   | 4      | 6-10 hours       | 5-7 days          |
| Phase 4   | 3      | 4-6 hours        | 5-7 days          |
| Phase 5   | 3      | 2-3 hours        | Ongoing           |
| **Total** | **16** | **16-26 hours**  | **2-4 weeks**     |

**Note**: Effort estimates assume no blocking issues (e.g., many security vulnerabilities requiring remediation). Actual duration may vary based on findings.

---

## 5. Implementation Phases

### 5.1 Phase 1: Security Remediation (CRITICAL)

**Objective**: Ensure security scanners can actually fail the build.

**Priority**: P0 - Immediate  
**Duration**: 1-2 days  
**Effort**: 2-4 hours

#### Tasks

| Task ID | Description                                               | Owner | Status |
| ------- | --------------------------------------------------------- | ----- | ------ |
| P1-T1   | Update Bandit command to use `--exit-zero` flag           | TBD   | ☐      |
| P1-T2   | Update pip-audit to use `--strict` flag                   | TBD   | ☐      |
| P1-T3   | Refactor test_main.py to remove silent exception handling | TBD   | ☐      |
| P1-T4   | Run security scans to verify failures are detected        | TBD   | ☐      |
| P1-T5   | Address any immediate security findings                   | TBD   | ☐      |

#### Acceptance Criteria

- [ ] Bandit security findings cause CI to produce warnings (not silent success)
- [ ] pip-audit vulnerabilities cause CI to fail
- [ ] test_main.py no longer silently passes on ImportError
- [ ] All tests still pass (207/207)

#### Rollback Plan

If security scans reveal many issues, temporarily use `--exit-zero` for Bandit and `--ignore-vuln` for known issues while remediating.

---

### 5.2 Phase 2: Infrastructure Hardening

**Objective**: Add missing security infrastructure and supply-chain protections.

**Priority**: P1 - High  
**Duration**: 3-5 days  
**Effort**: 2-3 hours

#### Tasks: 5.2, Phase 2

| Task ID | Description                               | Owner | Status |
| ------- | ----------------------------------------- | ----- | ------ |
| P2-T1   | Create `.github/dependabot.yml`           | TBD   | ☐      |
| P2-T2   | Pin all GitHub Actions to commit SHAs     | TBD   | ☐      |
| P2-T3   | Create CodeQL analysis workflow           | TBD   | ☐      |
| P2-T4   | Verify Dependabot PRs are generated       | TBD   | ☐      |
| P2-T5   | Verify CodeQL scans complete successfully | TBD   | ☐      |

#### Acceptance Criteria: 5.2, Phase 2

- [ ] Dependabot PRs appear for outdated dependencies
- [ ] All GitHub Actions pinned to specific commit SHAs
- [ ] CodeQL scans run on push to main/develop
- [ ] Security tab shows CodeQL findings

---

### 5.3 Phase 3: Static Analysis Coverage

**Objective**: Extend static analysis to cover test code.

**Priority**: P1/P2 - High/Medium  
**Duration**: 5-7 days  
**Effort**: 6-10 hours

#### Tasks: 5.3, Phase 3

| Task ID | Description                                  | Owner | Status |
| ------- | -------------------------------------------- | ----- | ------ |
| P3-T1   | Add Flake8 hook for tests with relaxed rules | TBD   | ☐      |
| P3-T2   | Fix F401 (unused imports) in tests           | TBD   | ☐      |
| P3-T3   | Fix F541 (empty f-strings) in tests          | TBD   | ☐      |
| P3-T4   | Update MyPy hook to include tests            | TBD   | ☐      |
| P3-T5   | Remove E722 from Flake8 ignores              | TBD   | ☐      |
| P3-T6   | Remove F401 from Flake8 ignores              | TBD   | ☐      |
| P3-T7   | Fix 6 MyPy errors in production code         | TBD   | ☐      |
| P3-T8   | Run pre-commit on all files to verify        | TBD   | ☐      |

#### Acceptance Criteria: 5.3, Phase 3

- [ ] Flake8 runs on test files in pre-commit
- [ ] No F401 or F541 violations in tests
- [ ] MyPy runs on test files in pre-commit
- [ ] E722 and F401 removed from global ignores
- [ ] All 6 production MyPy errors fixed
- [ ] Pre-commit passes on all files

---

### 5.4 Phase 4: Configuration Quality

**Objective**: Remove signal suppression and improve configuration hygiene.

**Priority**: P2 - Medium  
**Duration**: 5-7 days  
**Effort**: 4-6 hours

#### Tasks: 5.4, Phase 4

| Task ID | Description                                    | Owner | Status |
| ------- | ---------------------------------------------- | ----- | ------ |
| P4-T1   | Remove `-p no:warnings` from pytest config     | TBD   | ☐      |
| P4-T2   | Add selective filterwarnings configuration     | TBD   | ☐      |
| P4-T3   | Fix SIM117 violations (nested with statements) | TBD   | ☐      |
| P4-T4   | Update pyproject.toml MyPy exclude pattern     | TBD   | ☐      |
| P4-T5   | Run full test suite to verify no regressions   | TBD   | ☐      |

#### Acceptance Criteria: 5.4, Phase 4

- [ ] Pytest shows relevant warnings (not all suppressed)
- [ ] No unexpected warnings causing noise
- [ ] Test code readability improved (SIM117 fixed)
- [ ] All configuration files consistent

---

### 5.5 Phase 5: Enhancements (Optional)

**Objective**: Add nice-to-have improvements.

**Priority**: P3 - Low  
**Duration**: Ongoing  
**Effort**: 2-3 hours

#### Tasks: 5.5, Phase 5

| Task ID | Description                               | Owner | Status |
| ------- | ----------------------------------------- | ----- | ------ |
| P5-T1   | Add Codecov/Coveralls integration         | TBD   | ☐      |
| P5-T2   | Document slow test policy in AGENTS.md    | TBD   | ☐      |
| P5-T3   | Review and potentially reduce line length | TBD   | ☐      |
| P5-T4   | Consider migration from flake8 to ruff    | TBD   | ☐      |

---

## 6. Risk Assessment & Mitigation

### 6.1 Implementation Risks

| Risk                                       | Probability | Impact | Mitigation                                               |
| ------------------------------------------ | ----------- | ------ | -------------------------------------------------------- |
| Security scans reveal many vulnerabilities | Medium      | High   | Use graduated enforcement (`--only-high-severity` first) |
| Flake8 on tests creates excessive churn    | High        | Medium | Start with F401/F541 only, expand gradually              |
| MyPy on tests is too noisy                 | Medium      | Medium | Use `--allow-untyped-defs` for tests initially           |
| Warnings flood pytest output               | Medium      | Low    | Configure filterwarnings carefully                       |
| Pre-commit hooks slow down                 | Low         | Low    | Monitor performance, adjust as needed                    |

### 6.2 Rollback Procedures

**Phase 1 Rollback**: Revert ci.yml changes if security scans reveal blocking issues that require extended remediation time.

**Phase 3 Rollback**: Remove test inclusion from hooks if violations are too numerous to address in allocated time. Create tech debt tracking issue.

---

## 7. Success Criteria

### 7.1 Phase-Specific Success Criteria

| Phase   | Criteria                             | Verification                            |
| ------- | ------------------------------------ | --------------------------------------- |
| Phase 1 | Security scans can fail CI           | Trigger scan on known-vulnerable state  |
| Phase 2 | Automated dependency monitoring      | Verify Dependabot PR generation         |
| Phase 3 | Test code covered by static analysis | `pre-commit run --all-files` passes     |
| Phase 4 | Warnings visible without noise       | Review pytest output for signal quality |

### 7.2 Overall Success Criteria

- [ ] All 207 tests continue to pass
- [ ] Coverage remains at ≥80%
- [ ] Security scans are blocking (not advisory)
- [ ] Static analysis covers 100% of Python code
- [ ] Pre-commit hooks complete in < 60 seconds
- [ ] No silent test passes possible

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

# Run Bandit manually
bandit -r juniper_data -f screen

# Run pip-audit manually
pip freeze > requirements.txt && pip-audit -r requirements.txt

# Verify test collection
pytest --collect-only -q
```

### B. Files Modified by This Plan

| File                                   | Phase | Changes                                |
| -------------------------------------- | ----- | -------------------------------------- |
| `.github/workflows/ci.yml`             | 1, 2  | Security scan commands, action pinning |
| `.github/dependabot.yml`               | 2     | New file                               |
| `.github/workflows/codeql.yml`         | 2     | New file                               |
| `.pre-commit-config.yaml`              | 3     | Test inclusion, ignore reduction       |
| `pyproject.toml`                       | 3, 4  | MyPy config, pytest warnings           |
| `juniper_data/tests/unit/test_main.py` | 1     | Remove silent exception handling       |
| `juniper_data/api/app.py`              | 3     | Type annotation fix                    |
| `juniper_data/api/routes/datasets.py`  | 3     | Type annotation fix                    |
| Multiple test files                    | 3, 4  | Unused imports, f-strings, nested with |

### C. Reference Documentation

- [GitHub Actions Security Best Practices](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions)
- [Dependabot Configuration](https://docs.github.com/en/code-security/dependabot/dependabot-version-updates/configuration-options-for-the-dependabot.yml-file)
- [CodeQL for Python](https://codeql.github.com/docs/codeql-language-guides/codeql-for-python/)
- [Bandit Documentation](https://bandit.readthedocs.io/)
- [pip-audit Documentation](https://pypi.org/project/pip-audit/)
- [Pre-commit Hooks](https://pre-commit.com/)

### D. Audit Sources

This development plan consolidates findings from:

1. `TEST_SUITE_AUDIT_REPORT.md` - Initial audit by Claude (Opus 4.5)
2. `TEST_SUITE_CICD_AUDIT_REPORT_20260203_170839.md` - Comprehensive audit by AI Code Review Agent

All findings were independently validated against the source configuration files.

---

**Document Status**: Ready for Review  
**Next Action**: Stakeholder approval and Phase 1 scheduling  
**Review Cycle**: Weekly progress reviews recommended during implementation

---

**Generated by Amp AI Agent - 2026-02-03.**
