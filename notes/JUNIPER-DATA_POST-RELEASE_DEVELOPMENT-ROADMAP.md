# JuniperData Post-Release Development Roadmap

**Project**: JuniperData - Dataset Generation Microservice
**Version**: 0.4.2 (Current Release)
**Created**: 2026-02-17
**Author**: Paul Calnon
**Status**: Active - Post-Release Assessment
**Audit Date**: 2026-02-17

---

## Executive Summary

This document consolidates all outstanding updates, changes, fixes, and enhancements for the JuniperData application identified through a comprehensive audit of the project notes directory and codebase validation. Items were extracted from:

1. **INTEGRATION_DEVELOPMENT_PLAN.md** — Deferred items DATA-018, DATA-019, DATA-020
2. **PRE-DEPLOYMENT_ROADMAP.md** — Cross-project integration items
3. **PRE-DEPLOYMENT_ROADMAP-2.md** — Section 7 enhancements (CAS-001–010, CAN-000–021) and deferred items
4. **INTEGRATION_ROADMAP.md** — Integration-specific issues
5. **TEST_SUITE_CICD_ENHANCEMENT_DEVELOPMENT_PLAN.md** — Deferred CI/CD improvements
6. **RELEASE_NOTES_v0.4.2.md** — Known issues and planned v0.5.0 items
7. **Codebase Validation** — Direct inspection of source code against documented state

### Current State (Validated 2026-02-17)

| Metric                           | Documented                | Validated             | Status                     |
| -------------------------------- | ------------------------- | --------------------- | -------------------------- |
| Version                          | 0.4.2                     | 0.4.2                 | Correct                    |
| Generators in GENERATOR_REGISTRY | 5 (per release notes)     | **8**                 | **Release notes outdated** |
| Storage Backends                 | 7                         | 7 (+ base + **init**) | Correct                    |
| Service Tests                    | 658                       | 658 (30 files)        | Correct                    |
| Client Tests                     | 41                        | 41                    | Correct                    |
| Client Coverage                  | 96%                       | 96%                   | Correct                    |
| Security (auth + rate limiting)  | Complete                  | Complete              | Correct                    |
| Lifecycle Management             | Complete                  | Complete              | Correct                    |
| Dockerfile                       | Complete                  | Present               | Correct                    |
| `.github/dependabot.yml`         | Complete (per audit plan) | **Missing**           | **Not created**            |
| `.github/workflows/codeql.yml`   | Complete                  | Present               | Correct                    |
| docs/api/JUNIPER_DATA_API.md     | Complete                  | Present               | Correct                    |

### Discrepancy Summary

| Item                     | Documented State                 | Actual State           | Action Required         |
| ------------------------ | -------------------------------- | ---------------------- | ----------------------- |
| GENERATOR_REGISTRY count | Release notes say 5 of 8         | All 8 registered       | Update release notes    |
| Coverage reporting       | INTEGRATION_DEV_PLAN says 95.18% | Release notes say ~60% | Verify current coverage |
| Dependabot configuration | TEST_SUITE plan says Complete    | File does not exist    | Create dependabot.yml   |
| v0.5.0 planned items     | "Register remaining generators"  | Already done           | Update v0.5.0 plan      |

---

## Table of Contents

1. [Phase 1: Documentation & Housekeeping](#phase-1-documentation--housekeeping)
2. [Phase 2: Test Coverage & Quality](#phase-2-test-coverage--quality)
3. [Phase 3: Client Library Publication](#phase-3-client-library-publication)
4. [Phase 4: CI/CD Tooling Modernization](#phase-4-cicd-tooling-modernization)
5. [Phase 5: Advanced Capabilities](#phase-5-advanced-capabilities)
6. [Validation Results](#validation-results)
7. [Design Analysis](#design-analysis)
8. [Cross-Project References](#cross-project-references)

---

## Phase 1: Documentation & Housekeeping

**Priority**: HIGH | **Risk**: LOW | **Effort**: Small (2-4 hours total)
**Rationale**: Quick wins that correct documentation discrepancies and close outstanding housekeeping items.

### RD-001: Update Release Notes Known Issues

**Priority**: HIGH | **Status**: NOT STARTED | **Effort**: Small (30 min)
**Source**: Codebase validation audit (2026-02-17)

**Problem**: The v0.4.2 release notes contain two outdated known issues:

1. "GENERATOR_REGISTRY: 5 of 8 generators registered" — **All 8 are now registered**
2. "Coverage at ~60% overall" — Needs verification against current state

**Required Actions**:

- [ ] Update `notes/releases/RELEASE_NOTES_v0.4.2.md` known issues section
- [ ] Remove GENERATOR_REGISTRY known issue (resolved)
- [ ] Verify current coverage percentage and update accordingly

**Validation**: All 8 generators confirmed registered in `api/routes/generators.py`: spiral, xor, gaussian, circles, checkerboard, csv_import, mnist, arc_agi.

---

### RD-002: Create Dependabot Configuration

**Priority**: HIGH | **Status**: NOT STARTED | **Effort**: Small (15 min)
**Source**: TEST_SUITE_CICD_ENHANCEMENT_DEVELOPMENT_PLAN.md (SEC-003)

**Problem**: The test suite audit identified missing Dependabot configuration as CRITICAL. The consolidated plan marks it as COMPLETED, but the file does not exist at `.github/dependabot.yml`.

**Required Actions**:

- [ ] Create `.github/dependabot.yml` for pip and github-actions ecosystems
- [ ] Configure weekly schedule with grouped minor/patch updates
- [ ] Set open-pull-requests-limit to prevent PR flooding

**Validation**: Confirmed `.github/` directory contains only `ci.yml` and `codeql.yml` — no `dependabot.yml`.

**Feasibility**: Straightforward. No risk. Standard GitHub configuration file.

---

### RD-003: Verify and Document CodeQL Scan Status

**Priority**: MEDIUM | **Status**: PENDING VERIFICATION | **Effort**: Small (30 min)
**Source**: TEST_SUITE_CICD_ENHANCEMENT_DEVELOPMENT_PLAN.md (P3-T8)

**Problem**: CodeQL workflow exists but has not been verified as completing successfully on GitHub.

**Required Actions**:

- [ ] Verify CodeQL scans complete on the repository
- [ ] Check for any CodeQL findings that need addressing
- [ ] Update documentation with verification results

**Validation**: `.github/workflows/codeql.yml` exists (2,078 bytes).

---

### RD-004: Update v0.5.0 Planned Items

**Priority**: MEDIUM | **Status**: NOT STARTED | **Effort**: Small (30 min)
**Source**: RELEASE_NOTES_v0.4.2.md (What's Next section)

**Problem**: The v0.5.0 plan includes "Register remaining generators (csv_import, mnist, arc_agi) in GENERATOR_REGISTRY" — this is already done. The plan needs revision.

**Required Actions**:

- [ ] Update v0.5.0 planned items to reflect actual remaining work
- [ ] Cross-reference with this roadmap for comprehensive feature list

---

### RD-005: Reconcile Coverage Metrics

**Priority**: HIGH | **Status**: NOT STARTED | **Effort**: Small (1 hour)
**Source**: Cross-document discrepancy

**Problem**: Conflicting coverage metrics across documents:

- `INTEGRATION_DEVELOPMENT_PLAN.md` states 95.18% (updated 2026-02-12)
- `RELEASE_NOTES_v0.4.2.md` states "~60% overall" with 6 modules at 0%
- Release notes say "80%+ (meets threshold)"

**Required Actions**:

- [ ] Run `pytest --cov=juniper_data --cov-report=term-missing` to get authoritative current coverage
- [ ] Update all documentation with consistent, validated metrics
- [ ] Identify which modules truly remain at low/zero coverage

**Security/Best Practice Note**: Accurate coverage metrics are essential for informed decision-making. Stale or conflicting metrics create false confidence.

---

## Phase 2: Test Coverage & Quality

**Priority**: HIGH | **Risk**: LOW-MEDIUM | **Effort**: Medium-Large (8-20 hours total)
**Rationale**: Addresses known quality gaps and security testing shortfalls.

### RD-006: Add Security-Focused Test Suite

**Priority**: HIGH | **Status**: NOT STARTED | **Effort**: Medium (4-6 hours)
**Source**: TEST_SUITE_AUDIT_DATA_CLAUDE.md (Section 1.8), TEST_SUITE_AUDIT_DATA_AMP_.md

**Problem**: Both independent audits identified missing security-focused tests. While Pydantic validation and API input validation exist, explicit security boundary tests are absent.

**Required Actions**:

- [ ] Create `juniper_data/tests/unit/test_security_boundaries.py`
- [ ] Add path traversal prevention tests for storage backends
- [ ] Add dataset ID injection prevention tests
- [ ] Add API request size limit tests
- [ ] Add parameter bound enforcement tests (extreme values)
- [ ] Add resource exhaustion protection tests (very large datasets)

**Design Options**:

1. **Option A (Recommended)**: Create a single `test_security_boundaries.py` with test classes per attack vector
2. **Option B**: Integrate security tests into existing test files alongside related functionality

**Validation**: Confirmed `test_security.py` exists for API security (auth/rate limiting), but no boundary/injection tests exist.

**Feasibility**: Fully feasible. Uses existing test infrastructure. Tests should be deterministic and fast.

---

### RD-007: Improve Coverage for Low-Coverage Modules

**Priority**: MEDIUM | **Status**: NOT STARTED | **Effort**: Large (8-12 hours)
**Source**: RELEASE_NOTES_v0.4.2.md (Known Issues), INTEGRATION_DEVELOPMENT_PLAN.md

**Problem**: Six modules reported at 0% coverage due to external dependency mocking gaps:

- `generators/arc_agi/` — Requires `arc-agi` package
- `generators/mnist/` — Requires HuggingFace `datasets` package
- `storage/hf_store.py` — Requires HuggingFace Hub
- `storage/kaggle_store.py` — Requires Kaggle API
- `storage/postgres_store.py` — Requires PostgreSQL
- `storage/redis_store.py` — Requires Redis

**Required Actions**:

- [ ] Verify which modules have tests but don't count toward coverage (source_pkgs config issue)
- [ ] Add/improve mock-based tests for external dependency modules
- [ ] Ensure coverage configuration (`pyproject.toml`) correctly reports all modules
- [ ] Target 80%+ per module

**Design Options**:

1. **Option A (Recommended)**: Use `unittest.mock` / `responses` library to mock external APIs. Tests already exist for most modules (test_hf_store.py, test_kaggle_store.py, etc.) — the issue may be in coverage configuration.
2. **Option B**: Use `pytest-xdist` with conditional test environments that have real dependencies installed.
3. **Option C**: Create fixture-based integration test environment with Docker containers for Redis, PostgreSQL.

**Validation**: Test files exist for all 6 modules (test_arc_agi_generator.py, test_mnist_generator.py, test_hf_store.py, test_kaggle_store.py, test_postgres_store.py, test_redis_store.py). The 0% coverage likely reflects a `source_pkgs` configuration issue, not truly untested code.

**Feasibility**: High. Tests likely exist but may not be counted. Configuration fix may resolve most gaps.

---

### RD-008: Fix SIM117 Test Code Violations

**Priority**: LOW | **Status**: DEFERRED | **Effort**: Small-Medium (2-3 hours)
**Source**: TEST_SUITE_CICD_ENHANCEMENT_DEVELOPMENT_PLAN.md (P4-T3)

**Problem**: 21 SIM117 violations (nested `with` statements) exist in test code. Currently suppressed via `.flake8` config.

**Required Actions**:

- [ ] Combine nested `with` statements where Python version allows (3.11+)
- [ ] Review each instance for readability trade-offs

**Validation**: `.flake8` config confirms SIM117 in extend-ignore list.

**Feasibility**: Straightforward refactoring. No functional impact. May improve test readability.

---

### RD-009: Performance Test Infrastructure

**Priority**: LOW | **Status**: NOT STARTED | **Effort**: Medium (3-5 hours)
**Source**: TEST_SUITE_CICD_ENHANCEMENT_DEVELOPMENT_PLAN.md (INF-005), TEST_SUITE_AUDIT_DATA_AMP_.md

**Problem**: The `performance` marker is defined in `pyproject.toml` but no performance tests exist and no infrastructure supports them.

**Required Actions**:

- [ ] Create `tests/performance/` directory with benchmark tests
- [ ] Add `pytest-benchmark` to dev dependencies
- [ ] Create baseline benchmarks for generator performance (points/second)
- [ ] Create baseline benchmarks for storage throughput (datasets/second)
- [ ] Add performance regression detection to CI (optional)

**Design Options**:

1. **Option A (Recommended)**: Use `pytest-benchmark` with `--benchmark-autosave` for regression detection
2. **Option B**: Custom timing fixtures with threshold assertions
3. **Option C**: External profiling scripts using py-spy (already documented in Cascor)

**Feasibility**: Feasible. No external dependencies required beyond `pytest-benchmark`. Provides ongoing regression detection.

---

## Phase 3: Client Library Publication

**Priority**: MEDIUM | **Risk**: MEDIUM | **Effort**: Medium (4-8 hours total)
**Rationale**: Consolidates the shared client package and removes duplicated code in consumer projects.

### RD-010: Publish juniper-data-client to PyPI

**Priority**: MEDIUM | **Status**: NOT STARTED | **Effort**: Medium (2-4 hours)
**Source**: RELEASE_NOTES_v0.4.2.md (What's Next), INTEGRATION_DEVELOPMENT_PLAN.md (DATA-012 Next Steps)

**Problem**: `juniper-data-client` exists as a local package but is not published to PyPI. Both JuniperCascor and JuniperCanopy have been updated to use the shared package, but installation requires local path references.

**Required Actions**:

- [ ] Verify `juniper_data_client/pyproject.toml` has correct PyPI metadata
- [ ] Set up PyPI account/token for publishing
- [ ] Create GitHub Actions workflow for automated publishing on release tags
- [ ] Test installation from PyPI in clean environment
- [ ] Update consumer projects to reference PyPI package instead of local path

**Design Options**:

1. **Option A: Public PyPI** — Simplest approach, suitable for MIT-licensed project
2. **Option B: Private PyPI (e.g., AWS CodeArtifact)** — For private/internal use
3. **Option C: Git dependency** — Use `pip install git+https://...` for now, publish later

**Security Note**: Ensure no credentials or internal URLs are included in the published package. Verify `pyproject.toml` excludes test fixtures and notes.

**Feasibility**: Fully feasible. Package structure is already pip-installable. Primary effort is PyPI account setup and CI integration.

---

### RD-011: Update Consumer Projects to Use Published Client

**Priority**: MEDIUM | **Status**: NOT STARTED | **Effort**: Small (1-2 hours)
**Source**: INTEGRATION_DEVELOPMENT_PLAN.md (DATA-012-A)

**Problem**: JuniperCascor and JuniperCanopy re-export from the shared package with local fallback. Once the package is published, they should reference the PyPI version.

**Required Actions**:

- [ ] Update JuniperCascor `pyproject.toml` to add `juniper-data-client` as dependency
- [ ] Update JuniperCanopy `pyproject.toml` to add `juniper-data-client` as dependency
- [ ] Remove local fallback code from both consumers
- [ ] Test import chains in both consumers
- [ ] Remove duplicated `juniper_data_client/` directories from consumers (if still present)

**Dependencies**: RD-010 (publish to PyPI) must be completed first.

---

## Phase 4: CI/CD Tooling Modernization

**Priority**: LOW | **Risk**: LOW | **Effort**: Medium (4-8 hours total)
**Rationale**: Improves development velocity and maintainability but not functionally critical.

### RD-012: Consider Migration from flake8 to ruff

**Priority**: LOW | **Status**: DEFERRED | **Effort**: Medium (3-5 hours)
**Source**: TEST_SUITE_CICD_ENHANCEMENT_DEVELOPMENT_PLAN.md (P4-T5), TEST_SUITE_AUDIT_DATA_AMP_.md

**Problem**: flake8 with multiple plugins (bugbear, etc.) is slower than ruff and requires more configuration. ruff is a drop-in replacement that combines flake8, isort, pyupgrade, and more.

**Required Actions**:

- [ ] Evaluate ruff compatibility with current flake8 rules
- [ ] Create `ruff.toml` or `[tool.ruff]` section in `pyproject.toml`
- [ ] Replace flake8 + isort hooks in `.pre-commit-config.yaml` with ruff
- [ ] Verify CI pipeline works with ruff
- [ ] Remove flake8 configuration files

**Design Options**:

1. **Option A (Recommended)**: Gradual migration — run ruff alongside flake8 initially
2. **Option B**: Full cutover — replace all flake8/isort with ruff in one change
3. **Option C**: Stay with flake8 — current setup works, migration is optional

**Feasibility**: High. ruff supports all configured flake8 rules. Migration is typically straightforward.

---

### RD-013: Review Line Length Configuration

**Priority**: LOW | **Status**: DEFERRED | **Effort**: Small (1-2 hours)
**Source**: TEST_SUITE_CICD_ENHANCEMENT_DEVELOPMENT_PLAN.md (P4-T4), CFG-002

**Problem**: Line length is set to 512 characters in `pyproject.toml` (black/isort) and `.flake8`. The CLAUDE.md says 120. 512 is excessively permissive and effectively disables line-length enforcement.

**Required Actions**:

- [ ] Decide on target line length (120 is standard, 88 is black default)
- [ ] Run black with chosen line length to identify affected files
- [ ] Update `pyproject.toml`, `.flake8`, and `.pre-commit-config.yaml`
- [ ] Fix any resulting formatting issues

**Validation**: Confirmed `pyproject.toml` uses 512 for black and isort. `.flake8` also uses 512. CLAUDE.md documents 120.

**Best Practice Note**: 120 characters is widely accepted for modern development. 512 defeats the purpose of line length enforcement.

---

### RD-014: Add Documentation Build Step to CI

**Priority**: LOW | **Status**: NOT STARTED | **Effort**: Small (1-2 hours)
**Source**: TEST_SUITE_CICD_ENHANCEMENT_DEVELOPMENT_PLAN.md (L-04)

**Problem**: Documentation is not validated in CI, allowing documentation drift.

**Required Actions**:

- [ ] Add documentation link validation step to CI workflow
- [ ] Consider mkdocs or sphinx for future documentation generation
- [ ] Validate markdown link integrity

**Feasibility**: Feasible when documentation volume justifies the effort. Currently low priority.

---

## Phase 5: Advanced Capabilities

**Priority**: LOW | **Risk**: MEDIUM-HIGH | **Effort**: Extra Large (weeks each)
**Rationale**: Major features deferred until foundational work stabilizes and demand materializes.

### RD-015: IPC Architecture (DATA-018)

**Priority**: LOW | **Status**: DEFERRED | **Effort**: XL (2-4 weeks)
**Source**: INTEGRATION_DEVELOPMENT_PLAN.md, PRE-DEPLOYMENT_ROADMAP-2.md (P1-NEW-001)

**Problem**: Integration is REST-only via HTTP. Full IPC architecture would add:

- gRPC support for high-performance binary streaming
- Message queue integration for async dataset generation
- Shared memory for co-located services

**Decision**: Deferred until REST performance becomes a bottleneck. Current HTTP+NPZ approach is sufficient for research workloads.

**Revisit Criteria**:

- Dataset sizes exceed 100MB regularly
- Multiple concurrent generation requests cause queuing
- Co-located services need sub-millisecond data transfer

**Design Options**:

1. **Option A: gRPC** — Best for binary streaming. Requires protobuf schema definition. Supports bidirectional streaming for progress updates.
2. **Option B: Redis Pub/Sub** — Lightweight, good for status broadcasting. Not ideal for large data transfer.
3. **Option C: Shared Memory (multiprocessing.shared_memory)** — Fastest for co-located services. Requires process coordination.
4. **Option D: WebSocket** — Already partially implemented in JuniperCanopy. Could extend to JuniperData.

---

### RD-016: GPU-Accelerated Data Generation (DATA-019)

**Priority**: LOW | **Status**: DEFERRED | **Effort**: XL (2-4 weeks)
**Source**: INTEGRATION_DEVELOPMENT_PLAN.md, PRE-DEPLOYMENT_ROADMAP-2.md (P3-NEW-003)

**Problem**: Current generation is CPU-only via NumPy. For very large datasets, GPU acceleration could improve throughput.

**Decision**: Not needed for current dataset sizes (hundreds to thousands of points). Revisit when dataset sizes reach millions.

**Revisit Criteria**:

- Dataset sizes exceed 1M points
- Generation time exceeds acceptable thresholds (>30 seconds)
- GPU resources are available in deployment environment

**Design Options**:

1. **Option A: CuPy** — Drop-in NumPy replacement for CUDA. Minimal code changes. Requires CUDA toolkit.
2. **Option B: JAX** — Google's accelerated computing library. More flexible device management. Larger dependency footprint.
3. **Option C: PyTorch (torch.cuda)** — Already a dependency for some generators. Leverages existing ecosystem.

---

### RD-017: Continuous Profiling Integration (DATA-020)

**Priority**: LOW | **Status**: DEFERRED | **Effort**: Large (1-2 weeks)
**Source**: INTEGRATION_DEVELOPMENT_PLAN.md, PRE-DEPLOYMENT_ROADMAP-2.md (P3-NEW-004)

**Problem**: No production-grade performance monitoring for the JuniperData service:

- Response time tracking per endpoint
- Memory usage profiling
- Dataset generation time metrics
- Prometheus/Grafana integration

**Decision**: Deferred until production deployment.

**Design Options**:

1. **Option A (Recommended): Grafana Pyroscope** — Open source, self-hostable, integrates with existing Grafana. 2-5% overhead. Supports Python via `pyroscope-io` SDK.
2. **Option B: Prometheus + custom metrics** — Standard observability stack. Requires defining and exposing custom metrics via `/metrics` endpoint.
3. **Option C: OpenTelemetry** — Vendor-neutral telemetry standard. Supports traces, metrics, and logs. Growing ecosystem.

---

## Validation Results

### Validation Methodology

Each documented change was validated by:

1. Checking file existence in the codebase
2. Verifying feature implementation via symbol/content search
3. Cross-referencing documented state against actual state
4. Identifying discrepancies between documents

### Items Confirmed Complete (No Action Required)

| ID       | Item                               | Validated                                                |
| -------- | ---------------------------------- | -------------------------------------------------------- |
| DATA-001 | Fix mypy type errors in test files | mypy reports 0 errors                                    |
| DATA-002 | Fix flake8 unused imports          | No F401 in production code                               |
| DATA-003 | Fix golden dataset issues          | File updated                                             |
| DATA-004 | B008 warnings handled              | `.flake8` per-file-ignores configured                    |
| DATA-005 | SIM117 in extend-ignore            | `.flake8` configured                                     |
| DATA-006 | Dockerfile                         | `Dockerfile` exists                                      |
| DATA-007 | Health check probes                | `/v1/health/live` and `/v1/health/ready` endpoints exist |
| DATA-008 | E2E integration tests              | `test_e2e_workflow.py` exists                            |
| DATA-009 | API versioning docs                | `docs/api/JUNIPER_DATA_API.md` exists                    |
| DATA-010 | NPZ schema docs                    | Documented in API docs                                   |
| DATA-011 | Parameter validation parity        | `SpiralParams` has AliasChoices                          |
| DATA-012 | Client package                     | `juniper_data_client/` complete with tests               |
| DATA-013 | Client test coverage               | 41 tests, 96% coverage                                   |
| DATA-014 | 8 generators                       | All 8 registered in GENERATOR_REGISTRY                   |
| DATA-015 | 7 storage backends                 | All 7 implementations present                            |
| DATA-016 | Lifecycle management               | `DatasetMeta` has tags, ttl_seconds, expires_at, etc.    |
| DATA-017 | API security                       | `security.py` with APIKeyAuth + RateLimiter              |

### Items with Discrepancies

| Item                     | Documented                                | Actual             | Impact                 |
| ------------------------ | ----------------------------------------- | ------------------ | ---------------------- |
| GENERATOR_REGISTRY count | Release notes: 5 of 8                     | 8 of 8 registered  | Documentation outdated |
| Coverage percentage      | 95.18% (dev plan) vs ~60% (release notes) | Needs verification | Conflicting metrics    |
| Dependabot configuration | "Complete" (audit plan)                   | File missing       | Security gap           |

### Items Validated as Reasonable and Feasible

| ID     | Item                  | Assessment                                      |
| ------ | --------------------- | ----------------------------------------------- |
| RD-001 | Update release notes  | Trivial edit, no risk                           |
| RD-002 | Create dependabot.yml | Standard configuration, no risk                 |
| RD-005 | Reconcile coverage    | Essential for accurate reporting                |
| RD-006 | Security tests        | Important gap, standard testing practices       |
| RD-007 | Coverage improvement  | Likely a configuration issue, not missing tests |
| RD-010 | PyPI publication      | Standard Python packaging, well-documented      |
| RD-012 | flake8→ruff migration | Optional but beneficial for speed               |
| RD-013 | Line length review    | Best practice alignment needed                  |

### Items with Concerns

| ID                | Item                 | Concern                                               |
| ----------------- | -------------------- | ----------------------------------------------------- |
| RD-015 (DATA-018) | IPC Architecture     | XL effort, architecturally complex, no current demand |
| RD-016 (DATA-019) | GPU Acceleration     | XL effort, small dataset sizes don't justify          |
| RD-017 (DATA-020) | Continuous Profiling | Requires infrastructure, pre-production               |

---

## Design Analysis

### Phase 1 Design (Documentation & Housekeeping)

**RD-001/RD-004**: Straightforward text edits. No design decisions required.

**RD-002 (Dependabot)**: Standard GitHub configuration. Recommended approach:

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    groups:
      python-minor:
        patterns: ["*"]
        update-types: ["minor", "patch"]
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
```

**RD-005 (Coverage Reconciliation)**: Run coverage locally and update all documents to consistent value. If coverage is truly at ~60%, investigate `source_pkgs` configuration vs `source` paths. The discrepancy between 95% and 60% likely reflects different measurement scopes.

### Phase 2 Design (Test Coverage & Quality)

**RD-006 (Security Tests)**: Recommended test structure:

```bash
tests/unit/test_security_boundaries.py
├── TestPathTraversalPrevention
│   ├── test_storage_path_with_dotdot
│   ├── test_storage_absolute_path_rejected
│   └── test_dataset_id_special_chars
├── TestInputBoundaryEnforcement
│   ├── test_extreme_n_points_value
│   ├── test_negative_parameters
│   └── test_string_injection_in_params
├── TestResourceExhaustion
│   ├── test_very_large_dataset_request
│   └── test_concurrent_generation_limits
└── TestAPIBoundaries
    ├── test_oversized_request_body
    └── test_malformed_json_handling
```

**RD-007 (Coverage)**: The likely root cause is coverage configuration. Tests exist for all "0% coverage" modules. Investigation steps:

1. Run coverage with `--cov=juniper_data` and check which modules are included
2. Verify `source_pkgs` in `pyproject.toml` includes all subpackages
3. Check if conditional imports (e.g., `try: import arc_agi except ImportError`) bypass coverage

### Phase 3 Design (Client Library)

**RD-010 (PyPI Publication)**: Recommended CI/CD workflow:

```bash
Trigger: Git tag matching "client-v*"
Steps:
  1. Checkout code
  2. Build package (python -m build)
  3. Run client tests
  4. Publish to PyPI (twine upload)
```

Alternatively, integrate into existing `ci.yml` with a conditional publish job.

### Phase 4-5 Design

Detailed in the Design Options sections of each item above.

---

## Cross-Project References

### Items Identified for JuniperCascor

See `JUNIPER-CASCOR_POST-RELEASE_DEVELOPMENT-ROADMAP.md` in JuniperCascor notes directory.

| ID          | Item                                      | Status                        | Source |
| ----------- | ----------------------------------------- | ----------------------------- | ------ |
| CAS-001     | Extract Spiral Generator to JuniperData   | COMPLETE (JuniperData exists) |        |
| CAS-002     | Separate Epoch Limits                     | NOT STARTED                   |        |
| CAS-003     | Max Train Session Iterations              | NOT STARTED                   |        |
| CAS-004     | Extract Remote Worker to JuniperBranch    | NOT STARTED                   |        |
| CAS-005     | Extract Common Dependencies to Modules    | NOT STARTED                   |        |
| CAS-006     | Auto-Snap Best Network (Accuracy Ratchet) | NOT STARTED                   |        |
| CAS-007     | Optimize Slow Tests                       | NOT STARTED                   |        |
| CAS-008     | Network Hierarchy Management              | NOT STARTED                   |        |
| CAS-009     | Network Population Management             | NOT STARTED                   |        |
| CAS-010     | Snapshot Vector DB Storage                | NOT STARTED                   |        |
| CAS-REF-001 | Code Coverage Below 90%                   | IN PROGRESS                   |        |
| CAS-REF-002 | CI/CD Coverage Gates                      | NOT STARTED                   |        |
| CAS-REF-003 | Type Errors Gradual Fix                   | IN PROGRESS                   |        |
| CAS-REF-004 | Legacy Spiral Code Removal                | NOT STARTED                   |        |
| CAS-REF-005 | RemoteWorkerClient Integration            | COMPLETE (per P1-NEW-002)     |        |

### Items Identified for JuniperCanopy

See `JUNIPER-CANOPY_POST-RELEASE_DEVELOPMENT-ROADMAP.md` in JuniperCanopy notes directory.

CAN-000 through CAN-021 (22 enhancement items) documented in PRE-DEPLOYMENT_ROADMAP-2.md Section 7.1.

---

## Implementation Priority Matrix

### Immediate (This Sprint)

| ID     | Item                              | Priority | Effort | Impact                 |
| ------ | --------------------------------- | -------- | ------ | ---------------------- |
| RD-001 | Update release notes known issues | HIGH     | S      | Documentation accuracy |
| RD-002 | Create dependabot.yml             | HIGH     | S      | Security automation    |
| RD-004 | Update v0.5.0 planned items       | MEDIUM   | S      | Roadmap accuracy       |
| RD-005 | Reconcile coverage metrics        | HIGH     | S      | Informed decisions     |

### Short-Term (Next 2 Sprints)

| ID     | Item                   | Priority | Effort | Impact                |
| ------ | ---------------------- | -------- | ------ | --------------------- |
| RD-003 | Verify CodeQL scans    | MEDIUM   | S      | CI/CD completeness    |
| RD-006 | Security-focused tests | HIGH     | M      | Security posture      |
| RD-007 | Coverage improvement   | MEDIUM   | L      | Quality confidence    |
| RD-010 | Publish client to PyPI | MEDIUM   | M      | Ecosystem integration |

### Medium-Term (Next Quarter)

| ID     | Item                     | Priority | Effort | Impact               |
| ------ | ------------------------ | -------- | ------ | -------------------- |
| RD-011 | Update consumer projects | MEDIUM   | S      | Code deduplication   |
| RD-008 | Fix SIM117 violations    | LOW      | S-M    | Code readability     |
| RD-009 | Performance test infra   | LOW      | M      | Regression detection |
| RD-012 | flake8→ruff migration    | LOW      | M      | Dev velocity         |
| RD-013 | Line length review       | LOW      | S      | Code standards       |

### Deferred (Future)

| ID     | Item                  | Priority | Effort | Trigger                |
| ------ | --------------------- | -------- | ------ | ---------------------- |
| RD-014 | Documentation CI step | LOW      | S      | Documentation growth   |
| RD-015 | IPC Architecture      | LOW      | XL     | Performance bottleneck |
| RD-016 | GPU Acceleration      | LOW      | XL     | Dataset size >1M       |
| RD-017 | Continuous Profiling  | LOW      | L      | Production deployment  |

---

## Summary Statistics

| Category                 | Count |
| ------------------------ | ----- |
| Total Items              | 17    |
| NOT STARTED              | 12    |
| DEFERRED                 | 4     |
| PENDING VERIFICATION     | 1     |
| Phase 1 (Housekeeping)   | 5     |
| Phase 2 (Quality)        | 4     |
| Phase 3 (Client Library) | 2     |
| Phase 4 (Tooling)        | 3     |
| Phase 5 (Advanced)       | 3     |

---

## Cross-References from JuniperCanopy Audit (2026-02-17)

Items identified during the JuniperCanopy comprehensive notes/ audit that have JuniperData dependencies.

### RD-CANOPY-001: Health Check Endpoint Relied Upon by Canopy

**Status**: INFORMATIONAL
**Source**: JuniperCanopy CAN-HIGH-001 (Startup Health Check)

**Description**: JuniperCanopy plans to add a startup health check that probes `{JUNIPER_DATA_URL}/health` during application lifespan startup. The existing `/v1/health/live` and `/v1/health/ready` endpoints (validated as complete in DATA-007) serve this purpose. No action required unless health check endpoint paths change.

**Impact**: JuniperCanopy will depend on health check endpoint availability.

---

### RD-CANOPY-002: Dataset Versioning API (Future)

**Status**: DEFERRED
**Source**: JuniperCanopy CAN-DEF-005 (JuniperData Dataset Versioning)

**Description**: JuniperCanopy plans to support dataset versioning — tracking which dataset version was used for each training session. This requires JuniperData to support versioned datasets (multiple versions of the same dataset with version metadata). Currently deferred in both projects.

---

### RD-CANOPY-003: Batch Operations API (Future)

**Status**: DEFERRED
**Source**: JuniperCanopy CAN-DEF-006 (JuniperData Batch Operations)

**Description**: JuniperCanopy plans to support batch dataset operations (bulk import, bulk export, batch metadata updates). This requires corresponding batch API endpoints in JuniperData. Currently deferred in both projects.

---

## Document History

| Date       | Author                 | Changes                                                       |
| ---------- | ---------------------- | ------------------------------------------------------------- |
| 2026-02-17 | Paul Calnon / AI Agent | Initial creation from comprehensive codebase audit            |
| 2026-02-17 | AI Agent               | Added cross-references from JuniperCanopy comprehensive audit |
