# JuniperData Post-Release Development Roadmap

**Project**: JuniperData - Dataset Generation Microservice
**Version**: 0.4.2 (Current Release)
**Created**: 2026-02-17
**Author**: Paul Calnon
**Status**: Active - Post-Migration Reassessment
**Audit Date**: 2026-02-17
**Migration Review Date**: 2026-02-24

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

### Polyrepo Migration Impact (2026-02-24)

On 2026-02-24, the JuniperData codebase was extracted from the monorepo (`pcalnon/Juniper`) into a standalone repository (`pcalnon/juniper-data`) as part of the Juniper polyrepo migration (Phases 0–5 of the POLYREPO_MIGRATION_PLAN). This migration has materially affected the status and relevance of several roadmap items:

- **RD-002 (Dependabot)**: Now **COMPLETE** — `dependabot.yml` was created during migration; 3 dependabot PRs are already open.
- **RD-003 (CodeQL)**: Now **COMPLETE** — CodeQL scans confirmed active on the standalone `juniper-data` repo.
- **RD-010 (Publish client to PyPI)**: Now **COMPLETE** — `juniper-data-client` v0.3.0 published to PyPI during Phase 1, with Trusted Publishing OIDC configured.
- **RD-011 (Update consumers)**: Now **COMPLETE** — all vendored copies removed from CasCor, Canopy, and Data; all consumers reference the PyPI package.
- **RD-004 (v0.5.0 plan)**: **COMPLETE** — v0.5.0 redefined as Quality + Tooling release (RD-006 security tests, RD-012/013 ruff migration + line length).
- **CI/CD**: Standalone repo CI now uses pip-based installation (not conda), with its own independent pipeline.
- **Cross-repo symlinks**: Several symlinks in `notes/` are now broken due to the repo separation (see RD-018).
- **Cross-project references**: Many source documents referenced in this roadmap reside in the old monorepo structure and are accessible only through local history symlinks.

### Current State (Validated 2026-02-24)

| Metric                           | Documented (2026-02-17)   | Validated (2026-02-24)                     | Status                         |
| -------------------------------- | ------------------------- | ------------------------------------------ | ------------------------------ |
| Version                          | 0.4.2                     | 0.4.2                                      | Correct                        |
| Generators in GENERATOR_REGISTRY | 5 (per release notes)     | **8**                                      | ~~Release notes outdated~~ **FIXED** (RD-001) |
| Storage Backends                 | 7                         | 7 (+ base + **init**)                      | Correct                        |
| Service Tests                    | 658                       | 658 (30 files)                             | Correct                        |
| Client Tests                     | 41                        | N/A — client is now a separate repo        | **Moved to juniper-data-client** |
| Client Coverage                  | 96%                       | N/A — client is now a separate repo        | **Moved to juniper-data-client** |
| Security (auth + rate limiting)  | Complete                  | Complete                                   | Correct                        |
| Lifecycle Management             | Complete                  | Complete                                   | Correct                        |
| Dockerfile                       | Complete                  | Present                                    | Correct                        |
| `.github/dependabot.yml`         | **Missing** (per 02-17)  | **Present** (created during migration)     | **RESOLVED**                   |
| `.github/workflows/codeql.yml`   | Complete                  | Present and active on standalone repo      | Correct                        |
| docs/api/JUNIPER_DATA_API.md     | Complete                  | Present                                    | Correct                        |
| Repository                       | Monorepo branch           | Standalone `pcalnon/juniper-data` (595 commits) | **Migrated**              |
| CI/CD                            | Monorepo-scoped           | Standalone pipeline (pip-based, CI green)  | **Migrated**                   |
| `juniper-data-client`            | Vendored in repo          | **Removed** — PyPI package v0.3.0          | **RESOLVED**                   |

### Discrepancy Summary (Updated 2026-02-24)

| Item                     | Documented State (2026-02-17)    | Actual State (2026-02-24)                         | Action Required              |
| ------------------------ | -------------------------------- | ------------------------------------------------- | ---------------------------- |
| GENERATOR_REGISTRY count | Release notes say 5 of 8         | All 8 registered                                  | ~~Update release notes~~ **DONE** (RD-001) |
| Coverage reporting       | INTEGRATION_DEV_PLAN says 95.18% | Release notes say ~60%                            | ~~Verify current coverage~~ **DONE** — 99.40% (RD-001) |
| Dependabot configuration | File does not exist              | **File exists, 3 PRs open**                       | ~~Create dependabot.yml~~ **DONE** |
| v0.5.0 planned items     | "Register remaining generators"  | All 4 items already done                          | Revise v0.5.0 scope          |
| Client package           | Local vendored copy              | **Published to PyPI as v0.3.0**                   | ~~Publish to PyPI~~ **DONE** |
| Broken notes symlinks    | N/A                              | 4 broken symlinks in `notes/` and `notes/history/` | ~~Fix or remove~~ **DONE** (RD-018) |

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

**Priority**: HIGH | **Status**: COMPLETE (2026-02-24) | **Effort**: Small (30 min)
**Source**: Codebase validation audit (2026-02-17)

**Problem**: The v0.4.2 release notes contained two outdated known issues and an outdated "What's Next" section.

**Resolution** (2026-02-24):

- [x] Updated `notes/releases/RELEASE_NOTES_v0.4.2.md` known issues section
- [x] Moved GENERATOR_REGISTRY known issue to "Resolved Since Release" — all 8 generators confirmed registered
- [x] Verified current coverage: **99.40%** (659 tests passing). Moved coverage known issue to "Resolved Since Release"
- [x] Updated "What's Next" section: moved completed items (PyPI publish, consumer updates, generator registration, coverage) to "Completed Since Release"; updated roadmap link to development roadmap
- [x] B008 flake8 warnings remain as the only active known issue (intentional, not a bug)

---

### RD-002: Create Dependabot Configuration

**Priority**: ~~HIGH~~ N/A | **Status**: **COMPLETE** (resolved during polyrepo migration) | **Effort**: N/A
**Source**: TEST_SUITE_CICD_ENHANCEMENT_DEVELOPMENT_PLAN.md (SEC-003)

**Resolution**: The `dependabot.yml` file was created as part of the polyrepo migration and is now present at `.github/dependabot.yml` in the standalone `juniper-data` repository. The configuration includes:

- `pip` ecosystem: weekly schedule (Monday 09:00 ET), grouped minor/patch updates, 5 PR limit
- `github-actions` ecosystem: weekly schedule (Monday), 3 PR limit
- Labels, commit message prefixes, and timezone all configured

**Validation (2026-02-24)**: File confirmed present (2,740 bytes). Three dependabot PRs are already open on the repository (`actions/cache`, `actions/setup-python`, `codecov/codecov-action`).

**No further action required.**

---

### RD-003: Verify and Document CodeQL Scan Status

**Priority**: ~~MEDIUM~~ N/A | **Status**: **COMPLETE** (confirmed active on standalone repo) | **Effort**: N/A
**Source**: TEST_SUITE_CICD_ENHANCEMENT_DEVELOPMENT_PLAN.md (P3-T8)

**Resolution**: CodeQL scans are confirmed active and running successfully on the standalone `juniper-data` repository. The Phase 5 migration verification (2026-02-22) explicitly confirmed "CodeQL + scheduled CI active" for the juniper-data repo.

**Validation (2026-02-24)**: `.github/workflows/codeql.yml` present and active on `pcalnon/juniper-data`.

**No further action required.**

---

### RD-004: Update v0.5.0 Planned Items

**Priority**: HIGH | **Status**: COMPLETE (2026-02-24) | **Effort**: Small (30 min)
**Source**: RELEASE_NOTES_v0.4.2.md (What's Next section)

**Problem**: The original v0.5.0 plan included four items that were all completed during the polyrepo migration. The version needed a new scope definition.

**Resolution** (2026-02-24):

- [x] Redefined v0.5.0 as the first release from the standalone `juniper-data` repo, scoped as **Quality + Tooling**:
  - RD-006: Security boundary test suite (HIGH priority)
  - RD-012: flake8→ruff migration (LOW priority, good housekeeping)
  - RD-013: Line length normalization to 120 (LOW priority, consistency)
- [x] Updated `notes/releases/RELEASE_NOTES_v0.4.2.md` "What's Next" section with new v0.5.0 scope (completed items already moved to "Completed Since Release" in RD-001)
- [x] v0.5.0 will be a MINOR release focused on quality infrastructure and linting modernization — no new features or API changes

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

**Post-Migration Note**: Coverage should now be measured exclusively within the standalone `juniper-data` repo. The `source_pkgs = ["juniper_data"]` configuration in `pyproject.toml` is correct for the standalone layout. The CI pipeline already enforces an 80% coverage gate (`COVERAGE_FAIL_UNDER: "80"` in `ci.yml`). The removal of the vendored `juniper_data_client` from this repo (commit `4bada2a`) means client code is no longer in scope — coverage metrics should now be purely for the service code.

---

### RD-018: Fix Broken Notes Symlinks (NEW — Post-Migration)

**Priority**: MEDIUM | **Status**: COMPLETE (2026-02-24) | **Effort**: Small (30 min)
**Source**: Polyrepo migration impact analysis (2026-02-24)

**Problem**: The polyrepo migration left 4 broken symlinks in the `notes/` directory that point to the old monorepo location (`JuniperCascor/juniper_cascor/notes/`). Additionally, `notes/MONOREPO_ANALYSIS.md` was a symlink that resolved but pointed to the old `JuniperCascor` monorepo location, which would break when that directory is removed.

**Resolution** (2026-02-24):

- [x] `notes/POLYREPO_MIGRATION_PLAN.md` — replaced broken symlink with redirect note pointing to `pcalnon/juniper-cascor`
- [x] `notes/MONOREPO_ANALYSIS.md` — replaced fragile symlink with redirect note pointing to `pcalnon/juniper-cascor`
- [x] `notes/history/PRE-DEPLOYMENT_ROADMAP.md` — removed (historical, archived in monorepo git history)
- [x] `notes/history/PRE-DEPLOYMENT_ROADMAP-2.md` — removed (historical, archived in monorepo git history)
- [x] `notes/history/INTEGRATION_ROADMAP.md` — removed (historical, archived in monorepo git history)
- [x] Updated `CLAUDE.md` Key Documentation table to reflect new state

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

**Post-Migration Note**: No migration impact. The test file paths and infrastructure remain unchanged in the standalone repo. The CI pipeline will automatically pick up new test files.

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

**Post-Migration Note**: The coverage configuration (`source_pkgs = ["juniper_data"]` in `pyproject.toml`) is correct for the standalone repo layout. The CI pipeline now uses `pip install ".[all]"` which may or may not install optional dependencies like `arc-agi`. Verify whether the CI environment installs these optional packages — if not, the 0% coverage modules may still be excluded from CI coverage reports even with correct `source_pkgs`. Consider adding `arc-agi` to the `[test]` extra or using conditional `importorskip` patterns.

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

**Post-Migration Note**: No migration impact. If RD-012 (flake8→ruff) is pursued, SIM117 handling can be addressed during that migration rather than as a separate effort.

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

**Post-Migration Note**: Performance benchmarks are now especially relevant in the polyrepo architecture. JuniperCascor and JuniperCanopy communicate with JuniperData over REST, so endpoint response time benchmarks would directly inform the viability of the service-oriented approach (cf. RD-015 IPC Architecture revisit criteria).

---

## Phase 3: Client Library Publication

**Priority**: ~~MEDIUM~~ N/A | **Risk**: N/A | **Effort**: N/A
**Rationale**: ~~Consolidates the shared client package and removes duplicated code in consumer projects.~~ **PHASE COMPLETE** — All items resolved during polyrepo migration Phase 1.

### RD-010: Publish juniper-data-client to PyPI

**Priority**: N/A | **Status**: **COMPLETE** (resolved during polyrepo migration Phase 1) | **Effort**: N/A
**Source**: RELEASE_NOTES_v0.4.2.md (What's Next), INTEGRATION_DEVELOPMENT_PLAN.md (DATA-012 Next Steps)

**Resolution**: `juniper-data-client` v0.3.0 was published to PyPI on 2026-02-20 as part of the polyrepo migration Phase 1. The package is available at `pypi.org/project/juniper-data-client/`.

**Completion details**:

- Standalone repository: `pcalnon/juniper-data-client` (6 commits on `main`, CI green)
- PyPI: `juniper-data-client` v0.3.0 published via Trusted Publishing (OIDC, no API tokens)
- CI/CD: `ci.yml` + `publish.yml` (two-stage: TestPyPI → PyPI)
- Tests: 41 tests pass, 96% coverage
- Consumers updated: JuniperCascor, JuniperCanopy, and JuniperData all reference `juniper-data-client>=0.3.0`
- Vendored copy removed from JuniperData (commit `4bada2a`)

**No further action required.**

---

### RD-011: Update Consumer Projects to Use Published Client

**Priority**: N/A | **Status**: **COMPLETE** (resolved during polyrepo migration Phases 1 and 5) | **Effort**: N/A
**Source**: INTEGRATION_DEVELOPMENT_PLAN.md (DATA-012-A)

**Resolution**: All vendored copies of `juniper_data_client` have been removed from all consumer projects. All consumers now reference the PyPI package:

| Consumer        | Reference Location                                   | Status                    |
| --------------- | ---------------------------------------------------- | ------------------------- |
| JuniperCascor   | `pyproject.toml [project.optional-dependencies].juniper-data` | `juniper-data-client>=0.3.0` |
| JuniperCanopy   | `pyproject.toml [project.optional-dependencies].juniper-data` | `juniper-data-client>=0.3.0` |
| JuniperData     | `pyproject.toml [project.optional-dependencies].test`         | `juniper-data-client>=0.3.0` |

All tests pass with the PyPI-installed package (JuniperData 659, JuniperCascor 226 — verified 2026-02-21).

**No further action required.**

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
- [ ] Remove flake8 configuration files (`.flake8`)

**Design Options**:

1. **Option A (Recommended)**: Gradual migration — run ruff alongside flake8 initially
2. **Option B**: Full cutover — replace all flake8/isort with ruff in one change
3. **Option C**: Stay with flake8 — current setup works, migration is optional

**Feasibility**: High. ruff supports all configured flake8 rules. Migration is typically straightforward.

**Post-Migration Note**: The standalone repo's CI pipeline is simpler and fully independent, making this a good time to pursue a tooling migration. The `juniper-cascor` repo already went through CI normalization during its Phase 5 extraction (pip-based install, shellcheck fixes, Bandit skip codes). A ruff migration on `juniper-data` would establish a pattern that could be adopted across other Juniper repos. Note that `juniper-cascor-client` and `juniper-cascor-worker` already use 120-char line length — alignment across repos would benefit from doing RD-012 and RD-013 together.

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

**Post-Migration Note**: The newer Juniper packages (`juniper-cascor-client`, `juniper-cascor-worker`) already use 120-char line length. Aligning `juniper-data` to 120 would establish consistency across the polyrepo ecosystem. The parent `CLAUDE.md` documents "512 for linters, 120 for flake8" — this should be reconciled across all repos as part of a cross-project style normalization effort. Consider tackling this alongside RD-012 (ruff migration).

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

**Post-Migration Note**: Documentation link validation is now more important in the polyrepo context. Links between projects that previously used relative paths within the monorepo must now use absolute GitHub URLs or be removed. The broken symlinks identified in RD-018 are a specific manifestation of this issue.

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

**Post-Migration Note**: The polyrepo migration has formalized inter-service communication via REST through published client packages (`juniper-data-client` v0.3.0, `juniper-cascor-client` v0.1.0). The CasCor service API (19 REST + 2 WebSocket endpoints) and the `CascorServiceAdapter` in Canopy demonstrate that the REST+WebSocket pattern works well for the current use cases. This further reduces urgency for alternative IPC mechanisms. However, the migration also means all communication is now network-based (no more in-process Python imports), which makes performance benchmarking (RD-009) a more valuable input to the IPC revisit decision.

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

**Post-Migration Note**: No migration impact. The standalone CI now uses pip-based installation and does not have CUDA available — any GPU work would require a separate CI job or local testing only. Note that PyTorch is no longer a dependency of `juniper-data` (it's only in `juniper-cascor` and `juniper-cascor-worker`), so Option C would introduce a new heavy dependency.

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

**Post-Migration Note**: The Phase 6 (Post-Migration Hardening) plan in the polyrepo migration document includes Docker Compose full-stack deployment with health checks. Continuous profiling would naturally fit into that deployment infrastructure. The migration plan also calls for a version compatibility matrix and integration test suite (Phase 6, Steps 6.1–6.2) — profiling could be layered on top of that.

---

## Validation Results

### Validation Methodology

Each documented change was validated by:

1. Checking file existence in the codebase
2. Verifying feature implementation via symbol/content search
3. Cross-referencing documented state against actual state
4. Identifying discrepancies between documents
5. **(NEW)** Cross-referencing against polyrepo migration outcomes (2026-02-24)

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
| DATA-012 | Client package                     | **Published to PyPI** as `juniper-data-client` v0.3.0    |
| DATA-013 | Client test coverage               | 41 tests, 96% coverage (in `juniper-data-client` repo)  |
| DATA-014 | 8 generators                       | All 8 registered in GENERATOR_REGISTRY                   |
| DATA-015 | 7 storage backends                 | All 7 implementations present                            |
| DATA-016 | Lifecycle management               | `DatasetMeta` has tags, ttl_seconds, expires_at, etc.    |
| DATA-017 | API security                       | `security.py` with APIKeyAuth + RateLimiter              |

### Items Resolved by Migration (Previously Had Discrepancies)

| Item                     | Previous State (2026-02-17)       | Post-Migration State (2026-02-24)                 | Resolution                                |
| ------------------------ | --------------------------------- | ------------------------------------------------- | ----------------------------------------- |
| Dependabot configuration | "Complete" in audit, file missing | **File present**, 3 PRs already open              | Created during polyrepo Phase 5 CI setup  |
| CodeQL scans             | PENDING VERIFICATION              | **Active and running** on standalone repo          | Confirmed during Phase 5 verification     |
| Client PyPI publication  | NOT STARTED                       | **Published** `juniper-data-client` v0.3.0        | Completed in polyrepo Phase 1             |
| Consumer projects update | NOT STARTED                       | **All consumers updated**, vendored copies removed | Completed in polyrepo Phases 1 and 5      |

### Items with Remaining Discrepancies

| Item                     | Documented                                | Actual             | Impact                 |
| ------------------------ | ----------------------------------------- | ------------------ | ---------------------- |
| GENERATOR_REGISTRY count | Release notes: 5 of 8                     | 8 of 8 registered  | ~~Documentation outdated~~ **FIXED** (RD-001) |
| Coverage percentage      | 95.18% (dev plan) vs ~60% (release notes) | **99.40%**         | ~~Conflicting metrics~~ **RESOLVED** (RD-001/RD-005) |
| Broken notes symlinks    | N/A (new issue)                           | **RESOLVED**       | ~~Reference integrity~~ **DONE** |

### Items Validated as Reasonable and Feasible

| ID     | Item                  | Assessment                                                  |
| ------ | --------------------- | ----------------------------------------------------------- |
| RD-001 | Update release notes  | **COMPLETE** — Known issues + What's Next updated           |
| RD-005 | Reconcile coverage    | Essential for accurate reporting                            |
| RD-006 | Security tests        | Important gap, standard testing practices                   |
| RD-007 | Coverage improvement  | Likely a configuration issue, not missing tests             |
| RD-012 | flake8→ruff migration | Optional but beneficial; good timing in standalone repo     |
| RD-013 | Line length review    | Best practice alignment needed; coordinate across repos     |
| RD-018 | Fix broken symlinks   | **COMPLETE** — Redirect notes + removed history symlinks    |

### Items with Concerns

| ID                | Item                 | Concern                                               |
| ----------------- | -------------------- | ----------------------------------------------------- |
| RD-015 (DATA-018) | IPC Architecture     | XL effort, architecturally complex, no current demand — further reduced by successful REST migration |
| RD-016 (DATA-019) | GPU Acceleration     | XL effort, small dataset sizes don't justify — PyTorch no longer a dependency in juniper-data |
| RD-017 (DATA-020) | Continuous Profiling | Requires infrastructure, pre-production — fits Phase 6 hardening |

---

## Design Analysis

### Phase 1 Design (Documentation & Housekeeping)

**RD-001/RD-004**: Both **COMPLETE**. RD-001 updated release notes known issues and What's Next. RD-004 redefined v0.5.0 as Quality + Tooling: RD-006 (security boundary tests) + RD-012/013 (flake8→ruff + line length normalization).

**RD-002 (Dependabot)**: **COMPLETE** — No design needed. File exists and is active.

**RD-005 (Coverage Reconciliation)**: Run coverage locally and update all documents to consistent value. If coverage is truly at ~60%, investigate `source_pkgs` configuration vs `source` paths. The discrepancy between 95% and 60% likely reflects different measurement scopes. Post-migration, the client code is no longer in this repo's scope, which should simplify the measurement.

**RD-018 (Broken Symlinks)**: Recommend a tiered approach:
- **POLYREPO_MIGRATION_PLAN.md**: Replace with a redirect note (canonical copy is in `juniper-cascor/notes/`)
- **History files**: Remove broken symlinks — these are archived documents accessible through monorepo git history
- **MONOREPO_ANALYSIS.md**: Copy if JuniperData-relevant, otherwise replace with redirect

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
4. **NEW**: Verify whether `pip install ".[all]"` in CI actually installs optional deps like `arc-agi`

### Phase 3 Design (Client Library)

**PHASE COMPLETE** — No design needed. See RD-010 and RD-011 completion summaries.

### Phase 4-5 Design

Detailed in the Design Options sections of each item above, with post-migration notes.

---

## Cross-Project References

### Items Identified for JuniperCascor

See `JUNIPER-CASCOR_POST-RELEASE_DEVELOPMENT-ROADMAP.md` in JuniperCascor notes directory.

**Post-Migration Note**: JuniperCascor is now a standalone repo at `pcalnon/juniper-cascor` (127 commits, CI green). The following status updates reflect migration outcomes:

| ID          | Item                                      | Status (2026-02-17)           | Status (2026-02-24)                                              |
| ----------- | ----------------------------------------- | ----------------------------- | ---------------------------------------------------------------- |
| CAS-001     | Extract Spiral Generator to JuniperData   | COMPLETE (JuniperData exists) | COMPLETE                                                         |
| CAS-002     | Separate Epoch Limits                     | NOT STARTED                   | NOT STARTED                                                      |
| CAS-003     | Max Train Session Iterations              | NOT STARTED                   | NOT STARTED                                                      |
| CAS-004     | Extract Remote Worker to JuniperBranch    | NOT STARTED                   | **COMPLETE** — `juniper-cascor-worker` published to PyPI v0.1.0  |
| CAS-005     | Extract Common Dependencies to Modules    | NOT STARTED                   | NOT STARTED                                                      |
| CAS-006     | Auto-Snap Best Network (Accuracy Ratchet) | NOT STARTED                   | NOT STARTED                                                      |
| CAS-007     | Optimize Slow Tests                       | NOT STARTED                   | NOT STARTED                                                      |
| CAS-008     | Network Hierarchy Management              | NOT STARTED                   | NOT STARTED                                                      |
| CAS-009     | Network Population Management             | NOT STARTED                   | NOT STARTED                                                      |
| CAS-010     | Snapshot Vector DB Storage                | NOT STARTED                   | NOT STARTED                                                      |
| CAS-REF-001 | Code Coverage Below 90%                   | IN PROGRESS                   | IN PROGRESS                                                      |
| CAS-REF-002 | CI/CD Coverage Gates                      | NOT STARTED                   | **PARTIAL** — CI pipeline exists with coverage gate in standalone repo |
| CAS-REF-003 | Type Errors Gradual Fix                   | IN PROGRESS                   | IN PROGRESS                                                      |
| CAS-REF-004 | Legacy Spiral Code Removal                | NOT STARTED                   | NOT STARTED                                                      |
| CAS-REF-005 | RemoteWorkerClient Integration            | COMPLETE (per P1-NEW-002)     | **SUPERSEDED** — standalone `juniper-cascor-worker` package      |

### Items Identified for JuniperCanopy

See `JUNIPER-CANOPY_POST-RELEASE_DEVELOPMENT-ROADMAP.md` in JuniperCanopy notes directory.

CAN-000 through CAN-021 (22 enhancement items) documented in PRE-DEPLOYMENT_ROADMAP-2.md Section 7.1.

**Post-Migration Note**: JuniperCanopy has not yet been extracted to its standalone repo (blocked by Phase 4 completion — adapter + 3-mode activation are implemented but legacy removal and integration testing remain). The `CascorServiceAdapter` (306 lines) and three-mode activation logic are committed and tested (3,460 tests pass), but the Canopy repo still lives in the monorepo at `pcalnon/Juniper` on the `canopy/migration` branch.

---

## Implementation Priority Matrix

### Immediate (This Sprint)

| ID     | Item                              | Priority | Effort | Impact                 |
| ------ | --------------------------------- | -------- | ------ | ---------------------- |
| RD-001 | Update release notes known issues | ~~HIGH~~ | ~~S~~ | **COMPLETE**           |
| RD-004 | Update v0.5.0 planned items       | ~~HIGH~~ | ~~S~~ | **COMPLETE**           |
| RD-005 | Reconcile coverage metrics        | HIGH     | S      | Informed decisions     |
| RD-018 | Fix broken notes symlinks         | ~~MEDIUM~~ | ~~S~~ | **COMPLETE**           |

### Short-Term (Next 2 Sprints)

| ID     | Item                   | Priority | Effort | Impact                |
| ------ | ---------------------- | -------- | ------ | --------------------- |
| RD-006 | Security-focused tests | HIGH     | M      | Security posture      |
| RD-007 | Coverage improvement   | MEDIUM   | L      | Quality confidence    |

### Medium-Term (Next Quarter)

| ID     | Item                     | Priority | Effort | Impact               |
| ------ | ------------------------ | -------- | ------ | -------------------- |
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

### Completed (Resolved by Polyrepo Migration)

| ID     | Item                     | Resolution Date | Resolution                                      |
| ------ | ------------------------ | --------------- | ----------------------------------------------- |
| RD-002 | Create dependabot.yml    | 2026-02-21      | Created during Phase 5 repo extraction          |
| RD-003 | Verify CodeQL scans      | 2026-02-22      | Confirmed active on standalone repo             |
| RD-010 | Publish client to PyPI   | 2026-02-20      | `juniper-data-client` v0.3.0 on PyPI            |
| RD-011 | Update consumer projects | 2026-02-21      | All consumers reference PyPI package            |
| RD-018 | Fix broken notes symlinks| 2026-02-24      | Redirect notes for migration plan + monorepo analysis; removed 3 history symlinks |
| RD-001 | Update release notes     | 2026-02-24      | Updated known issues (GENERATOR_REGISTRY + coverage resolved), What's Next section |
| RD-004 | Redefine v0.5.0 scope    | 2026-02-24      | v0.5.0 = Quality + Tooling: RD-006 security tests + RD-012/013 ruff + line length |

---

## Summary Statistics

| Category                             | Count |
| ------------------------------------ | ----- |
| Total Items                          | 18    |
| **COMPLETE**                         | 7     |
| NOT STARTED                          | 5     |
| DEFERRED                             | 4     |
| PENDING VERIFICATION                 | 0     |
| NEW (post-migration)                 | 0     |
| Phase 1 (Housekeeping)              | 6     |
| Phase 2 (Quality)                    | 4     |
| Phase 3 (Client Library)             | 2 (both COMPLETE) |
| Phase 4 (Tooling)                    | 3     |
| Phase 5 (Advanced)                   | 3     |

---

## Cross-References from JuniperCanopy Audit (2026-02-17)

Items identified during the JuniperCanopy comprehensive notes/ audit that have JuniperData dependencies.

### RD-CANOPY-001: Health Check Endpoint Relied Upon by Canopy

**Status**: INFORMATIONAL
**Source**: JuniperCanopy CAN-HIGH-001 (Startup Health Check)

**Description**: JuniperCanopy plans to add a startup health check that probes `{JUNIPER_DATA_URL}/health` during application lifespan startup. The existing `/v1/health/live` and `/v1/health/ready` endpoints (validated as complete in DATA-007) serve this purpose. No action required unless health check endpoint paths change.

**Impact**: JuniperCanopy will depend on health check endpoint availability.

**Post-Migration Note**: The polyrepo migration does not change the JuniperData health check endpoints. JuniperCanopy will access them over the network via the same REST URLs. The Phase 6 Docker Compose plan (POLYREPO_MIGRATION_PLAN, Step 6.3) already includes health check configuration: `test: ["CMD", "curl", "-f", "http://localhost:8100/v1/health"]`.

---

### RD-CANOPY-002: Dataset Versioning API (Future)

**Status**: DEFERRED
**Source**: JuniperCanopy CAN-DEF-005 (JuniperData Dataset Versioning)

**Description**: JuniperCanopy plans to support dataset versioning — tracking which dataset version was used for each training session. This requires JuniperData to support versioned datasets (multiple versions of the same dataset with version metadata). Currently deferred in both projects.

**Post-Migration Note**: In the polyrepo architecture, any dataset versioning API changes to JuniperData would also require a corresponding update to `juniper-data-client`. The versioning coordination described in Phase 6 (Step 6.1) should include a compatibility matrix entry for this feature when it is implemented.

---

### RD-CANOPY-003: Batch Operations API (Future)

**Status**: DEFERRED
**Source**: JuniperCanopy CAN-DEF-006 (JuniperData Batch Operations)

**Description**: JuniperCanopy plans to support batch dataset operations (bulk import, bulk export, batch metadata updates). This requires corresponding batch API endpoints in JuniperData. Currently deferred in both projects.

**Post-Migration Note**: Same versioning considerations as RD-CANOPY-002. Any new API endpoints in JuniperData require corresponding client methods in `juniper-data-client`, which must be published to PyPI before consumers can use them. This is a 3-step release process: (1) add endpoint to `juniper-data`, (2) add client method to `juniper-data-client` and publish, (3) update consumers.

---

## Document History

| Date       | Author                 | Changes                                                       |
| ---------- | ---------------------- | ------------------------------------------------------------- |
| 2026-02-17 | Paul Calnon / AI Agent | Initial creation from comprehensive codebase audit            |
| 2026-02-17 | AI Agent               | Added cross-references from JuniperCanopy comprehensive audit |
| 2026-02-24 | AI Agent               | Post-migration reassessment: updated status of RD-002, RD-003, RD-010, RD-011 to COMPLETE; added RD-018 (broken symlinks); added post-migration notes to all items; updated CAS cross-references (CAS-004 COMPLETE, CAS-REF-005 SUPERSEDED); updated priority matrix and summary statistics; archived pre-migration version to `history/` |
| 2026-02-24 | AI Agent               | RD-018 COMPLETE: removed 3 broken history symlinks, replaced `POLYREPO_MIGRATION_PLAN.md` and `MONOREPO_ANALYSIS.md` symlinks with redirect notes, updated CLAUDE.md Key Documentation table, updated summary statistics |
| 2026-02-24 | AI Agent               | RD-001 COMPLETE: updated v0.4.2 release notes — moved GENERATOR_REGISTRY and coverage known issues to "Resolved Since Release" (coverage verified at 99.40%, 659 tests); updated What's Next with completed items and new roadmap link |
| 2026-02-24 | AI Agent               | RD-004 COMPLETE: redefined v0.5.0 scope as Quality + Tooling (RD-006 security tests + RD-012/013 flake8→ruff + line length normalization); updated release notes What's Next with concrete v0.5.0 plan |
