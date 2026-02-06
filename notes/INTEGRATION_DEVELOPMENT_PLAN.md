# JuniperData Integration Development Plan

**Project**: Juniper Data - Dataset Generation Service
**Version**: 0.3.0
**Author**: Paul Calnon
**Created**: 2026-02-05
**Status**: Planning Document - No Code Changes

---

## Overview

This document compiles all outstanding work items for the JuniperData project, synthesized from:

1. **JUNIPER_CASCOR_SPIRAL_DATA_GEN_REFACTOR_PLAN.md** - Phases 0-4 complete, Phase 5 deferred
2. **INTEGRATION_ROADMAP.md** - Cascor/Canopy integration, most items resolved
3. **PRE-DEPLOYMENT_ROADMAP.md** - Pre-deployment assessment, most P0/P1 items resolved
4. **PRE-DEPLOYMENT_ROADMAP-2.md** - Phase 2 remaining items, 74% complete
5. **Source Code Review** - Static analysis findings from mypy, flake8, and manual inspection

### Current State Summary

| Metric | Value |
|--------|-------|
| Version | 0.3.0 |
| Test Count | 207 (all passing) |
| Code Coverage | 100% |
| mypy Errors | 20 (all in test files) |
| flake8 Issues | ~30 (mix of real issues and intentional patterns) |
| black/isort | Clean |
| Python Support | >=3.11 (tested 3.11-3.14) |

---

## Table of Contents

- [Phase 1: Code Quality & Static Analysis Fixes](#phase-1-code-quality--static-analysis-fixes)
- [Phase 2: Integration Infrastructure](#phase-2-integration-infrastructure)
- [Phase 3: Client Package Consolidation](#phase-3-client-package-consolidation)
- [Phase 4: Extended Capabilities](#phase-4-extended-capabilities)
- [Phase 5: Ecosystem Enhancements](#phase-5-ecosystem-enhancements)
- [Deferred Items](#deferred-items)
- [Cross-Project Reference](#cross-project-reference)

---

## Phase 1: Code Quality & Static Analysis Fixes

**Priority**: HIGH | **Risk**: LOW | **Blocking**: None
**Rationale**: These are existing code quality issues that should be resolved to maintain the project's high standards and prevent technical debt accumulation.

### DATA-001: Fix mypy Type Errors in Test Files

**Priority**: HIGH | **Status**: NOT STARTED | **Effort**: Small
**Source**: Source code review (mypy analysis)

20 type errors across 4 test files. All errors are in test code only; production code is clean.

**`tests/unit/test_storage.py`** (6 errors):
- `union-attr`: Accessing `.metadata` / `.artifact_path` on `Optional[DatasetMeta]` without narrowing
- `arg-type`: Passing `MagicMock` where `DatasetMeta` expected

**`tests/integration/test_storage_workflow.py`** (6 errors):
- `union-attr`: Same pattern as test_storage.py - accessing attributes on Optional types
- `arg-type`: Mock type mismatches

**`tests/unit/test_spiral_generator.py`** (1 error):
- `arg-type`: Literal string `"invalid_algorithm"` incompatible with `Literal["modern", "legacy_cascor"]`

**`tests/unit/test_api_app.py`** (4 errors):
- `attr-defined`: Accessing `app` attribute on lifespan context manager return type

**Resolution**: Add type narrowing assertions, use `cast()`, or add targeted `# type: ignore` comments with explanations. Consider updating mypy overrides in `pyproject.toml` for test modules.

---

### DATA-002: Fix flake8 Unused Imports in datasets.py

**Priority**: HIGH | **Status**: NOT STARTED | **Effort**: Small
**Source**: Source code review (flake8 analysis)

**File**: `juniper_data/api/routes/datasets.py`
- F401: `typing.Any` imported but unused
- F401: `typing.Dict` imported but unused

**Resolution**: Remove unused imports.

---

### DATA-003: Fix flake8 Issues in generate_golden_datasets.py

**Priority**: MEDIUM | **Status**: NOT STARTED | **Effort**: Small
**Source**: Source code review (flake8 analysis)

**File**: `juniper_data/tests/fixtures/generate_golden_datasets.py`
- E402: Module-level import not at top of file (5 instances)
- F541: f-string without placeholders (5 instances)

**Resolution**: Reorganize imports and fix f-string expressions.

---

### DATA-004: Address B008 Warnings in API Route Defaults

**Priority**: LOW | **Status**: NOT STARTED | **Effort**: None (intentional)
**Source**: Source code review (flake8 analysis)

**File**: `juniper_data/api/routes/datasets.py`
- B008: 9 instances of function calls in argument defaults (e.g., `Query(default=...)`, `Depends(...)`)

**Resolution**: These are **intentional FastAPI patterns**. No action needed. Consider adding `# noqa: B008` comments or adding B008 to the per-file flake8 ignore list in `pyproject.toml` for route files.

---

### DATA-005: Address SIM117 Suggestions in Test Files

**Priority**: LOW | **Status**: NOT STARTED | **Effort**: Small
**Source**: Source code review (flake8 analysis)

Multiple test files have SIM117 suggestions to combine nested `with` statements.

**Resolution**: Already handled by relaxed SIM117 rules for test code (added in v0.3.0). No action needed unless cleanup is desired.

---

## Phase 2: Integration Infrastructure

**Priority**: HIGH | **Risk**: MEDIUM | **Blocking**: Phase 3
**Rationale**: JuniperData is consumed by both JuniperCascor and JuniperCanopy. Robust integration infrastructure is essential for the ecosystem to function as designed.

### DATA-006: Create Dockerfile for JuniperData Service

**Priority**: HIGH | **Status**: NOT STARTED | **Effort**: Medium
**Source**: Source code review, JUNIPER_CASCOR_SPIRAL_DATA_GEN_REFACTOR_PLAN.md

JuniperData has no Dockerfile despite being designed as a microservice. The refactoring plan documents a Docker Compose configuration where `juniper-data` runs on port 8100, but no Dockerfile exists to build the image.

**Requirements**:
- Multi-stage build (builder + runtime)
- Python >=3.11 base image
- Install with `pip install .[api]` (minimal dependencies)
- Expose port 8100
- Health check endpoint: `GET /v1/health`
- Non-root user for security
- `.dockerignore` to exclude tests, docs, notes

**Consumers**: Both JuniperCascor and JuniperCanopy docker-compose configurations reference a `juniper-data` service.

---

### DATA-007: Add Health Check Probes for Container Orchestration

**Priority**: HIGH | **Status**: NOT STARTED | **Effort**: Small
**Source**: Source code review, Docker Compose requirements

The `GET /v1/health` endpoint exists but there is no standardized health check configuration for Docker/Kubernetes readiness and liveness probes.

**Requirements**:
- Dockerfile `HEALTHCHECK` instruction using `GET /v1/health`
- Document probe configuration for docker-compose
- Consider adding `/v1/health/ready` (readiness) vs `/v1/health/live` (liveness) distinction

---

### DATA-008: End-to-End Integration Tests with Live Service

**Priority**: HIGH | **Status**: NOT STARTED | **Effort**: Large
**Source**: JUNIPER_CASCOR_SPIRAL_DATA_GEN_REFACTOR_PLAN.md (remaining work), INTEGRATION_ROADMAP.md

No E2E tests exist that verify the full flow:
1. Start JuniperData service
2. Client creates dataset via REST API
3. Client downloads NPZ artifact
4. Verify data integrity (shapes, dtypes, determinism)

**Requirements**:
- Test fixture that starts/stops JuniperData server (or uses `TestClient`)
- Verify `POST /v1/datasets` with spiral generator params
- Verify `GET /v1/datasets/{id}/artifact` returns valid NPZ
- Verify NPZ contains expected keys: `X_train`, `y_train`, `X_test`, `y_test`, `X_full`, `y_full`
- Verify array shapes, dtypes (`float32`), and deterministic output with seed
- Mark with `@pytest.mark.slow` for weekly CI runs
- Verify both `algorithm="modern"` and `algorithm="legacy_cascor"` modes

---

### DATA-009: API Versioning Strategy Documentation

**Priority**: MEDIUM | **Status**: NOT STARTED | **Effort**: Small
**Source**: Source code review

The API uses `/v1/` prefix but there is no documented versioning strategy for:
- When to increment API version
- How to handle backward-incompatible changes
- Deprecation policy for old API versions
- Client compatibility guarantees

**Resolution**: Document API versioning policy in README.md or a dedicated API docs file.

---

### DATA-010: NPZ Artifact Schema Documentation

**Priority**: MEDIUM | **Status**: NOT STARTED | **Effort**: Small
**Source**: JuniperCascor/JuniperCanopy integration analysis

The NPZ data contract is implicit. Both JuniperCascor and JuniperCanopy expect specific array keys and dtypes:

```
X_train: np.ndarray (n_train, 2) float32
y_train: np.ndarray (n_train, n_classes) float32 (one-hot)
X_test:  np.ndarray (n_test, 2) float32
y_test:  np.ndarray (n_test, n_classes) float32 (one-hot)
X_full:  np.ndarray (n_total, 2) float32
y_full:  np.ndarray (n_total, n_classes) float32 (one-hot)
```

**Resolution**: Add explicit schema documentation and consider adding schema validation on the server side to prevent breaking changes.

---

### DATA-011: Parameter Validation Parity with Consumers

**Priority**: MEDIUM | **Status**: NOT STARTED | **Effort**: Small
**Source**: JuniperCascor/JuniperCanopy integration analysis

JuniperCascor maps parameters with different names:
- `n_points` -> `n_points_per_spiral`
- `noise_level` -> `noise`

JuniperCanopy uses different default values between demo_mode (noise=0.1) and cascor_integration (noise=0.0).

**Resolution**: Document the canonical parameter names and ensure clear error messages when deprecated/incorrect parameter names are used. Consider adding parameter aliases for common consumer patterns.

---

## Phase 3: Client Package Consolidation

**Priority**: MEDIUM | **Risk**: MEDIUM | **Blocking**: None
**Rationale**: Both JuniperCascor and JuniperCanopy maintain their own copies of `juniper_data_client/`. This creates maintenance burden and divergence risk.

### DATA-012: Extract Shared JuniperData Client Package

**Priority**: MEDIUM | **Status**: NOT STARTED | **Effort**: Large
**Source**: JUNIPER_CASCOR_SPIRAL_DATA_GEN_REFACTOR_PLAN.md, JuniperCanopy INTEGRATION_DEVELOPMENT_PLAN.md

Both consumers have near-identical client code:
- `JuniperCascor/src/juniper_data_client/client.py`
- `JuniperCanopy/src/juniper_data_client/client.py`

**Options** (in order of preference):
1. **PyPI package**: Publish `juniper-data-client` as a pip-installable package
2. **Git submodule**: Share the client as a git submodule in both projects
3. **Monorepo**: Move all three projects into a single repository

**Recommendation**: Option 1 (PyPI package) provides the cleanest dependency management. The client could live in the JuniperData repository under `client/` and be published separately.

**Scope**:
- Extract client code into standalone package
- Add client-specific tests
- Publish to PyPI (or private index)
- Update JuniperCascor and JuniperCanopy to use the shared package
- Remove duplicated client code from both projects

---

### DATA-013: Client Test Coverage

**Priority**: MEDIUM | **Status**: NOT STARTED | **Effort**: Medium
**Source**: JuniperCanopy exploration (0% coverage on client module)

JuniperCanopy reports 0% coverage on its `juniper_data_client/` module. JuniperCascor has 17 tests for its copy. When consolidating, ensure comprehensive test coverage.

**Requirements**:
- URL normalization tests
- Request/response handling tests
- Error handling and timeout tests
- NPZ download and parsing tests
- Mock-based (no live service dependency for unit tests)

---

## Phase 4: Extended Capabilities

**Priority**: LOW | **Risk**: LOW | **Blocking**: None
**Rationale**: Enhancements that improve the service but are not required for current integration.

### DATA-014: Additional Generator Types

**Priority**: LOW | **Status**: NOT STARTED | **Effort**: Large
**Source**: JUNIPER_CASCOR_SPIRAL_DATA_GEN_REFACTOR_PLAN.md (Phase 5: Extended Data Sources)

Current generators: `spiral` only.

**Potential additions**:
- XOR classification dataset
- Gaussian mixture models
- Concentric circles/rings
- Checkerboard pattern
- Custom CSV/JSON import

**Framework**: The generator plugin architecture (`generators/` package, `GENERATOR_REGISTRY`) already supports adding new generators following the `SpiralGenerator` pattern.

---

### DATA-015: Storage Backend Extensions

**Priority**: LOW | **Status**: NOT STARTED | **Effort**: Large
**Source**: JUNIPER_CASCOR_SPIRAL_DATA_GEN_REFACTOR_PLAN.md (Phase 5)

Current storage backends: `InMemoryDatasetStore`, `LocalFSDatasetStore`.

**Potential additions**:
- S3/GCS object storage
- Database-backed metadata store (SQLite/PostgreSQL)
- HuggingFace Datasets integration
- Redis-based caching layer

**Framework**: The `DatasetStore` abstract base class already defines the interface for new backends.

---

### DATA-016: Dataset Lifecycle Management

**Priority**: LOW | **Status**: NOT STARTED | **Effort**: Medium
**Source**: Source code review

Current API supports create/read/delete but lacks:
- Dataset expiration / TTL
- Bulk operations (list with filtering, batch delete)
- Dataset tagging/labeling
- Usage tracking / access counts

---

### DATA-017: API Rate Limiting and Authentication

**Priority**: LOW | **Status**: NOT STARTED | **Effort**: Medium
**Source**: Source code review, PRE-DEPLOYMENT_ROADMAP.md

The API has no authentication or rate limiting. For internal use this is acceptable, but for any external exposure:
- Add API key authentication
- Add rate limiting middleware
- Add request logging/auditing

---

## Phase 5: Ecosystem Enhancements

**Priority**: LOW | **Risk**: LOW | **Blocking**: None
**Rationale**: Items from ecosystem roadmaps that have JuniperData implications.

### DATA-018: IPC Architecture (Full Inter-Process Communication)

**Priority**: LOW | **Status**: DEFERRED
**Source**: PRE-DEPLOYMENT_ROADMAP-2.md (P1-NEW-001)

Currently, integration is REST-only via HTTP. A full IPC architecture would add:
- gRPC support for high-performance binary streaming
- Message queue integration (for async dataset generation)
- Shared memory for co-located services

**Decision**: Deferred until REST performance becomes a bottleneck. Current HTTP+NPZ approach is sufficient for research workloads.

---

### DATA-019: GPU-Accelerated Data Generation

**Priority**: LOW | **Status**: NOT STARTED
**Source**: PRE-DEPLOYMENT_ROADMAP-2.md (P3-NEW-003)

Current generation is CPU-only via NumPy. For very large datasets, GPU acceleration via CuPy or JAX could improve throughput.

**Decision**: Not needed for current dataset sizes (hundreds to thousands of points). Revisit when dataset sizes reach millions.

---

### DATA-020: Continuous Profiling Integration

**Priority**: LOW | **Status**: NOT STARTED
**Source**: PRE-DEPLOYMENT_ROADMAP-2.md (P3-NEW-004)

Add performance monitoring for the JuniperData service:
- Response time tracking per endpoint
- Memory usage profiling
- Dataset generation time metrics
- Prometheus/Grafana integration

---

## Deferred Items

These items are explicitly deferred and will be revisited based on project needs.

| ID | Item | Source | Reason |
|----|------|--------|--------|
| DATA-018 | IPC Architecture | PRE-DEPLOYMENT_ROADMAP-2.md | REST sufficient for research workloads |
| DATA-014 | Extended Generators | Refactor Plan Phase 5 | Current spiral generator meets all needs |
| DATA-015 | Storage Extensions | Refactor Plan Phase 5 | LocalFS adequate for single-machine use |
| DATA-019 | GPU Acceleration | PRE-DEPLOYMENT_ROADMAP-2.md | Dataset sizes too small to benefit |

---

## Cross-Project Reference

### Items Owned by JuniperCascor (Not JuniperData Scope)

These items appear in the reviewed documentation but are owned by JuniperCascor:

| ID | Item | Status | Source |
|----|------|--------|--------|
| CAS-REF-001 | Code coverage below 90% | IN PROGRESS | PRE-DEPLOYMENT_ROADMAP-2.md (P2-NEW-001) |
| CAS-REF-002 | CI/CD coverage gates not enforced | NOT STARTED | PRE-DEPLOYMENT_ROADMAP-2.md (P2-NEW-002) |
| CAS-REF-003 | Type errors gradual fix | IN PROGRESS | PRE-DEPLOYMENT_ROADMAP-2.md (P2-NEW-006) |
| CAS-REF-004 | Legacy spiral code removal | NOT STARTED | Refactor Plan |
| CAS-REF-005 | RemoteWorkerClient integration | NOT STARTED | PRE-DEPLOYMENT_ROADMAP.md (INTEG-002) |

### Items Owned by JuniperCanopy (Not JuniperData Scope)

| ID | Item | Status | Source |
|----|------|--------|--------|
| CAN-REF-001 | JuniperData client not actively used | NOT STARTED | Canopy INTEGRATION_DEVELOPMENT_PLAN.md |
| CAN-REF-002 | No JUNIPER_DATA_URL in app_config.yaml | NOT STARTED | Canopy exploration |
| CAN-REF-003 | No JuniperData in docker-compose.yaml | NOT STARTED | Canopy exploration |
| CAN-REF-004 | Parameter inconsistencies (noise) | NOT STARTED | Canopy exploration |
| CAN-REF-005 | CAN-001 through CAN-021 enhancements | Various | PRE-DEPLOYMENT_ROADMAP.md Section 7 |

### Items Shared Across Projects

| ID | Item | Owners | Status |
|----|------|--------|--------|
| DATA-012 | Client package consolidation | Data, Cascor, Canopy | NOT STARTED |
| DATA-008 | E2E integration tests | Data (primary), Cascor, Canopy | NOT STARTED |
| DATA-006 | Dockerfile | Data (primary), Cascor/Canopy (consumers) | NOT STARTED |

---

## Implementation Priority Matrix

### Immediate (Next Sprint)

| ID | Item | Priority | Effort | Impact |
|----|------|----------|--------|--------|
| DATA-001 | Fix mypy type errors in tests | HIGH | Small | Code quality |
| DATA-002 | Fix unused imports in datasets.py | HIGH | Small | Code quality |
| DATA-003 | Fix golden dataset fixture issues | MEDIUM | Small | Code quality |

### Short-Term (1-2 Sprints)

| ID | Item | Priority | Effort | Impact |
|----|------|----------|--------|--------|
| DATA-006 | Create Dockerfile | HIGH | Medium | Deployment |
| DATA-007 | Health check probes | HIGH | Small | Operations |
| DATA-008 | E2E integration tests | HIGH | Large | Reliability |
| DATA-010 | NPZ schema documentation | MEDIUM | Small | Integration |

### Medium-Term (3-4 Sprints)

| ID | Item | Priority | Effort | Impact |
|----|------|----------|--------|--------|
| DATA-009 | API versioning documentation | MEDIUM | Small | Maintainability |
| DATA-011 | Parameter validation parity | MEDIUM | Small | Integration |
| DATA-012 | Client package consolidation | MEDIUM | Large | Maintainability |
| DATA-013 | Client test coverage | MEDIUM | Medium | Reliability |

### Long-Term (Backlog)

| ID | Item | Priority | Effort | Impact |
|----|------|----------|--------|--------|
| DATA-014 | Additional generators | LOW | Large | Capability |
| DATA-015 | Storage extensions | LOW | Large | Scalability |
| DATA-016 | Dataset lifecycle management | LOW | Medium | Operations |
| DATA-017 | API rate limiting/auth | LOW | Medium | Security |
| DATA-018 | IPC architecture | LOW | Large | Performance |
| DATA-019 | GPU acceleration | LOW | Large | Performance |
| DATA-020 | Continuous profiling | LOW | Medium | Operations |

---

## Summary Statistics

| Category | Count |
|----------|-------|
| Total Items | 20 |
| HIGH Priority | 5 (DATA-001, 002, 006, 007, 008) |
| MEDIUM Priority | 6 (DATA-003, 009, 010, 011, 012, 013) |
| LOW Priority | 6 (DATA-004, 005, 014, 015, 016, 017) |
| DEFERRED | 3 (DATA-018, 019, 020) |
| Cross-Project References | 10 (CAS: 5, CAN: 5) |

---

## Document History

| Date | Author | Changes |
|------|--------|---------|
| 2026-02-05 | Paul Calnon | Initial creation from documentation review and source code analysis |
