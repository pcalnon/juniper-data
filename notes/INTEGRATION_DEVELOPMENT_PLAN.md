# JuniperData Integration Development Plan

**Project**: Juniper Data - Dataset Generation Service
**Version**: 0.4.0
**Author**: Paul Calnon
**Created**: 2026-02-05
**Status**: Active Development
**Last Updated**: 2026-02-07

---

## Overview

This document compiles all outstanding work items for the JuniperData project, synthesized from:

1. **JUNIPER_CASCOR_SPIRAL_DATA_GEN_REFACTOR_PLAN.md** - Phases 0-4 complete, Phase 5 deferred
2. **INTEGRATION_ROADMAP.md** - Cascor/Canopy integration, most items resolved
3. **PRE-DEPLOYMENT_ROADMAP.md** - Pre-deployment assessment, most P0/P1 items resolved
4. **PRE-DEPLOYMENT_ROADMAP-2.md** - Phase 2 remaining items, 74% complete
5. **Source Code Review** - Static analysis findings from mypy, flake8, and manual inspection

### Current State Summary

| Metric         | Value                                                                                                 |
| -------------- | ----------------------------------------------------------------------------------------------------- |
| Version        | 0.4.0                                                                                                 |
| Test Count     | 411 (342 unit + 69 integration, all passing)                                                          |
| Code Coverage  | 60.57% total (**FAILS 80% threshold**); core modules ~95%+; see coverage gaps below                   |
| mypy Errors    | 0 (78 source files)                                                                                   |
| flake8 Issues  | 16 (7 F401 unused imports, 4 E741 ambiguous vars, 1 W293, 1 B007, + B008 intentional)                 |
| black/isort    | Clean                                                                                                 |
| Python Support | >=3.11 (tested 3.11-3.14)                                                                             |
| Generators     | 5 registered (spiral, xor, gaussian, circles, checkerboard); 3 code-only (csv_import, mnist, arc_agi) |
| Storage        | 3 tested (memory, localfs, cached); 4 code-only/0% coverage (redis, hf, postgres, kaggle)             |

### Coverage Gaps (Modules at 0% Coverage)

| Module                          | Lines | Status                                          |
| ------------------------------- | ----- | ----------------------------------------------- |
| `generators/arc_agi/` (3 files) | 133   | Code only - no tests, not in GENERATOR_REGISTRY |
| `generators/mnist/` (3 files)   | 63    | Code only - no tests, not in GENERATOR_REGISTRY |
| `storage/hf_store.py`           | 96    | Code only - no tests                            |
| `storage/kaggle_store.py`       | 127   | Code only - no tests, has F401 unused imports   |
| `storage/postgres_store.py`     | 101   | Code only - no tests, has F401 unused imports   |
| `storage/redis_store.py`        | 103   | Code only - no tests, has F401 unused imports   |

### Partial Coverage Concerns

| Module                                 | Coverage | Notes                                        |
| -------------------------------------- | -------- | -------------------------------------------- |
| `storage/__init__.py`                  | 52.94%   | Conditional imports for optional backends    |
| `storage/cached.py`                    | 76.47%   | Has 11 unit tests but gaps remain            |
| `storage/local_fs.py`                  | 79.57%   | Missing coverage on some error paths         |
| `generators/csv_import/generator.py`   | 88.14%   | Has 14 unit tests, not in GENERATOR_REGISTRY |
| `generators/checkerboard/generator.py` | 94.44%   | Minor gap on line 88                         |
| `generators/gaussian/generator.py`     | 95.52%   | Minor gap on line 143                        |

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

**Priority**: HIGH | **Status**: COMPLETE | **Effort**: Small
**Source**: Source code review (mypy analysis)
**Completed**: 2026-02-05

20 type errors across 4 test files were fixed. All errors were in test code only; production code was already clean.

**Resolution Applied**:

- **`tests/unit/test_storage.py`**: Added `assert ... is not None` type narrowing assertions before accessing Optional attributes
- **`tests/integration/test_storage_workflow.py`**: Added type narrowing assertions for `DatasetMeta` and `bytes` returns
- **`tests/unit/test_spiral_generator.py`**: Added `# type: ignore[arg-type]` with explanation for negative test case
- **`tests/unit/test_api_app.py`**: Used `getattr()` pattern to access dynamic route attributes safely

**Verification**: `mypy` now reports "Success: no issues found in 4 source files"

---

### DATA-002: Fix flake8 Unused Imports in datasets.py

**Priority**: HIGH | **Status**: COMPLETE | **Effort**: Small
**Source**: Source code review (flake8 analysis)
**Completed**: 2026-02-05

**File**: `juniper_data/api/routes/datasets.py`

- F401: `typing.Any` imported but unused
- F401: `typing.Dict` imported but unused

**Resolution Applied**: Removed unused `Any` and `Dict` imports from typing module.

**Verification**: `flake8 --select=F401` now reports no issues.

---

### DATA-003: Fix flake8 Issues in generate_golden_datasets.py

**Priority**: MEDIUM | **Status**: COMPLETE | **Effort**: Small
**Source**: Source code review (flake8 analysis)
**Completed**: 2026-02-05

**File**: `juniper_data/tests/fixtures/generate_golden_datasets.py`
6 (DATA-001, 002, 003, 006, 007, 008)

- E402: Module-level import not at top of file (2 instances - `SpiralProblem`, `torch`)
- F541: f-string without placeholders (5 instances)

**Resolution Applied**:

- Added `# noqa: E402` comments with explanations for late imports (required due to `sys.path` manipulation for JuniperCascor import)
- Converted f-strings without placeholders to regular strings

**Verification**: `flake8 --select=F541,E402` now reports no issues.

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

**Priority**: HIGH | **Status**: COMPLETE | **Effort**: Medium
**Source**: Source code review, JUNIPER_CASCOR_SPIRAL_DATA_GEN_REFACTOR_PLAN.md
**Completed**: 2026-02-05

JuniperData has no Dockerfile despite being designed as a microservice. The refactoring plan documents a Docker Compose configuration where `juniper-data` runs on port 8100, but no Dockerfile exists to build the image.

**Requirements**:

- Multi-stage build (builder + runtime)
- Python >=3.11 base image
- Install with `pip install .[api]` (minimal dependencies)
- Expose port 8100
- Health check endpoint: `GET /v1/health`
- Non-root user for security
- `.dockerignore` to exclude tests, docs, notes

**Resolution Applied**:

- Created `Dockerfile` with multi-stage build (builder + runtime stages)
- Uses `python:3.11-slim` base image for minimal footprint
- Installs with `pip install .[api]` for minimal dependencies
- Creates non-root `juniper` user (UID 1000) for security
- Exposes port 8100 with environment variable configuration
- Includes HEALTHCHECK instruction for container orchestration
- Created `.dockerignore` to exclude tests, docs, notes, and other development files

**Consumers**: Both JuniperCascor and JuniperCanopy docker-compose configurations reference a `juniper-data` service.

---

### DATA-007: Add Health Check Probes for Container Orchestration

**Priority**: HIGH | **Status**: COMPLETE | **Effort**: Small
**Source**: Source code review, Docker Compose requirements
**Completed**: 2026-02-05

The `GET /v1/health` endpoint exists but there is no standardized health check configuration for Docker/Kubernetes readiness and liveness probes.

**Requirements**:

- Dockerfile `HEALTHCHECK` instruction using `GET /v1/health`
- Document probe configuration for docker-compose
- Consider adding `/v1/health/ready` (readiness) vs `/v1/health/live` (liveness) distinction

**Resolution Applied**:

- Added `HEALTHCHECK` instruction to Dockerfile (30s interval, 10s timeout, 5s start period, 3 retries)
- Added `/v1/health/live` endpoint for liveness probes (returns `{"status": "alive"}`)
- Added `/v1/health/ready` endpoint for readiness probes (returns `{"status": "ready", "version": "..."}`)
- Original `/v1/health` endpoint preserved for backward compatibility
- Added 4 new integration tests for health probe endpoints

---

### DATA-008: End-to-End Integration Tests with Live Service

**Priority**: HIGH | **Status**: COMPLETE | **Effort**: Large
**Source**: JUNIPER_CASCOR_SPIRAL_DATA_GEN_REFACTOR_PLAN.md (remaining work), INTEGRATION_ROADMAP.md
**Completed**: 2026-02-05

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

**Resolution Applied**:

- Created `juniper_data/tests/integration/test_e2e_workflow.py` with 14 comprehensive E2E tests
- **TestE2EModernAlgorithm** (3 tests): create/download/verify flow, determinism with seed, different seed produces different data
- **TestE2ELegacyCascorAlgorithm** (2 tests): legacy algorithm flow, legacy vs modern comparison
- **TestE2EDataContract** (5 tests): NPZ keys contract, feature dimensions, one-hot labels, train/test split ratios, metadata consistency
- **TestE2EErrorHandling** (4 tests): invalid generator, invalid params, nonexistent dataset, delete verification
- All tests marked with `@pytest.mark.integration` and `@pytest.mark.slow`
- Uses FastAPI `TestClient` with in-memory storage for isolation

---

### DATA-009: API Versioning Strategy Documentation

**Priority**: MEDIUM | **Status**: COMPLETE | **Effort**: Small
**Source**: Source code review
**Completed**: 2026-02-05

The API uses `/v1/` prefix but there is no documented versioning strategy for:

- When to increment API version
- How to handle backward-incompatible changes
- Deprecation policy for old API versions
- Client compatibility guarantees

**Resolution Applied**:

- Created comprehensive API documentation: `docs/api/JUNIPER_DATA_API.md`
- Documents versioning policy following SemVer principles
- Defines backward compatibility guarantees for major versions
- Specifies deprecation policy (2 minor versions notice, 6 months support)
- Lists breaking changes that require major version bump

---

### DATA-010: NPZ Artifact Schema Documentation

**Priority**: MEDIUM | **Status**: COMPLETE | **Effort**: Small
**Source**: JuniperCascor/JuniperCanopy integration analysis
**Completed**: 2026-02-05

The NPZ data contract is implicit. Both JuniperCascor and JuniperCanopy expect specific array keys and dtypes:

```bash
X_train: np.ndarray (n_train, 2) float32
y_train: np.ndarray (n_train, n_classes) float32 (one-hot)
X_test:  np.ndarray (n_test, 2) float32
y_test:  np.ndarray (n_test, n_classes) float32 (one-hot)
X_full:  np.ndarray (n_total, 2) float32
y_full:  np.ndarray (n_total, n_classes) float32 (one-hot)
```

**Resolution Applied**:

- Added dedicated "NPZ Artifact Schema" section in `docs/api/JUNIPER_DATA_API.md`
- Documents all 6 required array keys with shapes and dtypes
- Specifies one-hot encoding format for labels
- Includes schema validation guarantees
- Provides Python and PyTorch loading examples

---

### DATA-011: Parameter Validation Parity with Consumers

**Priority**: MEDIUM | **Status**: COMPLETE | **Effort**: Small
**Source**: JuniperCascor/JuniperCanopy integration analysis
**Completed**: 2026-02-05

JuniperCascor maps parameters with different names:

- `n_points` -> `n_points_per_spiral`
- `noise_level` -> `noise`

JuniperCanopy uses different default values between demo_mode (noise=0.1) and cascor_integration (noise=0.0).

**Resolution Applied**:

- Added parameter aliases using Pydantic `AliasChoices` in `SpiralParams`
- `n_points` accepted as alias for `n_points_per_spiral`
- `noise_level` accepted as alias for `noise`
- Canonical names preserved in JSON schema for documentation
- Added 5 new unit tests verifying alias behavior
- Documented aliases in module docstring and API documentation

---

## Phase 3: Client Package Consolidation

**Priority**: MEDIUM | **Risk**: MEDIUM | **Blocking**: None
**Rationale**: Both JuniperCascor and JuniperCanopy maintain their own copies of `juniper_data_client/`. This creates maintenance burden and divergence risk.

### DATA-012: Extract Shared JuniperData Client Package

**Priority**: MEDIUM | **Status**: COMPLETE | **Effort**: Large
**Source**: JUNIPER_CASCOR_SPIRAL_DATA_GEN_REFACTOR_PLAN.md, JuniperCanopy INTEGRATION_DEVELOPMENT_PLAN.md
**Completed**: 2026-02-06

Both consumers had near-identical client code:

- `JuniperCascor/juniper_cascor/src/juniper_data_client/client.py`
- `JuniperCanopy/juniper_canopy/src/juniper_data_client/client.py`

**Resolution Applied** (Option 1 - PyPI package):

Created `juniper-data-client` as a standalone pip-installable package in `juniper_data_client/`:

- **Package structure**: `juniper_data_client/` with `client.py`, `exceptions.py`, `__init__.py`, `py.typed`
- **pyproject.toml**: Full package configuration with dependencies (numpy, requests, urllib3)
- **README.md**: Comprehensive documentation with usage examples
- **Test suite**: 35 unit tests using `responses` library for HTTP mocking (96% coverage)
- **Type annotations**: Full mypy strict mode compliance with `py.typed` marker

**Features consolidated from both implementations**:

- URL normalization (scheme handling, trailing slashes, /v1 suffix removal)
- Session management with connection pooling
- All dataset endpoints (create, list, get, delete, preview, artifact download)
- Generator endpoints (list, schema)
- Health check endpoints (health, live, ready, wait_for_ready)
- Convenience method `create_spiral_dataset()` with common parameters

**Enhancements over original implementations**:

- Automatic retry logic with configurable backoff (429, 5xx errors)
- Connection pooling via `requests.Session` with `HTTPAdapter`
- Custom exceptions hierarchy (`JuniperDataClientError`, `JuniperDataConnectionError`, etc.)
- Context manager support for resource cleanup
- `wait_for_ready()` method for service availability polling
- Full type hints with mypy strict mode support

**Installation**:

```bash
pip install -e juniper_data_client/[test]  # Development
pip install juniper-data-client            # From PyPI (when published)
```

**Next Steps**: DATA-012-A (update JuniperCascor and JuniperCanopy to use shared package)

- Publish to PyPI (or private index)
- Update JuniperCascor and JuniperCanopy to use the shared package
- Remove duplicated client code from both projects

---

### DATA-013: Client Test Coverage

**Priority**: MEDIUM | **Status**: COMPLETE | **Effort**: Medium
**Source**: JuniperCanopy exploration (0% coverage on client module)
**Completed**: 2026-02-06

JuniperCanopy reported 0% coverage on its `juniper_data_client/` module. JuniperCascor had 17 tests for its copy.

**Resolution Applied**:

The consolidated `juniper-data-client` package includes 35 comprehensive unit tests:

- **TestUrlNormalization** (7 tests): Scheme handling, trailing slashes, /v1 suffix, HTTPS, whitespace
- **TestClientConfiguration** (3 tests): Default values, custom values, context manager
- **TestHealthEndpoints** (4 tests): Health check, is_ready true/false, connection errors
- **TestGeneratorEndpoints** (3 tests): List generators, get schema, not found
- **TestDatasetCreation** (4 tests): Success, convenience method, validation errors (400/422)
- **TestDatasetRetrieval** (3 tests): List datasets, get metadata, not found
- **TestArtifactDownload** (3 tests): NPZ parsing, raw bytes, not found
- **TestPreview** (1 test): Get preview
- **TestDatasetDeletion** (2 tests): Delete success, not found
- **TestErrorHandling** (5 tests): Connection, timeout, generic, server, detail extraction

**Coverage**: 96% (35 tests, 0 failures)
**Mocking**: Uses `responses` library for HTTP mocking (no live service required)

---

## Phase 4: Extended Capabilities

**Priority**: LOW | **Risk**: LOW | **Blocking**: None
**Rationale**: Enhancements that improve the service but are not required for current integration.

### DATA-014: Additional Generator Types

**Priority**: LOW | **Status**: IN PROGRESS | **Effort**: Large
**Source**: JUNIPER_CASCOR_SPIRAL_DATA_GEN_REFACTOR_PLAN.md (Phase 5: Extended Data Sources)

**GENERATOR_REGISTRY** (5 registered, fully functional):

| Generator | Directory | Registered | Tests | Coverage |
| --------- | --------- | ---------- | ----- | -------- |
| spiral | `generators/spiral/` | Yes | 43 | 100% |
| xor | `generators/xor/` | Yes | 18 | 100% |
| gaussian | `generators/gaussian/` | Yes | 26 | ~96% |
| circles | `generators/circles/` | Yes | 22 | 100% |
| checkerboard | `generators/checkerboard/` | Yes | 17 | ~94% |

**Code-only generators** (3 not registered, incomplete):

| Generator | Directory | Registered | Tests | Coverage | Issue |
| --------- | --------- | ---------- | ----- | -------- | ----- |
| csv_import | `generators/csv_import/` | **No** | 14 | 88.14% | Not in GENERATOR_REGISTRY; has E741 flake8 warnings |
| mnist | `generators/mnist/` | **No** | 0 | **0%** | No tests, not registered, requires `datasets` package |
| arc_agi | `generators/arc_agi/` | **No** | 0 | **0%** | No tests, not registered, requires `datasets` package |

**XOR Generator Added** (2026-02-06):

Created `juniper_data/generators/xor/` package:

- `params.py` - `XorParams` model with:
  - `n_points_per_quadrant`: Points per quadrant (default: 50)
  - `x_range`, `y_range`: Coordinate ranges (default: 1.0)
  - `margin`: Exclusion zone around axes (default: 0.1)
  - `noise`: Gaussian noise level (default: 0.0)
  - `seed`, `train_ratio`, `test_ratio`, `shuffle`
- `generator.py` - `XorGenerator` class following `SpiralGenerator` pattern
- 18 unit tests with full coverage

**XOR dataset characteristics**:

- 4 quadrants around origin
- Quadrants 1 and 3 (x*y > 0) → Class 0
- Quadrants 2 and 4 (x*y < 0) → Class 1
- Balanced classes (2 quadrants each)
- Configurable margin prevents points too close to axes

**Other completed generators** (2026-02-06/07):

- **Gaussian blobs** (`generators/gaussian/`): Mixture-of-Gaussians classification with configurable centers, covariance, and noise. 26 unit tests.
- **Concentric circles** (`generators/circles/`): Binary classification with inner and outer circle classes. 22 unit tests.
- **Checkerboard** (`generators/checkerboard/`): 2D grid with alternating class squares. 17 unit tests.

**Partially complete generators** (code exists, needs finishing):

- **CSV/JSON Import** (`generators/csv_import/`): Import custom datasets from CSV/JSON files. 14 unit tests. **Needs: registration in GENERATOR_REGISTRY, fix E741 flake8 warnings.**
- **MNIST** (`generators/mnist/`): MNIST and Fashion-MNIST via HuggingFace. **Needs: unit tests, registration in GENERATOR_REGISTRY.** Requires `datasets` package.
- **ARC-AGI** (`generators/arc_agi/`): Abstraction and Reasoning Corpus tasks via HuggingFace or local JSON. **Needs: unit tests, registration in GENERATOR_REGISTRY.** Requires `datasets` package.

**Remaining work to complete DATA-014**:

1. Register `csv_import` in `GENERATOR_REGISTRY` (in `api/routes/generators.py`)
2. Fix E741 flake8 warnings in `csv_import/generator.py` (lines 172, 176)
3. Write unit tests for `mnist` generator
4. Write unit tests for `arc_agi` generator
5. Register `mnist` and `arc_agi` in `GENERATOR_REGISTRY` (conditional on `datasets` package availability)

**Framework**: The generator plugin architecture (`generators/` package, `GENERATOR_REGISTRY`) supports adding new generators following the established patterns.

---

### DATA-015: Storage Backend Extensions

**Priority**: LOW | **Status**: IN PROGRESS | **Effort**: Large
**Source**: JUNIPER_CASCOR_SPIRAL_DATA_GEN_REFACTOR_PLAN.md (Phase 5)

Core storage backends: `InMemoryDatasetStore` (100%), `LocalFSDatasetStore` (79.57%).

**Storage backend status**:

| Backend | File | Tests | Coverage | Status |
| ------- | ---- | ----- | -------- | ------ |
| InMemoryDatasetStore | `storage/memory.py` | 44 (shared) | 100% | Complete |
| LocalFSDatasetStore | `storage/local_fs.py` | 44 (shared) | 79.57% | Complete (minor gaps) |
| CachedDatasetStore | `storage/cached.py` | 11 | 76.47% | Partial - needs more test coverage |
| RedisDatasetStore | `storage/redis_store.py` | 0 | **0%** | Code only - no tests, has F401 warnings |
| HuggingFaceDatasetStore | `storage/hf_store.py` | 0 | **0%** | Code only - no tests |
| PostgresDatasetStore | `storage/postgres_store.py` | 0 | **0%** | Code only - no tests, has F401/W293 warnings |
| KaggleDatasetStore | `storage/kaggle_store.py` | 0 | **0%** | Code only - no tests, has F401/E741 warnings |

**Completed implementations**:

- **CachedDatasetStore** (`storage/cached.py`): Composable caching wrapper that wraps a primary store with a cache store for read-through caching. Supports write-through mode, cache invalidation, and cache warming. 11 unit tests (76.47% coverage).

**Code-only implementations** (0% coverage, no tests):

- **RedisDatasetStore** (`storage/redis_store.py`): Redis-backed storage for distributed deployments. Supports TTL, key prefixes, and connection pooling. Requires optional `redis` package.
- **HuggingFaceDatasetStore** (`storage/hf_store.py`): Integration with Hugging Face datasets hub. Can load MNIST, Fashion-MNIST, and other datasets. Supports feature extraction, normalization, and one-hot encoding. Requires optional `datasets` package.
- **PostgresDatasetStore** (`storage/postgres_store.py`): PostgreSQL-backed metadata storage with filesystem artifacts. Full CRUD operations with JSONB params. Requires optional `psycopg2-binary` package.
- **KaggleDatasetStore** (`storage/kaggle_store.py`): Kaggle API integration for downloading and caching datasets. Supports dataset download, CSV parsing, and competition listing. Requires optional `kaggle` package.

**Remaining work to complete DATA-015**:

1. Write unit tests for `RedisDatasetStore` (mock `redis` package)
2. Write unit tests for `HuggingFaceDatasetStore` (mock `datasets` package)
3. Write unit tests for `PostgresDatasetStore` (mock `psycopg2`)
4. Write unit tests for `KaggleDatasetStore` (mock `kaggle` API)
5. Fix F401 unused imports in `redis_store.py`, `postgres_store.py`, `kaggle_store.py`
6. Fix W293 whitespace in `postgres_store.py`
7. Fix E741 ambiguous variable names in `kaggle_store.py`
8. Improve `CachedDatasetStore` test coverage from 76.47% to 90%+

**Remaining potential additions**:

- S3/GCS object storage
- SQLite database backend (lightweight alternative to PostgreSQL)

**Framework**: The `DatasetStore` abstract base class already defines the interface for new backends.

---

### DATA-016: Dataset Lifecycle Management

**Priority**: LOW | **Status**: COMPLETE | **Effort**: Medium
**Source**: Source code review
**Completed**: 2026-02-06

**Resolution Applied**:

Added comprehensive lifecycle management features to JuniperData:

**Enhanced DatasetMeta model** (`core/models.py`):

- `tags: List[str]` - Dataset tagging/labeling
- `ttl_seconds: Optional[int]` - Time-to-live configuration
- `expires_at: Optional[datetime]` - Computed expiration time
- `last_accessed_at: Optional[datetime]` - Access tracking
- `access_count: int` - Usage tracking

**New API models**:

- `DatasetListFilter` - Filter criteria for listing
- `DatasetListResponse` - Filtered list with pagination
- `BatchDeleteRequest/Response` - Bulk delete operations
- `UpdateTagsRequest` - Tag modification
- `DatasetStats` - Aggregate statistics

**Enhanced DatasetStore** (`storage/base.py`):

- `update_meta()` - Update metadata
- `list_all_metadata()` - List all metadata for filtering
- `record_access()` - Track access count and timestamp
- `is_expired()` - Check dataset expiration
- `delete_expired()` - Cleanup expired datasets
- `filter_datasets()` - Filter by generator, tags, dates, sample count
- `batch_delete()` - Delete multiple datasets
- `get_stats()` - Aggregate statistics

**New API endpoints** (`api/routes/datasets.py`):

- `GET /v1/datasets/filter` - Filter datasets with pagination
- `GET /v1/datasets/stats` - Aggregate statistics
- `POST /v1/datasets/batch-delete` - Bulk delete
- `POST /v1/datasets/cleanup-expired` - Remove expired datasets
- `PATCH /v1/datasets/{id}/tags` - Add/remove tags

**Test coverage**: 44 new tests (27 unit + 17 integration), 97% coverage

---

### DATA-017: API Rate Limiting and Authentication

**Priority**: LOW | **Status**: COMPLETE | **Effort**: Medium
**Source**: Source code review, PRE-DEPLOYMENT_ROADMAP.md
**Completed**: 2026-02-06

**Resolution Applied**:

Implemented comprehensive API security features:

**API Key Authentication** (`api/security.py`):

- `APIKeyAuth` class validates requests against configured API keys
- Header-based authentication: `X-API-Key: <key>`
- Configurable via `JUNIPER_DATA_API_KEYS` environment variable (comma-separated)
- Disabled by default for development (open access mode)

**Rate Limiting** (`api/security.py`):

- `RateLimiter` class implements fixed-window rate limiting
- Configurable via `JUNIPER_DATA_RATE_LIMIT_ENABLED` and `JUNIPER_DATA_RATE_LIMIT_REQUESTS_PER_MINUTE`
- Per-client tracking (by API key or IP address)
- Thread-safe implementation for single-process deployments
- Returns standard rate limit headers: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`

**Security Middleware** (`api/middleware.py`):

- `SecurityMiddleware` applies auth and rate limiting to all requests
- Exempt paths: `/v1/health`, `/v1/health/live`, `/v1/health/ready`, `/docs`, `/openapi.json`, `/redoc`
- Proper error responses with JSON body and appropriate headers

**Settings** (`api/settings.py`):

- `api_keys: Optional[list[str]]` - List of valid API keys
- `rate_limit_enabled: bool` - Enable/disable rate limiting
- `rate_limit_requests_per_minute: int` - Max requests per minute

**Test coverage**: 31 tests (19 unit + 12 integration)

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

| ID       | Item             | Source                      | Reason                                 |
| -------- | ---------------- | --------------------------- | -------------------------------------- |
| DATA-018 | IPC Architecture | PRE-DEPLOYMENT_ROADMAP-2.md | REST sufficient for research workloads |
| DATA-019 | GPU Acceleration | PRE-DEPLOYMENT_ROADMAP-2.md | Dataset sizes too small to benefit     |

---

## Cross-Project Reference

### Items Owned by JuniperCascor (Not JuniperData Scope)

These items appear in the reviewed documentation but are owned by JuniperCascor:

| ID          | Item                              | Status      | Source                                   |
| ----------- | --------------------------------- | ----------- | ---------------------------------------- |
| CAS-REF-001 | Code coverage below 90%           | IN PROGRESS | PRE-DEPLOYMENT_ROADMAP-2.md (P2-NEW-001) |
| CAS-REF-002 | CI/CD coverage gates not enforced | NOT STARTED | PRE-DEPLOYMENT_ROADMAP-2.md (P2-NEW-002) |
| CAS-REF-003 | Type errors gradual fix           | IN PROGRESS | PRE-DEPLOYMENT_ROADMAP-2.md (P2-NEW-006) |
| CAS-REF-004 | Legacy spiral code removal        | NOT STARTED | Refactor Plan                            |
| CAS-REF-005 | RemoteWorkerClient integration    | NOT STARTED | PRE-DEPLOYMENT_ROADMAP.md (INTEG-002)    |

### Items Owned by JuniperCanopy (Not JuniperData Scope)

| ID          | Item                                   | Status      | Source                                 |
| ----------- | -------------------------------------- | ----------- | -------------------------------------- |
| CAN-REF-001 | JuniperData client not actively used   | NOT STARTED | Canopy INTEGRATION_DEVELOPMENT_PLAN.md |
| CAN-REF-002 | No JUNIPER_DATA_URL in app_config.yaml | NOT STARTED | Canopy exploration                     |
| CAN-REF-003 | No JuniperData in docker-compose.yaml  | NOT STARTED | Canopy exploration                     |
| CAN-REF-004 | Parameter inconsistencies (noise)      | NOT STARTED | Canopy exploration                     |
| CAN-REF-005 | CAN-001 through CAN-021 enhancements   | Various     | PRE-DEPLOYMENT_ROADMAP.md Section 7    |

### Items Shared Across Projects

| ID       | Item                         | Owners                                    | Status   |
| -------- | ---------------------------- | ----------------------------------------- | -------- |
| DATA-012 | Client package consolidation | Data, Cascor, Canopy                      | COMPLETE |
| DATA-008 | E2E integration tests        | Data (primary), Cascor, Canopy            | COMPLETE |
| DATA-006 | Dockerfile                   | Data (primary), Cascor/Canopy (consumers) | COMPLETE |

---

## Implementation Priority Matrix

### Immediate (Next Sprint) - Coverage Recovery

The overall coverage has dropped to 60.57% (below the 80% threshold) due to new modules with 0% test coverage. These must be addressed before any release.

| ID       | Item                                          | Priority | Effort | Impact       |
| -------- | --------------------------------------------- | -------- | ------ | ------------ |
| DATA-015 | Write tests for storage backends (4 at 0%)    | **HIGH** | Large  | Coverage     |
| DATA-014 | Write tests for mnist/arc_agi generators (0%) | **HIGH** | Medium | Coverage     |
| DATA-014 | Register csv_import/mnist/arc_agi in registry | MEDIUM   | Small  | Capability   |
| ---      | Fix 16 flake8 issues in new modules           | MEDIUM   | Small  | Code quality |
|          |                                               |          |        |              |

### Low Priority (Backlog)

| ID       | Item                    | Priority | Effort | Impact       |
| -------- | ----------------------- | -------- | ------ | ------------ |
| DATA-004 | Address B008 warnings   | LOW      | Small  | Code quality |
| DATA-005 | Address SIM117 in tests | LOW      | Small  | Code quality |
| DATA-018 | IPC architecture        | LOW      | Large  | Performance  |
| DATA-019 | GPU acceleration        | LOW      | Large  | Performance  |
| DATA-020 | Continuous profiling    | LOW      | Medium | Operations   |

### Completed Items

| ID       | Item                         | Completed  |
| -------- | ---------------------------- | ---------- |
| DATA-001 | Fix mypy type errors         | 2026-02-05 |
| DATA-002 | Fix unused imports           | 2026-02-05 |
| DATA-003 | Fix golden dataset issues    | 2026-02-05 |
| DATA-006 | Create Dockerfile            | 2026-02-05 |
| DATA-007 | Health check probes          | 2026-02-05 |
| DATA-008 | E2E integration tests        | 2026-02-05 |
| DATA-009 | API versioning docs          | 2026-02-05 |
| DATA-010 | NPZ schema docs              | 2026-02-05 |
| DATA-011 | Parameter validation parity  | 2026-02-05 |
| DATA-012 | Client package consolidation | 2026-02-06 |
| DATA-013 | Client test coverage         | 2026-02-06 |
| DATA-016 | Dataset lifecycle management | 2026-02-06 |
| DATA-017 | API rate limiting/auth       | 2026-02-06 |
|          |                              |            |

---

## Summary Statistics

| Category                   | Count                                                                        |
| -------------------------- | ---------------------------------------------------------------------------- |
| Total Items                | 20                                                                           |
| COMPLETE                   | 13 (DATA-001, 002, 003, 006, 007, 008, 009, 010, 011, 012, 013, 016, 017)    |
| IN PROGRESS                | 2 (DATA-014, 015) - code exists but missing tests/registration/coverage      |
| NOT STARTED (Low Priority) | 2 (DATA-004, 005)                                                            |
| DEFERRED                   | 3 (DATA-018, 019, 020)                                                       |
| Cross-Project References   | 10 (CAS: 5, CAN: 5)                                                          |
| **Coverage Status**        | **60.57% total (FAILS 80% threshold)** - 623 untested lines in 6 new modules |

---

## Document History

| Date       | Author      | Changes                                                                                               |
| ---------- | ----------- | ----------------------------------------------------------------------------------------------------- |
| 2026-02-05 | Paul Calnon | Initial creation from documentation review and source code analysis                                   |
| 2026-02-05 | AI Agent    | Completed DATA-001, DATA-002, DATA-003 - All mypy errors fixed, flake8 F401/E402/F541 issues resolved |
| 2026-02-05 | AI Agent    | Completed DATA-006, 007, 008 - Dockerfile, health probes, E2E tests                                   |
| 2026-02-05 | AI Agent    | Completed DATA-009, 010, 011 - API docs, NPZ schema docs, parameter aliases                           |
| 2026-02-06 | AI Agent    | Completed DATA-012, 013 - Created juniper-data-client package with 35 tests (96% coverage)            |
| 2026-02-06 | AI Agent    | Completed DATA-016 - Dataset lifecycle management (TTL, filtering, batch delete, tags, stats)         |
| 2026-02-06 | AI Agent    | DATA-014 IN PROGRESS - Added XOR generator with 18 tests                                              |
| 2026-02-06 | AI Agent    | Completed DATA-014 - Added Gaussian blobs (26 tests) and Concentric circles (22 tests) generators     |
| 2026-02-06 | AI Agent    | Completed DATA-017 - API security (API key auth, rate limiting middleware, 31 tests)                  |
| 2026-02-06 | AI Agent    | DATA-015 IN PROGRESS - Added CachedDatasetStore, RedisDatasetStore, HuggingFaceDatasetStore           |
| 2026-02-07 | AI Agent    | Completed DATA-014 - Added Checkerboard, CSV/JSON import, MNIST, ARC-AGI generators                   |
| 2026-02-07 | AI Agent    | Completed DATA-015 - Added PostgresDatasetStore and KaggleDatasetStore                                |
| 2026-02-07 | AI Agent    | **Validation audit**: DATA-014 reverted to IN PROGRESS (3 generators not registered, 2 have 0% tests) |
| 2026-02-07 | AI Agent    | **Validation audit**: DATA-015 reverted to IN PROGRESS (4 backends at 0% coverage, no tests)          |
| 2026-02-07 | AI Agent    | Updated Current State Summary with actual metrics: 60.57% coverage, 16 flake8 issues, coverage gaps   |
| 2026-02-07 | AI Agent    | Fixed stale cross-project references (DATA-006/008/012 were COMPLETE, not NOT STARTED)                |
| 2026-02-07 | AI Agent    | Removed DATA-014 from Deferred Items table (it's IN PROGRESS, not deferred)                           |
