# Pull Request: JuniperData Service Extraction & Integration Infrastructure

**Date:** 2026-02-06
**Version(s):** 0.1.0 → 0.4.0
**Author:** Paul Calnon
**Status:** READY_FOR_MERGE

---

## Summary

This PR extracts JuniperData as a standalone dataset generation microservice from the Juniper monorepo, replacing the legacy JuniperCanopy codebase with a purpose-built FastAPI service. The branch delivers the complete JuniperData application (v0.1.0–v0.4.0), including 8 dataset generators, 7 storage backends, REST API with 16 endpoints, Docker containerization, a shared client library, dataset lifecycle management, comprehensive CI/CD pipeline with security scanning, and 699 tests across service and client.

**SemVer Impact:** MINOR (cumulative across v0.1.0–v0.4.0)
**Breaking Changes:** YES — Complete repository transformation from JuniperCanopy to JuniperData

---

## Context / Motivation

JuniperData was designed as a standalone microservice to provide dataset generation capabilities to the Juniper ecosystem. Previously, dataset generation was embedded within JuniperCascor's `SpiralProblem` class. This extraction enables:

- **Independent deployment** — JuniperData runs as a containerized service on port 8100
- **Shared data contract** — NPZ artifacts with standardized keys (`X_train`, `y_train`, `X_test`, `y_test`, `X_full`, `y_full`)
- **Multi-consumer architecture** — Both JuniperCascor and JuniperCanopy consume datasets via REST API
- **Feature flag integration** — `JUNIPER_DATA_URL` environment variable enables JuniperData mode in consumers
- **Extensible generator framework** — Plugin architecture supports adding new dataset types

**Related Documentation:**

- [INTEGRATION_DEVELOPMENT_PLAN.md](../INTEGRATION_DEVELOPMENT_PLAN.md) — 20 work items, 12 complete
- [JUNIPER_DATA_API.md](../../docs/api/JUNIPER_DATA_API.md) — Comprehensive API reference
- [JUNIPER_CASCOR_SPIRAL_DATA_GEN_REFACTOR_PLAN.md](../../notes/INTEGRATION_ROADMAP.md) — Original refactoring plan (Phases 0-4)

---

## Priority & Work Status

| Priority | Work Item | ID | Status |
| -------- | --------- | -- | ------ |
| P0 | Fix mypy type errors in test files (20 errors → 0) | DATA-001 | ✅ Complete |
| P0 | Fix flake8 unused imports in datasets.py | DATA-002 | ✅ Complete |
| P1 | Fix flake8 issues in generate_golden_datasets.py | DATA-003 | ✅ Complete |
| P1 | Create Dockerfile for JuniperData service | DATA-006 | ✅ Complete |
| P1 | Add health check probes for container orchestration | DATA-007 | ✅ Complete |
| P1 | End-to-end integration tests | DATA-008 | ✅ Complete |
| P2 | API versioning strategy documentation | DATA-009 | ✅ Complete |
| P2 | NPZ artifact schema documentation | DATA-010 | ✅ Complete |
| P2 | Parameter validation parity with consumers | DATA-011 | ✅ Complete |
| P2 | Shared JuniperData client package | DATA-012 | ✅ Complete |
| P2 | Client test coverage | DATA-013 | ✅ Complete |
| P2 | XOR generator (additional generator types) | DATA-014 | ✅ Partial |
| P2 | Dataset lifecycle management | DATA-016 | ✅ Complete |
| P3 | B008 warnings in API route defaults | DATA-004 | Not Started (intentional FastAPI patterns) |
| P3 | SIM117 suggestions in test files | DATA-005 | Not Started (handled by relaxed rules) |
| P3 | Storage backend extensions | DATA-015 | Not Started |
| P3 | API rate limiting and authentication | DATA-017 | Not Started |
| P3 | IPC architecture | DATA-018 | Deferred |
| P3 | GPU-accelerated data generation | DATA-019 | Deferred |
| P3 | Continuous profiling integration | DATA-020 | Deferred |

### Priority Legend

- **P0:** Critical — Core bugs or blockers
- **P1:** High — High-impact features or fixes
- **P2:** Medium — Polish and medium-priority
- **P3:** Low — Advanced/infrastructure features

---

## Changes

### Added

- **Core Application** (`juniper_data/`)
  - `__init__.py` — Package init with version (v0.3.0)
  - `__main__.py` — CLI entry point with argparse (host, port, storage-path, log-level, reload)
  - `core/models.py` — 11 Pydantic v2 models (DatasetMeta, CreateDatasetRequest/Response, GeneratorInfo, PreviewData, DatasetListFilter, BatchDeleteRequest/Response, UpdateTagsRequest, DatasetStats)
  - `core/split.py` — Dataset shuffle and split utilities (`shuffle_data()`, `split_data()`, `shuffle_and_split()`)
  - `core/dataset_id.py` — Deterministic SHA-256 hash-based dataset ID generation
  - `core/artifacts.py` — NPZ artifact handling with checksums

- **Generator Framework** (`juniper_data/generators/`) — 8 generators registered in `GENERATOR_REGISTRY`
  - `spiral/` — N-spiral classification dataset generator (modern + legacy_cascor algorithms)
  - `xor/` — XOR classification dataset generator (4-quadrant diagonal class assignment)
  - `gaussian/` — Gaussian blobs classification dataset generator (mixture-of-Gaussians)
  - `circles/` — Concentric circles classification dataset generator (binary inner/outer)
  - `checkerboard/` — Checkerboard pattern classification dataset generator (2D alternating grid)
  - `csv_import/` — CSV/JSON import generator for custom datasets
  - `mnist/` — MNIST and Fashion-MNIST dataset generator
  - `arc_agi/` — ARC-AGI visual reasoning tasks generator (requires `arc-agi>=0.9.0`)

- **Storage Layer** (`juniper_data/storage/`) — 7 backend implementations
  - `base.py` — Abstract `DatasetStore` with lifecycle methods
  - `memory.py` — `InMemoryDatasetStore` for testing and ephemeral use
  - `local_fs.py` — `LocalFSDatasetStore` for file-based persistence (JSON metadata + NPZ artifacts)
  - `cached.py` — `CachedDatasetStore` caching layer with error logging
  - `hf_store.py` — `HuggingFaceDatasetStore` for HuggingFace Hub storage
  - `kaggle_store.py` — `KaggleDatasetStore` for Kaggle Datasets storage
  - `postgres_store.py` — `PostgresDatasetStore` for PostgreSQL storage
  - `redis_store.py` — `RedisDatasetStore` for Redis storage

- **REST API** (`juniper_data/api/`)
  - `app.py` — FastAPI application factory with lifespan management, CORS middleware, exception handlers
  - `settings.py` — Pydantic-settings configuration with `JUNIPER_DATA_` env prefix
  - `routes/health.py` — Health endpoints: `GET /v1/health`, `GET /v1/health/live`, `GET /v1/health/ready`
  - `routes/generators.py` — Generator discovery: `GET /v1/generators`, `GET /v1/generators/{name}/schema`
  - `routes/datasets.py` — 11 dataset endpoints:
    - `POST /v1/datasets` — Create dataset (deterministic caching)
    - `GET /v1/datasets` — List with pagination (limit/offset)
    - `GET /v1/datasets/filter` — Advanced filtering (generator, tags, date range, sample count)
    - `GET /v1/datasets/stats` — Aggregate statistics
    - `POST /v1/datasets/batch-delete` — Bulk delete (up to 100 IDs)
    - `POST /v1/datasets/cleanup-expired` — TTL-based cleanup
    - `GET /v1/datasets/{id}` — Metadata retrieval
    - `GET /v1/datasets/{id}/artifact` — NPZ download (StreamingResponse)
    - `GET /v1/datasets/{id}/preview` — JSON sample preview
    - `DELETE /v1/datasets/{id}` — Single dataset deletion
    - `PATCH /v1/datasets/{id}/tags` — Tag management (add/remove)

- **Shared Client Library** (`juniper_data_client/`)
  - `client.py` — `JuniperDataClient` with all API methods, automatic retry logic, connection pooling
  - `exceptions.py` — Exception hierarchy (ConnectionError, TimeoutError, NotFoundError, ValidationError)
  - `pyproject.toml` — Standalone pip-installable package configuration
  - `py.typed` — PEP 561 type marker for mypy strict mode
  - `README.md` — Comprehensive documentation with usage examples

- **Docker Containerization**
  - `Dockerfile` — Multi-stage build (python:3.11-slim), non-root user (juniper:1000), port 8100
  - `.dockerignore` — Build context exclusions (tests, docs, notes, caches)
  - `HEALTHCHECK` instruction (30s interval, 10s timeout, 5s start period, 3 retries)

- **CI/CD Pipeline**
  - `.github/workflows/ci.yml` — Full pipeline: pre-commit (Python 3.12-3.14), unit tests (80% coverage gate), integration tests, security scanning (Gitleaks, Bandit SARIF, pip-audit), slow tests (weekly), build verification
  - `.github/workflows/codeql.yml` — Weekly semantic code analysis
  - `.github/dependabot.yml` — Automated dependency updates (pip + GitHub Actions)
  - `.pre-commit-config.yaml` — 16+ hooks: black, isort, flake8 (with bugbear, comprehensions, simplify), mypy, bandit, pyupgrade (py311+), shellcheck, YAML/TOML/JSON validation
  - GitHub Actions pinned to SHA for supply chain security

- **Test Suite** (658 tests + 41 client tests)
  - `tests/unit/` — 589 unit tests across 25 test files
  - `tests/integration/` — 69 integration tests across 5 test files
  - `tests/fixtures/` — Golden dataset fixtures (2-spiral, 3-spiral) for parity testing
  - `tests/conftest.py` — Shared fixtures (TestClient, storage, sample datasets)
  - `juniper_data_client/tests/test_client.py` — 41 client tests using `responses` HTTP mocking

- **Documentation**
  - `docs/api/JUNIPER_DATA_API.md` — 622-line API reference with versioning strategy, endpoint documentation, NPZ schema, client examples (Python, curl, Docker Compose)
  - `notes/INTEGRATION_DEVELOPMENT_PLAN.md` — 20 work items compiled from 5 documentation sources
  - `notes/test_suite_audit/` — Test suite audits and CI/CD enhancement plans
  - `notes/mcp_tools/` — MCP/Serena setup documentation
  - `AGENTS.md` — Complete project guide (updated from JuniperCanopy)
  - `CLAUDE.md` — Symlink to AGENTS.md for Claude Code integration

- **Infrastructure**
  - `conf/juniper_data.conf` — Application launch configuration (renamed from juniper_canopy.conf)
  - `conf/juniper_data_functions.conf` — Shell functions (renamed from juniper_canopy_functions.conf)
  - `conf/conda_environment.yaml` — Updated Conda environment for JuniperData
  - `conf/script_util.cfg` — Script utility configuration
  - `conf/logging_config-CASCOR.yaml` — Cascor logging configuration
  - `util/juniper_data.bash` — Application launch script
  - `pyproject.toml` — Complete project configuration (dependencies, tool settings, test markers)

### Changed

- **Repository Identity** — Transformed from JuniperCanopy (Dash visualization dashboard) to JuniperData (dataset generation microservice)
- **Directory Structure** — Replaced `src/` package structure with `juniper_data/` for proper package discovery
- **README.md** — Rewritten for JuniperData service documentation
- **AGENTS.md** — Rewritten from JuniperCanopy guide to JuniperData project guide
- **CHANGELOG.md** — Rewritten with JuniperData version history (v0.1.0–v0.4.0)
- **pyproject.toml** — Reconfigured for JuniperData (dependencies, tool config, test markers, coverage settings)
- **CI/CD Pipeline** — Rebuilt from scratch for JuniperData (simplified, focused, security-hardened)
- **Pre-commit Configuration** — Updated with JuniperData-specific settings and additional hooks
- **Shell Scripts** — Retained and updated utility scripts; removed Canopy-specific scripts
- **Conda Environment** — Updated for JuniperData development dependencies

### Fixed

- **DATA-001: mypy Type Errors in Test Files** — 20 errors → 0 across 4 test files
  - Type narrowing assertions in test_storage.py and test_storage_workflow.py
  - `# type: ignore[arg-type]` for negative tests in test_spiral_generator.py
  - `getattr()` pattern for dynamic attributes in test_api_app.py
- **DATA-002: flake8 Unused Imports** — Removed unused `Any` and `Dict` from datasets.py
- **DATA-003: flake8 Issues in generate_golden_datasets.py** — Added `# noqa: E402` for late imports, converted bare f-strings
- **CI-001: pip-audit failing on local package** — Fixed grep pattern in CI pipeline to handle modern pip's underscore-normalized package names (`juniper_data` vs `juniper-data`), which caused `pip-audit --strict` to fail with "Dependency not found on PyPI"
- **MNIST-001: 12 failing MNIST generator tests** — Generator's `_load_and_preprocess` called `ds.with_format("numpy")` for bulk column access, but test mocks didn't configure `with_format()`, returning a generic `MagicMock` that produced empty arrays on `np.array()`. Fixed by adding `formatted_ds` mocks returning proper numpy data. Also added missing `n_samples` support via `ds.select()` in the generator.
- **SEC-007: Bandit B615 nosec placement** — Moved `# nosec B615` from the comment line above `hf_load_dataset()` to inline on the call itself; bandit only honors `# nosec` directives on the same line as the flagged code

### Removed

- **JuniperCanopy Application** (`src/`) — Entire Dash dashboard application (185 files)
  - `src/backend/` — Cascor integration, Redis, Cassandra, statistics, training monitor/state machine
  - `src/frontend/` — Dashboard manager, all UI components (metrics, network visualizer, dataset plotter, etc.)
  - `src/communication/` — WebSocket manager
  - `src/demo_mode.py`, `src/config_manager.py`, `src/main.py` — Application entry points
  - `src/tests/` — All JuniperCanopy tests (~2,900 tests)
  - `src/assets/` — Logo images and icons
  - `src/logger/` — Custom logging module
- **Legacy Configuration** — `.codecov.yml`, `.coveragerc`, `.markdownlint.json`, `.yamllint.yaml`, `conftest.py` (root)
- **Legacy Documentation** — Development phases (phase0-3), fix notes, release notes (v0.14–v0.25), analysis reports, roadmaps
- **Legacy Infrastructure** — Canopy-specific conf/ files (docker-compose, app_config, main.conf, proto.conf), requirements.txt files, conda CI environment, setup_environment scripts, performance profiling, Canopy utility scripts
- **Legacy Artifacts** — `snapshots/`, `demo` symlink, `src/logs` symlink

### Security

- **SEC-001: Bandit Security Scanning** — Blocking checks for medium+ severity findings
- **SEC-002: pip-audit Strict Mode** — Fails on any known vulnerability
- **SEC-003: Dependabot Configuration** — Automated weekly dependency updates (pip + GitHub Actions)
- **SEC-004: GitHub Actions Pinned to SHA** — Supply chain security for all Actions
- **SEC-005: Non-root Docker User** — Container runs as `juniper` (UID 1000)
- **SEC-006: CodeQL Analysis** — Weekly semantic code analysis workflow

---

## Impact & SemVer

- **SemVer impact:** MINOR (0.3.0 → 0.4.0) — New features added; v0.1.0–v0.4.0 cumulative
- **User-visible behavior change:** YES — Complete application replacement
- **Breaking changes:** YES
  - **What breaks:** The entire JuniperCanopy application (`src/`) is removed and replaced with JuniperData (`juniper_data/`)
  - **Migration steps:** This is a planned extraction; JuniperCanopy continues independently in its own repository
- **Performance impact:** N/A — New application, not comparable to predecessor
- **Security/privacy impact:** IMPROVED — Non-root Docker, Bandit SAST, pip-audit, CodeQL, Dependabot, SHA-pinned Actions
- **Guarded by feature flag:** YES — `JUNIPER_DATA_URL` environment variable enables JuniperData mode in consumers

---

## Testing & Results

### Test Summary

| Test Type | Passed | Failed | Skipped | Notes |
| --------- | ------ | ------ | ------- | ----- |
| Unit | 589 | 0 | 0 | 25 test files covering all source modules |
| Integration | 69 | 0 | 0 | 5 test files (API, E2E, lifecycle, storage) |
| Client | 41 | 0 | 0 | juniper-data-client package (96% coverage) |
| Total | 699 | 0 | 0 | All passing on Python 3.14.2 |

### Coverage

| Component | Coverage | Target | Status |
| --------- | -------- | ------ | ------ |
| juniper_data (overall) | 97.47% | 80% | ✅ Exceeded |
| 24 of 26 source files | 100% | 80% | ✅ Exceeded |
| storage/base.py | 99.10% | 80% | ✅ Exceeded |
| storage/local_fs.py | 79.57% | 80% | ⚠️ Near |
| juniper-data-client | 96% | N/A | ✅ Exceeded |

### Environments Tested

- Python 3.14.2 (local development): ✅ All 699 tests pass
- CI matrix: Python 3.12, 3.13, 3.14 (configured)
- Conda environment (JuniperData): ✅ Functional

---

## Verification Checklist

- [x] Main user flow(s) verified: Dataset create → download → verify cycle (E2E tests)
- [x] Edge cases checked: Invalid generators, missing datasets, expired TTL, batch operations
- [x] No regression in related areas: JuniperCascor/JuniperCanopy maintain independent codebases
- [x] Feature defaults correct and documented: API defaults, generator defaults, storage defaults
- [x] Logging/metrics updated: Structured logging via uvicorn, configurable log levels
- [x] Documentation updated: AGENTS.md, API docs, CHANGELOG.md, INTEGRATION_DEVELOPMENT_PLAN.md
- [x] Legacy parity verified: `algorithm="legacy_cascor"` matches JuniperCascor SpiralProblem statistics
- [x] Parameter aliases verified: `n_points`/`noise_level` accepted for consumer compatibility
- [x] Docker configuration verified: Multi-stage build, health checks, non-root user
- [x] CI/CD pipeline verified: All hooks pass, security scanning configured

---

## API Changes

### New Endpoints (All New)

| Method | Endpoint | Description | Breaking? |
| ------ | -------- | ----------- | --------- |
| GET | `/v1/health` | Health check (backward compatible) | No |
| GET | `/v1/health/live` | Kubernetes liveness probe | No |
| GET | `/v1/health/ready` | Kubernetes readiness probe | No |
| GET | `/v1/generators` | List available generators | No |
| GET | `/v1/generators/{name}/schema` | Get generator parameter schema | No |
| POST | `/v1/datasets` | Create/generate dataset | No |
| GET | `/v1/datasets` | List datasets with pagination | No |
| GET | `/v1/datasets/filter` | Advanced dataset filtering | No |
| GET | `/v1/datasets/stats` | Aggregate statistics | No |
| POST | `/v1/datasets/batch-delete` | Bulk delete (up to 100) | No |
| POST | `/v1/datasets/cleanup-expired` | Remove expired datasets | No |
| GET | `/v1/datasets/{id}` | Get dataset metadata | No |
| GET | `/v1/datasets/{id}/artifact` | Download NPZ artifact | No |
| GET | `/v1/datasets/{id}/preview` | JSON sample preview | No |
| DELETE | `/v1/datasets/{id}` | Delete dataset | No |
| PATCH | `/v1/datasets/{id}/tags` | Add/remove dataset tags | No |

### NPZ Artifact Schema (Data Contract)

| Key | Shape | Dtype | Description |
| --- | ----- | ----- | ----------- |
| `X_train` | `(n_train, n_features)` | `float32` | Training features |
| `y_train` | `(n_train, n_classes)` | `float32` | Training labels (one-hot) |
| `X_test` | `(n_test, n_features)` | `float32` | Test features |
| `y_test` | `(n_test, n_classes)` | `float32` | Test labels (one-hot) |
| `X_full` | `(n_samples, n_features)` | `float32` | Full dataset features |
| `y_full` | `(n_samples, n_classes)` | `float32` | Full dataset labels (one-hot) |

---

## Files Changed

**Total: 451 files changed, 19,150 insertions(+), 94,825 deletions(-):**

| Category | Count |
| -------- | ----- |
| New files | 128 |
| Modified files | 43 |
| Deleted files | 274 |

### New Components

**Core Application:**

- `juniper_data/__init__.py` — Package init with version
- `juniper_data/__main__.py` — CLI entry point (argparse + uvicorn)
- `juniper_data/core/models.py` — 11 Pydantic data models
- `juniper_data/core/split.py` — Shuffle and split utilities
- `juniper_data/core/dataset_id.py` — Deterministic ID generation
- `juniper_data/core/artifacts.py` — NPZ artifact handling

**Generators:**

- `juniper_data/generators/spiral/generator.py` — Spiral dataset generator (modern + legacy algorithms)
- `juniper_data/generators/spiral/params.py` — Spiral params with consumer aliases
- `juniper_data/generators/spiral/defaults.py` — Default constants
- `juniper_data/generators/xor/generator.py` — XOR classification generator
- `juniper_data/generators/xor/params.py` — XOR parameter model

**Storage:**

- `juniper_data/storage/base.py` — Abstract store with lifecycle methods
- `juniper_data/storage/memory.py` — In-memory store
- `juniper_data/storage/local_fs.py` — Filesystem store

**API:**

- `juniper_data/api/app.py` — FastAPI application factory
- `juniper_data/api/settings.py` — Pydantic-settings configuration
- `juniper_data/api/routes/health.py` — Health probe endpoints
- `juniper_data/api/routes/generators.py` — Generator discovery endpoints
- `juniper_data/api/routes/datasets.py` — Dataset CRUD + lifecycle endpoints

**Client Library:**

- `juniper_data_client/client.py` — REST client with retries and connection pooling
- `juniper_data_client/exceptions.py` — Exception hierarchy
- `juniper_data_client/pyproject.toml` — Package configuration

**Infrastructure:**

- `Dockerfile` — Multi-stage Docker build
- `.dockerignore` — Build context exclusions
- `.github/workflows/ci.yml` — CI/CD pipeline
- `.github/workflows/codeql.yml` — CodeQL analysis
- `.github/dependabot.yml` — Dependency automation
- `conf/juniper_data.conf` — Application launch config
- `util/juniper_data.bash` — Launch script

**Tests:**

- `juniper_data/tests/unit/test_spiral_generator.py` — 43 spiral generator tests
- `juniper_data/tests/unit/test_split.py` — 18 data split tests
- `juniper_data/tests/unit/test_dataset_id.py` — 16 ID generation tests
- `juniper_data/tests/unit/test_storage.py` — 44 storage tests
- `juniper_data/tests/unit/test_artifacts.py` — 11 artifact tests
- `juniper_data/tests/unit/test_api_app.py` — 17 app factory tests
- `juniper_data/tests/unit/test_api_routes.py` — 18 route tests
- `juniper_data/tests/unit/test_api_settings.py` — 11 settings tests
- `juniper_data/tests/unit/test_main.py` — 10 entry point tests
- `juniper_data/tests/unit/test_lifecycle.py` — 27 lifecycle tests
- `juniper_data/tests/unit/test_xor_generator.py` — 18 XOR generator tests
- `juniper_data/tests/integration/test_api.py` — 18 API integration tests
- `juniper_data/tests/integration/test_e2e_workflow.py` — 14 E2E workflow tests
- `juniper_data/tests/integration/test_lifecycle_api.py` — 17 lifecycle API tests
- `juniper_data/tests/integration/test_storage_workflow.py` — 8 storage workflow tests
- `juniper_data_client/tests/test_client.py` — 35 client tests

**Documentation:**

- `docs/api/JUNIPER_DATA_API.md` — 622-line API reference
- `notes/INTEGRATION_DEVELOPMENT_PLAN.md` — Integration work items
- `notes/test_suite_audit/*.md` — Test suite audit reports
- `AGENTS.md` — Updated project guide

### Modified Components

**Configuration:**

- `pyproject.toml` — Reconfigured for JuniperData (dependencies, tools, markers)
- `.pre-commit-config.yaml` — Updated hooks and settings
- `.gitignore` — Updated for JuniperData structure
- `conf/conda_environment.yaml` — Updated for JuniperData dependencies
- `conf/*.conf` — Updated shell configuration files

**Reports:**

- `reports/junit/results.xml` — Updated test results

### Removed Components

**JuniperCanopy Application (185 `src/` files):**

- `src/backend/` — cascor_integration.py, data_adapter.py, redis_client.py, cassandra_client.py, statistics.py, training_monitor.py, training_state_machine.py
- `src/frontend/` — dashboard_manager.py, 10 component files (metrics_panel, network_visualizer, dataset_plotter, decision_boundary, about_panel, etc.)
- `src/communication/` — websocket_manager.py
- `src/main.py` — 2,030-line application entry point
- `src/demo_mode.py` — 1,004-line demo mode
- `src/config_manager.py` — 492-line configuration manager
- `src/tests/` — ~2,900 tests across unit, integration, regression, and performance suites
- `src/assets/` — 15 logo/icon files

**Legacy Infrastructure (86 files):**

- `conf/docker-compose.yaml`, `conf/Dockerfile`, `conf/app_config.yaml`
- `conf/requirements.txt`, `conf/requirements_ci.txt`
- `conf/setup_environment.conf`, `conf/setup_environment_functions.conf`
- 30+ utility scripts (util/*.bash)
- Legacy notes/development/, notes/fixes/, notes/releases/ documentation

---

## Risks & Rollback Plan

- **Key risks:**
  - Repository transformation is non-reversible in the traditional sense (old JuniperCanopy code removed)
  - Consumer integration requires updating JuniperCascor and JuniperCanopy to use the shared client package
- **Monitoring / alerts to watch:**
  - JuniperData service health endpoint: `GET /v1/health`
  - CI/CD pipeline status on main branch
  - Dependency vulnerability alerts via Dependabot
- **Rollback plan:**
  - Git revert to main branch restores JuniperCanopy codebase
  - JuniperCanopy application continues independently in its own repository
  - Consumer feature flags (`JUNIPER_DATA_URL`) allow graceful degradation when JuniperData is unavailable

---

## Related Issues / Tickets

- Issues: DATA-001 through DATA-020 (see [INTEGRATION_DEVELOPMENT_PLAN.md](../INTEGRATION_DEVELOPMENT_PLAN.md))
- Design / Spec: [JUNIPER_CASCOR_SPIRAL_DATA_GEN_REFACTOR_PLAN.md](../../notes/INTEGRATION_ROADMAP.md) (Phases 0-4)
- Related PRs: PR_DESCRIPTION_RELEASE_v0.25.0_2026-01-25.md (last JuniperCanopy release)
- Phase Documentation: All Phase 0-4 complete; Phase 5 (Extended Data Sources) in progress

---

## What's Next

### Remaining Items

| Feature | Status | Priority |
| ------- | ------ | -------- |
| Publish juniper-data-client to PyPI | Not Started | P1 |
| Update JuniperCascor to use shared client package | Not Started | P1 |
| Update JuniperCanopy to use shared client package | Not Started | P1 |
| Additional generators (Gaussian mixture, circles, checkerboard) | ✅ Complete  | P3 |
| Storage backend extensions (cached, HF, Kaggle, Postgres, Redis) | ✅ Complete  | P3 |
| API rate limiting and authentication | Not Started | P3 |
| IPC architecture (gRPC) | Deferred | P3 |
| GPU-accelerated data generation | Deferred | P3 |
| Continuous profiling integration | Not Started | P3 |

---

## Notes for Release

**Release: JuniperData v0.4.0 — Integration Infrastructure & Client Library:**

Key highlights:

- Complete standalone dataset generation microservice
- FastAPI REST API with 16 endpoints on port 8100
- Two generators: spiral (modern + legacy algorithms) and XOR classification
- Dataset lifecycle management (TTL, tagging, filtering, batch operations)
- Shared client library (juniper-data-client) for JuniperCascor/JuniperCanopy integration
- Docker containerization with multi-stage build and health probes
- 699 tests (658 service + 41 client)
- CI/CD pipeline with Python 3.11-3.14, security scanning, pre-commit hooks
- Comprehensive API documentation with NPZ artifact schema

---

## Review Notes

1. **Repository Transformation:** This PR replaces the entire JuniperCanopy codebase with JuniperData. The `src/` directory (185 files) is removed and `juniper_data/` (82 new files) takes its place. This is a planned extraction, not an accidental deletion.

2. **Version History:** The CHANGELOG documents the complete development history from v0.1.0 (initial release) through v0.4.0 (this PR). Each version built incrementally on the previous one.

3. **Client Library:** The `juniper_data_client/` package is designed to be published to PyPI and consumed by both JuniperCascor and JuniperCanopy, replacing their current duplicate client code.

4. **Legacy Parity:** The `algorithm="legacy_cascor"` mode in SpiralGenerator reproduces the statistical properties of JuniperCascor's original SpiralProblem implementation, ensuring backward compatibility.

5. **New Modules:** 6 new generators and 5 new storage backends added after the initial PR description was written.

6. **B008 Warnings:** The 9 B008 flake8 warnings in `datasets.py` are intentional FastAPI patterns (using `Query()`, `Depends()` in function defaults) and should not be "fixed."

7. **Test Count Growth:** From 0 tests (main branch had JuniperCanopy tests) to 699 JuniperData-specific tests (658 service + 41 client).
