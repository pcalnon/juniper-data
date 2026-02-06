# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-02-05

**Summary**: Integration Infrastructure - Docker containerization, health probes, and E2E testing for ecosystem integration.

### Added: [Unreleased]

- **Integration Development Plan** (`notes/INTEGRATION_DEVELOPMENT_PLAN.md`)
  - Compiled 20 outstanding work items from 4 documentation files and source code analysis
  - 6 HIGH priority items now COMPLETE (mypy fixes, unused imports, Dockerfile, health checks, E2E tests)
  - 5 MEDIUM priority items remaining (API docs, parameter validation, client consolidation)
  - 6 LOW priority items (generators, storage, lifecycle, auth)
  - 3 DEFERRED items (IPC, GPU, profiling)
  - 10 cross-project references (JuniperCascor: 5, JuniperCanopy: 5)

- **DATA-006: Dockerfile for JuniperData Service**
  - Multi-stage build (builder + runtime) using `python:3.11-slim`
  - Installs with `pip install .[api]` for minimal dependencies
  - Non-root `juniper` user (UID 1000) for security
  - Exposes port 8100 with environment variable configuration
  - `.dockerignore` to exclude tests, docs, notes, and development files

- **DATA-007: Health Check Probes for Container Orchestration**
  - `HEALTHCHECK` instruction in Dockerfile (30s interval, 10s timeout, 5s start period, 3 retries)
  - `GET /v1/health/live` - Liveness probe (returns `{"status": "alive"}`)
  - `GET /v1/health/ready` - Readiness probe (returns `{"status": "ready", "version": "..."}`)
  - Original `/v1/health` endpoint preserved for backward compatibility

- **DATA-008: End-to-End Integration Tests**
  - `juniper_data/tests/integration/test_e2e_workflow.py` with 14 comprehensive E2E tests
  - **TestE2EModernAlgorithm**: create/download/verify flow, determinism, seed variation
  - **TestE2ELegacyCascorAlgorithm**: legacy algorithm flow, legacy vs modern comparison
  - **TestE2EDataContract**: NPZ keys, feature dimensions, one-hot labels, split ratios, metadata
  - **TestE2EErrorHandling**: invalid generator, invalid params, nonexistent dataset, delete verification
  - All tests marked with `@pytest.mark.integration` and `@pytest.mark.slow`

- **DATA-009: API Versioning Strategy Documentation**
  - Created `docs/api/JUNIPER_DATA_API.md` with comprehensive API reference
  - Documents versioning policy following SemVer principles
  - Specifies backward compatibility guarantees and deprecation policy

- **DATA-010: NPZ Artifact Schema Documentation**
  - Added dedicated "NPZ Artifact Schema" section in API documentation
  - Documents all 6 required array keys with shapes and dtypes
  - Includes Python and PyTorch loading examples

- **DATA-011: Parameter Validation Parity with Consumers**
  - Added parameter aliases using Pydantic `AliasChoices` in `SpiralParams`
  - `n_points` accepted as alias for `n_points_per_spiral`
  - `noise_level` accepted as alias for `noise`
  - Added 5 new unit tests verifying alias behavior

### Changed: [Unreleased]

- **CLAUDE.md** updated with Integration Context section
  - Added integration points documentation (port, feature flag, data contract, consumers)
  - Added key documentation reference table

### Fixed: [Unreleased]

- **DATA-001: mypy Type Errors in Test Files** (20 errors → 0)
  - Added type narrowing assertions (`assert x is not None`) in test_storage.py and test_storage_workflow.py
  - Added `# type: ignore[arg-type]` with explanation for negative test in test_spiral_generator.py
  - Used `getattr()` pattern for dynamic route/middleware attribute access in test_api_app.py

- **DATA-002: flake8 Unused Imports in datasets.py**
  - Removed unused `Any` and `Dict` imports from `typing` module

- **DATA-003: flake8 Issues in generate_golden_datasets.py**
  - Added `# noqa: E402` comments for late imports after `sys.path` manipulation
  - Converted f-strings without placeholders to regular strings (5 instances)

### Technical Notes: [Unreleased]

- **SemVer impact**: MINOR - New Docker infrastructure, API endpoints, and parameter aliases; backward compatible
- **Source analysis findings**: 0 mypy errors (was 20), ~9 flake8 issues (all B008 - intentional FastAPI patterns)
- **Test count**: 228 tests (up from 207, all passing)
- **Coverage**: 100% maintained
- **New files**: `Dockerfile`, `.dockerignore`, `juniper_data/tests/integration/test_e2e_workflow.py`, `docs/api/JUNIPER_DATA_API.md`
- **Modified**: `juniper_data/generators/spiral/params.py` (parameter aliases), `juniper_data/api/routes/health.py` (liveness/readiness probes)

---

## [0.3.0] - 2026-02-04

**Summary**: Comprehensive Test Suite and CI/CD Enhancement - Security hardening, static analysis expansion, infrastructure improvements.

### Security: [0.3.0]

- **SEC-001: Bandit Security Scanning Now Blocking**
  - Replaced `|| true` with `--exit-zero` for SARIF generation
  - Added blocking check for medium+ severity findings

- **SEC-002: pip-audit Now Strict Mode**
  - Changed from warning-only to `--strict` flag to fail on vulnerabilities

- **SEC-003: Dependabot Configuration**
  - New `.github/dependabot.yml` for automated dependency updates
  - Configured for both pip and GitHub Actions ecosystems
  - Weekly schedule with grouped updates

- **SEC-004: GitHub Actions Pinned to SHA**
  - All GitHub Actions now pinned to specific commit SHAs for supply chain security
  - Includes: checkout, setup-python, cache, upload-artifact, codecov, codeql, gitleaks

### Added: [0.3.0]

- **CodeQL Analysis Workflow** (`.github/workflows/codeql.yml`)
  - Weekly semantic code analysis for security vulnerabilities
  - Runs on push to main/develop and on PRs

- **Codecov Integration**
  - Coverage reports now uploaded to Codecov for trend tracking
  - Added `codecov-action` step in unit-tests job

- **Slow Test Job**
  - New `slow-tests` job for tests marked with `@pytest.mark.slow`
  - Runs weekly and on manual trigger

- **Pre-commit Hooks**
  - Added `pyupgrade` hook for Python syntax modernization (py311+)
  - Added `shellcheck` hook for shell script linting

### Changed: [0.3.0]

- **Static Analysis Now Covers Tests**
  - Flake8 now lints test code (with relaxed SIM117 rules)
  - MyPy now type-checks test code (with `--allow-untyped-defs`)
  - Removed E722 and F401 from global Flake8 ignores

- **Pytest Warnings Configuration**
  - Removed global `-p no:warnings` suppression
  - Added targeted `filterwarnings` for expected dependency warnings
  - Removed `--continue-on-collection-errors` flag

- **MyPy Configuration** (`pyproject.toml`)
  - Tests now included in type checking
  - Added relaxed overrides for test modules
  - Removed test exclusion pattern

### Fixed: [0.3.0]

- **TST-001: Silent ImportError Test Pass**
  - Refactored `test_main.py` to use `pytest.skip()` instead of silent `pass`

- **CFG-003: MyPy Type Errors in Production Code**
  - Added type ignore comments for numpy stubs in `core/artifacts.py`, `storage/memory.py`, `storage/local_fs.py`
  - Fixed `Optional` type annotations in `api/routes/datasets.py` and `api/app.py`

- **Unused Imports** (7 fixed)
  - `tests/fixtures/generate_golden_datasets.py`: removed `os`
  - `tests/integration/test_storage_workflow.py`: removed `Dict`
  - `tests/unit/test_api_app.py`: removed `AsyncMock`
  - `tests/unit/test_api_routes.py`: removed `Dict`, `generators`, `io`
  - `tests/unit/test_main.py`: removed `MagicMock`

### Technical Notes: [0.3.0]

- **SemVer impact**: MINOR – Significant CI/CD infrastructure changes; no API changes
- **Test count**: 207 tests (unchanged, all passing)
- **Coverage**: 100% maintained
- **Documentation**: See `notes/TEST_SUITE_CICD_ENHANCEMENT_DEVELOPMENT_PLAN.md` for full implementation details

---

## [0.2.2] - 2026-02-02

**Summary**: Fixed code coverage configuration and achieved 100% test coverage across all source files.

### Fixed: [0.2.2]

- **Code Coverage Configuration** (`pyproject.toml`, `ci.yml`)
  - Changed from path-based `source` to package-based `source_pkgs = ["juniper_data"]`
  - Simplified CI coverage flags from three path-based `--cov` flags to single `--cov=juniper_data`
  - Coverage now correctly measures at 100% (was 0% due to path mismatch)

### Added: [0.2.2]

- **Comprehensive Unit Tests**
  - `test_main.py` - Tests for `__main__.py` entry point (argument parsing, uvicorn launch)
  - `test_api_app.py` - Tests for FastAPI app factory, lifespan, exception handlers
  - `test_api_routes.py` - Tests for datasets/generators/health route edge cases
  - `test_api_settings.py` - Tests for Settings class and get_settings function
  - Added tests for `get_schema()` in spiral generator
  - Added tests for split edge case when rounding exceeds sample count
  - Added tests for LocalFS storage edge cases (JSON serializer, partial file deletion)
  - Added tests verifying abstract base class behavior

### Technical Notes: [0.2.2]

- **SemVer impact**: PATCH – Configuration fix and test additions only; no API changes
- **Test count**: 207 tests (up from 141)
- **Coverage**: 100% across all 23 source files
- **Root cause**: Nested `juniper_data/juniper_data/` structure caused path-based coverage targets to not exist
- **Documentation**: See `notes/CODE_COVERAGE_FIX.md` for detailed analysis

---

## [0.2.1] - 2026-02-01

**Summary**: CI/CD parity achieved across JuniperCascor, JuniperData, and JuniperCanopy with standardized settings.

### Changed: [0.2.1]

- **CI/CD Configuration Parity**
  - `.pre-commit-config.yaml` (v0.1.1)
    - Line length: 512 for black, isort, flake8
    - Added yamllint hook (v1.35.1, relaxed config)
    - Enabled mypy in CI (fully active)
  - `.github/workflows/ci.yml` (v0.1.1)
    - Coverage threshold: 80% (up from 50%)
    - Added build job with package verification
    - Standardized artifact paths: reports/junit/, reports/htmlcov/, reports/coverage.xml
  - `pyproject.toml` (v0.1.1)
    - Line length: 512 for black/isort
    - Coverage fail_under: 80%

### Technical Notes: [0.2.1]

- **SemVer impact**: PATCH – Configuration changes only; no API changes
- **CI Parity**: All 3 Juniper applications now use identical CI/CD settings

---

## [0.2.0] - 2026-01-31

**Summary**: Added legacy parity mode for spiral generator to achieve statistical compatibility with JuniperCascor's SpiralProblem implementation.

### Added: [0.2.0]

- **Legacy Parity Mode** (`generators/spiral/`)
  - New `algorithm` parameter: `"modern"` (default) or `"legacy_cascor"`
  - New `radius` parameter: Maximum radius/distance (default: 10.0)
  - New `origin` parameter: Center point offset as `(x, y)` tuple (default: `(0.0, 0.0)`)

- **Legacy Cascor Algorithm** (`algorithm="legacy_cascor"`)
  - Sqrt-uniform radial sampling: `sqrt(random) * radius`
  - Distance-as-angle formula: `angle = direction * (distance + offset)`
  - Uniform noise in `[0, noise)` (not zero-centered)
  - Matches statistical properties of original JuniperCascor SpiralProblem

- **New Unit Tests** (8 tests for legacy mode)
  - `test_legacy_mode_generates_correct_shapes`
  - `test_legacy_mode_deterministic_with_seed`
  - `test_legacy_mode_different_from_modern`
  - `test_legacy_mode_uniform_noise_range`
  - `test_legacy_mode_radii_distribution`
  - `test_origin_offset_works`
  - `test_radius_parameter_controls_spread`
  - `test_algorithm_param_validation`

### Technical Notes: [0.2.0]

- **SemVer impact**: MINOR – New features added; backward compatible
- **Test count**: 84 tests passing (up from 76)
- Default behavior unchanged (`algorithm="modern"`)

### Usage: [0.2.0]

```python
# Modern mode (default, same as before)
params = SpiralParams(n_spirals=2, n_points_per_spiral=100)

# Legacy Cascor mode (for parity with JuniperCascor)
params = SpiralParams(
    n_spirals=2,
    n_points_per_spiral=100,
    algorithm="legacy_cascor",
    radius=10.0,
    origin=(0.0, 0.0),
)
```

---

## [0.1.2] - 2026-01-31

**Summary**: Added Conda environment configuration for JuniperData development.

### Added: [0.1.2]

- **Conda Environment** (`conf/conda_environment.yaml`)
  - Python >=3.11 with numpy, pytest, dev tools via conda-forge
  - Editable package installation with `pip install -e .[all]`
  - Full test suite validation (76 tests passing)

### Technical Notes: [0.1.2]

- **SemVer impact**: PATCH – Environment configuration only; no API changes
- **Environment name**: JuniperData
- **Test count**: 76 tests passing in new environment

---

## [0.1.1] - 2026-01-30

**Summary**: Added comprehensive CI/CD pipeline with pre-commit hooks, GitHub Actions workflow, and security scanning. Renamed source directory from `src/` to `juniper_data/` for proper package discovery.

### Added: [0.1.1]

- **CI/CD Pipeline** (`.github/workflows/ci.yml`)
  - Pre-commit job across Python 3.11-3.14 matrix
  - Unit tests with 50% coverage gate
  - Integration tests for PRs and main/develop
  - Security scanning: Gitleaks, Bandit SARIF, pip-audit
  - Quality gate aggregator with proper failure handling
  - pip-based (no conda required)

- **Pre-commit Configuration** (`.pre-commit-config.yaml`)
  - General file checks (YAML, TOML, JSON, merge conflicts)
  - Python formatting: Black (line-length=120)
  - Import sorting: isort (black profile)
  - Linting: Flake8 with bugbear, comprehensions, simplify
  - Type checking: MyPy
  - Security: Bandit SAST scanning

- **Enhanced pyproject.toml**
  - New `test` optional dependency group
  - Bandit configuration section
  - Updated pytest paths for new structure
  - Added dev tools: flake8, bandit, pip-audit, pre-commit

### Changed: [0.1.1]

- **Directory Structure**: Renamed `src/` to `juniper_data/` for proper package discovery
- **Test paths**: Updated from `tests/` to `juniper_data/tests/`

### Technical Notes: [0.1.1]

- **SemVer impact**: PATCH – CI/CD infrastructure only; no API changes
- **Pre-commit status**: All 16 hooks pass
- **Test count**: 76 tests passing

---

## [0.1.0] - 2026-01-29

**Summary**: Initial release of JuniperData - a standalone dataset generation and management service extracted from JuniperCascor as part of the Juniper ecosystem refactoring initiative.

### Added: [0.1.0]

- **Core Generator Module** (`juniper_data/generators/spiral/`)
  - `SpiralParams` - Pydantic model with comprehensive validation
  - `SpiralGenerator` - Pure NumPy N-spiral dataset generator
  - `defaults.py` - Default constants migrated from Cascor
  - Static methods: `generate()`, `_generate_raw()`, `_create_one_hot_labels()`
  - Deterministic reproducibility via `np.random.default_rng(seed)`

- **Core Utilities** (`juniper_data/core/`)
  - `split.py` - Dataset shuffle and split utilities
    - `shuffle_data()` - Shuffle X, y together
    - `split_data()` - Partition into train/test
    - `shuffle_and_split()` - Combined high-level function
  - `dataset_id.py` - Deterministic hash-based dataset ID generation
  - `models.py` - Pydantic models for API contracts
    - `DatasetMeta` - Dataset metadata schema
    - `CreateDatasetRequest/Response` - API request/response models
    - `GeneratorInfo`, `PreviewData` - Additional schemas
  - `artifacts.py` - NPZ artifact handling and checksums

- **Storage Layer** (`juniper_data/storage/`)
  - `DatasetStore` - Abstract base class for storage backends
  - `InMemoryDatasetStore` - In-memory storage for testing
  - `LocalFSDatasetStore` - File-based storage with JSON metadata + NPZ artifacts

- **REST API** (`juniper_data/api/`)
  - FastAPI-based service on port 8100
  - **Health**: `GET /v1/health`
  - **Generators**:
    - `GET /v1/generators` - List available generators
    - `GET /v1/generators/{name}/schema` - Get parameter schema
  - **Datasets**:
    - `POST /v1/datasets` - Create/generate dataset (with caching)
    - `GET /v1/datasets` - List all datasets
    - `GET /v1/datasets/{id}` - Get dataset metadata
    - `GET /v1/datasets/{id}/artifact` - Download NPZ file
    - `GET /v1/datasets/{id}/preview` - Preview samples as JSON
    - `DELETE /v1/datasets/{id}` - Delete dataset
  - Pydantic-settings configuration with `JUNIPER_DATA_` env prefix
  - CORS middleware support

- **Test Suite** (76 tests)
  - 60 unit tests covering generators, split, dataset_id
  - 16 integration tests covering all API endpoints
  - Golden dataset fixtures for parity testing

- **Project Infrastructure**
  - `pyproject.toml` with dependencies and tool configuration
  - `AGENTS.md` with build/test commands
  - `README.md` with installation and usage guide
  - CLI entry point: `python -m juniper_data`

### Technical Notes: [0.1.0]

- **Design Principle**: Pure NumPy core (no PyTorch dependency)
- **Artifact-First API**: Returns NPZ files instead of large JSON payloads
- **Deterministic IDs**: SHA-256 hash of generator + version + params
- **Python Version**: >=3.11
- **Key Dependencies**: numpy>=1.24.0, pydantic>=2.0.0, fastapi>=0.100.0

### Migration Notes: [0.1.0]

- This release corresponds to Phases 0-2 of the JuniperCascor refactoring plan
- Cascor integration (Phase 3) completed in JuniperCascor 0.6.0
- Canopy integration (Phase 4) pending

---

## Version History

| Version    | Date       | Description                        |
| ---------- | ---------- | ---------------------------------- |
| Unreleased | 2026-02-05 | Integration Development Plan       |
| 0.3.0      | 2026-02-04 | Test suite & CI/CD enhancement     |
| 0.2.2      | 2026-02-02 | Code coverage configuration fix    |
| 0.2.1      | 2026-02-01 | CI/CD parity across Juniper        |
| 0.2.0      | 2026-01-31 | Legacy parity mode for spiral      |
| 0.1.2      | 2026-01-31 | Conda environment setup            |
| 0.1.1      | 2026-01-30 | CI/CD Pipeline & Pre-commit        |
| 0.1.0      | 2026-01-29 | Initial release (Phases 0-2)       |

---

## Related Changes

### JuniperCascor 0.6.0 (2026-01-30)

Phase 3 Cascor integration completed:

- Added `JuniperDataClient` for API communication
- Added `SpiralDataProvider` for torch tensor conversion
- Feature flag `JUNIPER_DATA_URL` enables JuniperData mode in SpiralProblem
