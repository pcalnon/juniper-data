# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

| Version | Date       | Description                    |
| ------- | ---------- | ------------------------------ |
| 0.2.2   | 2026-02-02 | Code coverage configuration fix|
| 0.2.1   | 2026-02-01 | CI/CD parity across Juniper    |
| 0.2.0   | 2026-01-31 | Legacy parity mode for spiral  |
| 0.1.2   | 2026-01-31 | Conda environment setup        |
| 0.1.1   | 2026-01-30 | CI/CD Pipeline & Pre-commit    |
| 0.1.0   | 2026-01-29 | Initial release (Phases 0-2)   |

---

## Related Changes

### JuniperCascor 0.6.0 (2026-01-30)

Phase 3 Cascor integration completed:

- Added `JuniperDataClient` for API communication
- Added `SpiralDataProvider` for torch tensor conversion
- Feature flag `JUNIPER_DATA_URL` enables JuniperData mode in SpiralProblem
