# AGENTS.md - Juniper Data Project Guide

**Project**: Juniper Data - Dataset Generation Service
**Repository**: pcalnon/juniper-data
**Author**: Paul Calnon
**License**: MIT License
**Version**: 0.12.0
**Last Updated**: 2026-08-30

---

## Hazards (resident — do not relocate)

Directives whose **non-application destroys work**. Everything else in this file may be demoted to
`docs/REFERENCE.md` under the memory budget; these may not, because a pointer only helps an agent
that already knows to look. Adding a new hazard here is legitimate — ratchet space out of a
reference section in the same PR rather than waiving the budget gate.

- **`/tmp/` is prohibited** as the home for any script that produces, modifies or analyzes
  repository content — it is reaped when sessions, sandboxes or containers end, and the scripts are
  irrecoverable. Scratch *data* there is fine; source files are not. Permanent utilities live in
  `util/`, single-use ones in `util/ad-hoc/`. Full rule: § Script Placement.

## Quick Reference

### Conda Environment

> **Required:** Activate the `JuniperData` conda environment before running any commands.

```bash
conda activate JuniperData
```

### Essential Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Install with API support
pip install -e ".[api]"

# Install everything
pip install -e ".[all]"

# Run all tests
pytest

# Run unit tests only
pytest juniper_data/tests/unit/

# Run integration tests only
pytest juniper_data/tests/integration/

# Run performance benchmarks
pytest juniper_data/tests/performance/ --benchmark-enable -v

# Run tests with coverage (uses source_pkgs from pyproject.toml)
pytest juniper_data/tests/ --cov=juniper_data --cov-report=html --cov-report=term-missing --cov-fail-under=80

# Run a specific test file
pytest juniper_data/tests/unit/test_spiral_generator.py -v

# Type checking with mypy
mypy juniper_data --ignore-missing-imports

# Linting with ruff (replaces flake8, isort, pyupgrade)
ruff check juniper_data

# Format checking with ruff (replaces black)
ruff format --check juniper_data

# Pre-commit hooks (CI/CD local validation)
pip install pre-commit                    # Install pre-commit (one-time)
pre-commit install                        # Install git hooks (one-time)
pre-commit run --all-files                # Run all hooks on all files

# Security scanning
bandit -r juniper_data                    # Run Bandit SAST scan
pip-audit                                 # Check for dependency vulnerabilities

# Start API server (development)
python -m juniper_data                    # Use module entry point on port 8100

# Start API server (production)
python -m juniper_data --host 0.0.0.0 --port 8100
```

---

## Project Architecture

The service's layered architecture, request lifecycle, and generator plug-in model. Moved to [`docs/REFERENCE.md` § Project Architecture Reference](docs/REFERENCE.md#project-architecture-reference) — read it when working on this area.

## Observability — Prometheus Collectors

Every Prometheus collector this service registers, and the register-or-reuse contract behind it. Moved to [`docs/REFERENCE.md` § Prometheus Collector Reference](docs/REFERENCE.md#prometheus-collector-reference) — read it when working on this area.

## Code Style Conventions

### Naming Conventions

**Constants**:

- Uppercase with underscores, prefixed by component: `_DATA_DEFAULT_NOISE`
- Hierarchical naming: `_SPIRAL_GENERATOR_DEFAULT_POINTS`
- Layer-scoped constants live in their layer's `constants.py`:
  - `juniper_data/api/constants.py` — header names, status code defaults, body/rate-limit limits, error message templates
  - `juniper_data/storage/constants.py` — filenames, metadata keys, table/column names, storage size limits
  - `juniper_data/core/constants.py` — encoding strings (`utf-8`), magic numbers, fixed metadata keys
  - `juniper_data/generators/<name>/params.py` — per-generator parameter defaults referenced by Pydantic `Field(default=...)`
- Application code (middleware, security, observability, storage backends, generators, route handlers) imports from these modules; inline literals are reserved for genuinely local one-shot values
- HTTP status codes use `starlette.status` constants instead of magic numbers (`HTTP_404_NOT_FOUND` rather than `404`)

**Classes**:

- PascalCase: `SpiralGenerator`, `DatasetStore`, `LocalFSDatasetStore`

**Methods/Functions**:

- snake_case: `generate_dataset`, `get_configuration`

**Private Members**:

- Single underscore prefix: `_internal_method`, `_private_attribute`

**Dunder Methods**:

- Double underscore: `__init__`, `__repr__`

### Code Formatting

- Line length: 320 characters (configured in `[tool.ruff] line-length` in pyproject.toml)
- Ruff formatter (replaces black) with `ruff>=0.9.0`
- Ruff isort rules for imports (profile: known-first-party = `juniper_data`)
- Quote style: double quotes, LF line endings
- Type hints required for all public methods
- Max cyclomatic complexity: 15

### Documentation

- Docstrings for all public classes and methods
- Google-style docstring format
- Type annotations in signatures, not docstrings

---

## Dependencies

### Core Dependencies

| Library | Purpose |
|---------|---------|
| `numpy>=1.24.0` | Numerical computations, array operations |
| `pydantic>=2.0.0` | Data validation, model definitions |
| `python-dotenv>=1.0.0` | Environment variable loading from `.env` files |

### API Dependencies (Optional: `pip install -e ".[api]"`)

| Library | Purpose |
|---------|---------|
| `fastapi>=0.100.0` | REST API framework |
| `uvicorn[standard]>=0.23.0` | ASGI server |
| `pydantic-settings>=2.0.0` | Settings management with env var prefix support |

### ARC-AGI Dependencies (Optional: `pip install -e ".[arc-agi]"`)

| Library | Purpose |
|---------|---------|
| `arc-agi>=0.9.0` | ARC-AGI visual reasoning dataset support |

### MNIST Dependencies (Optional: `pip install -e ".[mnist]"`)

| Library | Purpose |
|---------|---------|
| `datasets[vision]>=4.0.0` | MNIST / Fashion-MNIST from the Hugging Face Hub; `[vision]` pulls Pillow for image decode. Heavy chain (pyarrow, pandas) — extra-gated, in the Docker image via `requirements.lock`. First call downloads into `HF_HOME`; offline deployments need a seeded cache |

### Test Dependencies (Optional: `pip install -e ".[test]"`)

| Library | Purpose |
|---------|---------|
| `pytest>=7.0.0` | Testing framework |
| `pytest-cov>=4.0.0` | Coverage reporting |
| `pytest-timeout>=2.2.0` | Test timeout enforcement (60s default) |
| `pytest-asyncio>=0.21.0` | Async test support |
| `pytest-benchmark>=4.0.0` | Performance benchmarking |
| `httpx>=0.24.0` | HTTP test client for FastAPI |
| `coverage[toml]>=7.0.0` | Coverage with pyproject.toml config |
| `juniper-data-client>=0.3.0` | Client library for integration tests |

### Observability Dependencies (Optional: `pip install -e ".[observability]"`)

| Library | Purpose |
|---------|---------|
| `prometheus-client>=0.20.0` | Prometheus metrics exposition |
| `sentry-sdk[fastapi]>=2.0.0` | Error tracking and monitoring |

### Development Dependencies (Optional: `pip install -e ".[dev]"`)

| Library | Purpose |
|---------|---------|
| `ruff>=0.9.0` | Linting and formatting (replaces black, isort, flake8, pyupgrade) |
| `mypy>=1.0.0` | Static type checking |
| `bandit[sarif]>=1.9.4` | Security static analysis (SAST) |
| `pip-audit>=2.7.0` | Dependency vulnerability scanning |
| `pre-commit>=3.0.0` | Git hook management |

---

## Testing

### Test Organization

- `tests/unit/` -- Unit tests for individual components (30+ files)
- `tests/integration/` -- Integration tests for full workflows (5 files)
- `tests/performance/` -- Generator and storage benchmarks via pytest-benchmark (41 tests)
- `tests/api/` -- API-specific tests (batch operations)
- `tests/fixtures/` -- Golden dataset fixtures (NPZ archives + metadata JSON)
- `conftest.py` -- Shared fixtures (default/custom SpiralParams, generated datasets, sample arrays)

### Test Markers

```python
@pytest.mark.unit          # Unit tests
@pytest.mark.integration   # Integration tests
@pytest.mark.performance   # Performance and benchmarking tests
@pytest.mark.slow          # Tests that take a long time to run
@pytest.mark.spiral        # Spiral generator tests
@pytest.mark.api           # API endpoint tests
@pytest.mark.generators    # Generator tests
@pytest.mark.storage       # Storage tests
```

### Test Naming

- Files: `test_<component>.py`
- Classes: `Test<ComponentName>`
- Methods: `test_<behavior_under_test>`

### Performance Testing

Performance benchmarks use pytest-benchmark and are disabled by default (`--benchmark-disable` in addopts).

```bash
# Default: benchmarks run as quick smoke tests (no timing)
pytest juniper_data/tests/performance/

# Enable timing
pytest juniper_data/tests/performance/ --benchmark-enable -v

# Save baseline for regression tracking
pytest juniper_data/tests/performance/ --benchmark-enable --benchmark-autosave

# Compare against saved baseline
pytest juniper_data/tests/performance/ --benchmark-enable --benchmark-compare
```

### Coverage

- Aggregate minimum: 80% (enforced in `pyproject.toml` and CI)
- Per-module minimum: 85% (enforced by `scripts/check_module_coverage.py` in CI)
- Source: `juniper_data` package (tests excluded from metrics)
- Branch coverage enabled

Reproduce the CI coverage gates locally (full suite — aggregate + per-module):

```bash
make coverage                 # convenience wrapper
bash util/run_coverage.bash   # source of truth (mirrors .github/workflows/ci.yml)
```

Gates: 80% aggregate (override with `COVERAGE_FAIL_UNDER=<n>`) plus the per-module floor enforced by `scripts/check_module_coverage.py`. Full suite by design; for a narrower run use plain `pytest`.

---

## API Design

Route-by-route design notes: status codes, pagination, content negotiation, and the binary routes. Moved to [`docs/REFERENCE.md` § API Design Reference](docs/REFERENCE.md#api-design-reference) — read it when working on this area.

## Storage Backends

Each storage backend, its configuration, and the durability guarantee it does and does not make. Moved to [`docs/REFERENCE.md` § Storage Backend Reference](docs/REFERENCE.md#storage-backend-reference) — read it when working on this area.

## Security

### API Key Authentication

- Enabled when `JUNIPER_DATA_API_KEYS` is set (comma-separated list)
- Validated via `X-API-Key` request header
- Docker secrets supported via `JUNIPER_DATA_API_KEYS_FILE`
- Exempt paths: `/v1/health*`, `/docs`, `/openapi.json`, `/redoc`

### Rate Limiting

- Fixed-window per-minute rate limiting (thread-safe)
- Key: API key (authenticated) or client IP (unauthenticated)
- Configurable via `JUNIPER_DATA_RATE_LIMIT_REQUESTS_PER_MINUTE` (default: 60)
- Response headers: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`
- Returns 429 Too Many Requests when exceeded

### Security Headers

`SecurityHeadersMiddleware` adds to all responses:

- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Permissions-Policy: camera=(), microphone=(), geolocation=()`
- `Content-Security-Policy: default-src 'none'; frame-ancestors 'none'`
- `Strict-Transport-Security` (only when behind HTTPS proxy)

### Request Body Limits

`RequestBodyLimitMiddleware` rejects request bodies exceeding 10 MB (returns 413 Payload Too Large).

### CORS

- Disabled by default (empty `cors_origins`)
- Configured via `JUNIPER_DATA_CORS_ORIGINS` (comma-separated)
- Credentials support when origins are explicitly listed (not `*`)

### API Documentation

- `/docs` (Swagger) and `/redoc` endpoints available in development
- Conditional: can be disabled via configuration for production deployments

### Security Scanning (CI)

- **Bandit**: SAST scanning with SARIF output
- **pip-audit**: Dependency vulnerability checking
- **gitleaks**: Secret detection in git history
- **CodeQL**: GitHub code scanning

### Best Practices

- No secrets or API keys committed to codebase
- All input validated via Pydantic models
- Sensitive files excluded via `.gitignore` and `.gitleaks.toml`
- Docker secrets preferred over environment variables for credentials

---

## Observability

### Prometheus Metrics

Enabled via `JUNIPER_DATA_METRICS_ENABLED=true`. Exposed at `/metrics`.

| Metric | Type | Labels |
|--------|------|--------|
| `juniper_data_http_requests_total` | Counter | method, endpoint, status |
| `juniper_data_http_request_duration_seconds` | Histogram | method, endpoint |
| `juniper_data_dataset_generations_total` | Counter | generator, status |
| `juniper_data_dataset_generation_duration_seconds` | Histogram | generator |
| `juniper_data_datasets_cached` | Gauge | -- |
| `juniper_data_build` | Info | version, python_version |

### Structured Logging

- `JuniperJsonFormatter` outputs structured JSON (timestamp, level, logger, message, service, request_id)
- Configurable via `JUNIPER_DATA_LOG_FORMAT` (`text` or `json`)
- Log levels: TRACE, VERBOSE, DEBUG, INFO, WARNING, ERROR, CRITICAL, FATAL

### Sentry Integration

- Optional via `JUNIPER_DATA_SENTRY_DSN`
- Automatic release tagging, log capture, and trace collection

### Request ID Propagation

- `RequestIdMiddleware` injects `X-Request-ID` (UUID if not provided by caller)
- Propagated via `ContextVar` for correlation across log entries

---

## Configuration

All configuration uses the `JUNIPER_DATA_` environment variable prefix (via Pydantic BaseSettings).

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `JUNIPER_DATA_HOST` | str | `127.0.0.1` | API server bind host |
| `JUNIPER_DATA_PORT` | int | `8100` | API server bind port |
| `JUNIPER_DATA_STORAGE_PATH` | str | `./data/datasets` | Local storage directory |
| `JUNIPER_DATA_LOG_LEVEL` | str | `INFO` | Logging level |
| `JUNIPER_DATA_LOG_FORMAT` | str | `text` | Log format (`text` or `json`) |
| `JUNIPER_DATA_CORS_ORIGINS` | list | `[]` | Allowed CORS origins |
| `JUNIPER_DATA_API_KEYS` | str | `None` | Comma-separated API keys (disabled if unset) |
| `JUNIPER_DATA_API_KEYS_FILE` | str | `None` | Docker secrets file path for API keys |
| `JUNIPER_DATA_RATE_LIMIT_ENABLED` | bool | `true` | Enable rate limiting |
| `JUNIPER_DATA_RATE_LIMIT_REQUESTS_PER_MINUTE` | int | `60` | Rate limit per client per minute |
| `JUNIPER_DATA_SENTRY_DSN` | str | `None` | Sentry DSN for error tracking |
| `JUNIPER_DATA_METRICS_ENABLED` | bool | `false` | Enable Prometheus metrics at `/metrics` |

Reference: `.env.example` provides a template with all variables.

---

## CI/CD Pipeline

Per-workflow reference for `.github/workflows/`, including the contract each job must not break. Moved to [`docs/REFERENCE.md` § CI/CD Pipeline Reference](docs/REFERENCE.md#cicd-pipeline-reference) — read it when working on this area.

## Docker

Image build, compose wiring, and the environment each container expects. Moved to [`docs/REFERENCE.md` § Docker Reference](docs/REFERENCE.md#docker-reference) — read it when working on this area.

## Development Workflow

### Adding New Features

1. Create feature in appropriate module
2. Add Pydantic models for validation
3. Add tests in `tests/unit/` or `tests/integration/`
4. Run security scanning (`bandit -r juniper_data`)
5. Run pre-commit hooks (`pre-commit run --all-files`)
6. Update documentation
7. Run full test suite with coverage

### Adding New Generators

1. Create new subpackage under `generators/` with `__init__.py`, `params.py`, and `generator.py`
2. Implement `params.py` with a Pydantic `GeneratorParams` model
3. Implement `generator.py` with a `@staticmethod generate(params)` method returning `dict[str, np.ndarray]`
4. Register generator in `GENERATOR_REGISTRY` in `api/routes/generators.py`
5. Add unit tests in `tests/unit/test_<generator>_generator.py`
6. Add integration test coverage
7. Run full test suite

---

## Integration Context

JuniperData is part of the Juniper ecosystem alongside **juniper-cascor** (CasCor neural network backend) and **JuniperCanopy** (real-time monitoring dashboard).

### Ecosystem Architecture

All projects are standalone repositories with independent CI/CD. Inter-service communication uses REST via published PyPI client packages.

```
juniper-data-client (PyPI) --> juniper-data (REST API, port 8100)
juniper-cascor-client (PyPI) --> juniper-cascor (REST/WS API, port 8201)
juniper-canopy --> uses both clients
juniper-deploy --> Docker Compose orchestration
```

### Integration Points

- **Port**: 8100 (default)
- **Feature Flag**: `JUNIPER_DATA_URL` environment variable enables JuniperData mode in consumers
- **Data Contract**: NPZ artifacts with keys `X_train`, `y_train`, `X_test`, `y_test`, `X_full`, `y_full` (all `float32`)
- **API Prefix**: `/v1/`
- **Client Library**: `juniper-data-client>=0.3.0` on PyPI
- **Consumers**: juniper-cascor, JuniperCanopy (via `juniper-data-client`)
- **Health Endpoint**: `/v1/health` (used by Docker Compose and consumer startup checks)

### Key Documentation

| Document | Location | Purpose |
|----------|----------|---------|
| Development Roadmap | `notes/JUNIPER-DATA_POST-RELEASE_DEVELOPMENT-ROADMAP.md` | Outstanding work items, priorities, and status |
| API Reference | `docs/api/JUNIPER_DATA_API.md` | Complete API documentation |
| User Manual | `docs/USER_MANUAL.md` | Full user documentation |
| Developer Cheatsheet | `docs/DEVELOPER_CHEATSHEET.md` | Quick reference for developers |
| Documentation Overview | `docs/DOCUMENTATION_OVERVIEW.md` | Navigation guide for all docs |
| Polyrepo Migration Plan | `notes/POLYREPO_MIGRATION_PLAN.md` | Redirect to canonical copy in `juniper-cascor` |
| Monorepo Analysis | `notes/MONOREPO_ANALYSIS.md` | Redirect to canonical copy in `juniper-cascor` |

---

## Script Placement

**Permanent utilities** live in `util/`. **Single-use / temporary / unfinished scripts** go in `util/ad-hoc/` (create on first use). See [`util/ad-hoc/README.md`](util/ad-hoc/README.md) for the per-script header / lifecycle conventions.

`/tmp/` is **prohibited** as the home for any script that produces, modifies, or analyzes repository content. `/tmp/` is reaped when sessions / sandboxes / containers end, and scripts placed there are lost (irrecoverable). `/tmp/` remains fine as a scratch *workspace* for intermediate artifacts the script itself creates and reads — the prohibition is on script *source files*.

This is an ecosystem-wide rule restated in the parent `Juniper/AGENTS.md` "Cross-Project Conventions" section. Motivating incident: irrecoverable loss of `phase4_consolidate.py` and `v2_citation_validate.py` from the juniper-ml requirements-snapshot effort.

---

## Worktree Procedures (Mandatory — Task Isolation)

> **OPERATING INSTRUCTION**: All feature, bugfix, and task work SHOULD use git worktrees for isolation. Worktrees keep the main working directory on the default branch while task work proceeds in a separate checkout.

### What This Is

Git worktrees allow multiple branches of a repository to be checked out simultaneously in separate directories. For the Juniper ecosystem, all worktrees are centralized in **`/home/pcalnon/Development/python/Juniper/worktrees/`** using a standardized naming convention.

The full setup and cleanup procedures are defined in:

- **`notes/WORKTREE_SETUP_PROCEDURE.md`** — Creating a worktree for a new task
- **`notes/WORKTREE_CLEANUP_PROCEDURE_V2.md`** — Merging, removing, and pushing after task completion (V2 — fixes CWD-trap bug)

Read the appropriate file when starting or completing a task.

### Worktree Directory Naming

Format: `<repo-name>--<branch-name>--<YYYYMMDD-HHMM>--<short-hash>`

Example: `juniper-data--feature--add-generator--20260225-1430--973ae391`

- Slashes in branch names are replaced with `--`
- All worktrees reside in `/home/pcalnon/Development/python/Juniper/worktrees/`

### When to Use Worktrees

| Scenario | Use Worktree? |
| -------- | ------------- |
| Feature development (new feature branch) | **Yes** |
| Bug fix requiring a dedicated branch | **Yes** |
| Quick single-file documentation fix on main | No |
| Exploratory work that may be discarded | **Yes** |
| Hotfix requiring immediate merge | **Yes** |

### Quick Reference

**Setup** (full procedure in `notes/WORKTREE_SETUP_PROCEDURE.md`):

```bash
cd /home/pcalnon/Development/python/Juniper/juniper-data
git fetch origin && git checkout main && git pull origin main
BRANCH_NAME="feature/my-task"
git branch "$BRANCH_NAME" main
REPO_NAME=$(basename "$(pwd)")
SAFE_BRANCH=$(echo "$BRANCH_NAME" | sed 's|/|--|g')
WORKTREE_DIR="/home/pcalnon/Development/python/Juniper/worktrees/${REPO_NAME}--${SAFE_BRANCH}--$(date +%Y%m%d-%H%M)--$(git rev-parse --short=8 HEAD)"
git worktree add "$WORKTREE_DIR" "$BRANCH_NAME"
cd "$WORKTREE_DIR"
```

**Cleanup** (full procedure in `notes/WORKTREE_CLEANUP_PROCEDURE_V2.md`):

```bash
# Phase 1: Push current work
cd "$OLD_WORKTREE_DIR" && git push origin "$OLD_BRANCH"
# Phase 2: Create new worktree BEFORE removing old (prevents CWD-trap)
git fetch origin
git worktree add "$NEW_WORKTREE_DIR" -b "$NEW_BRANCH" origin/main
cd "$NEW_WORKTREE_DIR"
# Phase 3: Create PR (do NOT merge directly to main)
gh pr create --base main --head "$OLD_BRANCH" --title "<title>" --body "<body>"
# Phase 4: Cleanup
git worktree remove "$OLD_WORKTREE_DIR"
git branch -d "$OLD_BRANCH"
git worktree prune
```

### Rules

- **Centralized location**: All worktrees go in `/home/pcalnon/Development/python/Juniper/worktrees/`. Never create worktrees inside the repo directory.
- **Clean before you start**: Ensure the main working directory is clean before creating a worktree.
- **Push before you merge**: Always push the working branch to remote before merging (backup).
- **Prune after cleanup**: Run `git worktree prune` after removing a worktree to clean metadata.
- **Do not leave stale worktrees**: Clean up worktrees promptly after merging.

---

## Thread Handoff (Mandatory — Replaces Thread Compaction)

> **CRITICAL OPERATING INSTRUCTION**: Thread handoff MUST be performed instead of thread compaction. This section defines mandatory behavior that overrides Claude Code's default compaction process.

### What This Is

When a long-running conversation approaches context limits, Claude Code normally performs **thread compaction** — summarizing prior context to free token capacity. This introduces information loss. Instead, Claude Code instances working on this project MUST perform a **proactive thread handoff**: transferring a curated, high-signal summary to a fresh thread with full context capacity.

The full handoff protocol is defined in **`notes/THREAD_HANDOFF_PROCEDURE.md`**. Read that file when a handoff is triggered.

### When to Trigger a Handoff

**Automatic trigger (pre-compaction threshold):** Initiate a thread handoff when token utilization reaches **95% to 99%** of the level at which thread compaction would normally be triggered. This means the handoff fires when you are within **1% to 5%** of the compaction threshold, ensuring the handoff completes before compaction would occur.

Concretely:

- If compaction would trigger at N% context utilization, begin handoff at (N - 5)% to (N - 1)%.
- **Self-assessment rule**: At each turn where you are performing multi-step work, assess whether you are approaching the compaction threshold. If you estimate you are within 5% of it, begin the handoff protocol immediately.
- When the system compresses prior messages or you receive a context compression notification, treat this as a signal that handoff should have already occurred -- immediately initiate one.

**Additional triggers** (from `notes/THREAD_HANDOFF_PROCEDURE.md`):

| Condition                   | Indicator                                                            |
| --------------------------- | -------------------------------------------------------------------- |
| **Context saturation**      | Thread has performed 15+ tool calls or edited 5+ files               |
| **Phase boundary**          | A logical phase of work is complete                                  |
| **Degraded recall**         | Re-reading a file already read, or re-asking a resolved question     |
| **Multi-module transition** | Moving between major components                                      |
| **User request**            | User says "hand off", "new thread", or similar                       |

**Do NOT handoff** when:

- The task is nearly complete (< 2 remaining steps)
- The current thread is still sharp and producing correct output
- The work is tightly coupled and splitting would lose critical in-flight state

### How to Execute a Handoff

1. **Checkpoint**: Inventory what was done, what remains, what was discovered, and what files are in play
2. **Compose the handoff goal**: Write a concise, actionable summary (see templates in `notes/THREAD_HANDOFF_PROCEDURE.md`)
3. **Present to user**: Output the handoff goal to the user and recommend starting a new thread with that goal as the initial prompt
4. **Include verification commands**: Always specify how the new thread should verify its starting state (test commands, file checks)
5. **State git status**: Mention branch, staged files, and any uncommitted work

### Rules

- **This is not optional.** Every Claude Code instance on this project must follow these rules.
- **Handoff early, not late.** A handoff at 70% context usage is better than compaction at 95%.
- **Do not duplicate CLAUDE.md content** in the handoff goal -- the new thread reads CLAUDE.md automatically.
- **Be specific** in the handoff goal: include file paths, decisions made, and test status.
