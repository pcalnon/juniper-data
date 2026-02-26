# AGENTS.md - Juniper Data Project Guide

**Project**: Juniper Data - Dataset Generation Service
**Version**: 0.4.0
**License**: MIT License
**Author**: Paul Calnon
**Last Updated**: 2026-02-06

---

## Quick Reference

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
pip install bandit pip-audit              # Install security tools
bandit -r juniper_data                    # Run Bandit SAST scan
pip-audit                                 # Check for dependency vulnerabilities

# Start API server (development)
python -m juniper_data                    # Use module entry point on port 8100

# Start API server (production)
uvicorn juniper_data.api.app:app --host 0.0.0.0 --port 8100
```

---

## Project Architecture

### Directory Structure

```bash
JuniperData/
├── juniper_data/              # Main package
│   ├── __init__.py            # Package init with version
│   ├── core/                  # Core functionality
│   │   └── __init__.py
│   ├── generators/            # Dataset generators
│   │   ├── __init__.py
│   │   └── spiral/            # Spiral dataset generator
│   │       └── __init__.py
│   ├── storage/               # Dataset persistence
│   │   └── __init__.py
│   └── api/                   # FastAPI application
│       ├── __init__.py
│       └── routes/            # API route handlers
│           └── __init__.py
├── tests/                     # Test suite
│   ├── __init__.py
│   ├── unit/                  # Unit tests
│   │   └── __init__.py
│   └── integration/           # Integration tests
│       └── __init__.py
├── pyproject.toml             # Project configuration
├── README.md                  # Project documentation
└── AGENTS.md                  # This file
```

### Component Overview

| Component            | Purpose                                 |
| -------------------- | --------------------------------------- |
| `core/`              | Base classes, exceptions, configuration |
| `generators/`        | Dataset generation implementations      |
| `generators/spiral/` | Two-spiral classification dataset       |
| `storage/`           | Dataset persistence and retrieval       |
| `api/`               | FastAPI REST service                    |
| `api/routes/`        | API endpoint handlers                   |

---

## Code Style Conventions

Following JuniperCascor patterns:

### Naming Conventions

**Constants**:

- Uppercase with underscores, prefixed by component: `_DATA_DEFAULT_NOISE`
- Hierarchical naming: `_SPIRAL_GENERATOR_DEFAULT_POINTS`

**Classes**:

- PascalCase: `SpiralGenerator`, `DatasetStorage`

**Methods/Functions**:

- snake_case: `generate_dataset`, `get_configuration`

**Private Members**:

- Single underscore prefix: `_internal_method`, `_private_attribute`

**Dunder Methods**:

- Double underscore: `__init__`, `__repr__`

### Code Formatting

- Line length: 120 characters
- Ruff formatter (replaces black)
- Ruff isort rules for imports
- Type hints required for all public methods

### Documentation

- Docstrings for all public classes and methods
- Google-style docstring format
- Type annotations in signatures, not docstrings

---

## Dependencies

### Core Dependencies

| Library    | Purpose                      |
| ---------- | ---------------------------- |
| `numpy`    | Numerical computations       |
| `pydantic` | Data validation and settings |

### API Dependencies (Optional)

| Library   | Purpose            |
| --------- | ------------------ |
| `fastapi` | REST API framework |
| `uvicorn` | ASGI server        |

### Development Dependencies

| Library      | Purpose                                  |
| ------------ | ---------------------------------------- |
| `pytest`     | Testing framework                        |
| `pytest-cov` | Coverage reporting                       |
| `ruff`       | Linting and formatting (replaces black, isort, flake8, pyupgrade) |
| `mypy`       | Static type checking                     |

---

## Testing

### Test Organization

- `tests/unit/` - Unit tests for individual components
- `tests/integration/` - Integration tests for full workflows

### Test Markers

```python
@pytest.mark.unit          # Unit tests
@pytest.mark.integration   # Integration tests
@pytest.mark.spiral        # Spiral generator tests
@pytest.mark.api           # API endpoint tests
@pytest.mark.generators    # Generator tests
@pytest.mark.storage       # Storage tests
```

### Test Naming

- Files: `test_<component>.py`
- Classes: `Test<ComponentName>`
- Methods: `test_<behavior_under_test>`

---

## API Design

### REST Conventions

- Use nouns for resources: `/datasets`, `/generators`
- Use HTTP methods appropriately: GET, POST, PUT, DELETE
- Return proper status codes
- Include pagination for list endpoints

### Response Format

```python
{
    "status": "success",
    "data": { ... },
    "meta": {
        "timestamp": "...",
        "version": "0.1.0"
    }
}
```

---

## Security Notes

- No secrets or API keys in codebase
- Validate all input data with Pydantic
- Sensitive files excluded via `.gitignore`

---

## Development Workflow

### Adding New Features

1. Create feature in appropriate module
2. Add Pydantic models for validation
3. Add tests in `tests/unit/` or `tests/integration/`
4. Update documentation
5. Run tests and type checking

### Adding New Generators

1. Create new subpackage under `generators/`
2. Implement generator class following `SpiralGenerator` pattern
3. Add API routes in `api/routes/`
4. Add comprehensive tests

---

## Integration Context

JuniperData is part of the Juniper ecosystem alongside **JuniperCascor** (CasCor neural network backend) and **JuniperCanopy** (web frontend dashboard).

### Integration Points

- **Port**: 8100 (default)
- **Feature Flag**: `JUNIPER_DATA_URL` environment variable enables JuniperData mode in consumers
- **Data Contract**: NPZ artifacts with keys `X_train`, `y_train`, `X_test`, `y_test`, `X_full`, `y_full` (all `float32`)
- **API Prefix**: `/v1/`
- **Consumers**: JuniperCascor (`SpiralDataProvider`), JuniperCanopy (`DemoMode`, `CascorIntegration`)

### Key Documentation

| Document | Location | Purpose |
|----------|----------|---------|
| Integration Development Plan | `notes/INTEGRATION_DEVELOPMENT_PLAN.md` | Outstanding work items and priorities |
| Polyrepo Migration Plan | `notes/POLYREPO_MIGRATION_PLAN.md` | Redirect to canonical copy in `juniper-cascor` |
| Monorepo Analysis | `notes/MONOREPO_ANALYSIS.md` | Redirect to canonical copy in `juniper-cascor` |

---

## Worktree Procedures (Mandatory — Task Isolation)

> **OPERATING INSTRUCTION**: All feature, bugfix, and task work SHOULD use git worktrees for isolation. Worktrees keep the main working directory on the default branch while task work proceeds in a separate checkout.

### What This Is

Git worktrees allow multiple branches of a repository to be checked out simultaneously in separate directories. For the Juniper ecosystem, all worktrees are centralized in **`/home/pcalnon/Development/python/Juniper/worktrees/`** using a standardized naming convention.

The full setup and cleanup procedures are defined in:
- **`notes/WORKTREE_SETUP_PROCEDURE.md`** — Creating a worktree for a new task
- **`notes/WORKTREE_CLEANUP_PROCEDURE.md`** — Merging, removing, and pushing after task completion

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

**Cleanup** (full procedure in `notes/WORKTREE_CLEANUP_PROCEDURE.md`):
```bash
cd "$WORKTREE_DIR" && git push origin "$BRANCH_NAME"
cd /home/pcalnon/Development/python/Juniper/juniper-data
git checkout main && git pull origin main
git merge "$BRANCH_NAME"
git push origin main
git worktree remove "$WORKTREE_DIR"
git branch -d "$BRANCH_NAME"
git push origin --delete "$BRANCH_NAME"
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

- If compaction would trigger at N% context utilization, begin handoff at (N − 5)% to (N − 1)%.
- **Self-assessment rule**: At each turn where you are performing multi-step work, assess whether you are approaching the compaction threshold. If you estimate you are within 5% of it, begin the handoff protocol immediately.
- When the system compresses prior messages or you receive a context compression notification, treat this as a signal that handoff should have already occurred — immediately initiate one.

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
- **Do not duplicate CLAUDE.md content** in the handoff goal — the new thread reads CLAUDE.md automatically.
- **Be specific** in the handoff goal: include file paths, decisions made, and test status.
