# Suggested Commands for Juniper Data Development

## Installation

```bash
# Development installation (recommended)
pip install -e ".[dev]"

# With API support
pip install -e ".[api]"

# Full installation (all optional dependencies)
pip install -e ".[all]"
```

## Testing

```bash
# Run all tests
pytest

# Run unit tests only
pytest juniper_data/tests/unit/

# Run integration tests only
pytest juniper_data/tests/integration/

# Run specific test file
pytest juniper_data/tests/unit/test_spiral_generator.py -v

# Run tests with coverage
pytest juniper_data/tests/ --cov=juniper_data --cov-report=html --cov-report=term-missing --cov-fail-under=80

# Run tests with specific marker
pytest -m "unit"          # Unit tests
pytest -m "integration"   # Integration tests
pytest -m "spiral"        # Spiral generator tests
pytest -m "api"           # API tests
```

## Code Quality

```bash
# Type checking
mypy juniper_data --ignore-missing-imports

# Linting
flake8 juniper_data --max-line-length=512 --extend-ignore=E203,E265,E266,E501,W503,E722,E402,E226,C409,C901,B008,B904,B905,B907,F401

# Format checking
black --check --diff juniper_data

# Import sorting check
isort --check-only --diff juniper_data

# Format code (auto-fix)
black juniper_data
isort juniper_data
```

## Pre-commit Hooks

```bash
# Install pre-commit (one-time)
pip install pre-commit

# Install git hooks (one-time)
pre-commit install

# Run all hooks on all files
pre-commit run --all-files
```

## Security Scanning

```bash
# Bandit SAST scan
bandit -r juniper_data --skip=B101,B311

# Dependency vulnerability check
pip-audit
```

## Running the API

```bash
# Development mode (via module entry point, port 8100)
python -m juniper_data

# Production mode
uvicorn juniper_data.api.app:app --host 0.0.0.0 --port 8100
```

## Utility Scripts (util/)

```bash
# Run all tests with full reporting
./util/run_all_tests.bash

# Other utility scripts
./util/run.bash
./util/run_tests.bash
```

## Git Operations

```bash
# Standard git commands
git status
git diff
git add <files>
git commit -m "message"
git push

# View recent commits
git log --oneline -10
```

## System Utilities (Linux)

```bash
ls, cd, grep, find, cat, head, tail, less, tree
```
