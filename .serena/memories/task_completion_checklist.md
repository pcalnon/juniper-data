# Task Completion Checklist - Juniper Data

When completing a task, run through this checklist:

## 1. Code Quality Checks

### Type Checking
```bash
mypy juniper_data --ignore-missing-imports
```
- Fix any type errors before committing

### Linting
```bash
flake8 juniper_data --max-line-length=512 --extend-ignore=E203,E265,E266,E501,W503,E722,E402,E226,C409,C901,B008,B904,B905,B907,F401
```
- Address all linting issues

### Formatting
```bash
black juniper_data
isort juniper_data
```
- Run formatters to ensure consistent style

## 2. Testing

### Run Tests
```bash
pytest
```
- All tests must pass

### Coverage (for significant changes)
```bash
pytest juniper_data/tests/ --cov=juniper_data --cov-report=term-missing --cov-fail-under=80
```
- Maintain 80%+ coverage
- Add tests for new functionality

## 3. Pre-commit Validation
```bash
pre-commit run --all-files
```
- All hooks must pass
- This runs: black, isort, flake8, mypy, bandit, yamllint

## 4. Security (for changes involving dependencies or sensitive code)
```bash
bandit -r juniper_data --skip=B101,B311
pip-audit
```

## 5. Documentation
- Update docstrings for changed/new public APIs
- Update CHANGELOG.md for notable changes
- Update README.md if user-facing features change

## 6. Commit Guidelines
- Write clear, descriptive commit messages
- Reference issue numbers if applicable
- Keep commits focused and atomic

## Quick One-Liner
For quick validation before committing:
```bash
black juniper_data && isort juniper_data && mypy juniper_data --ignore-missing-imports && pytest
```
