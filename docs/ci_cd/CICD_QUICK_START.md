# CI/CD Quick Start

## Run juniper-data CI Checks Locally in 5 Minutes

**Version:** 0.4.2
**Status:** Active
**Last Updated:** March 3, 2026
**Project:** Juniper - Dataset Generation Service

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Install Pre-commit](#install-pre-commit)
- [Run All Local Checks](#run-all-local-checks)
- [Run Individual Checks](#run-individual-checks)
- [Run Coverage Gate](#run-coverage-gate)
- [Verify Before Push](#verify-before-push)
- [Common Issues](#common-issues)
- [Next Steps](#next-steps)

---

## Prerequisites

- [ ] Python >= 3.12 installed
- [ ] Conda environment `JuniperData` activated (`conda activate JuniperData`)
- [ ] Dev dependencies installed: `pip install -e ".[all]"`

---

## Install Pre-commit

```bash
# Install pre-commit hooks (one-time)
pre-commit install
pre-commit install --hook-type pre-push
```

After installation, hooks run automatically on `git commit` (code quality) and `git push` (coverage gate).

---

## Run All Local Checks

```bash
# Run all pre-commit hooks on all files
pre-commit run --all-files
```

This runs:

1. **File checks** -- YAML, TOML, JSON validation; trailing whitespace; merge conflicts; large files; debug statements
2. **Ruff** -- linting + auto-fix + formatting
3. **MyPy** -- type checking (production code strict, test code relaxed)
4. **Bandit** -- security scan
5. **yamllint** -- YAML linting
6. **ShellCheck** -- shell script linting

---

## Run Individual Checks

```bash
# Ruff linting only
pre-commit run ruff --all-files

# Ruff formatting only
pre-commit run ruff-format --all-files

# MyPy type checking only
pre-commit run mypy --all-files

# Bandit security scan only
pre-commit run bandit --all-files

# YAML lint only
pre-commit run yamllint --all-files
```

---

## Run Coverage Gate

The coverage check is on the **pre-push** stage (not pre-commit) to keep commits fast:

```bash
# Run coverage check manually
pre-commit run coverage-check --all-files --hook-stage pre-push

# Or run the script directly
python scripts/check_module_coverage.py --run-tests
# Enforces: 80% aggregate, 85% per-module
```

---

## Verify Before Push

Complete pre-push checklist:

```bash
# 1. Run all code quality checks
pre-commit run --all-files

# 2. Run tests with coverage
pytest --cov=juniper_data --cov-report=term-missing --cov-fail-under=80

# 3. Check per-module coverage
python scripts/check_module_coverage.py --run-tests

# 4. Validate documentation links
python scripts/check_doc_links.py
```

---

## Common Issues

**`pre-commit` not found**: Install with `pip install pre-commit` or ensure `.[all]` extras are installed.

**Ruff auto-fix changes files**: This is expected. Ruff applies fixes on commit. Stage the changes and commit again.

**MyPy import errors**: MyPy is configured with `--ignore-missing-imports` for third-party packages. If you see persistent errors, ensure `types-redis` is installed.

**Coverage below 80%**: Run `python scripts/check_module_coverage.py --run-tests` to see per-module breakdown. Add tests for modules below 85%.

**Lockfile check fails in CI**: Regenerate with `uv pip compile pyproject.toml --extra api --extra observability -o requirements.lock`.

---

## Next Steps

- [CI/CD Manual](CICD_MANUAL.md) -- comprehensive guide to all CI/CD pipelines and workflows
- [CI/CD Reference](CICD_REFERENCE.md) -- complete job, hook, and environment variable reference
- [Testing Quick Start](../testing/TESTING_QUICK_START.md) -- get tests running locally

---

**Last Updated:** March 3, 2026
**Version:** 0.4.2
**Status:** Active
