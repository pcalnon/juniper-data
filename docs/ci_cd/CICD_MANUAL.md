# CI/CD Manual

## Comprehensive CI/CD Pipeline Guide for juniper-data

**Version:** 0.4.7
**Status:** Active
**Last Updated:** October 6, 2026
**Project:** Juniper - Dataset Generation Service

---

## Table of Contents

1. [Introduction](#introduction)
2. [Pipeline Overview](#pipeline-overview)
3. [GitHub Actions Workflows](#github-actions-workflows)
   - [ci.yml -- Main CI Pipeline](#ciyml----main-ci-pipeline)
   - [publish.yml -- PyPI Publishing](#publishyml----pypi-publishing)
   - [lockfile-update.yml -- Dependency Lockfile Auto-Update](#lockfile-updateyml----dependency-lockfile-auto-update)
   - [codeql.yml -- Code Quality Analysis](#codeqlyml----code-quality-analysis)
4. [Pre-commit Hooks](#pre-commit-hooks)
   - [Hook Overview](#hook-overview)
   - [File Checks](#file-checks)
   - [Ruff (Lint + Format)](#ruff-lint--format)
   - [MyPy (Type Checking)](#mypy-type-checking)
   - [Bandit (Security)](#bandit-security)
   - [Coverage Gate (Pre-push)](#coverage-gate-pre-push)
   - [Shell and YAML Linting](#shell-and-yaml-linting)
   - [SOPS Encrypted Files](#sops-encrypted-files)
5. [Dependabot Configuration](#dependabot-configuration)
6. [Quality Gate](#quality-gate)
7. [Release Process](#release-process)
8. [Troubleshooting](#troubleshooting)

---

## Introduction

juniper-data uses a multi-layer CI/CD strategy:

- **Pre-commit hooks**: Local code quality enforcement (ruff, mypy, bandit, yamllint, shellcheck)
- **Pre-push hooks**: Coverage gate (80% aggregate, 85% per-module)
- **GitHub Actions CI**: Automated testing across Python 3.12/3.13/3.14, security scanning, build verification, documentation validation
- **GitHub Actions publishing**: Two-stage PyPI publishing (TestPyPI then production PyPI)
- **Dependabot**: Automated dependency updates with lockfile synchronization
- **CodeQL**: Weekly semantic code analysis

---

## Pipeline Overview

### Job Dependency Graph

```bash
pre-commit ──┬─→ unit-tests ──┬─→ build ──→ dependency-docs ──┐
             │                │                               │
             │                └─→ integration-tests ──────────┤
             │                                                │
             ├─→ security ────────────────────────────────────┤
             ├─→ docs ────────────────────────────────────────┤
             └─→ lockfile-check ──────────────────────────────┘
                                                              │
                                                              └─→ required-checks ──→ notify
```

### Trigger Matrix

| Workflow              | Push to main/develop | Push to feature/fix | Pull Requests | Schedule            | Release | Manual |
|-----------------------|----------------------|---------------------|---------------|---------------------|---------|--------|
| `ci.yml`              | Yes                  | Yes                 | Yes           | Daily 6 AM UTC      | No      | Yes    |
| `publish.yml`         | No                   | No                  | No            | No                  | Yes     | No     |
| `lockfile-update.yml` | No                   | dependabot branches | No            | No                  | No      | No     |
| `codeql.yml`          | Yes                  | No                  | Yes (to main) | Weekly Mon 6 AM UTC | No      | No     |

---

## GitHub Actions Workflows

### ci.yml -- Main CI Pipeline

The primary CI workflow runs up to 11 jobs with concurrency control (`ci-${{ github.ref }}`, cancel-in-progress).

#### Job: `pre-commit`

Multi-version code quality validation across Python 3.12, 3.13, and 3.14 with `fail-fast: false`.

- Runs all pre-commit hooks: ruff, mypy, bandit, yamllint, file checks
- Caches `~/.cache/pre-commit` keyed on `.pre-commit-config.yaml` hash
- First job to run; blocks `unit-tests` and `security`

#### Job: `docs`

Validates documentation link integrity using `scripts/check_doc_links.py`.

- Checks all markdown files for broken internal links and anchors
- Excludes `templates/` and `history/` directories
- Runs independently (no dependencies)

#### Job: `unit-tests`

Core test execution with coverage enforcement across Python 3.12, 3.13, and 3.14.

- **Depends on**: `pre-commit`
- **Markers**: `-m "unit and not slow"`
- **Timeout**: 60 seconds per test, `--maxfail=5`
- **Coverage**: `--cov-fail-under=80` (aggregate), then `check_module_coverage.py` for 85% per-module
- **Installs juniper-data-client** from `main` branch via git+https
- **Reports**: JUnit XML, HTML coverage, Cobertura XML, JSON coverage
- **Codecov upload**: Python 3.14 only, using `CODECOV_TOKEN` secret

#### Job: `build`

Package build and verification.

- **Depends on**: `unit-tests`
- Builds sdist and wheel with `python -m build`
- Verifies `.tar.gz` and `.whl` files exist
- Uploads `dist/` as artifact (30 day retention)

#### Job: `dependency-docs`

Captures the environment that installed this repo and uploads it. The capture is a diagnostic artifact. CI does not commit it, and it leaves `conf/requirements.txt` and `conf/requirements-ORIG.txt` alone. Those two files remain the supplemental pip list for `util/setup_environment.bash`.

- **Depends on**: `build`
- **Python**: `PYTHON_TEST_VERSION` (`3.14`)
- **Conda setup**: `conda-incubator/setup-miniconda` with `miniforge-version: latest`, `auto-activate-base: true`, and `python-version` set to `PYTHON_TEST_VERSION`. The version record is the `# vX.Y.Z` comment on that step's SHA pin.
- **Shell**: the install and generate steps use `bash -l {0}` (a login shell). That login shell is what puts Miniforge `conda` and the activated base on `PATH`.
- **Installed before capture**: `juniper-data-client` from `git+https://github.com/pcalnon/juniper-data-client.git@main`, then `pip install ".[all]"`, then `juniper-ci-tools>=0.9.0,<0.10.0`. The freeze is that environment, which is separate from `requirements.lock`.
- **Command**: `juniper-generate-dep-docs` with no arguments. In `juniper-ci-tools` 0.9.0 that console script is `juniper_ci_tools.cli:main`. This repository has no `scripts/generate_dep_docs.sh`.
- **Pip file**: `conf/requirements_ci.txt`. The body is `python -m pip list --format=freeze` for the interpreter running the tool.
- **Conda file**: `conf/conda_environment_ci.yaml`. The body is the `dependencies:` block of `conda env export --no-builds`: lines after `dependencies:` and before the next top-level key (`prefix:` or `variables:`). The generator does not run `conda list --explicit`.
- **Headers**: with no arguments the tool reads `notes/JUNIPER_2026-03-11_JUNIPER-ML_PIP-DEPENDENCY-FILE-HEADER.md` and `notes/JUNIPER_2026-03-15_JUNIPER-ML_CONDA-DEPENDENCY-FILE-HEADER.md`. This repo ships `notes/PIP_DEPENDENCY_FILE_HEADER.md` and `notes/CONDA_DEPENDENCY_FILE_HEADER.md` under different names, so CI writes the two-line fallback (`# <filename> - Generated <date>` and `# Python: <version>`). The conda template ends with a `dependencies:` key. The fallback does not, so the uploaded conda document is a YAML list of dependency lines. `yaml.safe_load` accepts that list, and the step still exits 0.
- **Backups**: an existing output is copied to `conf/requirements_ci_<YYYY-MM-DD_HH-MM-SS>.txt` or `conf/conda_environment_ci_<YYYY-MM-DD_HH-MM-SS>.yaml` before overwrite.
- **Conda missing**: if `conda` is not on `PATH`, the yaml is skipped, a warning is printed, and the command still exits 0. A YAML parse failure exits 1. A missing `pyproject.toml`, or one with no `[project].version`, exits 1.
- **Upload**: artifact `dependency-docs`, retained 90 days: `conf/requirements_ci.txt`, `conf/requirements_ci_*.txt`, `conf/conda_environment_ci.yaml`, `conf/conda_environment_ci_*.yaml`.

Regenerate locally with this repo's headers, from a login shell whose base environment is active:

```bash
pip install "juniper-ci-tools>=0.9.0,<0.10.0"
juniper-generate-dep-docs \
  --pip-header PIP_DEPENDENCY_FILE_HEADER.md \
  --conda-header CONDA_DEPENDENCY_FILE_HEADER.md
```

The checked-in copies under `conf/` are snapshots from the previous generator. The next CI run uploads new artifacts and leaves those committed files in place.

#### Job: `integration-tests`

Full workflow integration tests.

- **Depends on**: `unit-tests`
- **Condition**: Only on PRs or main/develop branches
- **Markers**: `-m "integration and not slow"`
- **Timeout**: 120 seconds per test, `--maxfail=3`

#### Job: `security`

Multi-tool security scanning.

1. **Gitleaks**: Secret detection in repository history
2. **Bandit SAST**: Static security analysis with SARIF report
   - Blocking check: fails on medium+ severity and confidence
   - SARIF uploaded to GitHub Security tab
3. **pip-audit**: Dependency vulnerability scanning
   - Filters out `juniper-data` packages (self-referencing)
   - Runs in `--strict` mode

#### Job: `lockfile-check`

Validates `requirements.lock` freshness.

- Runs independently (no dependencies)
- Uses `uv pip compile` to regenerate lockfile
- Diffs against committed `requirements.lock`
- Prints remediation command on failure

#### Job: `required-checks`

Quality gate aggregator. Runs `if: always()` and checks status of all required jobs.

**Required to pass**: pre-commit, unit-tests, build, dependency-docs, security, docs, lockfile-check

**Optional** (failure = error, skip = OK): integration-tests

#### Job: `slow-tests`

Long-running test execution.

- **Trigger**: Schedule (daily 6 AM UTC) or manual dispatch only
- **Markers**: `-m "slow"`
- **Timeout**: 600 seconds (10 minutes)

#### Job: `notify`

Build completion notification summary.

---

### publish.yml -- PyPI Publishing

Two-stage publishing triggered by GitHub Release creation.

#### Stage 1: TestPyPI

1. Build package with `python -m build`
2. Validate with `twine check dist/*`
3. Publish to TestPyPI via `pypa/gh-action-pypi-publish` (OIDC trusted publishing)
4. Wait 30 seconds for index sync
5. Install from TestPyPI and verify import

**Environment**: `testpypi` (5-minute wait timer)

#### Stage 2: Production PyPI

Identical to TestPyPI stage but publishes to production.

**Environment**: `pypi` (5-minute wait timer + required reviewer approval)

Both stages use `attestations: false` and SHA-pinned actions. Version is extracted from the release tag (strips `v` prefix).

---

### lockfile-update.yml -- Dependency Lockfile Auto-Update

Automatically regenerates `requirements.lock` when Dependabot updates dependencies.

- **Trigger**: Push to `dependabot/pip/**` branches
- **Condition**: Only runs for `dependabot[bot]` actor
- Runs `uv pip compile pyproject.toml --extra api --extra observability --extra mnist --extra equities --upgrade -o requirements.lock`
- Uses `CROSS_REPO_DISPATCH_TOKEN` (not `GITHUB_TOKEN`) so the push re-triggers CI
- Commits with `[dependabot skip]` prefix to prevent Dependabot re-processing

The workflow only updates `requirements.lock`, which is generated from `pyproject.toml` for Docker and CI reproducibility. Dependabot PRs that target `conf/requirements.txt` or `conf/requirements-ORIG.txt` are changing the legacy/local setup snapshot used by `util/setup_environment.bash`; keep those two files in sync and do not expect the lockfile workflow to rewrite them.

---

### codeql.yml -- Code Quality Analysis

GitHub CodeQL semantic analysis for Python.

- **Trigger**: Push to main/develop, PRs to main, weekly Monday 6 AM UTC
- Runs `security-and-quality` query suite
- Results appear in GitHub Security tab

---

## Pre-commit Hooks

### Hook Overview

| Hook                    | Stage        | Tool                 | Files                           | Auto-fix?     |
|-------------------------|--------------|----------------------|---------------------------------|---------------|
| check-yaml              | commit       | pre-commit-hooks     | `*.yaml`, `*.yml`               | No            |
| check-toml              | commit       | pre-commit-hooks     | `*.toml`                        | No            |
| check-json              | commit       | pre-commit-hooks     | `*.json`                        | No            |
| end-of-file-fixer       | commit       | pre-commit-hooks     | All                             | Yes           |
| trailing-whitespace     | commit       | pre-commit-hooks     | All (md linebreaks preserved)   | Yes           |
| check-merge-conflict    | commit       | pre-commit-hooks     | All                             | No            |
| check-added-large-files | commit       | pre-commit-hooks     | All (max 1000 KB)               | No            |
| check-case-conflict     | commit       | pre-commit-hooks     | All                             | No            |
| check-ast               | commit       | pre-commit-hooks     | `*.py`                          | No            |
| debug-statements        | commit       | pre-commit-hooks     | `*.py`                          | No            |
| detect-private-key      | commit       | pre-commit-hooks     | All                             | No            |
| ruff                    | commit       | Ruff v0.15.2         | `juniper_data/**/*.py`          | Yes (`--fix`) |
| ruff-format             | commit       | Ruff v0.15.2         | `juniper_data/**/*.py`          | Yes           |
| mypy (prod)             | commit       | MyPy v1.13.0         | `juniper_data/(?!tests/).*\.py` | No            |
| mypy (test)             | commit       | MyPy v1.13.0         | `juniper_data/tests/.*\.py`     | No            |
| bandit                  | commit       | Bandit v1.7.9        | `juniper_data/(?!tests).*\.py`  | No            |
| yamllint                | commit       | yamllint v1.35.1     | `*.yaml`, `*.yml`               | No            |
| shellcheck              | commit       | ShellCheck v0.10.0.1 | Shell scripts                   | No            |
| no-unencrypted-env      | commit       | Local                | `.env`, `.env.secrets`          | No (blocks)   |
| coverage-check          | **pre-push** | Local                | All (always_run)                | No            |

### File Checks

Standard pre-commit-hooks (v6.0.0) validate file integrity: YAML/TOML/JSON syntax, trailing whitespace, merge conflict markers, large files (max 1000 KB), case conflicts, Python AST validity, debug statements, and private key patterns.

### Ruff (Lint + Format)

Ruff replaces black, isort, flake8, and related tools. Runs on `juniper_data/**/*.py` only.

- **Linting**: Auto-fixes violations with `--fix`
- **Formatting**: Enforces consistent style
- **Config**: `[tool.ruff]` in pyproject.toml (line-length=320, target-version=py312)

### MyPy (Type Checking)

Two separate hooks with different strictness levels:

- **Production code**: `--ignore-missing-imports --no-strict-optional` (requires `types-redis`)
- **Test code**: Same as above plus `--allow-untyped-defs` (relaxed for test functions)

### Bandit (Security)

Runs on production code only (excludes `tests/`). Skips `B101` (assert) and `B311` (random module).

### Coverage Gate (Pre-push)

Runs `scripts/check_module_coverage.py --run-tests` on the pre-push stage. Enforces 80% aggregate and 85% per-module coverage. This means you can commit freely, but cannot push code that drops coverage below thresholds.

### Shell and YAML Linting

- **ShellCheck**: Severity level `warning` (skips style/info). Excludes specific legacy scripts.
- **yamllint**: Relaxed preset with parsable output.

### SOPS Encrypted Files

Blocks commits of unencrypted `.env` or `.env.secrets` files. Ensures secrets are SOPS-encrypted before committing.

---

## Dependabot Configuration

### Python Dependencies

- **Schedule**: Weekly on Mondays at 9 AM ET
- **PR limit**: 5 open PRs
- **Labels**: `dependencies`, `security`
- **Grouping**: Minor + patch updates grouped together
- **Commit prefix**: `deps`

### GitHub Actions

- **Schedule**: Weekly on Mondays
- **PR limit**: 3 open PRs
- **Labels**: `dependencies`, `ci`
- **Commit prefix**: `ci`

When Dependabot pushes to `dependabot/pip/**`, the `lockfile-update.yml` workflow automatically regenerates `requirements.lock` and commits the update.

---

## Quality Gate

The `required-checks` job in ci.yml acts as the merge quality gate. All of these must pass:

| Check                                    | Required    | Failure Impact                             |
|------------------------------------------|-------------|--------------------------------------------|
| pre-commit (code quality)                | Yes         | Blocks merge                               |
| unit-tests (all Python versions)         | Yes         | Blocks merge                               |
| build (package verification)             | Yes         | Blocks merge                               |
| dependency-docs                          | Soft        | Failure blocks, skip OK                    |
| security (gitleaks + bandit + pip-audit) | Soft        | Failure blocks, skip OK                    |
| docs (link validation)                   | Yes         | Blocks merge                               |
| lockfile-check                           | Yes         | Blocks merge                               |
| integration-tests                        | Conditional | Failure blocks, skip OK (feature branches) |

---

## Release Process

1. **Create a GitHub Release** with a tag matching the version (e.g., `v0.4.2`)
2. `publish.yml` triggers automatically
3. **TestPyPI stage**: Build, verify, publish, install-test (5-min environment wait)
4. **PyPI stage**: Same process, requires manual reviewer approval
5. Approve the PyPI deployment via GitHub environment approval or `gh api`

See [PyPI Publishing Procedure](../../../juniper-ml/notes/JUNIPER_2026-06-18_JUNIPER-ECOSYSTEM_PYPI-PUBLISH-PROCEDURE.md) for ecosystem-wide publishing lessons.

---

## Troubleshooting

**Pre-commit hooks not running**: Ensure you've run both `pre-commit install` and `pre-commit install --hook-type pre-push`.

**Ruff modifies files on commit**: This is by design. Ruff auto-fixes lint violations and reformats code. Stage the changes and commit again.

**CI fails but local passes**: Check Python version matrix. CI tests on 3.12, 3.13, and 3.14. Ensure your local environment matches.

**Lockfile check fails**: Regenerate with:

```bash
uv pip compile pyproject.toml --extra api --extra observability --extra mnist --extra equities -o requirements.lock
```

**Dependabot PR missing lockfile update**: The `lockfile-update.yml` workflow handles `requirements.lock` automatically when the branch matches `dependabot/pip/**` and the actor is `dependabot[bot]`. If the PR only changes `conf/requirements.txt` and `conf/requirements-ORIG.txt`, a lockfile update may not be needed.

**CodeQL findings**: Review in GitHub Security tab. These are informational and don't block the merge quality gate.

**TestPyPI publish fails**: Check that the release tag matches the version in `pyproject.toml`. Version is extracted by stripping the `v` prefix from the tag.

**Coverage drops after push**: Run `python scripts/check_module_coverage.py --run-tests` locally to identify modules below the 85% threshold.

**`dependency-docs` conda artifact is a YAML list**: the job calls `juniper-generate-dep-docs` with no arguments, so it does not load `notes/CONDA_DEPENDENCY_FILE_HEADER.md`. Pass `--conda-header CONDA_DEPENDENCY_FILE_HEADER.md` and `--pip-header PIP_DEPENDENCY_FILE_HEADER.md` when you want the committed header shape.

**`dependency-docs` succeeds and uploads no conda file**: `conda` was not on `PATH`. The tool warns and exits 0. The job's install and generate steps need `shell: bash -l {0}` after `setup-miniconda`. A non-login shell does not see the activated base.

---

## End of CI/CD Manual
