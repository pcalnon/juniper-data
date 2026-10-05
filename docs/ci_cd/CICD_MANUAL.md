# CI/CD Manual

## Comprehensive CI/CD Pipeline Guide for juniper-data

**Version:** 0.4.6
**Status:** Active
**Last Updated:** October 5, 2026
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
   - [claude.yml -- Claude Code](#claudeyml----claude-code)
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
- **Claude Code**: `@claude` on issue comments, pull-request review comments, submitted reviews, and newly opened issues

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

`claude.yml` is omitted from this matrix. It runs on `issue_comment` (`created`), `pull_request_review_comment` (`created`), `pull_request_review` (`submitted`), and `issues` (`opened`, `assigned`). See [claude.yml](#claudeyml----claude-code).

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

Generates dependency documentation snapshots.

- **Depends on**: `build`
- Installs `.[all]` plus the `juniper-data-client` main branch before capture
- Runs `scripts/generate_dep_docs.sh`
- Outputs `conf/requirements_ci.txt` and `conf/conda_environment_ci.yaml` as workflow artifacts, with timestamped backups if prior files exist in the checkout
- Does not update the committed `conf/requirements.txt` or `conf/requirements-ORIG.txt` environment snapshots
- Uses Conda (Miniforge) for environment capture
- 90 day artifact retention

These generated files describe the environment that ran in CI. They are diagnostic artifacts, not source inputs for local setup. The checked-in setup script installs supplemental packages from `conf/requirements.txt`, and `conf/requirements-ORIG.txt` is kept as the paired baseline copy for that setup snapshot.

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

### claude.yml -- Claude Code

`.github/workflows/claude.yml` replies when someone writes `@claude` in a place the workflow watches. The job checks out with `fetch-depth: 1` and passes a single action input, `anthropic_api_key`, from `secrets.ANTHROPIC_API_KEY`. The file header says that secret is set at the org level and that the repo must be able to read it.

Write the phrase in one of these:

- the body of a new issue comment, including a comment on a pull request (`issue_comment`)
- the body of a pull-request review comment
- the body of a submitted pull-request review
- the title or body of an issue at the moment it is opened

The workflow `if` uses `contains(..., '@claude')`. That test is case-sensitive, so `@Claude` does not start the job. It is also a substring test, so `email@claude.com` and `@claudefoo` do start the job.

When the job starts, the action checks the phrase again. The default phrase is `@claude`. The match is case-insensitive, and the phrase must sit at the start of the field or after whitespace, and must end at whitespace or one of `. , ! ? ; :`. A miss logs `No trigger found, skipping remaining steps` and the job succeeds with no reply. That is what an `issues` `assigned` delivery does: the workflow starts because the title or body already contains the substring, and `assignee_trigger` is unset so the action does not treat the assignment as a request. The action reads the issue title and body only on `opened`.

The action checks write access before the phrase check. A commenter who is neither `admin` nor `write` fails with `Actor does not have write permissions to the repository`. A login ending in `[bot]` passes that lookup. After a real phrase match, a non-user actor fails with `Workflow initiated by non-human actor` because `allowed_bots` is empty.

A matching run with an empty `ANTHROPIC_API_KEY` fails later:

```text
Environment variable validation failed:
  - Either ANTHROPIC_API_KEY, CLAUDE_CODE_OAUTH_TOKEN, or workload identity federation (ANTHROPIC_FEDERATION_RULE_ID and ANTHROPIC_ORGANIZATION_ID) is required when using direct Anthropic API.
```

This workflow does not set the OAuth token or the federation inputs, and it leaves `use_bedrock`, `use_vertex`, and `use_foundry` at their default `false`.

A phrase in a pull-request title or body never schedules the job: `on:` does not include `pull_request`. The action's default `label_trigger` is `claude`, and this workflow does not subscribe to `labeled`. The action accepts an edited review body; this workflow subscribes only to `submitted`.

The action version is the `# vX.Y.Z` comment on the `uses:` line. Dependabot groups only `github/codeql-action*`, so a bump of `anthropics/claude-code-action` arrives as its own pull request.

Tables for events, permissions, and the secret are in [Claude Code Workflow](CICD_REFERENCE.md#claude-code-workflow).

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
- **Grouping**: only `github/codeql-action*` (group `codeql-action`). Other actions, including `anthropics/claude-code-action`, each open their own PR

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

**`@claude` produced no reply**: Put the phrase in an issue comment, a pull-request review comment, a submitted review body, or the title or body of an issue when it is opened. `@Claude` does not start the job. A green run that logs `No trigger found, skipping remaining steps` means the action's word-boundary check rejected the phrase, or the event was an assignment. See [claude.yml](#claudeyml----claude-code).

**`@claude` failed before a reply**: `Actor does not have write permissions to the repository` means the commenter lacks write. `Workflow initiated by non-human actor` means a bot; `allowed_bots` is unset. `Environment variable validation failed` naming `ANTHROPIC_API_KEY` means the secret was empty for this repo.

**TestPyPI publish fails**: Check that the release tag matches the version in `pyproject.toml`. Version is extracted by stripping the `v` prefix from the tag.

**Coverage drops after push**: Run `python scripts/check_module_coverage.py --run-tests` locally to identify modules below the 85% threshold.

---

## End of CI/CD Manual
