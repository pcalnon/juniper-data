# CI/CD Reference

## juniper-data CI/CD Jobs, Hooks, and Configuration

**Version:** 0.4.6
**Status:** Active
**Last Updated:** October 5, 2026
**Project:** Juniper - Dataset Generation Service

---

## Table of Contents

- [Workflow Files](#workflow-files)
- [Claude Code Workflow](#claude-code-workflow)
- [CI Jobs Reference](#ci-jobs-reference)
- [Pre-commit Hook Reference](#pre-commit-hook-reference)
- [Ruff Configuration](#ruff-configuration)
- [MyPy Configuration](#mypy-configuration)
- [GitHub Actions Pinned Versions](#github-actions-pinned-versions)
- [Environment Variables](#environment-variables)
- [GitHub Environments](#github-environments)
- [Secrets Reference](#secrets-reference)
- [Dependabot Configuration](#dependabot-configuration)
- [CI Scripts Reference](#ci-scripts-reference)
- [Artifact Retention](#artifact-retention)

---

## Workflow Files

| File | Purpose | Trigger |
|------|---------|---------|
| `.github/workflows/ci.yml` | Main CI pipeline (v0.4.0) | Push, PR, schedule, dispatch |
| `.github/workflows/publish.yml` | PyPI publishing | GitHub Release published |
| `.github/workflows/lockfile-update.yml` | Lockfile auto-update (v0.1.0) | Push to `dependabot/pip/**` |
| `.github/workflows/codeql.yml` | Code quality analysis (v1.0.0) | Push to main/develop, PRs, weekly |
| `.github/workflows/claude.yml` | `@claude` assistant | Issue comments, review comments, submitted reviews, issue opened/assigned |

---

## Claude Code Workflow

`.github/workflows/claude.yml` is the live `@claude` assistant for this repo. The file header names `juniper-ml/.github/workflows/claude.yml` as the fleet source-of-truth copy. The action pin stays in the `# vX.Y.Z` comment on the `uses:` line.

The job checks out with `fetch-depth: 1` and passes one input: `anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}`. `trigger_phrase`, `assignee_trigger`, `prompt`, and `allowed_bots` are left unset. The action's default phrase is `@claude`, assignee triggering is off, bots are not allowed, and a matching mention runs in tag mode because no `prompt` is passed.

### Events

| Event | Types subscribed | Workflow `if` |
|-------|------------------|---------------|
| `issue_comment` | `created` | `github.event.comment.body` contains `@claude` |
| `pull_request_review_comment` | `created` | `github.event.comment.body` contains `@claude` |
| `pull_request_review` | `submitted` | `github.event.review.body` contains `@claude` |
| `issues` | `opened`, `assigned` | `github.event.issue.body` or `github.event.issue.title` contains `@claude` |

GitHub's `contains` is case-sensitive and matches a substring. `@Claude` does not start the job. `email@claude.com` and `@claudefoo` do.

### Action gate

The action runs only after that `if` is true. In the action's `src/entrypoints/run.ts` the order is:

1. **Write check** (`checkWritePermissions`), for these entity events. Permission must be `admin` or `write`, or the step throws `Actor does not have write permissions to the repository`. A login ending in `[bot]` returns success from this check without a collaborator lookup.
2. **Phrase check** (`checkContainsTrigger`). The default phrase is `@claude`. The match is case-insensitive and must sit at the start of the text or after whitespace, and must end at whitespace or one of `. , ! ? ; :`. A miss logs `No trigger found, skipping remaining steps` and the job succeeds without a reply.
3. **Human check** (`checkHumanActor` in tag mode), only after the phrase matches. `allowed_bots` is empty, so a non-user actor fails with `Workflow initiated by non-human actor`.

An `issues` `assigned` delivery never satisfies step 2: `assignee_trigger` is unset, and the action reads the title and body only when `eventAction` is `opened`. The workflow `if` still starts the job when the title or body already contains the substring.

`label_trigger` defaults to `claude` inside the action. This workflow does not subscribe to `labeled`, so that default never runs. The workflow also does not subscribe to `pull_request`, so a phrase in a pull-request title or body never schedules the job. The action would honor `pull_request_review` `edited`; this workflow subscribes only to `submitted`.

A run that passes the phrase check and then finds `ANTHROPIC_API_KEY` empty throws `Environment variable validation failed` with: `Either ANTHROPIC_API_KEY, CLAUDE_CODE_OAUTH_TOKEN, or workload identity federation (ANTHROPIC_FEDERATION_RULE_ID and ANTHROPIC_ORGANIZATION_ID) is required when using direct Anthropic API.` This workflow sets none of the alternatives. `use_bedrock`, `use_vertex`, and `use_foundry` stay at their default `false`.

---

## CI Jobs Reference

### ci.yml Jobs

| Job | Depends On | Python | Condition | Purpose |
|-----|-----------|--------|-----------|---------|
| `pre-commit` | -- | 3.12, 3.13, 3.14 | Always | Code quality (ruff, mypy, bandit) |
| `docs` | -- | 3.12 | Always | Markdown link validation |
| `unit-tests` | pre-commit | 3.12, 3.13, 3.14 | Always | Unit tests with coverage |
| `build` | unit-tests | 3.14 | Always | Package build verification |
| `dependency-docs` | build | 3.14 | Always | Dependency documentation |
| `integration-tests` | unit-tests | 3.14 | PRs or main/develop | Integration workflow tests |
| `security` | pre-commit | 3.14 | Always | Gitleaks + Bandit + pip-audit |
| `lockfile-check` | -- | 3.14 | Always | Lockfile freshness validation |
| `required-checks` | All above | -- | `always()` | Quality gate aggregator |
| `slow-tests` | -- | 3.14 | Schedule/dispatch | Long-running tests |
| `notify` | required-checks | -- | `always()` | Build notification |

### publish.yml Jobs

| Job | Depends On | Environment | Purpose |
|-----|-----------|-------------|---------|
| `testpypi` | -- | `testpypi` | Publish to TestPyPI, verify install |
| `pypi` | testpypi | `pypi` | Publish to production PyPI |

---

## Pre-commit Hook Reference

### Hook Versions

| Repository | Version | Hooks |
|-----------|---------|-------|
| `pre-commit/pre-commit-hooks` | v6.0.0 | File checks (11 hooks) |
| `astral-sh/ruff-pre-commit` | v0.15.2 | ruff, ruff-format |
| `pre-commit/mirrors-mypy` | v1.13.0 | mypy (2 hooks) |
| `PyCQA/bandit` | v1.7.9 | bandit |
| `adrienverge/yamllint` | v1.35.1 | yamllint |
| `shellcheck-py/shellcheck-py` | v0.10.0.1 | shellcheck |
| Local | -- | coverage-check, no-unencrypted-env |

### Hook File Patterns

| Hook | Files Pattern | Excludes |
|------|--------------|----------|
| ruff | `^juniper_data/.*\.py$` | -- |
| ruff-format | `^juniper_data/.*\.py$` | -- |
| mypy (prod) | `^juniper_data/(?!tests/).*\.py$` | Test files |
| mypy (test) | `^juniper_data/tests/.*\.py$` | -- |
| bandit | `^juniper_data/(?!tests).*\.py$` | Test files |
| shellcheck | Shell scripts | `last_mod_update-ORIG.bash$`, `last_mod_update-local.bash$`, `no_canopy.bash$` |
| no-unencrypted-env | `^\.env(\.secrets)?$` | -- |

### Global Exclusions

```
.git/, venv/, __pycache__/, .pytest_cache/, .mypy_cache/,
data/, reports/, logs/, htmlcov/, images/, *.egg-info/
```

---

## Ruff Configuration

From `pyproject.toml`:

### General

| Setting | Value |
|---------|-------|
| `line-length` | 120 |
| `target-version` | `"py311"` |

### Lint Rules

**Enabled:**

| Code | Plugin | Description |
|------|--------|-------------|
| `E` | pycodestyle | Style errors |
| `W` | pycodestyle | Style warnings |
| `F` | pyflakes | Logical errors |
| `B` | flake8-bugbear | Bug-prone patterns (including B9xx) |
| `C4` | flake8-comprehensions | Comprehension simplification |
| `I` | isort | Import sorting |
| `UP` | pyupgrade | Python upgrade suggestions |
| `SIM` | flake8-simplify | Code simplification |

**Ignored:**

| Code | Reason |
|------|--------|
| `E203` | Formatter conflict (whitespace before ':') |
| `E265`, `E266` | Block comment formatting preference |
| `E501` | Line length (formatter handles) |
| `E402` | Module-level import ordering |
| `E226` | Missing whitespace around operator |
| `C409` | Unnecessary literal within tuple call |
| `SIM102`, `SIM105`, `SIM117` | Style preferences |
| `B904`, `B905` | Exception handling preferences |

### Per-file Ignores

| Pattern | Ignored | Reason |
|---------|---------|--------|
| `juniper_data/__main__.py` | `T201` | Allow print statements |
| `juniper_data/api/routes/*.py` | `B008` | Allow Pydantic defaults in function args |
| `juniper_data/tests/**/*.py` | `F841`, `C901` | Unused vars, complexity in tests |
| `juniper_data/tests/fixtures/*.py` | `T201` | Allow prints in fixture generators |

### Import Sorting

| Setting | Value |
|---------|-------|
| `known-first-party` | `["juniper_data"]` |
| `section-order` | future, standard-library, third-party, first-party, local-folder |

### Formatting

| Setting | Value |
|---------|-------|
| `quote-style` | `"double"` |
| `line-ending` | `"lf"` |

### McCabe Complexity

| Setting | Value |
|---------|-------|
| `max-complexity` | 15 |

---

## MyPy Configuration

From `pyproject.toml`:

### General Settings

| Setting | Value |
|---------|-------|
| `python_version` | `"3.14"` |
| `warn_return_any` | `false` |
| `warn_unused_configs` | `true` |
| `ignore_missing_imports` | `true` |
| `exclude` | `["^data/", "^logs/", "^reports/", "^htmlcov/"]` |

### Module Overrides

| Module Pattern | `ignore_missing_imports` | `disallow_untyped_defs` |
|---------------|------------------------|------------------------|
| `numpy.*`, `pydantic.*`, `fastapi.*`, `uvicorn.*` | `true` | -- |
| `juniper_data.tests.*` | -- | `false` (relaxed) |

---

## GitHub Actions Pinned Versions

All actions are SHA-pinned for reproducibility:

| Action | Version | SHA |
|--------|---------|-----|
| `actions/checkout` | v4.2.2 | `11bd71901bbe5b1630ceea73d27597364c9af683` |
| `actions/setup-python` | v5.6.0 | `a26af69be951a213d495a4c3e4e4022e16d87065` |
| `actions/cache` | v4.2.3 | `5a3ec84eff668545956fd18022155c47e93e2684` |
| `actions/upload-artifact` | v4.6.0 | `65c4c4a1ddee5b72f698fdd19549f0f0fb45cf08` |
| `codecov/codecov-action` | v5.4.2 | `ad3126e916f78f00edff4ed0317cf185271ccc2d` |
| `gitleaks/gitleaks-action` | v2.3.9 | `ff98106e4c7b2bc287b24eaf42907196329070c7` |
| `github/codeql-action/*` | v3.28.0 | `48ab28a6f5dbc2a99bf1e0131198dd8f1df78169` |
| `pypa/gh-action-pypi-publish` | v1.13.0 | `ed0c53931b1dc9bd32cbe73a98c7f6766f8a527e` |
| `conda-incubator/setup-miniconda` | v3.3.0 | `fc2d68f6413eb2d87b895e92f8584b5b94a10167` |

---

## Environment Variables

### CI Workflow Variables

| Variable | Value | Used By |
|----------|-------|---------|
| `SERVICE_NAME` | `"JuniperData"` | ci.yml |
| `PYTHON_TEST_VERSION` | `"3.14"` | ci.yml |
| `COVERAGE_FAIL_UNDER` | `"80"` | ci.yml, check_module_coverage.py |

### Concurrency

| Setting | Value |
|---------|-------|
| Group | `ci-${{ github.ref }}` |
| Cancel in progress | `true` |

### Permissions

| Workflow | Default | Additional |
|----------|---------|-----------|
| ci.yml | `contents: read` | `security-events: write` (security job) |
| publish.yml | -- | `id-token: write` (OIDC) |
| lockfile-update.yml | -- | `contents: write` |
| codeql.yml | -- | `actions: read`, `contents: read`, `security-events: write` |
| claude.yml | -- | job: `contents: write`, `pull-requests: write`, `issues: write`, `id-token: write`, `actions: read` |

---

## GitHub Environments

| Environment | Wait Timer | Reviewer Required | Purpose |
|-------------|-----------|-------------------|---------|
| `testpypi` | 5 minutes | No | TestPyPI publishing |
| `pypi` | 5 minutes | Yes | Production PyPI publishing |

---

## Secrets Reference

| Secret | Used By | Purpose |
|--------|---------|---------|
| `CODECOV_TOKEN` | ci.yml (unit-tests) | Codecov.io upload token |
| `GITHUB_TOKEN` | ci.yml (security) | Built-in, used by Gitleaks |
| `CROSS_REPO_DISPATCH_TOKEN` | lockfile-update.yml | Custom PAT for re-triggering CI on push |
| `ANTHROPIC_API_KEY` | claude.yml | Passed as the action's `anthropic_api_key`. The workflow header says this secret is set at the org level and that the repo must be able to read it |

---

## Dependabot Configuration

From `.github/dependabot.yml`:

### Python Dependencies (`pip`)

| Setting | Value |
|---------|-------|
| Directory | `/` |
| Schedule | Weekly, Monday 9 AM ET |
| Open PR limit | 5 |
| Labels | `dependencies`, `security` |
| Commit prefix | `deps` |
| Grouping | Minor + patch updates together |

### GitHub Actions

| Setting | Value |
|---------|-------|
| Directory | `/` |
| Schedule | Weekly, Monday |
| Open PR limit | 3 |
| Labels | `dependencies`, `ci` |
| Commit prefix | `ci` |
| Grouping | Only `github/codeql-action*` (group `codeql-action`). Other actions, including `anthropics/claude-code-action`, each open their own PR |

---

## CI Scripts Reference

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/check_module_coverage.py` | Per-module coverage enforcement (85% module, 80% aggregate) | `python scripts/check_module_coverage.py [--run-tests]` |
| `scripts/check_doc_links.py` | Markdown link validation | `python scripts/check_doc_links.py [--verbose] [--exclude DIR]` |
| `scripts/generate_dep_docs.sh` | Dependency documentation artifact generation (`conf/requirements_ci.txt`, `conf/conda_environment_ci.yaml`) | `bash scripts/generate_dep_docs.sh` |

---

## Artifact Retention

| Artifact | Retention | Job |
|----------|-----------|-----|
| Coverage reports (`coverage-report-py*`) | 30 days | unit-tests |
| Build artifacts (`dist/`) | 30 days | build |
| Dependency docs (`conf/requirements_ci*`, `conf/conda_environment_ci*`) | 90 days | dependency-docs |
| Security reports (`reports/security/`) | 30 days | security |
| Integration test reports | 30 days | integration-tests |

---

**Last Updated:** October 5, 2026
**Version:** 0.4.6
**Maintainer:** Paul Calnon
