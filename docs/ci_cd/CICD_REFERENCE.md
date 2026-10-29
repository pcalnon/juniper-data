# CI/CD Reference

## juniper-data CI/CD Jobs, Hooks, and Configuration

**Version:** 0.4.3
**Status:** Active
**Last Updated:** October 8, 2026
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
| `.github/workflows/notify-consumers.yml` | Tell PyPI consumers a release is installable, then wait for a run | `workflow_call` from `publish.yml` after PyPI; `workflow_dispatch` to re-send |
| `.github/workflows/lockfile-update.yml` | Lockfile auto-update (v0.1.0) | Push to `dependabot/pip/**` by `dependabot[bot]`; `pull_request` when `pyproject.toml` changes (same repo, not `release/**`) |
| `.github/workflows/codeql.yml` | Code quality analysis (v1.0.0) | Push to main/develop, PRs, weekly |
| `.github/workflows/publish-image.yml` | GHCR image (amd64 + arm64). Serve-and-version gate runs before any tag | Release `v*`, PR on image inputs (build-only), manual |
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

GitHub's `contains` is not case sensitive and matches a substring, so `@Claude`, `email@claude.com`, and `@claudefoo` all start the job. The action's phrase check below then accepts `@Claude` and rejects the other two.

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
| `notify-consumers` | pypi | -- (`permissions: {}`) | Call `notify-consumers.yml` with the release tag |

### Consumer release notification

`notify-consumers.yml` posts `juniper-data-published` to `pcalnon/juniper-recurrence` using `CROSS_REPO_DISPATCH_TOKEN` (`curl --fail-with-body`). The version is `X.Y.Z` after a leading `v` is stripped. A 204 only means GitHub accepted the event.

The confirm step then looks for a run titled `juniper-data-published` (no `run-name:` on the listener): 12 listings, 10 seconds apart, `created` at or after 30 seconds before the POST. No successful listing and a listing with no matching run are different failures.

Contract: [Consumer release notification](../REFERENCE.md#consumer-release-notification).

### publish-image.yml serve check

`util/check_image_serves.py` is invoked from the build job with `--dist juniper-data --module juniper_data --port 8100`. Build-only runs address `data-smoke:<arch>` and pass `pyproject.toml`'s version. Publish runs address `ghcr.io/pcalnon/juniper-data@<digest>`. On a release, `--expect-version` is the tag with the leading `v` removed, and the step exits before the script when that string disagrees with `pyproject.toml`.

Defaults left in place: health path `/v1/health` (version field required), no enveloped path, 120s timeout. The merge job is the only tag writer. Contract: [Image serve-and-version gate](../REFERENCE.md#image-serve-and-version-gate).

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

Every third-party action is pinned to a full commit SHA, with its release in a `# vX.Y.Z` comment on the `uses:` line, with one exception: `agents-md-touch-up.yml` checks out with `actions/checkout@v7.0.1`, a tag. The table was re-derived on 2026-10-29 from all 14 files in `.github/workflows/` at `95ce8190`, and each SHA is the commit its commented tag resolves to (`gh api repos/<owner>/<repo>/commits/<tag> --jq .sha`). `publish.yml` also calls `./.github/workflows/notify-consumers.yml`, a reusable workflow in this repository, which takes no pin. `notify-consumers.yml`, `pr-base-branch-guard.yml` and `pr-budget-alarm.yml` use no action.

| Action | Version | SHA | Used in |
|--------|---------|-----|---------|
| `actions/checkout` | v7.0.1 | `3d3c42e5aac5ba805825da76410c181273ba90b1` | `ci.yml`, `claude.yml`, `codeql.yml`, `lockfile-update.yml`, `main-verify.yml`, `publish-image.yml`, `publish.yml`, `scheduled-tests.yml`, `security-scan.yml`, `sequence-safety.yml`. `agents-md-touch-up.yml` uses the `v7.0.1` tag |
| `actions/setup-python` | v7.0.0 | `5fda3b95a4ea91299a34e894583c3862153e4b97` | `ci.yml`, `lockfile-update.yml`, `main-verify.yml`, `publish.yml`, `scheduled-tests.yml`, `security-scan.yml`, `sequence-safety.yml` |
| `actions/cache` | v6.1.0 | `55cc8345863c7cc4c66a329aec7e433d2d1c52a9` | `ci.yml` |
| `actions/upload-artifact` | v7.0.2 | `cf430e030ddbb5b0abf93d22962f4752f3646cd9` | `ci.yml`, `main-verify.yml`, `publish-image.yml`, `scheduled-tests.yml`, `security-scan.yml`, `sequence-safety.yml` |
| `actions/download-artifact` | v8.0.2 | `9000827ccba6bdab643e8b6fd33ac0654aef8333` | `publish-image.yml` |
| `codecov/codecov-action` | v7.1.1 | `303a32d7a59b442fa8d48b6a1cc6825c09c847a5` | `ci.yml` |
| `gitleaks/gitleaks-action` | v3.0.0 | `e0c47f4f8be36e29cdc102c57e68cb5cbf0e8d1e` | `ci.yml` |
| `github/codeql-action/*` (`init`, `autobuild`, `analyze`, `upload-sarif`) | v4.39.1 | `9b3254085494b55a0f7c4c2b3d2d52cc712ce370` | `codeql.yml`; `upload-sarif` in `ci.yml` |
| `pypa/gh-action-pypi-publish` | v1.14.3 | `7245906fb61fa33a13554f9d5fb856fe61758086` | `publish.yml` |
| `conda-incubator/setup-miniconda` | v4.1.0 | `be893c923ea9cf1cf7cd510fbdde27c7e18cbdcb` | `ci.yml` |
| `docker/setup-buildx-action` | v4.4.1 | `f87e5991a6d7451dcb8d9637bfbc97413f497069` | `publish-image.yml` |
| `docker/login-action` | v4.7.0 | `81c6f93d4bc991e510b342a00b419b3bb4efc5c6` | `publish-image.yml` |
| `docker/build-push-action` | v7.4.0 | `c3c9e263c25d99ce0380d002d59b67737d91b0dc` | `publish-image.yml` |
| `docker/metadata-action` | v6.2.0 | `dc802804100637a589fabce1cb79ff13a1411302` | `publish-image.yml` |
| `anthropics/claude-code-action` | v1.0.255 | `e4a6faad497d33dbc7144135c75e41f82f0bf6c0` | `claude.yml` |

Dependabot's `github-actions` updates move these pins (see [Dependabot Configuration](#dependabot-configuration)). After one lands, re-derive the table from `grep -rn "uses:" .github/workflows/`.

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

| Environment | Wait Timer | Reviewer Required | Deployments From | Purpose |
|-------------|-----------|-------------------|------------------|---------|
| `testpypi` | None | No | Tags `v*`, `rc*`, `hf*`, `juniper-*-v*`, `juniper-*-rc*`, `juniper-*-hf*` | TestPyPI publishing |
| `pypi` | 5 minutes | Yes (the owner) | The same six tag patterns | Production PyPI publishing |

Read on 2026-10-29 with `gh api repos/pcalnon/juniper-data/environments` and each environment's `deployment-branch-policies`. `testpypi`'s only protection rule is the tag policy: no timer and no approval hold the TestPyPI job once a release whose tag matches is published. The one wait in that job is the `sleep 30` in `publish.yml` before the install check, for index lag. The repository also has `copilot` and `dockerhub` environments; no workflow here names either.

---

## Secrets Reference

| Secret | Used By | Purpose |
|--------|---------|---------|
| `CODECOV_TOKEN` | ci.yml (unit-tests, Python 3.14 leg) | Codecov.io upload token. **Set** as an Actions secret since 2026-10-29 (22:15:30 UTC). The Dependabot secret store does not hold it, so a run that Dependabot triggers still uploads tokenless and Codecov decides whether to take it. Before 2026-10-29 every upload was tokenless: Codecov rejected each one through 2026-10-27 22:51 UTC with `Token required - not valid tokenless upload`, and accepted them from 23:47 UTC. `fail_ci_if_error: false` keeps the step green either way, so it gates nothing |
| `GITHUB_TOKEN` | ci.yml (security); publish-image.yml | Built-in. Gitleaks uses it in ci.yml, and `docker/login-action` uses it to log in to `ghcr.io` in publish-image.yml |
| `SLACK_WEBHOOK_URL` | main-verify.yml (`Notify on Failure` job); pr-budget-alarm.yml | Slack incoming-webhook URL for two notifications: a failed post-merge verification of `main`, and an open-PR budget breach. **Not set** (Actions secrets read 2026-10-29). Without it, main-verify.yml's step logs that it skipped, and pr-budget-alarm.yml's step writes a `::warning::` annotation instead. Both steps are `continue-on-error: true`, so a failed post never fails a run |
| `CROSS_REPO_DISPATCH_TOKEN` | lockfile-update.yml; notify-consumers.yml | PAT. The lockfile lane uses it to author the signed lockfile commit so CI re-triggers: Dependabot-triggered runs read the Dependabot secret store, where an empty token is a green skip, and a non-Dependabot run without the token fails. Consumer notification uses it to `repository_dispatch` into another repo (`GITHUB_TOKEN` cannot) and to list that repo's Actions runs |
| `ANTHROPIC_API_KEY` | claude.yml | Passed as the action's `anthropic_api_key`. A repository secret: juniper-data is user-owned, so no organization secret can reach it |

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
| Grouping | Minor + patch updates together (`python-minor`, patterns `*`) |

The pip group rewrites committed requirement files, including `conf/requirements.txt`, `conf/requirements-ORIG.txt`, and `conf/requirements_ci.txt`. `pyproject.toml` is unchanged when its `>=` floors already accept the new releases. `lockfile-update.yml` still compiles `requirements.lock` with `--upgrade` and the `api`, `observability`, `mnist`, and `equities` extras on that push. The lockfile pin set can differ from the freeze. `lockfile-check` uses `--constraint requirements.lock` and stays green while the lock still satisfies `pyproject.toml`.

Review steps: [Dependency Update Workflow](../../notes/DEPENDENCY_UPDATE_WORKFLOW.md).

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
| `juniper-coverage-gap-map` | CI's per-file coverage gate, in each `unit-tests` leg. With `--enforce` it fails when a file's statement coverage is below 90% or a sub-module's (directory's) pooled statement coverage is below 95%, the tool's defaults. Package `juniper-ci-tools>=0.9.0,<0.10.0`. See [Coverage Gates](../REFERENCE.md#coverage-gates) | `juniper-coverage-gap-map --coverage-json reports/coverage.json --enforce` |
| `scripts/check_module_coverage.py` | The pre-push `coverage-check` hook. Fails below 80% aggregate (`COVERAGE_FAIL_UNDER`) or when a test file is measured; a module below 85% is printed as `WARN` and does not fail. No CI job runs it | `python scripts/check_module_coverage.py [--run-tests]` |
| `juniper-check-doc-links` | Markdown link validation. Installed from `juniper-doc-tools>=0.1.0,<0.2.0` in the `docs` job and as a local pre-commit hook; there is no `scripts/check_doc_links.py` | `juniper-check-doc-links --exclude templates ... --cross-repo skip` (the `docs` job's exclude list) |
| `juniper-generate-dep-docs` | `dependency-docs` capture of `conf/requirements_ci.txt` and `conf/conda_environment_ci.yaml`. Pip body is `python -m pip list --format=freeze`. Conda body is the `dependencies:` block of `conda env export --no-builds` (stops before `prefix:` / `variables:`). Package `juniper-ci-tools>=0.9.0,<0.10.0`. The job uploads the capture and does not commit it; Dependabot may still edit the committed freeze. | CI: `juniper-generate-dep-docs` (no arguments, login shell). Local headers: `--pip-header PIP_DEPENDENCY_FILE_HEADER.md --conda-header CONDA_DEPENDENCY_FILE_HEADER.md` |

The `dependency-docs` job runs that command after `conda-incubator/setup-miniconda` (`miniforge-version: latest`, `auto-activate-base: true`, `python-version` = `PYTHON_TEST_VERSION`) in `shell: bash -l {0}`. No-argument mode looks for `notes/JUNIPER_2026-03-11_JUNIPER-ML_PIP-DEPENDENCY-FILE-HEADER.md` and `notes/JUNIPER_2026-03-15_JUNIPER-ML_CONDA-DEPENDENCY-FILE-HEADER.md`. This repo's templates are `notes/PIP_DEPENDENCY_FILE_HEADER.md` and `notes/CONDA_DEPENDENCY_FILE_HEADER.md`, so the upload uses the two-line fallback header. The conda template ends with `dependencies:`; the fallback does not, and the uploaded conda document is a YAML list. If `conda` is absent from `PATH`, the yaml is skipped and the command still exits 0. A YAML parse failure exits 1. Timestamped backups of the previous files are included in the 90-day artifact. The job does not commit the files. Full steps: [CI/CD manual](CICD_MANUAL.md#job-dependency-docs).

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

**Last Updated:** October 8, 2026
**Version:** 0.4.3
**Maintainer:** Paul Calnon
