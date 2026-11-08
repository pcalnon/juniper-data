# CI/CD Manual

## Comprehensive CI/CD Pipeline Guide for juniper-data

**Version:** 0.4.5
**Status:** Active
**Last Updated:** November 8, 2026
**Project:** Juniper - Dataset Generation Service

---

## Table of Contents

1. [Introduction](#introduction)
2. [Pipeline Overview](#pipeline-overview)
3. [GitHub Actions Workflows](#github-actions-workflows)
   - [ci.yml -- Main CI Pipeline](#ciyml----main-ci-pipeline)
   - [publish.yml -- PyPI Publishing](#publishyml----pypi-publishing)
   - [notify-consumers.yml -- Consumer release notification](#notify-consumersyml----consumer-release-notification)
   - [lockfile-update.yml -- Dependency Lockfile Auto-Update](#lockfile-updateyml----dependency-lockfile-auto-update)
   - [publish-image.yml -- Container Image](#publish-imageyml----container-image)
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
- **Pre-push hooks**: Coverage check (fails below 80% aggregate; lists modules below 85% without failing)
- **GitHub Actions CI**: Automated testing across Python 3.12/3.13/3.14 on Ubuntu and 3.12 on macOS, coverage gates (80% aggregate, 90% statement coverage per file, 95% pooled per sub-module), security scanning, build verification, documentation validation
- **GitHub Actions publishing**: Two-stage PyPI publishing (TestPyPI then production PyPI), then a consumer dispatch that waits for a run to start
- **Dependabot**: Automated dependency updates with lockfile synchronization
- **CodeQL**: Weekly semantic code analysis
- **Claude Code**: `@claude` on issue comments, pull-request review comments, submitted reviews, and newly opened issues

---

## Pipeline Overview

### Job Dependency Graph

```bash
pre-commit ──┬─→ unit-tests ──┬─→ build ──┬─→ dependency-docs ──┐
             │                │           └─→ docker-build ─────┤
             │                └─→ integration-tests ────────────┤
             └─→ security ──────────────────────────────────────┤
docs ───────────────────────────────────────────────────────────┤
lockfile-check ─────────────────────────────────────────────────┘
                                                                │
                                                                └─→ required-checks ──→ notify

Outside the Quality Gate, with no dependencies:
  async-route-audit   memory-budget (pull requests)   slow-tests (schedule, dispatch)
```

### Trigger Matrix

| Workflow              | Push to main/develop | Push to feature/fix | Pull Requests | Schedule            | Release | Manual |
|-----------------------|----------------------|---------------------|---------------|---------------------|---------|--------|
| `ci.yml`              | Yes                  | No (removed 2026-08-13) | Yes       | Daily 6 AM UTC      | No      | Yes    |
| `publish.yml`         | No                   | No                  | No            | No                  | Yes     | No     |
| `notify-consumers.yml` | No                  | No                  | No            | No                  | via `publish.yml` | Yes |
| `lockfile-update.yml` | No                   | `dependabot/pip/**` by `dependabot[bot]` | `pyproject.toml` paths, same repo, not `release/**` | No | No | No |
| `codeql.yml`          | Yes                  | No                  | Yes (to main) | Weekly Mon 6 AM UTC | No      | No     |
| `publish-image.yml`   | No                   | No                  | Image inputs    | No                  | `v*`    | Yes    |

`claude.yml` is omitted from this matrix. It runs on `issue_comment` (`created`), `pull_request_review_comment` (`created`), `pull_request_review` (`submitted`), and `issues` (`opened`, `assigned`). See [claude.yml](#claudeyml----claude-code).

---

## GitHub Actions Workflows

### ci.yml -- Main CI Pipeline

The primary CI workflow has 14 jobs, with concurrency control (`ci-${{ github.ref }}`, cancel-in-progress). Not every job runs on every event: each job's condition is below and in the [jobs table](CICD_REFERENCE.md#ciyml-jobs).

#### Job: `pre-commit`

Multi-version code quality validation across Python 3.12, 3.13, and 3.14 with `fail-fast: false`.

- Runs all pre-commit hooks: ruff, mypy, bandit, yamllint, file checks
- Caches `~/.cache/pre-commit` keyed on `.pre-commit-config.yaml` hash
- First job to run; blocks `unit-tests` and `security`

#### Job: `docs`

Required context name: **Documentation Links**. Python 3.12, no `needs:`, on every `ci.yml` event. The job runs four commands. The first failure turns the job red. Pre-commit runs only the link check, with the same arguments.

1. **Link check.** `juniper-check-doc-links` from `juniper-doc-tools>=0.1.0,<0.2.0`, with `--cross-repo skip`. Excludes `templates`, `history`, `legacy`, `pull_requests`, `releases`, `analysis`, `fixes`, `development`, `tmp`, and `CHANGELOG.md`. The local hook installs `juniper-doc-tools>=0.1.1,<0.2.0` and passes that same list.

2. **Workflow script paths.** `juniper-lint-workflow-paths` with no arguments, from `juniper-ci-tools>=0.9.0,<0.10.0`. The 0.9.1 release leaves this lint unchanged. It parses every `*.yml` and `*.yaml` under `.github/workflows/` and scans every YAML string for a token that contains a `/` and ends in `.py`, `.sh`, or `.bash`. A YAML comment is not a string, so a comment that names a deleted script does not fail. `python -m` is not a path. A token is skipped when it has no `/`, starts with `/` or `-`, contains `${` or `$(`, or starts with one of `juniper-canopy/`, `juniper-cascor/`, `juniper-cascor-client/`, `juniper-cascor-worker/`, `juniper-data/`, `juniper-data-client/`, `juniper-deploy/`, `juniper-ml/`. A path is missing only when it is absent both from the step's working directory and from the repo root. Working directory, highest precedence first: the step's `working-directory`, the job's `defaults.run.working-directory`, the workflow's `defaults.run.working-directory`, then the repo root. A workflow that does not parse as YAML contributes no paths (yamllint owns that failure). Exit 0 when nothing is missing, 1 when a path is missing, 2 when the repo root or the workflows directory cannot be found. When the report names a working directory, leave the path in the workflow as written: the step already runs in that directory, and prefixing the path applies the directory twice.

3. **AGENTS.md version.** `juniper-lint-agents-md-version` with no arguments. The one line `**Version**: <value>` must equal `pyproject.toml`'s `[project].version` as a string (both were `0.17.1` on 2026-11-07). The walk-up looks for a directory that contains both files. A file with no such line exits 0 (`SKIP`). Two or more such lines exit 2, and the search is the whole file, so a second line after the header rule still fails. A `pyproject.toml` with no `[project].version` exits 2. Drift exits 1.

4. **AGENTS.md header.** `juniper-lint-agents-md-header` with no arguments. The header is everything before the first line that is `---` or starts with `## `. It must contain these six bullets in this relative order, each with a non-empty value. Extra bullets may sit between them:

   - `**Project**:`
   - `**Repository**:`
   - `**Author**:`
   - `**License**:`
   - `**Version**:`
   - `**Last Updated**:` matching `YYYY-MM-DD` (that pattern; the lint does not check that the day exists)

   A missing field is named in the report, which also prints the required fields that were present, because that list no longer matches the six. Exit 0 when the header conforms, 1 on a schema miss, 2 when `AGENTS.md` or a repo root (`AGENTS.md` beside `.github/`) cannot be found. This lint does not read `pyproject.toml`. Deleting `**Version**:` to take the version lint's skip still fails here.

The header lint accepts any `**Last Updated**:` value that matches `YYYY-MM-DD`. It does not compare that date with today, and it does not edit the file. A pull request that changes `AGENTS.md` is also checked by `agents-md-touch-up.yml` (workflow name **AGENTS.md Date Check**, job **Verify AGENTS.md Last Updated**). That workflow is not a required context, and it never commits. It warns and passes when the field is absent, fails a value that is not `YYYY-MM-DD` or is later than today's UTC date, passes when the value already equals today's UTC date, and otherwise passes only when this pull request's diff adds a `**Last Updated**:` line. The `juniper-ci-tools` 0.9.1 module text still describes that workflow as rewriting the date on every push. This repository's workflow verifies.

Reproduce the three lints from a checkout:

```bash
pip install "juniper-ci-tools>=0.9.0,<0.10.0"
juniper-lint-workflow-paths
juniper-lint-agents-md-version
juniper-lint-agents-md-header
```

The job calls those console scripts. `scripts/check_doc_links.py`, `util/test_workflow_script_paths.py`, and `util/test_agents_md_version_drift.py` are not in this repository.

#### Job: `async-route-audit`

`ruff check --select ASYNC` over `juniper_data/` (Python 3.12), the CI half of the pre-commit async-route audit.

- Runs independently (no dependencies); a required context outside the Quality Gate

#### Job: `unit-tests`

Core test execution with coverage enforcement across Python 3.12, 3.13, and 3.14 on Ubuntu, and 3.12 on macOS. All four legs are required contexts; none is allowed to fail.

- **Depends on**: `pre-commit`
- **Markers**: `-m "unit and not slow"`
- **Timeout**: 60 seconds per test, `--maxfail=5`
- **Coverage**: `python -m coverage report --fail-under=80` (aggregate, branches counted), then `juniper-coverage-gap-map --coverage-json reports/coverage.json --enforce`, which fails the leg when a file's statement coverage is below 90% or a sub-module's (directory's) pooled statement coverage is below 95%. See [Coverage Gates](../REFERENCE.md#coverage-gates)
- **Installs juniper-data-client** from `main` branch via git+https
- **Reports**: JUnit XML, HTML coverage, Cobertura XML, JSON coverage
- **Codecov upload**: Python 3.14 leg only, with `token: ${{ secrets.CODECOV_TOKEN }}`. The `CODECOV_TOKEN` Actions secret exists since 2026-10-29. The Dependabot secret store does not hold it, so a run that Dependabot triggers uploads tokenless and Codecov decides whether to take it. `fail_ci_if_error: false` keeps the step green either way, so it gates nothing. See the notes under [`REFERENCE.md` § GitHub Actions Workflows](../REFERENCE.md#github-actions-workflows)

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

The checked-in copies under `conf/` are snapshots from the previous generator. The next CI run uploads new artifacts and leaves those committed files in place. Dependabot's `python-minor` group still rewrites the `==` pins in the committed `conf/requirements_ci.txt`, so that file is not the CI capture either.

#### Job: `integration-tests`

Full workflow integration tests.

- **Depends on**: `unit-tests`
- **Condition**: Only on PRs or main/develop branches
- **Markers**: `-m "integration and not slow"`
- **Timeout**: 120 seconds per test, `--maxfail=3`

#### Job: `security`

Multi-tool security scanning. Depends on `pre-commit`. Python 3.12.

1. **Gitleaks**: Secret detection. The ruleset, the engine pin, and which commits each event scans are in [Gitleaks ruleset](#gitleaks-ruleset).
2. **Bandit SAST**: Static security analysis with SARIF report
   - Blocking check: fails on medium+ severity and confidence
   - SARIF uploaded to GitHub Security tab
3. **pip-audit**: Dependency vulnerability scanning
   - Filters out `juniper-data` packages (self-referencing)
   - Runs in `--strict` mode
4. **Gitleaks config self-test**: `pytest juniper_data/tests/unit/test_gitleaks_config.py` with `GITLEAKS_PARITY_REQUIRED=1`, after pip-audit. Skipped on `repository_dispatch`.

#### Gitleaks ruleset

Secret detection inside the `security` job. The weekly `security-scan.yml` workflow does not run Gitleaks.

The step is `gitleaks/gitleaks-action` at `e0c47f4f8be36e29cdc102c57e68cb5cbf0e8d1e` (v3.0.0). `GITLEAKS_VERSION` is `8.24.3`, which is also the version that action uses when the variable is unset (`src/index.js` at that SHA). The variable is set so a later action SHA cannot change the engine on its own. `juniper_data/tests/unit/test_gitleaks_config.py` reads the pin from `ci.yml`.

The action accepts `push`, `pull_request`, `workflow_dispatch`, and `schedule`, and exits on any other event. `ci.yml` also runs on `repository_dispatch`, so this step and the self-test are both skipped when `github.event_name` is `repository_dispatch`.

The job checks out with `fetch-depth: 0`. In `actions/checkout` v7.0.1 that input fetches all history for all branches and tags. The range the action then passes to gitleaks depends on the event (`src/index.js` and `src/gitleaks.js` at the pinned SHA; gitleaks 8.24.3 `sources/git.go` when `--log-opts` is absent):

| Event | What is scanned |
|-------|-----------------|
| `push` with an empty `commits` array | Nothing. The action exits 0. |
| `push` of one commit | `git log -1` on the checked-out tip. |
| `push` of several commits | `--no-merges --first-parent <first>^..<last>`, where `<first>` and `<last>` are the first and last SHAs in the push payload. |
| `pull_request` | The same `--no-merges --first-parent` range across the pull request's commit list, first SHA through last. A finding is posted as a review comment unless `GITLEAKS_ENABLE_COMMENTS` is `false`. The fingerprint in that comment is `<commit>:<file>:<rule>:<line>`. |
| `schedule`, `workflow_dispatch` | No `--log-opts`. Gitleaks 8.24.3 then runs `git log -p -U0 --full-history --all`: every commit in that full checkout. |

`BASE_REF`, when set, replaces the start commit of a `push` or `pull_request` range.

The action passes no `--config` and sets neither config variable. Precedence in 8.24.3 (`cmd/root.go`) is `--config`, then `GITLEAKS_CONFIG` (a path), then `GITLEAKS_CONFIG_TOML` (the TOML text itself), then `<source>/.gitleaks.toml`. The real-engine tests unset both variables so they exercise that last file. A missing `.gitleaks.toml` loads the built-in ruleset. A file that is present replaces it.

The action looks up the repository owner. An organization owner has to set `GITLEAKS_LICENSE` or the step exits 1. A user owner skips that check. This workflow does not set the secret; juniper-data is user-owned.

**Default ruleset.** `[extend] useDefault = true` is what loads gitleaks' built-in rules. A v8 file with no `[extend]` table loads only the `[[rules]]` written in it. Until 2026-10-25 this file had an `[allowlist]` and no `[[rules]]`, so the step ran with zero rules and a planted token could not fail it. The file now also has a `generic-api-key` entry with no regex of its own. Removing `[extend]` while that entry remains makes gitleaks refuse the config (`generic-api-key: both |regex| and |path| are empty, this rule will have no effect`, from `config/rule.go` in 8.24.3) and the step fails. The zero-rule pass returns when the file is still present and both the table and that rule are gone. The real-engine test builds a `github-pat`-shaped value at runtime and requires that rule to fire through the auto-discovered config.

**Allowlists stay narrow.** In gitleaks 8.24.3 the global `[allowlist]` is OR-only (`condition` is not read there), so a path on it exempts every rule in that file. A top-level `[[allowlists]]` table and a `targetRules` key are both dropped by the loader. The global list in this repo names one historical commit, `4bd37a6c7057e7380cd4470fcc717a76289fa193`, an ARC-AGI public identifier that was later removed.

Conda build strings are the class that comes back on every environment refresh. A spec line is `<package>=<version>=<build>`. `generic-api-key` treats `api` or `key` inside the package name as a key name, and the secret it captures is `<version>=<build>`. A commit-scoped ignore entry cannot name the commit a future refresh will create. `.gitleaks.toml` therefore adds one `[[rules]]` entry with `id = "generic-api-key"` and no `regex` of its own. `extend()` merges that entry into the default rule and keeps the default regex, entropy, keywords, and allowlists. The added allowlist has `condition = "AND"`, so a finding is dropped only when both halves match:

- the path matches one of the two regexes in the file: `conf/conda_environment*.yaml` or `.yml`, or `conf/requirements-conda.txt`
- the whole secret matches the file's regex: a version that starts with a digit, `=`, then a conda build string of the form `h<7 hex>_<n>`, `pyh<7 hex>_<n>`, or `py<digits>h<7 hex>_<n>`

A secret of any other shape in those files still fires. The same spec line outside those paths still fires. Every other rule, including `github-pat`, still fires inside a conda file. The real-engine cases in `test_gitleaks_config.py` pin all three.

**Ignore file.** Each `.gitleaksignore` entry is `<40-hex commit>:<file>:<rule>:<line>`. The nearest non-entry line above a group is a comment that names the class. A file-scoped entry (`<file>:<rule>:<line>`) would also hide a later secret on that line, and the test rejects it. A pull-request fingerprint from the action already includes the commit; keep that prefix when you append the line.

**Self-test.** pip-audit's `.[all]` install is what puts pytest on the runner. After that install, on every event except `repository_dispatch`, the job runs:

```bash
GITLEAKS_PARITY_REQUIRED=1 python -m pytest juniper_data/tests/unit/test_gitleaks_config.py -v
```

The binary is the one the action placed on `PATH`. With `GITLEAKS_PARITY_REQUIRED=1`, a missing binary fails the step. The unit-test job runs the same file and skips the real-engine cases when `gitleaks` is absent.

Local commands, both of which discover the two files at the repo root:

```bash
gitleaks git --redact .
gitleaks dir --redact .
```

`git` scans history. `dir` scans the working tree, including gitignored files. The action still invokes the deprecated `detect` subcommand; on 8.24.3 that subcommand uses the same `git log` path as `git`.

#### Job: `lockfile-check`

Validates `requirements.lock` freshness.

- Runs independently (no dependencies)
- Uses `uv pip compile` to regenerate lockfile
- Diffs against committed `requirements.lock`
- Prints remediation command on failure

#### Job: `docker-build`

Builds the image, starts a container, and checks the package import and `GET /v1/health`.

- **Depends on**: `build`
- **Condition**: Only on PRs or main/develop branches

#### Job: `memory-budget`

Runs `util/memory_budget_check.py` against the base branch: each always-loaded memory file that `conf/memory_budget.json` governs (`AGENTS.md`) must stay under its character ceiling.

- **Condition**: pull requests only (its `merge_group` arm cannot fire: the workflow declares no `merge_group` trigger)
- Runs independently (no dependencies); a required context outside the Quality Gate

#### Job: `required-checks`

Quality gate aggregator. Runs `if: always()` and checks status of all required jobs.

**Required to pass** (`success` only): pre-commit, unit-tests, build, docs, lockfile-check

**Conditional** (`success` or `skipped` passes; `failure` or `cancelled` fails the gate): dependency-docs, security, integration-tests, docker-build

#### Job: `slow-tests`

Long-running test execution.

- **Trigger**: Schedule (daily 6 AM UTC) or manual dispatch only
- **Markers**: `-m "slow"`
- **Timeout**: 600 seconds (10 minutes) per test
- **Verdict**: pytest's exit status. A failing slow test fails the job, and so does a run that selects no test (pytest exit 5, with an `::error` annotation naming it): a lane that runs nothing must not pass. `Slow Tests` is a required context; a pull request reports it skipped, which counts as passing

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
5. Install from TestPyPI only (`--index-url https://test.pypi.org/simple/`, `--no-deps`, `juniper-data==<version>` with one leading `v` stripped from the tag) and print the installed version from its metadata. The print does not compare that version with the tag; the pip pin does. `import juniper_data` would fail a good publish, because `dotenv` is imported at import time and `--no-deps` does not install it. A second index does not fetch dependencies under `--no-deps`; it only widens which `juniper-data` pip will accept. A non-zero `pip` or `python` fails the step. See [TestPyPI install check](../REFERENCE.md#testpypi-install-check).

**Environment**: `testpypi` (no wait timer and no reviewer; deployments only from a release tag matching `v*`, `rc*`, `hf*`, `juniper-*-v*`, `juniper-*-rc*` or `juniper-*-hf*`)

#### Stage 2: Production PyPI

Builds and checks the package again, then publishes to production PyPI. There is no install check.

**Environment**: `pypi` (5-minute wait timer + required reviewer approval)

Both stages use SHA-pinned actions. Neither sets the publish action's `attestations` input, so the pinned `pypa/gh-action-pypi-publish` (v1.14.3) uses its default, `true`: under Trusted Publishing each upload carries PEP 740 attestations. Only the TestPyPI install check reads the version from the release tag (with a leading `v` stripped); the build takes it from `pyproject.toml`.

The `notify-consumers` job runs only after `pypi` succeeds. A failed notification does not unpublish the release. See the next section.

---

### notify-consumers.yml -- Consumer release notification

Tells repos that install `juniper-data` from PyPI that a release is on PyPI. Today the matrix is `juniper-recurrence` and `juniper-canopy`.

- **From a release:** `publish.yml` calls this workflow with the release tag after the production PyPI job. A leading `v` is stripped. The value must be `X.Y.Z`.
- **Manual:** `workflow_dispatch` with a `version` input re-sends for a version already on PyPI. It does not publish.
- **Auth:** `CROSS_REPO_DISPATCH_TOKEN`. `GITHUB_TOKEN` cannot dispatch into another repository. An empty token fails before the POST. `curl --fail-with-body` makes a 403 or 404 fail the step.
- **Event:** `juniper-data-published`, payload `{source: "juniper-data", version, sha}`. `sha` is `github.sha` of this run.
- **204 is not delivery.** GitHub returns 204 whether or not a workflow listens. The next step polls up to 12 times, 10 seconds apart, for a `repository_dispatch` run created at or after 30 seconds before the POST whose `display_title` is `juniper-data-published`. The listener must not set `run-name:`.
- **Two failures.** No successful listing means the step cannot tell whether a run started (check the repo name, and Actions: Read if the consumer is private). A successful listing with no matching run means no listener started. If some listings failed, that second error says how many.

Full contract: [Consumer release notification](../REFERENCE.md#consumer-release-notification).

---

### lockfile-update.yml -- Dependency Lockfile Auto-Update

Regenerates `requirements.lock` from `pyproject.toml` and, when the file changes, pushes a GitHub-signed commit.

- **Push trigger**: `dependabot/pip/**` when `github.actor == dependabot[bot]`. A grouped pip push qualifies even when it edits only `conf/requirements*.txt`.
- **Pull request trigger**: `pyproject.toml` changed, the head repo is this repo, and the branch is not `release/**`. Forks are skipped.
- **Token gate**: `CROSS_REPO_DISPATCH_TOKEN`. Dependabot-triggered runs read the Dependabot secret store. An empty token there skips the rest of the job (green notice). A non-Dependabot run without the token fails the job. Repository Actions secrets are a different store.
- **Compile**: `uv pip compile pyproject.toml --extra api --extra observability --extra mnist --extra equities --upgrade -o requirements.lock`
- **Commit**: `[dependabot skip] Update requirements.lock` via `createCommitOnBranch` when `git diff` shows the regenerated `requirements.lock` changed. The PAT is the author, so the push re-triggers CI.

The lockfile is the Docker pin set for those four extras. `conf/requirements.txt` and `conf/requirements-ORIG.txt` are the local setup snapshots (`util/setup_environment.bash`); keep that pair on the same floors. `conf/requirements_ci.txt` is the committed freeze the `python-minor` group rewrites. The lockfile and the freeze can name different versions, and `--upgrade` can add a transitive the freeze does not list.

`lockfile-check` compiles with `--constraint requirements.lock` and compares pin lines. It fails when the lock no longer satisfies `pyproject.toml`. A newer release inside the current ranges leaves it green, which is why a conf-only Dependabot push can be freshness-green before the `--upgrade` commit lands.

Full review steps: [Dependency Update Workflow](../../notes/DEPENDENCY_UPDATE_WORKFLOW.md).

---

### publish-image.yml -- Container Image

Publishes `ghcr.io/pcalnon/juniper-data` as one manifest for linux/amd64 and linux/arm64. It is not a required check, and it does not run as part of `publish.yml`.

- **Build-only** (pull request touching image inputs, or a manual run with `push` left false): load `data-smoke:<arch>`, no registry login. Smoke-test the import, assert CPU-only (`EXPECT_TORCH=absent`), scan for credentials, then run the serve-and-version check against `pyproject.toml`'s version.
- **Publish** (a `v*` release, or a manual run with `push` true): push each arch by digest with no tag, run the same content checks plus the serve check against that digest, then let the merge job write tags. A release's expected version is the tag minus the leading `v`, and the job stops when that string disagrees with `pyproject.toml`.

The serve check is `util/check_image_serves.py --dist juniper-data --module juniper_data --port 8100`. It starts the image `CMD` and requires `GET /v1/health` to return 200 with a `version` equal to the installed package metadata and to `juniper_data.__version__`. Full contract, including what a failed check leaves in GHCR: [Image serve-and-version gate](../REFERENCE.md#image-serve-and-version-gate).

The merge job (`Publish manifest`) is the only tag writer. It refuses an empty tag list, requires the published list to contain linux `amd64` and `arm64` (a non-linux entry does not count), and on a release requires an `X.Y.Z` tag so `:latest` alone cannot go green. Before the CPU-only pull of the tag, it resolves each pushed digest to the one linux image inside it. With provenance attestations on, that digest is an OCI index, and the published list names the image manifests; comparing the two directly fails the job. A step in this job names no `shell:`, so the runner uses `bash -e`. See [Publish manifest gates](../REFERENCE.md#publish-manifest-gates).

---

### codeql.yml -- Code Quality Analysis

GitHub CodeQL semantic analysis for Python.

- **Trigger**: Push to main/develop, PRs to main, weekly Monday 6 AM UTC
- Runs `security-and-quality` query suite
- Results appear in GitHub Security tab

---

### claude.yml -- Claude Code

`.github/workflows/claude.yml` replies when someone writes `@claude` in a place the workflow watches. The job checks out with `fetch-depth: 1` and passes a single action input, `anthropic_api_key`, from `secrets.ANTHROPIC_API_KEY`. That is a repository secret: juniper-data is user-owned, so no organization secret can reach it.

Write the phrase in one of these:

- the body of a new issue comment, including a comment on a pull request (`issue_comment`)
- the body of a pull-request review comment
- the body of a submitted pull-request review
- the title or body of an issue at the moment it is opened

The workflow `if` uses `contains(..., '@claude')`. GitHub's `contains` is not case sensitive, so `@Claude` starts the job too. It is also a substring test, so `email@claude.com` and `@claudefoo` start the job as well.

When the job starts, the action checks the phrase again. The default phrase is `@claude`. The match is case-insensitive, and the phrase must sit at the start of the field or after whitespace, and must end at whitespace or one of `. , ! ? ; :`. A miss logs `No trigger found, skipping remaining steps` and the job succeeds with no reply. That is what an `issues` `assigned` delivery does: the workflow starts because the title or body already contains the substring, and `assignee_trigger` is unset so the action does not treat the assignment as a request. The action reads the issue title and body only on `opened`.

The action checks write access before the phrase check. A commenter who is neither `admin` nor `write` fails with `Actor does not have write permissions to the repository`. A login ending in `[bot]` passes that lookup. After a real phrase match, a non-user actor fails with `Workflow initiated by non-human actor` because `allowed_bots` is empty.

A matching run with an empty `ANTHROPIC_API_KEY` fails later:

```text
Environment variable validation failed:
  - Either ANTHROPIC_API_KEY, CLAUDE_CODE_OAUTH_TOKEN, or workload identity federation (ANTHROPIC_FEDERATION_RULE_ID and ANTHROPIC_ORGANIZATION_ID) is required when using direct Anthropic API.
```

This workflow does not set the OAuth token or the federation inputs, and it leaves `use_bedrock`, `use_vertex`, and `use_foundry` at their default `false`.

A phrase in a pull-request title or body never schedules the job: `on:` does not include `pull_request`. The action's default `label_trigger` is `claude`, and this workflow does not subscribe to `labeled`. The action accepts an edited review body; this workflow subscribes only to `submitted`.

The action version is the `# vX.Y.Z` comment on the `uses:` line. Dependabot groups only `github/codeql-action*`, so a bump of `anthropics/claude-code-action` arrives as its own pull request. The behaviour below was read from v1.0.265, the pin on 2026-11-07. After a bump, re-read `src/github/operations/branch.ts`, `src/modes/tag/index.ts` and `src/github/data/fetcher.ts` at the new SHA ([Open pull request checkout](CICD_REFERENCE.md#open-pull-request-checkout)).

On an open pull request that release checks out the head SHA the action loaded (`headRefOid`). The newest repository-activity timestamp on the head branch must be strictly earlier than the mention: `comment.created_at` for a comment, `review.submitted_at` for a submitted review. A push at or after that time fails the job, and the tracking comment is replaced with:

```text
PR #<n> was updated after the trigger (<pushedAt> >= <triggerTime>). Mention Claude again to run on the latest version.
```

Push the commit you want, then mention `@claude`, and leave the branch still until the job checks out. A fork pull request is checked out as the local branch `pr-<number>`; the activity read uses the head repository. Checkout, activity lookup, and the other error strings: [Open pull request checkout](CICD_REFERENCE.md#open-pull-request-checkout).

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

Runs `scripts/check_module_coverage.py --run-tests` on the pre-push stage: the whole suite under coverage, then the checks. It blocks the push when aggregate coverage is below `COVERAGE_FAIL_UNDER` (default 80%) or a test file is measured. A module below 85% is printed as `WARN` and does not block it. A pass with those warnings prints `PASS: aggregate >= 80.0%; K of N modules below 85.0% (warned above, not enforced)`, not `PASS: All`. CI's per-file gate (90% statement coverage per file, 95% pooled per sub-module) is stricter and does not run here, so a push this hook allows can still fail CI. See [Coverage Gates](../REFERENCE.md#coverage-gates).

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

When Dependabot pushes to `dependabot/pip/**`, `lockfile-update.yml` compiles `requirements.lock` with `--upgrade` and commits when the file changes. That includes grouped updates that never touch `pyproject.toml`. The commit is skipped, green, when `CROSS_REPO_DISPATCH_TOKEN` is absent from the Dependabot secret store.

---

## Quality Gate

The `required-checks` job in ci.yml acts as the merge quality gate. All of these must pass:

| Check                                    | Required    | Failure Impact                                             |
|------------------------------------------|-------------|------------------------------------------------------------|
| pre-commit (code quality)                | Yes         | Blocks merge                                               |
| unit-tests (all Python versions)         | Yes         | Blocks merge                                               |
| build (package verification)             | Yes         | Blocks merge                                               |
| dependency-docs                          | Soft        | Failure or cancellation blocks, skip OK                    |
| security (gitleaks + bandit + pip-audit) | Soft        | Failure or cancellation blocks, skip OK                    |
| docs (link validation)                   | Yes         | Blocks merge                                               |
| lockfile-check                           | Yes         | Blocks merge                                               |
| integration-tests                        | Conditional | Failure or cancellation blocks, skip OK (feature branches) |
| docker-build (image build + smoke test)  | Conditional | Failure or cancellation blocks, skip OK (feature branches) |

A Soft or Conditional lane passes the gate only when its result is `success` or `skipped`. `skipped` is the pass for a lane that does not run on every event. `cancelled` fails the gate: a lane cut short by a newer push or a manual cancel never finished.

The Quality Gate is not the whole merge requirement. The `main` ruleset also requires contexts outside its `needs:`, among them **`Sequence Safety`**, which comes from its own workflow (`sequence-safety.yml`) and so cannot be named there. A green Quality Gate therefore does not mean a PR is mergeable. See [Sequence Safety (required check)](../REFERENCE.md#sequence-safety-required-check) for the ruleset query.

---

## Release Process

1. **Create a GitHub Release** with a tag matching the version (e.g., `v0.4.2`)
2. `publish.yml` triggers automatically
3. **TestPyPI stage**: Build, verify, publish, install-test (no environment wait; the job sleeps 30 seconds before the install check)
4. **PyPI stage**: Same process, requires manual reviewer approval
5. Approve the PyPI deployment via GitHub environment approval or `gh api`
6. **Notify consumers**: after PyPI succeeds, `notify-consumers.yml` dispatches `juniper-data-published` and waits until the consumer starts a run. A red notify job means the package is already on PyPI. The `pypi` job is the publish verdict.
7. The same release event starts `publish-image.yml` when the tag starts with `v`. That workflow is independent: PyPI success does not prove the image serves or that its tag matches the package. See [publish-image.yml](#publish-imageyml----container-image).

See [PyPI Publishing Procedure](../../../juniper-ml/notes/JUNIPER_2026-06-18_JUNIPER-ECOSYSTEM_PYPI-PUBLISH-PROCEDURE.md) for ecosystem-wide publishing lessons.

---

## Troubleshooting

**Pre-commit hooks not running**: Ensure you've run both `pre-commit install` and `pre-commit install --hook-type pre-push`.

**Ruff modifies files on commit**: This is by design. Ruff auto-fixes lint violations and reformats code. Stage the changes and commit again.

**CI fails but local passes**: Check Python version matrix. CI tests on 3.12, 3.13, and 3.14. Ensure your local environment matches.

**Lockfile check fails**: The committed pins no longer satisfy `pyproject.toml`. Refresh with the same extras the workflow uses:

```bash
uv pip compile pyproject.toml --extra api --extra observability --extra mnist --extra equities --upgrade -o requirements.lock
```

**Dependabot PR and `requirements.lock`**: A `python-minor` PR can edit only `conf/requirements*.txt` and still receive `[dependabot skip] Update requirements.lock`. That commit is the `--upgrade` resolution of `pyproject.toml`, so its pins can differ from `conf/requirements_ci.txt`. When the commit is missing, open the Update Lockfile run. A notice that `CROSS_REPO_DISPATCH_TOKEN` is unavailable means the PAT is missing from the Dependabot secret store; the job stays green and `lockfile-check` still enforces freshness. See [Dependency Update Workflow](../../notes/DEPENDENCY_UPDATE_WORKFLOW.md).

**Gitleaks stays green on a token you planted**: The step did not load the built-in ruleset, or the event's range did not include that commit. `.gitleaks.toml` has to keep `[extend] useDefault = true`. `GITLEAKS_CONFIG` (a path) and `GITLEAKS_CONFIG_TOML` (TOML text) are used instead of the file in the tree when either is set. A `push` whose payload has an empty `commits` array exits 0 without scanning. A `repository_dispatch` run skips the step. A present config with no `[extend]` table and no `[[rules]]` is the zero-rule pass; removing only `[extend]` from the current file fails the step, because the conda `generic-api-key` entry has no regex. See [Gitleaks ruleset](#gitleaks-ruleset).

**A conda environment refresh fails `generic-api-key`**: The new line sits outside the two path regexes, or the captured secret is not `<version>=<build>` in the three build-string shapes in [Gitleaks ruleset](#gitleaks-ruleset). Leave those paths off the global `[allowlist]`; a path there would exempt every rule in the file. A one-off historical finding is a commit-scoped `.gitleaksignore` line under a class comment. The AND allowlist is what covers the next refresh of a matching build string.

**Gitleaks config self-test errors at setup**: `GITLEAKS_PARITY_REQUIRED=1` is set and the fixture finds no binary: `GITLEAKS_BIN` is unset and no `gitleaks` is on `PATH`. Collection succeeds and the static tests pass. Each test that needs the binary reports `ERROR at setup` with `GITLEAKS_PARITY_REQUIRED=1 but no gitleaks binary`, and pytest exits 1, so the step fails. The action step is what installs the pinned 8.24.3 binary. The unit-test job is allowed to skip the real-engine cases when the binary is absent.

**CodeQL findings**: Review them in the GitHub Security tab. They are not part of the Quality Gate job (`codeql.yml` is its own workflow), but they do block the merge. The `main` ruleset (`juniper-data-rules`) has a `code_scanning` rule for CodeQL and Bandit that blocks a pull request on an alert of severity `error`, or a security alert of severity `high` or `critical` (thresholds `errors` and `high_or_higher`). `Analyze (python)` is a required check, and the ruleset requires review threads to be resolved, so an open code-scanning comment on the diff blocks too. Check the thresholds with `gh api repos/pcalnon/juniper-data/rulesets/14748749 --jq '.rules[] | select(.type == "code_scanning")'`.

**`@claude` produced no reply**: Put the phrase in an issue comment, a pull-request review comment, a submitted review body, or the title or body of an issue when it is opened. Case does not matter (`@Claude` works); a word boundary does. A green run that logs `No trigger found, skipping remaining steps` means the action's word-boundary check rejected the phrase, or the event was an assignment. See [claude.yml](#claudeyml----claude-code).

**`@claude` failed before a reply**: `Actor does not have write permissions to the repository` means the commenter lacks write. `Workflow initiated by non-human actor` means a bot; `allowed_bots` is unset. `Environment variable validation failed` naming `ANTHROPIC_API_KEY` means the secret was empty for this repo.

**`@claude` on a pull request stops with `was updated after the trigger`**: The head commit's push time is at or after the mention. The tracking comment is that error. Push the commit you want, mention `@claude` again, and leave the branch still until the job checks out. `Could not determine when <sha> was pushed` means the newest activity on the head branch did not end at that SHA. See [Open pull request checkout](CICD_REFERENCE.md#open-pull-request-checkout).

**TestPyPI publish fails**: Check that the release tag matches the version in `pyproject.toml`. Version is extracted by stripping one leading `v` from the tag. The install check uses only `https://test.pypi.org/simple/` and `--no-deps`. Do not add a second index. The following Python line prints installed metadata; it does not compare it with the tag, and `import juniper_data` is the wrong check (`dotenv` is not installed). See [TestPyPI install check](../REFERENCE.md#testpypi-install-check).

**Image merge job: `manifest list references` vs `verified build digests`**: The arch digest is an OCI index when provenance attestations are on. The published list names the linux image inside it. See [Publish manifest gates](../REFERENCE.md#publish-manifest-gates).

**Image merge job: `produced no X.Y.Z image tag`**: The release still applied `:latest`. The semver rules did not match the tag. `:latest` alone is not a successful versioned publish.

**Publish run red after PyPI succeeded**: The `notify-consumers` job failed. PyPI already has the release. A 204 from the dispatch API is only acceptance.

If the error says the consumer's runs could not be listed, check `CROSS_REPO_DISPATCH_TOKEN` and Actions: Read on a private consumer. If it says no `juniper-data-published` run started in about two minutes, the consumer workflow needs `repository_dispatch: types: [juniper-data-published]` and must not set `run-name:`. Re-send with `workflow_dispatch` once that is fixed; do not cut another release to retry.

See [Consumer release notification](../REFERENCE.md#consumer-release-notification).

**Image job fails the serve-and-version step**: The arch was already pushed by digest. No tag is written unless every arch exports its digest. Read the `::error::` from `util/check_image_serves.py`: metadata, `juniper_data.__version__`, and `/v1/health` `version` must all equal the expected `X.Y.Z`. On a release that value is the tag without `v`, and it must equal `pyproject.toml`. See [Image serve-and-version gate](../REFERENCE.md#image-serve-and-version-gate).

**Per-file coverage gate fails in CI**: The leg's log lists each file below 90% statement coverage and each sub-module below 95% pooled. Reproduce it with the three commands under [Coverage Gates](../REFERENCE.md#coverage-gates). `scripts/check_module_coverage.py` uses another threshold (85%, warning only) on another basis (branches counted), so it can pass where the gate fails.

**`Documentation Links` is red and pre-commit was green**: The hook runs `juniper-check-doc-links` only, so pre-commit never checks the workflow script paths, AGENTS.md version, or AGENTS.md header steps. When the failed step is the link check: locally, the hook runs only when the commit includes a `.md` file (`files: \.md$`). CI's `Pre-commit` job runs it on every push (`pre-commit run --all-files`), with the job's arguments, so a red link check beside a green `Pre-commit` check points at the two having resolved different `juniper-doc-tools` releases. The job installs `juniper-doc-tools>=0.1.0,<0.2.0` afresh, while the hook's environment is restored from `actions/cache`. Run `juniper-check-doc-links` with the job's arguments. Commands and exit codes: [Job: docs](#job-docs).

**The date check failed after `Documentation Links` passed**: `agents-md-touch-up.yml` is a different workflow. It runs when the pull request changes `AGENTS.md`, it is not a required context, and it never rewrites the date. Set `**Last Updated**:` to today's UTC date, or to another `YYYY-MM-DD` that this pull request's diff adds. A value that already equals today's UTC date passes without appearing in the diff.

**`dependency-docs` conda artifact is a YAML list**: the job calls `juniper-generate-dep-docs` with no arguments, so it does not load `notes/CONDA_DEPENDENCY_FILE_HEADER.md`. Pass `--conda-header CONDA_DEPENDENCY_FILE_HEADER.md` and `--pip-header PIP_DEPENDENCY_FILE_HEADER.md` when you want the committed header shape.

**`dependency-docs` succeeds and uploads no conda file**: `conda` was not on `PATH`. The tool warns and exits 0. The job's install and generate steps need `shell: bash -l {0}` after `setup-miniconda`. A non-login shell does not see the activated base.

---

## End of CI/CD Manual
