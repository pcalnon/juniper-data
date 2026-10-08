# Dependency Update Workflow — juniper-data

**Last Updated:** 2026-10-08
**Version:** 1.0.2
**Status:** Current

---

## Overview

This document describes how dependency updates flow through juniper-data, from Dependabot PR to merged dependency artifacts. The lockfile (`requirements.lock`) pins exact versions for Docker builds while `pyproject.toml` uses `>=` ranges for library compatibility. The `conf/requirements*.txt` files are pip environment snapshots; they are reviewed as dependency documentation, not as the runtime source of truth.

## Dependency Artifact Map

| File                             | Role                                                                                                           | Normal Update Path                                                                     |
|----------------------------------|----------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| `pyproject.toml`                 | Authoritative package metadata and optional dependency groups                                                  | Manual edits or Dependabot updates to root package requirements                        |
| `requirements.lock`              | Reproducible Docker/API runtime lockfile generated from `pyproject.toml` with `api`, `observability`, `mnist`, and `equities` extras | `uv pip compile pyproject.toml --extra api --extra observability --extra mnist --extra equities --upgrade -o requirements.lock` |
| `conf/requirements.txt`          | Legacy pip requirements used by `util/setup_environment.bash` during local Conda environment setup             | Maintained as a snapshot; Dependabot can update packages listed here                   |
| `conf/requirements-ORIG.txt`     | Baseline copy of the legacy pip requirements snapshot                                                          | Keep in sync with `conf/requirements.txt` when maintaining the pair                    |
| `conf/requirements_ci.txt`       | Committed pip freeze (`==` pins). Dependabot's `python-minor` group rewrites it. The CI capture body is `pip list --format=freeze` after `.[all]` is installed | `dependency-docs` captures a new copy with `juniper-generate-dep-docs` and uploads the artifact; that job does not commit |
| `conf/conda_environment_ci.yaml` | Committed `conda env export` snapshot. The CI capture body is the `dependencies:` block of `conda env export --no-builds`, not `conda list --explicit` | Same artifact upload; the job does not commit. See [CI/CD manual](../docs/ci_cd/CICD_MANUAL.md#job-dependency-docs) |

Juniper Data also keeps environment snapshots under `conf/`. Those files are a different pin set from `requirements.lock`. `lockfile-check` verifies the lockfile against `pyproject.toml` (constraint compile). A grouped Dependabot push can still add an `--upgrade` lockfile commit when it never edits `pyproject.toml`. See [Automated Flow](#automated-flow-dependabot).

## Dependency File Roles

Use `pyproject.toml` as the source of truth for installable package metadata and direct dependency contracts. Other dependency files serve narrower operational purposes:

| File                         | Role                                                                                     | Review Guidance                                                                                    |
|------------------------------|------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| `pyproject.toml`             | Authoritative dependency ranges and extras used by `pip install -e ...`                  | Confirm source or tests need the package/range, then refresh `requirements.lock`                   |
| `requirements.lock`          | Exact pins for Docker builds and lockfile freshness checks                               | Regenerate from `pyproject.toml`; do not manually merge conflict hunks                             |
| `conf/requirements.txt`      | Lightweight/no-CUDA pip environment snapshot used for environment review and replication | Treat Dependabot-only floor bumps as snapshot maintenance unless code imports the package directly |
| `conf/requirements-ORIG.txt` | Baseline copy of the same pip environment snapshot                                       | Keep synchronized with `conf/requirements.txt` for the same package line                           |
| `conf/requirements_ci.txt`   | Committed freeze. Dependabot updates `==` pins in grouped pip PRs; CI captures a new copy with `juniper-generate-dep-docs` and does not commit it | Review the pin diff on its own. Do not copy it into `requirements.lock` or `pyproject.toml`       |

Example: a PR that changes only `responses>=0.25.8` to `responses>=0.26.0` in `conf/requirements.txt` and `conf/requirements-ORIG.txt` updates a pip snapshot floor. Because the current `juniper_data/tests` tree does not import `responses`, it should not be added to `[project.optional-dependencies.test]` just because the snapshot changed.

## Automated Flow (Dependabot)

`.github/dependabot.yml` puts every pip minor and patch update into the `python-minor` group (weekly, Monday 09:00 `America/New_York`, directory `/`, patterns `*`). That group rewrites the requirement files Dependabot finds under the repo, including:

- `conf/requirements.txt` and `conf/requirements-ORIG.txt` (floors such as `package>=x.y.z`)
- `conf/requirements_ci.txt` (`==` pins)

`pyproject.toml` stays put when its `>=` floors already accept the new releases. A grouped PR can be conf-only. [#446](https://github.com/pcalnon/juniper-data/pull/446) is that shape: 21 snapshot updates, no `pyproject.toml` diff, then a lockfile commit.

```
1. Dependabot pushes the grouped bump to dependabot/pip/**
2. lockfile-update.yml runs on that push when github.actor == dependabot[bot]
   - The push arm does not require a pyproject.toml diff
   - CROSS_REPO_DISPATCH_TOKEN is read from the Dependabot secret store
   - Empty token: green skip plus a notice. Freshness still runs in CI
   - Token present: uv pip compile --upgrade (api, observability, mnist, equities)
   - Lockfile changed: GitHub-signed commit "[dependabot skip] Update requirements.lock"
     via createCommitOnBranch. The PAT is the author, so CI runs again
3. A separate pull_request arm covers same-repo pyproject.toml edits
   (forks and release/** are excluded). A missing token there fails the job,
   unless the actor is dependabot[bot] (then it is the same green skip)
4. lockfile-check recompiles with --constraint requirements.lock and diffs pin lines
   A newer release inside the current ranges does not fail this check
5. Review the snapshot diff and, when present, the lockfile commit as two artifacts
```

The `--upgrade` commit resolves `pyproject.toml`. It is a different pin set from `conf/requirements_ci.txt`. On #446 the lockfile commit added `opentelemetry-api` (`# via fastapi`) and moved pins the freeze left behind (`websockets` 17.1 in the freeze, 17.2 in the lockfile; `filelock` 4.0.9 in the freeze, 4.0.11 in the lockfile).

A package that appears only in the conf snapshots is environment metadata. Promote it into `pyproject.toml` when source or tests import it, or when an install extra already contracts it.

### First CI Run May Fail

On the initial Dependabot push, the `lockfile-check` job can fail when `pyproject.toml` has been updated but `requirements.lock` has not yet been regenerated. This is expected for direct dependency changes:

- The `lockfile-update.yml` workflow pushes the fix within seconds
- The concurrency group (`cancel-in-progress: true`) cancels the stale CI run
- The second CI run (triggered by the lockfile commit) passes cleanly

A conf-only grouped push leaves freshness green against the lockfile already on the branch. The `[dependabot skip]` commit can still appear when the Dependabot PAT is set, because that job always compiles with `--upgrade`. Absence of the commit means the token gate skipped, or the upgraded resolution reproduced the committed lockfile.

### Why the automated compile uses `--upgrade`

The Dependabot lockfile workflow runs:

```bash
uv pip compile pyproject.toml \
  --extra api \
  --extra observability \
  --extra mnist \
  --extra equities \
  --upgrade \
  -o requirements.lock
```

`--upgrade` moves every pin to the newest version `pyproject.toml` allows, including new transitives. The push arm runs that compile even when Dependabot left `pyproject.toml` untouched. `lockfile-check` omits `--upgrade` and passes `--constraint requirements.lock`, so it only fails when the committed pins no longer satisfy the ranges.

## Manual Flow (Editing pyproject.toml)

When you manually edit dependency ranges in `pyproject.toml`:

```bash
# 1. Edit pyproject.toml with your changes

# 2. Regenerate the lockfile
uv pip compile pyproject.toml \
  --extra api \
  --extra observability \
  --extra mnist \
  --extra equities \
  --upgrade \
  -o requirements.lock

# 3. Verify the lockfile is fresh (same command CI uses)
uv pip compile pyproject.toml \
  --extra api \
  --extra observability \
  --extra mnist \
  --extra equities \
  --constraint requirements.lock \
  -o /tmp/check.lock
grep '^[^[:space:]#]' requirements.lock | sort > /tmp/lock_pins
grep '^[^[:space:]#]' /tmp/check.lock | sort > /tmp/check_pins
diff -u /tmp/lock_pins /tmp/check_pins

# 4. Commit both files together
git add pyproject.toml requirements.lock
git commit -m "Update <package> to <version>"
```

## Compile Command Reference

```bash
uv pip compile pyproject.toml \
  --extra api \
  --extra observability \
  --extra mnist \
  --extra equities \
  --upgrade \
  -o requirements.lock
```

| Flag                             | Purpose                                                                            |
|----------------------------------|------------------------------------------------------------------------------------|
| `--extra api`                    | Include FastAPI, uvicorn, and API dependencies                                     |
| `--extra observability`          | Include Prometheus and structured logging dependencies                             |
| `--extra mnist`                  | Include the Hugging Face `datasets` chain for the MNIST / Fashion-MNIST generator  |
| `--extra equities`               | Include `yfinance` (with `pandas`) for the `equities` / `equities_seq` generators  |
| `--upgrade`                      | Allow Dependabot or manual range bumps to move existing pins                       |
| `--constraint requirements.lock` | Check whether committed pins still satisfy `pyproject.toml` without upgrading them |
| `-o requirements.lock`           | Output file                                                                        |

## Snapshot-Only Updates

Some Dependabot PRs target package lines in `conf/requirements.txt` and `conf/requirements-ORIG.txt` instead of `pyproject.toml`. These files are pip environment snapshots, not the install contract for the package.

Review checklist:

1. Confirm `conf/requirements.txt` and `conf/requirements-ORIG.txt` carry the same package floor.
2. Read `conf/requirements_ci.txt` as the freeze Dependabot edited. `dependency-docs` uploads another capture and does not push it.
3. Search the source and tests for direct imports before promoting a snapshot-only package into `pyproject.toml`.
4. When a `[dependabot skip] Update requirements.lock` commit is on the branch, review it as the `--upgrade` resolution. When it is absent, open the Update Lockfile run and look for the Dependabot-secret notice.

For the `responses` package specifically, it is an HTTP mocking library in the broader Python ecosystem. Only add it to the `test` extra if `juniper_data/tests` starts importing `responses` directly.

## Troubleshooting

### Lockfile check fails in CI

**Symptom:** `lockfile-check` job fails with "requirements.lock is stale"

**Cause:** `pyproject.toml` was edited without regenerating `requirements.lock`

**Fix:** Run the compile command above and commit the updated lockfile.

### Dependabot PR updates conf snapshots and the lockfile diverges

**Symptom:** A `python-minor` PR edits `conf/requirements*.txt` (often including `conf/requirements_ci.txt`) and a later commit rewrites `requirements.lock` with a different pin set. `pyproject.toml` is unchanged.

**Cause:** Dependabot rewrites committed requirement files. `lockfile-update.yml` then compiles `pyproject.toml` with `--upgrade` on every `dependabot/pip/**` push by `dependabot[bot]`.

**Review:** Keep the snapshot pair aligned. Treat the lockfile commit as Docker resolution for the `api`, `observability`, `mnist`, and `equities` extras. Leave `pyproject.toml` unchanged unless an import or an extra contract requires the package.

### Lockfile-update workflow doesn't commit

**Symptom:** Dependabot PR has no `[dependabot skip] Update requirements.lock` commit.

**Possible causes:**

1. Branch name does not match `dependabot/pip/**`, or the actor is not `dependabot[bot]` (push arm).
2. `CROSS_REPO_DISPATCH_TOKEN` is missing from the Dependabot secret store. The job stays green and emits a notice. Repository Actions secrets are a different store, so `gh secret list` can show the PAT while Dependabot runs still skip.
3. The `--upgrade` resolution reproduced the committed `requirements.lock`, so the job exits without a commit.
4. The pull-request arm skipped the branch (`release/**` or a fork).
5. The workflow file has a syntax error, so no run started at all.

**Debug:**

```bash
# Actions secret store. Dependabot uses Settings -> Secrets and variables -> Dependabot.
gh secret list -R pcalnon/juniper-data | grep CROSS_REPO_DISPATCH_TOKEN

# Look for the notice on a green Update Lockfile run
gh run list --workflow=lockfile-update.yml -R pcalnon/juniper-data
```

### Merge conflict in requirements.lock

**Symptom:** Dependabot PR shows merge conflict in `requirements.lock`

**Fix:** Regenerate from scratch — lockfiles should never be manually merged:

```bash
git checkout dependabot/pip/<branch>
uv pip compile pyproject.toml --extra api --extra observability --extra mnist --extra equities --upgrade -o requirements.lock
git add requirements.lock
git commit -m "[dependabot skip] Regenerate requirements.lock"
git push
```
