# Dependency Update Workflow — juniper-data

**Last Updated:** 2026-05-04
**Version:** 1.0.0
**Status:** Current

---

## Overview

This document describes how dependency updates flow through juniper-data, from Dependabot PR to merged lockfile. The lockfile (`requirements.lock`) pins exact versions for Docker builds while `pyproject.toml` uses `>=` ranges for library compatibility.

## Dependency File Roles

Use `pyproject.toml` as the source of truth for installable package metadata and direct dependency contracts. Other dependency files serve narrower operational purposes:

| File | Role | Review Guidance |
|------|------|-----------------|
| `pyproject.toml` | Authoritative dependency ranges and extras used by `pip install -e ...` | Confirm source or tests need the package/range, then refresh `requirements.lock` |
| `requirements.lock` | Exact pins for Docker builds and lockfile freshness checks | Regenerate from `pyproject.toml`; do not manually merge conflict hunks |
| `conf/requirements.txt` | Lightweight/no-CUDA pip environment snapshot used for environment review and replication | Treat Dependabot-only floor bumps as snapshot maintenance unless code imports the package directly |
| `conf/requirements-ORIG.txt` | Baseline copy of the same pip environment snapshot | Keep synchronized with `conf/requirements.txt` for the same package line |
| `conf/requirements_ci.txt` | Generated CI artifact from `scripts/generate_dep_docs.sh` | Do not edit manually |

Example: a PR that changes only `responses>=0.25.8` to `responses>=0.26.0` in `conf/requirements.txt` and `conf/requirements-ORIG.txt` updates a pip snapshot floor. Because the current `juniper_data/tests` tree does not import `responses`, it should not be added to `[project.optional-dependencies.test]` just because the snapshot changed.

## Automated Flow (Dependabot)

When Dependabot opens a PR to update a dependency:

```
1. Dependabot pushes to dependabot/pip/<package> branch
2. lockfile-update.yml triggers on push to dependabot/pip/**
   - Guard: only runs if github.actor == 'dependabot[bot]'
   - Regenerates `requirements.lock` via `uv pip compile` when the PR changes `pyproject.toml`
   - Commits with "[dependabot skip]" prefix (prevents Dependabot rebase loop)
   - Uses CROSS_REPO_DISPATCH_TOKEN so the push re-triggers CI
3. CI runs on the updated branch
   - `lockfile-check` verifies `requirements.lock` still satisfies `pyproject.toml`
   - All other quality gates run normally
4. Review and merge the Dependabot PR
```

### First CI Run May Fail

On the initial Dependabot push, the `lockfile-check` job can fail when `pyproject.toml` has been updated but `requirements.lock` has not yet been regenerated. This is expected for direct dependency changes:

- The `lockfile-update.yml` workflow pushes the fix within seconds
- The concurrency group (`cancel-in-progress: true`) cancels the stale CI run
- The second CI run (triggered by the lockfile commit) passes cleanly

If a Dependabot PR touches only `conf/requirements.txt` and `conf/requirements-ORIG.txt`, `pyproject.toml` did not change and no lockfile-update commit is required.

## Manual Flow (Editing pyproject.toml)

When you manually edit dependency ranges in `pyproject.toml`:

```bash
# 1. Edit pyproject.toml with your changes

# 2. Regenerate the lockfile
uv pip compile pyproject.toml \
  --extra api \
  --extra observability \
  --upgrade \
  -o requirements.lock

# 3. Verify the lockfile is fresh (same command CI uses)
uv pip compile pyproject.toml \
  --extra api \
  --extra observability \
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
  --upgrade \
  -o requirements.lock
```

| Flag | Purpose |
|------|---------|
| `--extra api` | Include FastAPI, uvicorn, and API dependencies |
| `--extra observability` | Include Prometheus and structured logging dependencies |
| `--upgrade` | Allow refreshed pins when direct dependency floors move |
| `--constraint requirements.lock` | CI freshness-check mode: verify committed pins satisfy `pyproject.toml` |
| `-o requirements.lock` | Output file |

## Snapshot-Only Updates

Some Dependabot PRs target package lines in `conf/requirements.txt` and `conf/requirements-ORIG.txt` instead of `pyproject.toml`. These files are pip environment snapshots, not the install contract for the package.

Review checklist:

1. Confirm both snapshot files carry the same package floor.
2. Search the source and tests for direct imports before promoting the package into `pyproject.toml`.
3. Run the relevant CI-style checks; no lockfile regeneration is needed unless `pyproject.toml` changed.

For the `responses` package specifically, it is an HTTP mocking library in the broader Python ecosystem. Only add it to the `test` extra if `juniper_data/tests` starts importing `responses` directly.

## Troubleshooting

### Lockfile check fails in CI

**Symptom:** `lockfile-check` job fails with "requirements.lock is stale"

**Cause:** `pyproject.toml` was edited without regenerating `requirements.lock`

**Fix:** Run the compile command above and commit the updated lockfile.

### Lockfile-update workflow doesn't trigger

**Symptom:** Dependabot PR has stale lockfile, no auto-commit appears

**Possible causes:**
1. Branch name doesn't match `dependabot/pip/**` pattern
2. `CROSS_REPO_DISPATCH_TOKEN` secret is missing or expired
3. Workflow file has a syntax error

**Debug:**
```bash
# Check if the secret exists
gh secret list -R pcalnon/juniper-data | grep CROSS_REPO_DISPATCH_TOKEN

# Check workflow runs
gh run list --workflow=lockfile-update.yml -R pcalnon/juniper-data
```

### Merge conflict in requirements.lock

**Symptom:** Dependabot PR shows merge conflict in `requirements.lock`

**Fix:** Regenerate from scratch — lockfiles should never be manually merged:
```bash
git checkout dependabot/pip/<branch>
uv pip compile pyproject.toml --extra api --extra observability -o requirements.lock
git add requirements.lock
git commit -m "[dependabot skip] Regenerate requirements.lock"
git push
```
