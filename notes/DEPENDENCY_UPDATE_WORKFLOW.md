# Dependency Update Workflow — juniper-data

**Last Updated:** 2026-05-04
**Version:** 1.0.0
**Status:** Current

---

## Overview

This document describes how dependency updates flow through juniper-data, from Dependabot PR to merged lockfile. The lockfile (`requirements.lock`) pins exact versions for Docker builds while `pyproject.toml` uses `>=` ranges for library compatibility.

## Dependency File Roles

| File | Role | Update rule |
|------|------|-------------|
| `pyproject.toml` | Authoritative project metadata and install extras. | Update when adding, removing, or changing supported dependency ranges. |
| `requirements.lock` | Reproducible Docker/runtime lockfile compiled from `pyproject.toml` with the `api` and `observability` extras. | Regenerate whenever `pyproject.toml` dependency ranges change. |
| `conf/requirements.txt` | Environment-oriented pip requirements snapshot used by legacy setup tooling such as `util/setup_environment.bash`. | Keep in sync with Dependabot PRs when it is the changed surface, but do not treat it as package metadata. |
| `conf/requirements-ORIG.txt` | Historical baseline for the environment requirements snapshot. | Update only when the matching `conf/requirements.txt` snapshot intentionally changes. |
| `conf/requirements_ci.txt` | CI-generated dependency documentation artifact from `scripts/generate_dep_docs.sh`. | Generated and uploaded by CI; not committed by the dependency-docs job. |

## Automated Flow (Dependabot)

When Dependabot opens a PR to update a dependency:

```
1. Dependabot pushes to dependabot/pip/<package> branch
2. lockfile-update.yml triggers on push to dependabot/pip/**
   - Guard: only runs if github.actor == 'dependabot[bot]'
   - Regenerates requirements.lock via uv pip compile --upgrade
   - Commits with "[dependabot skip]" prefix (prevents Dependabot rebase loop)
   - Uses CROSS_REPO_DISPATCH_TOKEN so the push re-triggers CI
3. CI runs on the updated branch
   - lockfile-check job verifies requirements.lock matches pyproject.toml
   - All other quality gates run normally
4. Review and merge the Dependabot PR
```

### First CI Run May Fail

On the initial Dependabot push, the lockfile-check job can fail when `pyproject.toml` has been updated but `requirements.lock` has not yet been regenerated. This is expected:

- The `lockfile-update.yml` workflow pushes the fix within seconds
- The concurrency group (`cancel-in-progress: true`) cancels the stale CI run
- The second CI run (triggered by the lockfile commit) passes cleanly

If Dependabot updates only `conf/requirements.txt` and `conf/requirements-ORIG.txt`, no `requirements.lock` change is required unless the compiled lockfile is also stale against `pyproject.toml`.

### Why the automated compile uses `--upgrade`

The Dependabot lockfile workflow runs:

```bash
uv pip compile pyproject.toml \
  --extra api \
  --extra observability \
  --upgrade \
  -o requirements.lock
```

`--upgrade` lets uv move existing pins when a bumped direct dependency or one of its transitive requirements would otherwise be blocked by the current lockfile. Manual freshness checks do not use `--upgrade` because they verify that the committed lockfile already matches the current dependency ranges.

## Manual Flow (Editing pyproject.toml)

When you manually edit dependency ranges in `pyproject.toml`:

```bash
# 1. Edit pyproject.toml with your changes

# 2. Regenerate the lockfile
uv pip compile pyproject.toml \
  --extra api \
  --extra observability \
  -o requirements.lock

# 3. Verify the lockfile is fresh (same command CI uses)
uv pip compile pyproject.toml \
  --extra api \
  --extra observability \
  -o /tmp/check.lock
diff requirements.lock /tmp/check.lock

# 4. Commit both files together
git add pyproject.toml requirements.lock
git commit -m "Update <package> to <version>"
```

## Compile Command Reference

```bash
uv pip compile pyproject.toml \
  --extra api \
  --extra observability \
  -o requirements.lock
```

| Flag | Purpose |
|------|---------|
| `--extra api` | Include FastAPI, uvicorn, and API dependencies |
| `--extra observability` | Include Prometheus and structured logging dependencies |
| `--upgrade` | Used by the Dependabot workflow to allow updated dependency pins |
| `-o requirements.lock` | Output file |

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
