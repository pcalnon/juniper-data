#!/usr/bin/env python3
"""Negative control for D-B (APD-DATA-017 / -029 / -032): prove the new tests can FAIL.

Project:     Juniper
Sub-Project: juniper-data
Application: ad-hoc verification
Author:      Paul Calnon
Version:     1.0.0
License:     MIT

WHY THIS EXISTS
---------------
A test that passes against both the fixed and the broken implementation pins
nothing (memory ``reference_vacuous_pass_check_class``). Every behaviour the D-B PR
claims is reverted here one at a time, and each reversion must turn exactly the tests
that describe it red -- while a named control test in the same area stays green, so a
mutation that breaks everything cannot score as "caught".

The load-bearing one is M1: put the access counters back into the representation.
That is the state APD-DATA-032 describes, and ``test_metadata_etag_survives_recorded_
accesses`` is the test that says a strong metadata ETag is impossible in it.

M5 exists because the first draft of the route rendered its body through the wrong
encoder, and only a fixture that separates the two encoders could see it. M11-M17 were
added when round-1 validation of the PR found If-Match unevaluated, a 304 answered for
data that was gone, and a corrupt metadata document failing the download.

Run from the repo root::

    /opt/miniforge3/envs/JuniperData/bin/python util/ad-hoc/2026-09-22_verify_conditional_request_tests_are_not_vacuous.py

Exit 0 when every mutation is caught by the expected tests, the controls stay green,
and the unmutated baseline is green; exit 1 otherwise. It mutates a scratch COPY of the
repo and never writes the checkout, so the worktree stays safe to read and edit while it
runs (earlier versions mutated in place; see ``WORK``).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PYTHON = "/opt/miniforge3/envs/JuniperData/bin/python"

ROUTES = REPO / "juniper_data/api/routes/datasets.py"
MODELS = REPO / "juniper_data/core/models.py"
CACHE = REPO / "juniper_data/api/http_cache.py"

T = "juniper_data/tests/unit/test_conditional_requests.py"


def node(cls: str, name: str) -> str:
    return f"{T}::{cls}::{name}"


META = "TestMetadataValidator"
ART = "TestArtifactValidator"
LATEST = "TestLatestContentLocation"
ACCESS = "TestAccessCountersMoved"
PARSE = "TestIfNoneMatchParsing"


@dataclass
class Mutation:
    name: str
    why: str
    edits: list[tuple[Path, str, str]]
    must_fail: list[str]
    must_still_pass: list[str]


IFMATCH = "TestIfMatch"
WRITE = "TestConditionalTagWrite"
EXIST = "TestPreconditionsRespectExistence"
STORE = REPO / "juniper_data/storage/base.py"

MUTATIONS = [
    Mutation(
        name="M1: the metadata body is rendered with the STORED model again",
        why="the APD-DATA-032 state: counters in the representation, so every read is new bytes",
        edits=[(ROUTES, "_PUBLIC_META = TypeAdapter(PublicDatasetMeta)", "_PUBLIC_META = TypeAdapter(DatasetMeta)")],
        must_fail=[
            node(META, "test_metadata_etag_survives_recorded_accesses"),
            node(META, "test_body_carries_no_access_counter"),
            node(META, "test_bytes_match_fastapi_rendering_of_the_public_model"),
        ],
        must_still_pass=[node(META, "test_etag_is_strong_and_is_the_hash_of_the_exact_body"), node(ART, "test_etag_is_the_stored_checksum")],
    ),
    Mutation(
        name="M2a: CreateDatasetResponse.meta typed with the stored model",
        why="an embedding representation keeps the counters",
        edits=[(MODELS, "    generator: str\n    meta: PublicDatasetMeta\n", "    generator: str\n    meta: DatasetMeta\n")],
        must_fail=[node(ACCESS, "test_no_representation_that_embeds_metadata_carries_a_counter")],
        must_still_pass=[node(META, "test_body_carries_no_access_counter")],
    ),
    Mutation(
        name="M2b: DatasetListResponse.datasets typed with the stored model",
        why="the /filter listing keeps the counters",
        edits=[(MODELS, "    datasets: list[PublicDatasetMeta]\n", "    datasets: list[DatasetMeta]\n")],
        must_fail=[node(ACCESS, "test_no_representation_that_embeds_metadata_carries_a_counter")],
        must_still_pass=[node(META, "test_body_carries_no_access_counter")],
    ),
    Mutation(
        name="M2c: DatasetVersionListResponse.versions typed with the stored model",
        why="the /versions listing keeps the counters",
        edits=[(MODELS, "    versions: list[PublicDatasetMeta]\n", "    versions: list[DatasetMeta]\n")],
        must_fail=[node(ACCESS, "test_no_representation_that_embeds_metadata_carries_a_counter")],
        must_still_pass=[node(META, "test_body_carries_no_access_counter")],
    ),
    Mutation(
        name="M3: If-None-Match is never honoured",
        why="validators emitted, conditional reads ignored -- half of APD-DATA-017",
        edits=[(CACHE, "    return bool(if_none_match) and _list_names(if_none_match, etag, strong=False)\n", "    return False\n")],
        must_fail=[
            node(META, "test_matching_if_none_match_answers_304_with_no_body"),
            node(ART, "test_matching_if_none_match_answers_304_and_reads_no_artifact"),
            node(LATEST, "test_latest_304_keeps_content_location"),
            node(WRITE, "test_if_none_match_star_on_an_existing_dataset_is_412"),
        ],
        must_still_pass=[node(META, "test_etag_is_strong_and_is_the_hash_of_the_exact_body"), node(META, "test_stale_if_none_match_gets_the_full_body")],
    ),
    Mutation(
        name="M4: the artifact is opened BEFORE the 304 decision",
        why="a revalidation would pay the artifact I/O it exists to avoid",
        edits=[(ROUTES, "        if await asyncio.to_thread(store.exists, dataset_id):\n", "        await asyncio.to_thread(store.open_artifact_stream, dataset_id)\n        if await asyncio.to_thread(store.exists, dataset_id):\n")],
        must_fail=[node(ART, "test_matching_if_none_match_answers_304_and_reads_no_artifact")],
        must_still_pass=[node(ART, "test_etag_is_the_stored_checksum"), node(ART, "test_stale_if_none_match_gets_the_same_full_body")],
    ),
    Mutation(
        name="M5: the body is rendered through json.dumps instead of FastAPI's path",
        why="changes the wire format of exponent floats, and hashes a rendering FastAPI would not send",
        # The route module no longer imports JSONResponse, so the mutation brings its own.
        edits=[(ROUTES, "    body = _PUBLIC_META.dump_json(meta, by_alias=True)\n", "    from fastapi.responses import JSONResponse\n\n    body = JSONResponse(content=_PUBLIC_META.dump_python(meta, mode=\"json\")).body\n")],
        must_fail=[node(META, "test_bytes_match_fastapi_rendering_of_the_public_model")],
        must_still_pass=[node(META, "test_etag_is_strong_and_is_the_hash_of_the_exact_body"), node(META, "test_metadata_etag_survives_recorded_accesses")],
    ),
    Mutation(
        name="M6: /latest names no canonical URI",
        why="the APD-DATA-029 state",
        edits=[(ROUTES, "content_location=_canonical_path(meta.dataset_id)", "content_location=None")],
        must_fail=[node(LATEST, "test_latest_names_its_canonical_uri_and_shares_its_etag"), node(LATEST, "test_latest_304_keeps_content_location")],
        must_still_pass=[node(META, "test_matching_if_none_match_answers_304_with_no_body")],
    ),
    Mutation(
        name="M7: the artifact ETag is not the stored checksum",
        why="a validator not derived from the stored SHA-256, which is what the ruling named",
        edits=[(ROUTES, "    etag = weak_etag(meta.checksum) if meta is not None", "    etag = weak_etag(meta.dataset_id) if meta is not None")],
        must_fail=[node(ART, "test_etag_is_the_stored_checksum")],
        must_still_pass=[node(ART, "test_matching_if_none_match_answers_304_and_reads_no_artifact")],
    ),
    Mutation(
        name="M8a: the list-grammar check is skipped",
        why="a tag embedded in garbage would then match",
        edits=[(CACHE, "    if etag is None or _ENTITY_TAG_LIST.fullmatch(field) is None:\n", "    if etag is None:\n")],
        must_fail=[node(PARSE, "test_unparseable_field_serves_the_full_body")],
        must_still_pass=[node(PARSE, "test_list_form_and_weak_comparison")],
    ),
    Mutation(
        name="M8b: the list is split on commas",
        why="a comma inside an opaque tag then matches on a fragment",
        edits=[(CACHE, "    for candidate in _ENTITY_TAG.finditer(field):\n        weak, opaque = candidate.group(1) is not None, candidate.group(2)\n", "    for part in field.split(\",\"):\n        weak, opaque = part.strip().startswith(\"W/\"), part.strip().removeprefix(\"W/\").strip('\"')\n")],
        must_fail=[node(PARSE, "test_a_comma_inside_an_opaque_tag_is_not_a_list_separator")],
        must_still_pass=[node(PARSE, "test_list_form_and_weak_comparison")],
    ),
    Mutation(
        name="M9: reading /access records an access",
        why="the count would describe its own observation",
        edits=[(ROUTES, "    stats = DatasetAccessStats(", "    store.record_access(dataset_id)\n    meta = await asyncio.to_thread(store.get_meta, dataset_id)\n    stats = DatasetAccessStats(")],
        must_fail=[node(ACCESS, "test_reading_the_counters_is_not_itself_an_access")],
        must_still_pass=[node(ACCESS, "test_access_endpoint_serves_the_counters_uncached")],
    ),
    Mutation(
        name="M10: a metadata 304 is not recorded as an access",
        why="a revalidation is a read of the dataset",
        edits=[(ROUTES, "    asyncio.get_event_loop().call_soon(lambda: store.record_access(dataset_id))\n    return response\n", "    if response.status_code == 200:\n        asyncio.get_event_loop().call_soon(lambda: store.record_access(dataset_id))\n    return response\n")],
        must_fail=[node(META, "test_a_304_is_recorded_as_an_access")],
        must_still_pass=[node(META, "test_matching_if_none_match_answers_304_with_no_body")],
    ),
    Mutation(
        name="M11: If-Match is never evaluated",
        why="RFC 9110 §13.1.1 -- the header a strong ETag invites clients to send",
        edits=[(CACHE, "    return if_match is not None and not _list_names(if_match, etag, strong=True)\n", "    return False\n")],
        must_fail=[
            node(IFMATCH, "test_reads_honour_if_match"),
            node(IFMATCH, "test_failed_if_match_wins_over_a_matching_if_none_match"),
            node(ART, "test_stale_if_match_is_412_before_any_artifact_is_served"),
            node(WRITE, "test_stale_if_match_is_412_and_writes_nothing"),
        ],
        must_still_pass=[node(WRITE, "test_current_if_match_applies_the_edit")],
    ),
    Mutation(
        name="M12: If-Match uses the weak comparison",
        why="a weak tag must never satisfy If-Match (RFC 9110 §8.8.3.2)",
        edits=[(CACHE, "_list_names(if_match, etag, strong=True)", "_list_names(if_match, etag, strong=False)")],
        must_fail=[node(IFMATCH, "test_reads_honour_if_match"), node(PARSE, "test_if_match_uses_the_strong_comparison")],
        must_still_pass=[node(WRITE, "test_current_if_match_applies_the_edit")],
    ),
    Mutation(
        name="M13: the store ignores the write precondition",
        why="an If-Match that cannot stop the write is not optimistic concurrency",
        edits=[(STORE, "            if precondition is not None and not precondition(meta):\n                raise PreconditionFailedError(dataset_id)\n", "")],
        must_fail=[
            node(WRITE, "test_stale_if_match_is_412_and_writes_nothing"),
            node(WRITE, "test_if_none_match_star_on_an_existing_dataset_is_412"),
            node(WRITE, "test_the_store_evaluates_the_precondition_against_current_metadata_and_writes_nothing_on_false"),
        ],
        must_still_pass=[node(WRITE, "test_current_if_match_applies_the_edit")],
    ),
    Mutation(
        name="M14: the artifact existence gate is removed",
        why="RFC 9110 §13.2.1 -- a 304 for data that is gone",
        edits=[(ROUTES, "        if await asyncio.to_thread(store.exists, dataset_id):\n", "        if True:\n")],
        must_fail=[node(EXIST, "test_a_deleted_artifact_is_404_even_when_if_none_match_matches")],
        must_still_pass=[node(ART, "test_matching_if_none_match_answers_304_and_reads_no_artifact")],
    ),
    Mutation(
        name="M15: unreadable metadata fails the download",
        why="before validators existed a corrupt .meta.json still served the artifact",
        edits=[(ROUTES, "    try:\n        meta = await asyncio.to_thread(store.get_meta, dataset_id)\n    except Exception:  # noqa: BLE001 -- any metadata read failure degrades to \"no validator\", never to a failed download\n        logger.warning(\"Artifact download: dataset metadata unreadable; serving without an ETag\", exc_info=True)\n        meta = None\n", "    meta = await asyncio.to_thread(store.get_meta, dataset_id)\n")],
        must_fail=[node(ART, "test_unreadable_metadata_still_serves_the_artifact")],
        must_still_pass=[node(ART, "test_etag_is_the_stored_checksum")],
    ),
    Mutation(
        name="M16: a list header on two lines is read as its first line only",
        why="RFC 9110 §5.3 makes the lines one list",
        edits=[(CACHE, "    return \", \".join(values) if values else None\n", "    return values[0] if values else None\n")],
        must_fail=[node(META, "test_if_none_match_on_two_header_lines_is_one_list"), node(PARSE, "test_list_valued_header_lines_are_combined")],
        must_still_pass=[node(META, "test_matching_if_none_match_answers_304_with_no_body")],
    ),
    Mutation(
        name="M17: the write precondition hashes the STORED model",
        why="a client's If-Match comes from a read, which never carries the counters -- it could never match",
        edits=[(ROUTES, "body_etag(_PUBLIC_META.dump_json(current, by_alias=True))", "body_etag(current.model_dump_json().encode())")],
        must_fail=[node(WRITE, "test_current_if_match_applies_the_edit")],
        must_still_pass=[node(WRITE, "test_the_store_evaluates_the_precondition_against_current_metadata_and_writes_nothing_on_false")],
    ),
    Mutation(
        name="M18: the artifact ETag claims to be strong again",
        why="the owner ruled W/ on 2026-09-23: the checksum covers the arrays, not the served bytes",
        edits=[(ROUTES, "    etag = weak_etag(meta.checksum) if meta is not None", "    etag = f'\"{meta.checksum}\"' if meta is not None")],
        must_fail=[node(ART, "test_etag_is_the_stored_checksum"), node(ART, "test_if_match_cannot_name_a_weak_artifact_tag_but_star_matches")],
        must_still_pass=[node(ART, "test_matching_if_none_match_answers_304_and_reads_no_artifact")],
    ),
    Mutation(
        name="M19: the PATCH response names no resource",
        why="its request target, .../tags, has no GET; the ETag must say which representation it describes",
        edits=[(ROUTES, "    return _metadata_response(meta, None, None, content_location=_canonical_path(dataset_id))", "    return _metadata_response(meta, None, None)")],
        must_fail=[node(WRITE, "test_the_patch_response_names_the_resource_its_etag_describes")],
        must_still_pass=[node(WRITE, "test_current_if_match_applies_the_edit")],
    ),
]


# Where mutations are applied and tests run: a SCRATCH COPY of the repo, never the checkout.
# The first versions of this script rewrote tracked source in place, and round-1 validation
# of juniper-data#428 caught the cost: other lanes reading the shared worktree mid-run read
# mutated source, and an edit saved while a file was mutated would have been silently
# reverted by the restore. ``main`` sets it; ``REPO``-anchored paths are translated here.
WORK: Path = REPO


def _in_work(path: Path) -> Path:
    return WORK / path.relative_to(REPO)


def run_tests(node_ids: list[str]) -> dict[str, bool]:
    """Return {node_id: passed}; each id runs alone so one red cannot mask another.

    ``PYTHONDONTWRITEBYTECODE`` because the source is rewritten between runs, sometimes
    within one mtime tick, and a stale ``.pyc`` would serve the previous variant
    (memory ``reference_mutation_check_stale_pyc_and_piped_exit``).
    """
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    results: dict[str, bool] = {}
    for node_id in node_ids:
        proc = subprocess.run([PYTHON, "-m", "pytest", node_id, "-q", "--no-header", "-p", "no:cacheprovider"], cwd=WORK, capture_output=True, text=True, env=env)
        results[node_id] = proc.returncode == 0
    return results


def apply(edits: list[tuple[Path, str, str]]) -> dict[Path, str]:
    """Apply every edit in the scratch copy, returning each touched file's original text; each pattern must match once."""
    originals: dict[Path, str] = {}
    for repo_path, find, replace in edits:
        path = _in_work(repo_path)
        if path not in originals:
            originals[path] = path.read_text(encoding="utf-8")
        text = path.read_text(encoding="utf-8")
        count = text.count(find)
        if count != 1:
            for restore_path, restore_text in originals.items():
                restore_path.write_text(restore_text, encoding="utf-8")
            raise SystemExit(f"FATAL: pattern matched {count} times (expected 1) in {repo_path.relative_to(REPO)}:\n  {find.strip()}")
        path.write_text(text.replace(find, replace), encoding="utf-8")
    return originals


def restore(originals: dict[Path, str]) -> bool:
    """Put every touched file back, then PROVE it; returns True when the copy is clean."""
    for path, text in originals.items():
        path.write_text(text, encoding="utf-8")
    return all(path.read_text(encoding="utf-8") == text for path, text in originals.items())


_ANY_LEVEL_SKIP = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
# Top level only: ``data`` is the runtime dataset store there, but a package may bundle a
# ``data/`` directory of its own deeper down, which the copy must keep.
_TOP_LEVEL_SKIP = {".git", "data", "logs"}


def _ignore(directory: str, names: list[str]) -> set[str]:
    skip = {n for n in names if n in _ANY_LEVEL_SKIP or n.endswith(".egg-info")}
    if Path(directory) == REPO:
        skip |= {n for n in names if n in _TOP_LEVEL_SKIP}
    return skip


def main() -> int:
    global WORK
    scratch = Path(tempfile.mkdtemp(prefix="db-nonvacuity-"))
    WORK = scratch / "repo"
    shutil.copytree(REPO, WORK, ignore=_ignore)
    print(f"scratch copy: {WORK} (the checkout is never written)")
    try:
        return _run()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _run() -> int:
    failures: list[str] = []
    every = sorted({n for m in MUTATIONS for n in m.must_fail + m.must_still_pass})
    print(f"baseline (unmutated tree) -- all {len(every)} named tests must PASS")
    for node_id, passed in run_tests(every).items():
        if not passed:
            failures.append(f"baseline: {node_id} is RED before any mutation")
            print(f"  RED  {node_id.split('::')[-1]}")
    for mutation in MUTATIONS:
        print(f"\n{mutation.name}\n  ({mutation.why})")
        originals: dict[Path, str] = {}
        caught: dict[str, bool] = {}
        survived: dict[str, bool] = {}
        try:
            originals = apply(mutation.edits)
            caught = run_tests(mutation.must_fail)
            survived = run_tests(mutation.must_still_pass)
        finally:
            if not restore(originals):
                failures.append(f"{mutation.name}: RESTORE FAILED -- the working tree still holds mutated source")
        for node_id, passed in caught.items():
            print(f"  {'VACUOUS' if passed else 'CAUGHT '} {node_id.split('::')[-1]}")
            if passed:
                failures.append(f"{mutation.name}: {node_id} stayed GREEN -- it pins nothing")
        for node_id, passed in survived.items():
            print(f"  {'OK     ' if passed else 'OVERBRD'} {node_id.split('::')[-1]} (control)")
            if not passed:
                failures.append(f"{mutation.name}: control {node_id} went red -- the mutation is not isolated")
    print()
    if failures:
        print("FAIL")
        for line in failures:
            print(f"  - {line}")
        return 1
    print(f"PASS: {len(MUTATIONS)} mutations, each caught by the tests that describe it, with every control green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
