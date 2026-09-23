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
encoder, and only a fixture that separates the two encoders could see it.

Run from the repo root::

    /opt/miniforge3/envs/JuniperData/bin/python util/ad-hoc/2026-09-22_verify_conditional_request_tests_are_not_vacuous.py

Exit 0 when every mutation is caught by the expected tests, the controls stay green,
and the unmutated baseline is green; exit 1 otherwise. **It rewrites tracked source
while it runs** -- do not commit, push or edit this worktree until it exits.
"""

from __future__ import annotations

import os
import subprocess
import sys
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
        why="validators emitted, conditional requests ignored -- half of APD-DATA-017",
        edits=[(CACHE, "    if not if_none_match:\n        return False\n", "    return False\n")],
        must_fail=[
            node(META, "test_matching_if_none_match_answers_304_with_no_body"),
            node(ART, "test_matching_if_none_match_answers_304_and_reads_no_artifact"),
            node(LATEST, "test_latest_304_keeps_content_location"),
            node(PARSE, "test_star_matches_any_current_representation"),
        ],
        must_still_pass=[node(META, "test_etag_is_strong_and_is_the_hash_of_the_exact_body"), node(META, "test_stale_if_none_match_gets_the_full_body")],
    ),
    Mutation(
        name="M4: the artifact is opened BEFORE the 304 decision",
        why="a revalidation would pay the artifact I/O it exists to avoid",
        edits=[(ROUTES, "    meta = await asyncio.to_thread(store.get_meta, dataset_id)\n    headers = {\"Cache-Control\": CACHE_CONTROL_REVALIDATE}\n", "    await asyncio.to_thread(store.open_artifact_stream, dataset_id)\n    meta = await asyncio.to_thread(store.get_meta, dataset_id)\n    headers = {\"Cache-Control\": CACHE_CONTROL_REVALIDATE}\n")],
        must_fail=[node(ART, "test_matching_if_none_match_answers_304_and_reads_no_artifact")],
        must_still_pass=[node(ART, "test_etag_is_the_stored_checksum"), node(ART, "test_stale_if_none_match_gets_the_same_full_body")],
    ),
    Mutation(
        name="M5: the body is rendered through json.dumps instead of FastAPI's path",
        why="changes the wire format of exponent floats, and hashes a rendering FastAPI would not send",
        edits=[(ROUTES, "    body = _PUBLIC_META.dump_json(meta, by_alias=True)\n", "    body = JSONResponse(content=_PUBLIC_META.dump_python(meta, mode=\"json\")).body\n")],
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
        edits=[(ROUTES, "        headers[\"ETag\"] = strong_etag(meta.checksum)\n", "        headers[\"ETag\"] = strong_etag(meta.dataset_id)\n")],
        must_fail=[node(ART, "test_etag_is_the_stored_checksum")],
        must_still_pass=[node(ART, "test_matching_if_none_match_answers_304_and_reads_no_artifact")],
    ),
    Mutation(
        name="M8: If-None-Match split on commas",
        why="a comma inside an opaque tag then matches on a fragment",
        edits=[(CACHE, "    return any(candidate.group(1) == target.group(1) for candidate in _ENTITY_TAG.finditer(if_none_match))\n", "    return any(part.strip().removeprefix(\"W/\").strip('\"') == target.group(1) for part in if_none_match.split(\",\"))\n")],
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
        edits=[(ROUTES, "    asyncio.get_event_loop().call_soon(lambda: store.record_access(dataset_id))\n    return _metadata_response(meta, if_none_match)\n", "    response = _metadata_response(meta, if_none_match)\n    if response.status_code == 200:\n        asyncio.get_event_loop().call_soon(lambda: store.record_access(dataset_id))\n    return response\n")],
        must_fail=[node(META, "test_a_304_is_recorded_as_an_access")],
        must_still_pass=[node(META, "test_matching_if_none_match_answers_304_with_no_body")],
    ),
]


def run_tests(node_ids: list[str]) -> dict[str, bool]:
    """Return {node_id: passed}; each id runs alone so one red cannot mask another.

    ``PYTHONDONTWRITEBYTECODE`` because the source is rewritten in place, sometimes
    within one mtime tick, and a stale ``.pyc`` would serve the previous variant
    (memory ``reference_mutation_check_stale_pyc_and_piped_exit``).
    """
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    results: dict[str, bool] = {}
    for node_id in node_ids:
        proc = subprocess.run([PYTHON, "-m", "pytest", node_id, "-q", "--no-header", "-p", "no:cacheprovider"], cwd=REPO, capture_output=True, text=True, env=env)
        results[node_id] = proc.returncode == 0
    return results


def apply(edits: list[tuple[Path, str, str]]) -> dict[Path, str]:
    """Apply every edit, returning each touched file's original text; each pattern must match once."""
    originals: dict[Path, str] = {}
    for path, find, replace in edits:
        if path not in originals:
            originals[path] = path.read_text(encoding="utf-8")
        text = path.read_text(encoding="utf-8")
        count = text.count(find)
        if count != 1:
            for restore_path, restore_text in originals.items():
                restore_path.write_text(restore_text, encoding="utf-8")
            raise SystemExit(f"FATAL: pattern matched {count} times (expected 1) in {path.relative_to(REPO)}:\n  {find.strip()}")
        path.write_text(text.replace(find, replace), encoding="utf-8")
    return originals


def restore(originals: dict[Path, str]) -> bool:
    """Put every touched file back, then PROVE it; returns True when the tree is clean."""
    for path, text in originals.items():
        path.write_text(text, encoding="utf-8")
    return all(path.read_text(encoding="utf-8") == text for path, text in originals.items())


def main() -> int:
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
