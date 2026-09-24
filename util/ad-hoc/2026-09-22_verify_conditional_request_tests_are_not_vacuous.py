#!/usr/bin/env python3
"""Negative control for D-B (APD-DATA-017 / -029 / -032): prove the new tests can FAIL.

Project:     Juniper
Sub-Project: juniper-data
Application: ad-hoc verification
Author:      Paul Calnon
Version:     1.2.0
License:     MIT

WHY THIS EXISTS
---------------
A test that passes against both the fixed and the broken implementation pins
nothing (memory ``reference_vacuous_pass_check_class``). Each arm below reverts one
behaviour and must turn the tests named for it red, while a named control test in the
same area stays green, so a mutation that breaks everything cannot score as "caught".
Other tests may go red too; only the named ones are checked.

WHAT IT COVERS, EXACTLY
-----------------------
Every test in ``juniper_data/tests/unit/test_conditional_requests.py`` -- 69 -- is named by
at least one arm. 58 are a must-fail of some arm, so this run shows each can fail. The other
11 are named only as controls: the run shows they stay green under a nearby mutation, not that
they can fail. They are ``test_access_endpoint_serves_the_counters_uncached``,
``test_a_304_on_the_artifact_is_recorded_as_an_access``,
``test_stale_if_none_match_gets_the_same_full_body``,
``test_a_well_formed_if_none_match_naming_another_tag_applies_the_edit``,
``test_list_form_and_weak_comparison``, ``test_star_matches_any_current_representation``,
``test_etag_is_strong_and_is_the_hash_of_the_exact_body``,
``test_stale_if_none_match_gets_the_full_body``,
``test_a_dataset_that_is_really_absent_is_404_under_every_precondition``,
``test_an_orphaned_artifact_still_satisfies_if_match_star`` and
``test_a_star_wrapped_in_spaces_and_tabs_is_still_a_star``. One test outside that file is a
control (M33). Version 1.1.0 said every behaviour the PR claimed was reverted; seven of the
file's then 56 tests were named by no arm (round-3 validation, lane A2, F6).
``util/ad-hoc/2026-09-24_count_conditional_request_harness_coverage.py`` re-derives these
numbers from ``MUTATIONS`` and pytest's own collection.

The load-bearing one is M1: put the access counters back into the representation.
That is the state APD-DATA-032 describes, and ``test_metadata_etag_survives_recorded_
accesses`` is the test that says a strong metadata ETag is impossible in it.

M5 exists because the first draft of the route rendered its body through the wrong
encoder, and only a fixture that separates the two encoders could see it. M11-M17 were
added when round-1 validation of the PR found If-Match unevaluated, a 304 answered for
data that was gone, and a corrupt metadata document failing the download.

M20-M31 were added when round-2 validation found the list grammar backtracking
exponentially (M20 restores it; its test parses in a subprocess with a timeout, so this
arm costs about 30 s by design), nothing pinning that the PATCH precondition is checked
INSIDE the store lock (M22, M23 -- each passes every functional test), a PATCH whose
If-None-Match named the current tag untested (M24), an orphaned artifact ignoring a failing
If-Match (M30, M31), the metadata fallback logging a traceback that can carry the caller's
id (M28) and swallowing a malformed id (M29), a malformed If-None-Match letting a write
through (M25), and the artifact 304's Cache-Control and a 412's access count unasserted
(M26, M27). M3 now covers reads only: writes no longer share its helper.

M32-M44 were added when round-3 validation found the PATCH guarantee broken by two writers
that took no lock -- batch-tags (M33) and DELETE (M34, M35; batch delete and expired-dataset
cleanup, M36 and M37) -- and by ``update_tags`` ignoring a failed write (M38); the cross-process
half of the precondition lock unpinned (M32, lane B's N7, which passed the whole suite); ``*``
read after ``str.strip()``, which also strips NBSP and NEL (M39, M40); a symlinked metadata file
blamed on the caller (M41, M42); and an empty ``If-Match`` pinned by nothing (M43, M44 -- lane
B's C3 and C4). M45-M50 name the tests no earlier arm did.

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
LINEAR = "TestEntityTagListRunsInLinearTime"
ATOMIC = "TestConditionalWriteIsAtomic"
STORE = REPO / "juniper_data/storage/base.py"

# The artifact route's metadata read and its fallback, as one block (M15 removes it whole).
FALLBACK_BLOCK = (
    "    try:\n"
    "        meta = await asyncio.to_thread(store.get_meta, dataset_id)\n"
    "    except InvalidDatasetIdError:\n"
    "        raise\n"
    '    except Exception as exc:  # noqa: BLE001 -- any other metadata read failure degrades to "no validator", never to a failed download\n'
    '        logger.warning("Artifact download: dataset metadata unreadable (%s); serving without an ETag", type(exc).__name__)\n'
    "        meta = None\n"
    "        metadata_readable = False\n"
)
FALLBACK_WARNING = '        logger.warning("Artifact download: dataset metadata unreadable (%s); serving without an ETag", type(exc).__name__)\n'

# The list grammar as shipped, and the backtracking form round-2 validation found (M20).
LINEAR_GRAMMAR = r"""_ENTITY_TAG_LIST = re.compile(r'[ \t]*(?:(?:W/)?"[^"]*"[ \t]*)?(?:,[ \t]*(?:(?:W/)?"[^"]*"[ \t]*)?)*')"""
BACKTRACKING_GRAMMAR = r"""_ENTITY_TAG_LIST = re.compile(r'[ \t]*(?:(?:W/)?"[^"]*")?[ \t]*(?:,[ \t]*(?:(?:W/)?"[^"]*")?[ \t]*)*')"""

# The write-direction If-None-Match check (M24, M25 each rewrite it).
WRITE_INM = "    return if_none_match is not None and (not _well_formed(if_none_match) or _list_names(if_none_match, etag, strong=False))\n"

# update_tags' locked read and precondition (M22 hoists the check out of the lock).
LOCKED_CHECK = (
    "        with self._version_lock, self._meta_write_lock(dataset_id):\n"
    "            meta = self.get_meta(dataset_id)\n"
    "            if meta is None:\n"
    "                return None\n"
    "            if precondition is not None and not precondition(meta):\n"
    "                raise PreconditionFailedError(dataset_id)\n"
)
CHECK_BEFORE_LOCK = (
    "        if precondition is not None:\n"
    "            early = self.get_meta(dataset_id)\n"
    "            if early is not None and not precondition(early):\n"
    "                raise PreconditionFailedError(dataset_id)\n"
    "        with self._version_lock, self._meta_write_lock(dataset_id):\n"
    "            meta = self.get_meta(dataset_id)\n"
    "            if meta is None:\n"
    "                return None\n"
)

# The route's hand-off of the precondition to the store (M23 checks in the route instead).
STORE_CHECKS = "    try:\n        meta = await asyncio.to_thread(store.update_tags, dataset_id, request.add_tags, request.remove_tags, precondition if conditional else None)\n"
ROUTE_CHECKS = (
    "    if conditional:\n"
    "        early = await asyncio.to_thread(store.get_meta, dataset_id)\n"
    "        if early is not None and not precondition(early):\n"
    "            raise _precondition_failed()\n"
    "    try:\n"
    "        meta = await asyncio.to_thread(store.update_tags, dataset_id, request.add_tags, request.remove_tags, None)\n"
)

# The artifact route's existence-gated 412/304 (M26, M27 edit it; 16-space indent is unique to it).
ARTIFACT_304 = "                return Response(status_code=status.HTTP_304_NOT_MODIFIED, headers=headers)\n"
ARTIFACT_412 = "            if outcome == status.HTTP_412_PRECONDITION_FAILED:\n                raise _precondition_failed()\n            if outcome == status.HTTP_304_NOT_MODIFIED:\n                # Revalidating"

# ---- Round 3 (M32-M44): the defects round-3 validation of juniper-data#428 found ----------------
LOCAL_FS = REPO / "juniper_data/storage/local_fs.py"
STAR = "TestStarTakesOnlySpacesAndTabs"
BATCH_API = "juniper_data/tests/api/test_batch_operations.py::TestBatchUpdateTags"

# The two places ``*`` is recognised (M8a skips the first; M39 and M40 widen what it strips).
STAR_OR_LIST = '    return field.strip(_OWS) == "*" or _ENTITY_TAG_LIST.fullmatch(field) is not None\n'
STAR_IN_LIST_NAMES = '    if field.strip(_OWS) == "*":\n        return True\n'

# update_tags' whole locked block, and lane B's "N7" (M32): the check inside _version_lock but
# outside the cross-process lock, which then re-reads and writes under it.
LOCKED_BLOCK = (
    "        with self._version_lock, self._meta_write_lock(dataset_id):\n"
    "            meta = self.get_meta(dataset_id)\n"
    "            if meta is None:\n"
    "                return None\n"
    "            if precondition is not None and not precondition(meta):\n"
    "                raise PreconditionFailedError(dataset_id)\n"
    "            tags = set(meta.tags)\n"
    "            tags.update(add_tags)\n"
    "            tags -= set(remove_tags)\n"
    "            meta.tags = sorted(tags)\n"
    "            if not self.update_meta(dataset_id, meta):\n"
    "                return None\n"
    "            return meta\n"
)
CHECK_OUTSIDE_FILE_LOCK = (
    "        with self._version_lock:\n"
    "            meta = self.get_meta(dataset_id)\n"
    "            if meta is None:\n"
    "                return None\n"
    "            if precondition is not None and not precondition(meta):\n"
    "                raise PreconditionFailedError(dataset_id)\n"
    "            with self._meta_write_lock(dataset_id):\n"
    "                meta = self.get_meta(dataset_id)\n"
    "                if meta is None:\n"
    "                    return None\n"
    "                tags = set(meta.tags)\n"
    "                tags.update(add_tags)\n"
    "                tags -= set(remove_tags)\n"
    "                meta.tags = sorted(tags)\n"
    "                if not self.update_meta(dataset_id, meta):\n"
    "                    return None\n"
    "                return meta\n"
)

# batch-tags' per-dataset edit, and the two unlocked hops it replaced (M33).
BATCH_LOCKED = (
    "        meta = await asyncio.to_thread(store.update_tags, dataset_id, request.add_tags, request.remove_tags)\n"
    "        if meta is None:\n"
    "            not_found.append(dataset_id)\n"
    "            continue\n"
    "        updated.append(dataset_id)\n"
)
BATCH_UNLOCKED = (
    "        meta = await asyncio.to_thread(store.get_meta, dataset_id)\n"
    "        if meta is None:\n"
    "            not_found.append(dataset_id)\n"
    "            continue\n"
    "        meta.tags = sorted((set(meta.tags) | set(request.add_tags)) - set(request.remove_tags))\n"
    "        await asyncio.to_thread(store.update_meta, dataset_id, meta)\n"
    "        updated.append(dataset_id)\n"
)

# delete_under_lock's body (M35 drops the cross-process half).
DELETE_BOTH_LOCKS = "        with self._version_lock, self._meta_write_lock(dataset_id):\n            return self.delete(dataset_id)\n"


def deletes(route: str) -> str:
    """The node id of one case of the every-route-that-deletes test."""
    return node(ATOMIC, f"test_every_route_that_deletes_holds_both_locks[{route}]")


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
        name="M3: If-None-Match is never honoured on a read",
        why="validators emitted, conditional reads ignored -- half of APD-DATA-017 (writes have their own helper: M24, M25)",
        edits=[(CACHE, "    return bool(if_none_match) and _list_names(if_none_match, etag, strong=False)\n", "    return False\n")],
        must_fail=[
            node(META, "test_matching_if_none_match_answers_304_with_no_body"),
            node(ART, "test_matching_if_none_match_answers_304_and_reads_no_artifact"),
            node(LATEST, "test_latest_304_keeps_content_location"),
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
        edits=[(CACHE, STAR_OR_LIST, "    return True\n")],
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
        edits=[(ROUTES, FALLBACK_BLOCK, "    meta = await asyncio.to_thread(store.get_meta, dataset_id)\n")],
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
    Mutation(
        name="M20: the list grammar backtracks again",
        why="the form round-2 validation found: ', ' * 22 + 'x' took ~1.8 s, doubling per element, on the event loop",
        edits=[(CACHE, LINEAR_GRAMMAR, BACKTRACKING_GRAMMAR)],
        must_fail=[node(LINEAR, "test_hostile_fields_are_refused_well_inside_a_generous_bound")],
        # Same language: every functional parsing test stays green under the slow form.
        must_still_pass=[node(PARSE, "test_list_form_and_weak_comparison"), node(PARSE, "test_unparseable_field_serves_the_full_body")],
    ),
    Mutation(
        name="M21: the precondition field-length cap is removed",
        why="defence in depth beside the linear grammar: an over-cap field must be malformed",
        edits=[(CACHE, "    if len(field) > MAX_PRECONDITION_FIELD_LENGTH:\n        return False\n", "")],
        must_fail=[node(PARSE, "test_a_field_over_the_length_cap_is_malformed"), node(WRITE, "test_a_malformed_if_none_match_is_412_and_writes_nothing")],
        must_still_pass=[node(PARSE, "test_a_write_fails_closed_on_an_unreadable_if_none_match")],
    ),
    Mutation(
        name="M22: update_tags evaluates the precondition BEFORE taking its lock",
        why="check-then-act: a writer can land between the check and the write, and a stale If-Match still gets its 412",
        edits=[(STORE, LOCKED_CHECK, CHECK_BEFORE_LOCK)],
        must_fail=[node(ATOMIC, "test_the_store_evaluates_the_precondition_under_its_version_lock")],
        must_still_pass=[node(WRITE, "test_the_store_evaluates_the_precondition_against_current_metadata_and_writes_nothing_on_false"), node(WRITE, "test_stale_if_match_is_412_and_writes_nothing")],
    ),
    Mutation(
        name="M23: the route evaluates the precondition itself and hands the store None",
        why="the check leaves the lock: another writer can land after the route decides",
        edits=[(ROUTES, STORE_CHECKS, ROUTE_CHECKS)],
        must_fail=[node(ATOMIC, "test_a_write_that_lands_after_the_route_and_before_the_store_is_412")],
        must_still_pass=[node(WRITE, "test_stale_if_match_is_412_and_writes_nothing"), node(WRITE, "test_current_if_match_applies_the_edit")],
    ),
    Mutation(
        name="M24: a write's If-None-Match honours only '*'",
        why="RFC 9110 §13.1.2: a tag naming the current representation fails the write too",
        edits=[(CACHE, WRITE_INM, WRITE_INM.replace("_list_names(if_none_match, etag, strong=False)", 'if_none_match.strip() == "*"'))],
        must_fail=[node(WRITE, "test_if_none_match_naming_the_current_tag_is_412_and_writes_nothing")],
        must_still_pass=[node(WRITE, "test_if_none_match_star_on_an_existing_dataset_is_412"), node(WRITE, "test_a_malformed_if_none_match_is_412_and_writes_nothing")],
    ),
    Mutation(
        name="M25: a malformed If-None-Match lets a write proceed",
        why="a write must fail closed on a condition it cannot read, as If-Match does",
        edits=[(CACHE, WRITE_INM, "    return if_none_match is not None and _list_names(if_none_match, etag, strong=False)\n")],
        must_fail=[node(WRITE, "test_a_malformed_if_none_match_is_412_and_writes_nothing"), node(PARSE, "test_a_write_fails_closed_on_an_unreadable_if_none_match")],
        must_still_pass=[node(WRITE, "test_if_none_match_naming_the_current_tag_is_412_and_writes_nothing"), node(WRITE, "test_a_well_formed_if_none_match_naming_another_tag_applies_the_edit")],
    ),
    Mutation(
        name="M26: the artifact 304 drops Cache-Control",
        why="RFC 9110 §15.4.5: a 304 carries the Cache-Control the 200 would have",
        edits=[(ROUTES, ARTIFACT_304, ARTIFACT_304.replace("headers=headers)", 'headers={k: v for k, v in headers.items() if k != "Cache-Control"})'))],
        must_fail=[node(ART, "test_an_artifact_304_carries_its_caching_fields")],
        must_still_pass=[node(ART, "test_matching_if_none_match_answers_304_and_reads_no_artifact")],
    ),
    Mutation(
        name="M27: a 412 on the artifact route is recorded as an access",
        why="nothing was read; the metadata route already pins the same rule",
        edits=[(ROUTES, ARTIFACT_412, ARTIFACT_412.replace("                raise _precondition_failed()\n", "                store.record_access(dataset_id)\n                raise _precondition_failed()\n"))],
        must_fail=[node(ART, "test_a_412_on_the_artifact_is_not_an_access")],
        must_still_pass=[node(ART, "test_a_304_on_the_artifact_is_recorded_as_an_access"), node(ART, "test_stale_if_match_is_412_before_any_artifact_is_served")],
    ),
    Mutation(
        name="M28: the metadata fallback logs its traceback",
        why="the traceback can carry the caller-controlled id into a WARNING at request rate (ERR-08)",
        edits=[(ROUTES, FALLBACK_WARNING, FALLBACK_WARNING.replace("type(exc).__name__)", "type(exc).__name__, exc_info=True)"))],
        must_fail=[node(ART, "test_the_unreadable_metadata_warning_names_the_type_and_carries_no_caller_text")],
        must_still_pass=[node(ART, "test_unreadable_metadata_still_serves_the_artifact")],
    ),
    Mutation(
        name="M29: a malformed id takes the unreadable-metadata fallback",
        why="the caller's error would log a WARNING per request instead of taking the normal 400",
        edits=[(ROUTES, "    except InvalidDatasetIdError:\n        raise\n", "")],
        must_fail=[node(EXIST, "test_an_invalid_id_on_the_artifact_route_is_the_normal_400_and_logs_no_warning")],
        must_still_pass=[node(ART, "test_unreadable_metadata_still_serves_the_artifact")],
    ),
    Mutation(
        name="M30: an orphaned artifact ignores its preconditions",
        why="RFC 9110 §13.1.1: a false If-Match means the method MUST NOT be performed",
        edits=[(ROUTES, "    if conditional and not evaluated:\n", "    if False:\n")],
        must_fail=[node(EXIST, "test_an_orphaned_artifact_is_412_on_a_failing_if_match_and_the_stream_is_closed"), node(EXIST, "test_an_orphaned_artifact_answers_if_none_match_star_with_a_304")],
        must_still_pass=[node(EXIST, "test_a_dataset_that_is_really_absent_is_404_under_every_precondition"), node(EXIST, "test_a_deleted_artifact_is_404_even_when_if_none_match_matches")],
    ),
    Mutation(
        name="M31: an orphan's unsent stream is left open",
        why="a stream opened and then refused must release what it holds",
        edits=[(ROUTES, "            _close_artifact_stream(artifact_stream)\n", "")],
        must_fail=[node(EXIST, "test_an_orphaned_artifact_is_412_on_a_failing_if_match_and_the_stream_is_closed"), node(EXIST, "test_an_orphaned_artifact_answers_if_none_match_star_with_a_304")],
        must_still_pass=[node(EXIST, "test_an_orphaned_artifact_still_satisfies_if_match_star")],
    ),
    Mutation(
        name="M32: update_tags checks the precondition inside _version_lock but outside the cross-process lock",
        why="lane B's N7: it passed the whole suite, and let another PROCESS write between the check and the write",
        edits=[(STORE, LOCKED_BLOCK, CHECK_OUTSIDE_FILE_LOCK)],
        must_fail=[node(ATOMIC, "test_the_store_evaluates_the_precondition_under_its_cross_process_file_lock")],
        must_still_pass=[node(ATOMIC, "test_the_store_evaluates_the_precondition_under_its_version_lock"), node(WRITE, "test_stale_if_match_is_412_and_writes_nothing")],
    ),
    Mutation(
        name="M33: batch-tags reads and writes in two unlocked hops again",
        why="its edit lands inside a conditional PATCH's window, and the PATCH erases it",
        edits=[(ROUTES, BATCH_LOCKED, BATCH_UNLOCKED)],
        must_fail=[node(ATOMIC, "test_batch_tags_cannot_land_inside_a_conditional_patchs_window")],
        must_still_pass=[node(ATOMIC, "test_a_delete_cannot_land_inside_a_conditional_patchs_window"), f"{BATCH_API}::test_add_and_remove_tags_simultaneously"],
    ),
    Mutation(
        name="M34: DELETE calls the store's unlocked delete again",
        why="a delete lands inside a conditional PATCH's window, and the PATCH answers 200 for a dataset that is gone",
        edits=[(ROUTES, "    deleted = await asyncio.to_thread(store.delete_under_lock, dataset_id)\n", "    deleted = await asyncio.to_thread(store.delete, dataset_id)\n")],
        must_fail=[node(ATOMIC, "test_a_delete_cannot_land_inside_a_conditional_patchs_window"), deletes("delete")],
        must_still_pass=[deletes("batch-delete"), node(ATOMIC, "test_batch_tags_cannot_land_inside_a_conditional_patchs_window")],
    ),
    Mutation(
        name="M35: delete_under_lock takes _version_lock only",
        why="the flock is what orders a delete against a PATCH in another worker process",
        edits=[(STORE, DELETE_BOTH_LOCKS, "        with self._version_lock:\n            return self.delete(dataset_id)\n")],
        must_fail=[deletes("delete"), deletes("batch-delete"), deletes("cleanup-expired")],
        # In one process _version_lock alone still orders the two requests.
        must_still_pass=[node(ATOMIC, "test_a_delete_cannot_land_inside_a_conditional_patchs_window")],
    ),
    Mutation(
        name="M36: batch delete calls the unlocked delete",
        why="every route that deletes must take the locks, or the PATCH guarantee has an exception",
        edits=[(STORE, "                ok = self.delete_under_lock(dataset_id)\n", "                ok = self.delete(dataset_id)\n")],
        must_fail=[deletes("batch-delete")],
        must_still_pass=[deletes("delete"), deletes("cleanup-expired")],
    ),
    Mutation(
        name="M37: expired-dataset cleanup calls the unlocked delete",
        why="every route that deletes must take the locks, or the PATCH guarantee has an exception",
        edits=[(STORE, "self.is_expired(meta) and self.delete_under_lock(meta.dataset_id))", "self.is_expired(meta) and self.delete(meta.dataset_id))")],
        must_fail=[deletes("cleanup-expired")],
        must_still_pass=[deletes("delete"), deletes("batch-delete")],
    ),
    Mutation(
        name="M38: update_tags ignores update_meta reporting the dataset gone",
        why="the PATCH answers 200 with a new ETag for a dataset that no longer exists",
        edits=[(STORE, "            if not self.update_meta(dataset_id, meta):\n                return None\n            return meta\n", "            self.update_meta(dataset_id, meta)\n            return meta\n")],
        must_fail=[node(ATOMIC, "test_a_dataset_gone_by_the_write_is_404_and_carries_no_etag")],
        must_still_pass=[node(WRITE, "test_current_if_match_applies_the_edit")],
    ),
    Mutation(
        name="M39: '*' is matched after str.strip() again, at both sites",
        why="NBSP and NEL are obs-text, not OWS: '*' wrapped in either read as '*', and a write went ahead",
        edits=[(CACHE, STAR_OR_LIST, STAR_OR_LIST.replace(".strip(_OWS)", ".strip()")), (CACHE, STAR_IN_LIST_NAMES, STAR_IN_LIST_NAMES.replace(".strip(_OWS)", ".strip()"))],
        must_fail=[node(STAR, "test_on_a_read_a_star_wrapped_in_nbsp_or_nel_is_malformed"), node(STAR, "test_on_the_patch_a_star_wrapped_in_nbsp_or_nel_fails_closed")],
        must_still_pass=[node(STAR, "test_a_star_wrapped_in_spaces_and_tabs_is_still_a_star"), node(PARSE, "test_star_matches_any_current_representation")],
    ),
    Mutation(
        name="M40: only the well-formedness check strips Unicode whitespace",
        why="the reads still come out right, and only a write's If-None-Match shows the field was misread",
        edits=[(CACHE, STAR_OR_LIST, STAR_OR_LIST.replace(".strip(_OWS)", ".strip()"))],
        must_fail=[node(STAR, "test_on_the_patch_a_star_wrapped_in_nbsp_or_nel_fails_closed")],
        must_still_pass=[node(STAR, "test_on_a_read_a_star_wrapped_in_nbsp_or_nel_is_malformed"), node(STAR, "test_a_star_wrapped_in_spaces_and_tabs_is_still_a_star")],
    ),
    Mutation(
        name="M41: a metadata file that leads out of the store is the caller's error again",
        why="the artifact route then re-raises it as a 400 where it should serve without a validator",
        edits=[(LOCAL_FS, '            raise StorageContainmentError(f"Path traversal detected for dataset_id: {dataset_id!r}")\n', '            raise InvalidDatasetIdError(f"Path traversal detected for dataset_id: {dataset_id!r}")\n')],
        must_fail=[node(EXIST, "test_a_metadata_file_that_leads_out_of_the_store_still_serves_the_artifact")],
        must_still_pass=[node(EXIST, "test_an_invalid_id_on_the_artifact_route_is_the_normal_400_and_logs_no_warning")],
    ),
    Mutation(
        name="M42: the artifact route consults exists() even when the metadata could not be read",
        why="exists() reads the same metadata, so a conditional request for that dataset fails instead of degrading",
        edits=[(ROUTES, "    if conditional and metadata_readable:\n", "    if conditional:\n")],
        must_fail=[node(EXIST, "test_a_metadata_file_that_leads_out_of_the_store_still_serves_the_artifact")],
        must_still_pass=[node(ART, "test_unreadable_metadata_still_serves_the_artifact"), node(EXIST, "test_an_orphaned_artifact_answers_if_none_match_star_with_a_304")],
    ),
    Mutation(
        name="M43: the PATCH decides 'conditional' by truthiness",
        why="lane B's C3: an empty If-Match -- a precondition that names nothing -- lets the write through",
        edits=[(ROUTES, "    conditional = if_match_field is not None or if_none_match_field is not None\n", "    conditional = bool(if_match_field or if_none_match_field)\n")],
        must_fail=[node(IFMATCH, "test_an_empty_if_match_is_412_on_the_patch_and_writes_nothing")],
        must_still_pass=[node(IFMATCH, "test_an_empty_if_match_is_412_on_a_read"), node(WRITE, "test_stale_if_match_is_412_and_writes_nothing")],
    ),
    Mutation(
        name="M44: an empty If-Match is read as no header",
        why="lane B's C4: an empty list names nothing, so If-Match must fail, on a read and on a write",
        edits=[(CACHE, "    return if_match is not None and not _list_names(if_match, etag, strong=True)\n", "    return bool(if_match) and not _list_names(if_match, etag, strong=True)\n")],
        must_fail=[node(IFMATCH, "test_an_empty_if_match_is_412_on_a_read"), node(IFMATCH, "test_an_empty_if_match_is_412_on_the_patch_and_writes_nothing")],
        must_still_pass=[node(IFMATCH, "test_reads_honour_if_match")],
    ),
    # M45-M50 give an arm to each test no earlier arm named (round-3 lane A2, F6).
    Mutation(
        name="M45: /access answers an unknown dataset with empty counters",
        why="a 200 for a dataset that does not exist",
        edits=[
            (
                ROUTES,
                "        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f\"Dataset '{dataset_id}' not found\")\n    stats = DatasetAccessStats(",
                '        return PrerenderedJSONResponse(content=_ACCESS_STATS.dump_json(DatasetAccessStats(dataset_id=dataset_id, access_count=0, last_accessed_at=None), by_alias=True), headers={"Cache-Control": CACHE_CONTROL_NO_STORE})\n    stats = DatasetAccessStats(',
            )
        ],
        must_fail=[node(ACCESS, "test_access_endpoint_404s_for_an_unknown_dataset")],
        must_still_pass=[node(ACCESS, "test_access_endpoint_serves_the_counters_uncached")],
    ),
    Mutation(
        name="M46: '*' matches only a representation that has a validator",
        why="RFC 9110 §13.1.2: '*' matches any current representation, an artifact with no checksum included",
        edits=[(CACHE, STAR_IN_LIST_NAMES, STAR_IN_LIST_NAMES.replace("        return True\n", "        return etag is not None\n"))],
        must_fail=[node(ART, "test_a_dataset_without_a_checksum_has_no_etag_but_star_still_matches")],
        must_still_pass=[node(PARSE, "test_star_matches_any_current_representation")],
    ),
    Mutation(
        name="M47: a read records its access before its preconditions are judged",
        why="a 412 reads nothing, so it must not count as an access",
        edits=[(ROUTES, "    response = _metadata_response(meta, combine_field_lines(if_match), combine_field_lines(if_none_match))\n", "    store.record_access(dataset_id)\n    response = _metadata_response(meta, combine_field_lines(if_match), combine_field_lines(if_none_match))\n")],
        must_fail=[node(IFMATCH, "test_a_412_on_a_read_is_not_an_access")],
        must_still_pass=[node(IFMATCH, "test_reads_honour_if_match")],
    ),
    Mutation(
        name="M48: an empty If-None-Match is read as '*'",
        why="an empty field is an empty list: it names nothing, and a 304 for it would leave a client on data it should not use",
        edits=[(CACHE, "    return bool(if_none_match) and _list_names(if_none_match, etag, strong=False)\n", "    return if_none_match is not None and (not if_none_match.strip(_OWS) or _list_names(if_none_match, etag, strong=False))\n")],
        must_fail=[node(PARSE, "test_absent_or_empty_matches_nothing")],
        must_still_pass=[node(PARSE, "test_list_form_and_weak_comparison"), node(META, "test_stale_if_none_match_gets_the_full_body")],
    ),
    Mutation(
        name="M49: the list grammar refuses an empty list element",
        why="RFC 9110 §5.6.1 lets a list carry empty elements; refusing them makes a valid field malformed",
        edits=[(CACHE, LINEAR_GRAMMAR, r"""_ENTITY_TAG_LIST = re.compile(r'[ \t]*(?:W/)?"[^"]*"[ \t]*(?:,[ \t]*(?:W/)?"[^"]*"[ \t]*)*')""")],
        must_fail=[node(PARSE, "test_empty_list_elements_are_allowed")],
        must_still_pass=[node(PARSE, "test_list_form_and_weak_comparison")],
    ),
    Mutation(
        name="M50: the metadata ETag is the stored checksum, not the hash of the body",
        why="a tag edit then leaves the ETag unchanged, so a client holding the pre-edit copy is told it is current",
        edits=[(ROUTES, '    headers = {"ETag": body_etag(body), "Cache-Control": CACHE_CONTROL_REVALIDATE}\n', '    headers = {"ETag": f\'"{meta.checksum}"\', "Cache-Control": CACHE_CONTROL_REVALIDATE}\n')],
        must_fail=[node(META, "test_a_tag_edit_moves_the_etag_and_the_patch_carries_the_new_one")],
        must_still_pass=[node(META, "test_matching_if_none_match_answers_304_with_no_body")],
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
