"""ETags, conditional requests and the access-counter split (APD-DATA-017 / -029 / -032).

The owner rulings. 2026-09-11: an ``ETag`` derived from the stored SHA-256, and the access
counters moved OUT of the representation so the metadata body can carry a strong one;
``Content-Location`` on ``/latest`` naming the canonical ``/{dataset_id}``. Rejected: ETags
on artifacts only; a weak validator that churns on every read; a 307 from ``/latest``.
2026-09-23: the ARTIFACT's tag is WEAK, ``W/"<checksum>"`` -- that SHA-256 covers the arrays,
not the bytes served -- while the metadata tags stay strong.

The load-bearing test is ``test_metadata_etag_survives_recorded_accesses``. Before the
split, ``access_count`` / ``last_accessed_at`` sat in the body and changed on every read,
so any honest hash of the body changed with them -- a strong validator was impossible.
"""

import json
import logging
import subprocess
import sys
from datetime import UTC, datetime

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import TypeAdapter

from juniper_data.api import http_cache
from juniper_data.api.app import create_app
from juniper_data.api.http_cache import MAX_PRECONDITION_FIELD_LENGTH, body_etag, combine_field_lines, if_match_fails, if_none_match_fails_write, if_none_match_hits, strong_etag, weak_etag, write_preconditions_hold
from juniper_data.api.routes import datasets
from juniper_data.api.settings import Settings
from juniper_data.core.models import DatasetMeta, PublicDatasetMeta
from juniper_data.storage.local_fs import LocalFSDatasetStore
from juniper_data.storage.memory import InMemoryDatasetStore

COUNTERS = ("access_count", "last_accessed_at")


@pytest.fixture
def store() -> InMemoryDatasetStore:
    """In-memory store the app under test is wired to."""
    return InMemoryDatasetStore()


@pytest.fixture
def client(store: InMemoryDatasetStore, tmp_path) -> TestClient:
    """A test client over ``store``, with an existing storage directory for readiness."""
    storage = tmp_path / "juniper_data_storage"
    storage.mkdir()
    app = create_app(settings=Settings(storage_path=str(storage)))
    datasets.set_store(store)
    return TestClient(app)


def _create(client: TestClient, *, seed: int = 1, name: str | None = None) -> str:
    """Create a small spiral dataset and return its id."""
    body: dict = {"generator": "spiral", "params": {"n_spirals": 2, "n_points_per_spiral": 20, "seed": seed}, "persist": True}
    if name is not None:
        body["name"] = name
    response = client.post("/v1/datasets", json=body)
    assert response.status_code == 201, response.text
    return response.json()["dataset_id"]


def _stored_meta(dataset_id: str, **overrides) -> DatasetMeta:
    """A hand-built stored metadata record for tests that need exact field values."""
    fields = {
        "dataset_id": dataset_id,
        "generator": "spiral",
        "generator_version": "3.0.0",
        "params": {"seed": 1},
        "n_samples": 4,
        "n_features": 2,
        "n_train": 2,
        "n_test": 2,
        "created_at": datetime(2026, 9, 22, 20, 0, tzinfo=UTC),
        "checksum": "ab" * 32,
    }
    fields.update(overrides)
    return DatasetMeta(**fields)


def _arrays() -> dict[str, np.ndarray]:
    x = np.arange(8, dtype=np.float32).reshape(4, 2)
    y = np.eye(2, dtype=np.float32)[[0, 1, 0, 1]]
    return {"X_train": x[:2], "y_train": y[:2], "X_test": x[2:], "y_test": y[2:]}


@pytest.fixture
def localfs(tmp_path) -> tuple[TestClient, LocalFSDatasetStore]:
    """A client over a LocalFS store: the store the service wires, and the one that validates ids."""
    storage = tmp_path / "juniper_data_storage"
    storage.mkdir()
    store = LocalFSDatasetStore(storage)
    app = create_app(settings=Settings(storage_path=str(storage)))
    datasets.set_store(store)
    return TestClient(app), store


class _ClosingProbe:
    """Wraps an artifact stream and records whether the route closed it."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self) -> bytes:
        return next(self._inner)

    def close(self) -> None:
        self.closed = True
        getattr(self._inner, "close", lambda: None)()


def _probe_streams(store, monkeypatch: pytest.MonkeyPatch) -> list[_ClosingProbe]:
    """Wrap every artifact stream ``store`` opens; the returned list fills as the route opens them."""
    opened: list[_ClosingProbe] = []
    real_open = store.open_artifact_stream

    def probing_open(*args, **kwargs):
        stream = real_open(*args, **kwargs)
        if stream is None:
            return None
        opened.append(_ClosingProbe(stream))
        return opened[-1]

    monkeypatch.setattr(store, "open_artifact_stream", probing_open)
    return opened


@pytest.mark.unit
class TestIfNoneMatchParsing:
    """The comparison rules of RFC 9110 §13.1.2, and the safe direction on garbage."""

    ETAG = strong_etag("abc")

    def test_absent_or_empty_matches_nothing(self) -> None:
        assert not if_none_match_hits(None, self.ETAG)
        assert not if_none_match_hits("", self.ETAG)

    def test_star_matches_any_current_representation(self) -> None:
        assert if_none_match_hits("*", self.ETAG)
        assert if_none_match_hits("  *  ", self.ETAG)

    def test_list_form_and_weak_comparison(self) -> None:
        assert if_none_match_hits('"zzz", "abc"', self.ETAG)
        assert if_none_match_hits('W/"abc"', self.ETAG)
        assert not if_none_match_hits('"abd"', self.ETAG)

    def test_a_comma_inside_an_opaque_tag_is_not_a_list_separator(self) -> None:
        # Splitting on "," would cut '"x,abc"' into '"x' and 'abc"' -- neither of which
        # is a tag -- and a sloppier split could match the fragment. Scanning for quoted
        # tags keeps the tag whole.
        assert not if_none_match_hits('"x,abc"', self.ETAG)
        assert if_none_match_hits('"x,abc"', strong_etag("x,abc"))

    def test_unparseable_field_serves_the_full_body(self) -> None:
        assert not if_none_match_hits("abc", self.ETAG)
        # A tag embedded in garbage is not a list element; scanning without the grammar
        # check would find it.
        assert not if_none_match_hits('foo"abc"bar', self.ETAG)
        assert not if_none_match_hits('x, "abc"', self.ETAG)

    def test_empty_list_elements_are_allowed(self) -> None:
        assert if_none_match_hits('"zzz", , "abc"', self.ETAG)

    def test_if_match_uses_the_strong_comparison(self) -> None:
        assert not if_match_fails('"abc"', self.ETAG)
        assert not if_match_fails("*", self.ETAG)
        assert if_match_fails('W/"abc"', self.ETAG)
        assert if_match_fails('"abd"', self.ETAG)
        assert if_match_fails("garbage", self.ETAG), "a precondition the server cannot read must not pass"
        assert not if_match_fails(None, self.ETAG), "no header, no precondition"

    def test_list_valued_header_lines_are_combined(self) -> None:
        assert combine_field_lines(['"zzz"', '"abc"']) == '"zzz", "abc"'
        assert combine_field_lines(None) is None

    def test_a_field_over_the_length_cap_is_malformed(self) -> None:
        # Defence in depth beside the linear grammar: a longer field is never parsed. At the
        # cap it still is; one character over, each direction treats it as unreadable.
        at_cap = '"abc"' + " " * (MAX_PRECONDITION_FIELD_LENGTH - len('"abc"'))
        over_cap = at_cap + " "
        assert len(at_cap) == MAX_PRECONDITION_FIELD_LENGTH
        assert if_none_match_hits(at_cap, self.ETAG), "at the cap the field is still read"
        assert not if_match_fails(at_cap, self.ETAG), "at the cap the field is still read"
        assert not if_none_match_hits(over_cap, self.ETAG), "a read: an unreadable If-None-Match names nothing"
        assert if_match_fails(over_cap, self.ETAG), "an unreadable If-Match fails"
        assert if_none_match_fails_write(over_cap, self.ETAG), "a write fails closed"
        assert not if_none_match_hits("*" + " " * MAX_PRECONDITION_FIELD_LENGTH, self.ETAG), "the cap applies to * as well"
        # The cap counts the COMBINED field: two lines, each under it, that join to over it.
        lines = ['"abc"', " " * (MAX_PRECONDITION_FIELD_LENGTH - 4)]
        assert all(len(line) <= MAX_PRECONDITION_FIELD_LENGTH for line in lines)
        assert not if_none_match_hits(combine_field_lines(lines), self.ETAG)

    def test_a_write_fails_closed_on_an_unreadable_if_none_match(self) -> None:
        # A read treats an unreadable If-None-Match as naming nothing, because a wrong 304 is
        # the harm there; a write treats it as a failed precondition, because proceeding
        # under a condition the server could not read is the harm here.
        for garbage in ("abc", 'foo"abc"bar', 'x, "abc"', '"unterminated'):
            assert not if_none_match_hits(garbage, self.ETAG), garbage
            assert if_none_match_fails_write(garbage, self.ETAG), garbage
            assert not write_preconditions_hold(None, garbage, self.ETAG), garbage
        # A well-formed field keeps its meaning on a write.
        assert if_none_match_fails_write("*", self.ETAG)
        assert if_none_match_fails_write('"zzz", W/"abc"', self.ETAG), "weak comparison, as on a read"
        assert not if_none_match_fails_write('"zzz"', self.ETAG)
        assert not if_none_match_fails_write("", self.ETAG), "an empty field is an empty list and names nothing"
        assert not if_none_match_fails_write(None, self.ETAG), "no header, no precondition"


# Run in a CHILD interpreter, so a parse that never returns cannot hang the suite: the parent
# kills it at the bound and fails. The module is loaded from the exact file this process
# imported, so the child exercises the code under test -- in the non-vacuity harness, the
# mutated scratch copy -- and never an installed copy.
_PARSE_HOSTILE_FIELDS = """
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("http_cache_under_test", sys.argv[1])
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)
etag = cache.strong_etag("abc")
fields = json.loads(sys.argv[2])
print(json.dumps([[cache.if_none_match_hits(f, etag), cache.if_match_fails(f, etag), cache.write_preconditions_hold(None, f, etag)] for f in fields]))
"""


@pytest.mark.unit
class TestEntityTagListRunsInLinearTime:
    """The list grammar runs on the event loop (GET) and under ``_version_lock`` (PATCH): no backtracking.

    The first form of ``_ENTITY_TAG_LIST`` took about 1.8 s for ``", " * 22 + "x"`` and doubled
    per element, so each field below takes minutes or longer under it (``", " * 30 + "x"`` alone
    is about seven minutes on a dev box). The parse runs in a SUBPROCESS with a timeout, so a
    return of that form fails here in seconds instead of hanging CI. The bound is generous on
    purpose: this asserts "finishes", not a speed -- a wall-clock threshold flakes on a loaded
    runner.
    """

    HOSTILE = (", " * 30 + "x", ",\t\t" * 22 + "x", " , " * 26 + "x")
    BOUND_SECONDS = 30

    def test_hostile_fields_are_refused_well_inside_a_generous_bound(self) -> None:
        # Each field must REACH the grammar. One the length cap refused first would pass
        # against the backtracking form too, and pin nothing.
        assert all(len(field) <= MAX_PRECONDITION_FIELD_LENGTH for field in self.HOSTILE)
        try:
            child = subprocess.run([sys.executable, "-c", _PARSE_HOSTILE_FIELDS, http_cache.__file__, json.dumps(self.HOSTILE)], capture_output=True, text=True, timeout=self.BOUND_SECONDS)
        except subprocess.TimeoutExpired:
            pytest.fail(f"parsing {len(self.HOSTILE)} hostile precondition fields did not finish in {self.BOUND_SECONDS} s: the entity-tag list grammar backtracks")
        assert child.returncode == 0, child.stderr
        # All malformed: a read names nothing, If-Match fails, and a write fails closed.
        assert json.loads(child.stdout) == [[False, True, False]] * len(self.HOSTILE)


@pytest.mark.unit
class TestMetadataValidator:
    """``GET /v1/datasets/{dataset_id}``: a strong ETag over the exact body."""

    def test_etag_is_strong_and_is_the_hash_of_the_exact_body(self, client: TestClient) -> None:
        dataset_id = _create(client)
        response = client.get(f"/v1/datasets/{dataset_id}")
        assert response.status_code == 200
        etag = response.headers["etag"]
        assert not etag.startswith("W/"), "the ruling is a STRONG validator"
        assert etag == body_etag(response.content)
        assert response.headers["cache-control"] == "private, no-cache"

    def test_metadata_etag_survives_recorded_accesses(self, client: TestClient, store: InMemoryDatasetStore) -> None:
        dataset_id = _create(client)
        first = client.get(f"/v1/datasets/{dataset_id}").headers["etag"]
        store.record_access(dataset_id)
        store.record_access(dataset_id)
        assert store.get_meta(dataset_id).access_count >= 2, "the counters really did move"
        assert client.get(f"/v1/datasets/{dataset_id}").headers["etag"] == first

    def test_body_carries_no_access_counter(self, client: TestClient, store: InMemoryDatasetStore) -> None:
        dataset_id = _create(client)
        store.record_access(dataset_id)
        body = client.get(f"/v1/datasets/{dataset_id}").json()
        for counter in COUNTERS:
            assert counter not in body
        assert store.get_meta(dataset_id).access_count >= 1, "stored, just not represented"

    def test_matching_if_none_match_answers_304_with_no_body(self, client: TestClient) -> None:
        dataset_id = _create(client)
        etag = client.get(f"/v1/datasets/{dataset_id}").headers["etag"]
        for header in (etag, "*", f'"elsewhere", {etag}', f"W/{etag}"):
            response = client.get(f"/v1/datasets/{dataset_id}", headers={"If-None-Match": header})
            assert response.status_code == 304, header
            assert response.content == b""
            assert response.headers["etag"] == etag
            assert response.headers["cache-control"] == "private, no-cache"

    def test_stale_if_none_match_gets_the_full_body(self, client: TestClient) -> None:
        dataset_id = _create(client)
        response = client.get(f"/v1/datasets/{dataset_id}", headers={"If-None-Match": '"not-this-one"'})
        assert response.status_code == 200
        assert response.json()["dataset_id"] == dataset_id

    def test_if_none_match_on_two_header_lines_is_one_list(self, client: TestClient) -> None:
        # RFC 9110 §5.3: the two lines are the same list. Typed ``str``, FastAPI would read
        # only the first line and serve a 200 the client did not need.
        dataset_id = _create(client)
        etag = client.get(f"/v1/datasets/{dataset_id}").headers["etag"]
        response = client.get(f"/v1/datasets/{dataset_id}", headers=[("If-None-Match", '"zzz"'), ("If-None-Match", etag)])
        assert response.status_code == 304

    def test_a_tag_edit_moves_the_etag_and_the_patch_carries_the_new_one(self, client: TestClient) -> None:
        dataset_id = _create(client)
        before = client.get(f"/v1/datasets/{dataset_id}").headers["etag"]
        patched = client.patch(f"/v1/datasets/{dataset_id}/tags", json={"add_tags": ["edited"]})
        assert patched.status_code == 200
        assert "edited" in patched.json()["tags"]
        after = patched.headers["etag"]
        assert after != before
        assert after == body_etag(patched.content)
        assert client.get(f"/v1/datasets/{dataset_id}").headers["etag"] == after
        # A client holding the pre-edit copy must not be told it is current.
        assert client.get(f"/v1/datasets/{dataset_id}", headers={"If-None-Match": before}).status_code == 200

    def test_a_304_is_recorded_as_an_access(self, client: TestClient) -> None:
        dataset_id = _create(client)
        etag = client.get(f"/v1/datasets/{dataset_id}").headers["etag"]
        count_before = client.get(f"/v1/datasets/{dataset_id}/access").json()["access_count"]
        assert client.get(f"/v1/datasets/{dataset_id}", headers={"If-None-Match": etag}).status_code == 304
        assert client.get(f"/v1/datasets/{dataset_id}/access").json()["access_count"] == count_before + 1

    def test_bytes_match_fastapi_rendering_of_the_public_model(self, client: TestClient, store: InMemoryDatasetStore) -> None:
        """The route renders its own body; it must be the body FastAPI would have sent.

        The reference is a real ``response_model=PublicDatasetMeta`` route, so the test
        follows FastAPI's rendering rather than asserting which encoder FastAPI uses --
        that choice has moved across FastAPI versions. The fixture must DISCRIMINATE: a
        float in exponent form is where pydantic-core (``1e-7``) and ``json.dumps``
        (``1e-07``) disagree, so a route rendering through the other encoder fails here
        instead of silently changing the wire format and hashing bytes nobody sent.
        """
        meta = _stored_meta("exact-bytes", params={"noise": 1e-07, "seed": 1}, description="héllo")
        store.save("exact-bytes", meta, _arrays())
        reference = FastAPI()

        @reference.get("/m", response_model=PublicDatasetMeta)
        def _m() -> DatasetMeta:
            return store.get_meta("exact-bytes")

        expected = TestClient(reference).get("/m").content
        other_encoder = JSONResponse(content=TypeAdapter(PublicDatasetMeta).dump_python(store.get_meta("exact-bytes"), mode="json")).body
        assert expected != other_encoder, "the fixture must separate the two encoders, or this test proves nothing"
        assert "héllo".encode() in expected
        assert client.get("/v1/datasets/exact-bytes").content == expected


@pytest.mark.unit
class TestArtifactValidator:
    """``GET /v1/datasets/{dataset_id}/artifact``: the stored checksum as a WEAK ETag, ``W/"<checksum>"``."""

    def test_etag_is_the_stored_checksum(self, client: TestClient, store: InMemoryDatasetStore) -> None:
        dataset_id = _create(client)
        response = client.get(f"/v1/datasets/{dataset_id}/artifact")
        assert response.status_code == 200
        checksum = store.get_meta(dataset_id).checksum
        assert checksum
        # WEAK (owner ruling 2026-09-23): the checksum covers the arrays, not the served bytes.
        assert response.headers["etag"] == weak_etag(checksum)
        assert response.headers["cache-control"] == "private, no-cache"
        assert response.headers["content-disposition"] == f"attachment; filename={dataset_id}.npz"

    def test_matching_if_none_match_answers_304_and_reads_no_artifact(self, client: TestClient, store: InMemoryDatasetStore, monkeypatch: pytest.MonkeyPatch) -> None:
        dataset_id = _create(client)
        full = client.get(f"/v1/datasets/{dataset_id}/artifact")
        opened: list[str] = []
        real_open, real_bytes = store.open_artifact_stream, store.get_artifact_bytes

        def counting_open(*args, **kwargs):
            opened.append("open")
            return real_open(*args, **kwargs)

        def counting_bytes(*args, **kwargs):
            opened.append("bytes")
            return real_bytes(*args, **kwargs)

        # Both artifact readers: a 304 that read the whole artifact through get_artifact_bytes
        # would be as wrong as one that opened the stream.
        monkeypatch.setattr(store, "open_artifact_stream", counting_open)
        monkeypatch.setattr(store, "get_artifact_bytes", counting_bytes)
        response = client.get(f"/v1/datasets/{dataset_id}/artifact", headers={"If-None-Match": full.headers["etag"]})
        assert response.status_code == 304
        assert response.content == b""
        assert response.headers["etag"] == full.headers["etag"]
        assert opened == [], "a 304 must be decided before the artifact is opened"

    def test_stale_if_none_match_gets_the_same_full_body(self, client: TestClient) -> None:
        dataset_id = _create(client)
        full = client.get(f"/v1/datasets/{dataset_id}/artifact")
        again = client.get(f"/v1/datasets/{dataset_id}/artifact", headers={"If-None-Match": '"stale"'})
        assert again.status_code == 200
        assert again.content == full.content

    def test_a_dataset_without_a_checksum_has_no_etag_but_star_still_matches(self, client: TestClient, store: InMemoryDatasetStore) -> None:
        # No checksum, no validator: a tagged If-None-Match can name nothing and the full body
        # is served. ``*`` is different -- RFC 9110 §13.1.2 makes it match any CURRENT
        # representation, validator or not, so it answers 304.
        store.save("no-checksum", _stored_meta("no-checksum", checksum=None), _arrays())
        tagged = client.get("/v1/datasets/no-checksum/artifact", headers={"If-None-Match": '"anything"'})
        assert tagged.status_code == 200
        assert "etag" not in tagged.headers
        assert tagged.content
        assert client.get("/v1/datasets/no-checksum/artifact", headers={"If-None-Match": "*"}).status_code == 304

    def test_a_304_on_the_artifact_is_recorded_as_an_access(self, client: TestClient) -> None:
        dataset_id = _create(client)
        etag = client.get(f"/v1/datasets/{dataset_id}/artifact").headers["etag"]
        count_before = client.get(f"/v1/datasets/{dataset_id}/access").json()["access_count"]
        assert client.get(f"/v1/datasets/{dataset_id}/artifact", headers={"If-None-Match": etag}).status_code == 304
        assert client.get(f"/v1/datasets/{dataset_id}/access").json()["access_count"] == count_before + 1

    def test_stale_if_match_is_412_before_any_artifact_is_served(self, client: TestClient) -> None:
        dataset_id = _create(client)
        response = client.get(f"/v1/datasets/{dataset_id}/artifact", headers={"If-Match": '"stale"'})
        assert response.status_code == 412

    def test_if_match_cannot_name_a_weak_artifact_tag_but_star_matches(self, client: TestClient) -> None:
        # If-Match compares STRONGLY, so the artifact's weak tag never satisfies it -- not even
        # its own current value. That is the cost the weak ruling accepted; ``*`` still works.
        dataset_id = _create(client)
        etag = client.get(f"/v1/datasets/{dataset_id}/artifact").headers["etag"]
        assert etag.startswith("W/")
        assert client.get(f"/v1/datasets/{dataset_id}/artifact", headers={"If-Match": etag}).status_code == 412
        assert client.get(f"/v1/datasets/{dataset_id}/artifact", headers={"If-Match": "*"}).status_code == 200

    def test_unreadable_metadata_still_serves_the_artifact(self, client: TestClient, store: InMemoryDatasetStore, monkeypatch: pytest.MonkeyPatch) -> None:
        # Before validators existed this route never read the metadata, so a corrupt
        # metadata document still served the artifact. It must still, just without an ETag.
        dataset_id = _create(client)

        def broken_get_meta(_dataset_id: str) -> DatasetMeta:
            raise ValueError("truncated metadata document")

        monkeypatch.setattr(store, "get_meta", broken_get_meta)
        response = client.get(f"/v1/datasets/{dataset_id}/artifact")
        assert response.status_code == 200
        assert response.content
        assert "etag" not in response.headers

    def test_an_artifact_304_carries_its_caching_fields(self, client: TestClient) -> None:
        # RFC 9110 §15.4.5: a 304 carries the Cache-Control (and validator) the 200 would have.
        # Dropping Cache-Control lets a cache that stored the 200 lose "private, no-cache".
        dataset_id = _create(client)
        full = client.get(f"/v1/datasets/{dataset_id}/artifact")
        response = client.get(f"/v1/datasets/{dataset_id}/artifact", headers={"If-None-Match": full.headers["etag"]})
        assert response.status_code == 304
        assert response.headers["cache-control"] == full.headers["cache-control"] == "private, no-cache"
        assert response.headers["etag"] == full.headers["etag"]

    def test_a_412_on_the_artifact_is_not_an_access(self, client: TestClient) -> None:
        # Nothing was read: a failed If-Match stops the request before the artifact is touched.
        dataset_id = _create(client)
        count_before = client.get(f"/v1/datasets/{dataset_id}/access").json()["access_count"]
        assert client.get(f"/v1/datasets/{dataset_id}/artifact", headers={"If-Match": '"stale"'}).status_code == 412
        assert client.get(f"/v1/datasets/{dataset_id}/access").json()["access_count"] == count_before

    def test_the_unreadable_metadata_warning_names_the_type_and_carries_no_caller_text(self, client: TestClient, store: InMemoryDatasetStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        # The id in the URL is caller-supplied, and an exception message can carry it -- this
        # one does, deliberately. A traceback at WARNING would put it in a log record at
        # request rate; ERR-08 keeps caller strings out of log records.
        caller_id = "caller-chosen-7f3a91"
        store.save(caller_id, _stored_meta(caller_id), _arrays())
        real_get_meta = store.get_meta
        failed: list[str] = []

        def unreadable_once(dataset_id: str) -> DatasetMeta | None:
            # Only the route's read fails; record_access, which runs after it, reads normally.
            if not failed:
                failed.append(dataset_id)
                raise RuntimeError(f"cannot decode the metadata document of {dataset_id!r}")
            return real_get_meta(dataset_id)

        monkeypatch.setattr(store, "get_meta", unreadable_once)
        with caplog.at_level(logging.WARNING, logger="juniper_data"):
            response = client.get(f"/v1/datasets/{caller_id}/artifact")
        assert response.status_code == 200, "the artifact is still served, without a validator"
        assert "etag" not in response.headers
        assert failed == [caller_id], "the route's metadata read must actually have failed"
        loud = [record for record in caplog.records if record.levelno >= logging.WARNING]
        # Formatting renders any traceback too, which is where the caller's text would travel.
        assert not any(caller_id in logging.Formatter("%(message)s").format(record) for record in loud)
        ours = [record for record in loud if record.name.startswith("juniper_data")]
        assert [record.name for record in ours] == ["juniper_data.api.routes.datasets"]
        (record,) = ours
        assert record.exc_info is None, "no traceback above DEBUG"
        assert "RuntimeError" in record.getMessage(), "the exception TYPE is what the operator gets"


@pytest.mark.unit
class TestPreconditionsRespectExistence:
    """RFC 9110 §13.2.1: a precondition is never answered for a target that would 404 -- and never ignored for one that would be served."""

    def test_a_deleted_artifact_is_404_even_when_if_none_match_matches(self, tmp_path) -> None:
        storage = tmp_path / "juniper_data_storage"
        storage.mkdir()
        store = LocalFSDatasetStore(storage)
        app = create_app(settings=Settings(storage_path=str(storage)))
        datasets.set_store(store)
        client = TestClient(app)
        dataset_id = _create(client)
        etag = client.get(f"/v1/datasets/{dataset_id}/artifact").headers["etag"]
        store._npz_path(dataset_id).unlink()  # metadata left behind, artifact gone
        assert client.get(f"/v1/datasets/{dataset_id}/artifact").status_code == 404
        assert client.get(f"/v1/datasets/{dataset_id}/artifact", headers={"If-None-Match": etag}).status_code == 404
        assert client.get(f"/v1/datasets/{dataset_id}/artifact", headers={"If-None-Match": "*"}).status_code == 404

    @staticmethod
    def _orphan(localfs: tuple[TestClient, LocalFSDatasetStore]) -> tuple[TestClient, LocalFSDatasetStore, str]:
        """A dataset whose metadata is gone and whose artifact is still on disk."""
        client, store = localfs
        dataset_id = _create(client)
        store._meta_path(dataset_id).unlink()
        assert not store.exists(dataset_id), "LocalFS exists() needs the metadata too"
        assert client.get(f"/v1/datasets/{dataset_id}/artifact").status_code == 200, "unconditionally, an orphan is served"
        return client, store, dataset_id

    def test_an_orphaned_artifact_is_412_on_a_failing_if_match_and_the_stream_is_closed(self, localfs: tuple[TestClient, LocalFSDatasetStore], monkeypatch: pytest.MonkeyPatch) -> None:
        # RFC 9110 §13.1.1: when If-Match is false the method MUST NOT be performed. The
        # orphan has no validator, so no listed tag can name it -- but it would be served.
        client, store, dataset_id = self._orphan(localfs)
        opened = _probe_streams(store, monkeypatch)
        response = client.get(f"/v1/datasets/{dataset_id}/artifact", headers={"If-Match": '"not-current"'})
        assert response.status_code == 412
        assert [probe.closed for probe in opened] == [True], "a stream that will not be sent must be closed"

    def test_an_orphaned_artifact_still_satisfies_if_match_star(self, localfs: tuple[TestClient, LocalFSDatasetStore]) -> None:
        # ``*`` names any CURRENT representation, and the artifact is one.
        client, _store, dataset_id = self._orphan(localfs)
        response = client.get(f"/v1/datasets/{dataset_id}/artifact", headers={"If-Match": "*"})
        assert response.status_code == 200
        assert response.content
        assert "etag" not in response.headers

    def test_an_orphaned_artifact_answers_if_none_match_star_with_a_304(self, localfs: tuple[TestClient, LocalFSDatasetStore], monkeypatch: pytest.MonkeyPatch) -> None:
        # What read_precondition_status answers for any present representation: ``*`` hits.
        # A listed tag cannot name a representation with no validator, so the body is served.
        client, store, dataset_id = self._orphan(localfs)
        opened = _probe_streams(store, monkeypatch)
        response = client.get(f"/v1/datasets/{dataset_id}/artifact", headers={"If-None-Match": "*"})
        assert response.status_code == 304
        assert response.content == b""
        assert response.headers["cache-control"] == "private, no-cache"
        assert "etag" not in response.headers
        assert [probe.closed for probe in opened] == [True], "a stream that will not be sent must be closed"
        tagged = client.get(f"/v1/datasets/{dataset_id}/artifact", headers={"If-None-Match": '"anything"'})
        assert tagged.status_code == 200
        assert tagged.content

    def test_a_dataset_that_is_really_absent_is_404_under_every_precondition(self, localfs: tuple[TestClient, LocalFSDatasetStore]) -> None:
        # Neither metadata nor artifact: there is no representation for a precondition to judge.
        client, _store = localfs
        for headers in ({}, {"If-Match": "*"}, {"If-Match": '"x"'}, {"If-None-Match": "*"}, {"If-None-Match": '"x"'}):
            assert client.get("/v1/datasets/never-created-0000/artifact", headers=headers).status_code == 404, headers

    def test_an_invalid_id_on_the_artifact_route_is_the_normal_400_and_logs_no_warning(self, localfs: tuple[TestClient, LocalFSDatasetStore], caplog: pytest.LogCaptureFixture) -> None:
        # A malformed id is the CALLER's error, not unreadable metadata: it must not take the
        # "serve without a validator" fallback, whose warning would fire at request rate, and
        # it gets the same 400 the metadata route gives the same id.
        client, _store = localfs
        invalid = "CALLER$CONTROLLED"
        expected = client.get(f"/v1/datasets/{invalid}")
        assert expected.status_code == 400
        with caplog.at_level(logging.DEBUG, logger="juniper_data"):
            for headers in ({}, {"If-None-Match": "*"}, {"If-Match": '"x"'}):
                response = client.get(f"/v1/datasets/{invalid}/artifact", headers=headers)
                assert response.status_code == 400, headers
                assert response.json() == expected.json()
        loud = [record for record in caplog.records if record.levelno >= logging.WARNING]
        assert not any(invalid in logging.Formatter("%(message)s").format(record) for record in loud)
        ours = [record.getMessage() for record in loud if record.name.startswith("juniper_data")]
        assert ours == [], ours


@pytest.mark.unit
class TestIfMatch:
    """If-Match: strong comparison, evaluated before If-None-Match (RFC 9110 §13.1.1, §13.2.2)."""

    def test_reads_honour_if_match(self, client: TestClient) -> None:
        dataset_id = _create(client)
        etag = client.get(f"/v1/datasets/{dataset_id}").headers["etag"]
        assert client.get(f"/v1/datasets/{dataset_id}", headers={"If-Match": etag}).status_code == 200
        assert client.get(f"/v1/datasets/{dataset_id}", headers={"If-Match": "*"}).status_code == 200
        assert client.get(f"/v1/datasets/{dataset_id}", headers={"If-Match": '"stale"'}).status_code == 412
        # Strong comparison: a weak tag never satisfies If-Match, even with the same opaque value.
        assert client.get(f"/v1/datasets/{dataset_id}", headers={"If-Match": f"W/{etag}"}).status_code == 412

    def test_failed_if_match_wins_over_a_matching_if_none_match(self, client: TestClient) -> None:
        dataset_id = _create(client)
        etag = client.get(f"/v1/datasets/{dataset_id}").headers["etag"]
        response = client.get(f"/v1/datasets/{dataset_id}", headers={"If-Match": '"stale"', "If-None-Match": etag})
        assert response.status_code == 412

    def test_a_412_on_a_read_is_not_an_access(self, client: TestClient) -> None:
        dataset_id = _create(client)
        count_before = client.get(f"/v1/datasets/{dataset_id}/access").json()["access_count"]
        assert client.get(f"/v1/datasets/{dataset_id}", headers={"If-Match": '"stale"'}).status_code == 412
        assert client.get(f"/v1/datasets/{dataset_id}/access").json()["access_count"] == count_before


@pytest.mark.unit
class TestConditionalTagWrite:
    """PATCH .../tags as an optimistic-concurrency write (If-Match / If-None-Match -> 412)."""

    def test_the_patch_response_names_the_resource_its_etag_describes(self, client: TestClient) -> None:
        # The request target, .../tags, has no GET; Content-Location names the representation.
        dataset_id = _create(client)
        response = client.patch(f"/v1/datasets/{dataset_id}/tags", json={"add_tags": ["loc"]})
        assert response.headers["content-location"] == f"/v1/datasets/{dataset_id}"
        assert response.headers["etag"] == client.get(f"/v1/datasets/{dataset_id}").headers["etag"]

    def test_current_if_match_applies_the_edit(self, client: TestClient) -> None:
        dataset_id = _create(client)
        etag = client.get(f"/v1/datasets/{dataset_id}").headers["etag"]
        response = client.patch(f"/v1/datasets/{dataset_id}/tags", json={"add_tags": ["cas"]}, headers={"If-Match": etag})
        assert response.status_code == 200
        assert "cas" in response.json()["tags"]

    def test_stale_if_match_is_412_and_writes_nothing(self, client: TestClient, store: InMemoryDatasetStore) -> None:
        dataset_id = _create(client)
        stale = client.get(f"/v1/datasets/{dataset_id}").headers["etag"]
        assert client.patch(f"/v1/datasets/{dataset_id}/tags", json={"add_tags": ["first"]}).status_code == 200
        response = client.patch(f"/v1/datasets/{dataset_id}/tags", json={"add_tags": ["lost-update"]}, headers={"If-Match": stale})
        assert response.status_code == 412
        assert "lost-update" not in store.get_meta(dataset_id).tags

    def test_if_none_match_star_on_an_existing_dataset_is_412(self, client: TestClient, store: InMemoryDatasetStore) -> None:
        dataset_id = _create(client)
        response = client.patch(f"/v1/datasets/{dataset_id}/tags", json={"add_tags": ["x"]}, headers={"If-None-Match": "*"})
        assert response.status_code == 412
        assert "x" not in store.get_meta(dataset_id).tags

    def test_if_none_match_naming_the_current_tag_is_412_and_writes_nothing(self, client: TestClient, store: InMemoryDatasetStore) -> None:
        # Not only ``*``: a tag naming the CURRENT representation fails the write too
        # (RFC 9110 §13.1.2), compared weakly as on a read -- so ``W/<etag>`` does as well.
        dataset_id = _create(client)
        etag = client.get(f"/v1/datasets/{dataset_id}").headers["etag"]
        for header in (etag, f'"elsewhere", {etag}', f"W/{etag}"):
            response = client.patch(f"/v1/datasets/{dataset_id}/tags", json={"add_tags": ["named"]}, headers={"If-None-Match": header})
            assert response.status_code == 412, header
            assert "named" not in store.get_meta(dataset_id).tags, header

    def test_a_malformed_if_none_match_is_412_and_writes_nothing(self, client: TestClient, store: InMemoryDatasetStore) -> None:
        # Fail CLOSED: a write does not proceed under a condition the server could not read.
        # A read serves the full body for the same field (``test_unparseable_field_serves_the_full_body``).
        dataset_id = _create(client)
        etag = client.get(f"/v1/datasets/{dataset_id}").headers["etag"]
        # Well-formed and naming nothing current, so only the length cap can refuse it.
        over_cap = ", ".join(['"elsewhere"'] * (MAX_PRECONDITION_FIELD_LENGTH // len('"elsewhere", ') + 1))
        assert len(over_cap) > MAX_PRECONDITION_FIELD_LENGTH
        for header in ("garbage", f"foo{etag}bar", '"unterminated', over_cap):
            response = client.patch(f"/v1/datasets/{dataset_id}/tags", json={"add_tags": ["blind"]}, headers={"If-None-Match": header})
            assert response.status_code == 412, header[:40]
            assert "blind" not in store.get_meta(dataset_id).tags, header[:40]

    def test_a_well_formed_if_none_match_naming_another_tag_applies_the_edit(self, client: TestClient) -> None:
        # The control for the two tests above: a readable field that names nothing current,
        # and an empty one (an empty list), let the write through.
        dataset_id = _create(client)
        for tag, header in (("other", '"some-other-representation"'), ("empty", "")):
            response = client.patch(f"/v1/datasets/{dataset_id}/tags", json={"add_tags": [tag]}, headers={"If-None-Match": header})
            assert response.status_code == 200, header
            assert tag in response.json()["tags"]

    def test_the_store_evaluates_the_precondition_against_current_metadata_and_writes_nothing_on_false(self, store: InMemoryDatasetStore) -> None:
        from juniper_data.storage.base import PreconditionFailedError

        store.save("guarded", _stored_meta("guarded", tags=["a"]), _arrays())
        seen: list[list[str]] = []

        def refuse(current: DatasetMeta) -> bool:
            seen.append(list(current.tags))
            return False

        with pytest.raises(PreconditionFailedError):
            store.update_tags("guarded", ["b"], [], refuse)
        assert seen == [["a"]], "the precondition must see the CURRENT metadata"
        assert store.get_meta("guarded").tags == ["a"], "a failed precondition must write nothing"


@pytest.mark.unit
class TestConditionalWriteIsAtomic:
    """The PATCH precondition is checked INSIDE the lock that guards the write, or it is a race.

    Two ways to lose that, each pinned by one test here: ``update_tags`` evaluating the
    precondition before it takes ``_version_lock``, and the route evaluating it itself and
    handing the store ``None``. Both pass every functional test -- a stale ``If-Match`` still
    gets its 412 -- because the check still happens, just where another writer can slip in
    between it and the write.
    """

    def test_the_store_evaluates_the_precondition_under_its_version_lock(self, store: InMemoryDatasetStore) -> None:
        store.save("guarded", _stored_meta("guarded"), _arrays())
        assert not store._version_lock.locked(), "nothing else may hold the lock, or this proves nothing"
        held: list[bool] = []

        def check(_current: DatasetMeta) -> bool:
            held.append(store._version_lock.locked())
            return True

        store.update_tags("guarded", ["b"], [], check)
        assert held == [True], "the precondition must run while update_tags holds its lock"
        assert "b" in store.get_meta("guarded").tags

    def test_a_write_that_lands_after_the_route_and_before_the_store_is_412(self, client: TestClient, store: InMemoryDatasetStore, monkeypatch: pytest.MonkeyPatch) -> None:
        # Another writer wins the race at the latest point a route-level check could miss:
        # after the route has decided, before the store takes its lock. Only a precondition
        # the store evaluates under that lock sees the concurrent tag.
        dataset_id = _create(client)
        etag = client.get(f"/v1/datasets/{dataset_id}").headers["etag"]
        real_update_tags = store.update_tags

        def another_writer_first(target: str, add_tags: list[str], remove_tags: list[str], precondition=None):
            real_update_tags(target, ["concurrent"], [])
            return real_update_tags(target, add_tags, remove_tags, precondition)

        monkeypatch.setattr(store, "update_tags", another_writer_first)
        response = client.patch(f"/v1/datasets/{dataset_id}/tags", json={"add_tags": ["mine"]}, headers={"If-Match": etag})
        tags = store.get_meta(dataset_id).tags
        assert "concurrent" in tags, "the other writer's edit must have landed, or no race was simulated"
        assert response.status_code == 412
        assert "mine" not in tags, "the stale write must not be applied over the concurrent one"


@pytest.mark.unit
class TestLatestContentLocation:
    """``GET /v1/datasets/latest``: the canonical URI and the canonical validator."""

    def test_latest_names_its_canonical_uri_and_shares_its_etag(self, client: TestClient) -> None:
        _create(client, seed=1, name="cl-demo")
        newest = _create(client, seed=2, name="cl-demo")
        latest = client.get("/v1/datasets/latest", params={"name": "cl-demo"})
        assert latest.status_code == 200
        assert latest.json()["dataset_id"] == newest
        assert latest.headers["content-location"] == f"/v1/datasets/{newest}"
        canonical = client.get(f"/v1/datasets/{newest}")
        assert latest.headers["etag"] == canonical.headers["etag"]
        assert latest.content == canonical.content

    def test_latest_304_keeps_content_location(self, client: TestClient) -> None:
        newest = _create(client, seed=3, name="cl-304")
        etag = client.get("/v1/datasets/latest", params={"name": "cl-304"}).headers["etag"]
        response = client.get("/v1/datasets/latest", params={"name": "cl-304"}, headers={"If-None-Match": etag})
        assert response.status_code == 304
        assert response.headers["content-location"] == f"/v1/datasets/{newest}"


@pytest.mark.unit
class TestAccessCountersMoved:
    """The counters are still maintained, and are read from their own sub-resource."""

    def test_access_endpoint_serves_the_counters_uncached(self, client: TestClient, store: InMemoryDatasetStore) -> None:
        dataset_id = _create(client)
        store.record_access(dataset_id)
        store.record_access(dataset_id)
        response = client.get(f"/v1/datasets/{dataset_id}/access")
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        body = response.json()
        assert body["dataset_id"] == dataset_id
        assert body["access_count"] == store.get_meta(dataset_id).access_count
        assert body["last_accessed_at"] is not None

    def test_reading_the_counters_is_not_itself_an_access(self, client: TestClient) -> None:
        dataset_id = _create(client)
        first = client.get(f"/v1/datasets/{dataset_id}/access").json()["access_count"]
        second = client.get(f"/v1/datasets/{dataset_id}/access").json()["access_count"]
        assert second == first

    def test_access_endpoint_404s_for_an_unknown_dataset(self, client: TestClient) -> None:
        assert client.get("/v1/datasets/no-such-dataset/access").status_code == 404

    def test_no_representation_that_embeds_metadata_carries_a_counter(self, client: TestClient) -> None:
        created = client.post("/v1/datasets", json={"generator": "spiral", "params": {"n_spirals": 2, "n_points_per_spiral": 20, "seed": 9}, "name": "embed"})
        assert created.status_code == 201
        listed = client.get("/v1/datasets/filter").json()["datasets"]
        versions = client.get("/v1/datasets/versions", params={"name": "embed"}).json()["versions"]
        assert listed and versions, "an empty listing would make this loop pass vacuously"
        for representation in [created.json()["meta"], *listed, *versions]:
            for counter in COUNTERS:
                assert counter not in representation
