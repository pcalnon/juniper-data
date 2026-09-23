"""Strong ETags, conditional GETs and the access-counter split (APD-DATA-017 / -029 / -032).

The owner ruling (2026-09-11): a strong ``ETag`` derived from the stored SHA-256, and the
access counters moved OUT of the representation so the metadata body can carry one too;
``Content-Location`` on ``/latest`` naming the canonical ``/{dataset_id}``. Rejected: ETags
on artifacts only; a weak validator that churns on every read; a 307 from ``/latest``.

The load-bearing test is ``test_metadata_etag_survives_recorded_accesses``. Before the
split, ``access_count`` / ``last_accessed_at`` sat in the body and changed on every read,
so any honest hash of the body changed with them -- a strong validator was impossible.
"""

from datetime import UTC, datetime

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import TypeAdapter

from juniper_data.api.app import create_app
from juniper_data.api.http_cache import body_etag, combine_field_lines, if_match_fails, if_none_match_hits, strong_etag, weak_etag
from juniper_data.api.routes import datasets
from juniper_data.api.settings import Settings
from juniper_data.core.models import DatasetMeta, PublicDatasetMeta
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
    """``GET /v1/datasets/{dataset_id}/artifact``: the stored checksum as a strong ETag."""

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


@pytest.mark.unit
class TestPreconditionsRespectExistence:
    """RFC 9110 §13.2.1: a precondition is never answered for a target that would 404."""

    def test_a_deleted_artifact_is_404_even_when_if_none_match_matches(self, tmp_path) -> None:
        from juniper_data.storage.local_fs import LocalFSDatasetStore

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
