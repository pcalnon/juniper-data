"""HTTP validators and caching headers for dataset reads (APD-DATA-017 / -029 / -032).

Three GET routes carry a strong ``ETag`` and honour ``If-None-Match``:

* ``/v1/datasets/{dataset_id}`` and ``/v1/datasets/latest`` -- the ETag is the SHA-256
  of the exact response body, so it is strong by construction: it changes when, and
  only when, a byte of the representation changes. That is possible only because the
  access counters left the representation (APD-DATA-032). With them in it, every read
  produced a new byte sequence, and no strong validator could describe it.
* ``/v1/datasets/{dataset_id}/artifact`` -- the ETag is the stored ``checksum``: the
  SHA-256 that ``compute_checksum`` takes when the dataset is created. **Read what that
  hash covers before relying on it.** It is taken over a CANONICAL serialization of the
  arrays (uncompressed ``np.savez``, keys sorted), not over the compressed bytes a store
  serves -- ``sha256(served bytes) != checksum`` for every store. It changes whenever the
  arrays change, and a stored artifact's bytes never change while it exists: stores
  write it once. What it cannot see is a re-serialization of IDENTICAL arrays (another
  numpy or zlib, another key order), which can change the compressed bytes and keep the
  hash. The validator is therefore strong for everything this route does, which is
  revalidate a whole body: it serves no byte ranges, so no client can splice bytes from
  two serializations. An exact byte hash would need a new per-store field, which is
  storage work (APD-DATA-019's territory), not route work.

``Cache-Control: private, no-cache`` on all three. ``private`` because these routes sit
behind the API-key dependency and the key travels in ``X-API-Key``, not
``Authorization``: RFC 9111 §3.5 obliges a shared cache to hold back only responses to
requests carrying ``Authorization``, so without ``private`` a proxy could store one
caller's authorised response and replay it to another. ``no-cache`` because a dataset
can be deleted, or its tags edited, at any time: a client may keep the body but must
revalidate before each use, and the 304 is what makes that cheap.
"""

from __future__ import annotations

import hashlib
import re

from fastapi.responses import JSONResponse

#: ``Cache-Control`` for the three validated reads. See the module docstring.
CACHE_CONTROL_REVALIDATE = "private, no-cache"

#: ``Cache-Control`` for a response that changes on every read of its subject.
CACHE_CONTROL_NO_STORE = "no-store"

# One entity-tag (RFC 9110 §8.8.3): an optional weakness prefix and a quoted opaque tag.
# The opaque part may itself contain commas, so the list form of ``If-None-Match`` is
# scanned for quoted tags rather than split on "," -- splitting would cut such a tag in
# two and could match on a fragment.
_ENTITY_TAG = re.compile(r'(?:W/)?"([^"]*)"')


class PrerenderedJSONResponse(JSONResponse):
    """A ``JSONResponse`` whose body arrives already rendered, and is sent byte-for-byte.

    A strong ``ETag`` has to hash the exact bytes on the wire, so a route that emits one
    renders its body first, hashes it, and must then send THOSE bytes -- not hand the
    content to an encoder that could produce different ones. ``render`` is therefore the
    identity on ``bytes``. The JSON media type comes from the class, as it does for every
    ``JSONResponse``; no route spells it. (That also keeps ``test_binary_media_types``'s
    call-site pin meaning what it says: it requires every ``media_type=`` keyword in the
    routes to name ``BINARY_MEDIA_TYPE``, and it was written for the binary routes.)
    """

    def render(self, content: bytes) -> bytes:  # type: ignore[override]  # bytes in, bytes out
        return content


def strong_etag(opaque: str) -> str:
    """Quote ``opaque`` as a strong entity-tag (no ``W/`` prefix)."""
    return f'"{opaque}"'


def body_etag(body: bytes) -> str:
    """Return the strong entity-tag of an exact response body: its SHA-256, quoted."""
    return strong_etag(hashlib.sha256(body).hexdigest())


def if_none_match_hits(if_none_match: str | None, etag: str) -> bool:
    """Return True when ``If-None-Match`` names ``etag``, i.e. a GET should answer 304.

    RFC 9110 §13.1.2: ``*`` matches any current representation; otherwise the field is
    a list of entity-tags compared with the WEAK function, so ``W/"x"`` matches ``"x"``.
    An absent, empty or unparseable field matches nothing, which serves the full 200 --
    the safe direction, since a wrong 304 leaves a client using data it should not.
    """
    if not if_none_match:
        return False
    if if_none_match.strip() == "*":
        return True
    target = _ENTITY_TAG.fullmatch(etag)
    if target is None:
        return False
    return any(candidate.group(1) == target.group(1) for candidate in _ENTITY_TAG.finditer(if_none_match))
