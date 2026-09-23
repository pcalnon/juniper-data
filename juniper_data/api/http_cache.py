"""HTTP validators and caching headers for dataset reads (APD-DATA-017 / -029 / -032).

Three GET routes carry an ``ETag`` and honour ``If-Match`` / ``If-None-Match``:

* ``/v1/datasets/{dataset_id}`` and ``/v1/datasets/latest`` -- a STRONG ETag: the SHA-256
  of the exact response body, so it changes when, and only when, a byte of the
  representation changes. That is possible only because the access counters left the
  representation (APD-DATA-032). With them in it, every read produced a new byte
  sequence, and no strong validator could describe it.
* ``/v1/datasets/{dataset_id}/artifact`` -- a WEAK ETag, ``W/"<checksum>"``, over the
  stored ``checksum`` that ``compute_checksum`` takes when the dataset is created. Weak
  because of what that hash covers: a CANONICAL serialization of the arrays
  (uncompressed ``np.savez``, keys sorted), not the compressed bytes a store serves --
  ``sha256(served bytes) != checksum`` for every store. It changes whenever the arrays
  change, which is what whole-body revalidation needs, but identical arrays
  re-serialized (another numpy or zlib, another key order -- the in-memory store sorts
  keys, the others do not) can change the served bytes and keep the hash. "Same
  content, possibly different bytes" is the definition of a weak validator (RFC 9110
  §8.8.1), and the owner ruled 2026-09-23 to say so rather than claim a strong one the
  hash cannot back. ``If-None-Match`` compares weakly, so revalidation is unaffected;
  ``If-Match``, which compares strongly, can match an artifact only through ``*``.
  **Known gap, recorded for future work**: a truly strong artifact validator needs each
  store to record the SHA-256 of the exact bytes it writes -- a per-store change to every
  save path, with a fallback for artifacts written before it.

``Cache-Control: private, no-cache`` on all three. ``private`` because these routes sit
behind the API-key dependency and the key travels in ``X-API-Key``, not
``Authorization``: RFC 9111 §3.5 obliges a shared cache to hold back only responses to
requests carrying ``Authorization``, so without ``private`` a proxy could store one
caller's authorised response and replay it to another. ``no-cache`` -- revalidate before
every use -- because none of these URIs is immutable. Metadata changes with a tag edit.
The artifact is NOT content-addressed, whatever its id suggests: ``dataset_id`` hashes the
REQUEST (generator, version, params), so a dataset deleted or expired and re-created
serves whatever the generator now produces at the same URI -- ``equities`` with its
default ``end_date=None`` means "today", so the same params yield different data on
different days. ``immutable`` or a long ``max-age``, which the API primer
(``juniper-ml/notes/JUNIPER_2026-08-13_JUNIPER-ECOSYSTEM_API-DESIGN-AND-IMPLEMENTATION-PRIMER.md``)
prescribes for the artifact on the content-addressed premise, would serve stale data for
as long as it lasted; that prescription is rejected here. The 304 makes a revalidation
cheap on the WIRE -- no body -- not on the server: a 304 on ``/{dataset_id}`` or the
artifact is still recorded as an access, which on LocalFS rewrites the metadata document.

``Content-Location`` names the canonical ``/v1/datasets/<dataset_id>`` on ``/latest`` and on
the ``PATCH .../tags`` response, whose request target has no GET of its own. It tells a
client which resource the body represents; it does not merge cache entries, which RFC
9111 keys by the request target. The CORS middleware exposes none of these headers
(no ``expose_headers``), so browser JavaScript on another origin cannot read them.
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
# The opaque part may itself contain commas, so a list is never split on "," -- that would
# cut such a tag in two and could match on a fragment. Instead the WHOLE field must first
# be a well-formed list (``_ENTITY_TAG_LIST``: tags separated by commas, empty elements
# allowed, as RFC 9110 §5.6.1's list rule permits), and only then are the tags read out.
# Scanning without that check would find a tag embedded in garbage (``foo"<etag>"bar``).
_ENTITY_TAG = re.compile(r'(W/)?"([^"]*)"')
_ENTITY_TAG_LIST = re.compile(r'[ \t]*(?:(?:W/)?"[^"]*")?[ \t]*(?:,[ \t]*(?:(?:W/)?"[^"]*")?[ \t]*)*')


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


def weak_etag(opaque: str) -> str:
    """Quote ``opaque`` as a weak entity-tag: same content, not necessarily the same bytes."""
    return f'W/"{opaque}"'


def body_etag(body: bytes) -> str:
    """Return the strong entity-tag of an exact response body: its SHA-256, quoted."""
    return strong_etag(hashlib.sha256(body).hexdigest())


def combine_field_lines(values: list[str] | None) -> str | None:
    """Join a list-valued header received on several lines into one list (RFC 9110 §5.3)."""
    return ", ".join(values) if values else None


def _list_names(field: str, etag: str | None, *, strong: bool) -> bool:
    """Whether the entity-tag list ``field`` names ``etag`` -- the core of both preconditions.

    ``*`` names any CURRENT representation; every caller evaluates only a target that exists
    (RFC 9110 §13.2.1), so ``*`` is simply true here, with or without an ``etag``. A field
    that is not a well-formed list names nothing. ``strong`` selects the comparison function
    (RFC 9110 §8.8.3.2): If-Match uses the STRONG one, under which a ``W/`` tag never matches;
    If-None-Match uses the WEAK one, under which ``W/"x"`` matches ``"x"``.
    """
    if field.strip() == "*":
        return True
    if etag is None or _ENTITY_TAG_LIST.fullmatch(field) is None:
        return False
    current = _ENTITY_TAG.fullmatch(etag)
    if current is None:
        return False
    current_weak, current_opaque = current.group(1) is not None, current.group(2)
    for candidate in _ENTITY_TAG.finditer(field):
        weak, opaque = candidate.group(1) is not None, candidate.group(2)
        if opaque == current_opaque and (not strong or not (weak or current_weak)):
            return True
    return False


def if_none_match_hits(if_none_match: str | None, etag: str | None) -> bool:
    """Return True when ``If-None-Match`` names the current representation (weak comparison).

    For a GET that means 304; for a state-changing method, 412. An absent, empty or malformed
    field names nothing -- for a GET the safe direction, since a wrong 304 leaves a client
    using data it should not.
    """
    return bool(if_none_match) and _list_names(if_none_match, etag, strong=False)


def if_match_fails(if_match: str | None, etag: str | None) -> bool:
    """Return True when an ``If-Match`` precondition is present and does NOT hold -> 412.

    Strong comparison, so a weak tag never satisfies it. A malformed field fails: performing
    a method under a precondition the server could not read is the unsafe direction.
    """
    return if_match is not None and not _list_names(if_match, etag, strong=True)


def read_precondition_status(if_match: str | None, if_none_match: str | None, etag: str | None) -> int | None:
    """RFC 9110 §13.2.2 for a GET whose target exists: ``412``, ``304``, or ``None`` to serve it.

    If-Match is evaluated first and, when it fails, decides the response even if
    If-None-Match would have matched. If-Modified-Since / If-Unmodified-Since are not
    evaluated: these resources carry no ``Last-Modified`` (§13.2.2 steps 2 and 4 apply only
    to a representation that has one).
    """
    if if_match_fails(if_match, etag):
        return 412
    if if_none_match_hits(if_none_match, etag):
        return 304
    return None


def write_preconditions_hold(if_match: str | None, if_none_match: str | None, etag: str | None) -> bool:
    """Whether a state-changing request may proceed against the representation tagged ``etag``.

    False means 412: If-Match failed, or If-None-Match named the current representation
    (for a method other than GET/HEAD that is a failure, not a 304 -- §13.1.2).
    """
    return not if_match_fails(if_match, etag) and not if_none_match_hits(if_none_match, etag)
