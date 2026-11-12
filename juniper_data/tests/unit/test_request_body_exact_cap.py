"""Pin the exact byte where ``RequestBodyLimitMiddleware`` rejects.

Both checks are strict greater-than: a declared ``Content-Length`` of
``max_bytes``, and a body that reads as exactly ``max_bytes``, are legal.
One more byte is 413, and the route does not run.

``test_middleware.py`` sends 5 bytes against a cap of 100 and 50 bytes (or
two 8-byte chunks) against a cap of 10. Both a ``>`` and a ``>=`` reject
those, so flipping the comparison stays green and refuses a body the cap
was written to allow. A cap of 0 makes the same split on an empty body:
``0 > 0`` is false, ``0 >= 0`` is true.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.testclient import TestClient

from juniper_data.api.middleware import RequestBodyLimitMiddleware

pytestmark = [pytest.mark.unit]

_CAP = 10


def _app(max_bytes: int, seen: list[bytes]) -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestBodyLimitMiddleware, max_bytes=max_bytes)

    @app.post("/v1/echo")
    async def echo(request: Request) -> Response:
        body = await request.body()
        seen.append(body)
        return Response(content=body, media_type="application/octet-stream")

    return app


async def _drive(app: FastAPI, *, headers: list[tuple[bytes, bytes]], chunks: list[bytes]) -> tuple[int, bytes]:
    """POST at the ASGI layer so the declared length can disagree with the chunks.

    TestClient recomputes ``Content-Length`` from the payload, which cannot
    express a chunked body with no length, or a declared length larger than
    the bytes that follow.
    """
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/echo",
        "raw_path": b"/v1/echo",
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 123),
        "server": ("test", 80),
    }
    pending = list(chunks)
    sent: list[dict] = []

    async def receive():
        if pending:
            chunk = pending.pop(0)
            return {"type": "http.request", "body": chunk, "more_body": bool(pending)}
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)
    start = next(message for message in sent if message["type"] == "http.response.start")
    payload = b"".join(message.get("body", b"") for message in sent if message["type"] == "http.response.body")
    return start["status"], payload


class TestRequestBodyExactCap:
    @pytest.mark.parametrize("size", [0, _CAP - 1, _CAP])
    def test_a_body_up_to_the_cap_is_delivered_whole(self, size: int) -> None:
        """The last accepted byte is the cap itself, and the route sees all of it."""
        seen: list[bytes] = []
        body = b"x" * size
        response = TestClient(_app(_CAP, seen)).post("/v1/echo", content=body)
        assert response.status_code == 200
        assert response.content == body
        assert seen == [body]

    def test_one_byte_over_the_cap_is_rejected_and_the_route_does_not_run(self) -> None:
        seen: list[bytes] = []
        response = TestClient(_app(_CAP, seen)).post("/v1/echo", content=b"x" * (_CAP + 1))
        assert response.status_code == 413
        assert response.json()["detail"] == "Request body too large"
        assert seen == []

    def test_an_empty_body_is_legal_when_the_cap_is_zero(self) -> None:
        """``Content-Length: 0`` against ``max_bytes=0`` is the declared-length boundary.

        ``0 > 0`` does not reject. ``0 >= 0`` would, and would refuse every
        empty POST once an operator set the cap to zero.
        """
        seen: list[bytes] = []
        response = TestClient(_app(0, seen)).post("/v1/echo", content=b"")
        assert response.status_code == 200
        assert response.content == b""
        assert seen == [b""]

    def test_one_byte_is_rejected_when_the_cap_is_zero(self) -> None:
        seen: list[bytes] = []
        response = TestClient(_app(0, seen)).post("/v1/echo", content=b"x")
        assert response.status_code == 413
        assert seen == []

    @pytest.mark.parametrize(
        ("chunks", "status"),
        [
            ([b"x" * 6, b"x" * 3], 200),
            ([b"x" * 6, b"x" * 4], 200),
            ([b"x" * 6, b"x" * 5], 413),
        ],
    )
    async def test_a_chunked_body_with_no_content_length_uses_the_same_boundary(self, chunks: list[bytes], status: int) -> None:
        """No declared length, so only the running total can reject.

        6+4 lands on the cap and must be cached whole. 6+5 crosses it on the
        second chunk. Checking each chunk on its own accepts
        the cross; using ``>=`` rejects the exact landing.
        """
        seen: list[bytes] = []
        status_code, payload = await _drive(
            _app(_CAP, seen),
            headers=[(b"content-type", b"application/octet-stream")],
            chunks=chunks,
        )
        assert status_code == status
        if status == 200:
            assert payload == b"".join(chunks)
            assert seen == [b"".join(chunks)]
        else:
            assert b"Request body too large" in payload
            assert seen == []

    async def test_a_declared_length_one_over_the_cap_rejects_a_shorter_body(self) -> None:
        """The early check is its own bound: one byte declared over the cap is 413.

        The body that follows would have fit. Deleting the declared-length
        check, or comparing with ``>=`` only on the bytes read, lets it through.
        """
        seen: list[bytes] = []
        status_code, payload = await _drive(
            _app(_CAP, seen),
            headers=[
                (b"content-length", str(_CAP + 1).encode()),
                (b"content-type", b"application/octet-stream"),
            ],
            chunks=[b"x"],
        )
        assert status_code == 413
        assert b"Request body too large" in payload
        assert seen == []

    async def test_a_declared_length_equal_to_the_cap_still_delivers_that_body(self) -> None:
        """Equality on the declared length is not an early reject."""
        seen: list[bytes] = []
        body = b"y" * _CAP
        status_code, payload = await _drive(
            _app(_CAP, seen),
            headers=[
                (b"content-length", str(_CAP).encode()),
                (b"content-type", b"application/octet-stream"),
            ],
            chunks=[body],
        )
        assert status_code == 200
        assert payload == body
        assert seen == [body]
