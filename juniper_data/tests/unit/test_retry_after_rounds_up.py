"""``Retry-After`` and ``X-RateLimit-Reset`` round the wait UP and never say 0.

``RateLimiter.check`` returned ``int(window - elapsed)``, which truncates toward zero. A
sub-second remainder became ``Retry-After: 0``, an instruction to retry at once into a
limiter certain to reject again, and every other remainder with a fractional part was
under-reported, so a caller that obeyed the header woke while the window was still closed.
The allowed path fed ``X-RateLimit-Reset`` the same truncation. ``FailedAuthThrottle.check``
floored its value at 1, so it never said 0, but it truncated too: 9.4 s left said 9.

Both now return ``max(1, math.ceil(window - elapsed))``, the formula juniper-service-core
uses (APD-SVCCORE-004 for its ``RateLimiter``, juniper-ml#2524 for its
``FailedAuthThrottle``). The older checks do not separate the two roundings:
``test_security.py`` asserts ``reset_in > 0`` a moment into a 60 s window, and
``test_middleware.py`` asserts ``>= 1``, or exact values at whole-second and half-second
remainders, which truncation with a floor of 1 also produces.

The clock is frozen, so each remainder is exact. ``RateLimiter``'s ``TTLCache`` expires
entries on ``time.monotonic``, which the frozen ``time.time`` does not move, so the
sub-second cases use a 60 s window: no entry can expire while a test runs.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from juniper_data.api.middleware import SecurityMiddleware
from juniper_data.api.security import APIKeyAuth, FailedAuthThrottle, RateLimiter, build_failed_auth_throttle

pytestmark = [pytest.mark.unit]

_START = 1_000.0

# Seconds into a 10 s window, and the whole seconds the caller must then be told to wait.
# Away from these points the cases above one second leave 9.6, 9.4, 6, 59.6, 60, 89.75 and
# 119.5 s, and a wrong rounding can give the right answer for all of them: add 0.7 and
# truncate, add 0.4 and round, ``ceil`` of 0.3 less, truncate and add one only above a quarter
# of a second, add a half and round (Python rounds a half to the even number, so 6.5 is 6 and
# 60.5 is 60). The first three points leave 9.01, 9.001 and 1.001 s, where the first four of
# those say a second too few. The last two leave exactly 7 and 5 s, where adding a half and
# rounding says 8 and 6.
_ROUNDING_POINTS = [(0.99, 10), (0.999, 10), (8.999, 2), (3.0, 7), (5.0, 5)]

# Seconds into a 120 s window, and the wait. No other case expects a wait above 60 (the
# other windows are 10 s and 60 s, and one far longer whose answer is 1), so a wait capped at
# 60 would pass them all. Both of these are above 60, and 90 is not the window either.
_LONG_WINDOW_POINTS = [(0.5, 120), (30.25, 90)]


class _Clock:
    """A stand-in for ``time.time`` that a test moves by assignment."""

    def __init__(self, now: float) -> None:
        self.now = now

    def time(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    frozen = _Clock(_START)
    monkeypatch.setattr("juniper_data.api.security.time.time", frozen.time)
    return frozen


def _app(*, rate_limiter: RateLimiter, api_keys: list[str] | None = None, throttle: FailedAuthThrottle | None = None) -> FastAPI:
    app = FastAPI()

    @app.get("/v1/datasets")
    async def datasets() -> dict[str, list[str]]:
        return {"data": []}

    app.add_middleware(SecurityMiddleware, api_key_auth=APIKeyAuth(api_keys), rate_limiter=rate_limiter, failed_auth_throttle=throttle)
    return app


class TestRateLimiterResetRoundsUp:
    """``RateLimiter.check``'s third value, on the rejected and the allowed path."""

    @pytest.mark.parametrize("elapsed", [59.30, 59.60, 59.90, 59.99])
    def test_a_subsecond_remainder_is_one_second_not_zero(self, clock: _Clock, elapsed: float) -> None:
        """The defect itself: ``int()`` of a remainder under one second is 0."""
        limiter = RateLimiter(requests_per_minute=1, window_seconds=60)
        assert limiter.check("k") == (True, 0, 60)

        clock.now = _START + elapsed

        assert limiter.check("k") == (False, 0, 1)

    @pytest.mark.parametrize("elapsed", [0.4, 0.6])
    def test_a_fractional_remainder_rounds_up(self, clock: _Clock, elapsed: float) -> None:
        """9.6 s and 9.4 s left are both 10. Truncation says 9 for both; rounding to nearest says 9 for 9.4."""
        limiter = RateLimiter(requests_per_minute=1, window_seconds=10)
        limiter.check("k")

        clock.now = _START + elapsed

        assert limiter.check("k") == (False, 0, 10)

    def test_an_exact_remainder_is_not_bumped(self, clock: _Clock) -> None:
        """6.0 s left stays 6. A repair that always adds one second fails here."""
        limiter = RateLimiter(requests_per_minute=1, window_seconds=10)
        limiter.check("k")

        clock.now = _START + 4.0

        assert limiter.check("k") == (False, 0, 6)

    @pytest.mark.parametrize(("elapsed", "wait"), _ROUNDING_POINTS)
    def test_a_rejection_is_the_ceiling_at_every_rounding_point(self, clock: _Clock, elapsed: float, wait: int) -> None:
        """The rejected path at ``_ROUNDING_POINTS``, where a wrong rounding stops agreeing with ``ceil``."""
        limiter = RateLimiter(requests_per_minute=1, window_seconds=10)
        limiter.check("k")

        clock.now = _START + elapsed

        assert limiter.check("k") == (False, 0, wait)

    @pytest.mark.parametrize(("elapsed", "wait"), _LONG_WINDOW_POINTS)
    def test_a_rejection_in_a_long_window_is_not_capped(self, clock: _Clock, elapsed: float, wait: int) -> None:
        """The rejected path in a 120 s window: a wait above 60 comes back whole."""
        limiter = RateLimiter(requests_per_minute=1, window_seconds=120)
        limiter.check("k")

        clock.now = _START + elapsed

        assert limiter.check("k") == (False, 0, wait)

    def test_a_fresh_block_reports_the_whole_window(self, clock: _Clock) -> None:
        """The floor of 1 must not flatten every answer to 1."""
        limiter = RateLimiter(requests_per_minute=1, window_seconds=60)
        limiter.check("k")

        assert limiter.check("k") == (False, 0, 60)

    def test_the_allowed_path_rounds_the_same_way(self, clock: _Clock) -> None:
        """The allowed path feeds ``X-RateLimit-Reset``. It carried the same truncation, to 0 at the end."""
        limiter = RateLimiter(requests_per_minute=5, window_seconds=60)
        assert limiter.check("k") == (True, 4, 60)

        clock.now = _START + 0.4
        assert limiter.check("k") == (True, 3, 60)

        clock.now = _START + 59.99
        assert limiter.check("k") == (True, 2, 1)

    @pytest.mark.parametrize(("elapsed", "wait"), _ROUNDING_POINTS)
    def test_an_allowed_check_is_the_ceiling_at_every_rounding_point(self, clock: _Clock, elapsed: float, wait: int) -> None:
        """The allowed path at the same points, with a budget of 5.

        ``check`` has one ``return`` for each path, so one can round wrongly while the other
        is right: the rejected path's cases do not cover this one.
        """
        limiter = RateLimiter(requests_per_minute=5, window_seconds=10)
        assert limiter.check("k") == (True, 4, 10)

        clock.now = _START + elapsed

        assert limiter.check("k") == (True, 3, wait)

    @pytest.mark.parametrize(("elapsed", "wait"), _LONG_WINDOW_POINTS)
    def test_an_allowed_check_in_a_long_window_is_not_capped(self, clock: _Clock, elapsed: float, wait: int) -> None:
        """The allowed path in a 120 s window, with a budget of 5."""
        limiter = RateLimiter(requests_per_minute=5, window_seconds=120)
        assert limiter.check("k") == (True, 4, 120)

        clock.now = _START + elapsed

        assert limiter.check("k") == (True, 3, wait)

    def test_at_exactly_the_window_the_count_starts_over(self, clock: _Clock) -> None:
        """At ``window`` seconds the old count no longer applies, and the new window reports all of itself."""
        limiter = RateLimiter(requests_per_minute=1, window_seconds=10)
        limiter.check("k")

        clock.now = _START + 10.0

        assert limiter.check("k") == (True, 0, 10)

    def test_the_helper_floors_a_zero_remainder_at_one(self) -> None:
        """``_seconds_until_reset`` answers 1 when no time is left: ``ceil`` alone would say 0.

        Called directly, because with any window a float holds exactly ``check`` cannot get
        here. Once ``elapsed >= window`` it starts a new window and returns before the helper
        runs (the test above), so the remainder it hands over is positive and ``ceil`` is
        already at least 1. At ``elapsed == window`` the remainder is 0, ``ceil`` gives 0, and
        only the floor makes that 1. ``TestFailedAuthThrottleRoundsUp`` reaches the throttle's
        floor through ``check``, with a window no float holds.
        """
        limiter = RateLimiter(requests_per_minute=1, window_seconds=10)

        assert limiter._seconds_until_reset(_START + 10.0, _START) == 1


class TestRateLimiterHeaders:
    """The headers a caller receives, through ``SecurityMiddleware``.

    Asserting on ``check`` alone would pass even if the middleware formatted a different
    value into the response. ``X-RateLimit-Remaining`` (the requests left in the window after
    this one; 0 on a 429) is read with ``.get``, so a response that has lost the header fails
    the comparison instead of the lookup.
    """

    def test_a_429_in_the_last_second_says_retry_after_one(self, clock: _Clock) -> None:
        client = TestClient(_app(rate_limiter=RateLimiter(requests_per_minute=1, window_seconds=60)))
        first = client.get("/v1/datasets")
        assert first.status_code == 200
        assert first.headers["X-RateLimit-Reset"] == "60"
        assert first.headers.get("X-RateLimit-Remaining") == "0"

        clock.now = _START + 59.5
        response = client.get("/v1/datasets")

        assert response.status_code == 429
        assert response.headers["Retry-After"] == "1"
        assert response.headers["X-RateLimit-Reset"] == "1"
        assert response.headers.get("X-RateLimit-Remaining") == "0"
        assert response.json()["detail"] == "Rate limit exceeded. Try again in 1 seconds."

    def test_allowed_and_rejected_headers_name_the_same_rounded_instant(self, clock: _Clock) -> None:
        client = TestClient(_app(rate_limiter=RateLimiter(requests_per_minute=2, window_seconds=60)))
        assert client.get("/v1/datasets").headers["X-RateLimit-Reset"] == "60"

        clock.now = _START + 59.5
        allowed = client.get("/v1/datasets")
        rejected = client.get("/v1/datasets")

        assert allowed.status_code == 200
        assert allowed.headers.get("X-RateLimit-Remaining") == "0"
        assert allowed.headers["X-RateLimit-Reset"] == "1"
        assert rejected.status_code == 429
        assert rejected.headers["Retry-After"] == "1"
        assert rejected.headers["X-RateLimit-Reset"] == "1"
        assert rejected.headers.get("X-RateLimit-Remaining") == "0"

    def test_a_429_early_in_the_window_carries_the_whole_wait(self, clock: _Clock) -> None:
        """9.4 s left: the 429 says 10 in ``Retry-After``, in ``X-RateLimit-Reset`` and in its message.

        The two tests above read a 429 only where the answer is 1. An answer that is always
        1, a message or a header one second short and floored at 1, and a middleware that
        rewrites ``Retry-After`` to 1 all pass them.
        """
        client = TestClient(_app(rate_limiter=RateLimiter(requests_per_minute=1, window_seconds=10)))
        first = client.get("/v1/datasets")
        assert first.status_code == 200
        assert first.headers.get("X-RateLimit-Remaining") == "0"

        clock.now = _START + 0.6
        response = client.get("/v1/datasets")

        assert response.status_code == 429
        assert response.headers["Retry-After"] == "10"
        assert response.headers["X-RateLimit-Reset"] == "10"
        assert response.headers.get("X-RateLimit-Remaining") == "0"
        assert response.json()["detail"] == "Rate limit exceeded. Try again in 10 seconds."

    def test_an_allowed_response_late_in_the_window_says_one(self, clock: _Clock) -> None:
        """0.3 s left in a 10 s window, with budget to spare: the allowed response's ``X-RateLimit-Reset`` is 1, not 0."""
        client = TestClient(_app(rate_limiter=RateLimiter(requests_per_minute=5, window_seconds=10)))
        first = client.get("/v1/datasets")
        assert first.headers.get("X-RateLimit-Remaining") == "4"
        assert first.headers["X-RateLimit-Reset"] == "10"

        clock.now = _START + 9.7
        response = client.get("/v1/datasets")

        assert response.status_code == 200
        assert response.headers.get("X-RateLimit-Remaining") == "3"
        assert response.headers["X-RateLimit-Reset"] == "1"

    def test_a_wait_above_a_minute_reaches_the_caller_whole(self, clock: _Clock) -> None:
        """89.75 s left in a 120 s window: the allowed response and the 429 both say 90.

        Every other window read through the middleware is 10 s or 60 s, so a value capped at
        60 on its way out would pass them.
        """
        client = TestClient(_app(rate_limiter=RateLimiter(requests_per_minute=2, window_seconds=120)))
        first = client.get("/v1/datasets")
        assert first.headers.get("X-RateLimit-Remaining") == "1"
        assert first.headers["X-RateLimit-Reset"] == "120"

        clock.now = _START + 30.25
        allowed = client.get("/v1/datasets")
        rejected = client.get("/v1/datasets")

        assert allowed.status_code == 200
        assert allowed.headers.get("X-RateLimit-Remaining") == "0"
        assert allowed.headers["X-RateLimit-Reset"] == "90"
        assert rejected.status_code == 429
        assert rejected.headers["Retry-After"] == "90"
        assert rejected.headers["X-RateLimit-Reset"] == "90"
        assert rejected.headers.get("X-RateLimit-Remaining") == "0"
        assert rejected.json()["detail"] == "Rate limit exceeded. Try again in 90 seconds."


class TestFailedAuthThrottleRoundsUp:
    """``FailedAuthThrottle.check``'s ``retry_after``, which feeds the pre-auth 429's ``Retry-After``."""

    def test_blocked_retry_after_rounds_up(self, clock: _Clock) -> None:
        """9.4 s left is 10, not 9. Truncation and round-half-up both say 9."""
        throttle = FailedAuthThrottle(max_failures=1, window_seconds=10)
        throttle.record_failure("1.2.3.4")

        clock.now = _START + 0.6

        assert throttle.check("1.2.3.4") == (True, 10)

    def test_an_exact_remainder_is_not_bumped(self, clock: _Clock) -> None:
        """6.0 s left stays 6. A repair that always adds one second fails here."""
        throttle = FailedAuthThrottle(max_failures=1, window_seconds=10)
        throttle.record_failure("1.2.3.4")

        clock.now = _START + 4.0

        assert throttle.check("1.2.3.4") == (True, 6)

    @pytest.mark.parametrize(("elapsed", "wait"), _ROUNDING_POINTS)
    def test_a_blocked_wait_is_the_ceiling_at_every_rounding_point(self, clock: _Clock, elapsed: float, wait: int) -> None:
        """The throttle has its own copy of the formula, so it gets the same points."""
        throttle = FailedAuthThrottle(max_failures=1, window_seconds=10)
        throttle.record_failure("1.2.3.4")

        clock.now = _START + elapsed

        assert throttle.check("1.2.3.4") == (True, wait)

    @pytest.mark.parametrize(("elapsed", "wait"), _LONG_WINDOW_POINTS)
    def test_a_blocked_wait_in_a_long_window_is_not_capped(self, clock: _Clock, elapsed: float, wait: int) -> None:
        """The throttle in a 120 s window: its own copy of the formula must not cap the wait either."""
        throttle = FailedAuthThrottle(max_failures=1, window_seconds=120)
        throttle.record_failure("1.2.3.4")

        clock.now = _START + elapsed

        assert throttle.check("1.2.3.4") == (True, wait)

    @pytest.mark.parametrize("elapsed", [59.30, 59.60, 59.90, 59.99])
    def test_a_subsecond_remainder_is_one_second(self, clock: _Clock, elapsed: float) -> None:
        """A sub-second remainder is 1, never 0.

        The old ``max(1, int(...))`` gave 1 here too, so this passes on the old code: it
        pins the floor under the new rounding.
        """
        throttle = FailedAuthThrottle(max_failures=1, window_seconds=60)
        throttle.record_failure("1.2.3.4")

        clock.now = _START + elapsed

        assert throttle.check("1.2.3.4") == (True, 1)

    def test_a_fresh_block_reports_the_whole_window(self, clock: _Clock) -> None:
        throttle = FailedAuthThrottle(max_failures=1, window_seconds=60)
        throttle.record_failure("1.2.3.4")

        assert throttle.check("1.2.3.4") == (True, 60)

    def test_a_zero_remainder_is_floored_at_one(self, clock: _Clock) -> None:
        """``check`` answers 1 when the subtraction leaves no remainder: ``ceil`` alone would say 0.

        ``FailedAuthThrottle`` has no helper to call, and with any window a float holds exactly
        ``check`` returns on the rolled-over window before a remainder of 0 can reach the
        formula. A window of 2**53 + 1 is not one: no float holds that int. Python compares an
        int with a float exactly, so 2.0**53 seconds in is still inside the window, but it
        subtracts them as floats, and as a float 2**53 + 1 is 2.0**53: the remainder is 0.0.
        Nobody configures such a window. It is here because nothing else tells
        ``max(1, ceil(...))`` from ``ceil(...)`` through ``check``. The old ``max(1, int(...))``
        gave 1 here too, so this passes on the old code.
        """
        throttle = FailedAuthThrottle(max_failures=1, window_seconds=2**53 + 1)
        clock.now = 0.0
        throttle.record_failure("1.2.3.4")

        clock.now = 2.0**53

        assert throttle.check("1.2.3.4") == (True, 1)

    def test_the_pre_auth_429_header_is_the_rounded_up_wait(self, clock: _Clock) -> None:
        """The key is valid and the identity-keyed limiter is far from its budget, so this 429 is the throttle's."""
        throttle = build_failed_auth_throttle(max_failures=1, window_seconds=10)
        throttle.record_failure("testclient")
        clock.now = _START + 0.6
        app = _app(rate_limiter=RateLimiter(requests_per_minute=1000, window_seconds=60), api_keys=["s3cret"], throttle=throttle)

        response = TestClient(app).get("/v1/datasets", headers={"X-API-Key": "s3cret"})

        assert response.status_code == 429
        assert response.headers["Retry-After"] == "10"
        assert response.json()["detail"] == "Too many failed authentication attempts. Try again in 10 seconds."

    def test_the_pre_auth_429_carries_a_wait_above_a_minute_whole(self, clock: _Clock) -> None:
        """89.75 s left in a 120 s throttle window: ``Retry-After`` and the message say 90, not a capped 60."""
        throttle = build_failed_auth_throttle(max_failures=1, window_seconds=120)
        throttle.record_failure("testclient")
        clock.now = _START + 30.25
        app = _app(rate_limiter=RateLimiter(requests_per_minute=1000, window_seconds=60), api_keys=["s3cret"], throttle=throttle)

        response = TestClient(app).get("/v1/datasets", headers={"X-API-Key": "s3cret"})

        assert response.status_code == 429
        assert response.headers["Retry-After"] == "90"
        assert response.json()["detail"] == "Too many failed authentication attempts. Try again in 90 seconds."
