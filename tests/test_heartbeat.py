"""Kuma push heartbeat, exercised against a mock transport."""

import httpx
import pytest
from pydantic import SecretStr

from poolguard.alerts.heartbeat import (
    HeartbeatError,
    HeartbeatPinger,
    HeartbeatStatus,
    report_for_frame_age,
)

PUSH_URL = "http://kuma.example:3001/api/push/secret-token"


def pinger(transport: httpx.MockTransport) -> HeartbeatPinger:
    return HeartbeatPinger(SecretStr(PUSH_URL), httpx.Client(transport=transport))


def capture(seen: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True})

    return httpx.MockTransport(handler)


class TestPing:
    def test_reports_up_by_default(self) -> None:
        seen: list[httpx.Request] = []
        pinger(capture(seen)).ping()

        (request,) = seen
        assert request.method == "GET"
        assert request.url.path == "/api/push/secret-token"
        assert request.url.params["status"] == "up"
        assert request.url.params["msg"] == "OK"

    def test_reports_down_with_a_reason(self) -> None:
        seen: list[httpx.Request] = []
        pinger(capture(seen)).ping(HeartbeatStatus.DOWN, "camera offline")

        (request,) = seen
        assert request.url.params["status"] == "down"
        assert request.url.params["msg"] == "camera offline"

    def test_http_error_raises_without_leaking_the_token(self) -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(404))
        with pytest.raises(HeartbeatError, match="HTTP 404") as raised:
            pinger(transport).ping()
        assert "secret-token" not in str(raised.value)
        assert raised.value.__cause__ is None

    def test_network_failure_raises_without_leaking_the_token(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("timed out", request=request)

        with pytest.raises(HeartbeatError, match="unreachable") as raised:
            pinger(httpx.MockTransport(handler)).ping()
        assert "secret-token" not in str(raised.value)


class TestReportForFrameAge:
    def test_fresh_frames_report_up(self) -> None:
        report = report_for_frame_age(2.0, blind_after_seconds=180.0)
        assert report.status is HeartbeatStatus.UP

    def test_a_short_gap_is_tolerated(self) -> None:
        # Stalls up to 34 s recovered unaided on 2026-10-12; they must not alert.
        report = report_for_frame_age(179.0, blind_after_seconds=180.0)
        assert report.status is HeartbeatStatus.UP

    def test_a_long_gap_reports_down_and_says_how_long(self) -> None:
        report = report_for_frame_age(1290.0, blind_after_seconds=180.0)
        assert report.status is HeartbeatStatus.DOWN
        assert "21 min" in report.message

    def test_the_threshold_itself_counts_as_blind(self) -> None:
        report = report_for_frame_age(180.0, blind_after_seconds=180.0)
        assert report.status is HeartbeatStatus.DOWN
