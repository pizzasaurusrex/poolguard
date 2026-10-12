"""Kuma push heartbeat, exercised against a mock transport."""

import httpx
import pytest
from pydantic import SecretStr

from poolguard.alerts.heartbeat import HeartbeatError, HeartbeatPinger, HeartbeatStatus

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
