"""ntfy channel: message building is pure, delivery runs against a mock transport."""

from datetime import UTC, datetime

import httpx
import pytest
from pydantic import SecretStr

from poolguard.alerts.ntfy import (
    NTFY_MAX_PRIORITY,
    NTFY_MIN_PRIORITY,
    NtfyDeliveryError,
    NtfyMessage,
    NtfyNotifier,
    build_message,
    priority_for,
)
from poolguard.events import AlertTier, EventType, OperatingMode, SafetyEvent

TOPIC_URL = "https://ntfy.example/secret-topic-name"


def event(tier: AlertTier = AlertTier.WARN) -> SafetyEvent:
    return SafetyEvent(
        event_type=EventType.UNSUPERVISED_ENTRY,
        tier=tier,
        started_at=datetime(2026, 10, 11, 14, 30, 5, tzinfo=UTC),
        mode=OperatingMode.ARMED,
        confidence=0.8,
        detail="Person entered the water",
    )


def message() -> NtfyMessage:
    return NtfyMessage(title="PoolGuard warn", body="Person entered the water", priority=3)


def notifier(handler: httpx.MockTransport) -> NtfyNotifier:
    return NtfyNotifier(SecretStr(TOPIC_URL), httpx.Client(transport=handler))


class TestPriority:
    @pytest.mark.parametrize("tier", list(AlertTier))
    def test_every_tier_maps_into_ntfy_range(self, tier: AlertTier) -> None:
        assert NTFY_MIN_PRIORITY <= priority_for(tier) <= NTFY_MAX_PRIORITY

    def test_priority_rises_with_tier(self) -> None:
        """A louder tier must never arrive quieter than a softer one."""
        assert (
            priority_for(AlertTier.WATCH)
            < priority_for(AlertTier.WARN)
            < priority_for(AlertTier.EMERGENCY)
        )


class TestBuildMessage:
    def test_title_names_tier_and_event(self) -> None:
        built = build_message(event(AlertTier.WARN))
        assert built.title == "PoolGuard warn: unsupervised entry"

    def test_body_carries_detail_time_and_mode(self) -> None:
        built = build_message(event())
        assert "Person entered the water" in built.body
        assert "14:30:05" in built.body
        assert "armed" in built.body

    def test_priority_comes_from_the_tier(self) -> None:
        assert build_message(event(AlertTier.WATCH)).priority == priority_for(AlertTier.WATCH)


class TestSend:
    def test_posts_body_and_headers_to_the_topic(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200)

        notifier(httpx.MockTransport(handler)).send(
            NtfyMessage(title="T", body="hello", priority=4, tags=("warn", "distress"))
        )

        (request,) = seen
        assert request.method == "POST"
        assert str(request.url) == TOPIC_URL
        assert request.content == b"hello"
        assert request.headers["Title"] == "T"
        assert request.headers["Priority"] == "4"
        assert request.headers["Tags"] == "warn,distress"

    def test_http_error_raises_without_leaking_the_topic(self) -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(500))
        with pytest.raises(NtfyDeliveryError, match="HTTP 500") as raised:
            notifier(transport).send(message())
        assert "secret-topic-name" not in str(raised.value)
        assert raised.value.__cause__ is None

    def test_network_failure_raises_without_leaking_the_topic(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route", request=request)

        with pytest.raises(NtfyDeliveryError, match="unreachable") as raised:
            notifier(httpx.MockTransport(handler)).send(message())
        assert "secret-topic-name" not in str(raised.value)
