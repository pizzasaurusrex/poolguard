"""Send one real test notification through each configured alert channel.

Usage:
    poolguard-alert-test

Reads POOLGUARD_ALERT_* from the environment or .env. A channel with no URL
configured is reported as skipped. Exits non-zero if any configured channel
fails, so it can gate a deploy.
"""

from datetime import UTC, datetime
from enum import StrEnum

import httpx
from pydantic import BaseModel, ConfigDict

from poolguard.alerts.heartbeat import HeartbeatError, HeartbeatPinger
from poolguard.alerts.ntfy import NtfyDeliveryError, NtfyNotifier, build_message
from poolguard.config import AlertSettings
from poolguard.events import AlertTier, EventType, OperatingMode, SafetyEvent


class Outcome(StrEnum):
    SENT = "sent"
    SKIPPED = "skipped"
    FAILED = "failed"


class ChannelResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    channel: str
    outcome: Outcome
    detail: str = ""


def _test_event() -> SafetyEvent:
    return SafetyEvent(
        event_type=EventType.SYSTEM_DEGRADED,
        tier=AlertTier.WARN,
        started_at=datetime.now(UTC).astimezone(),
        mode=OperatingMode.MAINTENANCE,
        confidence=1.0,
        detail="Test notification, no action needed",
    )


def _check_ntfy(settings: AlertSettings, client: httpx.Client) -> ChannelResult:
    if settings.ntfy_topic_url is None:
        return ChannelResult(
            channel="ntfy",
            outcome=Outcome.SKIPPED,
            detail="POOLGUARD_ALERT_NTFY_TOPIC_URL not set",
        )
    try:
        NtfyNotifier(settings.ntfy_topic_url, client).send(build_message(_test_event()))
    except NtfyDeliveryError as error:
        return ChannelResult(channel="ntfy", outcome=Outcome.FAILED, detail=str(error))
    return ChannelResult(channel="ntfy", outcome=Outcome.SENT)


def _check_heartbeat(settings: AlertSettings, client: httpx.Client) -> ChannelResult:
    if settings.heartbeat_push_url is None:
        return ChannelResult(
            channel="heartbeat",
            outcome=Outcome.SKIPPED,
            detail="POOLGUARD_ALERT_HEARTBEAT_PUSH_URL not set",
        )
    try:
        HeartbeatPinger(settings.heartbeat_push_url, client).ping(message="alert test")
    except HeartbeatError as error:
        return ChannelResult(channel="heartbeat", outcome=Outcome.FAILED, detail=str(error))
    return ChannelResult(channel="heartbeat", outcome=Outcome.SENT)


def run_checks(settings: AlertSettings, client: httpx.Client) -> tuple[ChannelResult, ...]:
    return (_check_ntfy(settings, client), _check_heartbeat(settings, client))


def main() -> None:
    with httpx.Client() as client:
        results = run_checks(AlertSettings(), client)

    for result in results:
        suffix = f" ({result.detail})" if result.detail else ""
        print(f"{result.channel:<10} {result.outcome.value}{suffix}")

    if any(result.outcome is Outcome.FAILED for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
