"""ntfy push channel for the watch and warn tiers.

ntfy is a plain HTTP POST to a topic URL. The topic name is the only secret:
anyone who knows it can read and post, so the URL is a SecretStr and never
appears in an error message or log line.

This channel needs the internet. It is never the only path for an emergency;
the siren covers that without a network (PRD §5).
"""

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from poolguard.events import AlertTier, SafetyEvent

NTFY_MIN_PRIORITY = 1
NTFY_MAX_PRIORITY = 5
NTFY_TIMEOUT_SECONDS = 5.0

_PRIORITY_BY_TIER: dict[AlertTier, int] = {
    # Silent: watch fires often, and a noisy watch tier gets the whole topic muted.
    AlertTier.WATCH: 2,
    AlertTier.WARN: 4,
    AlertTier.EMERGENCY: NTFY_MAX_PRIORITY,
}


class NtfyDeliveryError(RuntimeError):
    """The push did not reach ntfy. The caller decides whether to retry or escalate."""


class NtfyMessage(BaseModel):
    """One push notification, built from an event before anything touches the network."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(min_length=1)
    body: str = Field(min_length=1)
    priority: int = Field(ge=NTFY_MIN_PRIORITY, le=NTFY_MAX_PRIORITY)
    tags: tuple[str, ...] = ()


def priority_for(tier: AlertTier) -> int:
    """ntfy priority (1 = silent, 3 = default, 5 = urgent) for an alert tier."""
    return _PRIORITY_BY_TIER[tier]


def build_message(event: SafetyEvent) -> NtfyMessage:
    """Translate a rules-engine conclusion into what the phone shows."""
    return NtfyMessage(
        title=f"PoolGuard {event.tier.value}: {event.event_type.value.replace('_', ' ')}",
        body=f"{event.detail} ({event.started_at:%H:%M:%S}, mode {event.mode.value})",
        priority=priority_for(event.tier),
        tags=(event.tier.value, event.event_type.value),
    )


class NtfyNotifier:
    """Sends messages to one ntfy topic over an injected HTTP client."""

    def __init__(self, topic_url: SecretStr, client: httpx.Client) -> None:
        self._topic_url = topic_url
        self._client = client

    def send(self, message: NtfyMessage) -> None:
        headers = {
            "Title": message.title,
            "Priority": str(message.priority),
            "Tags": ",".join(message.tags),
        }
        try:
            response = self._client.post(
                self._topic_url.get_secret_value(),
                content=message.body.encode("utf-8"),
                headers=headers,
                timeout=NTFY_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            # `from None`: httpx puts the request URL in its message, and the URL is the secret.
            raise NtfyDeliveryError(
                f"ntfy rejected the push with HTTP {error.response.status_code}"
            ) from None
        except httpx.HTTPError as error:
            raise NtfyDeliveryError(f"ntfy unreachable ({type(error).__name__})") from None
