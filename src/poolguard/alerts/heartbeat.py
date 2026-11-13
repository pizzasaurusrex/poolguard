"""Dead-man's switch: tell Uptime Kuma the pipeline is still alive.

A Kuma "Push" monitor hands out a URL and alerts when requests to it stop.
The pipeline pings it on a timer, so a crashed or hung detector produces an
alert without the detector having to do anything, which is the one failure it
cannot report itself.

The push URL embeds the monitor's token, so it is a SecretStr and never
appears in an error message.
"""

from enum import StrEnum

import httpx
from pydantic import BaseModel, ConfigDict, SecretStr

HEARTBEAT_TIMEOUT_SECONDS = 5.0


class HeartbeatStatus(StrEnum):
    UP = "up"
    DOWN = "down"


class HeartbeatReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: HeartbeatStatus
    message: str


def report_for_frame_age(
    frame_age_seconds: float, *, blind_after_seconds: float
) -> HeartbeatReport:
    """What to tell the monitor, given how long it has been since the last frame.

    A process that is running but receiving no video must not look healthy,
    so UP is earned by fresh frames, not by being alive.
    """
    if frame_age_seconds >= blind_after_seconds:
        minutes = int(frame_age_seconds // 60)
        return HeartbeatReport(status=HeartbeatStatus.DOWN, message=f"no video for {minutes} min")
    return HeartbeatReport(status=HeartbeatStatus.UP, message="OK")


class HeartbeatError(RuntimeError):
    """The ping did not reach the monitor.

    Callers log this and keep running: a monitoring outage must never stop
    the detector, and the missing ping is itself what raises the alarm.
    """


class HeartbeatPinger:
    """Pings one Kuma push monitor over an injected HTTP client."""

    def __init__(self, push_url: SecretStr, client: httpx.Client) -> None:
        self._push_url = push_url
        self._client = client

    def ping(self, status: HeartbeatStatus = HeartbeatStatus.UP, message: str = "OK") -> None:
        """Report liveness, or report DOWN to alert at once instead of waiting for a timeout."""
        try:
            response = self._client.get(
                self._push_url.get_secret_value(),
                params={"status": status.value, "msg": message},
                timeout=HEARTBEAT_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            # `from None`: httpx puts the request URL in its message, and the URL holds the token.
            raise HeartbeatError(
                f"monitor rejected the heartbeat with HTTP {error.response.status_code}"
            ) from None
        except httpx.HTTPError as error:
            raise HeartbeatError(f"monitor unreachable ({type(error).__name__})") from None
