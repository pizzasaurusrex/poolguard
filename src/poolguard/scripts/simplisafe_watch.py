"""Watch the SimpliSafe camera's live stream and report its health.

Usage:
    poolguard-simplisafe-watch

Reads POOLGUARD_SIMPLISAFE_* and POOLGUARD_ALERT_* from the environment or
.env. Runs until interrupted, logging a line a minute with the frame rate.
If POOLGUARD_ALERT_HEARTBEAT_PUSH_URL is set it also pings Uptime Kuma: UP
while frames are fresh, DOWN once the stream has been blind for
POOLGUARD_ALERT_BLIND_AFTER_SECONDS.

No detection runs here yet. This is the bench source on its own, so its
uptime can be measured before the pipeline is wired to it.
"""

import logging
import time
from collections.abc import Callable
from typing import Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field

from poolguard.alerts.heartbeat import (
    HeartbeatError,
    HeartbeatPinger,
    HeartbeatReport,
    HeartbeatStatus,
    report_for_frame_age,
)
from poolguard.config import AlertSettings, SimpliSafeSettings
from poolguard.vision.frames import Frame

logger = logging.getLogger(__name__)

POLL_SECONDS = 1.0
REPORT_INTERVAL_SECONDS = 60.0

Ping = Callable[[HeartbeatReport], None]


class WatchableSource(Protocol):
    @property
    def dropped_frames(self) -> int: ...

    def poll(self, timeout_seconds: float) -> Frame | None: ...

    def seconds_since_last_frame(self) -> float: ...


class WatchSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    blind_after_seconds: float = Field(gt=0)
    heartbeat_interval_seconds: float = Field(gt=0)
    report_interval_seconds: float = Field(gt=0)


def watch(
    source: WatchableSource,
    ping: Ping | None,
    settings: WatchSettings,
    *,
    clock: Callable[[], float] = time.monotonic,
    should_stop: Callable[[], bool] = lambda: False,
) -> None:
    """Consume frames, logging the rate and reporting health until told to stop."""
    last_ping_at = last_report_at = clock()
    frames_since_report = 0
    while not should_stop():
        if source.poll(POLL_SECONDS) is not None:
            frames_since_report += 1
        now = clock()
        if now - last_ping_at >= settings.heartbeat_interval_seconds:
            last_ping_at = now
            _report_health(source, ping, settings.blind_after_seconds)
        if now - last_report_at >= settings.report_interval_seconds:
            fps = frames_since_report / (now - last_report_at)
            logger.info(
                "%.1f fps, last frame %.0f s ago, %d frames dropped unread",
                fps,
                source.seconds_since_last_frame(),
                source.dropped_frames,
            )
            last_report_at = now
            frames_since_report = 0


def _report_health(source: WatchableSource, ping: Ping | None, blind_after_seconds: float) -> None:
    report = report_for_frame_age(
        source.seconds_since_last_frame(), blind_after_seconds=blind_after_seconds
    )
    if report.status is HeartbeatStatus.DOWN:
        logger.error("stream is blind: %s", report.message)
    if ping is None:
        return
    try:
        ping(report)
    except HeartbeatError as error:
        # A monitoring outage must never stop the watcher.
        logger.warning("heartbeat not delivered: %s", error)


def main() -> None:
    # Imported here so the loop above stays importable without the `simplisafe` extra.
    from poolguard.vision.simplisafe import simplisafe_frame_source

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # The SDK logs connection URLs and room names at INFO.
    logging.getLogger("livekit").setLevel(logging.WARNING)
    alerts = AlertSettings()
    source = simplisafe_frame_source(SimpliSafeSettings())
    settings = WatchSettings(
        blind_after_seconds=alerts.blind_after_seconds,
        heartbeat_interval_seconds=alerts.heartbeat_interval_seconds,
        report_interval_seconds=REPORT_INTERVAL_SECONDS,
    )
    try:
        with httpx.Client() as client:
            ping: Ping | None = None
            if alerts.heartbeat_push_url is not None:
                pinger = HeartbeatPinger(alerts.heartbeat_push_url, client)
                ping = lambda report: pinger.ping(report.status, report.message)  # noqa: E731
            else:
                logger.info("no heartbeat URL configured; health is logged only")
            watch(source, ping, settings)
    except KeyboardInterrupt:
        logger.info("stopped")
    finally:
        source.close()


if __name__ == "__main__":
    main()
