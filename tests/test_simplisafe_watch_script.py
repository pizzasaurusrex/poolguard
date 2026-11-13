"""The watch loop's heartbeat behaviour, with a fake source and a fake clock."""

import numpy as np
import pytest

from poolguard.alerts.heartbeat import HeartbeatError, HeartbeatReport, HeartbeatStatus
from poolguard.scripts.simplisafe_watch import Ping, WatchSettings, watch
from poolguard.vision.frames import REPLAY_EPOCH, Frame

SETTINGS = WatchSettings(
    blind_after_seconds=180.0, heartbeat_interval_seconds=30.0, report_interval_seconds=60.0
)
POLL_STEP_SECONDS = 1.0


class FakeSource:
    """Delivers a frame per poll until `blind_from`, then nothing."""

    def __init__(self, clock: "FakeClock", blind_from: float) -> None:
        self._clock = clock
        self._blind_from = blind_from
        self._last_frame_at = clock.now
        self.dropped_frames = 0

    def poll(self, timeout_seconds: float) -> Frame | None:
        self._clock.now += POLL_STEP_SECONDS
        if self._clock.now >= self._blind_from:
            return None
        self._last_frame_at = self._clock.now
        return Frame(image=np.zeros((2, 2, 3), dtype=np.uint8), ts=REPLAY_EPOCH, index=0)

    def seconds_since_last_frame(self) -> float:
        return self._clock.now - self._last_frame_at


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def run_for(seconds: float, blind_from: float, ping: Ping | None) -> None:
    clock = FakeClock()
    source = FakeSource(clock, blind_from)
    watch(source, ping, SETTINGS, clock=clock, should_stop=lambda: clock.now >= seconds)


def test_pings_up_on_the_heartbeat_interval_while_frames_arrive() -> None:
    reports: list[HeartbeatReport] = []
    run_for(95.0, blind_from=1e9, ping=reports.append)
    assert [report.status for report in reports] == [HeartbeatStatus.UP] * 3


def test_reports_down_once_blind_past_the_threshold() -> None:
    reports: list[HeartbeatReport] = []
    run_for(400.0, blind_from=60.0, ping=reports.append)
    statuses = [report.status for report in reports]
    first_down = statuses.index(HeartbeatStatus.DOWN)
    # Blind from 60 s, threshold 180 s: the first ping at or after 240 s is the 8th.
    assert first_down == 7
    assert all(status is HeartbeatStatus.DOWN for status in statuses[first_down:])


def test_a_failed_ping_is_logged_and_watching_continues(
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: list[HeartbeatReport] = []

    def failing_ping(report: HeartbeatReport) -> None:
        calls.append(report)
        raise HeartbeatError("monitor unreachable (ConnectError)")

    with caplog.at_level("WARNING"):
        run_for(95.0, blind_from=1e9, ping=failing_ping)
    assert len(calls) == 3
    assert "monitor unreachable" in caplog.text


def test_runs_without_a_heartbeat_configured() -> None:
    run_for(95.0, blind_from=1e9, ping=None)
