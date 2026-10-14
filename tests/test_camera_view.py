"""Tests for the camera-view verdict and the degraded event it raises.

The measuring (matching landmarks between two images) needs OpenCV and lives
in vision/view_compare.py. What is tested here is the dependency-free part:
turning those measurements into a verdict, and the verdict into an alert.
"""

from datetime import UTC, datetime

import pytest

from poolguard.camera_view import ViewCheck, ViewVerdict, classify_view, degraded_event
from poolguard.config import ViewSettings
from poolguard.events import AlertTier, EventType, OperatingMode

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
SETTINGS = ViewSettings(
    _env_file=None, min_landmarks=20, min_landmark_spread=0.3, max_zone_shift=0.02
)


class TestClassifyView:
    def test_plenty_of_landmarks_and_no_shift_is_aligned(self) -> None:
        check = classify_view(
            matched_landmarks=300, landmark_spread=0.8, zone_shift=0.004, settings=SETTINGS
        )
        assert check.verdict is ViewVerdict.ALIGNED

    def test_shift_beyond_the_limit_is_moved(self) -> None:
        check = classify_view(
            matched_landmarks=300, landmark_spread=0.8, zone_shift=0.05, settings=SETTINGS
        )
        assert check.verdict is ViewVerdict.MOVED
        assert check.zone_shift == pytest.approx(0.05)

    def test_too_few_landmarks_is_unknown(self) -> None:
        """A handful of matches cannot say where the outline went; do not trust them."""
        check = classify_view(
            matched_landmarks=6, landmark_spread=0.8, zone_shift=0.0, settings=SETTINGS
        )
        assert check.verdict is ViewVerdict.UNKNOWN

    def test_landmarks_bunched_in_one_spot_is_unknown(self) -> None:
        """A camera's burned-in logo matches perfectly at zero shift after any move.

        Requiring the matches to be spread across the frame stops that small
        cluster from vouching for the whole view.
        """
        check = classify_view(
            matched_landmarks=60, landmark_spread=0.1, zone_shift=0.0, settings=SETTINGS
        )
        assert check.verdict is ViewVerdict.UNKNOWN

    def test_no_measurable_shift_is_unknown(self) -> None:
        check = classify_view(
            matched_landmarks=0, landmark_spread=0.0, zone_shift=None, settings=SETTINGS
        )
        assert check.verdict is ViewVerdict.UNKNOWN


class TestDegradedEvent:
    @staticmethod
    def check(verdict: ViewVerdict, zone_shift: float | None = 0.0) -> ViewCheck:
        return ViewCheck(
            verdict=verdict, matched_landmarks=100, landmark_spread=0.8, zone_shift=zone_shift
        )

    def test_aligned_view_raises_nothing(self) -> None:
        assert degraded_event(self.check(ViewVerdict.ALIGNED), NOW, OperatingMode.ARMED) is None

    def test_moved_view_raises_a_degraded_warning(self) -> None:
        event = degraded_event(self.check(ViewVerdict.MOVED, 0.08), NOW, OperatingMode.ARMED)

        assert event is not None
        assert event.event_type is EventType.SYSTEM_DEGRADED
        assert event.tier is AlertTier.WARN
        assert event.started_at == NOW
        assert event.mode is OperatingMode.ARMED
        assert "moved" in event.detail
        assert "8%" in event.detail

    def test_unknown_view_also_raises(self) -> None:
        """Not being able to confirm the view is a reason to say so, not to stay quiet."""
        event = degraded_event(self.check(ViewVerdict.UNKNOWN, None), NOW, OperatingMode.SWIM)

        assert event is not None
        assert event.event_type is EventType.SYSTEM_DEGRADED
        assert "could not be confirmed" in event.detail
