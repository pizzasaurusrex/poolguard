from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from poolguard.events import (
    AlertTier,
    BoundingBox,
    Detection,
    EventType,
    OperatingMode,
    SafetyEvent,
    TrackedPerson,
)


def make_detection() -> Detection:
    return Detection(
        frame_ts=datetime(2026, 7, 24, 12, 0, 0, tzinfo=UTC),
        box=BoundingBox(x=0.4, y=0.5, width=0.1, height=0.2),
        confidence=0.9,
    )


class TestBoundingBox:
    def test_rejects_out_of_range_coordinates(self):
        with pytest.raises(ValidationError):
            BoundingBox(x=1.5, y=0.5, width=0.1, height=0.2)

    def test_rejects_zero_size(self):
        with pytest.raises(ValidationError):
            BoundingBox(x=0.1, y=0.1, width=0.0, height=0.2)

    def test_is_immutable(self):
        box = BoundingBox(x=0.1, y=0.1, width=0.2, height=0.2)
        with pytest.raises(ValidationError):
            box.x = 0.9


class TestDetection:
    def test_rejects_confidence_above_one(self):
        with pytest.raises(ValidationError):
            Detection(
                frame_ts=datetime(2026, 7, 24, tzinfo=UTC),
                box=BoundingBox(x=0.1, y=0.1, width=0.2, height=0.2),
                confidence=1.1,
            )

    def test_keypoints_optional(self):
        assert make_detection().keypoints is None


class TestSafetyEvent:
    def test_round_trips_through_json(self):
        event = SafetyEvent(
            event_type=EventType.SUBMERSION,
            tier=AlertTier.EMERGENCY,
            started_at=datetime(2026, 7, 24, 12, 0, 5, tzinfo=UTC),
            track_id=3,
            mode=OperatingMode.SWIM,
            confidence=0.85,
            detail="track 3 not resurfaced after 15s",
        )
        restored = SafetyEvent.model_validate_json(event.model_dump_json())
        assert restored == event

    def test_is_immutable(self):
        event = SafetyEvent(
            event_type=EventType.DISTRESS,
            tier=AlertTier.WARN,
            started_at=datetime(2026, 7, 24, tzinfo=UTC),
            mode=OperatingMode.ARMED,
            confidence=0.5,
            detail="sustained vertical flailing",
        )
        with pytest.raises(ValidationError):
            event.tier = AlertTier.EMERGENCY


class TestTrackedPerson:
    def test_rejects_negative_staleness(self):
        with pytest.raises(ValidationError):
            TrackedPerson(
                track_id=1,
                detection=make_detection(),
                in_water=True,
                seconds_since_last_seen=-1.0,
            )
