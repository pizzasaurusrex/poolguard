"""Tests for the two-pass IoU tracker.

The scenarios mirror the safety cases from PLAN.md: ID persistence while a
person is visible, low-confidence rescue while they splash or slip under,
and coasting (seconds_since_last_seen growing) once they disappear — the
raw signal for the submersion rule.
"""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from poolguard.config import TrackingSettings
from poolguard.events import BoundingBox, Detection
from poolguard.tracking import TrackerState, advance, iou

START = datetime(2026, 1, 1, tzinfo=UTC)

SETTINGS = TrackingSettings(
    high_confidence=0.5,
    low_confidence=0.1,
    iou_min=0.3,
    max_coast_seconds=30.0,
    pool_zone=(0.0, 0.0, 1.0, 1.0),
)


def det(
    x: float,
    y: float,
    confidence: float = 0.9,
    width: float = 0.1,
    height: float = 0.2,
    ts: datetime = START,
) -> Detection:
    return Detection(
        frame_ts=ts,
        box=BoundingBox(x=x, y=y, width=width, height=height),
        confidence=confidence,
    )


class TestIou:
    def test_identical_boxes(self) -> None:
        box = BoundingBox(x=0.1, y=0.1, width=0.2, height=0.2)
        assert iou(box, box) == pytest.approx(1.0)

    def test_disjoint_boxes(self) -> None:
        a = BoundingBox(x=0.0, y=0.0, width=0.1, height=0.1)
        b = BoundingBox(x=0.5, y=0.5, width=0.1, height=0.1)
        assert iou(a, b) == 0.0

    def test_half_overlap(self) -> None:
        a = BoundingBox(x=0.0, y=0.0, width=0.2, height=0.2)
        b = BoundingBox(x=0.1, y=0.0, width=0.2, height=0.2)
        # intersection 0.1x0.2 = 0.02, union 0.08 - 0.02 = 0.06
        assert iou(a, b) == pytest.approx(0.02 / 0.06)


class TestNewTracks:
    def test_high_confidence_detections_spawn_tracks(self) -> None:
        state, people = advance(
            TrackerState(), (det(0.1, 0.1), det(0.6, 0.6)), START, SETTINGS
        )

        assert sorted(p.track_id for p in people) == [1, 2]
        assert all(p.seconds_since_last_seen == 0.0 for p in people)
        assert len(state.tracks) == 2

    def test_low_confidence_detection_never_spawns(self) -> None:
        state, people = advance(
            TrackerState(), (det(0.1, 0.1, confidence=0.2),), START, SETTINGS
        )

        assert people == ()
        assert state.tracks == ()


class TestIdPersistence:
    def test_same_person_keeps_id_across_frames(self) -> None:
        state, _ = advance(TrackerState(), (det(0.1, 0.1),), START, SETTINGS)
        later = START + timedelta(seconds=1)
        state, people = advance(
            state, (det(0.12, 0.11, ts=later),), later, SETTINGS
        )

        assert [p.track_id for p in people] == [1]
        assert people[0].seconds_since_last_seen == 0.0

    def test_two_people_keep_distinct_ids(self) -> None:
        state, first = advance(
            TrackerState(), (det(0.1, 0.1), det(0.6, 0.6)), START, SETTINGS
        )
        by_pos = {round(p.detection.box.x, 1): p.track_id for p in first}

        later = START + timedelta(seconds=1)
        # Same two people, slightly moved, reported in swapped order.
        state, people = advance(
            state,
            (det(0.61, 0.61, ts=later), det(0.11, 0.11, ts=later)),
            later,
            SETTINGS,
        )

        for person in people:
            assert person.track_id == by_pos[round(person.detection.box.x, 1)]

    def test_strongest_overlap_wins_when_tracks_compete(self) -> None:
        # Detection at 0.15 overlaps track 0 (at 0.10) just over iou_min but
        # overlaps track 1 (at 0.16) strongly. Track-order matching lets
        # track 0 steal it before track 1 gets a turn; strongest-first must
        # give it to track 1 and leave the 0.09 detection for track 0.
        state, first = advance(
            TrackerState(), (det(0.10, 0.1), det(0.16, 0.1)), START, SETTINGS
        )
        ids = {round(p.detection.box.x, 2): p.track_id for p in first}

        later = START + timedelta(seconds=1)
        state, people = advance(
            state,
            (det(0.15, 0.1, ts=later), det(0.09, 0.1, ts=later)),
            later,
            SETTINGS,
        )
        new_ids = {round(p.detection.box.x, 2): p.track_id for p in people}

        assert new_ids[0.15] == ids[0.16]  # strongest claim kept its owner
        assert new_ids[0.09] == ids[0.10]

    def test_one_detection_cannot_satisfy_two_tracks(self) -> None:
        # Two people converge; only one box comes back. The other track must
        # coast (its timer growing), not silently share the survivor's box.
        state, _ = advance(
            TrackerState(), (det(0.10, 0.1), det(0.18, 0.1)), START, SETTINGS
        )

        later = START + timedelta(seconds=1)
        state, people = advance(state, (det(0.14, 0.1, ts=later),), later, SETTINGS)

        assert len(people) == 2  # both tracks still alive
        timers = sorted(p.seconds_since_last_seen for p in people)
        assert timers[0] == 0.0  # one track got the detection
        assert timers[1] == pytest.approx(1.0)  # the other is coasting

    def test_distant_detection_gets_new_id(self) -> None:
        state, _ = advance(TrackerState(), (det(0.1, 0.1),), START, SETTINGS)
        later = START + timedelta(seconds=1)
        state, people = advance(state, (det(0.8, 0.8, ts=later),), later, SETTINGS)

        assert sorted(p.track_id for p in people) == [1, 2]  # old coasts, new spawns


class TestLowConfidenceRescue:
    def test_weak_detection_keeps_existing_track_locked(self) -> None:
        state, _ = advance(TrackerState(), (det(0.1, 0.1),), START, SETTINGS)
        later = START + timedelta(seconds=1)
        # Splashing: same place, confidence collapsed below high threshold.
        state, people = advance(
            state, (det(0.11, 0.1, confidence=0.2, ts=later),), later, SETTINGS
        )

        assert [p.track_id for p in people] == [1]
        assert people[0].seconds_since_last_seen == 0.0
        assert people[0].detection.confidence == pytest.approx(0.2)


class TestCoasting:
    def test_unseen_track_coasts_with_growing_timer(self) -> None:
        state, _ = advance(TrackerState(), (det(0.1, 0.1),), START, SETTINGS)
        later = START + timedelta(seconds=10)
        state, people = advance(state, (), later, SETTINGS)

        assert [p.track_id for p in people] == [1]
        assert people[0].seconds_since_last_seen == pytest.approx(10.0)

    def test_track_expires_after_max_coast(self) -> None:
        state, _ = advance(TrackerState(), (det(0.1, 0.1),), START, SETTINGS)
        later = START + timedelta(seconds=31)
        state, people = advance(state, (), later, SETTINGS)

        assert people == ()
        assert state.tracks == ()


class TestInWater:
    def test_zone_membership_by_box_center(self) -> None:
        settings = SETTINGS.model_copy(
            update={"pool_zone": (0.0, 0.0, 0.5, 1.0)}
        )
        _, people = advance(
            TrackerState(), (det(0.1, 0.1), det(0.7, 0.1)), START, settings
        )
        by_x = {round(p.detection.box.x, 1): p.in_water for p in people}

        assert by_x[0.1] is True
        assert by_x[0.7] is False


class TestImmutability:
    def test_advance_returns_new_state(self) -> None:
        empty = TrackerState()
        state, _ = advance(empty, (det(0.1, 0.1),), START, SETTINGS)

        assert empty.tracks == ()
        assert state is not empty

    def test_tracker_state_is_frozen(self) -> None:
        state = TrackerState()
        with pytest.raises(ValidationError):
            state.next_track_id = 5  # type: ignore[misc]
