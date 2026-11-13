"""Tests for the rules engine's submersion rule.

The rule's input is the tracker's output: a person last seen in the water who
has gone unseen for too long. These tests build TrackedPerson values directly,
so they pin the rule's behaviour without a tracker or a video.
"""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from poolguard.config import RulesSettings
from poolguard.events import (
    AlertTier,
    BoundingBox,
    Detection,
    EventType,
    OperatingMode,
    TrackedPerson,
)
from poolguard.rules import RulesState, evaluate

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
SETTINGS = RulesSettings(_env_file=None, submersion_timeout_seconds=12.0)
SWIM = OperatingMode.SWIM


def person(track_id: int = 1, unseen: float = 0.0, in_water: bool = True) -> TrackedPerson:
    return TrackedPerson(
        track_id=track_id,
        detection=Detection(
            frame_ts=NOW,
            box=BoundingBox(x=0.4, y=0.4, width=0.1, height=0.1),
            confidence=0.8,
        ),
        in_water=in_water,
        seconds_since_last_seen=unseen,
    )


class TestSubmersion:
    def test_visible_swimmer_raises_nothing(self) -> None:
        _, events = evaluate(RulesState(), (person(unseen=0.0),), NOW, SWIM, SETTINGS)
        assert events == ()

    def test_briefly_unseen_swimmer_raises_nothing(self) -> None:
        """Kids duck under all the time; the timeout is what separates that from trouble."""
        _, events = evaluate(RulesState(), (person(unseen=11.9),), NOW, SWIM, SETTINGS)
        assert events == ()

    def test_swimmer_unseen_for_the_timeout_raises_an_emergency(self) -> None:
        _, events = evaluate(RulesState(), (person(track_id=7, unseen=12.0),), NOW, SWIM, SETTINGS)

        assert len(events) == 1
        event = events[0]
        assert event.event_type is EventType.SUBMERSION
        assert event.tier is AlertTier.EMERGENCY
        assert event.track_id == 7
        assert event.mode is SWIM
        assert event.started_at == NOW - timedelta(seconds=12.0)
        assert "12" in event.detail

    def test_person_lost_outside_the_water_raises_nothing(self) -> None:
        """Walking out of frame coasts a track too; that is not a submersion."""
        lost_on_deck = person(unseen=25.0, in_water=False)
        _, events = evaluate(RulesState(), (lost_on_deck,), NOW, SWIM, SETTINGS)
        assert events == ()

    def test_each_submerged_person_gets_their_own_event(self) -> None:
        people = (person(track_id=1, unseen=13.0), person(track_id=2, unseen=14.0))
        _, events = evaluate(RulesState(), people, NOW, SWIM, SETTINGS)
        assert sorted(event.track_id for event in events) == [1, 2]


class TestFiresOncePerIncident:
    def test_event_is_not_repeated_while_the_person_stays_under(self) -> None:
        state, first = evaluate(RulesState(), (person(unseen=12.0),), NOW, SWIM, SETTINGS)
        later = NOW + timedelta(seconds=1)
        state, second = evaluate(state, (person(unseen=13.0),), later, SWIM, SETTINGS)

        assert len(first) == 1
        assert second == ()

    def test_resurfacing_rearms_the_rule(self) -> None:
        state, _ = evaluate(RulesState(), (person(unseen=12.0),), NOW, SWIM, SETTINGS)
        state, _ = evaluate(state, (person(unseen=0.0),), NOW, SWIM, SETTINGS)
        _, events = evaluate(state, (person(unseen=12.5),), NOW, SWIM, SETTINGS)

        assert len(events) == 1

    def test_state_forgets_tracks_that_are_gone(self) -> None:
        state, _ = evaluate(RulesState(), (person(track_id=3, unseen=12.0),), NOW, SWIM, SETTINGS)
        state, _ = evaluate(state, (), NOW, SWIM, SETTINGS)

        assert state.alerted_track_ids == frozenset()


class TestOperatingMode:
    def test_armed_mode_also_raises_submersion(self) -> None:
        _, events = evaluate(
            RulesState(), (person(unseen=12.0),), NOW, OperatingMode.ARMED, SETTINGS
        )
        assert len(events) == 1
        assert events[0].mode is OperatingMode.ARMED

    def test_maintenance_mode_is_paused(self) -> None:
        _, events = evaluate(
            RulesState(), (person(unseen=30.0),), NOW, OperatingMode.MAINTENANCE, SETTINGS
        )
        assert events == ()

    def test_leaving_maintenance_with_someone_still_under_raises(self) -> None:
        """Pausing must not swallow an incident that is still in progress."""
        under = (person(unseen=30.0),)
        state, _ = evaluate(RulesState(), under, NOW, OperatingMode.MAINTENANCE, SETTINGS)
        _, events = evaluate(state, under, NOW, SWIM, SETTINGS)

        assert len(events) == 1


class TestImmutability:
    def test_evaluate_returns_a_new_state(self) -> None:
        before = RulesState()
        after, _ = evaluate(before, (person(unseen=12.0),), NOW, SWIM, SETTINGS)

        assert before.alerted_track_ids == frozenset()
        assert after.alerted_track_ids == frozenset({1})

    def test_state_is_frozen(self) -> None:
        with pytest.raises(ValidationError):
            RulesState().alerted_track_ids = frozenset({1})  # type: ignore[misc]
