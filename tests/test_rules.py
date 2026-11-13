"""Tests for the rules engine: the submersion and entry rules.

The rules' input is the tracker's output. These tests build TrackedPerson values directly,
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
    SafetyEvent,
    TrackedPerson,
)
from poolguard.rules import RulesState, evaluate

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
SETTINGS = RulesSettings(_env_file=None, submersion_timeout_seconds=12.0)
SWIM = OperatingMode.SWIM
ARMED = OperatingMode.ARMED


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
        _, events = evaluate(RulesState(), (person(unseen=12.0),), NOW, ARMED, SETTINGS)
        submersions = [e for e in events if e.event_type is EventType.SUBMERSION]
        assert len(submersions) == 1
        assert submersions[0].mode is ARMED

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


def entries(events: tuple[SafetyEvent, ...]) -> list[SafetyEvent]:
    return [event for event in events if event.event_type is EventType.UNSUPERVISED_ENTRY]


class TestEntry:
    def test_person_in_the_water_while_armed_raises_an_emergency(self) -> None:
        _, events = evaluate(RulesState(), (person(track_id=4),), NOW, ARMED, SETTINGS)

        assert len(events) == 1
        event = events[0]
        assert event.event_type is EventType.UNSUPERVISED_ENTRY
        assert event.tier is AlertTier.EMERGENCY
        assert event.track_id == 4
        assert event.mode is ARMED
        assert event.started_at == NOW

    def test_person_on_the_deck_while_armed_raises_nothing(self) -> None:
        _, events = evaluate(RulesState(), (person(in_water=False),), NOW, ARMED, SETTINGS)
        assert events == ()

    def test_swim_mode_allows_people_in_the_water(self) -> None:
        _, events = evaluate(RulesState(), (person(),), NOW, SWIM, SETTINGS)
        assert events == ()

    def test_maintenance_mode_raises_no_entry(self) -> None:
        _, events = evaluate(RulesState(), (person(),), NOW, OperatingMode.MAINTENANCE, SETTINGS)
        assert events == ()

    def test_a_person_who_enters_and_goes_straight_under_still_raises(self) -> None:
        """First seen in the water and already lost: entry must not wait for a second sighting."""
        _, events = evaluate(RulesState(), (person(unseen=0.5),), NOW, ARMED, SETTINGS)
        assert len(entries(events)) == 1

    def test_each_person_in_the_water_gets_their_own_event(self) -> None:
        people = (person(track_id=1), person(track_id=2), person(track_id=3, in_water=False))
        _, events = evaluate(RulesState(), people, NOW, ARMED, SETTINGS)
        assert sorted(event.track_id for event in events) == [1, 2]


class TestEntryFiresOncePerTrack:
    def test_event_is_not_repeated_while_the_person_stays_in(self) -> None:
        state, _ = evaluate(RulesState(), (person(),), NOW, ARMED, SETTINGS)
        _, events = evaluate(state, (person(),), NOW + timedelta(seconds=1), ARMED, SETTINGS)
        assert events == ()

    def test_stepping_out_and_back_in_does_not_repeat_it(self) -> None:
        """A box centre jittering across the coping must not re-alert every few frames."""
        state, _ = evaluate(RulesState(), (person(),), NOW, ARMED, SETTINGS)
        state, _ = evaluate(state, (person(in_water=False),), NOW, ARMED, SETTINGS)
        _, events = evaluate(state, (person(),), NOW, ARMED, SETTINGS)
        assert events == ()

    def test_a_second_person_still_raises(self) -> None:
        state, _ = evaluate(RulesState(), (person(track_id=1),), NOW, ARMED, SETTINGS)
        _, events = evaluate(state, (person(track_id=1), person(track_id=2)), NOW, ARMED, SETTINGS)
        assert [event.track_id for event in events] == [2]

    def test_arming_with_someone_still_in_the_water_raises(self) -> None:
        """Switching swim mode off while a child is still in the pool is exactly the case."""
        state, _ = evaluate(RulesState(), (person(),), NOW, ARMED, SETTINGS)
        state, _ = evaluate(state, (person(),), NOW, SWIM, SETTINGS)
        _, events = evaluate(state, (person(),), NOW, ARMED, SETTINGS)
        assert len(entries(events)) == 1

    def test_state_forgets_tracks_that_are_gone(self) -> None:
        state, _ = evaluate(RulesState(), (person(track_id=5),), NOW, ARMED, SETTINGS)
        state, _ = evaluate(state, (), NOW, ARMED, SETTINGS)
        assert state.entered_track_ids == frozenset()


class TestImmutability:
    def test_evaluate_returns_a_new_state(self) -> None:
        before = RulesState()
        after, _ = evaluate(before, (person(unseen=12.0),), NOW, SWIM, SETTINGS)

        assert before.alerted_track_ids == frozenset()
        assert after.alerted_track_ids == frozenset({1})

    def test_state_is_frozen(self) -> None:
        with pytest.raises(ValidationError):
            RulesState().alerted_track_ids = frozenset({1})  # type: ignore[misc]
