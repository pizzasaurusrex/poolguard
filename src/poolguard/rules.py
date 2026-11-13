"""Rules engine: turn tracked people into safety events (ADR-004).

Each rule is a plain, readable condition over the tracker's output, with its
thresholds in RulesSettings. v0 has two rules:

  submersion: a person last seen in the water has gone unseen for
              submersion_timeout_seconds or longer. Runs in every mode
              except maintenance.
  entry:      while armed, a person is in the water. Raised on the first
              frame, with no confirmation delay: the PRD makes any armed
              entry an immediate emergency.

Like the tracker, the engine is a pure function: evaluate(state, ...) returns
a new state and the events raised by this frame. Events are edge-triggered,
raised once when a person crosses the threshold and not again on every later
frame. The alert manager decides what to do with an event; repeating it here
would only hand it duplicates to de-duplicate.
"""

from datetime import timedelta

from pydantic import AwareDatetime, BaseModel, ConfigDict

from poolguard.config import RulesSettings
from poolguard.events import AlertTier, EventType, OperatingMode, SafetyEvent, TrackedPerson


class RulesState(BaseModel):
    """All rules-engine memory; evaluate() folds this immutably per frame."""

    model_config = ConfigDict(frozen=True)

    alerted_track_ids: frozenset[int] = frozenset()
    """Tracks currently past the submersion threshold. A track drops out when
    its person is seen again or the track expires, which re-arms the rule."""

    entered_track_ids: frozenset[int] = frozenset()
    """Tracks that have raised an entry since the system was last armed. A
    track stays here while it exists, even if it climbs out, so a box centre
    jittering across the coping does not re-alert. Cleared whenever the mode
    is not armed, so arming with someone still in the water raises."""


def evaluate(
    state: RulesState,
    people: tuple[TrackedPerson, ...],
    now: AwareDatetime,
    mode: OperatingMode,
    settings: RulesSettings,
) -> tuple[RulesState, tuple[SafetyEvent, ...]]:
    """Apply every rule to one frame of tracked people.

    Returns the new state and the events that start on this frame.
    """
    if mode is OperatingMode.MAINTENANCE:
        # Paused. Forgetting who was already alerted means an incident still
        # in progress when the pause ends is raised, not swallowed.
        return RulesState(), ()

    alerted, submersions = _apply_submersion(state.alerted_track_ids, people, now, mode, settings)
    entered, entries = _apply_entry(state.entered_track_ids, people, mode)
    new_state = RulesState(alerted_track_ids=alerted, entered_track_ids=entered)
    return new_state, entries + submersions


def _apply_submersion(
    alerted: frozenset[int],
    people: tuple[TrackedPerson, ...],
    now: AwareDatetime,
    mode: OperatingMode,
    settings: RulesSettings,
) -> tuple[frozenset[int], tuple[SafetyEvent, ...]]:
    submerged = tuple(person for person in people if _is_submerged(person, settings))
    events = tuple(
        _submersion_event(person, now, mode)
        for person in submerged
        if person.track_id not in alerted
    )
    return frozenset(person.track_id for person in submerged), events


def _apply_entry(
    entered: frozenset[int], people: tuple[TrackedPerson, ...], mode: OperatingMode
) -> tuple[frozenset[int], tuple[SafetyEvent, ...]]:
    if mode is not OperatingMode.ARMED:
        return frozenset(), ()
    events = tuple(
        _entry_event(person, mode)
        for person in people
        if person.in_water and person.track_id not in entered
    )
    still_entered = frozenset(
        person.track_id for person in people if person.in_water or person.track_id in entered
    )
    return still_entered, events


def _is_submerged(person: TrackedPerson, settings: RulesSettings) -> bool:
    return person.in_water and person.seconds_since_last_seen >= settings.submersion_timeout_seconds


def _submersion_event(
    person: TrackedPerson, now: AwareDatetime, mode: OperatingMode
) -> SafetyEvent:
    unseen_seconds = person.seconds_since_last_seen
    return SafetyEvent(
        event_type=EventType.SUBMERSION,
        tier=AlertTier.EMERGENCY,
        started_at=now - timedelta(seconds=unseen_seconds),
        track_id=person.track_id,
        mode=mode,
        confidence=person.detection.confidence,
        detail=f"Person last seen in the water {unseen_seconds:.0f}s ago and has not resurfaced",
    )


def _entry_event(person: TrackedPerson, mode: OperatingMode) -> SafetyEvent:
    return SafetyEvent(
        event_type=EventType.UNSUPERVISED_ENTRY,
        tier=AlertTier.EMERGENCY,
        started_at=person.detection.frame_ts,
        track_id=person.track_id,
        mode=mode,
        confidence=person.detection.confidence,
        detail="Person in the water while the system is armed",
    )
