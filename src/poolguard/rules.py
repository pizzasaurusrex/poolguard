"""Rules engine: turn tracked people into safety events (ADR-004).

Each rule is a plain, readable condition over the tracker's output, with its
thresholds in RulesSettings. v0 has one rule:

  submersion: a person last seen in the water has gone unseen for
              submersion_timeout_seconds or longer.

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

    submerged = tuple(person for person in people if _is_submerged(person, settings))
    events = tuple(
        _submersion_event(person, now, mode)
        for person in submerged
        if person.track_id not in state.alerted_track_ids
    )
    new_state = RulesState(alerted_track_ids=frozenset(person.track_id for person in submerged))
    return new_state, events


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
