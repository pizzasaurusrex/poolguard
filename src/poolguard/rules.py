"""Rules engine: turn tracked people into safety events (ADR-004).

Each rule is a plain, readable condition over the tracker's output, with its
thresholds in RulesSettings. v0 has three rules:

  submersion: a person last seen past edge_margin inside the outline has gone unseen for
              submersion_timeout_seconds or longer. Runs in every mode
              except maintenance.
  entry:      while armed, a person's feet are more than edge_margin inside
              the pool outline. Raised on the first such frame, with no
              confirmation delay: the PRD makes any armed entry an immediate
              emergency.
  edge:       while armed, a person's feet are within edge_margin of the
              outline, on either side. A warning, not an emergency: toes on
              the coping are within the error of the outline and the box, so
              they cannot be told apart from a step into the water.

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

    edge_warned_track_ids: frozenset[int] = frozenset()
    """Tracks that have raised a person-at-edge warning since the system was
    last armed. Kept while the track exists, like entered_track_ids, so pacing
    along the coping raises one warning, not one per step."""


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
    entered, entries = _apply_entry(state.entered_track_ids, people, mode, settings)
    warned, warnings = _apply_edge(state.edge_warned_track_ids, entered, people, mode, settings)
    new_state = RulesState(
        alerted_track_ids=alerted, entered_track_ids=entered, edge_warned_track_ids=warned
    )
    return new_state, entries + submersions + warnings


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
    entered: frozenset[int],
    people: tuple[TrackedPerson, ...],
    mode: OperatingMode,
    settings: RulesSettings,
) -> tuple[frozenset[int], tuple[SafetyEvent, ...]]:
    if mode is not OperatingMode.ARMED:
        return frozenset(), ()
    in_water = tuple(p for p in people if p.water_depth > settings.edge_margin)
    events = tuple(_entry_event(p, mode) for p in in_water if p.track_id not in entered)
    still_entered = frozenset(p.track_id for p in people if p in in_water or p.track_id in entered)
    return still_entered, events


def _apply_edge(
    warned: frozenset[int],
    entered: frozenset[int],
    people: tuple[TrackedPerson, ...],
    mode: OperatingMode,
    settings: RulesSettings,
) -> tuple[frozenset[int], tuple[SafetyEvent, ...]]:
    """Warn once per track at the edge; a track that has entered already raised more."""
    if mode is not OperatingMode.ARMED:
        return frozenset(), ()
    at_edge = tuple(p for p in people if abs(p.water_depth) <= settings.edge_margin)
    events = tuple(
        _edge_event(p, mode)
        for p in at_edge
        if p.track_id not in warned and p.track_id not in entered
    )
    still_warned = frozenset(p.track_id for p in people if p in at_edge or p.track_id in warned)
    return still_warned, events


def _is_submerged(person: TrackedPerson, settings: RulesSettings) -> bool:
    """Lost while past the edge band. Someone lost with their toes on the coping
    walked out of view; a swimmer's short box puts them far past the band."""
    return (
        person.water_depth > settings.edge_margin
        and person.seconds_since_last_seen >= settings.submersion_timeout_seconds
    )


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


def _edge_event(person: TrackedPerson, mode: OperatingMode) -> SafetyEvent:
    return SafetyEvent(
        event_type=EventType.PERSON_AT_EDGE,
        tier=AlertTier.WARN,
        started_at=person.detection.frame_ts,
        track_id=person.track_id,
        mode=mode,
        confidence=person.detection.confidence,
        detail="Person at the pool edge while the system is armed",
    )
