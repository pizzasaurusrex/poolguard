"""Greedy box-overlap tracker (ByteTrack-style association, no motion model).

Turns per-frame Detections into persistent TrackedPerson streams. The design
choice that matters for safety: a track that stops matching is *not* deleted —
it coasts, and its seconds_since_last_seen keeps growing. A coasting track
inside the pool zone is the raw signal for the submersion rule.

Association is four greedy passes:
  pass 1: high-confidence detections vs all live tracks, by IoU
  pass 2: high-confidence detections still unmatched vs tracks seen within
          the last containment_window_seconds, by containment — the model's
          box for one swimmer flips between head-only and whole-body, which
          IoU scores as a different person
  pass 3: low-confidence detections vs tracks still unmatched, by IoU (the
          "rescue" pass — a weak detection where a track already was is
          almost certainly still that person)
  pass 4: high-confidence detections still unmatched vs tracks lost within
          the last handoff_window_seconds, by distance in box lengths (the
          "hand-off" pass — a swimmer who jumps in vanishes for a couple of
          seconds and surfaces nearby with a box of a different shape)
Only unmatched high-confidence detections spawn new tracks; low-confidence
ones never do, so splash noise cannot create phantom people.

The tracker itself is a pure function: advance(state, ...) returns a new
state, so replay and tests can fold it like ReplaySummary.
"""

import math
from collections.abc import Callable
from typing import NamedTuple

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from poolguard.config import TrackingSettings
from poolguard.events import BoundingBox, Detection, TrackedPerson
from poolguard.geometry import Polygon, point_in_polygon


class Track(BaseModel):
    """Internal per-person state carried between frames."""

    model_config = ConfigDict(frozen=True)

    track_id: int
    last_detection: Detection
    last_seen_ts: AwareDatetime


class TrackerState(BaseModel):
    """All tracker memory; advance() folds this immutably per frame."""

    model_config = ConfigDict(frozen=True)

    next_track_id: int = Field(default=1, ge=1)
    tracks: tuple[Track, ...] = ()


def _intersection_area(a: BoundingBox, b: BoundingBox) -> float:
    left = max(a.x, b.x)
    top = max(a.y, b.y)
    right = min(a.x + a.width, b.x + b.width)
    bottom = min(a.y + a.height, b.y + b.height)
    if right <= left or bottom <= top:
        return 0.0
    return (right - left) * (bottom - top)


def iou(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection-over-union of two normalized boxes."""
    intersection = _intersection_area(a, b)
    union = a.width * a.height + b.width * b.height - intersection
    return intersection / union


def containment(a: BoundingBox, b: BoundingBox) -> float:
    """Share of the smaller box that lies inside the larger one.

    1.0 when one box sits wholly within the other, whatever their sizes. IoU
    scores that case low when the sizes differ, which is exactly what happens
    when the model boxes a swimmer's head in one frame and whole body the next.
    """
    smaller_area = min(a.width * a.height, b.width * b.height)
    return _intersection_area(a, b) / smaller_area


def handoff_distance(lost: BoundingBox, candidate: BoundingBox) -> float:
    """Centre-to-centre distance in lengths of the lost box's longer side.

    Measured in the lost person's own size so the same limit means about the
    same real distance near the camera and across the pool.
    """
    dx = (candidate.x + candidate.width / 2) - (lost.x + lost.width / 2)
    dy = (candidate.y + candidate.height / 2) - (lost.y + lost.height / 2)
    return math.hypot(dx, dy) / max(lost.width, lost.height)


def _nearness(lost: BoundingBox, candidate: BoundingBox) -> float:
    # Negated so match_greedy, which keeps the highest scores, keeps the nearest.
    return -handoff_distance(lost, candidate)


BoxScore = Callable[[BoundingBox, BoundingBox], float]


def match_greedy(
    tracks: tuple[Track, ...],
    detections: tuple[Detection, ...],
    minimum_score: float,
    score_boxes: BoxScore = iou,
) -> tuple[tuple[int, int], ...]:
    """Greedily pair tracks with detections, strongest score first.

    Returns (track_index, detection_index) pairs. Each track and each
    detection may appear in at most one pair, and a pair is only valid if
    score_boxes(track.last_detection.box, detection.box) >= minimum_score.
    """

    candidates = []
    for track_index, track in enumerate(tracks):
        for detection_index, detection in enumerate(detections):
            score = score_boxes(track.last_detection.box, detection.box)
            if score >= minimum_score:
                candidates.append((score, track_index, detection_index))

    matched = []

    for _score, track_index, detection_index in sorted(candidates, reverse=True):
        if any(track_index == t for t, _ in matched):
            continue
        if any(detection_index == d for _, d in matched):
            continue
        matched.append((track_index, detection_index))

    return tuple(matched)


class _Association(NamedTuple):
    """Outcome of one matching pass."""

    matched: tuple[tuple[Track, Detection], ...]
    unmatched_tracks: tuple[Track, ...]
    unmatched_detections: tuple[Detection, ...]


def _associate(
    tracks: tuple[Track, ...],
    detections: tuple[Detection, ...],
    minimum_score: float,
    score_boxes: BoxScore = iou,
) -> _Association:
    pairs = match_greedy(tracks, detections, minimum_score, score_boxes)
    matched_track_indices = {track_index for track_index, _ in pairs}
    matched_detection_indices = {detection_index for _, detection_index in pairs}
    return _Association(
        matched=tuple((tracks[t], detections[d]) for t, d in pairs),
        unmatched_tracks=tuple(
            track for i, track in enumerate(tracks) if i not in matched_track_indices
        ),
        unmatched_detections=tuple(
            detection
            for i, detection in enumerate(detections)
            if i not in matched_detection_indices
        ),
    )


def advance(
    state: TrackerState,
    detections: tuple[Detection, ...],
    now: AwareDatetime,
    settings: TrackingSettings,
) -> tuple[TrackerState, tuple[TrackedPerson, ...]]:
    """Fold one frame of detections into the tracker.

    Returns the new state and the TrackedPerson view of every live track,
    including coasting ones (matched nothing this frame but not yet expired).
    """
    high = tuple(d for d in detections if d.confidence >= settings.high_confidence)
    low = tuple(
        d for d in detections if settings.low_confidence <= d.confidence < settings.high_confidence
    )

    by_overlap = _associate(state.tracks, high, settings.iou_min)

    recently_seen = tuple(
        track
        for track in by_overlap.unmatched_tracks
        if _seconds_unseen(track, now) <= settings.containment_window_seconds
    )
    by_containment = _associate(
        recently_seen, by_overlap.unmatched_detections, settings.containment_min, containment
    )
    reshaped_ids = {track.track_id for track, _ in by_containment.matched}
    still_unmatched = tuple(
        track for track in by_overlap.unmatched_tracks if track.track_id not in reshaped_ids
    )

    rescue = _associate(still_unmatched, low, settings.iou_min)
    handoff = _hand_off(rescue.unmatched_tracks, by_containment.unmatched_detections, now, settings)

    updated = tuple(
        _touch(track, detection, now)
        for track, detection in by_overlap.matched
        + by_containment.matched
        + rescue.matched
        + handoff.matched
    )
    coasting = tuple(
        track
        for track in handoff.unmatched_tracks
        if _seconds_unseen(track, now) <= settings.max_coast_seconds
    )
    spawned = tuple(
        Track(track_id=state.next_track_id + offset, last_detection=detection, last_seen_ts=now)
        for offset, detection in enumerate(handoff.unmatched_detections)
    )

    tracks = updated + coasting + spawned
    new_state = TrackerState(next_track_id=state.next_track_id + len(spawned), tracks=tracks)
    people = tuple(_to_person(track, now, settings) for track in tracks)
    return new_state, people


def _hand_off(
    unmatched_tracks: tuple[Track, ...],
    newcomers: tuple[Detection, ...],
    now: AwareDatetime,
    settings: TrackingSettings,
) -> _Association:
    """Give a recently lost track to the nearest would-be newcomer.

    Only high-confidence detections that matched nothing else arrive here, so
    a visible person keeps their own track and splash noise cannot reset a
    lost person's clock. The returned unmatched tracks include the ones too
    old to be eligible, so the caller sees every track that is still lost.
    """
    eligible = tuple(
        track
        for track in unmatched_tracks
        if _seconds_unseen(track, now) <= settings.handoff_window_seconds
    )
    result = _associate(eligible, newcomers, -settings.handoff_max_distance, _nearness)
    taken_ids = {track.track_id for track, _ in result.matched}
    return _Association(
        matched=result.matched,
        unmatched_tracks=tuple(
            track for track in unmatched_tracks if track.track_id not in taken_ids
        ),
        unmatched_detections=result.unmatched_detections,
    )


def _touch(track: Track, detection: Detection, now: AwareDatetime) -> Track:
    return Track(track_id=track.track_id, last_detection=detection, last_seen_ts=now)


def _to_person(track: Track, now: AwareDatetime, settings: TrackingSettings) -> TrackedPerson:
    return TrackedPerson(
        track_id=track.track_id,
        detection=track.last_detection,
        in_water=_in_zone(track.last_detection.box, settings.pool_zone),
        seconds_since_last_seen=_seconds_unseen(track, now),
    )


def _seconds_unseen(track: Track, now: AwareDatetime) -> float:
    """Time since the track last matched a detection, never negative.

    A live clock can step backwards (an NTP correction), which would make the
    raw difference negative and fail TrackedPerson validation, stopping the
    tracker. Reading that as "just seen" keeps detection running.
    """
    return max(0.0, (now - track.last_seen_ts).total_seconds())


def _in_zone(box: BoundingBox, zone: Polygon) -> bool:
    center_x = box.x + box.width / 2
    center_y = box.y + box.height / 2
    return point_in_polygon(center_x, center_y, zone)
