"""Two-pass IoU tracker (ByteTrack-style association, no motion model).

Turns per-frame Detections into persistent TrackedPerson streams. The design
choice that matters for safety: a track that stops matching is *not* deleted —
it coasts, and its seconds_since_last_seen keeps growing. A coasting track
inside the pool zone is the raw signal for the submersion rule.

Association is two passes of greedy IoU matching:
  pass 1: high-confidence detections vs all live tracks
  pass 2: low-confidence detections vs tracks still unmatched (the "rescue"
          pass — a weak detection where a track already was is almost
          certainly still that person)
Only unmatched high-confidence detections spawn new tracks; low-confidence
ones never do, so splash noise cannot create phantom people.

The tracker itself is a pure function: advance(state, ...) returns a new
state, so replay and tests can fold it like ReplaySummary.
"""

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from poolguard.config import TrackingSettings
from poolguard.events import BoundingBox, Detection, TrackedPerson


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


def iou(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection-over-union of two normalized boxes."""
    left = max(a.x, b.x)
    top = max(a.y, b.y)
    right = min(a.x + a.width, b.x + b.width)
    bottom = min(a.y + a.height, b.y + b.height)
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    union = a.width * a.height + b.width * b.height - intersection
    return intersection / union


def match_greedy(
    tracks: tuple[Track, ...],
    detections: tuple[Detection, ...],
    iou_min: float,
) -> tuple[tuple[int, int], ...]:
    """Greedily pair tracks with detections by IoU.

    Returns (track_index, detection_index) pairs. Each track and each
    detection may appear in at most one pair, and a pair is only valid if
    iou(track.last_detection.box, detection.box) >= iou_min.
    """

    candidates = []
    for track_index, track in enumerate(tracks):
        for detection_index, detection in enumerate(detections):
            score = iou(track.last_detection.box, detection.box)
            if score >= iou_min:
                candidates.append((score, track_index, detection_index))

    matched = []

    for _score, track_index, detection_index in sorted(candidates, reverse=True):
        if any(track_index == t for t, _ in matched):
            continue
        if any(detection_index == d for _, d in matched):
            continue
        matched.append((track_index, detection_index))

    return tuple(matched)


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

    matched_pairs = match_greedy(state.tracks, high, settings.iou_min)
    matched_track_indices = {t for t, _ in matched_pairs}
    matched_high_indices = {d for _, d in matched_pairs}

    remaining = tuple(
        (i, track) for i, track in enumerate(state.tracks) if i not in matched_track_indices
    )
    rescue_pairs = match_greedy(tuple(track for _, track in remaining), low, settings.iou_min)

    updated: list[Track] = []
    for track_index, detection_index in matched_pairs:
        updated.append(_touch(state.tracks[track_index], high[detection_index], now))
    rescued_track_indices = set()
    for local_index, detection_index in rescue_pairs:
        original_index, track = remaining[local_index]
        rescued_track_indices.add(original_index)
        updated.append(_touch(track, low[detection_index], now))

    coasting = tuple(
        track
        for i, track in remaining
        if i not in rescued_track_indices
        and (now - track.last_seen_ts).total_seconds() <= settings.max_coast_seconds
    )

    next_id = state.next_track_id
    spawned: list[Track] = []
    for i, detection in enumerate(high):
        if i in matched_high_indices:
            continue
        spawned.append(Track(track_id=next_id, last_detection=detection, last_seen_ts=now))
        next_id += 1

    tracks = tuple(updated) + coasting + tuple(spawned)
    new_state = TrackerState(next_track_id=next_id, tracks=tracks)
    people = tuple(_to_person(track, now, settings) for track in tracks)
    return new_state, people


def _touch(track: Track, detection: Detection, now: AwareDatetime) -> Track:
    return Track(track_id=track.track_id, last_detection=detection, last_seen_ts=now)


def _to_person(track: Track, now: AwareDatetime, settings: TrackingSettings) -> TrackedPerson:
    return TrackedPerson(
        track_id=track.track_id,
        detection=track.last_detection,
        in_water=_in_zone(track.last_detection.box, settings.pool_zone),
        seconds_since_last_seen=(now - track.last_seen_ts).total_seconds(),
    )


def _in_zone(box: BoundingBox, zone: tuple[float, float, float, float]) -> bool:
    center_x = box.x + box.width / 2
    center_y = box.y + box.height / 2
    zone_x, zone_y, zone_width, zone_height = zone
    return zone_x <= center_x <= zone_x + zone_width and zone_y <= center_y <= zone_y + zone_height
