"""Annotated replay rendering: draw the tracker's view onto video frames.

Requires the `vision` extra (opencv). This is the debugging microscope for
everything downstream of detection — track identity (stable colors), the
pool zone, and coasting tracks with their unseen timers, which is the
submersion signal made visible.
"""

from collections import deque
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from poolguard.config import RulesSettings, TrackingSettings
from poolguard.events import TrackedPerson
from poolguard.geometry import Polygon
from poolguard.replay import TrackedFrameResult, run_tracked_replay
from poolguard.vision.frames import Frame, FrameSource
from poolguard.vision.hud import HudState, draw_hud
from poolguard.vision.pose import PoseEstimator

if TYPE_CHECKING:
    # Type-only: labels.py needs PyYAML, which the vision extra does not install.
    from poolguard.labels import ShotLabel

FALLBACK_FPS = 15.0

_PALETTE: tuple[tuple[int, int, int], ...] = (
    (80, 220, 60),  # green
    (60, 140, 255),  # orange
    (255, 120, 60),  # blue
    (60, 60, 230),  # red
    (230, 60, 200),  # magenta
    (40, 220, 220),  # yellow
    (200, 200, 80),  # cyan
    (160, 90, 240),  # pink
)

_ZONE_COLOR = (200, 160, 40)  # teal-ish, BGR


def color_for_track(track_id: int) -> tuple[int, int, int]:
    """Stable BGR color per track ID; a color change on screen = an ID break."""
    return _PALETTE[track_id % len(_PALETTE)]


def annotate_frame(
    image: np.ndarray,
    people: tuple[TrackedPerson, ...],
    pool_zone: Polygon,
) -> np.ndarray:
    """Return a copy of the frame with the tracker's view drawn on it."""
    canvas = image.copy()
    height, width = canvas.shape[:2]

    zone_pixels = np.array(
        [(int(x * width), int(y * height)) for x, y in pool_zone], dtype=np.int32
    )
    cv2.polylines(canvas, [zone_pixels], isClosed=True, color=_ZONE_COLOR, thickness=1)

    for person in people:
        box = person.detection.box
        top_left = (int(box.x * width), int(box.y * height))
        bottom_right = (
            int((box.x + box.width) * width),
            int((box.y + box.height) * height),
        )
        color = color_for_track(person.track_id)
        if person.seconds_since_last_seen == 0.0:
            _draw_live(canvas, person, top_left, bottom_right, color)
        else:
            _draw_coasting(canvas, person, top_left, bottom_right, color)
    return canvas


def _draw_live(
    canvas: np.ndarray,
    person: TrackedPerson,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    cv2.rectangle(canvas, top_left, bottom_right, color, 2)
    cv2.putText(
        canvas,
        f"#{person.track_id}",
        (top_left[0], max(top_left[1] - 6, 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def _draw_coasting(
    canvas: np.ndarray,
    person: TrackedPerson,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    """Draw a track the detector has lost: the ghost box at its last-seen spot.

    This is the frame's most important annotation — it must read as "the
    system is watching this spot" and show how long the person has been
    unseen (person.seconds_since_last_seen).
    """
    cv2.rectangle(canvas, top_left, bottom_right, color, 1)
    cv2.putText(
        canvas,
        f"#{person.track_id} (unseen/lost {person.seconds_since_last_seen:.1f}s)",
        (top_left[0], max(top_left[1] - 6, 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def _hud_state(
    result: TrackedFrameResult,
    start_ts: datetime,
    frames_seen: int,
    frames_with_detections: int,
    label: "ShotLabel | None",
    rules: RulesSettings,
) -> HudState:
    live = tuple(p for p in result.people if p.seconds_since_last_seen == 0.0)
    coasting = tuple(p for p in result.people if p.seconds_since_last_seen > 0.0)
    return HudState(
        frame_index=result.frame_index,
        elapsed_seconds=(result.ts - start_ts).total_seconds(),
        live_tracks=len(live),
        coasting_tracks=len(coasting),
        longest_unseen_seconds=max(
            (p.seconds_since_last_seen for p in coasting if p.in_water), default=0.0
        ),
        detection_rate=frames_with_detections / max(frames_seen, 1),
        expected_event=label.expected_event.value if label and label.expected_event else None,
        onset_window=label.onset_window if label else None,
        submersion_timeout_seconds=rules.submersion_timeout_seconds,
    )


def render_tracked_replay(
    source: FrameSource,
    estimator: PoseEstimator,
    settings: TrackingSettings,
    out_path: str | Path,
    label: "ShotLabel | None" = None,
    rules: RulesSettings | None = None,
) -> Iterator[TrackedFrameResult]:
    """run_tracked_replay that also writes an annotated video as it goes.

    Yields the same TrackedFrameResults, so callers fold summaries exactly
    as they would without rendering. The writer's fps comes from the gap
    between the first two frame timestamps (replay timestamps are synthetic
    and evenly spaced); a single-frame source falls back to FALLBACK_FPS.
    """
    writer: cv2.VideoWriter | None = None
    pending: tuple[np.ndarray, TrackedFrameResult] | None = None
    previous_ts = None
    rules = rules or RulesSettings()
    start_ts = None
    frames_seen = 0
    frames_with_detections = 0

    def open_writer(canvas: np.ndarray, fps: float) -> cv2.VideoWriter:
        height, width = canvas.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"could not open video writer for {out_path}")
        return writer

    try:
        for frame, result in _tracked_frames(source, estimator, settings):
            if start_ts is None:
                start_ts = result.ts
            frames_seen += 1
            if result.detections:
                frames_with_detections += 1
            canvas = annotate_frame(frame.image, result.people, settings.pool_zone)
            canvas = draw_hud(
                canvas,
                _hud_state(result, start_ts, frames_seen, frames_with_detections, label, rules),
            )
            if writer is None:
                if pending is None:
                    pending = (canvas, result)
                    previous_ts = result.ts
                    continue
                gap = (result.ts - previous_ts).total_seconds()
                fps = 1.0 / gap if gap > 0 else FALLBACK_FPS
                writer = open_writer(pending[0], fps)
                writer.write(pending[0])
                yield pending[1]
                pending = None
            writer.write(canvas)
            yield result

        if pending is not None:  # single-frame source
            writer = open_writer(pending[0], FALLBACK_FPS)
            writer.write(pending[0])
            yield pending[1]
    finally:
        if writer is not None:
            writer.release()


def _tracked_frames(
    source: FrameSource, estimator: PoseEstimator, settings: TrackingSettings
) -> Iterator[tuple[Frame, TrackedFrameResult]]:
    """Pair each source frame with its tracked result.

    run_tracked_replay consumes the source internally, so tee the frames
    here: wrap the source to remember frames in flight, oldest first, so
    pairing stays correct even if a future run_tracked_replay buffers or
    looks ahead instead of consuming in strict lockstep.
    """
    current: deque[Frame] = deque()

    class _Tap:
        def frames(self) -> Iterator[Frame]:
            for frame in source.frames():
                current.append(frame)
                yield frame

    for result in run_tracked_replay(_Tap(), estimator, settings):
        yield current.popleft(), result
