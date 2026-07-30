"""Annotated replay rendering: draw the tracker's view onto video frames.

Requires the `vision` extra (opencv). This is the debugging microscope for
everything downstream of detection — track identity (stable colors), the
pool zone, and coasting tracks with their unseen timers, which is the
submersion signal made visible.
"""

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from poolguard.config import TrackingSettings
from poolguard.events import TrackedPerson
from poolguard.replay import TrackedFrameResult, run_tracked_replay
from poolguard.vision.frames import FrameSource
from poolguard.vision.pose import PoseEstimator

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
    pool_zone: tuple[float, float, float, float],
) -> np.ndarray:
    """Return a copy of the frame with the tracker's view drawn on it."""
    canvas = image.copy()
    height, width = canvas.shape[:2]

    zone_x, zone_y, zone_w, zone_h = pool_zone
    cv2.rectangle(
        canvas,
        (int(zone_x * width), int(zone_y * height)),
        (int((zone_x + zone_w) * width), int((zone_y + zone_h) * height)),
        _ZONE_COLOR,
        1,
    )

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


def render_tracked_replay(
    source: FrameSource,
    estimator: PoseEstimator,
    settings: TrackingSettings,
    out_path: str | Path,
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

    def open_writer(canvas: np.ndarray, fps: float) -> cv2.VideoWriter:
        height, width = canvas.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    frames_and_results = (
        (frame, result) for frame, result in _tracked_frames(source, estimator, settings)
    )

    try:
        for frame, result in frames_and_results:
            canvas = annotate_frame(frame.image, result.people, settings.pool_zone)
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
) -> Iterator[tuple]:
    """Pair each source frame with its tracked result.

    run_tracked_replay consumes the source internally, so tee the frames
    here: wrap the source to remember the frame currently in flight.
    """
    current: list = []

    class _Tap:
        def frames(self) -> Iterator:
            for frame in source.frames():
                current.append(frame)
                yield frame

    for result in run_tracked_replay(_Tap(), estimator, settings):
        yield current.pop(), result
