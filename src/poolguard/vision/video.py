"""Video-file FrameSource for replay and offline testing.

Requires the `vision` extra (opencv). Timestamps are synthetic: a fixed
replay epoch plus frame_index / fps, so replays of the same file always
produce identical event timestamps.
"""

import math
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import cv2
from pydantic import AwareDatetime

from poolguard.vision.frames import REPLAY_EPOCH, Frame

FALLBACK_FPS = 30.0
"""Used when the container reports no frame rate (0/NaN), as some encoders do."""


class VideoFileFrameSource:
    def __init__(self, path: str | Path, start_time: AwareDatetime | None = None) -> None:
        self._path = Path(path)
        if start_time is not None and start_time.tzinfo is None:
            raise ValueError("start_time must be timezone-aware")
        self._start_time = start_time if start_time is not None else REPLAY_EPOCH
        if not self._path.is_file():
            raise FileNotFoundError(f"video file not found: {self._path}")

    def frames(self) -> Iterator[Frame]:
        capture = cv2.VideoCapture(str(self._path))
        if not capture.isOpened():
            raise ValueError(f"could not open video file: {self._path}")

        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps is None or math.isnan(fps) or fps <= 0:
            fps = FALLBACK_FPS

        index = 0
        try:
            while True:
                ok, image = capture.read()
                if not ok:
                    return
                ts = self._start_time + timedelta(seconds=index / fps)
                yield Frame(image=image, ts=ts, index=index)
                index += 1
        finally:
            capture.release()
