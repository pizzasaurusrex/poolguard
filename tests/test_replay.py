"""Tests for the replay loop and summary fold, using seam fakes."""

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from pydantic import ValidationError

from poolguard.events import BoundingBox, Detection
from poolguard.replay import FrameResult, ReplaySummary, run_replay
from poolguard.vision.frames import Frame

START = datetime(2026, 1, 1, tzinfo=UTC)


class ListFrameSource:
    def __init__(self, frames: list[Frame]) -> None:
        self._frames = frames

    def frames(self) -> Iterator[Frame]:
        return iter(self._frames)


class StubEstimator:
    """Returns a fixed number of detections per frame, cycling through counts."""

    def __init__(self, counts: list[int]) -> None:
        self._counts = counts

    def estimate(self, frame: Frame) -> tuple[Detection, ...]:
        count = self._counts[frame.index]
        detection = Detection(
            frame_ts=frame.ts,
            box=BoundingBox(x=0.1, y=0.1, width=0.2, height=0.3),
            confidence=0.9,
        )
        return (detection,) * count


def make_frames(count: int) -> list[Frame]:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    return [
        Frame(image=image, ts=START + timedelta(seconds=i), index=i) for i in range(count)
    ]


def test_run_replay_pairs_frames_with_detections() -> None:
    source = ListFrameSource(make_frames(3))
    estimator = StubEstimator(counts=[2, 0, 1])

    results = list(run_replay(source, estimator))

    assert [r.frame_index for r in results] == [0, 1, 2]
    assert [len(r.detections) for r in results] == [2, 0, 1]
    assert results[0].ts == START


def test_summary_fold_accumulates_immutably() -> None:
    source = ListFrameSource(make_frames(3))
    estimator = StubEstimator(counts=[2, 0, 1])

    empty = ReplaySummary()
    summary = empty
    for result in run_replay(source, estimator):
        summary = summary.fold(result)

    assert summary == ReplaySummary(frames=3, frames_with_people=2, detections=3)
    assert empty == ReplaySummary()


def test_replay_of_empty_source_yields_nothing() -> None:
    results = list(run_replay(ListFrameSource([]), StubEstimator(counts=[])))

    assert results == []


def test_frame_result_is_frozen() -> None:
    result = FrameResult(frame_index=0, ts=START, detections=())

    with pytest.raises(ValidationError):
        result.frame_index = 1  # type: ignore[misc]
