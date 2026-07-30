"""Tests for the annotated-replay renderer (vision extra)."""

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from poolguard.config import TrackingSettings  # noqa: E402
from poolguard.events import BoundingBox, Detection, TrackedPerson  # noqa: E402
from poolguard.vision.frames import Frame  # noqa: E402
from poolguard.vision.render import (  # noqa: E402
    annotate_frame,
    color_for_track,
    render_tracked_replay,
)

START = datetime(2026, 1, 1, tzinfo=UTC)
WIDTH, HEIGHT = 64, 48


def blank_image() -> np.ndarray:
    return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)


def person(track_id: int, unseen: float = 0.0) -> TrackedPerson:
    return TrackedPerson(
        track_id=track_id,
        detection=Detection(
            frame_ts=START,
            box=BoundingBox(x=0.25, y=0.25, width=0.5, height=0.5),
            confidence=0.9,
        ),
        in_water=True,
        seconds_since_last_seen=unseen,
    )


class TestAnnotateFrame:
    def test_returns_new_array_without_touching_input(self) -> None:
        image = blank_image()
        image.setflags(write=False)  # Frame images are read-only in the pipeline

        canvas = annotate_frame(image, (person(1),), pool_zone=(0, 0, 1, 1))

        assert canvas is not image
        assert not image.any()  # input still all-black

    def test_live_person_is_drawn(self) -> None:
        canvas = annotate_frame(blank_image(), (person(1),), pool_zone=(0, 0, 1, 1))

        assert canvas.any()  # something was drawn

    def test_coasting_person_is_drawn(self) -> None:
        canvas = annotate_frame(blank_image(), (person(1, unseen=4.2),), pool_zone=(0, 0, 1, 1))

        assert canvas.any()

    def test_coasting_looks_different_from_live(self) -> None:
        live = annotate_frame(blank_image(), (person(1),), pool_zone=(0, 0, 1, 1))
        ghost = annotate_frame(blank_image(), (person(1, unseen=4.2),), pool_zone=(0, 0, 1, 1))

        assert (live != ghost).any()


class TestTrackColors:
    def test_deterministic(self) -> None:
        assert color_for_track(7) == color_for_track(7)

    def test_adjacent_ids_differ(self) -> None:
        assert color_for_track(1) != color_for_track(2)


class ListFrameSource:
    def __init__(self, frames: list[Frame]) -> None:
        self._frames = frames

    def frames(self) -> Iterator[Frame]:
        return iter(self._frames)


class OnePersonEstimator:
    def estimate(self, frame: Frame) -> tuple[Detection, ...]:
        return (
            Detection(
                frame_ts=frame.ts,
                box=BoundingBox(x=0.25, y=0.25, width=0.5, height=0.5),
                confidence=0.9,
            ),
        )


def test_render_tracked_replay_writes_video(tmp_path: Path) -> None:
    image = blank_image()
    frames = [Frame(image=image, ts=START + timedelta(seconds=i / 4), index=i) for i in range(6)]
    out = tmp_path / "annotated.mp4"

    results = list(
        render_tracked_replay(
            ListFrameSource(frames), OnePersonEstimator(), TrackingSettings(), out
        )
    )

    assert len(results) == 6
    assert all(len(r.people) == 1 for r in results)

    capture = cv2.VideoCapture(str(out))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 6
    capture.release()


def test_render_tracked_replay_raises_if_writer_cannot_open(tmp_path: Path) -> None:
    # A nonexistent parent directory makes cv2.VideoWriter construct but
    # never open — the exact silent-failure mode open_writer must catch.
    image = blank_image()
    frames = [Frame(image=image, ts=START, index=0)]
    out = tmp_path / "missing_dir" / "annotated.mp4"

    with pytest.raises(RuntimeError, match="could not open video writer"):
        list(
            render_tracked_replay(
                ListFrameSource(frames), OnePersonEstimator(), TrackingSettings(), out
            )
        )
