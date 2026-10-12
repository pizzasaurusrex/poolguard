"""Tests for the annotated-replay renderer (vision extra)."""

import importlib
import sys
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from poolguard.config import RulesSettings, TrackingSettings  # noqa: E402
from poolguard.events import BoundingBox, Detection, TrackedPerson  # noqa: E402
from poolguard.geometry import WHOLE_FRAME  # noqa: E402
from poolguard.replay import TrackedFrameResult  # noqa: E402
from poolguard.vision.frames import Frame  # noqa: E402
from poolguard.vision.render import (  # noqa: E402
    _hud_state,
    annotate_frame,
    color_for_track,
    render_tracked_replay,
)

START = datetime(2026, 1, 1, tzinfo=UTC)
WIDTH, HEIGHT = 64, 48


def blank_image() -> np.ndarray:
    return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)


def person(track_id: int, unseen: float = 0.0, in_water: bool = True) -> TrackedPerson:
    return TrackedPerson(
        track_id=track_id,
        detection=Detection(
            frame_ts=START,
            box=BoundingBox(x=0.25, y=0.25, width=0.5, height=0.5),
            confidence=0.9,
        ),
        in_water=in_water,
        seconds_since_last_seen=unseen,
    )


class TestAnnotateFrame:
    def test_returns_new_array_without_touching_input(self) -> None:
        image = blank_image()
        image.setflags(write=False)  # Frame images are read-only in the pipeline

        canvas = annotate_frame(image, (person(1),), pool_zone=WHOLE_FRAME)

        assert canvas is not image
        assert not image.any()  # input still all-black

    def test_live_person_is_drawn(self) -> None:
        canvas = annotate_frame(blank_image(), (person(1),), pool_zone=WHOLE_FRAME)

        assert canvas.any()  # something was drawn

    def test_coasting_person_is_drawn(self) -> None:
        canvas = annotate_frame(blank_image(), (person(1, unseen=4.2),), pool_zone=WHOLE_FRAME)

        assert canvas.any()

    def test_coasting_looks_different_from_live(self) -> None:
        live = annotate_frame(blank_image(), (person(1),), pool_zone=WHOLE_FRAME)
        ghost = annotate_frame(blank_image(), (person(1, unseen=4.2),), pool_zone=WHOLE_FRAME)

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


def test_importing_render_does_not_need_pyyaml(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rendering only uses ShotLabel as a type; PyYAML is not a vision dependency."""
    monkeypatch.setitem(sys.modules, "yaml", None)  # makes `import yaml` raise ImportError
    monkeypatch.delitem(sys.modules, "poolguard.labels", raising=False)
    monkeypatch.delitem(sys.modules, "poolguard.vision.render", raising=False)

    importlib.import_module("poolguard.vision.render")


def test_pool_zone_outline_is_drawn() -> None:
    triangle = ((0.1, 0.1), (0.9, 0.1), (0.5, 0.9))

    canvas = annotate_frame(blank_image(), (), pool_zone=triangle)

    assert canvas.any()


class TestHudStateFromTracks:
    """Only a track lost *in the water* runs the submersion clock."""

    @staticmethod
    def hud(*people: TrackedPerson):  # type: ignore[no-untyped-def]
        result = TrackedFrameResult(frame_index=0, ts=START, detections=(), people=people)
        return _hud_state(
            result,
            start_ts=START,
            frames_seen=1,
            frames_with_detections=1,
            label=None,
            rules=RulesSettings(),
        )

    def test_track_lost_on_the_deck_does_not_run_the_clock(self) -> None:
        state = self.hud(person(1, unseen=20.0, in_water=False))

        assert state.coasting_tracks == 1
        assert state.longest_unseen_seconds == 0.0
        assert state.submersion_progress == 0.0

    def test_track_lost_in_the_water_runs_the_clock(self) -> None:
        state = self.hud(
            person(1, unseen=20.0, in_water=False),
            person(2, unseen=7.0, in_water=True),
        )

        assert state.coasting_tracks == 2
        assert state.longest_unseen_seconds == 7.0
