"""Tests for VideoFileFrameSource, using a tiny generated clip."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from poolguard.vision.video import VideoFileFrameSource  # noqa: E402

FPS = 4.0
FRAME_COUNT = 8
WIDTH, HEIGHT = 64, 48


@pytest.fixture
def video_path(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    assert writer.isOpened()
    for i in range(FRAME_COUNT):
        image = np.full((HEIGHT, WIDTH, 3), i * 10, dtype=np.uint8)
        writer.write(image)
    writer.release()
    return path


def test_yields_all_frames_in_order(video_path: Path) -> None:
    source = VideoFileFrameSource(video_path)

    frames = list(source.frames())

    assert len(frames) == FRAME_COUNT
    assert [f.index for f in frames] == list(range(FRAME_COUNT))
    assert all(f.width == WIDTH and f.height == HEIGHT for f in frames)


def test_synthetic_timestamps_follow_fps(video_path: Path) -> None:
    start = datetime(2026, 7, 29, 12, 0, 0, tzinfo=UTC)
    source = VideoFileFrameSource(video_path, start_time=start)

    frames = list(source.frames())

    assert frames[0].ts == start
    assert frames[1].ts - frames[0].ts == timedelta(seconds=1 / FPS)
    assert frames[-1].ts == start + timedelta(seconds=(FRAME_COUNT - 1) / FPS)


def test_replay_is_reproducible(video_path: Path) -> None:
    first = [f.ts for f in VideoFileFrameSource(video_path).frames()]
    second = [f.ts for f in VideoFileFrameSource(video_path).frames()]

    assert first == second


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        VideoFileFrameSource(tmp_path / "nope.mp4")


def test_naive_start_time_raises(video_path: Path) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        VideoFileFrameSource(video_path, start_time=datetime(2026, 7, 29, 12, 0, 0))
