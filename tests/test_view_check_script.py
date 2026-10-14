"""End-to-end test of poolguard-view-check on small synthetic videos (vision extra)."""

import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="needs the vision extra: uv sync --extra vision")

from poolguard.scripts import view_check  # noqa: E402

WIDTH, HEIGHT = 640, 480
ZONE_JSON = "[[0.3, 0.4], [0.7, 0.4], [0.7, 0.8], [0.3, 0.8]]"


def scene(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.full((HEIGHT, WIDTH, 3), 128, dtype=np.uint8)
    for _ in range(120):
        x, y = int(rng.integers(0, WIDTH)), int(rng.integers(0, HEIGHT))
        size = int(rng.integers(8, 40))
        color = tuple(int(c) for c in rng.integers(0, 255, size=3))
        cv2.rectangle(image, (x, y), (x + size, y + size), color, -1)
    return image


def write_video(path: Path, image: np.ndarray, frames: int = 12) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (WIDTH, HEIGHT))
    assert writer.isOpened()
    for _ in range(frames):
        writer.write(image)
    writer.release()
    return path


def run(monkeypatch: pytest.MonkeyPatch, *arguments: str) -> None:
    monkeypatch.setattr(sys, "argv", ["poolguard-view-check", *arguments])
    view_check.main()


@pytest.fixture
def zone_configured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)  # no stray .env
    monkeypatch.setenv("POOLGUARD_TRACKING_POOL_ZONE", ZONE_JSON)


def test_save_then_check_same_view_passes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    zone_configured: None,
) -> None:
    video = write_video(tmp_path / "view.mp4", scene(seed=1))
    reference = tmp_path / "calibration" / "cam.viewref.npz"

    run(monkeypatch, "save", "--video", str(video), "--reference", str(reference))
    run(monkeypatch, "check", "--video", str(video), "--reference", str(reference))

    assert reference.is_file()
    assert "aligned" in capsys.readouterr().out


def test_check_exits_non_zero_and_prints_the_event_when_the_view_moved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    zone_configured: None,
) -> None:
    image = scene(seed=1)
    moved = cv2.warpAffine(
        image, np.float32([[1, 0, 60], [0, 1, 0]]), (WIDTH, HEIGHT), borderValue=(128, 128, 128)
    )
    reference = tmp_path / "cam.viewref.npz"
    run(
        monkeypatch,
        "save",
        "--video",
        str(write_video(tmp_path / "before.mp4", image)),
        "--reference",
        str(reference),
    )

    with pytest.raises(SystemExit) as raised:
        run(
            monkeypatch,
            "check",
            "--video",
            str(write_video(tmp_path / "after.mp4", moved)),
            "--reference",
            str(reference),
        )

    assert raised.value.code == 1
    output = capsys.readouterr().out
    assert "moved" in output
    assert "system_degraded" in output


def test_save_refuses_when_no_pool_zone_is_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With the default whole-frame zone there is nothing outside it to use as a landmark."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("POOLGUARD_TRACKING_POOL_ZONE", raising=False)
    video = write_video(tmp_path / "view.mp4", scene(seed=1))

    with pytest.raises(SystemExit) as raised:
        run(monkeypatch, "save", "--video", str(video), "--reference", str(tmp_path / "r.npz"))

    assert "pool zone" in str(raised.value)
