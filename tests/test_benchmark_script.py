"""The FPS benchmark, run against a fake cv2 so no camera or vision extra is needed."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from poolguard.scripts import benchmark_fps

RTSP_URL = "rtsp://admin:hunter2@camera.example:554/stream"
PASSWORD = "hunter2"


class FakeCapture:
    def __init__(self, opened: bool) -> None:
        self._opened = opened
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 (cv2's name)
        return self._opened

    def read(self) -> tuple[bool, None]:
        return True, None

    def release(self) -> None:
        self.released = True


def fake_cv2(monkeypatch: pytest.MonkeyPatch, opened: bool = True) -> list[str]:
    """Install a stand-in cv2 module; returns the URLs it was asked to open."""
    opened_urls: list[str] = []

    def video_capture(url: str) -> FakeCapture:
        opened_urls.append(url)
        return FakeCapture(opened)

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(VideoCapture=video_capture))
    return opened_urls


@pytest.fixture
def no_env_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("POOLGUARD_CAMERA_RTSP_URL", raising=False)


def test_unopenable_stream_error_does_not_leak_the_url(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cv2(monkeypatch, opened=False)

    with pytest.raises(SystemExit) as raised:
        benchmark_fps.measure_fps(SecretStr(RTSP_URL), duration_seconds=0.0)

    assert PASSWORD not in str(raised.value)


def test_main_reads_the_url_from_settings_and_never_prints_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], no_env_file: None
) -> None:
    opened_urls = fake_cv2(monkeypatch)
    monkeypatch.setenv("POOLGUARD_CAMERA_RTSP_URL", RTSP_URL)
    monkeypatch.setattr(sys, "argv", ["poolguard-benchmark", "--duration", "0"])

    benchmark_fps.main()

    assert opened_urls == [RTSP_URL]
    assert PASSWORD not in capsys.readouterr().out


def test_main_names_the_missing_setting(monkeypatch: pytest.MonkeyPatch, no_env_file: None) -> None:
    fake_cv2(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["poolguard-benchmark"])

    with pytest.raises(SystemExit) as raised:
        benchmark_fps.main()

    assert "POOLGUARD_CAMERA_RTSP_URL" in str(raised.value)
