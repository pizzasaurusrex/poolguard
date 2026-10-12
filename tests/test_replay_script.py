"""The replay CLI's label lookup (vision extra, for the module import only)."""

import argparse
import sys

import pytest

pytest.importorskip("cv2")
pytest.importorskip("ultralytics")

from poolguard.scripts import replay  # noqa: E402


def test_labels_without_pyyaml_exits_with_the_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "yaml", None)  # makes `import yaml` raise ImportError
    monkeypatch.delitem(sys.modules, "poolguard.labels", raising=False)
    args = argparse.Namespace(labels="tests/labels/shots.yaml", video="clips/any.mp4")

    with pytest.raises(SystemExit) as raised:
        replay._find_label(args)

    assert "uv sync --extra labels" in str(raised.value)


def test_no_labels_argument_means_no_label() -> None:
    args = argparse.Namespace(labels=None, video="clips/any.mp4")

    assert replay._find_label(args) == (None, None)


def test_label_with_a_camera_brings_its_profile(tmp_path) -> None:  # type: ignore[no-untyped-def]
    manifest = tmp_path / "shots.yaml"
    manifest.write_text(
        """
cameras:
  backyard:
    water_polygon: [[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]]
shots:
  - shot: swim.mp4
    source: swim.mp4
    duration_seconds: 5.0
    people: 1
    camera: backyard
"""
    )
    args = argparse.Namespace(labels=str(manifest), video="clips/swim.mp4")

    label, camera = replay._find_label(args)

    assert label is not None and label.shot == "swim.mp4"
    assert camera is not None
    assert camera.water_polygon == ((0.1, 0.1), (0.9, 0.1), (0.5, 0.9))
