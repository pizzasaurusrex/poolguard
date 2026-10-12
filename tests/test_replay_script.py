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

    assert replay._find_label(args) is None
