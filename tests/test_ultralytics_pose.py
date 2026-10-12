"""Tests for UltralyticsPoseEstimator with a faked YOLO model.

Requires the `vision` extra only for the module import (ultralytics); the
model itself is faked so no weights are downloaded and no inference runs.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

pytest.importorskip("ultralytics")

from poolguard.config import TrackingSettings  # noqa: E402
from poolguard.vision.frames import Frame  # noqa: E402
from poolguard.vision.ultralytics_pose import (  # noqa: E402
    DEFAULT_MIN_CONFIDENCE,
    MODEL_STRIDE,
    UltralyticsPoseEstimator,
    inference_size,
)

TS = datetime(2026, 1, 1, tzinfo=UTC)


class FakeTensor:
    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._array


class FakeBoxes:
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray) -> None:
        self.xyxy = FakeTensor(xyxy)
        self.conf = FakeTensor(conf)

    def __len__(self) -> int:
        return self.xyxy.numpy().shape[0]


class FakeKeypoints:
    def __init__(self, data: np.ndarray) -> None:
        self.data = FakeTensor(data)


class FakeResult:
    def __init__(self, boxes: FakeBoxes | None, keypoints: FakeKeypoints | None) -> None:
        self.boxes = boxes
        self.keypoints = keypoints


class FakeYOLO:
    def __init__(self, result: FakeResult) -> None:
        self._result = result
        self.predict_kwargs: dict | None = None

    def predict(self, image: np.ndarray, **kwargs) -> list[FakeResult]:
        self.predict_kwargs = kwargs
        return [self._result]


@pytest.fixture
def frame() -> Frame:
    return Frame(image=np.zeros((100, 200, 3), dtype=np.uint8), ts=TS, index=0)


def make_estimator(monkeypatch: pytest.MonkeyPatch, result: FakeResult) -> UltralyticsPoseEstimator:
    fake = FakeYOLO(result)
    monkeypatch.setattr("poolguard.vision.ultralytics_pose.YOLO", lambda model: fake)
    return UltralyticsPoseEstimator()


def test_no_boxes_yields_no_detections(monkeypatch: pytest.MonkeyPatch, frame: Frame) -> None:
    estimator = make_estimator(monkeypatch, FakeResult(boxes=None, keypoints=None))

    assert estimator.estimate(frame) == ()


def test_empty_boxes_yields_no_detections(monkeypatch: pytest.MonkeyPatch, frame: Frame) -> None:
    boxes = FakeBoxes(np.zeros((0, 4)), np.zeros((0,)))
    estimator = make_estimator(monkeypatch, FakeResult(boxes=boxes, keypoints=None))

    assert estimator.estimate(frame) == ()


def test_boxes_and_keypoints_are_converted(monkeypatch: pytest.MonkeyPatch, frame: Frame) -> None:
    boxes = FakeBoxes(np.array([[20.0, 10.0, 120.0, 60.0]]), np.array([0.9]))
    keypoints = FakeKeypoints(np.full((1, 17, 3), 50.0))
    estimator = make_estimator(monkeypatch, FakeResult(boxes=boxes, keypoints=keypoints))

    detections = estimator.estimate(frame)

    assert len(detections) == 1
    assert detections[0].box.x == pytest.approx(0.1)
    assert detections[0].confidence == pytest.approx(0.9)
    assert detections[0].keypoints is not None
    assert len(detections[0].keypoints) == 17


def test_detection_only_result_has_no_keypoints(
    monkeypatch: pytest.MonkeyPatch, frame: Frame
) -> None:
    boxes = FakeBoxes(np.array([[20.0, 10.0, 120.0, 60.0]]), np.array([0.9]))
    estimator = make_estimator(monkeypatch, FakeResult(boxes=boxes, keypoints=None))

    detections = estimator.estimate(frame)

    assert len(detections) == 1
    assert detections[0].keypoints is None


def test_default_floor_lets_the_whole_rescue_band_through(
    monkeypatch: pytest.MonkeyPatch, frame: Frame
) -> None:
    """The tracker's rescue pass needs every detection down to low_confidence.

    A higher floor here silently discards the bottom of that band before the
    tracker can use it.
    """
    estimator = make_estimator(monkeypatch, FakeResult(boxes=None, keypoints=None))

    estimator.estimate(frame)

    assert TrackingSettings(_env_file=None).low_confidence == DEFAULT_MIN_CONFIDENCE
    assert estimator._model.predict_kwargs["conf"] == DEFAULT_MIN_CONFIDENCE


class TestInferenceSize:
    @pytest.mark.parametrize(
        ("width", "height", "max_size", "expected"),
        [
            (1920, 1080, 1920, 1920),  # the backyard camera: run at native size
            (1280, 720, 1920, 1280),  # smaller frame: no point upscaling past native
            (1080, 1920, 1920, 1920),  # portrait: the longest side is the height
            (3840, 2160, 1920, 1920),  # 4K: capped, or inference is too slow
            (200, 100, 1920, 224),  # not a multiple of the stride: round up, never lose pixels
            (2000, 1000, 1920, 1920),  # rounding up must not push past the cap
        ],
    )
    def test_size_for_frame(self, width: int, height: int, max_size: int, expected: int) -> None:
        assert inference_size(width, height, max_size) == expected

    def test_size_is_always_a_stride_multiple(self) -> None:
        for longest_side in range(MODEL_STRIDE, 2000, 7):
            assert inference_size(longest_side, 10, 1920) % MODEL_STRIDE == 0


def test_estimator_runs_near_native_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """The 640 default detects nobody on the 1080p backyard footage."""
    estimator = make_estimator(monkeypatch, FakeResult(boxes=None, keypoints=None))
    full_hd = Frame(image=np.zeros((1080, 1920, 3), dtype=np.uint8), ts=TS, index=0)

    estimator.estimate(full_hd)

    assert estimator._model.predict_kwargs["imgsz"] == 1920
