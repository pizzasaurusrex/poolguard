"""Tests for pixel-to-normalized detection conversion."""

from datetime import UTC, datetime

import numpy as np
import pytest

from poolguard.vision.frames import Frame
from poolguard.vision.pose import detections_from_arrays

TS = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def frame() -> Frame:
    return Frame(image=np.zeros((100, 200, 3), dtype=np.uint8), ts=TS, index=0)


def test_box_is_normalized_to_frame_size(frame: Frame) -> None:
    boxes = np.array([[20.0, 10.0, 120.0, 60.0]])
    confidences = np.array([0.9])

    detections = detections_from_arrays(boxes, confidences, None, frame)

    assert len(detections) == 1
    box = detections[0].box
    assert box.x == pytest.approx(0.1)
    assert box.y == pytest.approx(0.1)
    assert box.width == pytest.approx(0.5)
    assert box.height == pytest.approx(0.5)
    assert detections[0].confidence == pytest.approx(0.9)
    assert detections[0].frame_ts == TS
    assert detections[0].keypoints is None


def test_box_extending_past_frame_is_clipped(frame: Frame) -> None:
    boxes = np.array([[-10.0, -5.0, 250.0, 120.0]])
    confidences = np.array([0.8])

    detections = detections_from_arrays(boxes, confidences, None, frame)

    box = detections[0].box
    assert box.x == 0.0
    assert box.y == 0.0
    assert box.width == 1.0
    assert box.height == 1.0


def test_degenerate_box_is_dropped(frame: Frame) -> None:
    boxes = np.array([[50.0, 50.0, 50.0, 80.0], [20.0, 10.0, 40.0, 30.0]])
    confidences = np.array([0.9, 0.7])

    detections = detections_from_arrays(boxes, confidences, None, frame)

    assert len(detections) == 1
    assert detections[0].confidence == pytest.approx(0.7)


def test_box_entirely_outside_frame_is_dropped(frame: Frame) -> None:
    boxes = np.array([[300.0, 10.0, 400.0, 60.0]])
    confidences = np.array([0.9])

    assert detections_from_arrays(boxes, confidences, None, frame) == ()


def test_keypoints_are_normalized_and_clamped(frame: Frame) -> None:
    boxes = np.array([[0.0, 0.0, 200.0, 100.0]])
    confidences = np.array([0.9])
    keypoints = np.zeros((1, 17, 3))
    keypoints[0, 0] = [100.0, 50.0, 0.8]
    keypoints[0, 1] = [-20.0, 150.0, 1.5]

    detections = detections_from_arrays(boxes, confidences, keypoints, frame)

    kps = detections[0].keypoints
    assert kps is not None
    assert len(kps) == 17
    assert kps[0] == pytest.approx((0.5, 0.5, 0.8))
    assert kps[1] == pytest.approx((0.0, 1.0, 1.0))


def test_empty_input_yields_no_detections(frame: Frame) -> None:
    boxes = np.zeros((0, 4))
    confidences = np.zeros((0,))

    assert detections_from_arrays(boxes, confidences, None, frame) == ()


def test_out_of_range_confidence_is_clamped(frame: Frame) -> None:
    boxes = np.array([[10.0, 10.0, 50.0, 50.0]])
    confidences = np.array([1.2])

    detections = detections_from_arrays(boxes, confidences, None, frame)

    assert detections[0].confidence == 1.0
