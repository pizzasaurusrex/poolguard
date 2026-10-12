"""Ultralytics YOLO pose backend for dev machines (ADR-010).

Requires the `vision` extra. Runs FP weights (MPS-accelerated on Apple
Silicon via ultralytics' device auto-selection); the Pi runs the INT8 Hailo
backend instead — see docs/edge-inference.md for the accuracy caveat.
"""

import math
from pathlib import Path

from ultralytics import YOLO

from poolguard.events import Detection
from poolguard.vision.frames import Frame
from poolguard.vision.pose import detections_from_arrays

DEFAULT_MODEL = "yolo11n-pose.pt"

DEFAULT_MIN_CONFIDENCE = 0.1
"""Matches TrackingSettings.low_confidence: the tracker's rescue pass uses weak
detections down to that floor, so filtering higher here hides them from it."""

MODEL_STRIDE = 32
"""YOLO downsamples by this factor, so the input size must be a multiple of it."""

DEFAULT_MAX_INFERENCE_SIZE = 1920
"""Cap on the model's input size. Inference cost grows with the square of this."""


def inference_size(frame_width: int, frame_height: int, max_size: int) -> int:
    """Input size (longest side, in pixels) to run the model at for one frame.

    Ultralytics defaults to 640, which shrinks a 1080p frame until a distant
    swimmer is a few pixels tall and is not detected at all (decision log,
    2026-08-25). Running near the frame's native resolution fixes that.
    """
    longest_side = max(frame_width, frame_height)
    # Round up: rounding down would discard pixels, which is what loses small people.
    rounded_up = math.ceil(longest_side / MODEL_STRIDE) * MODEL_STRIDE
    largest_allowed = (max_size // MODEL_STRIDE) * MODEL_STRIDE
    return min(rounded_up, largest_allowed)


class UltralyticsPoseEstimator:
    def __init__(
        self,
        model: str | Path = DEFAULT_MODEL,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        device: str | None = None,
        max_inference_size: int = DEFAULT_MAX_INFERENCE_SIZE,
    ) -> None:
        self._model = YOLO(model)
        self._min_confidence = min_confidence
        self._max_inference_size = max_inference_size
        self._device = device

    def estimate(self, frame: Frame) -> tuple[Detection, ...]:
        results = self._model.predict(
            frame.image,
            conf=self._min_confidence,
            imgsz=inference_size(frame.width, frame.height, self._max_inference_size),
            device=self._device,
            verbose=False,
        )
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return ()

        keypoints = None
        if result.keypoints is not None:
            keypoints = result.keypoints.data.cpu().numpy()

        return detections_from_arrays(
            boxes_xyxy=result.boxes.xyxy.cpu().numpy(),
            confidences=result.boxes.conf.cpu().numpy(),
            keypoints_xyc=keypoints,
            frame=frame,
        )
