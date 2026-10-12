"""Ultralytics YOLO pose backend for dev machines (ADR-010).

Requires the `vision` extra. Runs FP weights (MPS-accelerated on Apple
Silicon via ultralytics' device auto-selection); the Pi runs the INT8 Hailo
backend instead — see docs/edge-inference.md for the accuracy caveat.
"""

from pathlib import Path

from ultralytics import YOLO

from poolguard.events import Detection
from poolguard.vision.frames import Frame
from poolguard.vision.pose import detections_from_arrays

DEFAULT_MODEL = "yolo11n-pose.pt"

DEFAULT_MIN_CONFIDENCE = 0.1
"""Matches TrackingSettings.low_confidence: the tracker's rescue pass uses weak
detections down to that floor, so filtering higher here hides them from it."""


class UltralyticsPoseEstimator:
    def __init__(
        self,
        model: str | Path = DEFAULT_MODEL,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        device: str | None = None,
    ) -> None:
        self._model = YOLO(model)
        self._min_confidence = min_confidence
        self._device = device

    def estimate(self, frame: Frame) -> tuple[Detection, ...]:
        results = self._model.predict(
            frame.image,
            conf=self._min_confidence,
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
