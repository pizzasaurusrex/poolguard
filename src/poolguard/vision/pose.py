"""Pose-estimation seam (ADR-010).

The pipeline depends on the PoseEstimator protocol only. Concrete backends:
UltralyticsPoseEstimator (dev machines, `ultralytics_pose.py`) and a Hailo
backend written at P0. Both emit the same normalized Detection events, so
downstream tracking and rules cannot tell them apart.
"""

from typing import Protocol

import numpy as np

from poolguard.events import BoundingBox, Detection
from poolguard.vision.frames import Frame

COCO_KEYPOINT_COUNT = 17


class PoseEstimator(Protocol):
    """Detects people (and optionally their pose keypoints) in one frame."""

    def estimate(self, frame: Frame) -> tuple[Detection, ...]: ...


def detections_from_arrays(
    boxes_xyxy: np.ndarray,
    confidences: np.ndarray,
    keypoints_xyc: np.ndarray | None,
    frame: Frame,
) -> tuple[Detection, ...]:
    """Convert pixel-space model output into normalized Detection events.

    boxes_xyxy: (n, 4) pixel corners; confidences: (n,); keypoints_xyc:
    (n, k, 3) pixel x, pixel y, confidence — or None for detection-only
    backends. Boxes are clipped to the frame; degenerate boxes are dropped.
    """
    width = float(frame.width)
    height = float(frame.height)

    detections: list[Detection] = []
    for i in range(boxes_xyxy.shape[0]):
        x1, y1, x2, y2 = boxes_xyxy[i]
        x1 = min(max(float(x1) / width, 0.0), 1.0)
        y1 = min(max(float(y1) / height, 0.0), 1.0)
        x2 = min(max(float(x2) / width, 0.0), 1.0)
        y2 = min(max(float(y2) / height, 0.0), 1.0)
        if x2 <= x1 or y2 <= y1:
            continue

        keypoints = None
        if keypoints_xyc is not None:
            keypoints = _normalize_keypoints(keypoints_xyc[i], width, height)

        detections.append(
            Detection(
                frame_ts=frame.ts,
                box=BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1),
                confidence=min(max(float(confidences[i]), 0.0), 1.0),
                keypoints=keypoints,
            )
        )
    return tuple(detections)


def _normalize_keypoints(
    keypoints: np.ndarray, width: float, height: float
) -> tuple[tuple[float, float, float], ...]:
    normalized = []
    for x, y, confidence in keypoints:
        normalized.append(
            (
                min(max(float(x) / width, 0.0), 1.0),
                min(max(float(y) / height, 0.0), 1.0),
                min(max(float(confidence), 0.0), 1.0),
            )
        )
    return tuple(normalized)
