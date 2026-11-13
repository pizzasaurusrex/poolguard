"""Measure how far the camera view has moved from a saved reference.

Requires the `vision` extra (opencv). The reference holds landmarks only:
the positions and descriptors of fixed points around the pool, not a picture.
It still describes a private scene, so it is saved outside the repository and
never committed.

The comparison matches landmarks between the reference and a live image, fits
the single transform most of them agree on, and reports how far that transform
moves the pool-zone vertices. camera_view.classify_view turns that into a
verdict.
"""

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from poolguard.camera_view import ViewCheck, classify_view
from poolguard.config import ViewSettings
from poolguard.geometry import Polygon

MAX_LANDMARKS = 2000
DESCRIPTOR_BYTES = 32
"""ORB describes each landmark with 256 bits."""
MATCH_RATIO = 0.75
"""A match counts only if it is clearly better than the runner-up (Lowe's ratio test)."""
RANSAC_TOLERANCE_PIXELS = 5.0
MIN_MATCHES_FOR_TRANSFORM = 4
"""A perspective transform has 8 unknowns, so it needs 4 point pairs."""


class ViewReference(BaseModel):
    """Landmarks of the view the pool zone was traced for."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    points: np.ndarray
    """Landmark positions, shape (n, 2), normalized (x, y)."""
    descriptors: np.ndarray
    """One ORB descriptor per landmark, shape (n, 32), uint8."""

    @model_validator(mode="after")
    def _shapes_agree(self) -> "ViewReference":
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError(f"points must have shape (n, 2), got {self.points.shape}")
        if self.descriptors.shape != (self.points.shape[0], DESCRIPTOR_BYTES):
            raise ValueError(
                f"need one descriptor per point: {self.points.shape[0]} points, "
                f"descriptors of shape {self.descriptors.shape}"
            )
        return self


def steady_image(images: Sequence[np.ndarray]) -> np.ndarray:
    """Per-pixel median of several frames: whatever moves between them drops out."""
    return np.median(np.stack(images), axis=0).astype(np.uint8)


def build_reference(image: np.ndarray, zone: Polygon) -> ViewReference:
    """Record the landmarks around the pool zone in this view.

    Points inside the zone are left out: water ripples and reflects, so it
    offers nothing that stays put.
    """
    height, width = image.shape[:2]
    outside_zone = np.full((height, width), 255, dtype=np.uint8)
    cv2.fillPoly(outside_zone, [_to_pixels(zone, width, height).astype(np.int32)], 0)

    points, descriptors = _landmarks(image, mask=outside_zone)
    return ViewReference(
        points=(points / np.float32([width, height])).astype(np.float32),
        descriptors=descriptors,
    )


def compare_view(
    reference: ViewReference, image: np.ndarray, zone: Polygon, settings: ViewSettings
) -> ViewCheck:
    """Compare a live image with the reference and judge whether the zone still fits."""
    height, width = image.shape[:2]
    frame_size = np.float32([width, height])
    live_points, live_descriptors = _landmarks(image, mask=None)

    pairs = _match(reference.descriptors, live_descriptors)
    if len(pairs) < MIN_MATCHES_FOR_TRANSFORM:
        return classify_view(len(pairs), 0.0, None, settings)

    reference_pixels = reference.points[[r for r, _ in pairs]] * frame_size
    live_pixels = live_points[[live for _, live in pairs]]
    transform, agreeing = cv2.findHomography(
        reference_pixels, live_pixels, cv2.RANSAC, RANSAC_TOLERANCE_PIXELS
    )
    if transform is None:
        return classify_view(0, 0.0, None, settings)

    agreeing_points = live_pixels[agreeing.ravel() == 1] / frame_size
    return classify_view(
        matched_landmarks=len(agreeing_points),
        landmark_spread=_spread(agreeing_points),
        zone_shift=_zone_shift(zone, transform, width, height),
        settings=settings,
    )


def save_reference(reference: ViewReference, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        np.savez_compressed(file, points=reference.points, descriptors=reference.descriptors)


def load_reference(path: Path) -> ViewReference:
    if not path.is_file():
        raise FileNotFoundError(
            f"view reference not found: {path} — create it with: poolguard-view-check save"
        )
    with np.load(path, allow_pickle=False) as data:
        return ViewReference(points=data["points"], descriptors=data["descriptors"])


def _landmarks(image: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Landmark pixel positions (n, 2) and their descriptors (n, 32)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = cv2.ORB_create(nfeatures=MAX_LANDMARKS).detectAndCompute(gray, mask)
    if descriptors is None:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, DESCRIPTOR_BYTES), dtype=np.uint8),
        )
    return np.float32([keypoint.pt for keypoint in keypoints]), descriptors


def _match(
    reference_descriptors: np.ndarray, live_descriptors: np.ndarray
) -> list[tuple[int, int]]:
    """(reference_index, live_index) for each confident landmark match."""
    if len(reference_descriptors) < 2 or len(live_descriptors) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = []
    for candidates in matcher.knnMatch(reference_descriptors, live_descriptors, k=2):
        if len(candidates) < 2:
            continue
        best, runner_up = candidates
        if best.distance < MATCH_RATIO * runner_up.distance:
            pairs.append((best.queryIdx, best.trainIdx))
    return pairs


def _spread(normalized_points: np.ndarray) -> float:
    """Smaller of the width and height of the box around the points, as a frame share."""
    if len(normalized_points) == 0:
        return 0.0
    extent = normalized_points.max(axis=0) - normalized_points.min(axis=0)
    return float(min(max(extent.min(), 0.0), 1.0))


def _zone_shift(zone: Polygon, transform: np.ndarray, width: int, height: int) -> float:
    """Farthest the transform moves any zone vertex, as a fraction of frame width."""
    before = _to_pixels(zone, width, height)
    after = cv2.perspectiveTransform(before.reshape(-1, 1, 2), transform).reshape(-1, 2)
    return float(np.linalg.norm(after - before, axis=1).max() / width)


def _to_pixels(polygon: Polygon, width: int, height: int) -> np.ndarray:
    return np.float32(polygon) * np.float32([width, height])
