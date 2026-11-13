"""Tests for measuring camera movement against a saved reference (vision extra).

Scenes are synthetic: random shapes give the landmark detector corners to
find, and shifting the picture by a known number of pixels gives a movement
whose size the check must recover.
"""

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="needs the vision extra: uv sync --extra vision")

from poolguard.camera_view import ViewVerdict  # noqa: E402
from poolguard.config import ViewSettings  # noqa: E402
from poolguard.vision.view_compare import (  # noqa: E402
    ViewReference,
    build_reference,
    compare_view,
    load_reference,
    save_reference,
    steady_image,
)

WIDTH, HEIGHT = 640, 480
ZONE = ((0.3, 0.4), (0.7, 0.4), (0.7, 0.8), (0.3, 0.8))
SETTINGS = ViewSettings(_env_file=None)


def scene(seed: int) -> np.ndarray:
    """A cluttered picture: 120 random rectangles and circles on mid-grey."""
    rng = np.random.default_rng(seed)
    image = np.full((HEIGHT, WIDTH, 3), 128, dtype=np.uint8)
    for _ in range(120):
        x, y = int(rng.integers(0, WIDTH)), int(rng.integers(0, HEIGHT))
        size = int(rng.integers(8, 40))
        color = tuple(int(c) for c in rng.integers(0, 255, size=3))
        if rng.random() < 0.5:
            cv2.rectangle(image, (x, y), (x + size, y + size), color, -1)
        else:
            cv2.circle(image, (x, y), size // 2, color, -1)
    return image


def shifted(image: np.ndarray, pixels_right: int) -> np.ndarray:
    matrix = np.float32([[1, 0, pixels_right], [0, 1, 0]])
    return cv2.warpAffine(image, matrix, (WIDTH, HEIGHT), borderValue=(128, 128, 128))


class TestCompareView:
    def test_same_view_is_aligned(self) -> None:
        image = scene(seed=1)
        check = compare_view(build_reference(image, ZONE), image, ZONE, SETTINGS)

        assert check.verdict is ViewVerdict.ALIGNED
        assert check.zone_shift == pytest.approx(0.0, abs=0.003)

    def test_shifted_view_is_moved_by_the_right_amount(self) -> None:
        image = scene(seed=1)
        check = compare_view(build_reference(image, ZONE), shifted(image, 48), ZONE, SETTINGS)

        assert check.verdict is ViewVerdict.MOVED
        assert check.zone_shift == pytest.approx(48 / WIDTH, abs=0.005)

    def test_small_shift_within_tolerance_is_aligned(self) -> None:
        image = scene(seed=1)
        check = compare_view(build_reference(image, ZONE), shifted(image, 4), ZONE, SETTINGS)

        assert check.verdict is ViewVerdict.ALIGNED

    def test_unrelated_view_is_unknown(self) -> None:
        check = compare_view(build_reference(scene(seed=1), ZONE), scene(seed=2), ZONE, SETTINGS)

        assert check.verdict is ViewVerdict.UNKNOWN

    def test_featureless_view_is_unknown(self) -> None:
        """A covered lens or a blank frame must not read as 'nothing moved'."""
        blank = np.full((HEIGHT, WIDTH, 3), 128, dtype=np.uint8)
        check = compare_view(build_reference(scene(seed=1), ZONE), blank, ZONE, SETTINGS)

        assert check.verdict is ViewVerdict.UNKNOWN
        assert check.zone_shift is None

    def test_shared_logo_does_not_vouch_for_a_different_view(self) -> None:
        """Two different scenes with the same burned-in corner logo are not aligned."""
        logo = scene(seed=9)[0:60, 0:140]
        before, after = scene(seed=1), scene(seed=2)
        before[0:60, 0:140] = logo
        after[0:60, 0:140] = logo

        check = compare_view(build_reference(before, ZONE), after, ZONE, SETTINGS)

        assert check.verdict is ViewVerdict.UNKNOWN


class TestReference:
    def test_landmarks_inside_the_pool_zone_are_left_out(self) -> None:
        """Water ripples; only the fixed surroundings make stable landmarks."""
        reference = build_reference(scene(seed=1), ZONE)

        inside = [(x, y) for x, y in reference.points if 0.32 < x < 0.68 and 0.42 < y < 0.78]
        assert len(reference.points) > 100
        assert inside == []

    def test_round_trips_through_a_file(self, tmp_path: Path) -> None:
        reference = build_reference(scene(seed=1), ZONE)
        path = tmp_path / "backyard.viewref.npz"

        save_reference(reference, path)
        loaded = load_reference(path)

        assert np.array_equal(loaded.points, reference.points)
        assert np.array_equal(loaded.descriptors, reference.descriptors)

    def test_rejects_mismatched_points_and_descriptors(self) -> None:
        with pytest.raises(ValueError, match="one descriptor per point"):
            ViewReference(
                points=np.zeros((3, 2), dtype=np.float32),
                descriptors=np.zeros((2, 32), dtype=np.uint8),
            )

    def test_missing_file_says_how_to_create_it(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="poolguard-view-check save"):
            load_reference(tmp_path / "absent.npz")


def test_steady_image_removes_something_that_moves() -> None:
    background = scene(seed=1)
    frames = []
    for offset in range(5):
        frame = background.copy()
        cv2.circle(frame, (100 + offset * 100, 240), 30, (0, 0, 255), -1)  # a "swimmer"
        frames.append(frame)

    steady = steady_image(frames)

    assert np.array_equal(steady, background)
