"""Tests for the Frame model."""

from datetime import UTC, datetime

import numpy as np
import pytest
from pydantic import ValidationError

from poolguard.vision.frames import Frame

TS = datetime(2026, 1, 1, tzinfo=UTC)


def make_image(height: int = 4, width: int = 6) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_frame_exposes_dimensions() -> None:
    frame = Frame(image=make_image(height=4, width=6), ts=TS, index=0)
    assert frame.height == 4
    assert frame.width == 6


def test_frame_image_becomes_read_only() -> None:
    frame = Frame(image=make_image(), ts=TS, index=0)
    with pytest.raises(ValueError):
        frame.image[0, 0, 0] = 255


@pytest.mark.parametrize("shape", [(4, 6), (4, 6, 1), (4, 6, 4)])
def test_frame_rejects_non_bgr_shapes(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValidationError):
        Frame(image=np.zeros(shape, dtype=np.uint8), ts=TS, index=0)


def test_frame_rejects_negative_index() -> None:
    with pytest.raises(ValidationError):
        Frame(image=make_image(), ts=TS, index=-1)
