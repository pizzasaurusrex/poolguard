"""Frame model and the frame-source seam.

A FrameSource yields decoded frames regardless of where they come from —
video file (replay), RTSP stream (live), or a test fake. Downstream stages
depend on this protocol, never on OpenCV directly.
"""

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Protocol

import numpy as np
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

REPLAY_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)
"""Fixed start time for file replays so synthetic timestamps are reproducible."""


class Frame(BaseModel):
    """One decoded video frame with its capture (or synthetic) timestamp."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    image: np.ndarray
    """BGR pixel data, shape (height, width, 3). Validation marks the array
    read-only in place (`setflags(write=False)`), so the caller's reference
    is frozen too — pass a copy if you still need a writable array."""
    ts: AwareDatetime
    index: int = Field(ge=0)

    @field_validator("image")
    @classmethod
    def _freeze_image(cls, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected BGR image of shape (h, w, 3), got {image.shape}")
        image.setflags(write=False)
        return image

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]


class FrameSource(Protocol):
    """Anything that can produce an ordered stream of frames."""

    def frames(self) -> Iterator[Frame]: ...
