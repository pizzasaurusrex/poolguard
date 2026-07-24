"""Core event and detection models shared across the pipeline.

Every stage (detect → track → rules → alert) communicates via these immutable
Pydantic models; no stage mutates another stage's output.
"""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class OperatingMode(StrEnum):
    ARMED = "armed"
    SWIM = "swim"
    MAINTENANCE = "maintenance"


class AlertTier(StrEnum):
    WATCH = "watch"
    WARN = "warn"
    EMERGENCY = "emergency"


class EventType(StrEnum):
    DISTRESS = "distress"
    SUBMERSION = "submersion"
    UNSUPERVISED_ENTRY = "unsupervised_entry"
    SYSTEM_DEGRADED = "system_degraded"


class BoundingBox(BaseModel):
    """Normalized [0, 1] coordinates, origin top-left."""

    model_config = ConfigDict(frozen=True)

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)


class Detection(BaseModel):
    """A single person detection in one frame."""

    model_config = ConfigDict(frozen=True)

    frame_ts: datetime
    box: BoundingBox
    confidence: float = Field(ge=0.0, le=1.0)
    keypoints: tuple[tuple[float, float, float], ...] | None = None
    """Optional pose keypoints as (x, y, confidence) triples, normalized."""


class TrackedPerson(BaseModel):
    """A detection associated with a persistent track ID."""

    model_config = ConfigDict(frozen=True)

    track_id: int
    detection: Detection
    in_water: bool
    seconds_since_last_seen: float = Field(ge=0.0, default=0.0)


class SafetyEvent(BaseModel):
    """A rules-engine conclusion that the alert manager acts on."""

    model_config = ConfigDict(frozen=True)

    event_type: EventType
    tier: AlertTier
    started_at: datetime
    track_id: int | None = None
    mode: OperatingMode
    confidence: float = Field(ge=0.0, le=1.0)
    detail: str
    snapshot_path: str | None = None
