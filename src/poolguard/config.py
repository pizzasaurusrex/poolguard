"""Runtime configuration, loaded from environment / .env / YAML overlay later.

Validated at startup; the process refuses to start with missing or invalid
required settings rather than degrading silently. Settings are frozen after
construction (ADR-007: config is immutable after startup).

Nested sections are BaseSettings populated via default_factory so each reads
its own POOLGUARD_<SECTION>_* env vars at Settings() construction time —
nesting them as plain fields would bypass their env machinery entirely.
"""

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from poolguard.events import OperatingMode


class CameraSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POOLGUARD_CAMERA_", frozen=True)

    rtsp_url: SecretStr = Field(
        description="RTSP stream URL, e.g. rtsp://user:pass@host:554/h264Preview_01_main"
    )
    target_fps: int = Field(default=15, ge=1, le=60)


class TrackingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POOLGUARD_TRACKING_", frozen=True)

    high_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Detections at or above this confidence match tracks and spawn new ones",
    )
    low_confidence: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Floor for the rescue pass; weaker detections are ignored entirely",
    )
    iou_min: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum box overlap for a detection to match an existing track",
    )
    max_coast_seconds: float = Field(
        default=30.0,
        gt=0,
        description="How long an unseen track survives before it is dropped",
    )
    pool_zone: tuple[float, float, float, float] = Field(
        default=(0.0, 0.0, 1.0, 1.0),
        description="Water region as normalized (x, y, width, height); default is whole frame",
    )

    @model_validator(mode="after")
    def _confidence_band_is_ordered(self) -> "TrackingSettings":
        # Equal values are allowed (an empty rescue band is a legitimate
        # way to disable the rescue pass).
        if self.low_confidence > self.high_confidence:
            raise ValueError(
                f"low_confidence ({self.low_confidence}) must be "
                f"<= high_confidence ({self.high_confidence})"
            )
        return self

    @model_validator(mode="after")
    def _pool_zone_is_normalized(self) -> "TrackingSettings":
        x, y, width, height = self.pool_zone
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"pool_zone origin must be within [0, 1]: {self.pool_zone}")
        if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            raise ValueError(
                f"pool_zone width/height must be in (0, 1]: {self.pool_zone}"
            )
        return self


class RulesSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POOLGUARD_RULES_", frozen=True)

    distress_sustained_seconds: float = Field(default=5.0, gt=0)
    submersion_timeout_seconds: float = Field(default=15.0, gt=0)
    default_mode: OperatingMode = OperatingMode.ARMED


class AlertSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POOLGUARD_ALERT_", frozen=True)

    siren_gpio_pin: int | None = Field(
        default=None, description="BCM pin driving the siren relay; None disables"
    )
    ntfy_topic_url: str | None = None
    mqtt_host: str | None = None
    escalation_ack_timeout_seconds: float = Field(default=30.0, gt=0)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", frozen=True)

    camera: CameraSettings = Field(default_factory=CameraSettings)
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)
    rules: RulesSettings = Field(default_factory=RulesSettings)
    alerts: AlertSettings = Field(default_factory=AlertSettings)
