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

ENV_FILE = ".env"

# Every section declares the env file itself: each is its own BaseSettings, so
# one declared only on the top-level Settings is never read by a section.
# extra="ignore" because a shared .env also holds the other sections' keys.
_SECTION_CONFIG = SettingsConfigDict(env_file=ENV_FILE, extra="ignore", frozen=True)


class CameraSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POOLGUARD_CAMERA_", **_SECTION_CONFIG)

    rtsp_url: SecretStr = Field(
        description="RTSP stream URL, e.g. rtsp://user:pass@host:554/h264Preview_01_main"
    )
    target_fps: int = Field(default=15, ge=1, le=60)


class TrackingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POOLGUARD_TRACKING_", **_SECTION_CONFIG)

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
            raise ValueError(f"pool_zone width/height must be in (0, 1]: {self.pool_zone}")
        return self


class RulesSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POOLGUARD_RULES_", **_SECTION_CONFIG)

    distress_sustained_seconds: float = Field(default=5.0, gt=0)
    submersion_timeout_seconds: float = Field(
        default=12.0,
        gt=0,
        description="Seconds a track may stay unseen inside the pool zone before submersion is "
        "raised. PROVISIONAL: lowered 15.0 -> 12.0 in POC to keep the rule reachable on short "
        "footage. No clip validates it yet — the one fully-submerged clip never spawns a track "
        "to lose (2% detection rate underwater). Re-derive from staged/P1 footage against the "
        "false-alarm budget — kids dive under routinely — not from clip lengths.",
    )
    default_mode: OperatingMode = OperatingMode.ARMED


class AlertSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POOLGUARD_ALERT_", **_SECTION_CONFIG)

    siren_gpio_pin: int | None = Field(
        default=None, description="BCM pin driving the siren relay; None disables"
    )
    ntfy_topic_url: SecretStr | None = Field(
        default=None,
        description="Full ntfy topic URL. The topic name is the only secret, so it is "
        "never logged; None disables the push channel",
    )
    heartbeat_push_url: SecretStr | None = Field(
        default=None,
        description="Uptime Kuma push-monitor URL, token included; None disables the heartbeat",
    )
    heartbeat_interval_seconds: float = Field(
        default=30.0,
        gt=0,
        description="Seconds between pings. Keep it under the monitor's own heartbeat "
        "interval so one late ping does not read as an outage",
    )
    mqtt_host: str | None = None
    escalation_ack_timeout_seconds: float = Field(default=30.0, gt=0)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(**_SECTION_CONFIG)

    camera: CameraSettings = Field(default_factory=CameraSettings)
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)
    rules: RulesSettings = Field(default_factory=RulesSettings)
    alerts: AlertSettings = Field(default_factory=AlertSettings)
