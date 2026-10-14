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
from poolguard.geometry import WHOLE_FRAME, Polygon, validate_polygon

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
    containment_min: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum share of the smaller box inside the larger for a detection to "
        "keep a recently seen track's ID when IoU fails (head-only vs whole-body boxes)",
    )
    containment_window_seconds: float = Field(
        default=1.0,
        ge=0.0,
        description="Only tracks seen within this long are eligible for containment matching. "
        "Kept short on purpose: a newcomer's large box must not adopt, and so reset the "
        "unseen clock of, someone who has been under for longer",
    )
    handoff_window_seconds: float = Field(
        default=3.0,
        ge=0.0,
        description="A detection that would start a new track instead takes over a track lost "
        "within this long, if it is close enough (a swimmer jumping in and surfacing). The "
        "safety limit: someone unseen for longer keeps their clock, whoever appears nearby",
    )
    handoff_max_distance: float = Field(
        default=1.5,
        ge=0.0,
        description="Furthest a hand-off can reach, in lengths of the lost track's box (its "
        "longer side), so the limit shrinks for people far from the camera",
    )
    max_coast_seconds: float = Field(
        default=30.0,
        gt=0,
        description="How long an unseen track survives before it is dropped",
    )
    pool_zone: Polygon = Field(
        default=WHOLE_FRAME,
        description="Waterline as normalized (x, y) vertices; default is the whole frame. "
        "From the environment, a JSON list of pairs: [[0.2, 0.3], [0.9, 0.4], [0.5, 0.7]]",
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
    def _pool_zone_is_a_usable_polygon(self) -> "TrackingSettings":
        validate_polygon(self.pool_zone, name="pool_zone")
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


class ViewSettings(BaseSettings):
    """Thresholds for deciding whether the camera still sees what the pool zone was traced for."""

    model_config = SettingsConfigDict(env_prefix="POOLGUARD_VIEW_", **_SECTION_CONFIG)

    min_landmarks: int = Field(
        default=20,
        ge=4,
        description="Fewest matching landmarks that can vouch for the view; fewer reads as "
        "unknown, which is treated as untrusted",
    )
    min_landmark_spread: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Matching landmarks must span at least this share of the frame in both "
        "directions, so a burned-in logo or timestamp cannot vouch for the whole view",
    )
    max_zone_shift: float = Field(
        default=0.02,
        gt=0.0,
        description="Largest movement of any pool-zone vertex, as a fraction of frame width, "
        "that still counts as aligned",
    )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(**_SECTION_CONFIG)

    camera: CameraSettings = Field(default_factory=CameraSettings)
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)
    rules: RulesSettings = Field(default_factory=RulesSettings)
    alerts: AlertSettings = Field(default_factory=AlertSettings)
    view: ViewSettings = Field(default_factory=ViewSettings)
