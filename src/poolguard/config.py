"""Runtime configuration, loaded from environment / .env / YAML overlay later.

Validated at startup; the process refuses to start with missing or invalid
required settings rather than degrading silently. Settings are frozen after
construction (ADR-007: config is immutable after startup).

Nested sections are BaseSettings populated via default_factory so each reads
its own POOLGUARD_<SECTION>_* env vars at Settings() construction time —
nesting them as plain fields would bypass their env machinery entirely.
"""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from poolguard.events import OperatingMode


class CameraSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POOLGUARD_CAMERA_", frozen=True)

    rtsp_url: SecretStr = Field(
        description="RTSP stream URL, e.g. rtsp://user:pass@host:554/h264Preview_01_main"
    )
    target_fps: int = Field(default=15, ge=1, le=60)


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
    rules: RulesSettings = Field(default_factory=RulesSettings)
    alerts: AlertSettings = Field(default_factory=AlertSettings)
