"""Runtime configuration, loaded from environment / .env / YAML overlay later.

Validated at startup; the process refuses to start with missing or invalid
required settings rather than degrading silently.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from poolguard.events import OperatingMode


class CameraSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POOLGUARD_CAMERA_")

    rtsp_url: str = Field(description="RTSP stream URL, e.g. rtsp://user:pass@host:554/h264Preview_01_main")
    target_fps: int = Field(default=15, ge=1, le=60)


class RulesSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POOLGUARD_RULES_")

    distress_sustained_seconds: float = Field(default=5.0, gt=0)
    submersion_timeout_seconds: float = Field(default=15.0, gt=0)
    default_mode: OperatingMode = OperatingMode.ARMED


class AlertSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POOLGUARD_ALERT_")

    siren_gpio_pin: int | None = Field(
        default=None, description="BCM pin driving the siren relay; None disables"
    )
    ntfy_topic_url: str | None = None
    mqtt_host: str | None = None
    escalation_ack_timeout_seconds: float = Field(default=30.0, gt=0)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__")

    camera: CameraSettings
    rules: RulesSettings = RulesSettings()
    alerts: AlertSettings = AlertSettings()
