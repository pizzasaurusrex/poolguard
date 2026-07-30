"""Regression tests for settings env loading.

The nested-BaseSettings composition previously dropped the documented
POOLGUARD_CAMERA_* env vars when constructing the top-level Settings.
"""

import pytest
from pydantic import ValidationError

from poolguard.config import Settings, TrackingSettings
from poolguard.events import OperatingMode

RTSP = "rtsp://user:pass@10.0.0.5:554/h265Preview_01_main"


@pytest.fixture
def camera_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POOLGUARD_CAMERA_RTSP_URL", RTSP)


def test_settings_loads_camera_from_documented_env_var(camera_env: None) -> None:
    settings = Settings()
    assert settings.camera.rtsp_url.get_secret_value() == RTSP


def test_nested_env_vars_read_at_construction_not_import(
    camera_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("POOLGUARD_RULES_SUBMERSION_TIMEOUT_SECONDS", "22.5")
    monkeypatch.setenv("POOLGUARD_ALERT_SIREN_GPIO_PIN", "17")
    settings = Settings()
    assert settings.rules.submersion_timeout_seconds == 22.5
    assert settings.alerts.siren_gpio_pin == 17


def test_missing_rtsp_url_refuses_to_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("POOLGUARD_CAMERA_RTSP_URL", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_rtsp_credentials_do_not_leak_in_repr(camera_env: None) -> None:
    settings = Settings()
    assert "pass" not in repr(settings)
    assert "pass" not in str(settings.camera)


class TestTrackingSettingsValidation:
    """Cross-field invariants the per-field ge/le checks can't express."""

    def test_low_confidence_above_high_confidence_rejected(self) -> None:
        with pytest.raises(ValidationError, match="low_confidence"):
            TrackingSettings(high_confidence=0.5, low_confidence=0.6)

    def test_equal_thresholds_allowed_to_disable_rescue(self) -> None:
        settings = TrackingSettings(high_confidence=0.5, low_confidence=0.5)
        assert settings.low_confidence == settings.high_confidence

    def test_pool_zone_origin_outside_frame_rejected(self) -> None:
        with pytest.raises(ValidationError, match="pool_zone"):
            TrackingSettings(pool_zone=(-0.1, 0.0, 1.0, 1.0))

    def test_pool_zone_zero_size_rejected(self) -> None:
        # A zero-width zone makes in_water always False — the silent
        # failure mode that would disable the submersion rule.
        with pytest.raises(ValidationError, match="pool_zone"):
            TrackingSettings(pool_zone=(0.2, 0.2, 0.0, 0.5))

    def test_pool_zone_oversized_component_rejected(self) -> None:
        with pytest.raises(ValidationError, match="pool_zone"):
            TrackingSettings(pool_zone=(0.0, 0.0, 1.5, 1.0))

    def test_default_pool_zone_still_valid(self) -> None:
        assert TrackingSettings().pool_zone == (0.0, 0.0, 1.0, 1.0)


def test_settings_are_frozen(camera_env: None) -> None:
    settings = Settings()
    with pytest.raises(ValidationError):
        settings.rules = settings.rules  # type: ignore[misc]
    with pytest.raises(ValidationError):
        settings.rules.default_mode = OperatingMode.SWIM  # type: ignore[misc]
