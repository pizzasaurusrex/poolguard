"""SimpliSafe token storage and response parsing. No SDK needed."""

import json
import stat
from pathlib import Path

import pytest
from pydantic import SecretStr

from poolguard.vision.simplisafe_auth import (
    RefreshTokenStore,
    SimpliSafeAuthError,
    extract_authorization_code,
    parse_live_view,
)

REDIRECT = (
    "com.simplisafe.mobile://auth.simplisafe.com/ios/com.simplisafe.mobile/callback"
    "?code=abc_DEF-123&state=xyz"
)


class TestRefreshTokenStore:
    def test_round_trips_a_token(self, tmp_path: Path) -> None:
        store = RefreshTokenStore(tmp_path / "token.json")
        store.save(SecretStr("refresh-1"))
        assert store.load().get_secret_value() == "refresh-1"

    def test_file_is_private_to_the_owner(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "token.json"
        RefreshTokenStore(path).save(SecretStr("refresh-1"))
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_saving_replaces_the_rotated_token(self, tmp_path: Path) -> None:
        store = RefreshTokenStore(tmp_path / "token.json")
        store.save(SecretStr("refresh-1"))
        store.save(SecretStr("refresh-2"))
        assert store.load().get_secret_value() == "refresh-2"
        assert list(tmp_path.iterdir()) == [tmp_path / "token.json"]

    def test_missing_file_says_how_to_fix_it(self, tmp_path: Path) -> None:
        with pytest.raises(SimpliSafeAuthError, match="poolguard-simplisafe-login"):
            RefreshTokenStore(tmp_path / "absent.json").load()

    def test_corrupt_file_is_rejected_without_echoing_its_content(self, tmp_path: Path) -> None:
        path = tmp_path / "token.json"
        path.write_text('{"refresh_token": 42, "leak": "secret-value"}')
        with pytest.raises(SimpliSafeAuthError) as raised:
            RefreshTokenStore(path).load()
        assert "secret-value" not in str(raised.value)

    def test_token_is_not_shown_in_repr(self, tmp_path: Path) -> None:
        store = RefreshTokenStore(tmp_path / "token.json")
        store.save(SecretStr("refresh-1"))
        assert "refresh-1" not in repr(store.load())


class TestParseLiveView:
    def test_reads_the_livekit_url_and_token(self) -> None:
        payload = json.loads(
            '{"liveKitDetails": {"liveKitURL": "wss://stream.example:7880", '
            '"userToken": "jwt-value"}, "cameraStatus": "online"}'
        )
        credentials = parse_live_view(payload)
        assert credentials.url == "wss://stream.example:7880"
        assert credentials.token.get_secret_value() == "jwt-value"
        assert "jwt-value" not in repr(credentials)

    def test_a_changed_response_shape_is_reported_by_field_name_only(self) -> None:
        with pytest.raises(SimpliSafeAuthError, match="liveKitDetails") as raised:
            parse_live_view({"streamDetails": {"userToken": "jwt-value"}})
        assert "jwt-value" not in str(raised.value)


class TestExtractAuthorizationCode:
    def test_pulls_the_code_from_the_redirect_url(self) -> None:
        assert extract_authorization_code(REDIRECT) == "abc_DEF-123"

    def test_pulls_the_code_from_the_whole_console_line(self) -> None:
        line = f"Failed to launch '{REDIRECT}' because the scheme does not have a handler."
        assert extract_authorization_code(line) == "abc_DEF-123"

    def test_accepts_a_bare_code(self) -> None:
        assert extract_authorization_code("  abc_DEF-123\n") == "abc_DEF-123"

    def test_rejects_text_with_no_code(self) -> None:
        with pytest.raises(SimpliSafeAuthError):
            extract_authorization_code("https://auth.simplisafe.com/u/login?state=xyz")
