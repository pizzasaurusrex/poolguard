"""SimpliSafe credentials: the stored refresh token and the live-view response.

Kept apart from `simplisafe.py` so none of this needs the `simplisafe` extra.

SimpliSafe's login (Auth0) rotates the refresh token on every use. The new
one has to be written back each time or the next start cannot log in, so the
store replaces the file atomically. Nothing here puts a token in an error
message or a repr.
"""

import json
import os
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

TOKEN_FILE_MODE = 0o600
LOGIN_COMMAND = "poolguard-simplisafe-login"

_CODE_IN_URL = re.compile(r"[?&]code=([A-Za-z0-9_\-]+)")
_BARE_CODE = re.compile(r"[A-Za-z0-9_\-]+")


class SimpliSafeAuthError(RuntimeError):
    """Credentials are missing, unreadable, or not in the expected shape."""


class _StoredToken(BaseModel):
    refresh_token: SecretStr


class RefreshTokenStore:
    """The refresh token on disk, readable only by its owner."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> SecretStr:
        try:
            stored = _StoredToken.model_validate_json(self._path.read_text())
        except FileNotFoundError:
            raise SimpliSafeAuthError(
                f"no SimpliSafe token at {self._path}; run {LOGIN_COMMAND}"
            ) from None
        except ValidationError:
            # `from None`: pydantic's message quotes the file's content.
            raise SimpliSafeAuthError(
                f"SimpliSafe token file {self._path} is not valid; run {LOGIN_COMMAND}"
            ) from None
        return stored.refresh_token

    def save(self, token: SecretStr) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_name(self._path.name + ".tmp")
        payload = json.dumps({"refresh_token": token.get_secret_value()})
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, TOKEN_FILE_MODE)
        with os.fdopen(descriptor, "w") as handle:
            handle.write(payload)
        os.chmod(temporary, TOKEN_FILE_MODE)
        os.replace(temporary, self._path)


class LiveKitCredentials(BaseModel):
    """What it takes to join the camera's LiveKit room."""

    model_config = ConfigDict(frozen=True)

    url: str
    token: SecretStr


class _LiveKitDetails(BaseModel):
    url: str = Field(alias="liveKitURL")
    token: SecretStr = Field(alias="userToken")


class _LiveViewResponse(BaseModel):
    details: _LiveKitDetails = Field(alias="liveKitDetails")


def parse_live_view(payload: object) -> LiveKitCredentials:
    """Validate SimpliSafe's live-view response, which is a private API and may change."""
    try:
        response = _LiveViewResponse.model_validate(payload)
    except ValidationError as error:
        fields = ", ".join(".".join(str(part) for part in issue["loc"]) for issue in error.errors())
        raise SimpliSafeAuthError(
            f"live-view response is not in the expected shape (problem fields: {fields})"
        ) from None
    return LiveKitCredentials(url=response.details.url, token=response.details.token)


def extract_authorization_code(text: str) -> str:
    """Find the one-time login code in a redirect URL, a console line, or on its own."""
    match = _CODE_IN_URL.search(text)
    if match is not None:
        return match.group(1)
    stripped = text.strip()
    if _BARE_CODE.fullmatch(stripped) is not None:
        return stripped
    raise SimpliSafeAuthError("no authorization code found in the pasted text")
