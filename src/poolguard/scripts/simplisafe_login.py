"""One-time SimpliSafe sign-in, to obtain the refresh token the live source uses.

Usage:
    poolguard-simplisafe-login

1. Open the printed URL in a browser with DevTools open on the Console tab,
   and sign in (including the SMS code).
2. The page appears to hang. The console shows a line starting
   "Failed to launch 'com.simplisafe.mobile://...". Copy that whole line.
3. Paste it at the prompt here. The code in it is single-use and expires
   within about a minute.

The refresh token is written to POOLGUARD_SIMPLISAFE_TOKEN_PATH, readable
only by you. Needed again if that file is lost, the token is revoked, or it
expires after a long unused gap (seen after 32 days, 2026-11-13).
"""

import asyncio
from getpass import getpass

from pydantic import SecretStr

from poolguard.config import SimpliSafeSettings
from poolguard.vision.simplisafe_auth import (
    RefreshTokenStore,
    SimpliSafeAuthError,
    extract_authorization_code,
)


async def _exchange(code: str, verifier: str) -> SecretStr:
    import aiohttp
    from simplipy import API

    async with aiohttp.ClientSession() as session:
        api = await API.async_from_auth(code, verifier, session=session)
    if api.refresh_token is None:
        raise SimpliSafeAuthError("SimpliSafe login returned no refresh token")
    return SecretStr(api.refresh_token)


def main() -> None:
    # Imported here so this module's helpers stay importable without the `simplisafe` extra.
    from simplipy.util.auth import (
        get_auth0_code_challenge,
        get_auth0_code_verifier,
        get_auth_url,
    )

    settings = SimpliSafeSettings()
    verifier = get_auth0_code_verifier()
    print("Open this URL, sign in, then copy the 'Failed to launch' console line:\n")
    print(get_auth_url(get_auth0_code_challenge(verifier)))
    # getpass: the code is a credential, so it is not echoed to the terminal.
    pasted = getpass("\nPaste the line (input is hidden) and press Enter: ")
    try:
        code = extract_authorization_code(pasted)
        token = asyncio.run(_exchange(code, verifier))
    except SimpliSafeAuthError as error:
        raise SystemExit(f"login failed: {error}") from None
    RefreshTokenStore(settings.token_path).save(token)
    print(f"Signed in. Refresh token saved to {settings.token_path}")


if __name__ == "__main__":
    main()
