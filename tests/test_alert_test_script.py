"""The alert smoke test, run against a mock transport instead of real services."""

import httpx
import pytest
from pydantic import SecretStr

from poolguard.config import AlertSettings
from poolguard.scripts import alert_test
from poolguard.scripts.alert_test import Outcome, run_checks

NTFY_URL = "https://ntfy.example/secret-topic"
PUSH_URL = "http://kuma.example/api/push/secret-token"


def settings(ntfy: str | None = NTFY_URL, heartbeat: str | None = PUSH_URL) -> AlertSettings:
    return AlertSettings(
        _env_file=None,
        ntfy_topic_url=SecretStr(ntfy) if ntfy else None,
        heartbeat_push_url=SecretStr(heartbeat) if heartbeat else None,
    )


def client(status_by_host: dict[str, int]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_by_host[request.url.host])

    return httpx.Client(transport=httpx.MockTransport(handler))


def outcomes(results: tuple[alert_test.ChannelResult, ...]) -> dict[str, Outcome]:
    return {result.channel: result.outcome for result in results}


def test_sends_through_every_configured_channel() -> None:
    results = run_checks(settings(), client({"ntfy.example": 200, "kuma.example": 200}))
    assert outcomes(results) == {"ntfy": Outcome.SENT, "heartbeat": Outcome.SENT}


def test_unconfigured_channels_are_skipped_not_failed() -> None:
    results = run_checks(settings(ntfy=None, heartbeat=None), client({}))
    assert outcomes(results) == {"ntfy": Outcome.SKIPPED, "heartbeat": Outcome.SKIPPED}


def test_one_failing_channel_does_not_hide_the_other() -> None:
    results = run_checks(settings(), client({"ntfy.example": 500, "kuma.example": 200}))
    assert outcomes(results) == {"ntfy": Outcome.FAILED, "heartbeat": Outcome.SENT}
    failed = next(result for result in results if result.channel == "ntfy")
    assert "HTTP 500" in failed.detail
    assert "secret-topic" not in failed.detail


def test_main_exits_nonzero_when_a_channel_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(alert_test, "AlertSettings", lambda: settings())
    # Built before patching: the helper itself calls the real httpx.Client.
    fake_client = client({"ntfy.example": 200, "kuma.example": 503})
    monkeypatch.setattr(alert_test.httpx, "Client", lambda: fake_client)

    with pytest.raises(SystemExit) as raised:
        alert_test.main()

    assert raised.value.code == 1
    printed = capsys.readouterr().out
    assert "ntfy       sent" in printed
    assert "heartbeat  failed" in printed
    assert "secret-token" not in printed
