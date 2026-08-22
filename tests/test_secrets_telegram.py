"""The Telegram sender must never expose the bot token.

Telegram puts the token in the URL and `requests` puts the URL in its error
string, so every naive failure path prints the credential. It leaked twice on
2026-08-20: once to the screen through a traceback, and once through the
`detail` string that `background_scanner` writes to its log on a failed alert.

No test here uses a real token or sends a real message.
"""
from __future__ import annotations

import pytest
import requests

from src.core import secrets

FAKE = "123456:FAKE-TOKEN-abcdef"


class _Resp:
    """Minimal stand-in for a requests.Response."""

    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload if payload is not None else {"ok": True}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if not 200 <= self.status_code < 300:
            raise requests.HTTPError(
                f"{self.status_code} Client Error: Bad Request for url: "
                f"https://api.telegram.org/bot{FAKE}/sendMessage")


@pytest.fixture(autouse=True)
def _configured(monkeypatch):
    monkeypatch.setattr(secrets, "telegram_config",
                        lambda: {"bot_token": FAKE, "chat_id": "1"})
    # Never pace a unit test on a real backoff.
    monkeypatch.setattr(secrets.time, "sleep", lambda s: None, raising=False)


class TestNoTokenEverEscapes:
    def test_a_network_error_detail_has_no_token(self, monkeypatch):
        def boom(*a, **k):
            raise requests.ConnectionError(
                f"Max retries exceeded with url: /bot{FAKE}/sendMessage")

        monkeypatch.setattr(requests, "post", boom)
        ok, detail = secrets.send_telegram_message("hi")
        assert ok is False
        assert FAKE not in detail
        assert "REDACTED" in detail

    def test_an_http_error_detail_has_no_token(self, monkeypatch):
        monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(
            400, {"ok": False, "description": "Bad Request: chat not found"}))
        ok, detail = secrets.send_telegram_message("hi")
        assert ok is False
        assert FAKE not in detail

    def test_it_reports_telegrams_own_description(self, monkeypatch):
        # "chat not found" is what the operator needs; the URL is noise.
        monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(
            400, {"ok": False, "description": "Bad Request: chat not found"}))
        _, detail = secrets.send_telegram_message("hi")
        assert "chat not found" in detail

    def test_redact_is_a_no_op_without_a_token(self, monkeypatch):
        monkeypatch.setattr(secrets, "telegram_config",
                            lambda: {"bot_token": "", "chat_id": ""})
        assert secrets.redact("nothing to hide") == "nothing to hide"

    def test_redact_removes_every_occurrence(self):
        assert FAKE not in secrets.redact(f"{FAKE} and again {FAKE}")

    def test_success_still_returns_true_and_sent(self, monkeypatch):
        monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(200))
        assert secrets.send_telegram_message("hi") == (True, "sent")

    def test_parse_mode_is_absent_by_default(self, monkeypatch):
        # The confluence body contains ENTRY_FIRED, whose lone underscore makes
        # Telegram reject the whole message under Markdown.
        seen = {}
        monkeypatch.setattr(requests, "post",
                            lambda url, json=None, timeout=None: seen.update(json or {}) or _Resp(200))
        secrets.send_telegram_message("hi")
        assert "parse_mode" not in seen

    def test_parse_mode_is_passed_through_when_given(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(requests, "post",
                            lambda url, json=None, timeout=None: seen.update(json or {}) or _Resp(200))
        secrets.send_telegram_message("*hi*", parse_mode="Markdown")
        assert seen["parse_mode"] == "Markdown"


class TestTransientRetry:
    """A momentary fault must not destroy a rare entry alert.

    On 2026-08-20 the scanner container briefly failed DNS and the send was
    simply lost; DNS was healthy again minutes later.
    """

    def test_a_connection_error_is_retried_and_can_succeed(self, monkeypatch):
        calls = []

        def flaky(*a, **k):
            calls.append(1)
            if len(calls) < 3:
                raise requests.ConnectionError("temporary failure in name resolution")
            return _Resp(200)

        monkeypatch.setattr(requests, "post", flaky)
        ok, _ = secrets.send_telegram_message("hi")
        assert ok is True and len(calls) == 3

    def test_it_gives_up_after_the_configured_attempts(self, monkeypatch):
        calls = []

        def always(*a, **k):
            calls.append(1)
            raise requests.ConnectionError(f"boom /bot{FAKE}/sendMessage")

        monkeypatch.setattr(requests, "post", always)
        ok, detail = secrets.send_telegram_message("hi")
        assert len(calls) == secrets.TELEGRAM_ATTEMPTS
        assert ok is False and FAKE not in detail

    def test_an_http_400_is_not_retried(self, monkeypatch):
        # It will never succeed on attempt two; retrying only delays the
        # scanner and hammers the API.
        calls = []
        monkeypatch.setattr(requests, "post", lambda *a, **k: calls.append(1) or _Resp(
            400, {"ok": False, "description": "Bad Request: chat not found"}))
        secrets.send_telegram_message("hi")
        assert len(calls) == 1

    def test_the_backoff_stays_inside_its_budget(self, monkeypatch):
        slept = []
        monkeypatch.setattr(secrets.time, "sleep", lambda s: slept.append(s))
        monkeypatch.setattr(requests, "post",
                            lambda *a, **k: (_ for _ in ()).throw(requests.Timeout("slow")))
        secrets.send_telegram_message("hi")
        assert sum(slept) <= 3.0
