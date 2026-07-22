"""The email audit log: every actual send attempt (success or SMTP failure)
writes one row to tool_usage_log. SMTP and the DB are both mocked, so no
network and no live Postgres are touched."""
import smtplib

import pytest

from src.services import alert_service


@pytest.fixture
def _configured_email(monkeypatch):
    """Pretend Gmail is configured so send_email reaches the SMTP path."""
    monkeypatch.setattr(alert_service.secrets, "email_config", lambda: {
        "smtp_host": "smtp.gmail.com", "smtp_port": 587,
        "user": "me@gmail.com", "password": "app-pw",
        "sender": "me@gmail.com", "recipient": "me@gmail.com",
    })


@pytest.fixture
def _capture_log(monkeypatch):
    rows = []
    monkeypatch.setattr("src.services.tool_log.log_tool_usage",
                        lambda tool, payload: rows.append((tool, payload)) or True)
    return rows


class _FakeSMTP:
    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def ehlo(self): pass
    def starttls(self, context=None): pass
    def login(self, u, p): pass
    def send_message(self, msg): pass


class _AuthFailSMTP(_FakeSMTP):
    def login(self, u, p):
        raise smtplib.SMTPAuthenticationError(535, b"bad creds")


def test_successful_send_logs_ok_row(monkeypatch, _configured_email, _capture_log):
    monkeypatch.setattr(alert_service.smtplib, "SMTP", _FakeSMTP)
    ok, msg = alert_service.send_email("🎰 2 new setups — XAU/USD", "<p>x</p>",
                                       "x", source="setup_ranker")
    assert ok is True
    assert len(_capture_log) == 1
    tool, payload = _capture_log[0]
    assert tool == "email_alert"
    assert payload["source"] == "setup_ranker"
    assert payload["ok"] is True
    assert payload["error"] is None
    assert payload["recipient"] == "me@gmail.com"
    assert "XAU/USD" in payload["subject"]


def test_auth_failure_logs_failed_row_with_error(monkeypatch, _configured_email, _capture_log):
    monkeypatch.setattr(alert_service.smtplib, "SMTP", _AuthFailSMTP)
    ok, msg = alert_service.send_email("s", "<p>x</p>", "x", source="test")
    assert ok is False
    assert len(_capture_log) == 1
    _, payload = _capture_log[0]
    assert payload["ok"] is False
    assert payload["source"] == "test"
    assert payload["error"]  # a non-empty error string


def test_not_configured_is_not_logged(monkeypatch, _capture_log):
    monkeypatch.setattr(alert_service.secrets, "email_config",
                        lambda: {"user": "", "password": "", "recipient": ""})
    ok, msg = alert_service.send_email("s", "<p>x</p>", "x")
    assert ok is False
    assert _capture_log == []  # no send attempted → no audit row


def test_logging_failure_never_breaks_send(monkeypatch, _configured_email):
    monkeypatch.setattr(alert_service.smtplib, "SMTP", _FakeSMTP)
    monkeypatch.setattr("src.services.tool_log.log_tool_usage",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db down")))
    ok, msg = alert_service.send_email("s", "<p>x</p>", "x")
    assert ok is True  # the send still succeeds despite the logging error
