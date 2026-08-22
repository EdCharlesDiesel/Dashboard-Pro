"""Single source of truth for all sensitive configuration.

Every secret (API keys, SMTP credentials, database password) is read through
this module so there is exactly one place that knows the layout of
``.streamlit/secrets.toml``. Pages must never read ``st.secrets`` directly.

Resolution order for every value:

1. ``st.secrets`` (parsed from ``.streamlit/secrets.toml``)
2. environment variable (so the app runs in containers / CI without a file)
3. the supplied default

Backward-compatible reads are kept for the historical layout (FRED key under a
top-level name, news keys stashed under ``[database]`` with the ``FMP_API-KEY``
hyphen, the SMTP block under ``[email]``) so existing secrets files keep
working while new ones can use the canonical schema documented in
``.streamlit/secrets.toml.example``.

Canonical schema::

    [api]
    FRED_API_KEY    = "..."
    FMP_API_KEY     = "..."
    FINNHUB_API_KEY = "..."

    [database]
    host     = "localhost"
    port     = 5432
    dbname   = "trading"
    user     = "postgres"
    password = "..."
    # optional: url = "postgresql://user:pass@host:port/dbname" (overrides the above)

    [gmail]
    sender       = "you@gmail.com"
    app_password = "abcd efgh ijkl mnop"
    recipient    = "alerts@example.com"
"""
from __future__ import annotations
import time

import os
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlparse

import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# Low-level accessors
# ─────────────────────────────────────────────────────────────────────────────

def _section(name: str) -> Mapping[str, Any]:
    """Return a secrets section as a plain mapping, or ``{}`` if absent.

    Safe to call when no ``secrets.toml`` exists — ``st.secrets`` raises in that
    case and we fall back to an empty mapping (env vars still apply).
    """
    try:
        if name in st.secrets:
            return dict(st.secrets[name])
    except Exception:
        pass
    return {}


def _get(section: str, key: str, env: str, default: str = "") -> str:
    """Read ``[section] key`` from secrets, then ``$env``, then ``default``."""
    sect = _section(section)
    if key in sect and str(sect[key]) != "":
        return str(sect[key])
    return os.environ.get(env, default)


# ─────────────────────────────────────────────────────────────────────────────
# API keys
# ─────────────────────────────────────────────────────────────────────────────

def fred_api_key() -> str:
    """FRED (St. Louis Fed) API key. Canonical: ``[api] FRED_API_KEY``."""
    val = _get("api", "FRED_API_KEY", "FRED_API_KEY")
    if val:
        return val
    # Legacy: key sometimes lived at the top level of secrets.toml.
    try:
        return str(st.secrets.get("FRED_API_KEY", "")) or ""
    except Exception:
        return ""


def fmp_api_key() -> str:
    """Financial Modeling Prep API key. Canonical: ``[api] FMP_API_KEY``."""
    val = _get("api", "FMP_API_KEY", "FMP_API_KEY")
    if val:
        return val
    # Legacy: stored under [database] with a hyphen in the key name.
    db = _section("database")
    return str(db.get("FMP_API-KEY", db.get("FMP_API_KEY", "")))


def finnhub_api_key() -> str:
    """Finnhub API key. Canonical: ``[api] FINNHUB_API_KEY``."""
    val = _get("api", "FINNHUB_API_KEY", "FINNHUB_API_KEY")
    if val:
        return val
    # Legacy: stored under [database].
    return str(_section("database").get("FINNHUB_API_KEY", ""))


def anthropic_api_key() -> str:
    """Anthropic (Claude) API key for AI report summaries.

    Canonical: ``[api] ANTHROPIC_API_KEY``; env ``ANTHROPIC_API_KEY``. Optional —
    the reports page falls back to rule-based analytics when this is absent.
    """
    return _get("api", "ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY")


# ─────────────────────────────────────────────────────────────────────────────
# Email / Gmail SMTP (AMD scanner + market-overview alerts)
# ─────────────────────────────────────────────────────────────────────────────

def gmail_config() -> Dict[str, str]:
    """Gmail SMTP credentials. Canonical: ``[gmail] sender/app_password/recipient``.

    Falls back to a legacy ``[email]`` block (``smtp_user``/``smtp_password``).
    """
    g = _section("gmail")
    e = _section("email")
    return {
        "sender": str(g.get("sender") or e.get("sender") or e.get("smtp_user")
                      or os.environ.get("GMAIL_SENDER", "")),
        "app_password": str(g.get("app_password") or e.get("smtp_password")
                            or os.environ.get("GMAIL_APP_PASSWORD", "")),
        "recipient": str(g.get("recipient") or e.get("recipient")
                         or os.environ.get("GMAIL_RECIPIENT", "")),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Telegram (Evening Sentry / Surprise Awareness alerts)
# ─────────────────────────────────────────────────────────────────────────────

def telegram_config() -> Dict[str, str]:
    """Telegram bot credentials for in-app alert sends (pages/surprise_tab.py).

    Canonical: ``[telegram] bot_token`` / ``chat_id``. Env fallback:
    ``TELEGRAM_BOT_TOKEN`` / ``TELEGRAM_CHAT_ID`` — the same two variables
    src/services/evening_sentry.py's standalone CLI reads directly from
    ``os.environ`` (it has no Streamlit runtime, so it can't use ``st.secrets``);
    set both env vars to the same values as this ``[telegram]`` block so the
    dashboard and the background sentry alert through the same bot.
    """
    return {
        "bot_token": _get("telegram", "bot_token", "TELEGRAM_BOT_TOKEN"),
        "chat_id": _get("telegram", "chat_id", "TELEGRAM_CHAT_ID"),
    }


TELEGRAM_ATTEMPTS = 3          # bounded: an alert must not stall a scan cycle
TELEGRAM_BACKOFF_S = 1.0


def redact(text: str) -> str:
    """Replace the configured bot token with ``<REDACTED>``.

    Telegram puts the token in the URL and ``requests`` puts the URL in its
    error string, so any naive ``str(exc)`` prints the credential. That
    happened twice on 2026-08-20 - once to a screen, once through the detail
    string ``background_scanner`` writes to its log on a failed alert.

    Exported so other modules can scrub their own logging.
    """
    token = (telegram_config() or {}).get("bot_token")
    return str(text).replace(token, "<REDACTED>") if token else str(text)


def send_telegram_message(text: str,
                          parse_mode: str | None = None) -> tuple[bool, str]:
    """Send ``text`` via the configured bot. Returns ``(ok, detail)``.

    Never raises, and never returns the token. On an HTTP error it reports
    Telegram's own ``description`` ("chat not found"), which is what an
    operator needs; the URL is noise that happens to be a secret.

    Connection and timeout failures are retried, HTTP status errors are not:
    a 400 will never succeed on attempt two, and retrying it only delays the
    scan cycle. The retry exists because a momentary DNS failure in the
    scanner container destroyed an entry alert outright - and a confluence
    signal is rare enough that losing one to a two-second fault matters.

    ``parse_mode`` is omitted unless given: the confluence body contains
    ``ENTRY_FIRED``, whose lone underscore makes Telegram reject the whole
    message under Markdown.
    """
    cfg = telegram_config()
    if not cfg["bot_token"] or not cfg["chat_id"]:
        return False, "Telegram not configured - add [telegram] bot_token/chat_id to secrets.toml"

    import requests

    payload = {"chat_id": cfg["chat_id"], "text": text}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    url = f"https://api.telegram.org/bot{cfg['bot_token']}/sendMessage"

    last = "no attempt made"
    for attempt in range(TELEGRAM_ATTEMPTS):
        try:
            resp = requests.post(url, json=payload, timeout=10)
        except (requests.ConnectionError, requests.Timeout) as exc:
            last = redact(str(exc))          # transient - worth another go
            if attempt < TELEGRAM_ATTEMPTS - 1:
                time.sleep(TELEGRAM_BACKOFF_S)
            continue
        except Exception as exc:             # noqa: BLE001 - never raise at a caller
            return False, redact(str(exc))

        if 200 <= getattr(resp, "status_code", 0) < 300:
            return True, "sent"

        # A status error is final. Prefer Telegram's description.
        try:
            detail = str((resp.json() or {}).get("description") or "")
        except Exception:                    # noqa: BLE001 - body may not be JSON
            detail = ""
        return False, redact(detail or f"HTTP {resp.status_code}")

    return False, last


def email_config() -> Dict[str, Any]:
    """SMTP settings in the shape ``market-overview.py`` expects.

    Built on top of :func:`gmail_config` so there is one credential source. Host
    and port default to Gmail and may be overridden by a legacy ``[email]``
    block or ``EMAIL_SMTP_HOST`` / ``EMAIL_SMTP_PORT`` env vars.
    """
    g = gmail_config()
    e = _section("email")
    # Auto-detect the SMTP host from the sender's domain when no explicit
    # override is given — a Hotmail/Outlook sender can't authenticate against
    # smtp.gmail.com. An [email] smtp_host / EMAIL_SMTP_HOST override always
    # wins; Gmail remains the default for everything unrecognized.
    sender_domain = g["sender"].rsplit("@", 1)[-1].lower() if "@" in g["sender"] else ""
    if sender_domain in ("hotmail.com", "outlook.com", "live.com", "msn.com"):
        auto_host = "smtp-mail.outlook.com"
    elif sender_domain in ("yahoo.com",):
        auto_host = "smtp.mail.yahoo.com"
    else:
        auto_host = "smtp.gmail.com"
    host = str(e.get("smtp_host") or os.environ.get("EMAIL_SMTP_HOST") or auto_host)
    port = int(e.get("smtp_port") or os.environ.get("EMAIL_SMTP_PORT", "587"))
    return {
        "smtp_host": host,
        "smtp_port": port,
        "user": g["sender"],
        "password": g["app_password"],
        "sender": g["sender"],
        "recipient": g["recipient"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────────────────────────────────────

_DB_DEFAULTS = {
    "host": "localhost",
    "port": 5432,
    "dbname": "trading",
    "user": "postgres",
    "password": "",
}


def db_config() -> Dict[str, Any]:
    """PostgreSQL connection settings.

    Reads ``[database]``. A ``url`` (``postgresql://user:pass@host:port/dbname``)
    takes precedence over individual keys. Individual keys fall back to
    ``DB_HOST`` / ``DB_PORT`` / ``DB_NAME`` / ``DB_USER`` / ``DB_PASSWORD`` env
    vars and finally to localhost defaults. ``name`` is accepted as an alias for
    ``dbname``.
    """
    db = _section("database")
    cfg: Dict[str, Any] = dict(_DB_DEFAULTS)

    url = str(db.get("url", "") or os.environ.get("DATABASE_URL", ""))
    if url:
        parsed = urlparse(url)
        if parsed.hostname:
            cfg["host"] = parsed.hostname
        if parsed.port:
            cfg["port"] = parsed.port
        if parsed.username:
            cfg["user"] = parsed.username
        if parsed.password:
            cfg["password"] = parsed.password
        path = parsed.path.lstrip("/")
        if path:
            cfg["dbname"] = path

    # Individual keys override / supplement the URL.
    cfg["host"] = str(db.get("host") or os.environ.get("DB_HOST", "") or cfg["host"])
    cfg["port"] = int(db.get("port") or os.environ.get("DB_PORT", "") or cfg["port"])
    cfg["dbname"] = str(db.get("dbname") or db.get("name")
                        or os.environ.get("DB_NAME", "") or cfg["dbname"])
    cfg["user"] = str(db.get("user") or os.environ.get("DB_USER", "") or cfg["user"])
    pw = db.get("password")
    if pw in (None, ""):
        pw = os.environ.get("DB_PASSWORD", "") or cfg["password"]
    cfg["password"] = str(pw)
    return cfg
