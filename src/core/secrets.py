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


def send_telegram_message(text: str) -> tuple[bool, str]:
    """Send ``text`` via the configured Telegram bot. Returns ``(ok, detail)``.

    Never raises — a missing/invalid config or a network failure is reported
    in the returned message, not thrown, so callers can surface it with
    ``st.error``/``st.caption`` without their own try/except.
    """
    cfg = telegram_config()
    if not cfg["bot_token"] or not cfg["chat_id"]:
        return False, "Telegram not configured — add [telegram] bot_token/chat_id to secrets.toml"
    import requests
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{cfg['bot_token']}/sendMessage",
            json={"chat_id": cfg["chat_id"], "text": text}, timeout=10,
        )
        resp.raise_for_status()
        return True, "sent"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def email_config() -> Dict[str, Any]:
    """SMTP settings in the shape ``market-overview.py`` expects.

    Built on top of :func:`gmail_config` so there is one credential source. Host
    and port default to Gmail and may be overridden by a legacy ``[email]``
    block or ``EMAIL_SMTP_HOST`` / ``EMAIL_SMTP_PORT`` env vars.
    """
    g = gmail_config()
    e = _section("email")
    host = str(e.get("smtp_host") or os.environ.get("EMAIL_SMTP_HOST", "smtp.gmail.com"))
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
