"""App-level database auto-connection.

So the app never needs a manual "Connect" click: on first use in a session this
resolves the DB target from ``st.session_state`` (whose ``db_*`` keys default to
``.streamlit/secrets.toml`` ``[database]``) and initialises the schema, flipping
the ``db_ok`` / ``journal_db_ok`` gates the checklist and Trade Journal read.

Idempotent per session (guarded by a sentinel key) and never raises — a missing
or unreachable DB simply leaves the gates ``False`` with a message, exactly as a
failed manual connect would. Streamlit-aware, so it lives here rather than in the
Streamlit-free ``trade_repository``.
"""
from __future__ import annotations

import streamlit as st

from src.core import secrets
from src.db.trade_repository import DBConfig, TradeRepository

_DONE_KEY = "_db_auto_connect_done"


def current_db_config() -> DBConfig:
    """The active DB target: live ``session_state`` ``db_*`` values when present,
    otherwise the ``secrets.toml`` ``[database]`` defaults."""
    base = secrets.db_config()
    return DBConfig.from_mapping({
        "host": st.session_state.get("db_host", base["host"]),
        "port": st.session_state.get("db_port", base["port"]),
        "dbname": st.session_state.get("db_name", base["dbname"]),
        "user": st.session_state.get("db_user", base["user"]),
        "password": st.session_state.get("db_pass", base["password"]),
    })


def auto_connect(force: bool = False) -> bool:
    """Connect + init schema once per session from the resolved config.

    Sets ``db_ok`` / ``db_msg`` / ``journal_db_ok`` / ``journal_db_cfg`` in
    session state and returns whether the connection succeeded. A no-op returning
    the current state if already attempted this session, unless ``force=True``
    (used by the sidebar's manual reconnect after editing credentials).
    """
    if not force and st.session_state.get(_DONE_KEY):
        return bool(st.session_state.get("db_ok"))
    st.session_state[_DONE_KEY] = True

    cfg = current_db_config()
    if not cfg.password:
        st.session_state["db_ok"] = False
        st.session_state["db_msg"] = "No DB credentials in secrets.toml"
        st.session_state["journal_db_ok"] = False
        return False

    # Plain (lazy) connection — bad credentials surface as a message instead of
    # crashing on eager pool creation. init_schema already swallows its errors.
    try:
        ok, msg = TradeRepository(cfg).init_schema()
    except Exception as exc:  # pragma: no cover - defensive; init_schema guards
        ok, msg = False, str(exc)

    st.session_state["db_ok"] = ok
    st.session_state["db_msg"] = msg
    st.session_state["journal_db_ok"] = ok
    st.session_state["journal_db_cfg"] = cfg.as_kwargs()
    # Mirror the resolved target into the sidebar input keys so they display the
    # active connection (without clobbering anything the user already typed).
    st.session_state.setdefault("db_host", cfg.host)
    st.session_state.setdefault("db_port", cfg.port)
    st.session_state.setdefault("db_name", cfg.dbname)
    st.session_state.setdefault("db_user", cfg.user)
    st.session_state.setdefault("db_pass", cfg.password)

    if ok:
        # Stand up the market-data cache tables on the same target (non-fatal —
        # the read-through cache degrades to live yfinance if this fails).
        try:
            from src.db.market_data_repository import MarketDataRepository
            MarketDataRepository(cfg).init_schema()
        except Exception:
            pass
        # Re-read journal/stats against the freshly-connected DB.
        try:
            from src.db.cache import clear_read_caches
            clear_read_caches()
        except Exception:
            pass
    return ok
