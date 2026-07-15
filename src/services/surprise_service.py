"""
src/services/surprise_service.py
=================================
Adapter around pages/surprise_tab.py's ``event_gate`` for headless callers —
today, the Swing Playbook. That function's real signature needs a SQLAlchemy
connection and one of surprise_tab's compact symbol keys ("EURUSD", not
"EUR/USD"); this module resolves the connection and the symbol translation so
callers can just pass a registry instrument name.

pages/surprise_tab.py owns the actual logic (sentry_news schema, the
event_gate query) and is guarded behind ``if __name__ == "__main__":`` (same
contract as pages/cot_signals.py / pages/disconnect_monitor_tab.py) so
importing its pure functions here never re-runs its page UI or collides on a
second set_page_config().

Same resilience contract as the rest of src/services/: never raises. A
down/unconfigured DB or a missing sentry_news table degrades to "not
blocked" rather than crashing the whole playbook — a gate that can't be
checked should not silently suppress every signal either, so it fails open.
"""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st

# Registry pair ("EUR/USD") -> surprise_tab's compact symbol key ("EURUSD"),
# mirroring pages/surprise_tab.py's own _SYMBOL_OVERRIDES.
_SYMBOL_OVERRIDES = {"WTI/USD": "WTI"}

_CLEAR: Dict[str, Any] = {
    "blocked": False, "events": [], "reason": "gate unavailable — treating as clear",
}


def _symbol_for(instrument: str) -> str:
    return _SYMBOL_OVERRIDES.get(instrument, instrument.replace("/", ""))


@st.cache_resource(show_spinner=False)
def _store():
    """Same Postgres target and Store as pages/surprise_tab.py's
    _sentry_store() — creates sentry_news/sentry_journal/etc. if missing.
    Cached so repeated gate checks (one per playbook row) reuse one
    connection pool instead of opening a fresh one per instrument."""
    from src.core.secrets import db_config
    from src.services.evening_sentry import Store
    cfg = db_config()
    url = (f"postgresql+psycopg2://{cfg['user']}:{cfg['password']}"
          f"@{cfg['host']}:{cfg['port']}/{cfg['dbname']}")
    return Store(url=url)


def event_gate(instrument: str, hours_before: float = 4.0,
               hours_after: float = 2.0) -> Dict[str, Any]:
    """``{'blocked': bool, 'events': [...], 'reason': str}`` for ``instrument``
    (a registry pair like "EUR/USD" or "XAU/USD"). Never raises."""
    try:
        from pages.surprise_tab import event_gate as _event_gate
        with _store().engine.connect() as conn:
            return _event_gate(conn, _symbol_for(instrument),
                               hours_before=hours_before, hours_after=hours_after)
    except Exception:
        return dict(_CLEAR)
