"""Session-state initialisation and accessors for the Daily Trading page."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import streamlit as st


CHECKS_TOTAL = 18
CRITICAL_CHECKS = (11, 12, 13, 14, 15, 16)


@dataclass
class StateDefaults:
    selected_instrument: str = "EUR/USD"
    trade_direction: str = "LONG"
    session: str = "London"
    notes: str = ""
    account_bal: float = 10000.0
    risk_pct: float = 1.0
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "trading"
    db_user: str = "postgres"
    db_pass: str = ""
    db_ok: bool = False
    db_msg: str = "Not connected"

    def as_dict(self) -> Dict[str, object]:
        return {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}


class SessionStateBootstrap:
    """Idempotent initialiser — call once at top of run()."""

    @staticmethod
    def init() -> None:
        # 18 checks
        for i in range(1, CHECKS_TOTAL + 1):
            st.session_state.setdefault(f"check_{i}", False)
        # trend-following toggles
        st.session_state.setdefault(
            "trend_selected_pairs", ["EUR/USD", "GBP/USD", "🥇 Gold"]
        )
        st.session_state.setdefault("trend_timeframe", "4 Hours")
        st.session_state.setdefault("trend_min_conds", 4)
        st.session_state.setdefault("current_page", "checklist")
        # generic defaults
        for k, v in StateDefaults().as_dict().items():
            st.session_state.setdefault(k, v)

    @staticmethod
    def reset_checks() -> None:
        for i in range(1, CHECKS_TOTAL + 1):
            st.session_state[f"check_{i}"] = False
