"""Daily Trading page composition — wires sidebars and controllers together."""
from __future__ import annotations

import streamlit as st

from src.pages_lib.base import BloombergPage, PageContext
from src.pages_lib.daily_trading.checklist import ChecklistController
from src.pages_lib.daily_trading.sidebar import ChecklistSidebar, TrendSidebar
from src.pages_lib.daily_trading.state import SessionStateBootstrap
from src.pages_lib.daily_trading.trend import TrendController


class DailyTradingPage(BloombergPage):
    """The top-level Bloomberg-style daily-trading page.

    Two operational modes are toggled by a sidebar radio:
      - CHECKLIST: 18-point pre-trade confluence flow.
      - TREND SIGNALS: multi-symbol trend scanner.
    """

    def configure(self) -> PageContext:
        SessionStateBootstrap.init()
        return PageContext(
            code="CHCK",
            title="Daily Trading System",
            icon="📈",
        )

    # ── sidebar ────────────────────────────────────────────────────────────
    def sidebar(self, ctx: PageContext) -> None:
        st.markdown(
            '<div style="color:#00ff41;font-weight:700;letter-spacing:0.18em;'
            'text-transform:uppercase;font-size:11px;margin:6px 0;">'
            'MODE</div>',
            unsafe_allow_html=True,
        )
        mode = st.radio(
            "Module", ["▸ CHECKLIST", "▸ TREND SIGNALS"],
            index=0 if st.session_state.current_page == "checklist" else 1,
            horizontal=True, label_visibility="collapsed",
        )
        st.session_state.current_page = (
            "checklist" if mode == "▸ CHECKLIST" else "trend"
        )
        st.markdown("---")
        if st.session_state.current_page == "checklist":
            ChecklistSidebar().render()
        else:
            TrendSidebar().render()

    # ── body ───────────────────────────────────────────────────────────────
    def body(self, ctx: PageContext) -> None:
        if st.session_state.current_page == "checklist":
            ChecklistController().render()
        else:
            TrendController().render()
