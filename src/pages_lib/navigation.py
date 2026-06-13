"""Single source of truth for the page navigation.

Renumbered in Bloomberg-function-code style. Every page file uses this list to
render its sidebar and its function-code grid — no more hand-maintained copies
across 22 files.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import streamlit as st


@dataclass(frozen=True)
class NavEntry:
    code: str          # Bloomberg-style mnemonic (e.g. "CHCK", "RANK")
    label: str         # Human-friendly label
    icon: str          # Emoji prefix
    path: str          # Streamlit page path (top-level file or pages/foo.py)


# Ordered the way a professional day trader works a session:
# scan for candidates → filter the day → establish bias → confirm → find the
# zone → wait for the trigger → size the risk → final gate → review.
NAV_SECTIONS: List[tuple] = [
    ("1 · SCAN", [
        NavEntry("CHCK", "00. Daily Checklist",      "📋", "app.py"),
        NavEntry("OVRV", "01. Market Overview",      "📊", "pages/market-overview.py"),
        NavEntry("RANK", "02. Setup Ranker",         "🎰", "pages/setup-ranker.py"),
        NavEntry("AMD",  "03. AMD Scanner",          "📊", "pages/amd-scanner.py"),
        NavEntry("FCST", "04. Forecast Dashboard",   "📈", "pages/forecast-dashboard.py"),
    ]),
    ("2 · FILTER THE DAY", [
        NavEntry("DXAU", "05. DXY vs Gold",          "💵", "pages/dxy-gold.py"),
        NavEntry("MACR", "06. Macro Bias",           "🌐", "pages/macro-bias.py"),
        NavEntry("NEWS", "07. News Filter",          "📰", "pages/news-filter.py"),
        NavEntry("CORR", "08. Correlations",         "🔗", "pages/correlations.py"),
        NavEntry("ATR",  "09. ATR Volatility",       "📊", "pages/atr-volatility.py"),
    ]),
    ("3 · WEEKLY BIAS", [
        NavEntry("WEMA", "10. Weekly EMA",           "📉", "pages/weekly-ema.py"),
        NavEntry("WRSI", "11. Weekly RSI",           "📡", "pages/weekly-rsi.py"),
        NavEntry("WSWG", "12. Weekly Swing",         "🔄", "pages/weekly-swing.py"),
    ]),
    ("4 · DAILY CONFIRM", [
        NavEntry("DTRN", "13. Daily Trend",          "📈", "pages/daily-trend.py"),
        NavEntry("DMCD", "14. Daily MACD",           "📊", "pages/daily-macd.py"),
        NavEntry("STRC", "15. Market Structure",     "🏗️", "pages/market-structure.py"),
    ]),
    ("5 · 4H ZONE", [
        NavEntry("4HCZ", "16. 4H Confluence Zone",   "🎯", "pages/4H-confluence-zone.py"),
        NavEntry("CONF", "17. 2/3 Confluence Check", "🔀", "pages/confluence-checker.py"),
    ]),
    ("6 · 15M TRIGGER", [
        NavEntry("15RJ", "18. 15M Rejection",        "🕯️", "pages/15m-rejection.py"),
        NavEntry("15EN", "19. 15M Entry Signal",     "⚡", "pages/15m-entry-signal.py"),
    ]),
    ("7 · RISK & EXECUTE", [
        NavEntry("STOP", "20. Stop Structure",       "🛡️", "pages/stop-structure.py"),
        NavEntry("RRC",  "21. R:R Calculator",       "⚖️", "pages/rr-calculator.py"),
    ]),
    ("8 · REVIEW", [
        NavEntry("JRNL", "22. Trade Journal",        "📓", "pages/trade-journal.py"),
        NavEntry("BTST", "23. Forecast & Backtest",  "🧪", "pages/backtest-workflow.py"),
    ]),
    ("9 · SYSTEM", [
        NavEntry("RPRT", "24. Reports",              "📑", "pages/reports.py"),
        NavEntry("LOGS", "25. System Logs",          "🧾", "pages/system-logs.py"),
    ]),
]

# Flat list kept for any caller that doesn't care about grouping.
NAV_ENTRIES: List[NavEntry] = [e for _, entries in NAV_SECTIONS for e in entries]


def render_sidebar_nav() -> None:
    """Render the uniform sidebar navigation (call inside `with st.sidebar:`).

    Uses st.page_link so navigation is SPA-style and preserves session state.
    Every page should call this instead of hand-maintaining its own link list.
    """
    for section, entries in NAV_SECTIONS:
        st.caption(section)
        for e in entries:
            try:
                st.page_link(e.path, label=e.label, icon=e.icon)
            except Exception:
                # st.page_link needs the multipage registry (absent in bare/test
                # runs) and raises if a page file was renamed — skip the entry
                # rather than crash the whole page.
                continue
