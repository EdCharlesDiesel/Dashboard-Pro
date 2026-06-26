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


# Ordered the way a professional day trader works a session, top-down:
# start at the cockpit → scan for candidates → filter the day (macro/risk) →
# weekly bias → daily confirm → 4H zone → 15M trigger → size the risk → review.
# This is the single source of truth for both the sidebar and the README's
# step-by-step guide — keep the two in sync when reordering.
NAV_SECTIONS: List[tuple] = [
    ("0 · START HERE", [
        NavEntry("CHCK", "00. Daily Checklist",      "📋", "app.py"),
        NavEntry("DCKP", "01. Daily Cockpit",        "🛫", "pages/daily_cockpit_tab.py"),
        NavEntry("OVRV", "02. Market Overview",      "📊", "pages/market-overview.py"),
    ]),
    ("1 · SCAN — SHORTLIST", [
        NavEntry("RANK", "03. Setup Ranker",         "🎰", "pages/setup-ranker.py"),
        NavEntry("AMD",  "04. AMD Scanner",          "📊", "pages/amd-scanner.py"),
        NavEntry("TSIG", "05. Trend Signals",        "📡", "pages/trend-signals.py"),
        NavEntry("20DB", "06. 20-Day Breakout",      "🚀", "pages/twenty_day_breakout_tab.py"),
        NavEntry("SMRT", "07. Smart Money",          "🕵️", "pages/smart_money_tab.py"),
        NavEntry("PRED", "08. Predictive Analytics", "💱", "pages/predictive.py"),
        NavEntry("FCST", "09. Forecast Lab",         "📈", "pages/forecast-dashboard.py"),
        NavEntry("VWAP", "10. VWAP-EMA Gold",        "🟡", "pages/vwap-ema-gold.py"),
    ]),
    ("2 · FILTER THE DAY", [
        NavEntry("MACR", "11. Macro Bias",           "🌐", "pages/macro-bias.py"),
        NavEntry("FUND", "12. Forex Fundamentals",   "🌍", "pages/forex_fundamentals_tab.py"),
        NavEntry("DXAU", "13. DXY vs Gold",          "💵", "pages/dxy-gold.py"),
        NavEntry("RGME", "14. Market Regime",        "🌡️", "pages/regime.py"),
        NavEntry("RVRS", "15. Risk Reversals",       "🎭", "pages/risk_reversal_tab.py"),
        NavEntry("NEWS", "16. News Filter",          "📰", "pages/news-filter.py"),
        NavEntry("EVNT", "17. Event Impact",         "📅", "pages/event_impact_tab.py"),
        NavEntry("SEAS", "18. Seasonality",          "📅", "pages/seasonality.py"),
        NavEntry("CORR", "19. Correlations",         "🔗", "pages/correlations.py"),
        NavEntry("ATR",  "20. ATR Volatility",       "📊", "pages/atr-volatility.py"),
    ]),
    ("3 · WEEKLY BIAS", [
        NavEntry("WEMA", "21. Weekly EMA",           "📉", "pages/weekly-ema.py"),
        NavEntry("WRSI", "22. Weekly RSI",           "📡", "pages/weekly-rsi.py"),
        NavEntry("WSWG", "23. Weekly Swing",         "🔄", "pages/weekly-swing.py"),
    ]),
    ("4 · DAILY CONFIRM", [
        NavEntry("DTRN", "24. Daily Trend",          "📈", "pages/daily-trend.py"),
        NavEntry("DMCD", "25. Daily MACD",           "📊", "pages/daily-macd.py"),
        NavEntry("STRC", "26. Market Structure",     "🏗️", "pages/market-structure.py"),
    ]),
    ("5 · 4H ZONE", [
        NavEntry("4HCZ", "27. 4H Confluence Zone",   "🎯", "pages/4H-confluence-zone.py"),
        NavEntry("CONF", "28. 2/3 Confluence Check", "🔀", "pages/confluence-checker.py"),
    ]),
    ("6 · 15M TRIGGER", [
        NavEntry("15FB", "29. 15M Fib Entry",        "⚡", "pages/15m-fib-entry.py"),
        NavEntry("DZER", "30. Double Zeros",         "🎯", "pages/double_zeros.py"),
    ]),
    ("7 · RISK & EXECUTE", [
        NavEntry("STOP", "31. Stop Structure",       "🛡️", "pages/stop-structure.py"),
        NavEntry("RRC",  "32. R:R Calculator",       "⚖️", "pages/rr-calculator.py"),
        NavEntry("ACCT", "33. Account Risk",         "💵", "pages/account-risk.py"),
    ]),
    ("8 · REVIEW", [
        NavEntry("JRNL", "34. Trade Journal",        "📓", "pages/trade-journal.py"),
        NavEntry("BTST", "35. Backtest Lab",         "🧪", "pages/backtest-workflow.py"),
        NavEntry("TLAB", "36. Trading Lab",          "🧪", "pages/trading_lab_tab.py"),
    ]),
    ("9 · SYSTEM", [
        NavEntry("RPRT", "37. Reports",              "📑", "pages/reports.py"),
        NavEntry("LOGS", "38. System Logs",          "🧾", "pages/system-logs.py"),
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
