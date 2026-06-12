"""Single source of truth for the page navigation.

Renumbered in Bloomberg-function-code style. Every page file uses this list to
render its sidebar and its function-code grid — no more hand-maintained copies
across 22 files.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class NavEntry:
    code: str          # Bloomberg-style mnemonic (e.g. "CHCK", "RANK")
    label: str         # Human-friendly label
    icon: str          # Emoji prefix
    path: str          # Streamlit page path (top-level file or pages/foo.py)


NAV_ENTRIES: List[NavEntry] = [
    NavEntry("CHCK", "00. Checklist",            "📋", "daily-trading-checklist.py"),
    NavEntry("RANK", "01. Setup Ranker",         "🎰", "pages/setup-ranker.py"),
    NavEntry("MACR", "02. Macro Bias",           "🌐", "pages/macro-bias.py"),
    NavEntry("NEWS", "03. News Filter",          "📰", "pages/news-filter.py"),
    NavEntry("CORR", "04. Correlations",         "🔗", "pages/correlations.py"),
    NavEntry("ATR",  "05. ATR Volatility",       "📊", "pages/atr-volatility.py"),
    NavEntry("WEMA", "06. Weekly EMA",           "📉", "pages/weekly-ema.py"),
    NavEntry("WRSI", "07. Weekly RSI",           "📡", "pages/weekly-rsi.py"),
    NavEntry("WSWG", "08. Weekly Swing",         "🔄", "pages/weekly-swing.py"),
    NavEntry("DTRN", "09. Daily Trend",          "📈", "pages/daily-trend.py"),
    NavEntry("DMCD", "10. Daily MACD",           "📊", "pages/daily-macd.py"),
    NavEntry("4HCZ", "11. 4H Confluence Zone",   "🎯", "pages/4H-confluence-zone.py"),
    NavEntry("CONF", "12. 2/3 Confluence Check", "🔀", "pages/confluence-checker.py"),
    NavEntry("15RJ", "13. 15M Rejection",        "🕯️", "pages/15m-rejection.py"),
    NavEntry("15EN", "14. 15M Entry Signal",     "⚡", "pages/15m-entry-signal.py"),
    NavEntry("STOP", "15. Stop Structure",       "🛡️", "pages/stop-structure.py"),
    NavEntry("RRC",  "16. R:R Calculator",       "⚖️", "pages/rr-calculator.py"),
    NavEntry("JRNL", "17. Trade Journal",        "📓", "pages/trade-journal.py"),
    NavEntry("STRC", "18. Market Structure",     "🏗️", "pages/market-structure.py"),
    NavEntry("BTST", "19. Backtest Workflow",    "🧪", "pages/backtest-workflow.py"),
    NavEntry("AMD",  "20. AMD Scanner",          "📊", "pages/amd-scanner.py"),
]
