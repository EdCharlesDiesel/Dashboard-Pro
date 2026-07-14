"""Single source of truth for the page navigation.

Renumbered in Bloomberg-function-code style. Every page file uses this list to
render its sidebar and its function-code grid — no more hand-maintained copies
across 22 files.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import streamlit as st

logger = logging.getLogger("ForexDashboard")


@dataclass(frozen=True)
class NavEntry:
    code: str          # Bloomberg-style mnemonic (e.g. "CHCK", "RANK")
    label: str         # Human-friendly label
    icon: str          # Emoji prefix
    path: str          # Streamlit page path (top-level file or pages/foo.py)


# Re-architected (2026-07) from the original 9-section layout. Ordered the way
# a professional day trader works a session, top-down: cockpit → scan for
# candidates → synthesize a composite read → filter the day (DAILY macro/risk
# only) → weekly bias → daily confirm → 4H zone → 15M trigger → size the risk
# → review → weekly/occasional research lab. This is the single source of
# truth for both the sidebar and the README's step-by-step guide — keep the
# two in sync when reordering.
#
# Two changes from the prior layout, both driven by actual cadence rather than
# where a page happened to be added historically:
#   1. New "2 · SYNTHESIZE" section for the Instrument Predictor — it combines
#      Scan-stage (Setup Score) and Filter-stage (Currency Strength, COT)
#      reads into one composite, so it naturally sits as a checkpoint between
#      the two rather than inside either.
#   2. "Filter the day" now holds ONLY genuinely daily-checked tools (DXY vs
#      Gold, Currency Strength, News Filter, Correlations). Every COT page
#      (17/18/20/20a/20b previously) plus Bonds→Gold→DXY are weekly-cadence
#      per their own captions/README notes ("check once a week, not every
#      session") and are consolidated into "10 · WEEKLY & RESEARCH LAB" with
#      the other weekly/occasional tools that already lived there — one
#      section for "not part of the daily top-to-bottom pass," not two.
NAV_SECTIONS: List[tuple] = [
    ("0 · COCKPIT", [
        NavEntry("CHCK", "00. Daily Checklist",           "📋", "app.py"),
        NavEntry("GUID", "00a. System Guide & Playbook",  "📖", "pages/system_guide.py"),
        NavEntry("DCKP", "01. Daily Cockpit",             "🛫", "pages/daily_cockpit_tab.py"),
        NavEntry("OVRV", "02. Market Overview",           "📊", "pages/market-overview.py"),
    ]),
    ("1 · SCAN — SHORTLIST", [
        NavEntry("RANK", "03. Setup Ranker",         "🎰", "pages/setup-ranker.py"),
        NavEntry("AMD",  "04. AMD Scanner",          "📊", "pages/amd-scanner.py"),
        NavEntry("TSIG", "05. Trend Signals",        "📡", "pages/trend-signals.py"),
        NavEntry("MTFX", "05a. MTF Matrix",          "🧭", "pages/mtf-matrix.py"),
        NavEntry("20DB", "06. 20-Day Breakout",      "🚀", "pages/twenty_day_breakout_tab.py"),
        NavEntry("CMEF", "07. CME FX Futures",       "📦", "pages/cme-futures-volume.py"),
        NavEntry("PRED", "08. Predictive Analytics", "💱", "pages/predictive.py"),
        NavEntry("VWAP", "09. VWAP-EMA Gold",        "🟡", "pages/vwap-ema-gold.py"),
        NavEntry("ABRT", "09a. ABR Toolkit",         "🧱", "pages/abr_toolkit_tab.py"),
    ]),
    # NEW — the composite cross-check: Setup Score (Scan) + Trend Signal +
    # Currency Strength + COT Composite (Filter), one weighted directional
    # read. Run it on your shortlist before sinking time into deeper macro
    # digging on a candidate that doesn't hold up under a second opinion.
    ("2 · SYNTHESIZE — composite read", [
        NavEntry("PRDX", "10. Instrument Predictor", "🔮", "pages/instrument-predictor.py"),
    ]),
    ("3 · FILTER THE DAY", [
        NavEntry("DXAU", "11. DXY vs Gold",          "💵", "pages/dxy-gold.py"),
        NavEntry("CCYS", "12. Currency Strength",    "💪", "pages/currency-strength.py"),
        NavEntry("NEWS", "13. News Filter",          "📰", "pages/news-filter.py"),
        NavEntry("CORR", "14. Correlations",         "🔗", "pages/correlations.py"),
    ]),
    ("4 · WEEKLY BIAS", [
        NavEntry("WEMA", "15. Weekly EMA",           "📉", "pages/weekly-ema.py"),
        NavEntry("WSWG", "16. Weekly Swing",         "🔄", "pages/weekly-swing.py"),
    ]),
    ("5 · DAILY CONFIRM", [
        NavEntry("DTRN", "17. Daily Trend",          "📈", "pages/daily-trend.py"),
        NavEntry("DMCD", "18. Daily MACD",           "📊", "pages/daily-macd.py"),
        NavEntry("STRC", "19. Market Structure",     "🏗️", "pages/market-structure.py"),
    ]),
    ("6 · 4H ZONE", [
        NavEntry("4HCZ", "20. 4H Confluence Zone",   "🎯", "pages/4H-confluence-zone.py"),
        NavEntry("CONF", "21. 2/3 Confluence Check", "🔀", "pages/confluence-checker.py"),
    ]),
    ("7 · 15M TRIGGER", [
        NavEntry("15FB", "22. 15M Fib Entry",        "⚡", "pages/15m-fib-entry.py"),
    ]),
    ("8 · RISK & EXECUTE", [
        NavEntry("STOP", "23. Stop Structure",       "🛡️", "pages/stop-structure.py"),
        NavEntry("RRC",  "24. R:R Calculator",       "⚖️", "pages/rr-calculator.py"),
        NavEntry("ACCT", "25. Account Risk",         "💵", "pages/account-risk.py"),
    ]),
    ("9 · REVIEW", [
        NavEntry("JRNL", "26. Trade Journal",        "📓", "pages/trade-journal.py"),
    ]),
    # Not a sequential daily step -- every weekly-cadence CFTC/COT page (moved
    # here from "Filter the day"), the educational cross-asset context page,
    # and the occasional stat-arb/backtest research tools. Check this section
    # weekly (Monday morning is a good habit), not every session.
    ("10 · WEEKLY & RESEARCH LAB", [
        NavEntry("COT",  "27. COT Positioning",        "🏛️", "pages/cot_tab.py"),
        NavEntry("COTS", "28. COT Signals",            "🧭", "pages/cot_signals.py"),
        NavEntry("COTO", "29. COT Open Interest",      "🧮", "pages/cot_open_interest.py"),
        NavEntry("GCOT", "29a. Gold COT",               "🥇", "pages/gold_cot_tab.py"),
        NavEntry("OCOT", "29b. Oil COT",                "🛢️", "pages/oil_cot_tab.py"),
        NavEntry("BGDX", "30. Bonds → Gold → DXY",     "🏦", "pages/bonds_gold_dxy_app.py"),
        NavEntry("COTX", "31. COT Composite Signal",   "🧩", "pages/cot_composite_trade_signal.py"),
        NavEntry("COTB", "32. COT Composite Backtest", "🧪", "pages/cot_trade_signal_walk_forward_backtest_harness.py"),
        NavEntry("EWKV", "33. Busy-Week Anatomy",      "📅", "pages/event_week_vol_tab.py"),
        NavEntry("DISC", "34. Disconnect Monitor",     "🔌", "pages/disconnect_monitor_tab.py"),
        NavEntry("ONDR", "35. Overnight Drift",        "🌙", "pages/overnight_drift_tab.py"),
        NavEntry("HOLD", "36. Optimal Holding Period", "⏱️", "pages/holding_period_tab.py"),
        NavEntry("QNTM", "37. Quant Models Lab",       "🧮", "pages/quant_models_tab.py"),
        NavEntry("FCST", "37a. Forecast",              "🔭", "pages/forecast_tab.py"),
        NavEntry("SURP", "38. Surprise Awareness",     "😲", "pages/surprise_tab.py"),
    ]),
]

# Archived (moved to archive/pages/ — not deleted, just off the daily nav):
# Trading Ideas, Technical Chart, Pivots & Fibonacci, Volume Profile, FRED
# Macro Grid, Smart Money, Forecast Lab, Macro Bias, Forex Fundamentals,
# Market Regime, Risk Reversals, Event Impact, Seasonality, ATR Volatility,
# Weekly RSI, Double Zeros, Backtest Lab, Trading Lab, Reports, System Logs.
# (MTF Matrix was restored to the live nav above — see MTFX.)

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
            except Exception as exc:
                # st.page_link needs the multipage registry (absent in bare/test
                # runs) and raises if a page file was renamed/missing — skip the
                # entry rather than crash the whole page, but log it: a silent
                # `continue` here previously made a bad entry indistinguishable
                # from "the dev server just needs a restart."
                logger.warning("[nav] page_link failed for %s (%s): %s",
                               e.code, e.path, exc)
                continue
