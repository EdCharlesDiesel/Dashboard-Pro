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


# Re-architected (2026-07) around WHEN a page gets touched during the day,
# not the order features were built or a numbered sequential walkthrough.
# 44 pages is too many stops for any daily routine to survive — the fix is
# separating the daily path (walked in order, ≤8 touches) from reference and
# research (visited on demand). Numeric label prefixes (00/05a/09a-style) are
# dropped entirely: the section grouping IS the ordering, and lettered
# suffixes were a smell that pages accreted rather than got designed. This is
# the single source of truth for both the sidebar and README/System_Guide's
# walkthroughs — keep those in sync when reordering.
#
#   Morning Brief   — 5 min, sets bias, no decisions. MTF Matrix leads: it is
#                     the house-view consensus board (src/core/bias.py) — the
#                     canonical direction every other page anchors to — so the
#                     day starts by reading the board, not by scanning.
#   Pre-Session     — build the shortlist: Setup Ranker (primary scan) +
#                     Trend Signals (confirm/deny the top picks) + Currency
#                     Strength/Correlations (kill conflicted exposure) +
#                     Confluence Check (the final gate).
#   Session         — execution only, nothing analytical. If you can re-open
#                     the Ranker mid-session you will, and that's how a flat
#                     day becomes a revenge trade — the shortlist is frozen
#                     at session open.
#   Weekend         — weekly-bias tools (EMA/Swing/Playbook) + the full COT
#                     suite; having them in the daily path just adds scroll.
#   Research Lab    — everything else: the other directional scanners (AMD,
#                     20-Day Breakout — one primary + one
#                     confirmer earns the daily-path slot, not all five;
#                     revisit once there's enough journaled history to know
#                     which one actually earns it), Daily Trend/MACD/Market
#                     Structure (genuinely more analysis than the Ranker's
#                     checklist booleans — kept as full pages, just not part
#                     of the 7-touch day), and the stat-arb/backtest/vol
#                     research tools. Visited when building or validating,
#                     never during a session.
#   Reference       — System Guide, ABR Toolkit, the 18-point Daily Checklist.
NAV_SECTIONS: List[tuple] = [
    ("🌅 MORNING BRIEF", [
        NavEntry("MTFX", "MTF Matrix (House View)", "🧭", "pages/mtf-matrix.py"),
        NavEntry("DCKP", "Daily Cockpit",   "🛫", "pages/daily_cockpit_tab.py"),
        NavEntry("OVRV", "Market Overview", "📊", "pages/market-overview.py"),
        NavEntry("NEWS", "News Filter",     "📰", "pages/news-filter.py"),
    ]),
    ("📋 PRE-SESSION", [
        NavEntry("TDAY", "Today's Trades",     "🎯", "pages/todays-trades.py"),
        NavEntry("RANK", "Setup Ranker",       "🎰", "pages/setup-ranker.py"),
        NavEntry("TSIG", "Trend Signals",      "📡", "pages/trend-signals.py"),
        NavEntry("CCYS", "Currency Strength",  "💪", "pages/currency-strength.py"),
        NavEntry("CORR", "Correlations",       "🔗", "pages/correlations.py"),
        NavEntry("CONF", "Confluence Check",   "🔀", "pages/confluence-checker.py"),
    ]),
    ("⚡ SESSION — execution only", [
        NavEntry("15FB", "15M Fib Entry", "⚡", "pages/15m-fib-entry.py"),
        NavEntry("RSKS", "Risk Suite",    "🛡️", "pages/risk-suite.py"),
        NavEntry("THRT", "Threat Board",  "🚨", "pages/threat_board_tab.py"),
        NavEntry("JRNL", "Trade Journal", "📓", "pages/trade-journal.py"),
    ]),
    ("📅 WEEKEND — weekly bias & COT", [
        NavEntry("WKAH", "Week Ahead",              "🗓️", "pages/week_ahead_tab.py"),
        NavEntry("WEMA", "Weekly EMA",              "📉", "pages/weekly-ema.py"),
        NavEntry("WSWG", "Weekly Swing",            "🔄", "pages/weekly-swing.py"),
        NavEntry("SWPB", "Swing Playbook",          "📔", "pages/swing_playbook_tab.py"),
        NavEntry("COT",  "COT Positioning",         "🏛️", "pages/cot_tab.py"),
        NavEntry("COTS", "COT Signals",             "🧭", "pages/cot_signals.py"),
        NavEntry("COTO", "COT Open Interest",       "🧮", "pages/cot_open_interest.py"),
        NavEntry("GCOT", "Gold COT",                "🥇", "pages/gold_cot_tab.py"),
        NavEntry("OCOT", "Oil COT",                 "🛢️", "pages/oil_cot_tab.py"),
        NavEntry("BGDX", "Bonds → Gold → DXY",      "🏦", "pages/bonds_gold_dxy_app.py"),
        NavEntry("COTX", "COT Composite Signal",    "🧩", "pages/cot_composite_trade_signal.py"),
        NavEntry("COTB", "COT Composite Backtest",  "🧪", "pages/cot_trade_signal_walk_forward_backtest_harness.py"),
    ]),
    ("🔬 RESEARCH LAB — on demand, never mid-session", [
        NavEntry("AMD",  "AMD Scanner",           "📊", "pages/amd-scanner.py"),
        NavEntry("LEAD", "Leading Indicators",    "🧲", "pages/leading-indicators.py"),
        NavEntry("20DB", "20-Day Breakout",       "🚀", "pages/twenty_day_breakout_tab.py"),
        NavEntry("CMEF", "CME FX Futures",        "📦", "pages/cme-futures-volume.py"),
        NavEntry("PRED", "Predictive Analytics",  "💱", "pages/predictive.py"),
        NavEntry("VWAP", "VWAP-EMA Gold",         "🟡", "pages/vwap-ema-gold.py"),
        NavEntry("PRDX", "Instrument Predictor",  "🔮", "pages/instrument-predictor.py"),
        NavEntry("DXAU", "DXY vs Gold",           "💵", "pages/dxy-gold.py"),
        NavEntry("IDXC", "Indices Correlation",   "🇺🇸", "pages/indices-correlation.py"),
        NavEntry("DTRN", "Daily Trend",           "📈", "pages/daily-trend.py"),
        NavEntry("DMCD", "Daily MACD",            "📊", "pages/daily-macd.py"),
        NavEntry("STRC", "Market Structure",      "🏗️", "pages/market-structure.py"),
        NavEntry("4HCZ", "4H Confluence Zone",    "🎯", "pages/4H-confluence-zone.py"),
        NavEntry("EWKV", "Busy-Week Anatomy",     "📅", "pages/event_week_vol_tab.py"),
        NavEntry("DISC", "Disconnect Monitor",    "🔌", "pages/disconnect_monitor_tab.py"),
        NavEntry("LQHT", "Liquidity Hunt Lab",    "🎣", "pages/liquidity_hunt_tab.py"),
        NavEntry("ONDR", "Overnight Drift",       "🌙", "pages/overnight_drift_tab.py"),
        NavEntry("HOLD", "Optimal Holding Period","⏱️", "pages/holding_period_tab.py"),
        NavEntry("QNTM", "Quant Models Lab",      "🧮", "pages/quant_models_tab.py"),
        NavEntry("MTGL", "Martingale Diagnostics","🎲", "pages/martingale_tab.py"),
        NavEntry("STOC", "Stochastic Calculus",  "🎓", "pages/stochastic_tab.py"),
        NavEntry("FCST", "Forecast",              "🔭", "pages/forecast_tab.py"),
        NavEntry("SURP", "Surprise Awareness",    "😲", "pages/surprise_tab.py"),
        NavEntry("SEAS", "Seasonality",           "📅", "pages/seasonality.py"),
        NavEntry("RSKR", "Risk Reversals",        "🎭", "pages/risk_reversal_tab.py"),
        NavEntry("SMRT", "Smart Money",           "🕵️", "pages/smart_money_tab.py"),
        NavEntry("RIBN", "Fibo Ribbon",           "🎗️", "pages/fibo-ribbon.py"),
        NavEntry("PIVB", "Biased Pivots",         "🧿", "pages/biased-pivots.py"),
    ]),
    ("📖 REFERENCE", [
        NavEntry("CHCK", "Daily Checklist (18-point)", "📋", "app.py"),
        NavEntry("GUID", "System Guide & Playbook",    "📖", "pages/system_guide.py"),
        NavEntry("ABRT", "ABR Toolkit",                "🧱", "pages/abr_toolkit_tab.py"),
    ]),
]

# Archived (moved to archive/pages/ — not deleted, just off the daily nav):
# Trading Ideas, Technical Chart, Pivots & Fibonacci, Volume Profile, FRED
# Macro Grid, Forecast Lab, Macro Bias, Forex Fundamentals, Market Regime,
# Event Impact, ATR Volatility, Weekly RSI, Double Zeros, Backtest Lab,
# Trading Lab, Reports, System Logs, Stop Structure, R:R Calculator,
# Account Risk (superseded by Risk Suite).
#
# Un-archived 2026-08-04: Seasonality, Risk Reversals and Smart Money were
# restored to pages/ and re-registered above. All three already called
# persist_signals before they were archived, so they were feeding the Source
# Scorecard until the 2026-07-03 cleanup silently cut off three signal sources.
# They are swept by src/services/signal_sweep.py. (Forecast Lab stays archived —
# pages/forecast_tab.py is its live successor and now carries the
# source='forecast_dashboard' tag.)

# Flat list kept for any caller that doesn't care about grouping.
NAV_ENTRIES: List[NavEntry] = [e for _, entries in NAV_SECTIONS for e in entries]


def render_sidebar_nav() -> None:
    """Render the uniform sidebar navigation (call inside `with st.sidebar:`).

    Uses st.page_link so navigation is SPA-style and preserves session state.
    Every page should call this instead of hand-maintaining its own link list.
    """
    # Every page calls this, which makes it the one guaranteed per-process
    # hook: kick off the unattended Grade-A scanner (idempotent — a lock +
    # liveness check after the first call; no-op under pytest). Lives here so
    # alerts keep firing even when no page (or browser) is open.
    try:
        from src.services.background_scanner import ensure_started
        ensure_started()
    except Exception as exc:  # the nav must render no matter what
        logger.warning("[nav] background scanner failed to start: %s", exc)

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
