"""Fibo Ribbon — the desk's own chart stack, scanned across the universe.

The chart came first here, which is unusual for this repo. `mt5/FiboRibbon.mq5`
recreates the moving-average stack read out of the trader's saved MT5 chart
profile; `src/core/fibo_ribbon.py` is the pure Python twin of that logic, and
this page runs it over every registry instrument so the read reaches the Source
Scorecard like any other signal.

The signal is the trigger cross **gated by the ribbon it sits inside** — a bare
5/8 cross fires constantly in chop, and the EMA21 > EMA55 > EMA200 ordering is
what separates a fast cross going with the larger trend from one fighting it.
Ungated crosses are still shown, tagged as gated-out, so the gate's value can be
judged rather than assumed.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from src.core.fibo_ribbon import LONG, NEUTRAL, SHORT, RibbonRead, read
from src.instruments.registry import INSTRUMENTS
from src.pages_lib.base import BloombergPage, PageContext
from src.services.market_data import daily_ohlc
from src.services.signal_store import persist_signals
from src.ui.theme import BloombergTheme as T

# The gate needs 200 daily bars before it means anything, and the canonical
# daily window is 300 — enough, but only just, so widen it here rather than
# silently scoring a half-formed EMA200. Interval/resample stay the spine's.
_DAILY_PERIOD = "3y"
_HORIZON_DAYS = 10


class FiboRibbonPage(BloombergPage):
    """Universe scan of the 5/21/55/200 EMA ribbon + 8-SMA(open) trigger."""

    def configure(self) -> PageContext:
        return PageContext(code="RIBN", title="Fibo Ribbon", icon="🎗️")

    # ── sidebar ─────────────────────────────────────────────────────────────
    def sidebar(self, ctx: PageContext) -> None:
        st.markdown("### 🎗️ RIBBON SCAN")
        st.caption("EMA 5/21/55/200 on close · SMA 8 on **open** — the stack "
                   "read from the desk's own MT5 chart profile.")
        # Display-only. The persisted set is derived from the data, never from a
        # widget — see the daily-trend lesson in CLAUDE.md.
        st.session_state.setdefault("ribn_show_gated", False)
        st.session_state.ribn_show_gated = st.checkbox(
            "Show gated-out crosses", value=st.session_state.ribn_show_gated,
            help="Crosses that fired against the ribbon. Shown for judgement, "
                 "never persisted.")
        st.divider()

    # ── scan ────────────────────────────────────────────────────────────────
    @staticmethod
    @st.cache_data(ttl=300, show_spinner=False)
    def _scan() -> Dict[str, dict]:
        """One ribbon read per instrument. Cached like every other fetcher."""
        out: Dict[str, dict] = {}
        for pair, inst in INSTRUMENTS.items():
            try:
                df = daily_ohlc(inst.ticker, period=_DAILY_PERIOD)
                r: Optional[RibbonRead] = read(df)
                if r is not None:
                    out[pair] = r.to_dict()
            except Exception:
                continue          # one bad symbol must not lose the scan
        return out

    # ── body ────────────────────────────────────────────────────────────────
    def body(self, ctx: PageContext) -> None:
        results = self._scan()
        if not results:
            st.warning("No instrument had enough daily history for a 200-EMA gate.")
            return

        longs = [p for p, r in results.items() if r["direction"] == LONG]
        shorts = [p for p, r in results.items() if r["direction"] == SHORT]
        gated = [p for p, r in results.items()
                 if r["direction"] == NEUTRAL and r["crossed"]]

        # Persist the gated-through crosses only. Derived from the data, so the
        # sidebar checkbox cannot change what reaches trade_setups.
        self._persist(results, longs + shorts)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("▲ Long crosses", len(longs))
        c2.metric("▼ Short crosses", len(shorts))
        c3.metric("⛔ Gated out", len(gated))
        c4.metric("Scanned", len(results))

        if not longs and not shorts:
            st.info("No cross cleared the ribbon gate on the last closed bar. "
                    "That is the common case — the gate exists to say no.")

        for pair in longs + shorts:
            self._card(pair, results[pair])

        if st.session_state.get("ribn_show_gated") and gated:
            st.markdown("##### ⛔ Gated-out crosses")
            st.caption("The trigger crossed but the ribbon disagreed. Never "
                       "persisted — listed so the gate can be judged.")
            for pair in gated:
                self._card(pair, results[pair], muted=True)

        st.caption("Signals are read from the last **closed** daily bar, so they "
                   "cannot repaint. Mirrors `mt5/FiboRibbon.mq5` buffers 5/6.")

    # ── rendering ───────────────────────────────────────────────────────────
    @staticmethod
    def _card(pair: str, r: dict, muted: bool = False) -> None:
        up = r["direction"] == LONG
        colour = T.AMBER if (up and not muted) else ("#ff3344" if not muted else "#9a9a9a")
        arrow = "▲" if up else "▼"
        st.markdown(
            f"<div style='border-left:3px solid {colour};padding:6px 10px;margin:6px 0;'>"
            f"<b style='color:{colour}'>{arrow} {pair}</b> "
            f"<span style='color:#9a9a9a'>close {r['close']:.5f} · "
            f"EMA5 {r['ema5']:.5f} vs SMA8 {r['sma8']:.5f}</span><br>"
            f"<span style='color:#9a9a9a;font-size:11px'>{r['reason']} · "
            f"bar {str(r['bar_time'])[:10]}</span></div>",
            unsafe_allow_html=True)

    # ── persistence ─────────────────────────────────────────────────────────
    @staticmethod
    def _persist(results: Dict[str, dict], pairs: List[str]) -> None:
        signals = []
        for pair in pairs:
            r = results[pair]
            signals.append({
                "pair": pair,
                "bias": r["direction"],
                "entry": r["close"],
                # Last closed daily bar — the read's own provenance, and what
                # keeps dedupe per-bar instead of per-day.
                "bar_time": r["bar_time"],
                "horizon_days": _HORIZON_DAYS,
                "conviction": "Ribbon cross",
                "thesis": ("Fibo Ribbon — EMA5 crossed {0} SMA8(open) with the "
                           "ribbon aligned (EMA21 {1} EMA55 {1} EMA200)".format(
                               "above" if r["direction"] == LONG else "below",
                               ">" if r["direction"] == LONG else "<")),
            })
        persist_signals("fibo_ribbon", signals)
