"""Biased Pivots — the FX Bootcamp pivot zones, scanned across the universe.

Python twin of `BiasedPivots.mq5`, which is on the live EUR/CAD chart. Same
levels, same "one period earlier" close, same zone semantics taken from the
indicator's own labels — so the scanner and the chart cannot disagree about
where the buying and selling zones are.

The signal is location, not momentum: price inside the **Bullish Buying Zone**
(around M2) reads Long, inside the **Bearish Selling Zone** (around M3) reads
Short. Targets come from the source's labelled levels rather than an R-multiple,
because that is what the indicator actually claims.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import streamlit as st

from src.core.biased_pivots import LONG, NEUTRAL, SHORT, Pivots, read, targets
from src.instruments.registry import INSTRUMENTS
from src.pages_lib.base import BloombergPage, PageContext
from src.services.market_data import weekly_ohlc
from src.services.signal_store import persist_signals
from src.ui.theme import BloombergTheme as T

# Pivot levels are recomputed every period: a bar closes and PP/S1/R1/M2/M3 all
# move. A signal derived from them is therefore valid for about ONE period and
# no longer. This page used daily pivots with a 5-day horizon, so every signal
# outlived its own levels by four days -- and the comment here conceded it:
# "pivot zones are a day-trade construct, not a swing one".
#
# Weekly pivots resolve it without touching the maths. `read()` is
# period-agnostic, and the MQL5 original is period-parameterised
# (`timePeriodConverter()`), so this is the port catching up. The horizon is
# derived from the period rather than written next to it, because two numbers
# that must agree will not stay in agreement if a human maintains both.
_PIVOT_PERIOD = "weekly"
_TRADING_DAYS = {"daily": 1, "weekly": 5, "monthly": 21}
_HORIZON_DAYS = _TRADING_DAYS[_PIVOT_PERIOD]


class BiasedPivotsPage(BloombergPage):
    """Universe scan of the FX Bootcamp pivot buying/selling zones."""

    def configure(self) -> PageContext:
        return PageContext(code="PIVB", title="Biased Pivots", icon="🧿")

    def sidebar(self, ctx: PageContext) -> None:
        st.markdown("### 🧿 PIVOT ZONES")
        st.caption("FX Bootcamp levels ported from `BiasedPivots.mq5`. "
                   "M2 = Bullish Buying Zone · M3 = Bearish Selling Zone.")
        st.caption("⚠ The close used for PP comes from one period **earlier** "
                   "than the high/low — the indicator's own quirk, preserved so "
                   "these levels match the chart.")
        st.caption("Levels come from the last **completed week**, so a read is "
                   "good for the week ahead. Daily pivots move every night — "
                   "too fast for a 5-day swing horizon.")
        st.divider()

    @staticmethod
    @st.cache_data(ttl=300, show_spinner=False)
    def _scan() -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for pair, inst in INSTRUMENTS.items():
            try:
                r: Optional[Pivots] = read(weekly_ohlc(inst.ticker))
                if r is not None:
                    d = r.to_dict()
                    d["_targets"] = targets(r)
                    out[pair] = d
            except Exception:
                continue
        return out

    def body(self, ctx: PageContext) -> None:
        results = self._scan()
        if not results:
            st.warning("No instrument returned enough daily history for pivots.")
            return

        longs = [p for p, r in results.items() if r["direction"] == LONG]
        shorts = [p for p, r in results.items() if r["direction"] == SHORT]
        self._persist(results, longs + shorts)

        c1, c2, c3 = st.columns(3)
        c1.metric("▲ In buying zone", len(longs))
        c2.metric("▼ In selling zone", len(shorts))
        c3.metric("Scanned", len(results))

        if not longs and not shorts:
            st.info("Nothing is sitting in a pivot zone right now.")

        for pair in longs + shorts:
            self._card(pair, results[pair])

        st.caption("Levels are built from the last **closed** daily period; only "
                   "the price being judged comes from the forming bar.")

    @staticmethod
    def _card(pair: str, r: dict) -> None:
        up = r["direction"] == LONG
        colour = T.AMBER if up else "#ff3344"
        t = r.get("_targets") or {}
        st.markdown(
            f"<div style='border-left:3px solid {colour};padding:6px 10px;margin:6px 0;'>"
            f"<b style='color:{colour}'>{'▲' if up else '▼'} {pair}</b> "
            f"<span style='color:#9a9a9a'>{r['zone']} · price {r['price']:.5f}</span><br>"
            f"<span style='color:#9a9a9a;font-size:11px'>"
            f"PP {r['pp']:.5f} · S1 {r['s1']:.5f} · R1 {r['r1']:.5f} · "
            f"target {t.get('conservative') or 0:.5f} (cons) / "
            f"{t.get('aggressive') or 0:.5f} (aggr)</span></div>",
            unsafe_allow_html=True)

    @staticmethod
    def _persist(results: Dict[str, dict], pairs: List[str]) -> None:
        signals = []
        for pair in pairs:
            r = results[pair]
            t = r.get("_targets") or {}
            signals.append({
                "pair": pair,
                "bias": r["direction"],
                "entry": r["price"],
                "stop_loss": t.get("stop_ref"),
                "take_profit_1": t.get("conservative"),
                "take_profit_2": t.get("aggressive"),
                "bar_time": r["bar_time"],
                "horizon_days": _HORIZON_DAYS,
                "conviction": r["zone"],
                "thesis": ("Biased Pivots — {0}; PP {1:.5f}, targets {2} "
                           "(conservative) / {3} (aggressive)".format(
                               r["zone"], r["pp"],
                               "{0:.5f}".format(t["conservative"]) if t.get("conservative") else "—",
                               "{0:.5f}".format(t["aggressive"]) if t.get("aggressive") else "—")),
            })
        persist_signals("biased_pivots", signals)
