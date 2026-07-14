"""Currency Strength Meter — rank the 9 base/quote currencies strong to weak.

Each currency's strength is the average % return, over a chosen lookback
window, of every registry forex pair it appears in (positive when it's the
base leg, negated when it's the quote leg — mirrors how every retail
strength-meter indicator works). Ranking currencies this way surfaces the
highest-probability pairs directly: strongest-vs-weakest divergence is the
cleanest trend combination available in the registry.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.instruments.registry import INSTRUMENTS
from src.pages_lib.base import BloombergPage, PageContext
from src.services.signal_store import persist_signals
from src.ui.components import CommandBar, MetricCell, Panel, render_metric_row
from src.ui.theme import BloombergTheme as T

# Trading-day windows shown for every currency (short/primary/long context).
_WINDOWS: Tuple[int, ...] = (5, 20, 60, 120)

_PRIMARY_OPTIONS: Dict[str, int] = {
    "5 Day": 5, "20 Day": 20, "60 Day": 60, "120 Day": 120,
}

# Minimum |strength diff| (in %) between a pair's two legs before it's treated
# as a clear enough divergence to persist as a trade idea.
_IDEA_THRESHOLD = 0.15


@st.cache_data(ttl=600, show_spinner=False)
def _fetch_pair_closes() -> pd.DataFrame:
    """Daily closes for every registry forex pair (commodities excluded —
    gold/silver/platinum aren't driven by a single fiat currency's strength).
    A full year gives the 120-day window enough headroom past holidays/gaps."""
    from src.db.market_cache import cached_closes

    pairs = INSTRUMENTS.forex_pairs()
    tickers = [INSTRUMENTS[p]["ticker"] for p in pairs]
    try:
        close = cached_closes(tickers, period="1y", interval="1d", ttl=600)
        return close if close is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _currency_returns(closes: pd.DataFrame) -> Dict[str, Dict[int, float]]:
    """currency -> {window: avg % return across every pair containing it}."""
    contributions: Dict[str, Dict[int, List[float]]] = {}
    for pair in INSTRUMENTS.forex_pairs():
        ticker = INSTRUMENTS[pair]["ticker"]
        if ticker not in closes.columns:
            continue
        series = closes[ticker].dropna()
        base, quote = pair.split("/")
        for w in _WINDOWS:
            if len(series) <= w:
                continue
            ret = (float(series.iloc[-1]) / float(series.iloc[-1 - w]) - 1) * 100
            contributions.setdefault(base, {}).setdefault(w, []).append(ret)
            contributions.setdefault(quote, {}).setdefault(w, []).append(-ret)

    strength: Dict[str, Dict[int, float]] = {}
    for ccy, windows in contributions.items():
        strength[ccy] = {
            w: (sum(vals) / len(vals) if vals else 0.0) for w, vals in windows.items()
        }
    return strength


def _pair_ideas(strength: Dict[str, Dict[int, float]], primary: int) -> List[dict]:
    """Every registry forex pair ranked by how far apart its two legs' strength
    scores are — the biggest spread is the highest-conviction trend combo."""
    ideas = []
    for pair in INSTRUMENTS.forex_pairs():
        base, quote = pair.split("/")
        base_s = strength.get(base, {}).get(primary, 0.0)
        quote_s = strength.get(quote, {}).get(primary, 0.0)
        diff = base_s - quote_s
        ideas.append({
            "pair": pair, "base": base, "quote": quote,
            "base_strength": base_s, "quote_strength": quote_s,
            "diff": diff, "bias": "Long" if diff >= 0 else "Short",
        })
    ideas.sort(key=lambda d: abs(d["diff"]), reverse=True)
    return ideas


class CurrencyStrengthPage(BloombergPage):
    """Bloomberg-styled currency strength ranking + best-pair ideas."""

    def configure(self) -> PageContext:
        st.session_state.setdefault("ccys_primary", "20 Day")
        return PageContext(code="CCYS", title="Currency Strength", icon="💪")

    def sidebar(self, ctx: PageContext) -> None:
        st.markdown(
            '<div style="color:#00ff41;font-weight:700;letter-spacing:0.15em;'
            'text-transform:uppercase;font-size:11px;">STRENGTH SETTINGS</div>',
            unsafe_allow_html=True,
        )
        keys = list(_PRIMARY_OPTIONS.keys())
        st.session_state.ccys_primary = st.selectbox(
            "Primary window", keys, index=keys.index(st.session_state.ccys_primary),
            help="Ranking, bars, and pair ideas all sort on this window; "
                 "5D/20D/60D are always shown together for context.",
        )
        st.divider()
        if st.button("◆ REFRESH DATA", width="stretch", type="primary"):
            _fetch_pair_closes.clear()
            st.rerun()

    def body(self, ctx: PageContext) -> None:
        primary = _PRIMARY_OPTIONS[st.session_state.ccys_primary]

        with st.spinner("Fetching market data…"):
            closes = _fetch_pair_closes()
        if closes.empty or len(closes) <= max(_WINDOWS):
            st.error("Could not load enough FX data. Try Refresh.")
            return

        strength = _currency_returns(closes)
        if not strength:
            st.error("No currency strength could be computed from loaded data.")
            return

        ranked = sorted(strength.keys(), key=lambda c: strength[c].get(primary, 0.0),
                         reverse=True)
        strongest, weakest = ranked[0], ranked[-1]
        spread = strength[strongest].get(primary, 0.0) - strength[weakest].get(primary, 0.0)

        CommandBar(label="CCYS", cells=[
            (f"WINDOW {st.session_state.ccys_primary.upper()}", ""),
            (f"CCYS {len(ranked)}", "cyan"),
            (f"STRONGEST {strongest}", "green"),
            (f"WEAKEST {weakest}", "red"),
        ]).show()

        render_metric_row([
            MetricCell("Strongest", strongest,
                       "green", delta=f"{strength[strongest].get(primary, 0.0):+.2f}%"),
            MetricCell("Weakest", weakest,
                       "red", delta=f"{strength[weakest].get(primary, 0.0):+.2f}%"),
            MetricCell("Spread", f"{spread:.2f}%", "amber"),
            MetricCell("Pairs Loaded", str(len(closes.columns)), "white"),
        ])

        ideas = _pair_ideas(strength, primary)
        self._persist_top_idea(ideas, closes)

        col_bars, col_ideas = st.columns([3, 2], gap="medium")
        with col_bars:
            with Panel("CURRENCY STRENGTH RANKING", tag=st.session_state.ccys_primary).context():
                st.plotly_chart(self._bar_chart(strength, ranked, primary),
                                width="stretch", config=dict(displayModeBar=False))
        with col_ideas:
            Panel("HIGHEST-PROBABILITY PAIRS", tag="TOP 5",
                  body_html=self._ideas_html(ideas[:5])).show()

        self._render_chip_grid(strength, ranked)

        st.caption(
            "Each currency's score is the average % return, over the selected "
            "window, of every registry pair it appears in (its own leg's sign "
            "flipped when it's the quote currency). The strongest-vs-weakest "
            "spread is the cleanest divergence available; pair ideas rank every "
            "tradable registry pair by how far apart its two legs score — the "
            "biggest gaps are the highest-conviction trend trades, not guaranteed "
            "wins. Commodities (XAU/XAG/XPT) are excluded from the currency "
            "scores since they aren't driven by a single fiat currency."
        )

    # ── persistence ────────────────────────────────────────────────────────
    @staticmethod
    def _persist_top_idea(ideas: List[dict], closes: pd.DataFrame) -> None:
        if not ideas:
            return
        top = ideas[0]
        if abs(top["diff"]) < _IDEA_THRESHOLD:
            return
        ticker = INSTRUMENTS[top["pair"]]["ticker"]
        if ticker not in closes.columns or closes[ticker].dropna().empty:
            return
        entry = float(closes[ticker].dropna().iloc[-1])
        persist_signals("currency_strength", [{
            "pair": top["pair"],
            "bias": top["bias"],
            "entry": entry,
            "conviction": "High" if abs(top["diff"]) >= 2 * _IDEA_THRESHOLD else "Medium",
            "thesis": (f"Currency strength — {top['base']} "
                       f"{top['base_strength']:+.2f}% vs {top['quote']} "
                       f"{top['quote_strength']:+.2f}% (spread {top['diff']:+.2f}%)."),
        }])

    # ── rendering helpers ────────────────────────────────────────────────────
    @staticmethod
    def _bar_chart(strength: Dict[str, Dict[int, float]], ranked: List[str],
                    primary: int) -> go.Figure:
        values = [strength[c].get(primary, 0.0) for c in ranked]
        colors = [T.GREEN if v >= 0 else T.RED for v in values]
        fig = go.Figure(go.Bar(
            x=values, y=ranked, orientation="h",
            marker_color=colors,
            text=[f"{v:+.2f}%" for v in values], textposition="outside",
            textfont=dict(color=T.WHITE, size=11),
        ))
        fig.add_vline(x=0, line_color=T.BORDER, line_width=1)
        fig.update_layout(
            paper_bgcolor=T.BG, plot_bgcolor=T.BG_PANEL,
            font=dict(family=T.FONT_MONO, color=T.WHITE, size=10),
            margin=dict(l=10, r=30, t=10, b=10), height=380,
            showlegend=False,
            xaxis=dict(showgrid=True, gridcolor=T.BORDER,
                       tickfont=dict(color=T.GREY), zeroline=False,
                       title="Avg % return"),
            yaxis=dict(autorange="reversed", tickfont=dict(color=T.WHITE, size=12)),
        )
        return fig

    @staticmethod
    def _ideas_html(ideas: List[dict]) -> str:
        if not ideas:
            return f'<div style="color:{T.GREY};">No pairs loaded.</div>'
        rows = []
        for idea in ideas:
            color = T.GREEN if idea["bias"] == "Long" else T.RED
            bar_w = min(100, int(abs(idea["diff"]) * 25))
            rows.append(
                f'<div style="padding:6px 0;border-bottom:1px dotted {T.BORDER};">'
                f'<div style="display:flex;justify-content:space-between;'
                f'font-family:{T.FONT_MONO};font-size:12px;">'
                f'<b style="color:{T.WHITE};">{idea["pair"]}</b>'
                f'<span style="color:{color};font-weight:700;">'
                f'{idea["bias"].upper()} · {idea["diff"]:+.2f}%</span></div>'
                f'<div style="background:{T.BORDER};height:3px;margin-top:3px;">'
                f'<div style="background:{color};width:{bar_w}%;height:100%;"></div>'
                f'</div></div>'
            )
        return "".join(rows)

    @staticmethod
    def _render_chip_grid(strength: Dict[str, Dict[int, float]], ranked: List[str]) -> None:
        st.markdown(
            '<div style="color:#00ff41;font-family:\'JetBrains Mono\',monospace;'
            'font-size:12px;letter-spacing:0.18em;margin:14px 0 6px 0;">'
            '▸ ALL WINDOWS BY CURRENCY</div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(3)
        for idx, ccy in enumerate(ranked):
            windows = strength[ccy]
            cells = "".join(
                f'<span style="color:{T.GREEN if windows.get(w, 0.0) >= 0 else T.RED};'
                f'margin-right:10px;">{w}D {windows.get(w, 0.0):+.2f}%</span>'
                for w in _WINDOWS
            )
            body = (
                f'<div style="font-family:{T.FONT_MONO};font-size:11px;">'
                f'{cells}</div>'
            )
            with cols[idx % 3]:
                Panel(ccy, body_html=body).show()
