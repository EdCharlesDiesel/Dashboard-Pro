"""DXY vs Gold — the single most important macro correlation for gold traders.

Gold is priced in dollars, so dollar strength (DXY up) mechanically pressures
XAU/USD. This page shows the relationship three ways: a rebased overlay, the
rolling correlation, and a plain-language regime read.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from src.pages_lib.base import BloombergPage, PageContext
from src.ui.components import CommandBar, MetricCell, Panel, render_metric_row
from src.ui.theme import BloombergTheme as T

DXY_TICKER = "DX-Y.NYB"
GOLD_TICKER = "GC=F"

_LOOKBACKS = {
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
}


@st.cache_data(ttl=600, show_spinner=False)
def _fetch_closes(period: str) -> pd.DataFrame:
    """Daily closes for DXY and Gold, aligned on shared sessions."""
    try:
        raw = yf.download([DXY_TICKER, GOLD_TICKER], period=period,
                          interval="1d", progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            return pd.DataFrame()
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        close = close.rename(columns={DXY_TICKER: "DXY", GOLD_TICKER: "XAU/USD"})
        return close[["DXY", "XAU/USD"]].dropna()
    except Exception:
        return pd.DataFrame()


class DxyGoldPage(BloombergPage):
    """Clean two-asset correlation monitor: DXY vs XAU/USD."""

    def configure(self) -> PageContext:
        st.session_state.setdefault("dxg_lookback", "6 Months")
        st.session_state.setdefault("dxg_window", 20)
        return PageContext(code="DXAU", title="DXY vs Gold", icon="💵")

    def sidebar(self, ctx: PageContext) -> None:
        st.markdown(
            '<div style="color:#00ff41;font-weight:700;letter-spacing:0.15em;'
            'text-transform:uppercase;font-size:11px;">DXY · XAU/USD</div>',
            unsafe_allow_html=True,
        )
        keys = list(_LOOKBACKS.keys())
        st.session_state.dxg_lookback = st.selectbox(
            "Lookback", keys, index=keys.index(st.session_state.dxg_lookback),
        )
        st.session_state.dxg_window = st.slider(
            "Rolling window (days)", 10, 60, st.session_state.dxg_window,
            help="20 days ≈ one trading month — the standard regime window.",
        )
        st.divider()
        if st.button("◆ REFRESH DATA", width="stretch", type="primary"):
            _fetch_closes.clear()
            st.rerun()

    def body(self, ctx: PageContext) -> None:
        lookback = st.session_state.dxg_lookback
        window = st.session_state.dxg_window

        closes = _fetch_closes(_LOOKBACKS[lookback])
        if closes.empty or len(closes) < window + 5:
            st.error("Could not load enough DXY / Gold data. Try Refresh.")
            return

        returns = closes.pct_change().dropna()
        period_corr = float(returns["DXY"].corr(returns["XAU/USD"]))
        rolling = returns["DXY"].rolling(window).corr(returns["XAU/USD"]).dropna()
        current_corr = float(rolling.iloc[-1])

        dxy_last = float(closes["DXY"].iloc[-1])
        xau_last = float(closes["XAU/USD"].iloc[-1])
        dxy_chg = (dxy_last / float(closes["DXY"].iloc[-2]) - 1) * 100
        xau_chg = (xau_last / float(closes["XAU/USD"].iloc[-2]) - 1) * 100

        CommandBar(label="DXAU", cells=[
            (f"LOOK {lookback.upper()}", ""),
            (f"ROLL {window}D", "cyan"),
            (f"CORR {current_corr:+.2f}", "red" if current_corr <= -0.4 else "green"),
            (datetime.now().strftime("%a %d %b %H:%M"), "green"),
        ]).show()

        regime, regime_color, regime_read = self._regime(current_corr)

        render_metric_row([
            MetricCell("DXY", f"{dxy_last:,.2f}",
                       "green" if dxy_chg >= 0 else "red", delta=f"{dxy_chg:+.2f}%"),
            MetricCell("XAU/USD", f"{xau_last:,.2f}",
                       "green" if xau_chg >= 0 else "red", delta=f"{xau_chg:+.2f}%"),
            MetricCell(f"Corr ({lookback})", f"{period_corr:+.2f}", "white"),
            MetricCell(f"Corr ({window}d now)", f"{current_corr:+.2f}", regime_color),
        ])

        # Who's stronger + what to do about it
        sig = self._strength_signal(closes)
        col_sig, col_regime = st.columns([3, 2], gap="medium")
        with col_sig:
            Panel("WHO'S STRONGER — TRADE SIGNAL", tag=sig["stronger"],
                  body_html=self._strength_html(sig)).show()
        with col_regime:
            Panel("REGIME READ", tag=regime, body_html=(
                f'<div style="font-family:{T.FONT_MONO};font-size:12px;'
                f'color:{T.WHITE};">{regime_read}</div>'
            )).show()

        col_overlay, col_roll = st.columns([3, 2], gap="medium")
        with col_overlay:
            with Panel(f"REBASED OVERLAY · {lookback}", tag="BASE 100").context():
                st.plotly_chart(self._overlay_chart(closes),
                                width="stretch",
                                config=dict(displayModeBar=False))
        with col_roll:
            with Panel(f"ROLLING {window}D CORRELATION", tag="DAILY RETURNS").context():
                st.plotly_chart(self._rolling_chart(rolling),
                                width="stretch",
                                config=dict(displayModeBar=False))

        st.caption(
            "Gold is dollar-priced, so the structural relationship is inverse "
            "(correlation below −0.4 most of the time). When the rolling "
            "correlation climbs toward 0 or positive, the usual driver is a "
            "risk event or real-yield shock — DXY stops being a reliable "
            "filter for gold entries until the inverse regime returns."
        )

    # ── helpers ────────────────────────────────────────────────────────────
    @staticmethod
    def _trend_score(series: pd.Series) -> dict:
        """0–4 bull score: positive 5d/20d/60d returns + price above 20-EMA."""
        rets = {}
        for w in (5, 20, 60):
            if len(series) > w:
                rets[w] = (float(series.iloc[-1]) / float(series.iloc[-1 - w]) - 1) * 100
            else:
                rets[w] = 0.0
        ema20 = float(series.ewm(span=20, adjust=False).mean().iloc[-1])
        score = sum(r > 0 for r in rets.values()) + (float(series.iloc[-1]) > ema20)
        state = "BULL" if score >= 3 else "BEAR" if score <= 1 else "MIXED"
        return {"rets": rets, "score": score, "state": state}

    @classmethod
    def _strength_signal(cls, closes: pd.DataFrame) -> dict:
        gold = cls._trend_score(closes["XAU/USD"])
        dxy = cls._trend_score(closes["DXY"])
        g, d = gold["state"], dxy["state"]
        if g == "BULL" and d == "BEAR":
            action, color, why = "BUY XAU/USD — STRONG", T.GREEN, (
                "Gold trending up while the dollar trends down — the cleanest "
                "tailwind gold gets. Dollar weakness is doing the heavy lifting.")
        elif g == "BEAR" and d == "BULL":
            action, color, why = "SELL XAU/USD — STRONG", T.RED, (
                "Dollar trending up while gold trends down — textbook pressure "
                "on gold. Favor shorts / stand aside on longs.")
        elif g == "BULL" and d == "BULL":
            action, color, why = "BUY XAU/USD — CAUTION", T.YELLOW, (
                "Gold is rising despite a strong dollar — intrinsic demand "
                "(risk-off / real yields). Valid long, but trade smaller: the "
                "dollar is a headwind, not a tailwind.")
        elif g == "BEAR" and d == "BEAR":
            action, color, why = "SELL XAU/USD — CAUTION", T.YELLOW, (
                "Gold is falling even with a weak dollar — that is real gold "
                "weakness. Shorts have the edge, but a dollar bounce is the risk.")
        else:
            action, color, why = "NO CLEAR SIDE — WAIT", T.GREY, (
                "Neither asset has a clean trend. No edge from this pair of "
                "drivers right now — let one of them pick a direction first.")
        if gold["score"] != dxy["score"]:
            stronger = "GOLD" if gold["score"] > dxy["score"] else "DOLLAR"
        else:
            # tie on trend score — let 20-day momentum decide (gold runs ~3x
            # DXY's volatility, so scale DXY up before comparing)
            stronger = ("GOLD" if gold["rets"][20] > dxy["rets"][20] * 3
                        else "DOLLAR")
        return {"gold": gold, "dxy": dxy, "action": action,
                "color": color, "why": why, "stronger": stronger}

    @staticmethod
    def _strength_html(sig: dict) -> str:
        def chips(label, ts):
            cells = "".join(
                f'<span style="color:{T.GREEN if r > 0 else T.RED};'
                f'margin-right:10px;">{w}D {r:+.1f}%</span>'
                for w, r in ts["rets"].items()
            )
            state_color = (T.GREEN if ts["state"] == "BULL"
                           else T.RED if ts["state"] == "BEAR" else T.GREY)
            return (
                f'<div style="margin:2px 0;"><b style="color:{T.WHITE};'
                f'display:inline-block;width:74px;">{label}</b>'
                f'<span style="color:{state_color};font-weight:700;'
                f'display:inline-block;width:54px;">{ts["state"]}</span>'
                f'{cells}</div>'
            )

        return (
            f'<div style="font-family:{T.FONT_MONO};font-size:11px;">'
            f'<div style="font-size:16px;font-weight:700;color:{sig["color"]};'
            f'letter-spacing:0.08em;margin-bottom:6px;">▸ {sig["action"]}</div>'
            + chips("XAU/USD", sig["gold"])
            + chips("DXY", sig["dxy"])
            + f'<div style="color:{T.GREY};margin-top:6px;">{sig["why"]}</div>'
            + '</div>'
        )

    @staticmethod
    def _regime(corr: float):
        if corr <= -0.7:
            return ("INVERSE — STRONG", "green",
                    "Textbook regime. DXY direction is a high-quality filter: "
                    "DXY breaking down supports gold longs; DXY rallying "
                    "argues against them.")
        if corr <= -0.4:
            return ("INVERSE — NORMAL", "green",
                    "The usual inverse relationship holds. Use DXY as a "
                    "confirming filter for XAU/USD entries, not as the "
                    "primary signal.")
        if corr < 0.2:
            return ("DECOUPLED", "yellow",
                    "Correlation has gone slack — gold is trading on its own "
                    "drivers (real yields, risk-off flows, central-bank "
                    "buying). Do not lean on DXY for gold entries right now.")
        return ("POSITIVE — UNUSUAL", "red",
                "Gold and the dollar are rising/falling together — typically "
                "a flight-to-safety or real-yield shock. Historically "
                "short-lived: expect a snap back to inverse. Trade smaller.")

    @staticmethod
    def _overlay_chart(closes: pd.DataFrame) -> go.Figure:
        base = closes / closes.iloc[0] * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=base.index, y=base["XAU/USD"], name="XAU/USD",
            line=dict(color=T.AMBER, width=1.8),
        ))
        fig.add_trace(go.Scatter(
            x=base.index, y=base["DXY"], name="DXY",
            line=dict(color=T.CYAN, width=1.8),
        ))
        fig.add_hline(y=100, line_dash="dot", line_color=T.GREY, line_width=1)
        fig.update_layout(
            paper_bgcolor=T.BG, plot_bgcolor=T.BG_PANEL,
            font=dict(family=T.FONT_MONO, color=T.WHITE, size=10),
            margin=dict(l=20, r=20, t=20, b=20), height=380,
            legend=dict(orientation="h", y=1.05, x=0,
                        bgcolor="rgba(0,0,0,0)", font=dict(color=T.GREY)),
            xaxis=dict(showgrid=False, linecolor=T.BORDER,
                       tickfont=dict(color=T.GREY)),
            yaxis=dict(showgrid=True, gridcolor=T.BORDER,
                       tickfont=dict(color=T.GREY), title="Rebased = 100"),
            hovermode="x unified",
        )
        return fig

    @staticmethod
    def _rolling_chart(rolling: pd.Series) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rolling.index, y=rolling.values, name="corr",
            line=dict(color=T.AMBER, width=1.6),
            fill="tozeroy", fillcolor="rgba(0,255,65,0.08)",
        ))
        fig.add_hline(y=0, line_dash="dot", line_color=T.GREY, line_width=1)
        fig.add_hline(y=-0.7, line_dash="dash", line_color=T.RED, line_width=1,
                      annotation_text="strong inverse",
                      annotation_font_color=T.RED)
        fig.update_layout(
            paper_bgcolor=T.BG, plot_bgcolor=T.BG_PANEL,
            font=dict(family=T.FONT_MONO, color=T.WHITE, size=10),
            margin=dict(l=20, r=20, t=20, b=20), height=380,
            showlegend=False,
            xaxis=dict(showgrid=False, linecolor=T.BORDER,
                       tickfont=dict(color=T.GREY)),
            yaxis=dict(showgrid=True, gridcolor=T.BORDER,
                       tickfont=dict(color=T.GREY), range=[-1.05, 1.05]),
        )
        return fig
