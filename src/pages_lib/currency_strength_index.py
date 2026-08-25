"""Currency Strength Index — per-currency indexes over time + correlation views.

Distinct from the existing Currency Strength page (`currency_strength.py`),
which only ranks a single-point-in-time % return snapshot. This page charts
that same basket-average logic as a running index over the lookback window
(`src.services.currency_index`), and pairs it with a correlation view — most
prominently a stacked Gold-over-DXY comparison chart with the actual rolling
correlation coefficient displayed front and center, since that relationship
("dollar weakens, gold strengthens") is what was explicitly asked for.

Descriptive/audit-only, like `indices-correlation.py` and `disconnect_mon`:
logs to `tool_usage_log`, not wired to `persist_signals` — a correlation read
isn't a directional pair+bias trade call.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.instruments.registry import INSTRUMENTS
from src.pages_lib.base import BloombergPage, PageContext
from src.pages_lib.currency_strength import _fetch_pair_closes
from src.services.alert_service import NotifyCache
from src.services.currency_index import currency_index_series
from src.services.index_analysis import correlation_summary, rolling_correlation
from src.services.tool_log import log_tool_usage
from src.ui.components import MetricCell, Panel, render_metric_row
from src.ui.theme import BloombergTheme as T

_XAU_TICKER = "GC=F"          # INSTRUMENTS["XAU/USD"].ticker
_DXY_TICKER = "DX-Y.NYB"      # not a registry instrument; ad hoc ticker used
                               # the same way correlations.py / cot_signals.py do

# label -> (yfinance period token for the Gold/DXY fetch, trading days to
# window the currency-index chart/heatmap to). "5d" is yfinance's closest
# native period to "last week" (no "1wk" token; the currency-index side
# already deals in trading days, not calendar days).
_PERIOD_OPTIONS: Dict[str, Tuple[str, int]] = {
    "Last Week": ("5d", 5),
    "3 Months": ("3mo", 63),
    "6 Months": ("6mo", 126),
    "1 Year": ("1y", 252),
}

_ALL_CURRENCIES: Tuple[str, ...] = ("USD", "EUR", "GBP", "AUD", "NZD", "JPY", "CHF", "CAD", "ZAR")


@st.cache_data(ttl=600, show_spinner=False)
def _fetch_gold_dxy_closes(period: str) -> pd.DataFrame:
    """Daily Gold + DXY closes, aligned on shared dates."""
    from src.db.market_cache import cached_closes
    try:
        close = cached_closes([_XAU_TICKER, _DXY_TICKER], period=period,
                              interval="1d", ttl=600)
        return close if close is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


class CurrencyStrengthIndexPage(BloombergPage):
    """Bloomberg-styled per-currency index chart + Gold/DXY correlation focus."""

    def configure(self) -> PageContext:
        st.session_state.setdefault("cidx_period", "6 Months")
        st.session_state.setdefault("cidx_currencies", list(_ALL_CURRENCIES))
        st.session_state.setdefault("cidx_rolling", 20)
        return PageContext(code="CIDX", title="Currency Strength Index", icon="📈")

    def sidebar(self, ctx: PageContext) -> None:
        st.markdown(
            '<div style="color:#00ff41;font-weight:700;letter-spacing:0.15em;'
            'text-transform:uppercase;font-size:11px;">INDEX SETTINGS</div>',
            unsafe_allow_html=True,
        )
        keys = list(_PERIOD_OPTIONS.keys())
        st.session_state.cidx_period = st.selectbox(
            "Lookback", keys, index=keys.index(st.session_state.cidx_period),
        )
        st.session_state.cidx_currencies = st.multiselect(
            "Currencies", list(_ALL_CURRENCIES),
            default=st.session_state.cidx_currencies,
        )
        st.session_state.cidx_rolling = st.slider(
            "Rolling correlation window (days)", 10, 60, st.session_state.cidx_rolling,
        )
        st.divider()
        if st.button("◆ REFRESH DATA", width="stretch", type="primary"):
            _fetch_pair_closes.clear()
            _fetch_gold_dxy_closes.clear()
            st.rerun()

    def body(self, ctx: PageContext) -> None:
        period, window_days = _PERIOD_OPTIONS[st.session_state.cidx_period]
        window = st.session_state.cidx_rolling

        with st.spinner("Fetching FX data…"):
            pair_closes = _fetch_pair_closes()
        if pair_closes.empty:
            st.error("Could not load FX data. Try Refresh.")
            return

        # Windowed to the selected Lookback so the index chart and heatmap
        # honor the same period as the Gold/DXY panel, instead of always
        # showing the full underlying 1y fetch regardless of selection.
        index_df = currency_index_series(pair_closes).tail(window_days)
        selected = [c for c in st.session_state.cidx_currencies if c in index_df.columns]
        if not selected:
            st.warning("Select at least one currency in the sidebar.")
            return

        with st.spinner("Fetching Gold/DXY data…"):
            gd_closes = _fetch_gold_dxy_closes(period)
        gold = gd_closes[_XAU_TICKER].dropna() if _XAU_TICKER in gd_closes.columns else pd.Series(dtype=float)
        dxy = gd_closes[_DXY_TICKER].dropna() if _DXY_TICKER in gd_closes.columns else pd.Series(dtype=float)
        has_gold_dxy = not gold.empty and not dxy.empty

        summary = correlation_summary(gold, dxy, windows=(window,)) if has_gold_dxy else {}
        latest_corr = summary.get(f"w{window}")

        render_metric_row([
            MetricCell("Currencies Loaded", str(len(index_df.columns)), "white"),
            MetricCell("Gold vs DXY (full period)",
                       f"{summary.get('full'):+.2f}" if summary.get("full") is not None else "—",
                       "cyan"),
            MetricCell(f"Gold vs DXY ({window}D rolling)",
                       f"{latest_corr:+.2f}" if latest_corr is not None else "—",
                       "green" if (latest_corr or 0) < 0 else "red"),
            MetricCell("Observations", str(summary.get("n", 0)), "amber"),
        ])

        with Panel("CURRENCY STRENGTH INDEX", tag=st.session_state.cidx_period).context():
            st.plotly_chart(self._index_chart(index_df, selected),
                            width="stretch", config=dict(displayModeBar=False))

        st.markdown(
            '<div style="color:#00ff41;font-family:\'JetBrains Mono\',monospace;'
            'font-size:12px;letter-spacing:0.18em;margin:14px 0 6px 0;">'
            '▸ GOLD vs DXY — THE HEADLINE RELATIONSHIP</div>',
            unsafe_allow_html=True,
        )
        if has_gold_dxy:
            with Panel("XAU/USD OVER DXY — SYNCED", tag="STACKED").context():
                st.plotly_chart(self._stacked_gold_dxy_chart(gold, dxy),
                                width="stretch", config=dict(displayModeBar=False))
                rc = rolling_correlation(gold, dxy, window).dropna()
                if not rc.empty:
                    st.plotly_chart(self._rolling_corr_chart(rc, window),
                                    width="stretch", config=dict(displayModeBar=False))
        else:
            st.warning("Could not load Gold/DXY data for the headline panel.")

        with Panel("CORRELATION MATRIX — CURRENCY INDEXES + GOLD + DXY",
                   tag=st.session_state.cidx_period).context():
            st.plotly_chart(self._heatmap(index_df, selected, gold, dxy),
                            width="stretch", config=dict(displayModeBar=False))

        st.caption(
            "Each currency's index tracks the average % return, compounded "
            "daily, of every registry FX pair it appears in (sign-flipped on "
            "the quote leg) — the same basket-average method as the Currency "
            "Strength page, run day-by-day instead of as a single snapshot. "
            "A descriptive/correlation read, not a trade signal on its own."
        )

        self._log_view(selected, period, window, summary)

    # ── data-shape helpers ───────────────────────────────────────────────
    @staticmethod
    def _log_view(selected: List[str], period: str, window: int, summary: dict) -> None:
        key = f"{','.join(sorted(selected))}|{period}|{window}"
        if NotifyCache("currency_strength_index_log").filter_new([key]):
            log_tool_usage("currency_strength_index", {
                "currencies": selected, "period": period, "rolling_window": window,
                "gold_dxy_full_corr": summary.get("full"),
                "gold_dxy_rolling_corr": summary.get(f"w{window}"),
            })

    # ── rendering helpers ────────────────────────────────────────────────
    @staticmethod
    def _index_chart(index_df: pd.DataFrame, selected: List[str]) -> go.Figure:
        palette = [T.CYAN, T.GREEN, T.AMBER, T.RED, T.PURPLE,
                  "#58a6ff", "#7ee787", "#ffa657", "#ff7b72"]
        fig = go.Figure()
        for idx, ccy in enumerate(selected):
            series = index_df[ccy].dropna()
            fig.add_trace(go.Scatter(
                x=series.index, y=series.values, name=ccy, mode="lines",
                line=dict(color=palette[idx % len(palette)], width=1.6),
            ))
        fig.add_hline(y=100, line_dash="dot", line_color=T.BORDER, line_width=1)
        fig.update_layout(
            paper_bgcolor=T.BG, plot_bgcolor=T.BG_PANEL,
            font=dict(family=T.FONT_MONO, color=T.WHITE, size=10),
            margin=dict(l=20, r=20, t=20, b=20), height=420,
            legend=dict(bgcolor=T.BG_PANEL, bordercolor=T.BORDER, borderwidth=1,
                        orientation="h", x=0, y=1.08,
                        font=dict(color=T.GREY, size=10)),
            xaxis=dict(showgrid=False, tickfont=dict(color=T.GREY), linecolor=T.BORDER),
            yaxis=dict(showgrid=True, gridcolor=T.BORDER, tickfont=dict(color=T.GREY),
                      title="Index (base 100)"),
            hovermode="x unified",
        )
        return fig

    @staticmethod
    def _stacked_gold_dxy_chart(gold: pd.Series, dxy: pd.Series) -> go.Figure:
        """XAU/USD on top, DXY directly below, shared x-axis — a TradingView-
        style stacked comparison rather than a dual-y-axis overlay, so the
        inverse relationship reads visually without an axis-scaling illusion."""
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            subplot_titles=("XAU/USD", "DXY"),
        )
        fig.add_trace(go.Scatter(x=gold.index, y=gold.values, name="XAU/USD",
                                 mode="lines", line=dict(color=T.AMBER, width=1.6)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=dxy.index, y=dxy.values, name="DXY",
                                 mode="lines", line=dict(color=T.CYAN, width=1.6)),
                      row=2, col=1)
        fig.update_layout(
            paper_bgcolor=T.BG, plot_bgcolor=T.BG_PANEL,
            font=dict(family=T.FONT_MONO, color=T.WHITE, size=10),
            margin=dict(l=20, r=20, t=30, b=20), height=440, showlegend=False,
            hovermode="x unified",
        )
        fig.update_xaxes(showgrid=False, tickfont=dict(color=T.GREY), linecolor=T.BORDER)
        fig.update_yaxes(showgrid=True, gridcolor=T.BORDER, tickfont=dict(color=T.GREY))
        for ann in fig.layout.annotations:
            ann.font = dict(color=T.WHITE, size=11)
        return fig

    @staticmethod
    def _rolling_corr_chart(rc: pd.Series, window: int) -> go.Figure:
        fig = go.Figure(go.Scatter(
            x=rc.index, y=rc.values, mode="lines",
            line=dict(color=T.GREEN, width=1.6), fill="tozeroy",
            fillcolor="rgba(0,255,65,0.08)",
        ))
        fig.add_hline(y=0, line_dash="dot", line_color=T.GREY, line_width=1)
        fig.update_layout(
            paper_bgcolor=T.BG, plot_bgcolor=T.BG_PANEL,
            font=dict(family=T.FONT_MONO, color=T.WHITE, size=10),
            margin=dict(l=20, r=20, t=10, b=20), height=180,
            title=dict(text=f"{window}D Rolling Correlation — Gold vs DXY",
                      font=dict(color=T.GREY, size=11)),
            xaxis=dict(showgrid=False, tickfont=dict(color=T.GREY), linecolor=T.BORDER),
            yaxis=dict(showgrid=True, gridcolor=T.BORDER, tickfont=dict(color=T.GREY),
                      range=[-1.05, 1.05]),
        )
        return fig

    @staticmethod
    def _heatmap(index_df: pd.DataFrame, selected: List[str],
                 gold: pd.Series, dxy: pd.Series) -> go.Figure:
        panel = index_df[selected].copy()
        if not gold.empty:
            panel["GOLD"] = gold.reindex(panel.index)
        if not dxy.empty:
            panel["DXY"] = dxy.reindex(panel.index)
        returns = panel.pct_change().dropna(how="all")
        corr = returns.corr()

        labels = corr.columns.tolist()
        z = corr.values
        text = [[f"{v:.2f}" if not pd.isna(v) else "" for v in row] for row in z]
        fig = go.Figure(go.Heatmap(
            z=z, x=labels, y=labels, text=text, texttemplate="%{text}",
            textfont=dict(size=9, color="#000"),
            colorscale=[
                [0.0, T.RED], [0.25, "#8b1a1a"],
                [0.45, T.BG_PANEL], [0.50, T.BG],
                [0.55, T.BG_PANEL], [0.75, "#0a8a3a"],
                [1.0, T.GREEN],
            ],
            zmid=0, zmin=-1, zmax=1,
            colorbar=dict(tickvals=[-1, -0.5, 0, 0.5, 1],
                          tickfont=dict(color=T.GREY, size=10),
                          bgcolor=T.BG_PANEL, bordercolor=T.BORDER),
            hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>r = %{z:.3f}<extra></extra>",
        ))
        fig.update_layout(
            paper_bgcolor=T.BG, plot_bgcolor=T.BG_PANEL,
            margin=dict(l=10, r=10, t=10, b=10), height=480,
            font=dict(family=T.FONT_MONO, color=T.WHITE, size=10),
            xaxis=dict(tickfont=dict(color=T.GREY, size=10), tickangle=-45),
            yaxis=dict(tickfont=dict(color=T.GREY, size=10), autorange="reversed"),
        )
        return fig
