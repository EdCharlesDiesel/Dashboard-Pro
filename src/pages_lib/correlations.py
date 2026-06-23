"""Correlations page — Bloomberg-terminal version.

All correlation math / KNOWN_RELATIONSHIPS commentary preserved unchanged.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from src.pages_lib.base import BloombergPage, PageContext
from src.ui.components import CommandBar, MetricCell, Panel, render_metric_row
from src.ui.theme import BloombergTheme as T


_ALL_INSTRUMENTS: Dict[str, str] = {
    "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "AUD/USD": "AUDUSD=X",
    "NZD/USD": "NZDUSD=X", "USD/JPY": "USDJPY=X", "USD/CHF": "USDCHF=X",
    "USD/CAD": "USDCAD=X", "EUR/GBP": "EURGBP=X", "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X", "AUD/JPY": "AUDJPY=X", "EUR/AUD": "EURAUD=X",
    "GBP/AUD": "GBPAUD=X", "EUR/CAD": "EURCAD=X", "GBP/CAD": "GBPCAD=X",
    "USD/ZAR": "USDZAR=X", "EUR/ZAR": "EURZAR=X", "GBP/ZAR": "GBPZAR=X",
    "XAU/USD": "GC=F", "XAG/USD": "SI=F", "XPT/USD": "PL=F",
    "DXY": "DX-Y.NYB", "S&P 500": "^GSPC",
}

_KNOWN_RELATIONSHIPS: List[Tuple[str, str, str, str]] = [
    ("EUR/USD", "GBP/USD", "Positive", "Both inversely correlated to DXY."),
    ("EUR/USD", "USD/CHF", "Negative", "Strong inverse — CHF pairs mirror EUR/USD."),
    ("EUR/USD", "USD/JPY", "Negative", "Risk-on dollar pairs diverge from EUR."),
    ("EUR/USD", "XAU/USD", "Positive", "Both USD-priced — weak dollar lifts both."),
    ("GBP/USD", "GBP/JPY", "Positive", "GBP leg drives both; JPY adds vol."),
    ("AUD/USD", "XAU/USD", "Positive", "AU is major gold exporter — AUD tracks gold."),
    ("AUD/USD", "NZD/USD", "Positive", "Commodity-currency bloc."),
    ("USD/CAD", "XAU/USD", "Negative", "CAD with commodities; Gold USD = inverse."),
    ("USD/JPY", "EUR/JPY", "Positive", "JPY leg dominates in risk-off."),
    ("USD/ZAR", "XAU/USD", "Negative", "ZAR strengthens with Gold."),
    ("EUR/ZAR", "USD/ZAR", "Positive", "Both ZAR pairs."),
    ("GBP/JPY", "AUD/JPY", "Positive", "JPY crosses move together."),
    ("DXY",  "EUR/USD", "Negative", "EUR is 57% of DXY — almost perfect inverse."),
    ("DXY",  "XAU/USD",  "Negative", "Gold USD-priced; USD strength suppresses Gold."),
    ("DXY",  "AUD/USD", "Negative", "Commodity currencies weaken with USD."),
    ("S&P 500", "AUD/USD", "Positive", "Risk-on lifts equities + commod ccys."),
    ("S&P 500", "USD/JPY", "Positive", "JPY weakens in risk-on."),
    ("S&P 500", "XAU/USD", "Negative", "Gold competes with equities."),
]


def _corr_color(val) -> Tuple[str, str]:
    """Return (color, label) for a correlation badge."""
    if pd.isna(val):
        return T.GREY, "Neutral"
    if val >= 0.70:  return T.GREEN,  "Strong +"
    if val >= 0.40:  return T.GREEN,  "Moderate +"
    if val <= -0.70: return T.RED,    "Strong −"
    if val <= -0.40: return T.RED,    "Moderate −"
    return T.GREY, "Neutral"


class _CorrelationFetcher:
    """Cached price-return loader."""

    @staticmethod
    @st.cache_data(ttl=600, show_spinner=False)
    def fetch(tickers: Tuple[str, ...], period: str, interval: str):
        from src.db.market_cache import cached_closes
        try:
            close = cached_closes(list(tickers), period=period, interval=interval, ttl=600)
            if close is None or close.empty:
                return None
            return close.pct_change().dropna()
        except Exception:
            return None


class CorrelationsPage(BloombergPage):
    """Bloomberg-styled correlation matrix + rolling correlation analysis."""

    def configure(self) -> PageContext:
        st.session_state.setdefault("corr_period", "3 Months")
        st.session_state.setdefault("corr_method", "Pearson")
        st.session_state.setdefault("corr_rolling", 20)
        st.session_state.setdefault("corr_selected", list(_ALL_INSTRUMENTS.keys()))
        st.session_state.setdefault("corr_focus", "EUR/USD")
        return PageContext(code="CORR", title="Correlations", icon="🔗")

    def sidebar(self, ctx: PageContext) -> None:
        st.markdown("""
               <style>
                   span[data-baseweb="tag"]{
                       background-color:#00ff41 !important;
                   }
                   span[data-baseweb="tag"] span{
                       color:#000000 !important;
                       font-weight:600 !important;
                   }
                   span[data-baseweb="tag"] svg{
                       fill:#000000 !important;
                   }
                   span[data-baseweb="tag"] [role="button"]:hover{
                       background-color:#00cc34 !important;
                   }
               </style>
               """, unsafe_allow_html=True)
        st.markdown(
            '<div style="color:#00ff41;font-weight:700;letter-spacing:0.15em;'
            'text-transform:uppercase;font-size:11px;">CORR SETTINGS</div>',
            unsafe_allow_html=True,
        )
        period_map = {
            "1 Month":  ("1mo", "1d"), "3 Months": ("3mo", "1d"),
            "6 Months": ("6mo", "1d"), "1 Year":   ("1y",  "1d"),
        }
        keys = list(period_map.keys())
        st.session_state.corr_period = st.selectbox(
            "Lookback", keys, index=keys.index(st.session_state.corr_period),
        )
        st.session_state.corr_selected = st.multiselect(
            "Instruments", list(_ALL_INSTRUMENTS.keys()),
            default=st.session_state.corr_selected,
        )
        st.session_state.corr_method = st.radio(
            "Method", ["Pearson", "Spearman"], horizontal=True,
            index=0 if st.session_state.corr_method == "Pearson" else 1,
        )
        st.session_state.corr_rolling = st.slider(
            "Rolling Window (days)", 10, 60, st.session_state.corr_rolling,
        )
        focus_options = (st.session_state.corr_selected
                          if st.session_state.corr_selected else ["EUR/USD"])
        if st.session_state.corr_focus not in focus_options:
            st.session_state.corr_focus = focus_options[0]
        st.session_state.corr_focus = st.selectbox(
            "Focus pair (rolling)", focus_options,
            index=focus_options.index(st.session_state.corr_focus),
        )
        st.divider()
        if st.button("◆ REFRESH DATA", width="stretch", type="primary"):
            st.cache_data.clear()
            st.rerun()

    def body(self, ctx: PageContext) -> None:
        period_map = {
            "1 Month":  ("1mo", "1d"), "3 Months": ("3mo", "1d"),
            "6 Months": ("6mo", "1d"), "1 Year":   ("1y",  "1d"),
        }
        yf_period, yf_interval = period_map[st.session_state.corr_period]
        selected = st.session_state.corr_selected
        if not selected:
            st.warning("Select at least 2 instruments in the sidebar.")
            return

        CommandBar(label="CORR", cells=[
            (f"LOOK {st.session_state.corr_period.upper()}", ""),
            (f"BARS {yf_interval}", "cyan"),
            (f"METHOD {st.session_state.corr_method.upper()}", "amber"),
            (f"ROLL {st.session_state.corr_rolling}d", "green"),
        ]).show()

        tickers = tuple(_ALL_INSTRUMENTS[p] for p in selected)
        with st.spinner("Fetching market data…"):
            returns = _CorrelationFetcher.fetch(tickers, yf_period, yf_interval)
        if returns is None or returns.empty:
            st.error("Could not fetch price data.")
            return

        ticker_to_pair = {v: k for k, v in _ALL_INSTRUMENTS.items()}
        returns.columns = [ticker_to_pair.get(c, c) for c in returns.columns]
        available = [p for p in selected if p in returns.columns]
        returns = returns[available]
        if len(available) < 2:
            st.error("Not enough data loaded.")
            return

        method = st.session_state.corr_method.lower()
        corr_matrix = returns.corr(method=method)
        flat = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        ).stack().dropna()

        avg_c = flat.mean() if not flat.empty else 0
        max_c = flat.max() if not flat.empty else 0
        min_c = flat.min() if not flat.empty else 0
        strong_pos = int((flat >= 0.7).sum())
        strong_neg = int((flat <= -0.7).sum())

        render_metric_row([
            MetricCell("Pairs Loaded",   str(len(available)), "white"),
            MetricCell("Avg Correlation", f"{avg_c:+.2f}",    "amber"),
            MetricCell("Strongest +",    f"{max_c:+.2f}",     "green"),
            MetricCell("Strongest −",    f"{min_c:+.2f}",     "red"),
            MetricCell("Strong + Pairs", str(strong_pos),     "green"),
            MetricCell("Strong − Pairs", str(strong_neg),     "red"),
        ])

        col_heat, col_top = st.columns([3, 2], gap="medium")
        with col_heat:
            self._render_heatmap(corr_matrix)
        with col_top:
            self._render_top_lists(flat)

        self._render_rolling(returns, available)
        self._render_known_insights(corr_matrix)
        self._render_table(flat)

    @staticmethod
    def _render_heatmap(corr_matrix: pd.DataFrame) -> None:
        labels = corr_matrix.columns.tolist()
        z = corr_matrix.values
        text = [[f"{v:.2f}" for v in row] for row in z]
        fig = go.Figure(go.Heatmap(
            z=z, x=labels, y=labels, text=text, texttemplate="%{text}",
            textfont=dict(size=9, color="#000"),
            colorscale=[
                [0.0,  T.RED], [0.25, "#8b1a1a"],
                [0.45, T.BG_PANEL], [0.50, T.BG],
                [0.55, T.BG_PANEL], [0.75, "#0a8a3a"],
                [1.0,  T.GREEN],
            ],
            zmid=0, zmin=-1, zmax=1,
            colorbar=dict(
                tickvals=[-1, -0.5, 0, 0.5, 1],
                tickfont=dict(color=T.GREY, size=10),
                bgcolor=T.BG_PANEL, bordercolor=T.BORDER,
            ),
            hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>r = %{z:.3f}<extra></extra>",
        ))
        fig.update_layout(
            paper_bgcolor=T.BG, plot_bgcolor=T.BG_PANEL,
            margin=dict(l=10, r=10, t=10, b=10), height=520,
            font=dict(family=T.FONT_MONO, color=T.WHITE, size=10),
            xaxis=dict(tickfont=dict(color=T.GREY, size=10), tickangle=-45),
            yaxis=dict(tickfont=dict(color=T.GREY, size=10), autorange="reversed"),
        )
        with Panel("CORRELATION HEATMAP", tag="LIVE").context():
            st.plotly_chart(fig, width="stretch",
                            config=dict(displayModeBar=False))

    @staticmethod
    def _render_top_lists(flat: pd.Series) -> None:
        pair_corrs = flat.reset_index()
        pair_corrs.columns = ["Pair A", "Pair B", "Correlation"]
        pair_corrs = pair_corrs.dropna(subset=["Correlation"])
        pair_corrs = pair_corrs.sort_values("Correlation", ascending=False)

        def _list_html(rows: pd.DataFrame, neg: bool = False) -> str:
            html_parts = []
            for _, r in rows.iterrows():
                color, label = _corr_color(r["Correlation"])
                bar_w = int(abs(r["Correlation"]) * 100)
                html_parts.append(
                    f'<div style="padding:4px 0;border-bottom:1px dotted {T.BORDER};">'
                    f'<div style="display:flex;justify-content:space-between;'
                    f'font-family:\'JetBrains Mono\',monospace;font-size:11px;">'
                    f'<span><b style="color:{T.WHITE};">{r["Pair A"]}</b>'
                    f'<span style="color:{T.GREY};"> ╱ </span>'
                    f'<b style="color:{T.WHITE};">{r["Pair B"]}</b></span>'
                    f'<span style="color:{color};font-weight:700;">{r["Correlation"]:+.2f}</span>'
                    f'</div>'
                    f'<div style="background:{T.BORDER};height:3px;margin-top:2px;">'
                    f'<div style="background:{color};width:{bar_w}%;height:100%;"></div>'
                    f'</div></div>'
                )
            return "".join(html_parts)

        Panel("STRONGEST CORRELATIONS", tag="TOP 8",
              body_html=_list_html(pair_corrs.head(8))).show()
        Panel("MOST NEGATIVE", tag="BOT 8",
              body_html=_list_html(pair_corrs.tail(8).iloc[::-1], neg=True)).show()

    def _render_rolling(self, returns: pd.DataFrame, available: List[str]) -> None:
        focus = st.session_state.corr_focus
        window = st.session_state.corr_rolling
        if focus not in returns.columns:
            st.info(f"{focus} not in loaded data.")
            return
        others = [p for p in available if p != focus]
        fig = go.Figure()
        palette = [T.CYAN, T.GREEN, T.AMBER, T.RED, T.PURPLE,
                    "#58a6ff", "#7ee787", "#ffa657", "#ff7b72", "#d2a8ff"]
        for idx, pair in enumerate(others):
            rc = returns[focus].rolling(window).corr(returns[pair]).dropna()
            if rc.empty:
                continue
            fig.add_trace(go.Scatter(
                x=rc.index, y=rc.values, name=pair, mode="lines",
                line=dict(color=palette[idx % len(palette)], width=1.4),
            ))
        fig.add_hline(y=0,    line_dash="dot",  line_color=T.GREY,  line_width=1)
        fig.add_hline(y=0.7,  line_dash="dash", line_color=T.GREEN, line_width=1)
        fig.add_hline(y=-0.7, line_dash="dash", line_color=T.RED,   line_width=1)
        fig.update_layout(
            paper_bgcolor=T.BG, plot_bgcolor=T.BG_PANEL,
            font=dict(family=T.FONT_MONO, color=T.WHITE, size=10),
            margin=dict(l=20, r=20, t=20, b=20), height=380,
            legend=dict(bgcolor=T.BG_PANEL, bordercolor=T.BORDER, borderwidth=1,
                        orientation="v", x=1.02, y=1,
                        font=dict(color=T.GREY, size=9)),
            xaxis=dict(showgrid=False, tickfont=dict(color=T.GREY),
                       linecolor=T.BORDER),
            yaxis=dict(showgrid=True, gridcolor=T.BORDER,
                       tickfont=dict(color=T.GREY), range=[-1.05, 1.05]),
            hovermode="x unified",
        )
        with Panel(f"ROLLING {window}D CORRELATION · {focus} vs ALL").context():
            st.plotly_chart(fig, width="stretch",
                            config=dict(displayModeBar=False))

    @staticmethod
    def _render_known_insights(corr_matrix: pd.DataFrame) -> None:
        st.markdown(
            '<div style="color:#00ff41;font-family:\'JetBrains Mono\',monospace;'
            'font-size:12px;letter-spacing:0.18em;margin:14px 0 6px 0;">'
            '▸ KEY RELATIONSHIPS</div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(2)
        for idx, (pa, pb, rel_type, desc) in enumerate(_KNOWN_RELATIONSHIPS):
            live_corr = None
            if pa in corr_matrix.columns and pb in corr_matrix.columns:
                live_corr = corr_matrix.loc[pa, pb]
            color, _ = _corr_color(live_corr) if live_corr is not None else (T.GREY, "")
            live_html = (
                f'<span style="color:{color};font-weight:700;">'
                f'{live_corr:+.2f}</span>'
                if live_corr is not None and not pd.isna(live_corr)
                else f'<span style="color:{T.GREY};">N/A</span>'
            )
            rel_color = T.GREEN if rel_type == "Positive" else T.RED
            body = (
                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:flex-start;font-family:\'JetBrains Mono\',monospace;'
                f'font-size:11px;">'
                f'<span><b style="color:{T.WHITE};">{pa}</b>'
                f'<span style="color:{T.GREY};"> vs </span>'
                f'<b style="color:{T.WHITE};">{pb}</b></span>'
                f'<span><span style="color:{rel_color};font-weight:700;">'
                f'{rel_type.upper()}</span> {live_html}</span></div>'
                f'<div style="font-size:11px;color:{T.GREY};margin-top:4px;">'
                f'{desc}</div>'
            )
            with cols[idx % 2]:
                Panel(f"{pa} ╱ {pb}", body_html=body).show()

    @staticmethod
    def _render_table(flat: pd.Series) -> None:
        pair_corrs = flat.reset_index()
        pair_corrs.columns = ["Pair A", "Pair B", "Correlation"]
        pair_corrs = pair_corrs.dropna(subset=["Correlation"])
        pair_corrs["Correlation"] = pair_corrs["Correlation"].round(4)
        pair_corrs = pair_corrs.sort_values("Correlation", ascending=False)
        pair_corrs.index = range(1, len(pair_corrs) + 1)
        with Panel("FULL CORRELATION TABLE", tag="ALL PAIRS").context():
            st.dataframe(pair_corrs, width="stretch")
