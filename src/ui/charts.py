"""Plotly chart factories — Bloomberg-tinted defaults.

`ChartBuilder.trend_panel()` is the OOP replacement for the procedural
`build_trend_chart()` function. It preserves the same subplot layout, colours
and indicator wiring.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.ui.theme import BloombergTheme as T


class ChartBuilder:
    """Stateless factory for Plotly figures used across the app."""

    @staticmethod
    def trend_panel(df: pd.DataFrame, pair: str) -> go.Figure:
        """3-panel candlestick + RSI + MACD chart, Bloomberg-tinted."""
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
            row_heights=[0.58, 0.21, 0.21],
            subplot_titles=[
                f"{pair} — Price with EMAs", "RSI (14)", "MACD (12 / 26 / 9)",
            ],
        )
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name="Price",
            increasing_line_color=T.GREEN, decreasing_line_color=T.RED,
            increasing_fillcolor=T.GREEN, decreasing_fillcolor=T.RED,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["EMA50"], name="50 EMA",
            line=dict(color=T.AMBER, width=2),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["EMA200"], name="200 EMA",
            line=dict(color=T.PURPLE, width=2.5, dash="dash"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI"], name="RSI",
            line=dict(color=T.CYAN, width=1.6),
        ), row=2, col=1)
        for level, color in [(70, T.RED), (30, T.GREEN), (50, T.GREY)]:
            fig.add_hline(y=level, line_dash="dash", line_color=color,
                          opacity=0.5, row=2, col=1)
        hist_colors = [T.GREEN if v >= 0 else T.RED for v in df["MACDHist"]]
        fig.add_trace(go.Bar(
            x=df.index, y=df["MACDHist"], name="Histogram",
            marker_color=hist_colors, opacity=0.7,
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD"], name="MACD",
            line=dict(color=T.CYAN, width=1.6),
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACDSig"], name="Signal",
            line=dict(color=T.AMBER, width=1.6),
        ), row=3, col=1)
        fig.update_layout(
            height=620,
            template="plotly_dark",
            paper_bgcolor=T.BG, plot_bgcolor=T.BG_PANEL,
            font=dict(color=T.WHITE, family=T.FONT_MONO, size=10),
            showlegend=True, xaxis_rangeslider_visible=False,
            margin=dict(l=30, r=20, t=40, b=20),
        )
        fig.update_yaxes(gridcolor=T.BORDER, linecolor=T.BORDER, zerolinecolor=T.BORDER)
        fig.update_xaxes(gridcolor=T.BORDER, linecolor=T.BORDER)
        return fig

    @staticmethod
    def radar(categories: list, values: list) -> go.Figure:
        """Polar/radar chart for confluence breakdown."""
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor=f"rgba(255,153,0,0.15)",
            line=dict(color=T.AMBER, width=2),
            marker=dict(color=T.AMBER, size=4),
        ))
        fig.update_layout(
            polar=dict(
                bgcolor=T.BG_PANEL,
                angularaxis=dict(
                    tickfont=dict(size=9, color=T.GREY, family=T.FONT_MONO),
                    linecolor=T.BORDER, gridcolor=T.BORDER,
                ),
                radialaxis=dict(
                    range=[0, 1], tickvals=[0, .5, 1], ticktext=["", "", ""],
                    gridcolor=T.BORDER, linecolor=T.BORDER,
                ),
            ),
            paper_bgcolor=T.BG, plot_bgcolor=T.BG_PANEL,
            showlegend=False, margin=dict(l=30, r=30, t=10, b=10), height=260,
            font=dict(family=T.FONT_MONO),
        )
        return fig
