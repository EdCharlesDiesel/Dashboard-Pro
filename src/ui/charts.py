"""Plotly chart factories — Bloomberg-tinted defaults.

Two layers:

* **ChartKit** — the chart *stencil*: small composable primitives
  (``panels`` → add traces → ``finish``) that give every figure in the app
  the identical terminal look from one place. New charts must be built from
  these; legacy hand-rolled figures migrate opportunistically. Pattern::

      fig = ChartKit.panels(["EUR/USD — Daily", "RSI (14)"], heights=[0.7, 0.3])
      ChartKit.candles(fig, df)
      ChartKit.line(fig, df.index, rsi, "RSI", color=T.CYAN, row=2)
      ChartKit.hline(fig, 70, T.RED, row=2)
      st.plotly_chart(ChartKit.finish(fig, height=560), width="stretch",
                      config=ChartKit.PLOTLY_CONFIG)

* **ChartBuilder** — page-specific composites (``trend_panel``, ``radar``)
  preserved byte-for-byte for their existing call sites.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.ui.theme import BloombergTheme as T
from src.core.config import CANDLE_STYLE


class ChartKit:
    """The chart stencil — one look for every figure, composed from parts."""

    PLOTLY_CONFIG = dict(displayModeBar=False)

    @staticmethod
    def panels(titles: Sequence[str], heights: Optional[Sequence[float]] = None,
               vertical_spacing: float = 0.04) -> go.Figure:
        """Stacked subplot rows sharing the x axis — the house skeleton."""
        n = len(titles)
        return make_subplots(
            rows=n, cols=1, shared_xaxes=True,
            vertical_spacing=vertical_spacing,
            row_heights=list(heights) if heights else None,
            subplot_titles=list(titles),
        )

    @staticmethod
    def candles(fig: go.Figure, df: pd.DataFrame, row: int = 1,
                name: str = "Price") -> None:
        """Candlestick trace in the canonical CANDLE_STYLE colours."""
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name=name,
            showlegend=False, **CANDLE_STYLE,
        ), row=row, col=1)

    @staticmethod
    def line(fig: go.Figure, x, y, name: str, color: Optional[str] = None,
             row: int = 1, width: float = 1.6, dash: Optional[str] = None,
             showlegend: bool = False) -> None:
        fig.add_trace(go.Scatter(
            x=x, y=y, name=name, mode="lines",
            line=dict(color=color or T.AMBER, width=width, dash=dash),
            showlegend=showlegend,
        ), row=row, col=1)

    @staticmethod
    def bars(fig: go.Figure, x, y, row: int = 1, name: str = "",
             colors: Optional[Sequence[str]] = None) -> None:
        """Bar trace; defaults to signed green/red colouring (histograms, flow)."""
        if colors is None:
            colors = [T.GREEN if (v or 0) >= 0 else T.RED for v in y]
        fig.add_trace(go.Bar(x=x, y=y, name=name, marker_color=list(colors),
                             showlegend=False), row=row, col=1)

    @staticmethod
    def band(fig: go.Figure, x, hi, lo, fillcolor: str, row: int = 1,
             name: str = "") -> None:
        """Shaded hi/lo band (volatility cones, deviation envelopes)."""
        fig.add_trace(go.Scatter(x=x, y=hi, mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"),
                      row=row, col=1)
        fig.add_trace(go.Scatter(x=x, y=lo, mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=fillcolor, name=name,
                                 showlegend=False, hoverinfo="skip"),
                      row=row, col=1)

    @staticmethod
    def hline(fig: go.Figure, y: float, color: str, row: int = 1,
              dash: str = "dot", width: float = 1,
              label: Optional[str] = None) -> None:
        """Horizontal level. ⚠ Plotly resolves ``row=`` targets from existing
        traces — call this AFTER adding at least one trace to that row, or the
        line is silently dropped (same applies to :meth:`levels`)."""
        kwargs = dict(y=y, line_dash=dash, line_color=color, line_width=width,
                      row=row, col=1)
        if label:
            kwargs.update(annotation_text=label, annotation_position="right",
                          annotation_font_color=color)
        fig.add_hline(**kwargs)

    @staticmethod
    def levels(fig: go.Figure, levels: Dict[str, float],
               colors: Optional[Dict[str, str]] = None, row: int = 1) -> None:
        """Named horizontal levels (pivots, fibs, zones) with right-edge labels."""
        colors = colors or {}
        for name, value in levels.items():
            ChartKit.hline(fig, value, colors.get(name, T.GREY),
                           row=row, label=name)

    @staticmethod
    def finish(fig: go.Figure, height: int = 620,
               showlegend: bool = False) -> go.Figure:
        """Apply the house layout — call once, last. Same tokens everywhere."""
        fig.update_layout(
            height=height,
            paper_bgcolor=T.BG, plot_bgcolor=T.BG_PANEL,
            font=dict(color=T.WHITE, family=T.FONT_MONO, size=10),
            showlegend=showlegend, xaxis_rangeslider_visible=False,
            margin=dict(l=30, r=60, t=40, b=20),
            hovermode="x unified",
            legend=dict(bgcolor=T.BG, bordercolor=T.AMBER, borderwidth=1,
                        font=dict(color=T.AMBER, family=T.FONT_MONO, size=10)),
            hoverlabel=dict(bgcolor=T.BG, bordercolor=T.AMBER,
                            font=dict(color=T.AMBER, family=T.FONT_MONO, size=11)),
        )
        for ann in fig.layout.annotations:
            if ann.font.color is None:      # subplot titles only, not hline labels
                ann.font.color = T.AMBER
                ann.font.size = 11
        fig.update_yaxes(gridcolor=T.BORDER, linecolor=T.BORDER,
                         zerolinecolor=T.BORDER,
                         tickfont=dict(color=T.GREY, size=9))
        fig.update_xaxes(gridcolor=T.BORDER, linecolor=T.BORDER, showgrid=False,
                         tickfont=dict(color=T.GREY, size=9))
        return fig


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
            **CANDLE_STYLE,
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
            # green text on a solid black box so EMA/series labels are legible
            legend=dict(
                bgcolor=T.BG, bordercolor=T.AMBER, borderwidth=1,
                font=dict(color=T.AMBER, family=T.FONT_MONO, size=10),
            ),
            hoverlabel=dict(
                bgcolor=T.BG, bordercolor=T.AMBER,
                font=dict(color=T.AMBER, family=T.FONT_MONO, size=11),
            ),
        )
        # subplot titles ("Price with EMAs", etc.) — green, not white
        for ann in fig.layout.annotations:
            ann.font.color = T.AMBER
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
            fillcolor=f"rgba(0,255,65,0.15)",
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
