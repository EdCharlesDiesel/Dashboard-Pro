"""Unit tests for the ChartKit stencil (src/ui/charts.py)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.core.config import CANDLE_STYLE
from src.ui.charts import ChartKit
from src.ui.theme import BloombergTheme as T


def _ohlc(n=30):
    close = np.linspace(1.10, 1.12, n)
    idx = pd.bdate_range("2026-01-05", periods=n)
    return pd.DataFrame({"Open": close, "High": close + 0.002,
                         "Low": close - 0.002, "Close": close,
                         "Volume": np.zeros(n)}, index=idx)


class TestChartKit:
    def test_panels_builds_stacked_rows(self):
        fig = ChartKit.panels(["A", "B", "C"], heights=[0.5, 0.25, 0.25])
        titles = [a.text for a in fig.layout.annotations]
        assert titles == ["A", "B", "C"]
        assert "yaxis3" in fig.layout   # three stacked rows exist

    def test_candles_uses_canonical_style(self):
        fig = ChartKit.panels(["P"])
        ChartKit.candles(fig, _ohlc())
        trace = fig.data[0]
        assert isinstance(trace, go.Candlestick)
        assert trace.increasing.line.color == CANDLE_STYLE["increasing_line_color"]
        assert trace.decreasing.fillcolor == CANDLE_STYLE["decreasing_fillcolor"]

    def test_bars_default_signed_coloring(self):
        fig = ChartKit.panels(["F"])
        ChartKit.bars(fig, [1, 2, 3], [0.5, -0.2, 0.1])
        assert list(fig.data[0].marker.color) == [T.GREEN, T.RED, T.GREEN]

    def test_band_adds_fill_pair(self):
        fig = ChartKit.panels(["B"])
        ChartKit.band(fig, [1, 2], [3, 3], [1, 1], "rgba(0,224,255,0.1)")
        assert len(fig.data) == 2
        assert fig.data[1].fill == "tonexty"

    def test_levels_annotates_each_named_level(self):
        # A trace must exist on the row first — plotly's add_hline silently
        # no-ops on trace-less subplots (documented on ChartKit.hline).
        fig = ChartKit.panels(["L"])
        ChartKit.line(fig, [1, 2], [1.10, 1.11], "px")
        ChartKit.levels(fig, {"P": 1.10, "R1": 1.11}, colors={"P": T.AMBER})
        labels = {a.text for a in fig.layout.annotations}
        assert {"P", "R1"}.issubset(labels)
        assert len(fig.layout.shapes) == 2

    def test_finish_applies_house_layout(self):
        fig = ChartKit.panels(["X"])
        ChartKit.line(fig, [1, 2], [3, 4], "l")
        out = ChartKit.finish(fig, height=500)
        assert out is fig
        assert fig.layout.paper_bgcolor == T.BG
        assert fig.layout.plot_bgcolor == T.BG_PANEL
        assert fig.layout.height == 500
        assert fig.layout.font.family == T.FONT_MONO
        assert fig.layout.xaxis.rangeslider.visible is False
        # subplot title recoloured to the accent
        assert fig.layout.annotations[0].font.color == T.AMBER
