"""Unit tests for the TradingView pilot's pure config builders.

The component itself is a browser widget; these tests exercise the data
conversion + config assembly (the part that can be wrong), never the frontend.
"""
import numpy as np
import pandas as pd
import pytest

from src.core.config import CANDLE_STYLE
from src.ui import tv_charts
from src.ui.theme import BloombergTheme as T


def _ohlc(n=30, tz=None):
    close = np.linspace(1900, 1950, n)
    idx = pd.date_range("2026-01-05", periods=n, freq="D", tz=tz)
    return pd.DataFrame({"Open": close, "High": close + 5,
                         "Low": close - 5, "Close": close,
                         "Volume": np.zeros(n)}, index=idx)


class TestConversion:
    def test_epoch_is_utc_int_and_monotonic(self):
        df = _ohlc()
        ts = [tv_charts._epoch(t) for t in df.index]
        assert all(isinstance(x, int) for x in ts)
        assert ts == sorted(ts) and len(set(ts)) == len(ts)

    def test_epoch_naive_and_aware_agree(self):
        naive = tv_charts._epoch(pd.Timestamp("2026-03-01 00:00"))
        aware = tv_charts._epoch(pd.Timestamp("2026-03-01 00:00", tz="UTC"))
        assert naive == aware

    def test_candle_data_shape(self):
        d = tv_charts.candle_data(_ohlc(3))
        assert len(d) == 3
        assert set(d[0]) == {"time", "open", "high", "low", "close"}
        assert isinstance(d[0]["open"], float)

    def test_candle_data_skips_nan_bars(self):
        df = _ohlc(3)
        df.iloc[1, df.columns.get_loc("Close")] = np.nan
        assert len(tv_charts.candle_data(df)) == 2

    def test_line_data_drops_nan(self):
        idx = _ohlc(4).index
        s = pd.Series([1.0, np.nan, 3.0, 4.0], index=idx)
        d = tv_charts.line_data(idx, s)
        assert [p["value"] for p in d] == [1.0, 3.0, 4.0]

    def test_histogram_signed_coloring(self):
        idx = _ohlc(3).index
        s = pd.Series([0.5, -0.2, 0.1], index=idx)
        d = tv_charts.histogram_data(idx, s)
        assert [p["color"] for p in d] == [T.GREEN, T.RED, T.GREEN]

    def test_flat_line_is_two_points(self):
        idx = _ohlc(5).index
        d = tv_charts._flat(list(idx), 1925.0)
        assert len(d) == 2
        assert d[0]["value"] == 1925.0 and d[1]["value"] == 1925.0
        assert d[0]["time"] < d[1]["time"]


class TestLeadingCharts:
    def _pivots(self):
        return {"P": 1925.0, "R1": 1940.0, "S1": 1910.0}

    def _colors(self):
        return {"P": T.AMBER, "R1": T.RED, "S1": T.GREEN}

    def test_three_panes_price_rsi_flow(self):
        df = _ohlc()
        rsi = pd.Series(np.linspace(40, 60, len(df)), index=df.index)
        flow = pd.Series(np.linspace(-0.5, 0.5, len(df)), index=df.index)
        charts = tv_charts.leading_charts(df, self._pivots(), self._colors(),
                                          rsi, flow, rsi_period=14)
        assert len(charts) == 3
        # price pane: 1 candlestick + 3 pivot lines
        price = charts[0]["series"]
        assert price[0]["type"] == "Candlestick"
        assert sum(1 for s in price if s["type"] == "Line") == 3
        # candle colours come from the canonical CANDLE_STYLE
        assert price[0]["options"]["upColor"] == CANDLE_STYLE["increasing_fillcolor"]
        # rsi pane: line + two reference lines
        assert charts[1]["series"][0]["options"]["title"] == "RSI 14"
        # flow pane: histogram
        assert charts[2]["series"][0]["type"] == "Histogram"

    def test_divergence_marker_added_only_with_direction_and_level(self):
        df = _ohlc()
        rsi = pd.Series(50.0, index=df.index)
        flow = pd.Series(0.0, index=df.index)
        none = tv_charts.leading_charts(df, self._pivots(), self._colors(),
                                        rsi, flow, direction=None, at_level=None)
        assert "markers" not in none[0]["series"][0]
        longd = tv_charts.leading_charts(df, self._pivots(), self._colors(),
                                         rsi, flow, direction="Long", at_level="S1")
        mk = longd[0]["series"][0]["markers"]
        assert mk[0]["shape"] == "arrowUp" and "S1" in mk[0]["text"]

    def test_every_pane_uses_terminal_background(self):
        df = _ohlc()
        rsi = pd.Series(50.0, index=df.index)
        flow = pd.Series(0.0, index=df.index)
        charts = tv_charts.leading_charts(df, self._pivots(), self._colors(),
                                          rsi, flow)
        for c in charts:
            assert c["chart"]["layout"]["background"]["color"] == T.BG


class TestGeneralBuilders:
    def test_marker_shape(self):
        m = tv_charts.marker(pd.Timestamp("2026-02-01"), "aboveBar",
                             T.GREEN, "arrowDown", "SH")
        assert set(m) == {"time", "position", "color", "shape", "text"}
        assert m["shape"] == "arrowDown" and m["text"] == "SH"

    def test_candle_series_attaches_markers(self):
        df = _ohlc()
        mk = [tv_charts.marker(df.index[-1], "belowBar", T.RED, "arrowUp", "SL")]
        s = tv_charts.candle_series(df, markers=mk)
        assert s["type"] == "Candlestick"
        assert s["markers"] == mk
        # no markers key when none passed
        assert "markers" not in tv_charts.candle_series(df)

    def test_line_series_dash_maps_to_style_enum(self):
        idx = _ohlc(3).index
        s = tv_charts.line_series(idx, pd.Series([1.0, 2.0, 3.0], index=idx),
                                  T.CYAN, "x", dash="dash")
        assert s["options"]["lineStyle"] == 2          # dashed
        assert s["type"] == "Line"

    def test_level_series_is_flat_pair(self):
        idx = list(_ohlc(5).index)
        s = tv_charts.level_series(idx, 1925.0, T.WHITE, "PP", dash="solid")
        assert len(s["data"]) == 2
        assert s["options"]["lineStyle"] == 0          # solid
        assert s["options"]["title"] == "PP"

    def test_price_chart_wraps_series_with_themed_opts(self):
        c = tv_charts.price_chart([tv_charts.candle_series(_ohlc())], height=300)
        assert c["chart"]["height"] == 300
        assert c["chart"]["layout"]["background"]["color"] == T.BG
        assert len(c["series"]) == 1


class TestAvailability:
    def test_available_returns_bool(self):
        assert isinstance(tv_charts.available(), bool)

    def test_engine_toggle_is_callable(self):
        # Can't drive Streamlit widgets here; just assert the symbol exists so
        # a rename can't silently break every wired page.
        assert callable(tv_charts.engine_toggle)
