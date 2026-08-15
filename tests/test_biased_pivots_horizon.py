"""The pivot horizon must match the period the levels come from.

Pivot levels are recomputed every period: a new bar closes, and PP/S1/R1/M2/M3
all move. So a signal derived from them is valid for about *one* period, and no
longer. The page shipped daily pivots with a 5-day horizon and a comment
conceding the mismatch — "pivot zones are a day-trade construct, not a swing
one" — which is the wrong half to concede in a swing system.

Weekly pivots fix it honestly: levels from last completed week, held for the
week ahead. Same indicator, and the MQL5 original is period-parameterised
(`timePeriodConverter()`), so this is the port catching up rather than a new
invention.
"""
from __future__ import annotations

import ast
import inspect
import os
import textwrap

import pandas as pd
import pytest

from src.core.biased_pivots import LONG, read
from src.pages_lib import biased_pivots_page as page

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRADING_DAYS_PER = {"weekly": 5, "daily": 1, "monthly": 21}


class TestHorizonMatchesPeriod:
    def test_the_period_is_declared(self):
        assert hasattr(page, "_PIVOT_PERIOD"), (
            "the page must say which period its pivots come from")
        assert page._PIVOT_PERIOD in _TRADING_DAYS_PER

    def test_horizon_equals_one_period(self):
        # The invariant. A horizon longer than the period outlives the levels it
        # was derived from; shorter, and the signal expires while its own zone
        # is still on the chart.
        assert page._HORIZON_DAYS == _TRADING_DAYS_PER[page._PIVOT_PERIOD]

    def test_it_is_not_a_day_trade_construct(self):
        assert page._PIVOT_PERIOD != "daily", (
            "daily pivots expire nightly — that is a day-trade horizon")
        assert page._HORIZON_DAYS >= 5

    def test_the_scan_reads_the_declared_period(self):
        # A declared period the scanner ignores is worse than none: the horizon
        # would look justified while the levels came from somewhere else.
        # dedent, not strip: `getsource` returns the method still indented
        # inside its class, and `.strip()` only unindents the first line, so
        # ast.parse dies with IndentationError before it can see any call.
        source = textwrap.dedent(inspect.getsource(page.BiasedPivotsPage._scan))
        calls = {n.func.id for n in ast.walk(ast.parse(source))
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert "weekly_ohlc" in calls
        assert "daily_ohlc" not in calls


class TestStillReadsCorrectlyOnWeeklyBars:
    def _weekly(self, rows):
        idx = pd.date_range("2026-01-04", periods=len(rows), freq="W")
        return pd.DataFrame(
            {"High": [r[0] for r in rows], "Low": [r[1] for r in rows],
             "Close": [r[2] for r in rows], "Open": [r[2] for r in rows]},
            index=idx)

    def test_zones_still_resolve(self):
        # H=110 L=90 Cprev=100 -> PP=100, M2=95. The maths is period-agnostic;
        # this pins that feeding it weekly bars changes nothing but the clock.
        r = read(self._weekly([(1, 1, 100.0), (110.0, 90.0, 95.0), (1, 1, 0.0)]))
        assert r is not None
        assert r.pp == pytest.approx(100.0)
        assert r.direction == LONG

    def test_bar_time_is_the_weekly_bar(self):
        frame = self._weekly([(1, 1, 100.0), (110.0, 90.0, 95.0), (1, 1, 0.0)])
        assert read(frame).bar_time == frame.index[-2]
