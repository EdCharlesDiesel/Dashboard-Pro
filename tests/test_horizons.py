"""Change over a horizon, computed identically for every timeframe.

Market Overview used to report `Close.pct_change().iloc[-1]` on whichever tab
was open, so "change" meant a different span depending on where you stood. These
functions take the frames the spine produces -- all resampled from one daily
series -- so a 1W figure and the 1D figures inside it are consistent.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.core import horizons as hz


def _closes(values):
    idx = pd.date_range("2026-01-01", periods=len(values), freq="D")
    return pd.DataFrame({"Close": values}, index=idx)


class TestPeriodChange:
    def test_rise_is_positive(self):
        assert hz.period_change(_closes([100.0, 110.0])) == pytest.approx(10.0)

    def test_fall_is_negative(self):
        assert hz.period_change(_closes([100.0, 90.0])) == pytest.approx(-10.0)

    def test_uses_only_the_last_two_closes(self):
        assert hz.period_change(_closes([1.0, 50.0, 100.0, 110.0])) == pytest.approx(10.0)

    def test_one_bar_has_no_previous_period(self):
        assert hz.period_change(_closes([100.0])) is None

    def test_empty_and_none_are_none(self):
        assert hz.period_change(pd.DataFrame()) is None
        assert hz.period_change(None) is None

    def test_a_frame_without_a_close_column_is_none(self):
        assert hz.period_change(pd.DataFrame({"Open": [1.0, 2.0]})) is None

    def test_zero_previous_close_does_not_divide_by_zero(self):
        assert hz.period_change(_closes([0.0, 100.0])) is None

    def test_nan_close_is_none_not_nan(self):
        # NaN would render as "nan%" and, if ever persisted, is invalid JSONB --
        # the failure that silently destroyed whole signal rows before.
        assert hz.period_change(_closes([float("nan"), 100.0])) is None
        assert hz.period_change(_closes([100.0, float("nan")])) is None

    def test_interior_nans_are_dropped_not_fatal(self):
        # A gap mid-series must not blank the whole reading.
        assert hz.period_change(
            _closes([100.0, float("nan"), 110.0])) == pytest.approx(10.0)


class TestHorizonRow:
    def test_builds_all_four_columns(self):
        frames = {"4 Hour": _closes([100.0, 101.0]),
                  "Daily": _closes([100.0, 102.0]),
                  "Weekly": _closes([100.0, 105.0]),
                  "Monthly": _closes([100.0, 110.0])}
        row = hz.horizon_row("EUR/USD", frames)
        assert row["Pair"] == "EUR/USD"
        assert row["4H %"] == pytest.approx(1.0)
        assert row["1D %"] == pytest.approx(2.0)
        assert row["1W %"] == pytest.approx(5.0)
        assert row["1M %"] == pytest.approx(10.0)

    def test_a_missing_timeframe_is_none_not_an_exception(self):
        row = hz.horizon_row("EUR/USD", {"Daily": _closes([100.0, 102.0])})
        assert row["1D %"] == pytest.approx(2.0)
        assert row["1M %"] is None

    def test_column_order_runs_shortest_to_longest(self):
        # The panel reads left to right as "just happened" -> "this month".
        assert [label for label, _ in hz.HORIZONS] == ["4H %", "1D %", "1W %", "1M %"]

    def test_every_horizon_maps_to_a_timeframe_the_spine_serves(self):
        from src.pages_lib.market_overview_lib import _SPINE
        for _label, timeframe in hz.HORIZONS:
            assert timeframe in _SPINE, (
                "%s has no spine function; the column would always be blank"
                % timeframe)


class TestSortByMover:
    def test_biggest_absolute_daily_move_comes_first(self):
        rows = [{"Pair": "A", "1D %": 0.2}, {"Pair": "B", "1D %": -3.0},
                {"Pair": "C", "1D %": 1.0}]
        assert [r["Pair"] for r in hz.sort_by_mover(rows)] == ["B", "C", "A"]

    def test_rows_without_a_reading_sink_to_the_bottom(self):
        rows = [{"Pair": "A", "1D %": None}, {"Pair": "B", "1D %": 1.0}]
        assert [r["Pair"] for r in hz.sort_by_mover(rows)] == ["B", "A"]

    def test_empty_input(self):
        assert hz.sort_by_mover([]) == []
