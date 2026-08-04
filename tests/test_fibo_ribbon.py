"""Unit tests for src/core/fibo_ribbon.py — pure, synthetic frames, no network.

The three things worth pinning down, because each is a way the signal could be
silently wrong rather than obviously broken:

1. the 8-SMA reads the **open**, not the close (it came off the chart that way,
   and the cross only works while the two series differ);
2. the ribbon gate actually suppresses counter-trend crosses;
3. the read comes from the last **closed** bar, so it cannot repaint.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core import fibo_ribbon as fr


def _frame(closes, opens=None, start="2024-01-01"):
    n = len(closes)
    closes = np.asarray(closes, dtype=float)
    opens = closes if opens is None else np.asarray(opens, dtype=float)
    idx = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame({
        "Open": opens,
        "High": np.maximum(opens, closes) + 0.5,
        "Low": np.minimum(opens, closes) - 0.5,
        "Close": closes,
        "Volume": np.ones(n),
    }, index=idx)


def _ramp(n, start=100.0, step=0.0):
    return [start + step * i for i in range(n)]


class TestRibbonColumns:
    def test_attaches_all_five_averages(self):
        out = fr.ribbon(_frame(_ramp(300, step=0.1)))
        for col in ("ema200", "ema55", "ema21", "ema5", "sma8"):
            assert col in out.columns

    def test_sma8_is_computed_from_the_open_not_the_close(self):
        # Opens deliberately offset from closes; if sma8 were built on Close it
        # would match the close-based rolling mean instead.
        closes = _ramp(60, 100.0, 1.0)
        opens = [c - 10.0 for c in closes]
        out = fr.ribbon(_frame(closes, opens))
        expected_open = pd.Series(opens, index=out.index).rolling(8).mean()
        expected_close = pd.Series(closes, index=out.index).rolling(8).mean()
        assert out["sma8"].iloc[-1] == pytest.approx(expected_open.iloc[-1])
        assert out["sma8"].iloc[-1] != pytest.approx(expected_close.iloc[-1])

    def test_does_not_mutate_the_callers_frame(self):
        df = _frame(_ramp(300, step=0.1))
        fr.ribbon(df)
        assert "ema200" not in df.columns

    def test_missing_open_column_degrades_to_empty(self):
        df = _frame(_ramp(300, step=0.1)).drop(columns=["Open"])
        assert fr.ribbon(df).empty


class TestReadGuards:
    def test_too_little_history_returns_none(self):
        assert fr.read(_frame(_ramp(50, step=0.1))) is None

    def test_empty_frame_returns_none(self):
        assert fr.read(pd.DataFrame()) is None

    def test_none_returns_none(self):
        assert fr.read(None) is None


class TestDirectionAndGate:
    def _uptrend_then_cross(self):
        # 260 bars up (ribbon stacks bullish), a dip that drags EMA5 under the
        # 8-SMA, then one sharp recovery bar that crosses it back up.
        # Bar layout matters: read() compares [-3] -> [-2], so the recovery must
        # be the *last closed* bar. An extra recovery bar would put the cross at
        # [-4] -> [-3] and read() would see no fresh cross.
        closes = _ramp(260, 100.0, 0.5)
        closes += [closes[-1] - 6, closes[-1] - 9]      # dip: EMA5 below SMA8
        closes += [closes[-1] + 14]                     # [-2] recovery: cross up
        closes += [closes[-1] + 1]                      # [-1] forming, discarded
        return _frame(closes)

    def _downtrend_then_cross(self):
        closes = _ramp(260, 300.0, -0.5)
        closes += [closes[-1] - 4, closes[-1] - 6]
        closes += [closes[-1] + 14]                     # [-2] counter-trend pop
        closes += [closes[-1] + 1]                      # [-1] forming
        return _frame(closes)

    def test_bullish_cross_in_a_stacked_ribbon_reads_long(self):
        r = fr.read(self._uptrend_then_cross())
        assert r is not None
        assert r.crossed is True
        assert r.direction == fr.LONG
        assert r.stacked is True

    def test_no_cross_reads_neutral(self):
        r = fr.read(_frame(_ramp(300, 100.0, 0.5)))   # smooth ramp, no cross
        assert r is not None and r.direction == fr.NEUTRAL and r.crossed is False

    def test_counter_trend_cross_is_gated_out(self):
        # Sustained downtrend (ribbon stacks bearish), then a bounce that
        # crosses EMA5 up through SMA8. The cross happens; the gate rejects it.
        r = fr.read(self._downtrend_then_cross())
        assert r is not None
        assert r.crossed is True          # the cross is real
        assert r.direction == fr.NEUTRAL  # but suppressed
        assert r.stacked is False
        assert "gated out" in r.reason

    def test_gate_can_be_disabled_for_measurement(self):
        r = fr.read(self._downtrend_then_cross(), require_stack=False)
        assert r.direction == fr.LONG and r.stacked is False


class TestClosedBarOnly:
    def test_the_forming_bar_is_never_read(self):
        base = _frame(_ramp(300, 100.0, 0.5))
        r1 = fr.read(base)
        # Replace only the final (forming) bar with a violent spike. If the read
        # touched it, direction/close would move.
        spiked = base.copy()
        spiked.iloc[-1, spiked.columns.get_loc("Close")] = 1e4
        spiked.iloc[-1, spiked.columns.get_loc("Open")] = 1e4
        r2 = fr.read(spiked)
        assert r1.close == r2.close
        assert r1.direction == r2.direction
        assert r1.ema5 == pytest.approx(r2.ema5)

    def test_bar_time_is_the_last_closed_bar(self):
        df = _frame(_ramp(300, 100.0, 0.5))
        r = fr.read(df)
        assert r.bar_time == df.index[-2]
        assert r.bar_time != df.index[-1]
