"""Unit tests for src/core/biased_pivots.py — the FX Bootcamp pivot port.

The formulas are pinned against hand-computed values, because the whole value of
this module is agreeing with what `BiasedPivots.mq5` draws on the chart. The two
non-obvious behaviours get their own tests: the close comes from one period
*earlier* than the high/low, and the levels come from the last **closed** bar
while only the price comes from the forming one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core import biased_pivots as bp


def _frame(rows):
    """rows: list of (high, low, close)."""
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="D")
    return pd.DataFrame(
        {"High": [r[0] for r in rows],
         "Low": [r[1] for r in rows],
         "Close": [r[2] for r in rows],
         "Open": [r[2] for r in rows]},
        index=idx)


class TestLevels:
    def test_matches_hand_computed_values(self):
        # H=110 L=90 Cprev=100 -> PP=100
        lv = bp.levels(110.0, 90.0, 100.0)
        assert lv["pp"] == pytest.approx(100.0)
        assert lv["r1"] == pytest.approx(2 * 100.0 - 90.0)      # 110
        assert lv["r2"] == pytest.approx(100.0 + (110.0 - 90.0))  # 120
        assert lv["s1"] == pytest.approx(2 * 100.0 - 110.0)     # 90
        assert lv["s2"] == pytest.approx(100.0 - (110.0 - 90.0))  # 80

    def test_midpoints_are_the_documented_halves(self):
        lv = bp.levels(110.0, 90.0, 100.0)
        assert lv["m1"] == pytest.approx(0.5 * (lv["s1"] + lv["s2"]))
        assert lv["m2"] == pytest.approx(0.5 * (lv["pp"] + lv["s1"]))
        assert lv["m3"] == pytest.approx(0.5 * (lv["pp"] + lv["r1"]))
        assert lv["m4"] == pytest.approx(0.5 * (lv["r1"] + lv["r2"]))

    def test_r3_s3_m0_m5_are_not_invented(self):
        # The source assigns them 0 and never uses them; reproducing fake zeros
        # would invite someone to trade a level that does not exist.
        lv = bp.levels(110.0, 90.0, 100.0)
        for absent in ("r3", "s3", "m0", "m5"):
            assert absent not in lv


class TestCloseComesFromOnePeriodEarlier:
    def test_close_is_taken_from_the_bar_before_the_high_low(self):
        # [-3] close = 50 (used), [-2] high/low = 110/90 (used),
        # [-2] close = 999 (must be ignored), [-1] = forming.
        df = _frame([(1, 1, 50.0), (110.0, 90.0, 999.0), (1, 1, 100.0)])
        r = bp.read(df)
        assert r is not None
        # PP built from 110, 90 and the *earlier* close of 50
        assert r.pp == pytest.approx((110.0 + 90.0 + 50.0) / 3.0)
        # and emphatically not from the same bar's close of 999
        assert r.pp != pytest.approx((110.0 + 90.0 + 999.0) / 3.0)

    def test_levels_ignore_the_forming_bar(self):
        base = _frame([(1, 1, 50.0), (110.0, 90.0, 999.0), (1, 1, 100.0)])
        r1 = bp.read(base)
        spiked = base.copy()
        spiked.iloc[-1, spiked.columns.get_loc("High")] = 1e6
        spiked.iloc[-1, spiked.columns.get_loc("Low")] = -1e6
        r2 = bp.read(spiked)
        assert r1.pp == pytest.approx(r2.pp)
        assert r1.r1 == pytest.approx(r2.r1)


class TestZones:
    def _at(self, price):
        # H=110 L=90 Cprev=100 -> PP=100, S1=90, R1=110, M2=95, M3=105
        return bp.read(_frame([(1, 1, 100.0), (110.0, 90.0, 0.0), (1, 1, price)]))

    def test_price_at_m2_reads_long(self):
        r = self._at(95.0)
        assert r.direction == bp.LONG and r.zone == "Bullish Buying Zone"

    def test_price_at_m3_reads_short(self):
        r = self._at(105.0)
        assert r.direction == bp.SHORT and r.zone == "Bearish Selling Zone"

    def test_price_at_the_pivot_is_neutral(self):
        assert self._at(100.0).direction == bp.NEUTRAL

    def test_zone_width_scales_with_the_pivot_geometry(self):
        # half-width = 0.25 * |PP - S1| = 2.5, so 97.4 is in and 97.6 is out
        assert self._at(97.4).direction == bp.LONG
        assert self._at(97.6).direction == bp.NEUTRAL


class TestTargets:
    def _long(self):
        return bp.read(_frame([(1, 1, 100.0), (110.0, 90.0, 0.0), (1, 1, 95.0)]))

    def test_long_targets_are_the_bullish_labels(self):
        r = self._long()
        t = bp.targets(r)
        assert t["conservative"] == pytest.approx(r.m4)   # "Bullish Profit Target"
        assert t["aggressive"] == pytest.approx(r.r2)     # "Bullish Profit Zone"
        assert t["stop_ref"] == pytest.approx(r.s1)

    def test_short_targets_are_the_bearish_labels(self):
        r = bp.read(_frame([(1, 1, 100.0), (110.0, 90.0, 0.0), (1, 1, 105.0)]))
        t = bp.targets(r)
        assert t["conservative"] == pytest.approx(r.m1)
        assert t["aggressive"] == pytest.approx(r.s2)

    def test_neutral_has_no_targets(self):
        t = bp.targets(bp.read(_frame([(1, 1, 100.0), (110.0, 90.0, 0.0), (1, 1, 100.0)])))
        assert all(v is None for v in t.values())


class TestGuards:
    @pytest.mark.parametrize("df", [None, pd.DataFrame()])
    def test_missing_frame_returns_none(self, df):
        assert bp.read(df) is None

    def test_too_few_bars_returns_none(self):
        assert bp.read(_frame([(1, 1, 1), (2, 2, 2)])) is None

    def test_nan_degrades_to_none(self):
        df = _frame([(1, 1, np.nan), (110.0, 90.0, 0.0), (1, 1, 100.0)])
        assert bp.read(df) is None

    def test_payload_is_json_safe(self):
        import json
        r = bp.read(_frame([(1, 1, 100.0), (110.0, 90.0, 0.0), (1, 1, 95.0)]))
        d = {k: v for k, v in r.to_dict().items() if k != "bar_time"}
        json.loads(json.dumps(d))
