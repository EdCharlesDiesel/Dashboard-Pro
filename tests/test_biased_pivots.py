"""Unit tests for src/core/biased_pivots.py — the FX Bootcamp pivot port.

The formulas are pinned against hand-computed values, because the whole value of
this module is agreeing with what `BiasedPivots.mq5` draws on the chart. The two
non-obvious behaviours get their own tests: the close comes from one period
*earlier* than the high/low, and everything — levels *and* the price judged
against them — comes from the last **closed** bar, so one bar always yields one
answer. The forming bar is ignored entirely; it used to supply the price, which
let a single bar produce opposite calls as the live price drifted across a zone
boundary.
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
        #
        # `price` now sits on the *period* bar (index -2), not the forming one.
        # It used to go on the forming bar with the period close set to a
        # don't-care 0.0 — which is exactly the behaviour that let one bar
        # produce opposite calls as the live price drifted across a boundary.
        # The forming bar's close is the don't-care now, and is genuinely
        # ignored.
        return bp.read(_frame([(1, 1, 100.0), (110.0, 90.0, price), (1, 1, 0.0)]))

    def test_the_forming_bar_close_is_ignored(self):
        # The inversion of the old contract, pinned: same period bar, absurd
        # live prices, identical read.
        reads = {bp.read(_frame([(1, 1, 100.0), (110.0, 90.0, 95.0),
                                 (1, 1, tick)])).direction
                 for tick in (0.0, 50.0, 105.0, 1e6)}
        assert reads == {bp.LONG}

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
        return bp.read(_frame([(1, 1, 100.0), (110.0, 90.0, 95.0), (1, 1, 0.0)]))

    def test_long_targets_are_the_bullish_labels(self):
        r = self._long()
        t = bp.targets(r)
        assert t["conservative"] == pytest.approx(r.m4)   # "Bullish Profit Target"
        assert t["aggressive"] == pytest.approx(r.r2)     # "Bullish Profit Zone"
        assert t["stop_ref"] == pytest.approx(r.s1)

    def test_short_targets_are_the_bearish_labels(self):
        r = bp.read(_frame([(1, 1, 100.0), (110.0, 90.0, 105.0), (1, 1, 0.0)]))
        t = bp.targets(r)
        assert t["conservative"] == pytest.approx(r.m1)
        assert t["aggressive"] == pytest.approx(r.s2)

    def test_neutral_has_no_targets(self):
        t = bp.targets(bp.read(_frame([(1, 1, 100.0), (110.0, 90.0, 100.0), (1, 1, 0.0)])))
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


# ── determinism: one bar, one answer ─────────────────────────────────────────
# The bug: the zone was judged against the *forming* bar's live close while
# being stamped with the last *closed* bar. The sweep re-reads every 5 minutes,
# so as price drifted across a zone boundary the same bar produced opposite
# calls -- AUD/USD long at 23:43 and short 40 minutes later, both stamped
# bar_time=2026-08-09, and both persisted because the dedupe key is
# (pair, direction, period) and the direction differed. `biased_pivots`
# contradicted itself on 20 of 27 instruments, more than any other source.


def _close_frame(closes, highs=None, lows=None):
    """A daily frame; the last row is the still-forming bar."""
    n = len(closes)
    idx = pd.date_range("2026-08-01", periods=n, freq="D")
    return pd.DataFrame({
        "Open": closes,
        "High": highs if highs is not None else [c * 1.002 for c in closes],
        "Low": lows if lows is not None else [c * 0.998 for c in closes],
        "Close": closes,
    }, index=idx)


class TestOneBarOneAnswer:
    def test_forming_bar_price_does_not_change_the_direction(self):
        # Identical history, wildly different live price. The read must not move.
        base = [1.1000, 1.1050, 1.1020]
        low_tick = _close_frame(base + [1.0800])
        high_tick = _close_frame(base + [1.1400])
        a, b = bp.read(low_tick), bp.read(high_tick)
        assert a is not None and b is not None
        assert a.direction == b.direction
        assert a.price == b.price          # judged on the closed bar, not the tick

    def test_the_judged_price_is_the_stamped_bar(self):
        df = _close_frame([1.1000, 1.1050, 1.1020, 1.1400])
        read = bp.read(df)
        assert read is not None
        # bar_time names df.index[-2]; the price must come from that same row.
        assert read.price == pytest.approx(float(df["Close"].iloc[-2]))
        assert read.bar_time == df.index[-2]

    def test_repeated_reads_of_one_frame_agree(self):
        df = _close_frame([1.1000, 1.1050, 1.1020, 1.1100])
        directions = {bp.read(df).direction for _ in range(5)}
        assert len(directions) == 1

    def test_a_new_closed_bar_is_allowed_to_change_the_read(self):
        # Determinism per bar, not permanence: appending a genuinely new closed
        # bar may legitimately flip the zone.
        short_frame = _close_frame([1.1000, 1.1050, 1.1020, 1.1100])
        long_frame = _close_frame([1.1000, 1.1050, 1.1020, 1.1100, 1.0900])
        assert bp.read(short_frame).bar_time != bp.read(long_frame).bar_time
