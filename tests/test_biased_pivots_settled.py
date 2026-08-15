"""A bar that closed minutes ago is closed by index, not settled in fact.

Judging the last closed bar removed 23 of 25 same-bar contradictions. The two
survivors named the same bar_time and disagreed about its close -- AUD/ZAR at
11.42590 then 11.39812, USD/ZAR at 16.19230 then 16.13787 -- both within two
hours of the 21:00 UTC FX day break. `read()` is deterministic given a frame;
the frame changed underneath it while the provider revised a just-closed bar.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.core import biased_pivots as bp


def _frame(rows, freq="D", end=None):
    """rows: list of (high, low, close). Last row is the forming bar."""
    end = end or datetime(2026, 8, 14)
    idx = pd.date_range(end=end, periods=len(rows), freq=freq)
    return pd.DataFrame(
        {"High": [r[0] for r in rows], "Low": [r[1] for r in rows],
         "Close": [r[2] for r in rows], "Open": [r[2] for r in rows]},
        index=idx)


def _three_bars():
    return _frame([(1, 1, 50.0), (110.0, 90.0, 95.0), (1, 1, 0.0)])


def _four_bars():
    # -4 close 40 (the shft+1 close for the fallback), -3 the older period,
    # -2 the freshly closed one, -1 forming.
    return _frame([(1, 1, 40.0), (210.0, 190.0, 195.0),
                   (110.0, 90.0, 95.0), (1, 1, 0.0)])


SETTLED = datetime(2026, 8, 21)          # a week after the forming bar opened
FRESH = datetime(2026, 8, 14, 1, 0)      # one hour after it opened


class TestSettledPeriodIndex:
    def test_an_old_closed_bar_is_used(self):
        assert bp.settled_period_index(_three_bars(), now=SETTLED) == -2

    def test_a_freshly_closed_bar_is_skipped(self):
        # Needs four bars: stepping back to -3 as the period leaves -4 as the
        # indicator's shft+1 close, so a three-bar frame cannot fall back.
        assert bp.settled_period_index(_four_bars(), now=FRESH) == -3

    def test_the_boundary_is_inclusive(self):
        exactly = datetime(2026, 8, 14) + timedelta(hours=bp.SETTLE_HOURS)
        assert bp.settled_period_index(_three_bars(), now=exactly) == -2

    def test_too_short_to_step_back_returns_none(self):
        # Three bars and the newest is too fresh: there is no safe read here.
        # Returning -3 would make `read()` index -4 on a three-row frame and
        # silently wrap to the newest bar -- the repaint this whole change
        # exists to stop.
        assert bp.settled_period_index(_three_bars(), now=FRESH) is None

    def test_two_bars_is_never_enough(self):
        assert bp.settled_period_index(
            _frame([(110.0, 90.0, 95.0), (1, 1, 0.0)]), now=SETTLED) is None

    def test_a_non_datetime_index_cannot_be_aged_so_is_trusted(self):
        # No clock means no evidence of freshness; refusing to read would be a
        # worse failure than reading a bar that might be fresh.
        df = pd.DataFrame({"High": [1, 110.0, 1], "Low": [1, 90.0, 1],
                           "Close": [50.0, 95.0, 0.0], "Open": [0, 0, 0]})
        assert bp.settled_period_index(df) == -2

    def test_empty_and_none_are_none(self):
        assert bp.settled_period_index(None) is None
        assert bp.settled_period_index(pd.DataFrame()) is None


class TestReadHonoursIt:
    def test_read_uses_the_fresh_bar_when_it_is_old_enough(self):
        r = bp.read(_four_bars(), now=SETTLED)
        assert r is not None
        assert r.price == pytest.approx(95.0)
        assert r.pp == pytest.approx((110.0 + 90.0 + 195.0) / 3.0)

    def test_read_falls_back_when_the_bar_is_too_fresh(self):
        r = bp.read(_four_bars(), now=FRESH)
        assert r is not None
        assert r.price == pytest.approx(195.0)          # the older bar's close
        assert r.pp == pytest.approx((210.0 + 190.0 + 40.0) / 3.0)

    def test_bar_time_names_the_bar_actually_read(self):
        df = _four_bars()
        assert bp.read(df, now=FRESH).bar_time == df.index[-3]
        assert bp.read(df, now=SETTLED).bar_time == df.index[-2]

    def test_read_returns_none_when_there_is_no_settled_bar(self):
        assert bp.read(_three_bars(), now=FRESH) is None

    def test_two_reads_of_one_frame_at_different_clock_times_agree(self):
        # The regression, in one test: the same frame read 50 minutes apart --
        # the gap that produced the USD/ZAR flip -- must give the same answer,
        # provided both are outside the settling window.
        df = _four_bars()
        a = bp.read(df, now=datetime(2026, 8, 21, 9, 0))
        b = bp.read(df, now=datetime(2026, 8, 21, 9, 50))
        assert (a.direction, a.price, a.bar_time) == (b.direction, b.price, b.bar_time)

    def test_the_default_clock_is_now(self):
        # Frames dated 2026 with no `now=` must still read: real "now" is far
        # past them, so every bar is settled. This is what keeps the existing
        # pivot tests passing untouched.
        assert bp.read(_four_bars()) is not None
