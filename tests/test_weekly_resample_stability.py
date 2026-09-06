"""A settled weekly bar's Close must not depend on a row that may not arrive.

`weekly_ohlc` resamples daily bars rather than pulling native weekly ones — the
desk's convention, so every timeframe derives from one series. That is fine, but
the resample rule decides *which day closes the week*, and pandas ``"W"`` means
week-ending-**Sunday**.

FX trades a partial Sunday session, and those rows flicker: the daily history
holds 4 Saturday and 4 Sunday bars in 220, present in some recent weeks and not
others. Under ``"W"`` that flickering row is the week's Close. Under ``"W-FRI"``
it opens the next week instead, which is both the FX convention and stable.

This is not theoretical. On 2026-09-05 `biased_pivots` read GBP/AUD twice from
the bar labelled 2026-08-23 and returned Long then Short, 46 minutes apart:

    2026-08-21 (Fri)  Close 1.91631   <- the second read
    2026-08-23 (Sun)  Close 1.90280   <- the first read

Same bar, same label, opposite sides of a pivot zone.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.services.market_data import _resample, weekly_ohlc

_AGG = {"Open": "first", "High": "max", "Low": "min", "Close": "last",
        "Volume": "sum"}


def _daily(with_weekend: bool) -> pd.DataFrame:
    """Four weeks of daily bars, optionally carrying the Sunday sessions.

    Prices rise monotonically, so any week whose Close moves between the two
    frames is visible immediately rather than needing a tolerance.
    """
    days = pd.bdate_range("2026-08-03", "2026-08-28")          # Mon-Fri only
    if with_weekend:
        sundays = pd.DatetimeIndex(["2026-08-09", "2026-08-16", "2026-08-23"])
        days = days.union(sundays).sort_values()
    # Prices derived from the DATE, never the row position. With positional
    # values (range(1, n+1)) inserting a Sunday row shifts every later day's
    # price, so the two frames differ everywhere and the test fails for a reason
    # that has nothing to do with the resample rule.
    base = [float(d.toordinal()) for d in days]
    return pd.DataFrame({
        "Open": base, "High": [b + 1 for b in base],
        "Low": [b - 1 for b in base], "Close": base,
        "Volume": [1.0] * len(days),
    }, index=days)


class TestTheInstability:
    def test_week_ending_sunday_moves_when_the_sunday_row_vanishes(self):
        """The defect, pinned so the fix cannot be undone silently."""
        with_wk = _daily(True).resample("W").agg(_AGG).dropna()["Close"]
        without = _daily(False).resample("W").agg(_AGG).dropna()["Close"]
        common = with_wk.index.intersection(without.index)
        moved = int((with_wk[common] != without[common]).sum())
        assert moved > 0, ("expected W to be unstable — if this passes, the "
                           "fixture no longer reproduces the original fault")


class TestTheGuarantee:
    def test_a_settled_weeks_close_survives_a_missing_weekend_row(self):
        with_wk = _resample(_daily(True), "W-FRI")["Close"]
        without = _resample(_daily(False), "W-FRI")["Close"]
        common = with_wk.index.intersection(without.index)
        assert len(common) >= 3, "fixture must span several settled weeks"
        moved = [str(d.date()) for d in common if with_wk[d] != without[d]]
        assert not moved, f"weeks whose Close moved: {moved}"

    def test_the_production_helper_uses_a_friday_rule(self):
        """`weekly_ohlc` itself, not just a local resample."""
        df = weekly_ohlc.__doc__ or ""
        from src.services import market_data
        import inspect
        src = inspect.getsource(market_data.weekly_ohlc)
        assert '"W-FRI"' in src or "'W-FRI'" in src, (
            "weekly_ohlc must resample W-FRI; plain W closes the week on the "
            "flickering Sunday session")

    def test_weekly_labels_land_on_fridays(self):
        wk = _resample(_daily(True), "W-FRI")
        bad = [str(d.date()) for d in wk.index if d.dayofweek != 4]
        assert not bad, f"non-Friday weekly labels: {bad}"

    def test_the_sunday_session_opens_the_following_week(self):
        # The convention this rests on: an FX week closes Friday, and Sunday
        # evening belongs to the week after. Under W-FRI the Sunday row must not
        # change the week that precedes it.
        wk_with = _resample(_daily(True), "W-FRI")["Close"]
        wk_without = _resample(_daily(False), "W-FRI")["Close"]
        friday = pd.Timestamp("2026-08-21")
        assert friday in wk_with.index
        assert wk_with[friday] == wk_without[friday]


class TestDegenerateFrames:
    def test_an_empty_frame_stays_empty(self):
        empty = pd.DataFrame(columns=list(_AGG), index=pd.DatetimeIndex([]))
        assert _resample(empty, "W-FRI").empty

    def test_a_single_day_still_produces_one_week(self):
        one = _daily(False).head(1)
        assert len(_resample(one, "W-FRI")) == 1


class TestMonthlyStability:
    """Months have the same disease and need a different cure.

    The same cure works: `BME` (business month end) moves the bin boundary to
    the last business day, so a month-ending Sunday falls into the next month —
    exactly as `W-FRI` pushes the Sunday session into the next week.

    A live measurement briefly said otherwise (ME 1/61, BME 1/61) and nearly led
    to dropping weekend rows outright. It was confounded: the single month moving
    was the *in-progress* one, whose Close moves as days arrive by design. These
    tests use settled months only, which is what reversed the conclusion.
    """

    def _daily_spanning_months(self, with_weekend: bool) -> pd.DataFrame:
        days = pd.bdate_range("2026-05-01", "2026-08-31")
        if with_weekend:
            # Month-ending weekend sessions - exactly the rows that flicker.
            extra = pd.DatetimeIndex(["2026-05-31", "2026-08-30"])
            days = days.union(extra).sort_values()
        base = [float(d.toordinal()) for d in days]
        return pd.DataFrame({
            "Open": base, "High": [b + 1 for b in base],
            "Low": [b - 1 for b in base], "Close": base,
            "Volume": [1.0] * len(days),
        }, index=days)

    def test_month_end_is_unstable_under_the_calendar_rule(self):
        """The defect, pinned so the fix cannot be reverted silently."""
        a = self._daily_spanning_months(True).resample("ME").agg(_AGG).dropna()["Close"]
        b = self._daily_spanning_months(False).resample("ME").agg(_AGG).dropna()["Close"]
        common = a.index.intersection(b.index)
        assert int((a[common] != b[common]).sum()) > 0, (
            "expected ME to be unstable — if this passes the fixture no longer "
            "reproduces the original fault")

    def test_the_production_helper_uses_a_business_month_rule(self):
        from src.services.market_data import monthly_ohlc
        import inspect
        src = inspect.getsource(monthly_ohlc)
        assert '"BME"' in src or "'BME'" in src, (
            "monthly_ohlc must resample BME; plain ME lets a month-ending "
            "Sunday decide the month's Close")

    def test_a_settled_months_close_survives_a_missing_weekend_row(self):
        a = _resample(self._daily_spanning_months(True), "BME")["Close"]
        b = _resample(self._daily_spanning_months(False), "BME")["Close"]
        common = a.index.intersection(b.index)
        assert len(common) >= 3
        moved = [str(d.date()) for d in common if a[d] != b[d]]
        assert not moved, f"months whose Close moved: {moved}"
