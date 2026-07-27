"""Unit tests for src/core/volume_profile.py — the fixed-range volume profile.

All synthetic: no network, no Streamlit, no DB. The frames are built so the
expected POC/value area is known by construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.volume_profile import (
    DEFAULT_VALUE_AREA,
    FX_DAY_BREAK_HOUR,
    SESSION_WINDOWS,
    ProfileRead,
    RangeWindow,
    VolumeProfile,
    activity_series,
    fixed_range_profile,
    has_usable_volume,
    last_n_bars_range,
    previous_day_range,
    previous_session_range,
    read_profile,
    session_breaks,
    session_day,
    true_range,
    value_area,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _hourly_index(days: int = 3, start: str = "2026-01-05 00:00") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=days * 24, freq="h", tz="UTC")


def _frame(index, highs, lows, opens, closes, volumes) -> pd.DataFrame:
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows,
         "Close": closes, "Volume": volumes},
        index=index,
    )


@pytest.fixture
def flat_frame() -> pd.DataFrame:
    """72 hourly bars, all spanning the same 1.0-wide band around 100."""
    idx = _hourly_index(3)
    n = len(idx)
    return _frame(idx,
                  highs=np.full(n, 100.5), lows=np.full(n, 99.5),
                  opens=np.full(n, 100.0), closes=np.full(n, 100.2),
                  volumes=np.full(n, 1000.0))


@pytest.fixture
def concentrated_frame() -> pd.DataFrame:
    """Most bars trade a tight band at 100; a handful spike away from it.

    The POC must land in the tight band, not at the spikes.
    """
    idx = _hourly_index(3)
    n = len(idx)
    highs = np.full(n, 100.10)
    lows = np.full(n, 99.90)
    # Six outliers stretch the range to [98, 102] but carry little activity.
    highs[:3] = 102.0
    lows[:3] = 101.5
    highs[3:6] = 98.5
    lows[3:6] = 98.0
    vols = np.full(n, 1000.0)
    vols[:6] = 1.0
    return _frame(idx, highs=highs, lows=lows,
                  opens=np.full(n, 100.0), closes=np.full(n, 100.05),
                  volumes=vols)


# --------------------------------------------------------------------------- #
# value_area — the Market Profile expansion
# --------------------------------------------------------------------------- #
class TestValueArea:
    def test_single_dominant_bin_satisfies_target_alone(self):
        vols = np.array([1.0, 1.0, 10.0, 1.0, 1.0])   # total 14, target 9.8
        lo, poc, hi = value_area(vols, 0.70)
        assert (lo, poc, hi) == (2, 2, 2)

    def test_expands_two_rows_at_a_time_toward_the_heavier_side(self):
        # total 22, target 15.4. POC=3 (10). First step: up pair (3+2=5) ties
        # the down pair (2+3=5) and ties go up -> acc 15, hi 5. Still short, so
        # a second step compares up (1) vs down (5) and takes down.
        vols = np.array([1.0, 2.0, 3.0, 10.0, 3.0, 2.0, 1.0])
        lo, poc, hi = value_area(vols, 0.70)
        assert (lo, poc, hi) == (1, 3, 5)

    def test_full_pct_covers_everything(self):
        vols = np.array([1.0, 2.0, 3.0, 4.0])
        lo, poc, hi = value_area(vols, 1.0)
        assert (lo, hi) == (0, 3)
        assert poc == 3

    def test_zero_activity_collapses_to_poc(self):
        lo, poc, hi = value_area(np.zeros(5), 0.70)
        assert lo == poc == hi == 0

    def test_empty(self):
        assert value_area(np.array([]), 0.70) == (0, 0, 0)

    def test_never_runs_off_either_end(self):
        # A POC at the very edge must still terminate and stay in bounds.
        vols = np.array([10.0, 1.0, 1.0])
        lo, poc, hi = value_area(vols, 0.99)
        assert 0 <= lo <= poc <= hi <= 2


# --------------------------------------------------------------------------- #
# Activity source
# --------------------------------------------------------------------------- #
class TestActivitySource:
    def test_zero_volume_is_not_usable(self):
        assert not has_usable_volume(pd.Series([0, 0, 0.0]))
        assert not has_usable_volume(pd.Series([np.nan, np.nan]))

    def test_real_volume_is_usable(self):
        assert has_usable_volume(pd.Series([0, 0, 5.0]))

    def test_fx_frame_falls_back_to_range_proxy(self, flat_frame):
        fx = flat_frame.copy()
        fx["Volume"] = 0.0                     # spot FX on Yahoo
        series, is_proxy = activity_series(fx)
        assert is_proxy
        # Every bar spans 1.0, so the proxy is the bar range.
        assert series.iloc[-1] == pytest.approx(1.0)

    def test_real_volume_wins_when_present(self, flat_frame):
        series, is_proxy = activity_series(flat_frame)
        assert not is_proxy
        assert series.iloc[0] == pytest.approx(1000.0)

    def test_use_volume_override_forces_the_proxy(self, flat_frame):
        _, is_proxy = activity_series(flat_frame, use_volume=False)
        assert is_proxy

    def test_true_range_uses_prev_close_gaps(self):
        idx = pd.date_range("2026-01-05", periods=2, freq="h", tz="UTC")
        df = _frame(idx, highs=[10.0, 20.0], lows=[9.0, 19.0],
                    opens=[9.5, 19.5], closes=[9.8, 19.8], volumes=[1, 1])
        tr = true_range(df)
        assert tr.iloc[0] == pytest.approx(1.0)       # no prev close
        assert tr.iloc[1] == pytest.approx(10.2)      # 20.0 - 9.8


# --------------------------------------------------------------------------- #
# Session ranges
# --------------------------------------------------------------------------- #
class TestSessionRanges:
    def test_session_day_rolls_at_the_break_hour(self):
        idx = pd.DatetimeIndex(["2026-01-05 20:00", "2026-01-05 21:00",
                                "2026-01-06 08:00"]).tz_localize("UTC")
        days = session_day(idx, day_break_hour=21)
        # 20:00 is still the previous FX day; 21:00 and next-morning are the same one.
        assert days[0] != days[1]
        assert days[1] == days[2]

    def test_naive_index_is_treated_as_utc(self):
        naive = pd.date_range("2026-01-05", periods=5, freq="h")
        assert len(session_day(naive)) == 5

    def test_session_breaks_marks_each_new_day(self, flat_frame):
        breaks = session_breaks(flat_frame.index, day_break_hour=0)
        assert len(breaks) == 2                       # 3 days => 2 interior breaks
        assert all(b.hour == 0 for b in breaks)

    def test_session_breaks_empty_index(self):
        assert session_breaks(pd.DatetimeIndex([])) == []

    def test_previous_day_range_skips_the_in_progress_day(self, flat_frame):
        win = previous_day_range(flat_frame.index, back=1, day_break_hour=0)
        assert win is not None
        # Data covers Jan 5/6/7; Jan 7 is still forming, so back=1 is Jan 6.
        assert win.start.date() == pd.Timestamp("2026-01-06").date()
        assert win.start.hour == 0 and win.end.hour == 23
        assert "2026-01-06" in win.label

    def test_previous_day_range_back_two(self, flat_frame):
        win = previous_day_range(flat_frame.index, back=2, day_break_hour=0)
        assert win.start.date() == pd.Timestamp("2026-01-05").date()

    def test_previous_day_range_needs_history(self):
        idx = pd.date_range("2026-01-05", periods=5, freq="h", tz="UTC")
        assert previous_day_range(idx, back=3, day_break_hour=0) is None

    def test_previous_day_range_empty(self):
        assert previous_day_range(pd.DatetimeIndex([])) is None

    def test_previous_session_range_picks_the_kill_zone_hours(self, flat_frame):
        win = previous_session_range(flat_frame.index, "London KZ",
                                     back=1, day_break_hour=0)
        assert win is not None
        lo, hi = SESSION_WINDOWS["London KZ"]
        assert win.start.hour == int(lo)
        assert win.end.hour == int(hi) - 1             # last bar inside the window
        assert win.label.startswith("London KZ")

    def test_previous_session_range_unknown_session(self, flat_frame):
        assert previous_session_range(flat_frame.index, "Frankfurt") is None

    def test_previous_session_range_drops_an_in_progress_block(self):
        # Index ends at 08:00, mid-London-KZ (07:00-09:00) — that block is live.
        idx = pd.date_range("2026-01-05 00:00", "2026-01-06 08:00",
                            freq="h", tz="UTC")
        win = previous_session_range(idx, "London KZ", back=1, day_break_hour=0)
        assert win is not None
        assert win.start.date() == pd.Timestamp("2026-01-05").date()

    def test_last_n_bars_range(self, flat_frame):
        win = last_n_bars_range(flat_frame.index, 10)
        assert win.start == flat_frame.index[-10]
        assert win.end == flat_frame.index[-1]
        assert win.label == "last 10 bars"

    def test_last_n_bars_range_clamps_to_available(self, flat_frame):
        win = last_n_bars_range(flat_frame.index, 10_000)
        assert win.start == flat_frame.index[0]

    def test_range_window_duration(self):
        a = pd.Timestamp("2026-01-05 00:00", tz="UTC")
        b = pd.Timestamp("2026-01-05 06:00", tz="UTC")
        assert RangeWindow(a, b, "x").duration == pd.Timedelta(hours=6)


# --------------------------------------------------------------------------- #
# fixed_range_profile
# --------------------------------------------------------------------------- #
class TestFixedRangeProfile:
    def test_poc_lands_in_the_concentrated_band(self, concentrated_frame):
        prof = fixed_range_profile(concentrated_frame, bins=40)
        assert prof is not None
        # 99.90-100.10 is where the activity is, despite a 98-102 total range.
        assert 99.8 <= prof.poc <= 100.2

    def test_value_area_brackets_the_poc(self, concentrated_frame):
        prof = fixed_range_profile(concentrated_frame, bins=40)
        assert prof.val <= prof.poc <= prof.vah
        assert prof.va_low_index <= prof.poc_index <= prof.va_high_index

    def test_value_area_holds_at_least_the_requested_share(self, concentrated_frame):
        prof = fixed_range_profile(concentrated_frame, bins=40,
                                   value_area_pct=DEFAULT_VALUE_AREA)
        inside = prof.total[prof.va_low_index:prof.va_high_index + 1].sum()
        assert inside / prof.total.sum() >= DEFAULT_VALUE_AREA - 1e-9

    def test_range_edges_match_the_window_extremes(self, concentrated_frame):
        prof = fixed_range_profile(concentrated_frame, bins=40)
        assert prof.range_low == pytest.approx(float(concentrated_frame["Low"].min()))
        assert prof.range_high == pytest.approx(float(concentrated_frame["High"].max()))

    def test_buy_and_sell_sum_to_total(self, concentrated_frame):
        prof = fixed_range_profile(concentrated_frame, bins=40)
        assert np.allclose(prof.buy + prof.sell, prof.total)

    def test_up_closing_bars_count_as_buy(self, flat_frame):
        # Every bar closes above its open, so the profile is 100% buy.
        prof = fixed_range_profile(flat_frame, bins=20)
        assert prof.buy_share == pytest.approx(1.0)
        assert prof.sell.sum() == pytest.approx(0.0)

    def test_down_closing_bars_count_as_sell(self, flat_frame):
        down = flat_frame.copy()
        down["Close"] = down["Open"] - 0.1
        prof = fixed_range_profile(down, bins=20)
        assert prof.buy_share == pytest.approx(0.0)

    def test_window_restricts_the_bars_used(self, flat_frame):
        win = previous_day_range(flat_frame.index, back=1, day_break_hour=0)
        prof = fixed_range_profile(flat_frame, window=win, bins=20)
        assert prof.bars == 24                     # exactly one day of hourly bars
        assert prof.window is win

    def test_defaults_to_the_whole_frame(self, flat_frame):
        prof = fixed_range_profile(flat_frame, bins=20)
        assert prof.bars == len(flat_frame)
        assert prof.window.label == "full range"

    def test_fx_zero_volume_marks_the_profile_as_a_proxy(self, concentrated_frame):
        fx = concentrated_frame.copy()
        fx["Volume"] = 0.0
        prof = fixed_range_profile(fx, bins=40)
        assert prof is not None and prof.is_proxy
        # The proxy weights by bar range, so the wide outlier bars now matter
        # more — but the 66 tight bars still outnumber the 6 spikes.
        assert 99.8 <= prof.poc <= 100.2

    def test_bin_width_is_uniform(self, concentrated_frame):
        prof = fixed_range_profile(concentrated_frame, bins=40)
        assert prof.bin_width == pytest.approx(np.diff(prof.edges).mean())
        assert len(prof.centers) == 40
        assert len(prof.edges) == 41

    def test_empty_frame(self):
        assert fixed_range_profile(pd.DataFrame()) is None
        assert fixed_range_profile(None) is None

    def test_window_outside_the_data(self, flat_frame):
        far = RangeWindow(pd.Timestamp("2030-01-01", tz="UTC"),
                          pd.Timestamp("2030-01-02", tz="UTC"), "nowhere")
        assert fixed_range_profile(flat_frame, window=far) is None

    def test_flat_price_has_no_extent(self):
        idx = _hourly_index(1)
        n = len(idx)
        df = _frame(idx, highs=np.full(n, 100.0), lows=np.full(n, 100.0),
                    opens=np.full(n, 100.0), closes=np.full(n, 100.0),
                    volumes=np.full(n, 10.0))
        assert fixed_range_profile(df) is None

    def test_too_few_bins(self, flat_frame):
        assert fixed_range_profile(flat_frame, bins=1) is None


# --------------------------------------------------------------------------- #
# read_profile
# --------------------------------------------------------------------------- #
class TestReadProfile:
    @pytest.fixture
    def prof(self, concentrated_frame) -> VolumeProfile:
        return fixed_range_profile(concentrated_frame, bins=40)

    def test_above_value_is_bullish(self, prof):
        read = read_profile(prof, prof.vah + 0.5)
        assert read.state == "ABOVE VALUE"
        assert read.lean == "Bullish"
        assert read.dist_to_poc > 0

    def test_below_value_is_bearish(self, prof):
        read = read_profile(prof, prof.val - 0.5)
        assert read.state == "BELOW VALUE"
        assert read.lean == "Bearish"
        assert read.dist_to_poc < 0

    def test_inside_value_is_neutral(self, prof):
        read = read_profile(prof, prof.poc)
        assert read.state == "IN VALUE"
        assert read.lean == "Neutral"
        assert read.dist_to_poc == pytest.approx(0.0)

    def test_sitting_on_the_poc_is_called_out(self, prof):
        read = read_profile(prof, prof.poc)
        assert "POC" in read.headline

    def test_boundaries_are_exclusive(self, prof):
        # Exactly at VAH/VAL is still "in value" — acceptance means beyond.
        assert read_profile(prof, prof.vah).state == "IN VALUE"
        assert read_profile(prof, prof.val).state == "IN VALUE"

    def test_read_is_a_frozen_dataclass(self, prof):
        read = read_profile(prof, prof.poc)
        assert isinstance(read, ProfileRead)
        with pytest.raises(Exception):
            read.state = "X"          # type: ignore[misc]


class TestConstants:
    def test_session_windows_mirror_session_service(self):
        from src.services.session_service import SessionService
        assert SESSION_WINDOWS["London KZ"] == SessionService.LONDON_KZ
        assert SESSION_WINDOWS["NY KZ"] == SessionService.NY_KZ
        assert SESSION_WINDOWS["Tokyo"] == SessionService.TOKYO
        assert SESSION_WINDOWS["London Close"] == SessionService.LONDON_CLOSE

    def test_fx_day_break_is_the_ny_close(self):
        assert FX_DAY_BREAK_HOUR == 21
