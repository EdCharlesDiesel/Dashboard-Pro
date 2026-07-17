"""Unit tests for src/services/house_view_backtest.py.

The equivalence test is the load-bearing one: the vectorized walk-forward
must reproduce, day by day, exactly what src/core/bias.house_view returns
when run on frames truncated to that day — proving both rule fidelity and
the absence of lookahead.
"""
import numpy as np
import pandas as pd
import pytest

from src.core.bias import BEARISH, BULLISH, NEUTRAL, house_view
from src.services.house_view_backtest import (
    ALIGNED,
    ALL_DAYS,
    run_universe,
    summarize,
    walk_forward,
)
from src.services.market_data import _resample


def _daily(n: int, slope: float = 0.0004, start: float = 1.10,
           noise: float = 0.0, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    drift = start * np.exp(slope * np.arange(n))
    close = drift * np.exp(noise * rng.standard_normal(n)) if noise else drift
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame({
        "Open": close, "High": close * 1.001, "Low": close * 0.999,
        "Close": close, "Volume": np.zeros(n),
    }, index=idx)


class TestEquivalenceWithBiasEngine:
    @pytest.mark.parametrize("t", [280, 300, 333, 361, 420])
    def test_direction_matches_house_view_on_truncated_frames(self, t):
        # Noisy series so all three states actually occur.
        daily = _daily(430, slope=0.0002, noise=0.004)
        bt = walk_forward(daily)

        trunc = daily.iloc[: t + 1]
        weekly = _resample(trunc, "W")
        hv = house_view("X", weekly=weekly, daily=trunc)

        day = trunc.index[-1]
        assert bt.loc[day, "direction"] == hv.direction
        assert bool(bt.loc[day, "aligned"]) == hv.aligned
        assert bt.loc[day, "composite"] == pytest.approx(hv.score, abs=1e-9)

    def test_weekly_partial_bar_step_matches_midweek(self):
        # A Wednesday (mid-week) — the partial-week EMA step must still agree.
        daily = _daily(400, slope=-0.0003, noise=0.003, seed=11)
        bt = walk_forward(daily)
        midweek = [d for d in daily.index[300:340] if d.dayofweek == 2][0]
        t = daily.index.get_loc(midweek)
        trunc = daily.iloc[: t + 1]
        hv = house_view("X", weekly=_resample(trunc, "W"), daily=trunc)
        assert bt.loc[midweek, "direction"] == hv.direction


class TestWalkForward:
    def test_steady_uptrend_is_bullish_and_aligned(self):
        bt = walk_forward(_daily(400, slope=0.0006))
        tail = bt.tail(50)
        assert (tail["direction"] == BULLISH).all()
        assert tail["aligned"].all()

    def test_steady_downtrend_is_bearish(self):
        bt = walk_forward(_daily(400, slope=-0.0006))
        assert (bt.tail(50)["direction"] == BEARISH).all()

    def test_daily_only_until_weekly_history_arrives(self):
        # 60 daily bars = enough for the daily TF (≥50) but only ~12 weeks —
        # the weekly TF is unavailable and the composite is daily-only.
        bt = walk_forward(_daily(60, slope=0.0006))
        assert not bt.empty
        assert (bt["direction"] == BULLISH).any()

    def test_too_short_history_is_empty(self):
        assert walk_forward(_daily(30)).empty
        assert walk_forward(pd.DataFrame()).empty
        assert walk_forward(None).empty

    def test_forward_returns_are_future_relative(self):
        daily = _daily(400, slope=0.0006)
        bt = walk_forward(daily)
        day = bt.index[-30]
        h = 5
        i = daily.index.get_loc(day)
        expected = daily["Close"].iloc[i + h] / daily["Close"].iloc[i] - 1
        assert bt.loc[day, f"fwd_{h}"] == pytest.approx(expected)


class TestSummarize:
    def test_uptrend_bullish_hit_rate_is_high(self):
        bt = walk_forward(_daily(500, slope=0.0006))
        rep = summarize(bt)
        assert rep.loc[BULLISH, "n"] > 100
        assert rep.loc[BULLISH, "hit_10"] > 90
        assert rep.loc[BULLISH, "avg_10_bps"] > 0
        assert rep.loc[ALIGNED, "n"] <= rep.loc[BULLISH, "n"] + rep.loc[BEARISH, "n"]

    def test_downtrend_bearish_signed_returns_positive(self):
        # BEARISH days in a falling market: signed (short) returns are gains.
        rep = summarize(walk_forward(_daily(500, slope=-0.0006)))
        assert rep.loc[BEARISH, "avg_10_bps"] > 0
        assert rep.loc[BEARISH, "hit_10"] > 90

    def test_neutral_and_all_days_are_unsigned(self):
        rep = summarize(walk_forward(_daily(500, slope=0.0002, noise=0.004)))
        assert rep.loc[NEUTRAL, "hit_10"] is None or np.isnan(rep.loc[NEUTRAL, "hit_10"])
        assert rep.loc[ALL_DAYS, "n"] == rep["n"].drop(ALIGNED).drop(ALL_DAYS).sum()

    def test_empty_input(self):
        assert summarize(pd.DataFrame()).empty
        assert summarize(None).empty


class TestRunUniverse:
    def test_pools_across_pairs(self):
        bars = {
            "EUR/USD": _daily(400, slope=0.0006),
            "GBP/USD": _daily(400, slope=-0.0006),
            "XAU/USD": _daily(20),  # too short — skipped but reported empty
        }
        pooled, per_pair = run_universe(bars)
        assert not pooled.empty
        assert pooled.loc[ALL_DAYS, "n"] == (
            per_pair["EUR/USD"].loc[ALL_DAYS, "n"]
            + per_pair["GBP/USD"].loc[ALL_DAYS, "n"]
        )
        assert per_pair["XAU/USD"].empty
