"""Unit tests for src/core/signals.py — pip sizing, structure, setup scoring."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.signals import (
    StopLossCalculator,
    evaluate_trend_following_signal,
    score_setup,
    swing_structure,
    typical_spread,
)


class TestPipSize:
    @pytest.mark.parametrize(
        "pair,expected",
        [
            ("EUR/USD", 0.0001),
            ("GBP/USD", 0.0001),
            ("USD/JPY", 0.01),
            ("EUR/JPY", 0.01),
            ("XAU/USD", 0.10),
            ("BTC/USD", 1.0),
            ("USD/ZAR", 0.001),
        ],
    )
    def test_pip_size(self, pair, expected):
        assert StopLossCalculator.pip_size(pair) == expected


class TestPriceToPips:
    def test_majors(self):
        calc = StopLossCalculator()
        # 50 pips on a 0.0001 pip instrument = 0.0050 price distance.
        assert calc.price_to_pips("EUR/USD", 0.0050) == pytest.approx(50.0)

    def test_jpy(self):
        calc = StopLossCalculator()
        assert calc.price_to_pips("USD/JPY", 0.50) == pytest.approx(50.0)

    def test_rounds_to_one_dp(self):
        calc = StopLossCalculator()
        assert calc.price_to_pips("EUR/USD", 0.00123) == pytest.approx(12.3)


class TestTypicalSpread:
    def test_known_pair(self):
        assert typical_spread("EUR/USD") == pytest.approx(1.2)

    def test_unknown_pair_is_zero(self):
        assert typical_spread("ZZZ/ZZZ") == 0.0


class TestSwingStructure:
    def test_rising_zigzag_is_bullish(self, rising_zigzag):
        assert swing_structure(rising_zigzag, 3) == "BULLISH"

    def test_falling_zigzag_is_bearish(self, falling_zigzag):
        assert swing_structure(falling_zigzag, 3) == "BEARISH"

    def test_short_frame_is_neutral(self, rising_zigzag):
        assert swing_structure(rising_zigzag.head(10), 3) == "NEUTRAL"

    def test_flat_market_is_neutral(self):
        df = pd.DataFrame({"High": [100.0] * 40, "Low": [99.0] * 40,
                           "Close": [99.5] * 40})
        assert swing_structure(df, 3) == "NEUTRAL"


class TestScoreSetup:
    def _frame(self, start, stop, n):
        close = np.linspace(start, stop, n)
        return pd.DataFrame(
            {"Open": close, "High": close + 0.5, "Low": close - 0.5,
             "Close": close, "Volume": np.full(n, 1000.0)}
        )

    def test_output_contract(self):
        res = score_setup(
            df_weekly=self._frame(1.00, 1.20, 260),
            df_daily=self._frame(1.10, 1.20, 120),
            df_4h=self._frame(1.18, 1.20, 120),
            direction="LONG",
            pip_size=0.0001,
            spread_pips=1.2,
        )
        # score/max_score are direction-only (7 criteria without Currency
        # Strength); the 3 quality-gate criteria (ATR Volatile, 4H Zone,
        # Spread/ATR) still appear in scores/details but are excluded here.
        assert res["max_score"] == 7
        assert 0 <= res["score"] <= 7
        assert res["grade"] in {"A", "B", "C", "D"}
        assert res["pct"] == int(round(res["score"] / 7 * 100))
        assert res["total_max"] == 10
        assert res["quality_max"] == 3
        assert res["quality_score"] + res["score"] == res["total_score"]
        assert set(res["scores"]) == set(res["details"])
        assert all(v in (0, 1) for v in res["scores"].values())

    def test_strong_uptrend_long_scores_well(self):
        res = score_setup(
            df_weekly=self._frame(1.00, 1.30, 260),
            df_daily=self._frame(1.10, 1.30, 120),
            df_4h=self._frame(1.28, 1.30, 120),
            direction="LONG",
            pip_size=0.0001,
            spread_pips=1.0,
        )
        # A clean uptrend should satisfy the weekly/daily EMA & trend checks.
        assert res["scores"]["Weekly EMA"] == 1
        assert res["scores"]["Daily Trend"] == 1

    # ── Daily 200MA regime filter ────────────────────────────────────────
    # Scored, never a veto (a new trend always starts by crossing the 200), and
    # omitted rather than failed when there isn't enough daily history.
    def _score(self, daily, direction, **kw):
        return score_setup(
            df_weekly=self._frame(1.00, 1.20, 260),
            df_daily=daily,
            df_4h=self._frame(1.18, 1.20, 120),
            direction=direction, pip_size=0.0001, spread_pips=1.0, **kw)

    def test_daily_200ma_rewards_long_above_the_average(self):
        rising = self._frame(1.00, 1.30, 250)      # last close well above SMA200
        assert self._score(rising, "LONG")["scores"]["Daily 200MA"] == 1
        assert self._score(rising, "SHORT")["scores"]["Daily 200MA"] == 0

    def test_daily_200ma_rewards_short_below_the_average(self):
        falling = self._frame(1.30, 1.00, 250)     # last close well below SMA200
        assert self._score(falling, "SHORT")["scores"]["Daily 200MA"] == 1
        assert self._score(falling, "LONG")["scores"]["Daily 200MA"] == 0

    def test_daily_200ma_omitted_when_history_too_short(self):
        # 120 daily bars can't seed a 200 average — the criterion must vanish
        # rather than score 0, so a thin-history instrument isn't penalised.
        res = self._score(self._frame(1.10, 1.20, 120), "LONG")
        assert "Daily 200MA" not in res["scores"]
        assert res["max_score"] == 7

    def test_daily_200ma_widens_the_direction_scale(self):
        res = self._score(self._frame(1.00, 1.30, 250), "LONG")
        assert "Daily 200MA" in res["scores"]
        assert res["max_score"] == 8            # 7 + the regime filter
        assert res["pct"] == int(round(res["score"] / 8 * 100))

    def test_daily_200ma_and_currency_strength_both_scale(self):
        res = self._score(self._frame(1.00, 1.30, 250), "LONG",
                          currency_strength_diff=0.5)
        assert res["max_score"] == 9            # 7 + 200MA + currency strength

    def test_daily_200ma_detail_reports_actual_side(self):
        # The label must describe reality, not the bullish case — the bug that
        # made a correct SHORT read "EMA20>50".
        falling = self._frame(1.30, 1.00, 250)
        for direction in ("LONG", "SHORT"):
            assert "price <" in self._score(falling, direction)["details"]["Daily 200MA"]

    def test_empty_frames_score_zero(self):
        res = score_setup(
            df_weekly=pd.DataFrame(),
            df_daily=pd.DataFrame(),
            df_4h=pd.DataFrame(),
            direction="LONG",
            pip_size=0.0001,
            spread_pips=0.0,
        )
        # With no data every directional check is 0 (Spread/ATR can no longer
        # rescue the direction score — it's a quality-gate criterion now).
        assert res["score"] == 0
        assert res["grade"] == "D"
        assert res["close"] == 0.0

    def test_none_frames_do_not_crash(self):
        res = score_setup(None, None, None, "SHORT", 0.0001, 0.0)
        assert res["max_score"] == 7
        assert 0 <= res["score"] <= 7

    def test_currency_strength_omitted_by_default(self):
        """Callers that don't pass currency_strength_diff stay on the
        original 7-criterion directional scale — no silent regression for
        unmigrated pages."""
        res = score_setup(
            df_weekly=self._frame(1.00, 1.20, 260),
            df_daily=self._frame(1.10, 1.20, 120),
            df_4h=self._frame(1.18, 1.20, 120),
            direction="LONG", pip_size=0.0001, spread_pips=1.2,
        )
        assert "Currency Strength" not in res["scores"]
        assert res["max_score"] == 7

    def test_currency_strength_adds_eighth_direction_criterion(self):
        res = score_setup(
            df_weekly=self._frame(1.00, 1.20, 260),
            df_daily=self._frame(1.10, 1.20, 120),
            df_4h=self._frame(1.18, 1.20, 120),
            direction="LONG", pip_size=0.0001, spread_pips=1.2,
            currency_strength_diff=1.5,
        )
        assert res["max_score"] == 8
        assert res["scores"]["Currency Strength"] == 1
        assert res["pct"] == int(round(res["score"] / 8 * 100))

    def test_currency_strength_disagreeing_direction_fails(self):
        res = score_setup(
            df_weekly=self._frame(1.00, 1.20, 260),
            df_daily=self._frame(1.10, 1.20, 120),
            df_4h=self._frame(1.18, 1.20, 120),
            direction="LONG", pip_size=0.0001, spread_pips=1.2,
            currency_strength_diff=-1.5,
        )
        assert res["scores"]["Currency Strength"] == 0

    def test_grade_bands_are_percentage_based(self):
        # 7/8 direction criteria = 87.5% still clears the 80% Grade-A bar that
        # 6/7 (85.7%) also would — percentage, not a fixed score range.
        res = score_setup(
            df_weekly=self._frame(1.00, 1.30, 260),
            df_daily=self._frame(1.10, 1.30, 120),
            df_4h=self._frame(1.28, 1.30, 120),
            direction="LONG", pip_size=0.0001, spread_pips=1.0,
            currency_strength_diff=1.0,
        )
        if res["score"] / res["max_score"] >= 0.8:
            assert res["grade"] == "A"


class TestEvaluateTrendFollowingSignal:
    def _frame(self, start, stop, n=260):
        close = np.linspace(start, stop, n)
        return pd.DataFrame(
            {"Open": close, "High": close + 0.5, "Low": close - 0.5,
             "Close": close, "Volume": np.full(n, 1000.0)}
        )

    def test_too_few_bars_neutral(self):
        label, score, max_score, conds, direction = evaluate_trend_following_signal(
            self._frame(100, 110, 50)
        )
        assert direction == "NEUTRAL"
        assert max_score == 6
        assert conds == {}

    def test_strong_uptrend_returns_buy_side(self):
        label, score, max_score, conds, direction = evaluate_trend_following_signal(
            self._frame(100, 200, 260)
        )
        assert direction in {"BUY", "STRONG_BUY"}
        assert score >= 4
        assert max_score == 6

    def test_strong_downtrend_returns_sell_side(self):
        label, score, max_score, conds, direction = evaluate_trend_following_signal(
            self._frame(200, 100, 260)
        )
        assert direction in {"SELL", "STRONG_SELL"}
        assert score >= 4
