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
        assert res["max_score"] == 10
        assert 0 <= res["score"] <= 10
        assert res["grade"] in {"A", "B", "C", "D"}
        assert res["pct"] == int(res["score"] / 10 * 100)
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

    def test_empty_frames_score_zero(self):
        res = score_setup(
            df_weekly=pd.DataFrame(),
            df_daily=pd.DataFrame(),
            df_4h=pd.DataFrame(),
            direction="LONG",
            pip_size=0.0001,
            spread_pips=0.0,
        )
        # With no data every data-dependent check is 0; only Spread/ATR can flip.
        assert res["score"] <= 1
        assert res["grade"] == "D"
        assert res["close"] == 0.0

    def test_none_frames_do_not_crash(self):
        res = score_setup(None, None, None, "SHORT", 0.0001, 0.0)
        assert res["max_score"] == 10
        assert 0 <= res["score"] <= 10


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
