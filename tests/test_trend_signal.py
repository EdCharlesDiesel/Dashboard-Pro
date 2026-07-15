"""Unit tests for src/indicators/trend_signal.py — 6-condition trend scorer.

Only the pure `evaluate()` path is exercised; `fetch()` hits yfinance and is
intentionally not unit-tested here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.trend_signal import TrendSignal, TrendSignalEvaluator


def _frame(**cols) -> pd.DataFrame:
    """Build a single-row indicator frame; the evaluator reads only the last row.

    Padded to >=10 rows so the length guard passes.
    """
    base = {
        "Close": 100.0, "EMA50": 100.0, "EMA200": 100.0,
        "RSI": 50.0, "MACD": 0.0, "MACDSig": 0.0, "ADX": 30.0,
    }
    base.update(cols)
    return pd.DataFrame([base] * 12)


class TestEvaluate:
    def test_too_few_bars_is_neutral(self):
        df = _frame().head(5)
        sig = TrendSignalEvaluator.evaluate(df)
        assert isinstance(sig, TrendSignal)
        assert sig.direction == "NEUTRAL"
        assert sig.score == 0

    def test_all_six_buy_conditions_strong_buy(self):
        df = _frame(
            Close=110.0, EMA50=105.0, EMA200=100.0,  # price > 200, 50>200, price>50
            RSI=55.0,                                 # 45–70
            MACD=1.0, MACDSig=0.5,                    # macd > sig
            ADX=30.0,                                 # > 25
        )
        sig = TrendSignalEvaluator.evaluate(df)
        assert sig.direction == "STRONG_BUY"
        assert sig.score == 6
        assert sig.max_score == 6
        assert "STRONG BUY" in sig.label

    def test_all_six_sell_conditions_strong_sell(self):
        df = _frame(
            Close=90.0, EMA50=95.0, EMA200=100.0,
            RSI=40.0,                                 # 30–45 bearish band
            MACD=-1.0, MACDSig=-0.5,
            ADX=30.0,
        )
        sig = TrendSignalEvaluator.evaluate(df)
        assert sig.direction == "STRONG_SELL"
        assert sig.score == 6

    def test_four_conditions_is_plain_buy(self):
        # price>200, 50>200, price>=50 (3 conds) + ADX>25 passes the gate and
        # itself counts as a 4th — RSI and MACD both fail, keeping it at 4/6
        # (plain BUY), not 5/6 (STRONG).
        df = _frame(
            Close=110.0, EMA50=105.0, EMA200=100.0,  # 3 conds
            RSI=80.0,                                 # out of 45-70 band (fail)
            MACD=-1.0, MACDSig=0.5,                   # macd<sig (fail)
            ADX=30.0,                                 # trending (gate pass, +1)
        )
        sig = TrendSignalEvaluator.evaluate(df)
        assert sig.direction == "BUY"
        assert sig.score == 4
        assert "📈 BUY" == sig.label

    def test_min_conditions_threshold_respected(self):
        # Same 4/6 setup as above but require 5 → falls through to NEUTRAL
        # (ADX still passes the gate; this isolates the min_conditions check).
        df = _frame(
            Close=110.0, EMA50=105.0, EMA200=100.0,
            RSI=80.0, MACD=-1.0, MACDSig=0.5, ADX=30.0,
        )
        sig = TrendSignalEvaluator.evaluate(df, min_conditions=5)
        assert sig.direction == "NEUTRAL"

    def test_conflicting_signals_resolve_neutral(self):
        # Flat market: EMAs equal, RSI 50, macd==sig, weak ADX.
        sig = TrendSignalEvaluator.evaluate(_frame(ADX=10.0))
        assert sig.direction == "NEUTRAL"

    def test_adx_gates_a_five_of_six_whipsaw_setup(self):
        # The reported bug: golden cross + price above EMAs + bullish RSI +
        # MACD positive all line up (5/6) but ADX says the market is ranging
        # (20.9, matching the GBP/USD case) — this must NOT read as a
        # directional signal, let alone STRONG BUY. It's exactly the setup
        # that whipsaws: those conditions are satisfied at range tops right
        # before mean reversion.
        df = _frame(
            Close=110.0, EMA50=105.0, EMA200=100.0,  # 3 conds
            RSI=55.0,                                 # +1 = 4
            MACD=1.0, MACDSig=0.5,                    # +1 = 5 of the non-ADX 5
            ADX=20.9,                                 # RANGE, not TREND
        )
        sig = TrendSignalEvaluator.evaluate(df)
        assert sig.direction == "NEUTRAL"
        assert "STRONG" not in sig.label
        assert "NO TREND" in sig.label

    def test_as_tuple_shape(self):
        sig = TrendSignalEvaluator.evaluate(_frame())
        label, score, max_score, conds, direction = sig.as_tuple()
        assert max_score == 6
        assert isinstance(conds, dict)
        assert direction == sig.direction
