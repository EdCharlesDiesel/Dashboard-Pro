"""Unit tests for the SL/TP calculators in src/core/signals.py.

Exact stop/target values and method labels were captured from the live
functions, then asserted here so any behavioural drift is caught.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.core.signals import StopLossCalculator, TakeProfitCalculator


def _flat(low: float, high: float, n: int = 25) -> pd.DataFrame:
    """Constant-band OHLC frame long enough to clear the lookback guard."""
    mid = (low + high) / 2
    return pd.DataFrame({"Low": [low] * n, "High": [high] * n, "Close": [mid] * n})


# ── StopLossCalculator ───────────────────────────────────────────────────────
class TestSwingStop:
    def test_long_uses_min_low(self):
        calc = StopLossCalculator()
        df = pd.DataFrame({"Low": [1.0, 0.9, 1.1] * 7, "High": [2.0] * 21})
        assert calc.get_swing_stop(df, "Long", 20) == pytest.approx(0.9)

    def test_short_uses_max_high(self):
        calc = StopLossCalculator()
        df = pd.DataFrame({"Low": [1.0] * 21, "High": [2.0, 2.5, 1.9] * 7})
        assert calc.get_swing_stop(df, "Short", 20) == pytest.approx(2.5)

    def test_short_frame_returns_none(self):
        calc = StopLossCalculator()
        assert calc.get_swing_stop(_flat(1.0, 2.0, 5), "Long", 20) is None


class TestStopLossCalculate:
    def setup_method(self):
        self.calc = StopLossCalculator()

    def test_long_atr_stop(self):
        # Swing low (1.0990) is tighter than the ATR stop, so ATR wins.
        r = self.calc.calculate(_flat(1.0990, 1.1010), "EUR/USD", "Long", 1.1000, 0.0010)
        assert r["method"] == "ATR"
        assert r["stop"] == pytest.approx(1.0985)
        assert r["distance_pips"] == pytest.approx(15.0)

    def test_long_swing_low_widens_stop(self):
        r = self.calc.calculate(_flat(1.0960, 1.1010), "EUR/USD", "Long", 1.1000, 0.0010)
        assert r["method"] == "Swing Low"
        assert r["stop"] == pytest.approx(1.09575)
        assert r["distance_pips"] == pytest.approx(42.5)

    def test_long_pivot_s1(self):
        r = self.calc.calculate(
            _flat(1.0990, 1.1010), "EUR/USD", "Long", 1.1000, 0.0010,
            pivots={"S1": 1.0980, "R1": 1.1020},
        )
        assert r["method"] == "Pivot S1"
        assert r["stop"] == pytest.approx(1.0980)
        assert r["distance_pips"] == pytest.approx(20.0)

    def test_min_distance_enforced(self):
        # Tiny ATR → raw stop is inside min_dist (0.0010) → enforced.
        r = self.calc.calculate(_flat(1.0999, 1.1001), "EUR/USD", "Long", 1.1000, 0.0001)
        assert "min-dist enforced" in r["method"]
        assert r["stop"] == pytest.approx(1.0990)
        assert r["distance_pips"] == pytest.approx(10.0)

    def test_short_atr_stop_is_above_price(self):
        r = self.calc.calculate(_flat(1.0990, 1.1010), "EUR/USD", "Short", 1.1000, 0.0010)
        assert r["method"] == "ATR"
        assert r["stop"] == pytest.approx(1.1015)
        assert r["distance_pips"] == pytest.approx(15.0)

    def test_short_pivot_r1(self):
        r = self.calc.calculate(
            _flat(1.0990, 1.1010), "EUR/USD", "Short", 1.1000, 0.0010,
            pivots={"R1": 1.1020, "S1": 1.0980},
        )
        assert r["method"] == "Pivot R1"
        assert r["stop"] == pytest.approx(1.1020)

    def test_unknown_pair_uses_default_multiplier(self):
        # No entry in pair_atr_multipliers → falls back to config.atr_sl_mult (1.5).
        r = self.calc.calculate(_flat(1.0990, 1.1010), "ZZZ/ZZZ", "Long", 1.1000, 0.0010)
        assert r["stop"] == pytest.approx(1.0985)


# ── TakeProfitCalculator ─────────────────────────────────────────────────────
class TestSwingTarget:
    def test_long_uses_max_high(self):
        calc = TakeProfitCalculator()
        df = pd.DataFrame({"Low": [1.0] * 21, "High": [2.0, 2.7, 1.9] * 7})
        assert calc.get_swing_target(df, "Long", 20) == pytest.approx(2.7)

    def test_short_uses_min_low(self):
        calc = TakeProfitCalculator()
        df = pd.DataFrame({"Low": [1.0, 0.7, 1.1] * 7, "High": [2.0] * 21})
        assert calc.get_swing_target(df, "Short", 20) == pytest.approx(0.7)

    def test_short_frame_returns_none(self):
        calc = TakeProfitCalculator()
        assert calc.get_swing_target(_flat(1.0, 2.0, 5), "Long", 20) is None


class TestTakeProfitCalculate:
    def setup_method(self):
        self.calc = TakeProfitCalculator()

    def test_long_atr_targets(self):
        r = self.calc.calculate(_flat(1.0990, 1.1000), "EUR/USD", "Long", 1.1000, 0.0010, 1.0985)
        assert r["tp1"] == pytest.approx(1.103)
        assert r["tp2"] == pytest.approx(1.105)
        assert r["method_tp1"] == "ATR ×3.0"
        assert r["method_tp2"] == "ATR ×5.0"
        assert r["rr1"] == pytest.approx(2.0)
        assert r["rr2"] == pytest.approx(3.33)
        # rr1 is fractionally below 2.0 pre-rounding → invalid; rr2 clears min_rr.
        assert r["tp1_valid"] is False
        assert r["tp2_valid"] is True

    def test_long_pivot_targets(self):
        r = self.calc.calculate(
            _flat(1.0990, 1.1000), "EUR/USD", "Long", 1.1000, 0.0010, 1.0985,
            pivots={"R1": 1.1020, "R2": 1.1060, "R3": 1.1090, "S1": 0, "S2": 0, "S3": 0},
        )
        assert r["method_tp1"] == "Pivot R1"
        assert r["method_tp2"] == "Pivot R2"
        assert r["tp1"] == pytest.approx(1.102)
        assert r["tp2"] == pytest.approx(1.106)

    def test_long_swing_extends_tp2(self):
        r = self.calc.calculate(_flat(1.0990, 1.1080), "EUR/USD", "Long", 1.1000, 0.0010, 1.0985)
        assert r["method_tp2"] == "Swing High (ext)"
        assert r["tp2"] == pytest.approx(1.108)

    def test_invalid_rr_when_stop_far(self):
        r = self.calc.calculate(_flat(1.0990, 1.1000), "EUR/USD", "Long", 1.1000, 0.0010, 1.0970)
        assert r["rr1"] == pytest.approx(1.0)
        assert r["tp1_valid"] is False
        assert r["tp2_valid"] is False

    def test_stop_dist_falls_back_to_atr(self):
        # stop_loss == current_price → stop_dist would be 0 → uses atr instead.
        r = self.calc.calculate(_flat(1.0990, 1.1000), "EUR/USD", "Long", 1.1000, 0.0010, 1.1000)
        assert r["rr1"] == pytest.approx(3.0)
        assert r["tp1_valid"] is True

    def test_short_pivot_targets(self):
        r = self.calc.calculate(
            _flat(1.0990, 1.1010), "EUR/USD", "Short", 1.1000, 0.0010, 1.1015,
            pivots={"S1": 1.0980, "S2": 1.0940, "S3": 1.0910, "R1": 0, "R2": 0, "R3": 0},
        )
        assert r["method_tp1"] == "Pivot S1"
        assert r["method_tp2"] == "Pivot S2"
        assert r["tp1"] == pytest.approx(1.098)
        assert r["tp2"] == pytest.approx(1.094)
