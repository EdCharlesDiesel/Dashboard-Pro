"""Unit tests for the multi-timeframe idea generators in src/core/signals.py.

These exercise analyze_multi_timeframe via the public generate_* entry points
with synthetic multi-timeframe data for the full instrument universe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.config import default_config as config
from src.core.signals import generate_trading_ideas, generate_weekly_swing_ideas

_TF_BARS = {"Weekly": 120, "Daily": 200, "4 Hour": 200, "Hourly": 200, "15 Minute": 200}


def _frame(n: int, lo: float, hi: float) -> pd.DataFrame:
    c = np.linspace(lo, hi, n)
    band = (hi - lo) * 0.01 + 0.001
    return pd.DataFrame(
        {"Open": c, "High": c + band, "Low": c - band, "Close": c,
         "Volume": np.full(n, 1000.0)}
    )


@pytest.fixture
def uptrend_universe():
    """A clean uptrend across every asset and every timeframe."""
    return {tf: {pair: _frame(n, 1.00, 1.30) for pair in config.assets}
            for tf, n in _TF_BARS.items()}


class TestGenerateTradingIdeas:
    def test_empty_data_skips_everything(self):
        ideas, skipped = generate_trading_ideas({})
        assert ideas == []
        assert len(skipped) == len(config.assets)
        assert all("insufficient bars" in s for s in skipped)

    def test_rich_uptrend_produces_ideas(self, uptrend_universe):
        ideas, skipped = generate_trading_ideas(uptrend_universe)
        assert skipped == []
        assert ideas  # at least one idea generated
        idea = ideas[0]
        for key in ("pair", "bias", "conviction", "strength_score",
                    "entry", "stop_loss", "take_profit_1", "risk_reward_1"):
            assert key in idea
        assert idea["bias"] in {"Long", "Short"}
        assert 0 <= idea["strength_score"] <= 10

    def test_ideas_sorted_by_conviction_then_strength(self, uptrend_universe):
        ideas, _ = generate_trading_ideas(uptrend_universe)
        keys = [(i["conviction"] == "High", i["strength_score"]) for i in ideas]
        assert keys == sorted(keys, reverse=True)


class TestGenerateWeeklySwingIdeas:
    def test_no_weekly_data_returns_empty(self):
        assert generate_weekly_swing_ideas({}) == []
        assert generate_weekly_swing_ideas({"Daily": {}}) == []

    def test_returns_list_for_rich_data(self, uptrend_universe):
        ideas = generate_weekly_swing_ideas(uptrend_universe)
        assert isinstance(ideas, list)
        # Each idea (if any survive the R:R filter) is a well-formed dict.
        for idea in ideas:
            assert idea["bias"] in {"Long", "Short"}
            assert idea["rr1"] >= 1.5
            assert {"pair", "entry", "sl", "tp1", "tp2", "tp3"} <= set(idea)

    def test_thin_weekly_pair_skipped(self):
        # Weekly present but too few bars → no ideas, no crash.
        data = {"Weekly": {p: _frame(10, 1.0, 1.3) for p in config.assets},
                "Daily": {p: _frame(120, 1.0, 1.3) for p in config.assets}}
        assert generate_weekly_swing_ideas(data) == []
