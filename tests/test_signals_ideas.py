"""Unit tests for the multi-timeframe idea generators in src/core/signals.py.

These exercise analyze_multi_timeframe via the public generate_* entry points
with synthetic multi-timeframe data for the full instrument universe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import json

from src.core.config import default_config as config
from src.core.signals import (
    generate_trading_ideas,
    generate_weekly_swing_ideas,
    signal_to_setup_row,
)

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


# Columns TradeRepository.save_setup writes — the row from signal_to_setup_row
# must supply every one (a missing key would KeyError on the named INSERT).
_SAVE_SETUP_COLUMNS = {
    "logged_at", "instrument", "ticker", "direction", "session", "score",
    "verdict", "atr14", "atr20", "sl_pips", "tp1_pips", "tp2_pips", "lot_size",
    "risk_amount", "rr_tp1", "rr_tp2", "account_bal", "risk_pct",
    "checks_passed", "checks_total", "checks_detail", "notes",
}


class TestSignalToSetupRow:
    def _idea(self, **over):
        idea = {
            "pair": "EUR/USD", "bias": "Long", "conviction": "High",
            "strength_score": 9, "confidence": 4, "thesis": "Strong MTF uptrend",
            "entry": 1.10000, "take_profit_1": 1.10500, "take_profit_2": 1.11000,
            "stop_loss": 1.09700, "stop_loss_method": "ATR", "stop_loss_pips": 30.0,
            "risk_reward_1": 1.67, "risk_reward_2": 3.33, "atr": 0.0015,
        }
        idea.update(over)
        return idea

    def test_row_has_every_save_setup_column(self):
        row = signal_to_setup_row(self._idea(), "NY Kill Zone")
        assert _SAVE_SETUP_COLUMNS <= set(row)

    def test_maps_registry_ticker_and_core_fields(self):
        row = signal_to_setup_row(self._idea(), "NY Kill Zone")
        assert row["instrument"] == "EUR/USD"
        assert row["ticker"] == "EURUSD=X"      # resolved from the registry
        assert row["direction"] == "Long"
        assert row["session"] == "NY Kill Zone"
        assert row["score"] == "9/10"
        assert row["verdict"] == "High"
        assert row["rr_tp1"] == 1.67

    def test_targets_converted_to_pips(self):
        # EUR/USD pip_size 0.0001 → 1.10500-1.10000 = 50 pips, 1.11000 = 100 pips.
        row = signal_to_setup_row(self._idea(), "")
        assert row["tp1_pips"] == 50.0
        assert row["tp2_pips"] == 100.0
        assert row["sl_pips"] == 30.0           # passed through from the idea

    def test_sizing_fields_are_none(self):
        row = signal_to_setup_row(self._idea(), "")
        for col in ("lot_size", "risk_amount", "account_bal", "risk_pct"):
            assert row[col] is None

    def test_checks_detail_is_valid_json_with_context(self):
        row = signal_to_setup_row(self._idea(), "")
        detail = json.loads(row["checks_detail"])
        assert detail["stop_loss"] == 1.09700
        assert detail["thesis"] == "Strong MTF uptrend"
        assert detail["strength_score"] == 9

    def test_verdict_capped_at_db_column_width(self):
        # verdict maps to a VARCHAR(20) column — a long conviction must be capped.
        row = signal_to_setup_row(
            self._idea(conviction="BUY XAU/USD — STRONG CONVICTION"), "")
        assert len(row["verdict"]) <= 20

    def test_unknown_pair_degrades_without_crashing(self):
        row = signal_to_setup_row(self._idea(pair="ZZZ/ZZZ"), "")
        assert row["instrument"] == "ZZZ/ZZZ"
        # Changed 2026-08-04: this used to assert `ticker is None`. A NULL ticker
        # is not a graceful degradation — the Source Scorecard fetches price bars
        # *by ticker*, so those rows could never resolve and sat OPEN forever
        # while still counting toward their source's totals. Falling back to the
        # symbol itself lets a free-text ticker (Smart Money's "SPY") resolve; a
        # genuinely bogus one simply finds no bars, which is the same outcome as
        # before minus the silent accounting distortion.
        assert row["ticker"] == "ZZZ/ZZZ"
        assert row["tp1_pips"] == 50.0          # default pip_size 0.0001 used

    def test_real_idea_maps_cleanly(self, uptrend_universe):
        ideas, _ = generate_trading_ideas(uptrend_universe)
        assert ideas
        row = signal_to_setup_row(ideas[0], "London Kill Zone")
        assert _SAVE_SETUP_COLUMNS <= set(row)
        json.loads(row["checks_detail"])        # serialisable


class TestSignalToSetupRowRejectsNonFinite:
    """NaN in a signal used to silently destroy the row.

    `json.dumps` emits a bare ``NaN`` token — legal Python-flavoured JSON, but
    Postgres rejects it as JSONB, and `signal_store` catches the failure per row
    so the signal vanished with only a log line. Seen live: the Smart Money page
    read the last close of a yfinance frame that had an unfilled final bar.
    """

    def _idea(self, **over):
        idea = {
            "pair": "EUR/USD", "bias": "Long", "conviction": "High",
            "strength_score": 9, "entry": 1.10000, "take_profit_1": 1.10500,
        }
        idea.update(over)
        return idea

    def _detail(self, row):
        return json.loads(row["checks_detail"])

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_checks_detail_stays_valid_json(self, bad):
        row = signal_to_setup_row(self._idea(entry=bad), "NY Kill Zone")
        # The real assertion: json.loads must not choke, which is what Postgres does.
        assert self._detail(row)["entry"] == 0.0

    @pytest.mark.parametrize("field", ["take_profit_1", "take_profit_2", "stop_loss"])
    def test_non_finite_price_levels_become_null(self, field):
        row = signal_to_setup_row(self._idea(**{field: float("nan")}), "NY Kill Zone")
        assert self._detail(row)[field] is None

    @pytest.mark.parametrize("field,column", [
        ("atr", "atr14"), ("stop_loss_pips", "sl_pips"),
        ("risk_reward_1", "rr_tp1"), ("risk_reward_2", "rr_tp2"),
    ])
    def test_non_finite_numeric_columns_become_null(self, field, column):
        row = signal_to_setup_row(self._idea(**{field: float("nan")}), "NY Kill Zone")
        assert row[column] is None

    def test_nan_entry_does_not_produce_bogus_pip_distances(self):
        # entry falls back to 0.0, and _pips treats 0 entry as "no distance" —
        # a NaN entry must never yield a confident-looking pip number.
        row = signal_to_setup_row(self._idea(entry=float("nan")), "NY Kill Zone")
        assert row["tp1_pips"] is None

    def test_finite_values_are_untouched(self):
        row = signal_to_setup_row(self._idea(atr=0.0015, stop_loss_pips=30.0),
                                  "NY Kill Zone")
        assert row["atr14"] == 0.0015 and row["sl_pips"] == 30.0
        assert self._detail(row)["entry"] == 1.10000

    def test_booleans_are_not_coerced_to_numbers(self):
        row = signal_to_setup_row(self._idea(confidence=True), "NY Kill Zone")
        assert self._detail(row)["confidence"] is True


class TestSignalToSetupRowSegmentationFields:
    """Fields recorded so the scorecard can segment, added 2026-08-04."""

    def _row(self, **over):
        idea = {"pair": "EUR/USD", "bias": "Long", "entry": 1.1}
        idea.update(over)
        return signal_to_setup_row(idea, "NY Kill Zone")

    def test_quality_passed_is_recorded_not_enforced(self):
        # False must still produce a row — the point is to measure the gate,
        # not to let it silently drop signals before it has been validated.
        row = self._row(quality_passed=False)
        assert json.loads(row["checks_detail"])["quality_passed"] is False
        assert row["instrument"] == "EUR/USD"

    def test_horizon_days_is_recorded(self):
        assert json.loads(self._row(horizon_days=20)["checks_detail"])["horizon_days"] == 20

    def test_fields_default_to_none_when_page_does_not_supply_them(self):
        detail = json.loads(self._row()["checks_detail"])
        assert detail["quality_passed"] is None and detail["horizon_days"] is None


class TestFreeTextTickerFallback:
    """Smart Money takes any symbol; a NULL ticker made its rows unresolvable."""

    def test_non_registry_pair_falls_back_to_itself_as_ticker(self):
        row = signal_to_setup_row({"pair": "SPY", "bias": "Long", "entry": 600.0},
                                  "NY Kill Zone")
        assert row["ticker"] == "SPY"

    def test_registry_pair_still_uses_the_registry_ticker(self):
        row = signal_to_setup_row({"pair": "EUR/USD", "bias": "Long", "entry": 1.1},
                                  "NY Kill Zone")
        assert row["ticker"] == "EURUSD=X"

    def test_blank_pair_stays_null_rather_than_empty_string(self):
        row = signal_to_setup_row({"pair": "", "bias": "Long", "entry": 1.1},
                                  "NY Kill Zone")
        assert row["ticker"] is None
