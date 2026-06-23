"""Unit tests for src/services/correlation_service.py — stacked-exposure guard."""
from __future__ import annotations

from src.services.correlation_service import CorrelationService


def _trade(instrument: str, direction: str) -> dict:
    return {"instrument": instrument, "direction": direction}


class TestCheckExposure:
    def test_warns_on_stacked_usd_longs(self):
        # Long GBP/USD already open; adding long EUR/USD doubles USD risk.
        warnings = CorrelationService.check_exposure(
            open_trades=[_trade("GBP/USD", "Long")],
            direction="Long",
            instrument="EUR/USD",
        )
        assert len(warnings) == 1
        assert "GBP/USD" in warnings[0]["conflicting"]
        assert warnings[0]["count"] == 1
        assert "USD vs majors" in warnings[0]["group"]

    def test_opposite_direction_does_not_warn(self):
        warnings = CorrelationService.check_exposure(
            open_trades=[_trade("GBP/USD", "Short")],
            direction="Long",
            instrument="EUR/USD",
        )
        assert warnings == []

    def test_same_instrument_is_ignored(self):
        # An existing trade in the SAME instrument is not a correlation conflict.
        warnings = CorrelationService.check_exposure(
            open_trades=[_trade("EUR/USD", "Long")],
            direction="Long",
            instrument="EUR/USD",
        )
        assert warnings == []

    def test_uncorrelated_instrument_no_warning(self):
        warnings = CorrelationService.check_exposure(
            open_trades=[_trade("EUR/USD", "Long")],
            direction="Long",
            instrument="EUR/AUD",  # in no CORR_GROUP
        )
        assert warnings == []

    def test_counts_multiple_conflicts(self):
        warnings = CorrelationService.check_exposure(
            open_trades=[_trade("GBP/USD", "Long"), _trade("AUD/USD", "Long")],
            direction="Long",
            instrument="EUR/USD",
        )
        assert len(warnings) == 1
        assert warnings[0]["count"] == 2
        assert "GBP/USD" in warnings[0]["conflicting"]
        assert "AUD/USD" in warnings[0]["conflicting"]

    def test_gold_silver_stacking(self):
        warnings = CorrelationService.check_exposure(
            open_trades=[_trade("XAG/USD", "Long")],
            direction="Long",
            instrument="XAU/USD",
        )
        assert len(warnings) == 1
        assert "Gold / Silver" in warnings[0]["group"]

    def test_empty_open_trades(self):
        assert CorrelationService.check_exposure([], "Long", "EUR/USD") == []

    def test_accepts_generator_of_trades(self):
        gen = (_trade(i, "Long") for i in ("GBP/USD", "AUD/USD"))
        warnings = CorrelationService.check_exposure(gen, "Long", "EUR/USD")
        assert warnings[0]["count"] == 2
