"""Unit tests for src/services/signal_expiry.py — pure price-invalidation logic.

No DB, no Streamlit, no network — plain dict/float fixtures.
"""
from __future__ import annotations

from src.services.signal_expiry import (
    extract_entry_price,
    find_newly_invalidated,
    is_invalidated,
    stop_price,
    _bias_sign,
)


# ── extract_entry_price ─────────────────────────────────────────────────────
class TestExtractEntryPrice:
    def test_prefers_entry_price_column(self):
        row = {"entry_price": 1.2345, "checks_detail": {"entry": 9.9999}}
        assert extract_entry_price(row) == 1.2345

    def test_falls_back_to_checks_detail_dict(self):
        row = {"entry_price": None, "checks_detail": {"entry": 1.0842}}
        assert extract_entry_price(row) == 1.0842

    def test_falls_back_to_checks_detail_json_string(self):
        row = {"entry_price": None, "checks_detail": '{"entry": 1.0842}'}
        assert extract_entry_price(row) == 1.0842

    def test_missing_entry_price_and_checks_detail_returns_none(self):
        assert extract_entry_price({}) is None

    def test_checks_detail_without_entry_key_returns_none(self):
        row = {"checks_detail": {"stop_loss": 1.08}}
        assert extract_entry_price(row) is None

    def test_zero_entry_in_checks_detail_treated_as_missing(self):
        # signal_to_setup_row defaults a missing entry to 0.0 — not a real price.
        row = {"checks_detail": {"entry": 0.0}}
        assert extract_entry_price(row) is None

    def test_malformed_json_string_returns_none(self):
        row = {"checks_detail": "not json"}
        assert extract_entry_price(row) is None

    def test_non_numeric_entry_price_returns_none(self):
        assert extract_entry_price({"entry_price": "oops"}) is None


# ── _bias_sign ───────────────────────────────────────────────────────────────
class TestBiasSign:
    def test_long_variants(self):
        for d in ("LONG", "Long", "long", "BUY", "Buy", "STRONG_BUY"):
            assert _bias_sign(d) == 1, d

    def test_short_variants(self):
        for d in ("SHORT", "Short", "short", "SELL", "Sell", "STRONG_SELL"):
            assert _bias_sign(d) == -1, d

    def test_unresolved_variants(self):
        for d in ("Neutral", "n/a", "", None, "unknown"):
            assert _bias_sign(d) is None, d


# ── stop_price ───────────────────────────────────────────────────────────────
class TestStopPrice:
    def test_long_stop_is_below_entry(self):
        assert stop_price(1.1000, 20, 0.0001, sign=1) == 1.0980

    def test_short_stop_is_above_entry(self):
        assert stop_price(1.1000, 20, 0.0001, sign=-1) == 1.1020


# ── is_invalidated ───────────────────────────────────────────────────────────
class TestIsInvalidated:
    def test_long_invalidated_when_price_at_or_below_stop(self):
        assert is_invalidated(1.1000, 20, 0.0001, "LONG", 1.0980) is True
        assert is_invalidated(1.1000, 20, 0.0001, "LONG", 1.0970) is True

    def test_long_not_invalidated_above_stop(self):
        assert is_invalidated(1.1000, 20, 0.0001, "LONG", 1.0990) is False

    def test_short_invalidated_when_price_at_or_above_stop(self):
        assert is_invalidated(1.1000, 20, 0.0001, "SHORT", 1.1020) is True

    def test_short_not_invalidated_below_stop(self):
        assert is_invalidated(1.1000, 20, 0.0001, "SHORT", 1.1010) is False

    def test_none_when_direction_unresolved(self):
        assert is_invalidated(1.1000, 20, 0.0001, "Neutral", 1.0900) is None

    def test_none_when_entry_missing(self):
        assert is_invalidated(None, 20, 0.0001, "LONG", 1.0900) is None

    def test_none_when_sl_pips_missing(self):
        assert is_invalidated(1.1000, None, 0.0001, "LONG", 1.0900) is None

    def test_none_when_pip_size_missing_or_zero(self):
        assert is_invalidated(1.1000, 20, None, "LONG", 1.0900) is None
        assert is_invalidated(1.1000, 20, 0.0, "LONG", 1.0900) is None

    def test_none_when_current_price_missing(self):
        assert is_invalidated(1.1000, 20, 0.0001, "LONG", None) is None


# ── find_newly_invalidated ────────────────────────────────────────────────────
class TestFindNewlyInvalidated:
    def _row(self, **overrides):
        row = {
            "id": 1, "direction": "LONG", "sl_pips": 20, "pip_size": 0.0001,
            "ticker": "EURUSD=X", "entry_price": 1.1000, "checks_detail": None,
        }
        row.update(overrides)
        return row

    def test_flags_a_long_that_crossed_its_stop(self):
        rows = [self._row(id=1)]
        hits = find_newly_invalidated(rows, {"EURUSD=X": 1.0970})
        assert hits == [(1, 1.0970)]

    def test_skips_a_long_still_above_stop(self):
        rows = [self._row(id=1)]
        hits = find_newly_invalidated(rows, {"EURUSD=X": 1.0995})
        assert hits == []

    def test_skips_row_with_no_price_for_its_ticker(self):
        rows = [self._row(id=1, ticker="GBPUSD=X")]
        hits = find_newly_invalidated(rows, {"EURUSD=X": 1.0900})
        assert hits == []

    def test_skips_row_missing_entry_price(self):
        rows = [self._row(id=1, entry_price=None, checks_detail=None)]
        hits = find_newly_invalidated(rows, {"EURUSD=X": 1.0900})
        assert hits == []

    def test_evaluates_checklist_and_signal_rows_independently(self):
        rows = [
            self._row(id=1, ticker="EURUSD=X"),                       # crosses stop
            self._row(id=2, ticker="GBPUSD=X", direction="SHORT",
                      entry_price=1.2500, sl_pips=15),                 # stays safe
        ]
        prices = {"EURUSD=X": 1.0960, "GBPUSD=X": 1.2400}
        hits = find_newly_invalidated(rows, prices)
        assert hits == [(1, 1.0960)]
