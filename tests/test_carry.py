"""Carry: the one component of expected return you do not have to infer.

GBM drift estimated from price history was statistically indistinguishable from
zero on every instrument in this universe (t = 0.13 to 1.64, measured
2026-08-15). The rate differential is not inferred -- it is read off the policy
rates, and at a 5-20 day horizon it is the dominant driver.

Sign convention: `carry_pct` is what you EARN for being LONG the pair. Long
USD/ZAR earns rate(USD) and pays rate(ZAR), which is deeply negative; the short
earns the mirror. Getting this backwards would invert the read on every EM pair,
so it has its own test.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.services import carry


def _series(value):
    return pd.Series([value], index=pd.to_datetime(["2026-08-01"]))


@pytest.fixture
def rates(monkeypatch):
    table = {"FEDFUNDS": 4.0, "IRSTCI01ZAM156N": 7.5, "ECBDFR": 2.0}

    def fake(series_id, start=None):
        return _series(table[series_id]) if series_id in table else pd.Series(dtype=float)

    monkeypatch.setattr(carry, "fred_series", fake)
    return table


class TestCarryPct:
    def test_long_a_low_yielder_against_a_high_yielder_is_negative(self, rates):
        # Long USD/ZAR = long USD (4.0), short ZAR (7.5) -> -3.5
        assert carry.carry_pct("USD/ZAR") == pytest.approx(-3.5)

    def test_the_sign_follows_the_quote_order(self, rates):
        # EUR/USD: long EUR (2.0), short USD (4.0) -> -2.0
        assert carry.carry_pct("EUR/USD") == pytest.approx(-2.0)

    def test_a_missing_rate_is_none_not_zero(self, rates):
        # Zero would read as "no carry", which is a claim. None is the truth.
        assert carry.carry_pct("XAU/USD") is None

    def test_an_unknown_pair_is_none(self, rates):
        assert carry.carry_pct("FOO/BAR") is None

    def test_a_malformed_pair_is_none(self, rates):
        for bad in ("", None, "EURUSD", "EUR/", "/USD"):
            assert carry.carry_pct(bad) is None

    def test_commodities_have_no_carry(self, rates):
        for pair in ("XAU/USD", "XAG/USD", "WTI/USD"):
            assert carry.carry_pct(pair) is None

    def test_a_fred_failure_never_raises(self, monkeypatch):
        def boom(series_id, start=None):
            raise RuntimeError("FRED down")

        monkeypatch.setattr(carry, "fred_series", boom)
        assert carry.carry_pct("USD/ZAR") is None

    def test_an_empty_series_is_none(self, monkeypatch):
        monkeypatch.setattr(carry, "fred_series",
                            lambda s, start=None: pd.Series(dtype=float))
        assert carry.carry_pct("USD/ZAR") is None

    def test_an_all_nan_series_is_none(self, monkeypatch):
        monkeypatch.setattr(carry, "fred_series",
                            lambda s, start=None: pd.Series(
                                [float("nan")], index=pd.to_datetime(["2026-08-01"])))
        assert carry.carry_pct("USD/ZAR") is None


class TestFavours:
    def test_the_short_side_is_favoured_when_the_base_yields_less(self, rates):
        # Long USD/ZAR carries -3.5, so the short earns +3.5.
        assert carry.favours("USD/ZAR", "Short") is True
        assert carry.favours("USD/ZAR", "Long") is False

    def test_broker_dialect_directions_are_understood(self, rates):
        for long_word in ("Long", "LONG", "buy", "BUY"):
            assert carry.favours("USD/ZAR", long_word) is False
        for short_word in ("Short", "SHORT", "sell", "SELL"):
            assert carry.favours("USD/ZAR", short_word) is True

    def test_unknown_carry_neither_favours_nor_opposes(self, rates):
        assert carry.favours("XAU/USD", "Long") is None

    def test_a_negligible_differential_is_neutral(self, monkeypatch):
        # Both legs 4.0 -> 0.0 differential, inside the dead band. Policy rates
        # move in 25bp steps; anything smaller is not an edge.
        monkeypatch.setattr(carry, "fred_series", lambda s, start=None: _series(4.0))
        assert carry.favours("EUR/USD", "Long") is None

    def test_a_differential_exactly_on_the_dead_band_is_neutral(self, monkeypatch):
        table = {"FEDFUNDS": 4.0, "ECBDFR": 4.0 - carry.DEAD_BAND_PCT}

        def fake(series_id, start=None):
            return _series(table[series_id]) if series_id in table else pd.Series(dtype=float)

        monkeypatch.setattr(carry, "fred_series", fake)
        # EUR/USD = EUR(3.75) - USD(4.0) = -0.25, exactly the band -> still neutral
        assert carry.favours("EUR/USD", "Long") is None


class TestRateSeriesTable:
    def test_every_currency_in_the_universe_has_a_rate(self):
        from src.instruments.registry import INSTRUMENTS
        missing = set()
        for pair in INSTRUMENTS:
            base, _, quote = pair.partition("/")
            for ccy in (base, quote):
                if ccy in ("XAU", "XAG", "XPT", "WTI"):
                    continue
                if ccy not in carry.RATE_SERIES:
                    missing.add(ccy)
        assert not missing, "no policy-rate series for: %s" % sorted(missing)

    def test_it_is_derived_from_fred_series_not_duplicated(self):
        # One source of truth: a currency added to FRED_SERIES with a Rates
        # entry must appear here without anyone editing this module.
        from src.core.data_provider import FRED_SERIES
        expected = {c for c, b in FRED_SERIES.items() if b.get("Rates")}
        assert set(carry.RATE_SERIES) == expected
