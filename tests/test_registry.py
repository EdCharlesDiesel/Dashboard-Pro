"""Unit tests for src/instruments/registry.py — the single source of truth."""
from __future__ import annotations

import pytest

from src.instruments.registry import (
    CORR_GROUPS,
    INSTRUMENTS,
    Instrument,
    InstrumentRegistry,
    TREND_COMMODITIES,
)


class TestDictStyleAccess:
    def test_legacy_ticker_lookup(self):
        assert INSTRUMENTS["EUR/USD"]["ticker"] == "EURUSD=X"

    def test_legacy_pip_fields(self):
        view = INSTRUMENTS["USD/JPY"]
        assert view["pip"] == pytest.approx(9.09)
        assert view["pip_size"] == 0.01

    def test_unknown_field_raises_keyerror(self):
        with pytest.raises(KeyError):
            _ = INSTRUMENTS["EUR/USD"]["nonexistent"]

    def test_contains(self):
        assert "XAU/USD" in INSTRUMENTS
        assert "FOO/BAR" not in INSTRUMENTS

    def test_len_is_full_universe(self):
        assert len(INSTRUMENTS) == 21


class TestAttributeAccess:
    def test_get_returns_instrument_dataclass(self):
        inst = INSTRUMENTS.get("GBP/USD")
        assert isinstance(inst, Instrument)
        assert inst.ticker == "GBPUSD=X"
        assert inst.pip_size == 0.0001

    def test_get_missing_returns_default(self):
        assert INSTRUMENTS.get("NOPE") is None
        sentinel = object()
        assert INSTRUMENTS.get("NOPE", sentinel) is sentinel

    def test_display_name_is_symbol(self):
        assert INSTRUMENTS.get("XAU/USD").display_name == "XAU/USD"

    def test_is_commodity_flag(self):
        assert INSTRUMENTS.get("XAU/USD").is_commodity is True
        assert INSTRUMENTS.get("EUR/USD").is_commodity is False


class TestDomainHelpers:
    def test_forex_pairs_excludes_metals(self):
        forex = INSTRUMENTS.forex_pairs()
        assert "EUR/USD" in forex
        assert not (set(forex) & TREND_COMMODITIES)

    def test_commodities_are_the_three_metals(self):
        assert set(INSTRUMENTS.commodities()) == {"XAU/USD", "XAG/USD", "XPT/USD"}

    def test_forex_and_commodities_partition_universe(self):
        assert len(INSTRUMENTS.forex_pairs()) + len(INSTRUMENTS.commodities()) == len(INSTRUMENTS)

    def test_correlated_groups_for_eurusd(self):
        groups = INSTRUMENTS.correlated_groups("EUR/USD")
        names = [g for g, _ in groups]
        assert any("USD vs majors" in n for n in names)
        # Every returned group actually contains the queried instrument.
        assert all("EUR/USD" in members for _, members in groups)

    def test_correlated_groups_empty_for_isolated(self):
        # EUR/AUD is in no CORR_GROUPS bucket.
        assert INSTRUMENTS.correlated_groups("EUR/AUD") == []


class TestRegistryConstruction:
    def test_custom_table(self):
        reg = InstrumentRegistry([("X/Y", "XY=X", 10.0, 0.0001, "note")])
        assert len(reg) == 1
        assert reg["X/Y"]["ticker"] == "XY=X"

    def test_corr_groups_only_reference_known_instruments(self):
        for members in CORR_GROUPS.values():
            for m in members:
                assert m in INSTRUMENTS, f"{m} in CORR_GROUPS but not in registry"
