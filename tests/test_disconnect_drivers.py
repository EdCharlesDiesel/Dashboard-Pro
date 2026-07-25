"""Unit tests for src/services/disconnect_drivers.py.

The value of the driver map is that every registry instrument has an
economically-defensible macro driver with the correct sign. These tests pin
that contract: full coverage of the universe, valid driver keys, sane signs,
and the specific relationships that must hold (dollar inverts XXX/USD, oil
inverts USD/CAD, gold lifts AUD, real yields press metals).
"""
from __future__ import annotations

import pytest

from src.instruments.registry import INSTRUMENTS
from src.services import disconnect_drivers as dd


class TestCoverage:
    def test_every_registry_instrument_has_a_driver(self):
        missing = [p for p in INSTRUMENTS.keys() if not dd.drivers_for(p)]
        assert missing == [], f"instruments with no mapped driver: {missing}"

    def test_no_orphan_instruments_in_map(self):
        # Every key in the map is a real registry instrument.
        unknown = [p for p in dd.INSTRUMENT_DRIVERS if p not in INSTRUMENTS]
        assert unknown == [], f"map references unknown instruments: {unknown}"

    def test_covered_instruments_matches_map(self):
        assert set(dd.covered_instruments()) == set(dd.INSTRUMENT_DRIVERS)


class TestRelationshipValidity:
    def test_all_drivers_resolve_and_signs_are_unit(self):
        for pair, rels in dd.INSTRUMENT_DRIVERS.items():
            for rel in rels:
                assert rel.driver in dd.DRIVERS, f"{pair}: bad driver {rel.driver}"
                assert rel.expected_sign in (-1, +1), f"{pair}: bad sign"
                assert rel.note.strip(), f"{pair}: empty rationale"

    def test_no_duplicate_driver_per_instrument(self):
        for pair, rels in dd.INSTRUMENT_DRIVERS.items():
            keys = [r.driver for r in rels]
            assert len(keys) == len(set(keys)), f"{pair} has a duplicate driver"

    def test_driver_labels_exist(self):
        for key in dd.DRIVERS:
            assert dd.driver_label(key) and dd.driver_label(key) != key


class TestEconomicSigns:
    """The relationships a forex trader would sanity-check."""

    def _sign(self, pair, driver):
        for r in dd.drivers_for(pair):
            if r.driver == driver:
                return r.expected_sign
        return None

    def test_dollar_inverts_xxx_usd(self):
        # DXY up → EUR/USD, GBP/USD, AUD/USD down.
        for pair in ("EUR/USD", "GBP/USD", "AUD/USD", "NZD/USD"):
            assert self._sign(pair, "DXY") == -1

    def test_dollar_lifts_usd_xxx(self):
        # DXY up → USD/JPY, USD/CHF, USD/CAD, USD/ZAR up.
        for pair in ("USD/JPY", "USD/CHF", "USD/CAD", "USD/ZAR"):
            assert self._sign(pair, "DXY") == +1

    def test_oil_inverts_usd_cad(self):
        assert self._sign("USD/CAD", "OIL") == -1

    def test_gold_lifts_commodity_currencies(self):
        assert self._sign("AUD/USD", "GOLD") == +1
        assert self._sign("NZD/USD", "GOLD") == +1

    def test_gold_strengthens_rand(self):
        # SA gold exporter: gold up → ZAR up → USD/ZAR down.
        assert self._sign("USD/ZAR", "GOLD") == -1

    def test_real_yields_press_metals(self):
        assert self._sign("XAU/USD", "REAL5Y") == -1
        assert self._sign("XAG/USD", "REAL5Y") == -1

    def test_risk_on_lifts_jpy_crosses(self):
        for pair in ("USD/JPY", "EUR/JPY", "GBP/JPY", "AUD/JPY"):
            assert self._sign(pair, "SPX") == +1
