"""Unit tests for src/services/risk_service.py — position sizing & R:R."""
from __future__ import annotations

import pytest

from src.services.risk_service import RiskBreakdown, RiskService


class TestCompute:
    def test_textbook_case(self):
        r = RiskService.compute(
            account_balance=10_000,
            risk_pct=1.0,
            pip_value=10.0,
            sl_pips=20.0,
            tp1_pips=40.0,
            tp2_pips=60.0,
        )
        assert isinstance(r, RiskBreakdown)
        assert r.risk_amount == pytest.approx(100.0)
        assert r.lot_size == pytest.approx(0.5)
        assert r.rr_tp1 == pytest.approx(2.0)
        assert r.rr_tp2 == pytest.approx(3.0)

    def test_risk_amount_scales_with_pct(self):
        r = RiskService.compute(20_000, 2.0, 10.0, 50.0, 100.0, 150.0)
        assert r.risk_amount == pytest.approx(400.0)

    def test_zero_sl_pips_is_safe(self):
        # Division guards must prevent ZeroDivisionError and return zeros.
        r = RiskService.compute(10_000, 1.0, 10.0, 0.0, 40.0, 60.0)
        assert r.lot_size == 0.0
        assert r.rr_tp1 == 0.0
        assert r.rr_tp2 == 0.0

    def test_zero_pip_value_yields_zero_lot(self):
        r = RiskService.compute(10_000, 1.0, 0.0, 20.0, 40.0, 60.0)
        assert r.lot_size == 0.0
        # R:R is independent of pip value and still computes.
        assert r.rr_tp1 == pytest.approx(2.0)

    def test_lot_size_rounded_to_two_dp(self):
        r = RiskService.compute(10_000, 1.0, 9.09, 23.0, 46.0, 69.0)
        # round(100 / 23 / 9.09, 2)
        assert r.lot_size == pytest.approx(round(100 / 23 / 9.09, 2))

    def test_breakdown_is_frozen(self):
        r = RiskService.compute(10_000, 1.0, 10.0, 20.0, 40.0, 60.0)
        with pytest.raises(Exception):
            r.lot_size = 1.0  # frozen dataclass
