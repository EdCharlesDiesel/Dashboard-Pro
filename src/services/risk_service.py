"""Position sizing and R:R computation — pure functions wrapped in a class."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskBreakdown:
    risk_amount: float
    lot_size: float
    rr_tp1: float
    rr_tp2: float


class RiskService:
    """Identical math to the inline calculations in the original checklist page."""

    @staticmethod
    def compute(
        account_balance: float,
        risk_pct: float,
        pip_value: float,
        sl_pips: float,
        tp1_pips: float,
        tp2_pips: float,
    ) -> RiskBreakdown:
        risk_amount = account_balance * (risk_pct / 100)
        lot_size = (
            round(risk_amount / sl_pips / pip_value, 2)
            if (sl_pips and pip_value)
            else 0.0
        )
        rr_tp1 = round(tp1_pips / sl_pips, 2) if sl_pips else 0.0
        rr_tp2 = round(tp2_pips / sl_pips, 2) if sl_pips else 0.0
        return RiskBreakdown(risk_amount, lot_size, rr_tp1, rr_tp2)
