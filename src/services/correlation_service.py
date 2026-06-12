"""Correlation-exposure guard — flag stacked directional risk."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping

from src.instruments.registry import CORR_GROUPS


@dataclass(frozen=True)
class CorrelationWarning:
    group: str
    conflicting: str
    count: int

    def as_dict(self) -> dict:
        return {"group": self.group, "conflicting": self.conflicting, "count": self.count}


class CorrelationService:
    """Stateless check against the CORR_GROUPS table."""

    @staticmethod
    def check_exposure(
        open_trades: Iterable[Mapping[str, str]],
        direction: str,
        instrument: str,
    ) -> List[dict]:
        warnings: List[dict] = []
        open_trades = list(open_trades)
        for group_name, group_instruments in CORR_GROUPS.items():
            if instrument not in group_instruments:
                continue
            conflicts = [
                t for t in open_trades
                if t.get("instrument") in group_instruments
                and t.get("direction") == direction
                and t.get("instrument") != instrument
            ]
            if conflicts:
                warnings.append(CorrelationWarning(
                    group=group_name,
                    conflicting=", ".join(t["instrument"] for t in conflicts),
                    count=len(conflicts),
                ).as_dict())
        return warnings
