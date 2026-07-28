"""Currency-leg exposure guard — catch the stack `CORR_GROUPS` can't see.

`CorrelationService.check_exposure` compares *pair names* within a hand-listed
group and requires the **same direction string**. That misses the single most
common way a retail book blows up::

    sell GBP/AUD   ->  short GBP, LONG AUD
    buy  AUD/JPY   ->  LONG AUD,  short JPY
    sell EUR/AUD   ->  short EUR, LONG AUD

Three positions, three different pairs, two different direction strings, and
no shared `CORR_GROUPS` entry (there is no AUD group) — yet it is one bet on
AUD, sized 3×. The direction strings even *disagree* ("sell", "buy", "sell"),
so a name-and-direction check reports nothing.

This module decomposes every position into its two **currency legs** and nets
them, so the exposure is measured in the thing actually being risked. It's the
same base/quote convention the Setup Ranker's net-exposure strip already uses
for displayed ideas — this generalises it to positions you already hold.

Pure: no Streamlit, no DB, no network. Weights are caller-defined (lots, $ risk,
notional — whatever you want the netting denominated in) and only ever compared
against each other.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

from src.core.bias import BEARISH, BULLISH, normalize_direction

LONG, SHORT = 1, -1


@dataclass(frozen=True)
class Position:
    """One held (or proposed) position, reduced to what exposure cares about."""

    pair: str
    direction: str
    weight: float = 1.0
    label: str = ""          # optional free text for the UI (ticket #, source)

    @property
    def side(self) -> int:
        """+1 long the pair, -1 short it, 0 when the direction is unreadable."""
        return direction_side(self.direction)

    def describe(self) -> str:
        side = "LONG" if self.side == LONG else "SHORT" if self.side == SHORT else "?"
        return "{0} {1}".format(self.pair, side) + (
            " ({0})".format(self.label) if self.label else "")


@dataclass(frozen=True)
class Leg:
    currency: str
    side: int                # +1 long that currency, -1 short it
    weight: float


@dataclass(frozen=True)
class StackWarning:
    """The candidate would add to a currency leg you already hold."""

    currency: str
    side: int                       # the shared direction on that currency
    existing: Tuple[Position, ...]  # held positions carrying the same leg
    held_weight: float              # summed weight of those positions' legs
    candidate_weight: float

    @property
    def count(self) -> int:
        """Total positions on this leg *including* the candidate."""
        return len(self.existing) + 1

    @property
    def side_word(self) -> str:
        return "long" if self.side == LONG else "short"

    def summary(self) -> str:
        held = ", ".join(p.describe() for p in self.existing)
        return (
            "{0} {1} — this would be position {2} {3} {0}; already open: {4}"
            .format(self.currency, self.side_word, self.count,
                    self.side_word, held)
        )

    def as_dict(self) -> dict:
        return {
            "currency": self.currency,
            "side": self.side,
            "side_word": self.side_word,
            "count": self.count,
            "held_weight": round(float(self.held_weight), 2),
            "candidate_weight": round(float(self.candidate_weight), 2),
            "existing": [p.describe() for p in self.existing],
        }


def direction_side(direction: Optional[str]) -> int:
    """Map any of the app's direction vocabularies onto +1 / -1 / 0.

    Reuses `bias.normalize_direction`, so LONG/BUY/Bullish/STRONG_BUY and
    SHORT/SELL/Bearish all resolve correctly — including MT4's lowercase
    ``buy``/``sell``, which is what makes this work on imported positions.
    """
    canonical = normalize_direction(direction)
    if canonical == BULLISH:
        return LONG
    if canonical == BEARISH:
        return SHORT
    return 0


def split_pair(pair: str) -> Optional[Tuple[str, str]]:
    """``"EUR/AUD"`` -> ``("EUR", "AUD")``; ``None`` for anything unparseable
    (free-text tickers, indices, a stray blank)."""
    if not pair or "/" not in str(pair):
        return None
    base, _, quote = str(pair).partition("/")
    base, quote = base.strip().upper(), quote.strip().upper()
    if not base or not quote:
        return None
    return base, quote


def pair_legs(pair: str, direction: str, weight: float = 1.0) -> List[Leg]:
    """The two currency legs of a position.

    Long EUR/AUD is long EUR *and* short AUD; short EUR/AUD is the mirror.
    Returns an empty list when the pair or direction can't be read, so callers
    never have to special-case a metal or a free-text symbol.
    """
    parts = split_pair(pair)
    side = direction_side(direction)
    if parts is None or side == 0:
        return []
    base, quote = parts
    return [Leg(base, side, float(weight)), Leg(quote, -side, float(weight))]


def _as_positions(positions: Iterable) -> List[Position]:
    """Accept `Position` objects or plain mappings (DB rows, MT4 dicts)."""
    out: List[Position] = []
    for p in positions or []:
        if isinstance(p, Position):
            out.append(p)
            continue
        if isinstance(p, Mapping):
            pair = p.get("pair") or p.get("instrument") or ""
            direction = p.get("direction") or p.get("type") or ""
            weight = p.get("weight")
            if weight is None:
                weight = p.get("lot_size") or p.get("size") or 1.0
            label = str(p.get("label") or p.get("source") or "")
            try:
                weight = float(weight)
            except (TypeError, ValueError):
                weight = 1.0
            out.append(Position(pair=str(pair), direction=str(direction),
                                weight=weight, label=label))
    return out


def net_currency_exposure(positions: Iterable) -> dict:
    """Signed weight per currency across ``positions``.

    Positive = net long that currency. Positions whose pair/direction can't be
    parsed contribute nothing rather than raising — an exposure guard that
    crashes on one odd row is worse than one that under-reports it.
    """
    exposure: dict = {}
    for pos in _as_positions(positions):
        for leg in pair_legs(pos.pair, pos.direction, pos.weight):
            exposure[leg.currency] = exposure.get(leg.currency, 0.0) + \
                leg.side * leg.weight
    return exposure


def check_stack(
    pair: str,
    direction: str,
    open_positions: Iterable,
    weight: float = 1.0,
    max_per_currency: int = 1,
) -> List[StackWarning]:
    """Would taking ``pair``/``direction`` stack a currency you already hold?

    Returns one warning per currency where the candidate's leg points the same
    way as at least ``max_per_currency`` existing legs. Default 1 means "warn
    from the second position on a currency" — the rule that would have caught
    a third long-AUD trade.

    The candidate itself is excluded by identity, not by name: an existing
    position in the *same pair and same direction* is a genuine second helping
    and is reported. An existing position in the same pair the *other* way is
    not a stack (it's a partial close) and is ignored.
    """
    candidate_legs = pair_legs(pair, direction, weight)
    if not candidate_legs:
        return []
    held = _as_positions(open_positions)

    warnings: List[StackWarning] = []
    for leg in candidate_legs:
        matches: List[Position] = []
        held_weight = 0.0
        for pos in held:
            for held_leg in pair_legs(pos.pair, pos.direction, pos.weight):
                if held_leg.currency == leg.currency and held_leg.side == leg.side:
                    matches.append(pos)
                    held_weight += held_leg.weight
        if len(matches) >= max(1, int(max_per_currency)):
            warnings.append(StackWarning(
                currency=leg.currency,
                side=leg.side,
                existing=tuple(matches),
                held_weight=held_weight,
                candidate_weight=float(weight),
            ))
    # Worst offender first: most positions, then heaviest.
    warnings.sort(key=lambda w: (-w.count, -w.held_weight, w.currency))
    return warnings


def worst_stack(warnings: Sequence[StackWarning]) -> Optional[StackWarning]:
    """The single warning a compact UI should show. ``None`` when clean."""
    return warnings[0] if warnings else None
