"""Today's tradeable ideas — consensus across every signal source, sized.

Pure logic, no I/O. This is the thing the desk actually asks for each morning:
*of everything the system found today, which ones can I take, at what size?*

Three steps, each of which is a way the answer can be wrong if skipped:

1. **Consensus.** One instrument can be flagged by a dozen pages; what matters
   is how many *independent* sources agree and whether any disagree. A pair with
   4 sources long and 0 short is a different proposition from 4 long and 3 short.
2. **Conflict.** An idea that stacks onto an open position is not a new trade,
   it is the same bet twice. `CORR_GROUPS` decides this, not intuition — the
   desk previously ran four simultaneous short-USD positions believing they were
   diversified, and carried 54.6% of the account at risk as a result.
3. **Size.** Position size is *derived from the stop*, never fixed. Using a flat
   lot size across instruments produced an 8x spread in real risk between the
   tightest and widest stop on the same nominal size.

Sizing here is deliberately broker-free arithmetic (registry pip value x stop
distance) so it works identically on the Linux container, where the Windows-only
MetaTrader5 package does not exist.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.instruments.registry import CORR_GROUPS, INSTRUMENTS

LONG, SHORT = "Long", "Short"


def _side(direction: Any) -> Optional[str]:
    """Normalise any dialect the app stores into Long/Short, or None."""
    d = str(direction or "").strip().upper()
    if not d:
        return None
    if any(m in d for m in ("LONG", "BUY", "BULL")):
        return LONG
    if any(m in d for m in ("SHORT", "SELL", "BEAR")):
        return SHORT
    return None          # Neutral and friends have no side to trade


@dataclass
class Idea:
    """One instrument's aggregated read, plus whether it can be taken."""

    pair: str
    direction: str
    agree: List[str] = field(default_factory=list)     # sources on this side
    against: List[str] = field(default_factory=list)   # sources on the other
    conflict: Optional[str] = None                     # why it can't be taken
    entry: Optional[float] = None
    stop: Optional[float] = None
    target: Optional[float] = None
    sl_pips: Optional[float] = None
    lots: Optional[float] = None
    risk_usd: Optional[float] = None
    risk_pct: Optional[float] = None

    @property
    def score(self) -> int:
        """Net agreement — the ranking key."""
        return len(self.agree) - len(self.against)

    @property
    def tradeable(self) -> bool:
        return self.conflict is None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["score"] = self.score
        d["tradeable"] = self.tradeable
        return d


def consensus(rows: Iterable[Dict[str, Any]]) -> List[Idea]:
    """Aggregate persisted signal rows into one Idea per instrument.

    ``rows`` need only carry ``instrument``, ``direction`` and ``source``. A
    source is counted once per side however many rows it wrote, so a page that
    fires more often cannot outvote one that fires less.
    """
    sides: Dict[str, Dict[str, set]] = {}
    for r in rows:
        pair = str(r.get("instrument") or "").strip()
        side = _side(r.get("direction"))
        src = str(r.get("source") or "").strip()
        if not pair or not side or not src:
            continue
        sides.setdefault(pair, {LONG: set(), SHORT: set()})[side].add(src)

    out: List[Idea] = []
    for pair, by_side in sides.items():
        longs, shorts = by_side[LONG], by_side[SHORT]
        if not longs and not shorts:
            continue
        if len(longs) == len(shorts):
            continue          # genuinely split — no read to offer
        direction = LONG if len(longs) > len(shorts) else SHORT
        agree, against = (longs, shorts) if direction == LONG else (shorts, longs)
        out.append(Idea(pair=pair, direction=direction,
                        agree=sorted(agree), against=sorted(against)))
    out.sort(key=lambda i: (i.score, len(i.agree)), reverse=True)
    return out


def find_conflict(pair: str, direction: str,
                  open_positions: Sequence[Dict[str, Any]]) -> Optional[str]:
    """Why this idea would double an existing bet, or ``None`` if it is clean.

    Two ways to collide, and the second is the one people miss:
      * the same instrument is already open in the same direction;
      * a *different* instrument in the same `CORR_GROUPS` group is open the
        same way, which is the same underlying bet wearing another ticket.
    """
    for pos in open_positions:
        held_pair = str(pos.get("pair") or "").strip()
        held_side = _side(pos.get("direction"))
        if not held_pair or not held_side:
            continue
        if held_pair == pair:
            if held_side == direction:
                return "already open — {0} {1}".format(held_pair, held_side.lower())
            continue          # opposite side of the same pair is a hedge, allowed
        for group, members in CORR_GROUPS.items():
            if pair in members and held_pair in members and held_side == direction:
                return "stacks with {0} {1} — same group: {2}".format(
                    held_pair, held_side.lower(), group)
    return None


def position_size(risk_budget: float, sl_pips: float, pip_value_per_lot: float,
                  volume_min: float = 0.01, volume_step: float = 0.01,
                  volume_max: float = 200.0) -> float:
    """Lots such that a stop-out costs about ``risk_budget``.

    Rounded **down** to the volume step: overshooting the risk budget to reach a
    round lot is exactly how a 1% trade becomes a 1.4% one.
    """
    if sl_pips <= 0 or pip_value_per_lot <= 0 or risk_budget <= 0:
        return 0.0
    raw = risk_budget / (sl_pips * pip_value_per_lot)
    stepped = int(raw / volume_step) * volume_step
    return round(max(volume_min, min(volume_max, stepped)), 8)


def risk_of(lots: float, sl_pips: float, pip_value_per_lot: float) -> float:
    """What a stop-out actually costs at that size — always shown, never assumed.

    `position_size` clamps to the broker minimum, so a wide stop on a small
    account can still exceed the budget. The caller must be able to see that.
    """
    return round(lots * sl_pips * pip_value_per_lot, 2)


def size_idea(idea: Idea, price: float, atr: float, balance: float,
              risk_pct: float = 1.0, atr_mult: float = 1.5,
              rr: float = 2.0) -> Idea:
    """Attach entry/stop/target and a size derived from the stop distance."""
    inst = INSTRUMENTS.get(idea.pair)
    if inst is None or not price or not atr:
        return idea
    pip_size = inst.pip_size or 0.0001
    dist = atr_mult * atr
    long = idea.direction == LONG

    idea.entry = round(price, 8)
    idea.stop = round(price - dist if long else price + dist, 8)
    idea.target = round(price + rr * dist if long else price - rr * dist, 8)
    idea.sl_pips = round(dist / pip_size, 1)

    budget = balance * (risk_pct / 100.0)
    idea.lots = position_size(budget, idea.sl_pips, float(inst.pip))
    idea.risk_usd = risk_of(idea.lots, idea.sl_pips, float(inst.pip))
    idea.risk_pct = round(idea.risk_usd / balance * 100.0, 2) if balance else None
    return idea
