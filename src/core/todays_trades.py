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


# ── currency exposure ─────────────────────────────────────────────────────────
# `find_conflict` compares *pair names* against hand-maintained groups, so it can
# only see collisions someone thought to list. Two blind spots follow from that,
# and both were live in the book on 2026-08-10:
#
#   * Across groups. AUD/USD sits in "USD vs majors", AUD/JPY in "JPY crosses",
#     so the guard never compares them — it passed AUD/USD long while AUD/JPY
#     short was open. Both are AUD trades; net AUD was -1,000 units, half a
#     position paying spread on both sides.
#   * In aggregate. Four tickets each cleared the guard individually and summed
#     to -4,056 USD on a 1,939 USD account — 2.1x the balance short dollar,
#     spread across four symbols so it never looked like concentration.
#
# This decomposes every position into its two currency legs instead, which needs
# no group table and cannot miss a pair nobody listed.

# Units of the base per 1.0 lot. Forex is the 100k standard; the commodities are
# the contract sizes this desk's broker (Exness) uses. Only the *quote* leg of a
# commodity is a currency, but it still lands on USD, so the size has to be right.
CONTRACT_SIZES: Dict[str, float] = {
    "XAU/USD": 100.0, "XAG/USD": 5000.0, "XPT/USD": 100.0, "WTI/USD": 1000.0,
}
DEFAULT_CONTRACT = 100_000.0

# Net exposure to one currency, as a multiple of account balance, above which a
# further trade in the same direction is refused. 1.0 = "no single currency may
# exceed the account". Deliberately loose — this is a backstop against the 2.1x
# case, not a portfolio optimiser.
CCY_LIMIT_RATIO = 1.0


def _pair_ccys(pair: str) -> Optional[Sequence[str]]:
    """('EUR', 'USD') for 'EUR/USD'; None if it isn't a two-sided symbol."""
    parts = [p.strip().upper() for p in str(pair or "").split("/")]
    if len(parts) != 2 or not all(parts):
        return None
    return parts


# Two schemas reach this code and neither is going to change: `open_positions`
# stores lot_size/entry_price, the live MT5 read returns volume/price_current.
# Accepting both here beats making every caller remap and forget a field.
def pos_volume(pos: Dict[str, Any]) -> float:
    for key in ("volume", "lots", "lot_size"):
        value = pos.get(key)
        if value:
            return float(value)
    return 0.0


def pos_price(pos: Dict[str, Any]) -> float:
    for key in ("price", "price_current", "price_open", "entry_price"):
        value = pos.get(key)
        if value:
            return float(value)
    return 0.0


def usd_rates(positions: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """USD value of one unit of each currency, derived from the book itself.

    Every open position carries its own price, and any pair with USD on one side
    fixes that currency outright (AUD/USD at 0.70659 => 1 AUD = 0.70659 USD).
    Crosses then resolve off whatever is already known (AUD/JPY at 111.569 with
    AUD fixed gives JPY), so repeated passes converge without a single quote
    lookup. That matters because this has to run in the Linux container, where
    there is no MetaTrader and no broker connection at all.
    """
    rates: Dict[str, float] = {"USD": 1.0}
    for _ in range(len(positions) + 1):     # each pass can only fix new currencies
        progressed = False
        for pos in positions:
            ccys = _pair_ccys(pos.get("pair"))
            price = pos_price(pos)
            if not ccys or not price:
                continue
            base, quote = ccys
            # 1 base = `price` quote, so base_usd = price * quote_usd.
            if base not in rates and quote in rates:
                rates[base] = float(price) * rates[quote]
                progressed = True
            elif quote not in rates and base in rates and float(price):
                rates[quote] = rates[base] / float(price)
                progressed = True
        if not progressed:
            break
    return rates


def currency_legs(pair: str, direction: Any, lots: float, price: float,
                  rates: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """One position as signed USD notional per currency.

    A long is +base / -quote: buying EUR/USD is owning euros funded by dollars.
    Both legs count — netting only the base is how a book ends up short dollar
    four times over without anyone noticing.
    """
    ccys = _pair_ccys(pair)
    side = _side(direction)
    if not ccys or not side or not lots or not price:
        return {}
    base, quote = ccys
    rates = rates or {}
    units = float(lots) * CONTRACT_SIZES.get(pair, DEFAULT_CONTRACT)
    sign = 1.0 if side == LONG else -1.0
    inst = INSTRUMENTS.get(pair)
    out: Dict[str, float] = {}
    # A commodity's base leg is metal or oil, not a currency — skip it, but keep
    # the USD leg, which is the part that concentrates.
    if not (inst is not None and inst.is_commodity):
        out[base] = sign * units * rates.get(base, 1.0)
    out[quote] = out.get(quote, 0.0) - sign * units * float(price) * rates.get(quote, 1.0)
    return out


def net_exposure(positions: Sequence[Dict[str, Any]],
                 rates: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Signed USD notional per currency across the whole book."""
    rates = rates if rates is not None else usd_rates(positions)
    net: Dict[str, float] = {}
    for pos in positions:
        legs = currency_legs(pos.get("pair"), pos.get("direction"),
                             pos_volume(pos), pos_price(pos), rates)
        for ccy, amount in legs.items():
            net[ccy] = net.get(ccy, 0.0) + amount
    return net


def exposure_conflict(pair: str, direction: Any,
                      open_positions: Sequence[Dict[str, Any]],
                      balance: float,
                      rates: Optional[Dict[str, float]] = None,
                      limit_ratio: float = CCY_LIMIT_RATIO) -> Optional[str]:
    """Why this idea would deepen an already-oversized currency bet, or ``None``.

    Only refuses trades that push a currency *further* the way it is already
    leaning. A trade that offsets an existing exposure is the opposite of the
    problem and passes — USD/CHF long against a short-dollar book reduces it.
    """
    side = _side(direction)
    ccys = _pair_ccys(pair)
    if not side or not ccys or not balance or balance <= 0:
        return None
    exposure = net_exposure(open_positions, rates)
    limit = float(balance) * limit_ratio
    base, quote = ccys
    long = side == LONG
    # What this trade would add to each leg: +1 accumulates, -1 offsets.
    adds = {base: 1.0 if long else -1.0, quote: -1.0 if long else 1.0}
    inst = INSTRUMENTS.get(pair)
    if inst is not None and inst.is_commodity:
        adds.pop(base, None)
    for ccy, direction_sign in adds.items():
        held = exposure.get(ccy, 0.0)
        if not held or abs(held) < limit:
            continue
        if (held > 0) != (direction_sign > 0):
            continue                      # this trade reduces that exposure
        return ("deepens {0} exposure — book is already {1} {2:,.0f} USD "
                "({3:.1f}x balance)".format(
                    ccy, "long" if held > 0 else "short", abs(held),
                    abs(held) / float(balance)))
    return None


def offsetting_legs(positions: Sequence[Dict[str, Any]],
                    rates: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    """Currencies the book holds on both sides at once.

    Not an error — a deliberate hedge looks identical — but it is nearly always
    accidental, and it costs spread and swap on both legs to hold the difference.
    """
    rates = rates if rates is not None else usd_rates(positions)
    by_ccy: Dict[str, Dict[str, List[str]]] = {}
    for pos in positions:
        pair = str(pos.get("pair") or "")
        legs = currency_legs(pair, pos.get("direction"),
                             pos_volume(pos), pos_price(pos), rates)
        for ccy, amount in legs.items():
            if not amount:
                continue
            slot = by_ccy.setdefault(ccy, {"long": [], "short": []})
            slot["long" if amount > 0 else "short"].append(pair)
    out: List[Dict[str, Any]] = []
    for ccy, slot in by_ccy.items():
        if slot["long"] and slot["short"]:
            out.append({"currency": ccy, "long": sorted(set(slot["long"])),
                        "short": sorted(set(slot["short"]))})
    return sorted(out, key=lambda d: d["currency"])


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
