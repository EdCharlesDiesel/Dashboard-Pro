"""What the system found this week — aggregation only. Pure.

No Streamlit and no database, so this is measured by coverage; the page that
renders it lives in ``src/pages_lib/weeks_trades_page.py``.

**Why this is not "Today's Trades with a 7-day filter".** That page reads a
45-day window on purpose: ``consensus()`` holds each signal for its own declared
horizon, and narrowing it to a day made the board reverse whenever yesterday's
votes aged out, with nothing in the market having changed. So it answers *"what
is live and takeable now"*. Re-filtering the same query to seven days would
produce a staler copy of a board that already exists — a second answer to a
question that already has one.

This module answers a different question: **flow**. How many distinct days did a
pair keep appearing, did the system hold one side all week or change its mind,
and which sources drove it. That is a review question, and the existing store
already answers it without a new query shape.

**"This week" is Monday-anchored, not rolling.** A rolling seven days moves its
boundary every time the page is opened, so Tuesday's "this week" and Thursday's
"this week" cover different spans and no two reviews are comparable. Monday
00:00 UTC is fixed until the week turns over.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set

from src.core.todays_trades import LONG, SHORT, _side

__all__ = ["PairWeek", "by_day", "in_week", "pair_activity", "week_start"]


@dataclass(frozen=True)
class PairWeek:
    """One pair's week: how often, which side, which sources, and whether it
    changed its mind."""

    pair: str
    longs: int
    shorts: int
    days_seen: int
    sources: Set[str] = field(default_factory=set)
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    #: True when *one source* took both sides — not merely when two sources
    #: disagreed. See `pair_activity`.
    flipped: bool = False
    reversing_sources: Set[str] = field(default_factory=set)


def _as_utc(value: Any) -> Optional[datetime]:
    """Coerce a stored timestamp to aware UTC, or ``None``.

    Postgres hands back naive datetimes through this path. Dropping them would
    empty the page against a perfectly healthy store, so a naive stamp is read
    as UTC rather than discarded.
    """
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def week_start(now: datetime) -> datetime:
    """Monday 00:00 of ``now``'s week, in UTC."""
    moment = _as_utc(now) or datetime.now(timezone.utc)
    midnight = moment.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight - timedelta(days=midnight.weekday())


def in_week(row: Mapping[str, Any], now: datetime) -> bool:
    """True when ``row`` was logged inside ``now``'s Monday-anchored week."""
    when = _as_utc(row.get("logged_at"))
    if when is None:
        return False
    return week_start(now) <= when <= (_as_utc(now) or now)


def by_day(rows: Iterable[Mapping[str, Any]],
           now: datetime) -> "OrderedDict[date, List[Mapping[str, Any]]]":
    """This week's rows grouped by calendar day, oldest first."""
    buckets: Dict[date, List[Mapping[str, Any]]] = {}
    for row in rows:
        if not in_week(row, now):
            continue
        when = _as_utc(row.get("logged_at"))
        buckets.setdefault(when.date(), []).append(row)
    return OrderedDict((day, buckets[day]) for day in sorted(buckets))


def pair_activity(rows: Iterable[Mapping[str, Any]],
                  now: datetime) -> List[PairWeek]:
    """One :class:`PairWeek` per instrument, most persistent first.

    ``days_seen`` counts *distinct days*, not rows: three sources firing on
    Monday is one day of conviction, not three, and counting rows would let a
    single noisy morning outrank a pair that held all week.

    ``flipped`` means **one source contradicted itself**, not that both sides
    appeared. The earlier definition — ``longs and shorts`` — reported 22 of 27
    pairs as having changed side in a week when only 5 had. Eighteen independent
    indicators reading a pair differently *at the same moment* is the normal
    state of the system, and precisely what ``consensus()`` exists to resolve;
    reporting it as a reversal manufactures an alarm from routine behaviour.
    The long/short counts are kept because the split is the informative part:
    10/3 is conviction with one dissenter, 3/3 is a coin toss, and a boolean
    erases the difference.
    """
    acc: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not in_week(row, now):
            continue
        pair = row.get("instrument")
        when = _as_utc(row.get("logged_at"))
        if not pair or when is None:
            continue

        entry = acc.setdefault(pair, {
            "longs": 0, "shorts": 0, "days": set(), "sources": set(),
            "first": when, "last": when, "by_source": {}})
        side = _side(row.get("direction"))
        if side == LONG:
            entry["longs"] += 1
        elif side == SHORT:
            entry["shorts"] += 1
        if side and row.get("source"):
            entry["by_source"].setdefault(str(row["source"]), set()).add(side)
        entry["days"].add(when.date())
        if row.get("source"):
            entry["sources"].add(str(row["source"]))
        entry["first"] = min(entry["first"], when)
        entry["last"] = max(entry["last"], when)

    out = []
    for pair, e in acc.items():
        reversing = {src for src, sides in e["by_source"].items() if len(sides) > 1}
        out.append(PairWeek(
            pair=pair, longs=e["longs"], shorts=e["shorts"],
            days_seen=len(e["days"]), sources=e["sources"],
            first_seen=e["first"], last_seen=e["last"],
            flipped=bool(reversing), reversing_sources=reversing))
    out.sort(key=lambda p: (-p.days_seen, -(p.longs + p.shorts), p.pair))
    return out
