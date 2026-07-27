"""Economic calendar — the week's scheduled events, as a reusable service.

``pages/news-filter.py`` has its own copy of this fetch, but that page runs its
Streamlit body at import time (``st.set_page_config`` at module scope, no
``__main__`` guard), so it **cannot be imported** by another page. This module
is the importable version for new pages; if news-filter is ever refactored into
the framework, it should converge onto this rather than a third copy.

Source is the same public Forex Factory weekly JSON feed news-filter uses.
Times are published in America/New_York; this module keeps the date and the
raw time label rather than re-deriving timezones (news-filter owns the full
tz-aware presentation — here we only need "which day, which currency, how big").

Streamlit-free: the caching wrapper belongs to the caller. Never raises — a
dead feed returns an empty list so a page degrades to "no calendar" instead of
breaking.
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Dict, Iterable, List, Optional, Set

logger = logging.getLogger("ForexDashboard")

FF_URL = "https://nfs.faireconomy.media/ff_calendar_{week}.json"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}

IMPACTS = ("High", "Medium", "Low", "Holiday")


def fetch_week(week: str = "nextweek", timeout: int = 10) -> List[Dict]:
    """Raw Forex Factory events for ``"thisweek"`` or ``"nextweek"``.

    Returns ``[]`` on any failure (network, bad JSON, empty payload) — the
    calendar is context, never a hard dependency.
    """
    if week not in ("thisweek", "nextweek"):
        raise ValueError("week must be 'thisweek' or 'nextweek'")
    try:
        import requests
        r = requests.get(FF_URL.format(week=week), headers=_HEADERS, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception as exc:  # noqa: BLE001 — best effort by design
        logger.warning("[econ_calendar] fetch failed (%s): %s", week, exc)
        return []


def normalise(raw: Iterable[Dict]) -> List[Dict]:
    """Flatten raw feed rows to ``{date, time, currency, title, impact}``.

    Rows missing a date/title/currency are dropped. ``date`` is a ``date``
    object (the feed's ``YYYY-MM-DD`` prefix); ``time`` stays the feed's own
    label (ET, e.g. ``"8:30am"``, ``"All Day"``) — pure display.
    """
    out: List[Dict] = []
    for e in raw or []:
        try:
            date_str = (e.get("date") or "")[:10]
            title = (e.get("title") or "").strip()
            ccy = (e.get("country") or "").strip().upper()
            if not date_str or not title or not ccy:
                continue
            d = datetime.strptime(date_str, "%Y-%m-%d").date()
            impact = (e.get("impact") or "Low").capitalize()
            if impact not in IMPACTS:
                impact = "Low"
            out.append({
                "date": d,
                "time": (e.get("time") or "").strip() or "—",
                "currency": ccy,
                "title": title,
                "impact": impact,
                "forecast": (e.get("forecast") or "") or "—",
                "previous": (e.get("previous") or "") or "—",
            })
        except Exception:
            continue   # one malformed row must not drop the whole calendar
    out.sort(key=lambda x: (x["date"], x["currency"]))
    return out


def filter_events(
    events: Iterable[Dict],
    currencies: Optional[Set[str]] = None,
    impacts: Iterable[str] = ("High",),
) -> List[Dict]:
    """Events matching the given currencies (``None`` = all) and impact levels."""
    want = {i.capitalize() for i in impacts}
    ccys = {c.upper() for c in currencies} if currencies else None
    return [
        e for e in events or []
        if e.get("impact") in want and (ccys is None or e.get("currency") in ccys)
    ]


def count_by_currency(events: Iterable[Dict], impacts: Iterable[str] = ("High",)) -> Dict[str, int]:
    """How many events of the given impact each currency carries this week."""
    want = {i.capitalize() for i in impacts}
    counts: Dict[str, int] = {}
    for e in events or []:
        if e.get("impact") in want:
            counts[e["currency"]] = counts.get(e["currency"], 0) + 1
    return counts


def group_by_day(events: Iterable[Dict]) -> Dict[date, List[Dict]]:
    """Events keyed by calendar day, each day's list kept in feed order."""
    days: Dict[date, List[Dict]] = {}
    for e in events or []:
        days.setdefault(e["date"], []).append(e)
    return dict(sorted(days.items()))
