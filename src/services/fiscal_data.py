"""Treasury Fiscal Data — URL building and response parsing. Pure.

No network and no database here, so this module sits inside the coverage gate;
the HTTP call and the upsert live in ``src/data_backbone/fiscal_jobs.py``, which
is omitted for the same reason every other collector is.

**Why this exists rather than a scraper.** usdebtclock.org draws every figure in
JavaScript over transparent GIFs, and its per-second ticking is linear
interpolation between releases — several derived fields are the site's own
projection assumptions, not published statistics. It is fine to look at and
useless as a time series. This endpoint is the authoritative daily print,
unauthenticated and versioned.

**Everything arrives as a string, including nulls.** Coercion is ours, and the
two ways to get it wrong are both silent:

* ``float()`` loses the cents. ``40102964278586.10`` needs 16 significant digits
  and float64 gives ~15.95, so it becomes ``40102964278586.1015625``. It still
  *formats* to the right cents today, which is exactly why it would survive
  review, and it stops being right as the figure grows. Everything here is
  ``Decimal``.
* A suppressed row arrives as the string ``"null"``. Read as ``0`` it enters the
  series as a cliff rather than a gap, and a cliff is a fiscal event.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Mapping
from urllib.parse import urlencode

BASE_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"

#: Debt to the Penny — the headline figure, published daily.
DEBT_TO_PENNY_PATH = "/v2/accounting/od/debt_to_penny"

#: The amount columns worth storing. Each becomes its own series, so the
#: components stay separable from the total rather than being re-derived later.
AMOUNT_FIELDS: tuple[str, ...] = (
    "tot_pub_debt_out_amt",     # total public debt outstanding
    "debt_held_public_amt",     # held by the public
    "intragov_hold_amt",        # intragovernmental holdings
)

#: Values Treasury uses for "no figure". `"null"` is a JSON string, not a null.
_MISSING = {"", "null", "none", "n/a", "-"}

_PAGE_NUMBER = re.compile(r"page(?:%5B|\[)number(?:%5D|\])=(\d+)")


@dataclass(frozen=True)
class FiscalPoint:
    """One (series, date) observation, exact to the cent."""

    record_date: date
    series_id: str
    value: Decimal


def build_url(path: str, *, page_size: int = 100, page_number: int = 1,
              sort: str = "-record_date",
              fields: tuple[str, ...] | None = None) -> str:
    """A fully-qualified request URL.

    The endpoint is unauthenticated by design — no key is read or attached, so
    this module never touches ``secrets.toml``.
    """
    params: list[tuple[str, str]] = [
        ("sort", sort),
        ("page[size]", str(page_size)),
        ("page[number]", str(page_number)),
    ]
    if fields:
        params.append(("fields", ",".join(fields)))
    return f"{BASE_URL}{path}?{urlencode(params)}"


def _to_decimal(raw: Any) -> Decimal | None:
    """``Decimal`` or ``None`` — never a float, never a silent zero."""
    if raw is None:
        return None
    text = str(raw).strip()
    if text.lower() in _MISSING:
        return None
    try:
        return Decimal(text)
    except (InvalidOperation, ValueError):
        return None


def _to_date(raw: Any) -> date | None:
    if not raw:
        return None
    try:
        return datetime.strptime(str(raw).strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


def parse_rows(payload: Mapping[str, Any],
               fields: tuple[str, ...] = AMOUNT_FIELDS) -> list[FiscalPoint]:
    """Flatten a response into one point per (row, amount field).

    Rows without a usable date, and fields without a usable amount, are dropped
    rather than defaulted — a gap is honest, a zero is a fabricated data point.
    """
    points: list[FiscalPoint] = []
    for row in payload.get("data") or []:
        when = _to_date(row.get("record_date"))
        if when is None:
            continue
        for field in fields:
            value = _to_decimal(row.get(field))
            if value is None:
                continue
            points.append(FiscalPoint(record_date=when, series_id=field,
                                      value=value))
    return points


def collect_points(fetch: Callable[[str], Mapping[str, Any]], path: str, *,
                   backfill: bool = False, page_size: int = 100,
                   max_pages: int = 5000,
                   fields: tuple[str, ...] = AMOUNT_FIELDS) -> list[FiscalPoint]:
    """Walk the feed and return every point, with the transport injected.

    ``fetch`` takes a URL and returns the decoded payload. Keeping it a
    parameter is what lets pagination be tested exhaustively without a network
    call, and is why this lives here rather than in the collector.

    ``backfill`` is the whole distinction between the two callers. The first run
    must walk all 4,000-odd pages of history; the daily job must fetch **one**,
    because Treasury appends a single row a day and following ``links.next``
    every night would re-pull the entire series.

    ``max_pages`` is a circuit breaker, not a tuning knob: a feed that always
    returns a next link would otherwise loop forever.
    """
    points: list[FiscalPoint] = []
    page = 1
    for _ in range(max_pages):
        payload = fetch(build_url(path, page_size=page_size, page_number=page))
        batch = parse_rows(payload, fields=fields)
        points.extend(batch)
        if not backfill or not batch:
            break
        nxt = next_page_number(payload)
        if nxt is None:
            break
        page = nxt
    return points


def next_page_number(payload: Mapping[str, Any]) -> int | None:
    """The next page number, or ``None`` on the last page.

    The API returns ``links.next`` as a partial query string with the brackets
    percent-encoded, so the number is pulled out rather than the link followed
    verbatim.
    """
    link = (payload.get("links") or {}).get("next")
    if not link:
        return None
    found = _PAGE_NUMBER.search(str(link))
    return int(found.group(1)) if found else None
