"""Treasury Fiscal Data collector — HTTP and the upsert. Nothing else.

Deliberately thin. Every decision worth testing (URL construction, coercion,
pagination) lives in ``src/services/fiscal_data.py``, which is pure and measured
by coverage; this module is the part that cannot be unit-tested without a
network and a database, and is omitted for the same reason every other collector
under ``src/data_backbone/`` is.

Register with the scheduler in ``worker.py``:

    from src.data_backbone.fiscal_jobs import collect_debt_to_penny
    sched.add_job(collect_debt_to_penny, "cron", hour=23, minute=30,
                  args=[engine], id="fiscal_debt", replace_existing=True)

First run, once, by hand:

    python -c "from src.data_backbone.fiscal_jobs import collect_debt_to_penny; \\
               collect_debt_to_penny(engine, backfill=True)"
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Any, Mapping

from sqlalchemy import text

from src.services.fiscal_data import (
    DEBT_TO_PENNY_PATH,
    FiscalPoint,
    collect_points,
)

log = logging.getLogger(__name__)

_SCHEMA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "fiscal_schema.sql")

_TIMEOUT = float(os.environ.get("FISCAL_HTTP_TIMEOUT", "30"))


def _http_get(url: str) -> Mapping[str, Any]:
    """One GET, decoded. No key: the endpoint is unauthenticated."""
    with urllib.request.urlopen(url, timeout=_TIMEOUT) as resp:
        return json.load(resp)


def _statements(ddl: str) -> list[str]:
    """Split DDL on ``;``, dropping fragments that are only comments.

    Splitting naively leaves the trailing comment block after the final
    semicolon as a "statement" that is non-empty after ``strip()`` but contains
    no SQL, and Postgres rejects it with "can't execute an empty query" — which
    fails schema creation on a file that is perfectly valid.
    """
    out = []
    for chunk in ddl.split(";"):
        body = "\n".join(line for line in chunk.splitlines()
                         if line.strip() and not line.strip().startswith("--"))
        if body.strip():
            out.append(chunk.strip())
    return out


def ensure_schema(engine) -> None:
    """Create the table if absent. Idempotent, safe on every start."""
    with open(_SCHEMA, encoding="utf-8") as fh:
        ddl = fh.read()
    with engine.begin() as conn:
        for statement in _statements(ddl):
            conn.execute(text(statement))


def upsert_points(engine, points: list[FiscalPoint]) -> int:
    """Write points, replacing any existing (series_id, record_date).

    Upsert rather than insert because **Treasury revises**: a record_date
    already stored can be restated, and an insert-only collector would keep the
    superseded figure forever while believing it was current.

    The value is bound as ``str(Decimal)`` so psycopg hands Postgres an exact
    numeric literal — passing the Decimal through float anywhere on this path
    would undo the entire point of the NUMERIC column.
    """
    if not points:
        return 0
    sql = text("""
        INSERT INTO fiscal_series (series_id, record_date, value)
        VALUES (:series_id, :record_date, CAST(:value AS NUMERIC))
        ON CONFLICT (series_id, record_date) DO UPDATE
           SET value = EXCLUDED.value,
               fetched_at = now()
    """)
    with engine.begin() as conn:
        conn.execute(sql, [{"series_id": p.series_id,
                            "record_date": p.record_date,
                            "value": str(p.value)} for p in points])
    return len(points)


def collect_debt_to_penny(engine, *, backfill: bool = False,
                          fetch=None) -> int:
    """Fetch Debt to the Penny and store it. Returns rows written.

    ``backfill=True`` walks the whole history (4,000-odd pages) and is a
    one-time operation; the scheduled call takes the default and fetches a
    single page, because Treasury appends one row a day.

    ``fetch`` exists so the caller can inject a transport; production leaves it
    None and gets ``_http_get``.
    """
    ensure_schema(engine)
    points = collect_points(fetch or _http_get, DEBT_TO_PENNY_PATH,
                            backfill=backfill,
                            page_size=1000 if backfill else 10)
    written = upsert_points(engine, points)
    log.info("[fiscal] %s: %d point(s) written",
             "backfill" if backfill else "daily", written)
    return written
