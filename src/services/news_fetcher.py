"""
news_fetcher.py
===============
Robust economic-calendar feed for the sentry's sentry_news table.

Primary source: ForexFactory weekly calendar JSON, served by
faireconomy.media — free, no API key, includes impact level, currency,
forecast and previous values, updated continuously:

    https://nfs.faireconomy.media/ff_calendar_thisweek.json

Robustness features (the reasons your current feed feels stale):
  * Retries with backoff + hard timeout  — a hung request returns
    STALE, never blocks.
  * Freshness ledger — every successful fetch is recorded in
    sentry_meta; freshness() tells you (and the briefing) how old the
    calendar actually is. Stale feed becomes VISIBLE instead of silent.
  * Upsert dedupe — events keyed by (ts, currency, title); re-fetching
    updates changed times (FF often reshuffles) without duplicating.
  * Timezone-correct — feed timestamps carry offsets; everything is
    normalized to naive UTC, matching the sentry's convention.
  * Impact filter — keeps High (and optionally Medium) only, so the
    news-window tagging isn't polluted by tentative low-impact noise.

Usage:
  python -m src.services.news_fetcher --fetch            # pull this week into DB
  python -m src.services.news_fetcher --upcoming 12      # show next 12h of events
  python -m src.services.news_fetcher --freshness        # how old is the calendar?

Scheduling (add to evening_sentry.run(), or its own cron):
  07:00 SAST daily  -> fetch (morning refresh)
  16:30 SAST mon-fri-> fetch (pre-briefing refresh, catches reschedules)
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timedelta, timezone

import requests
import sqlalchemy as sa

from src.services.evening_sentry import Store

log = logging.getLogger("news_fetcher")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

FF_URLS = [
    "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
    "https://nfs.faireconomy.media/ff_calendar_nextweek.json",
]
KEEP_IMPACTS = {"High", "Medium"}      # set to {"High"} for high only
RETRIES = 3
TIMEOUT_S = 10
BACKOFF_S = 2.0
STALE_AFTER_HOURS = 18                 # calendar older than this = stale
UA = {"User-Agent": "Mozilla/5.0 (sentry-news-fetcher)"}

MAJORS = {"USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF", "CNY", "ZAR"}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class NewsFetcher:
    def __init__(self, store: Store | None = None):
        self.store = store or Store()
        md = sa.MetaData()
        self.meta = sa.Table(
            "sentry_meta", md,
            sa.Column("key", sa.String(50), primary_key=True),
            sa.Column("value", sa.Text),
            sa.Column("ts_utc", sa.DateTime),
        )
        md.create_all(self.store.engine)

    # ------------------------------------------------------------ http
    def _get_json(self, url: str) -> list | None:
        for attempt in range(1, RETRIES + 1):
            try:
                r = requests.get(url, timeout=TIMEOUT_S, headers=UA)
                r.raise_for_status()
                data = r.json()
                if isinstance(data, list):
                    return data
                log.warning("unexpected payload shape from %s", url)
            except Exception as e:
                log.warning("fetch attempt %d/%d failed (%s): %s",
                            attempt, RETRIES, url, e)
                if attempt < RETRIES:
                    time.sleep(BACKOFF_S * attempt)
        return None

    # ----------------------------------------------------------- parse
    @staticmethod
    def parse_events(raw: list) -> list[dict]:
        """FF item: {title, country, date(ISO w/ offset), impact,
        forecast, previous}. Returns naive-UTC rows for sentry_news."""
        out = []
        for it in raw:
            try:
                ccy = str(it.get("country", "")).upper()
                impact = str(it.get("impact", "")).title()
                if ccy not in MAJORS or impact not in KEEP_IMPACTS:
                    continue
                dt = datetime.fromisoformat(str(it["date"]))
                ts = (dt.astimezone(timezone.utc).replace(tzinfo=None)
                      if dt.tzinfo else dt)
                out.append({
                    "ts_utc": ts,
                    "currency": ccy,
                    "impact": impact.lower(),
                    "title": str(it.get("title", "")).strip()[:200],
                    "actual": str(it.get("actual", "")).strip(),
                    "forecast": str(it.get("forecast", "")).strip(),
                    "previous": str(it.get("previous", "")).strip(),
                })
            except Exception as e:
                log.debug("skipping malformed event %r: %s", it, e)
        return out

    # ---------------------------------------------------------- upsert
    def upsert(self, events: list[dict]) -> tuple[int, int]:
        """Insert new events; update ts for rescheduled ones.
        Key: (currency, title, calendar week). Returns (added, updated)."""
        added = updated = 0
        news = self.store.news
        with self.store.engine.begin() as cx:
            for ev in events:
                week_lo = ev["ts_utc"] - timedelta(days=4)
                week_hi = ev["ts_utc"] + timedelta(days=4)
                q = (sa.select(news.c.id, news.c.ts_utc)
                     .where(news.c.currency == ev["currency"])
                     .where(news.c.title == ev["title"])
                     .where(news.c.ts_utc.between(week_lo, week_hi)))
                row = cx.execute(q).fetchone()
                if row is None:
                    cx.execute(news.insert().values(**ev))
                    added += 1
                else:
                    # Always refresh actual/forecast/previous (a scheduled
                    # release fills these in later) as well as ts_utc/impact
                    # (FF often reshuffles times).
                    cx.execute(news.update()
                               .where(news.c.id == row.id)
                               .values(ts_utc=ev["ts_utc"], impact=ev["impact"],
                                       actual=ev["actual"], forecast=ev["forecast"],
                                       previous=ev["previous"]))
                    updated += 1
        return added, updated

    # ------------------------------------------------------- freshness
    def _mark_fetched(self, n_events: int) -> None:
        with self.store.engine.begin() as cx:
            cx.execute(sa.delete(self.meta)
                       .where(self.meta.c.key == "news_last_fetch"))
            cx.execute(self.meta.insert().values(
                key="news_last_fetch", value=str(n_events),
                ts_utc=_utcnow()))

    def freshness(self) -> tuple[float | None, bool]:
        """(hours since last successful fetch, is_stale)."""
        q = sa.select(self.meta.c.ts_utc).where(
            self.meta.c.key == "news_last_fetch")
        with self.store.engine.connect() as cx:
            row = cx.execute(q).fetchone()
        if row is None or row.ts_utc is None:
            return None, True
        hours = (_utcnow() - row.ts_utc).total_seconds() / 3600
        return hours, hours > STALE_AFTER_HOURS

    # ------------------------------------------------------------- api
    def fetch(self) -> bool:
        """Pull this week (+ next week) into sentry_news. True on success."""
        total_added = total_updated = 0
        got_any = False
        for url in FF_URLS:
            raw = self._get_json(url)
            if raw is None:
                continue
            got_any = True
            events = self.parse_events(raw)
            a, u = self.upsert(events)
            total_added += a
            total_updated += u
        if got_any:
            self._mark_fetched(total_added + total_updated)
            log.info("calendar refreshed: +%d new, %d rescheduled",
                     total_added, total_updated)
        else:
            log.error("ALL calendar sources failed — feed is stale")
        return got_any

    def upcoming(self, hours: int = 24) -> list:
        now = _utcnow()
        q = (sa.select(self.store.news)
             .where(self.store.news.c.ts_utc.between(
                 now, now + timedelta(hours=hours)))
             .order_by(self.store.news.c.ts_utc))
        with self.store.engine.connect() as cx:
            return list(cx.execute(q).fetchall())

    def briefing_line(self) -> str:
        """One line for the sentry's 16:45 briefing."""
        hours, stale = self.freshness()
        if stale:
            age = "never" if hours is None else f"{hours:.0f}h ago"
            return f"⚠ NEWS FEED STALE (last fetch: {age})"
        nxt = self.upcoming(6)
        if not nxt:
            return "News: no high/medium impact events next 6h ✓"
        lines = [f"{r.ts_utc:%H:%M}Z {r.currency} {r.title}" for r in nxt[:5]]
        return "News next 6h:\n  " + "\n  ".join(lines)


# ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fetch", action="store_true")
    g.add_argument("--upcoming", type=int, metavar="HOURS")
    g.add_argument("--freshness", action="store_true")
    args = ap.parse_args()

    nf = NewsFetcher()
    if args.fetch:
        ok = nf.fetch()
        raise SystemExit(0 if ok else 1)
    if args.upcoming:
        for r in nf.upcoming(args.upcoming):
            print(f"{r.ts_utc:%a %H:%M}Z  {r.currency:4s} "
                  f"[{r.impact:6s}] {r.title}")
    if args.freshness:
        hours, stale = nf.freshness()
        print(f"last fetch: {'never' if hours is None else f'{hours:.1f}h ago'}"
              f"  stale={stale}")


if __name__ == "__main__":
    main()
