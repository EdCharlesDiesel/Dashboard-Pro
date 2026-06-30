"""
worker.py — background service.

Initialises the database, warms the store on startup, then refreshes every
watched ticker and FRED series on a daily schedule. Run as its own process /
container — never inside Streamlit:

    python -m src.data_backbone.worker
"""
from __future__ import annotations

import logging

from apscheduler.schedulers.blocking import BlockingScheduler

from . import db
from . import data_access as da
from .config import WATCH_TICKERS, WATCH_FRED, PRICE_PERIOD

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("worker")


def refresh_ticker(ticker: str) -> None:
    df = da.fetch_yf(ticker, PRICE_PERIOD)
    if df.empty:
        log.warning("no data for %s", ticker)
        return
    n = db.upsert_price_bars(ticker, df)
    log.info("price %s: upserted %d bars", ticker, n)


def refresh_fred(series_id: str) -> None:
    s = da.fetch_fred(series_id)
    if s.empty:
        log.warning("no data for FRED %s", series_id)
        return
    n = db.upsert_fred(series_id, s)
    log.info("fred %s: upserted %d points", series_id, n)


def refresh_all() -> None:
    log.info("refresh starting")
    for tk in WATCH_TICKERS:
        try:
            refresh_ticker(tk)
        except Exception as e:
            log.warning("ticker %s failed: %s", tk, e)
    for sid in WATCH_FRED:
        try:
            refresh_fred(sid)
        except Exception as e:
            log.warning("fred %s failed: %s", sid, e)
    log.info("refresh complete")


def main() -> None:
    db.init_db()
    log.info("db initialised")
    refresh_all()  # warm the store immediately

    sched = BlockingScheduler(timezone="UTC")
    # after the US cash close (22:00 UTC ~ 17:00 ET) on weekdays
    sched.add_job(refresh_all, "cron", day_of_week="mon-fri", hour=22, minute=0,
                  id="daily_refresh", max_instances=1, coalesce=True)
    log.info("scheduler started — daily refresh weekdays 22:00 UTC")
    sched.start()


if __name__ == "__main__":
    main()
