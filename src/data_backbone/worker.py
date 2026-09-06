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


def schedule_jobs(sched) -> None:
    """Register every recurring job on `sched`.

    Separate from `main()` so the wiring can be asserted: `main()` ends in
    `sched.start()`, which blocks, so a job that is never registered would
    otherwise be invisible — the collector imports fine, passes its own tests,
    and simply never runs.
    """
    # after the US cash close (22:00 UTC ~ 17:00 ET) on weekdays
    sched.add_job(refresh_all, "cron", day_of_week="mon-fri", hour=22, minute=0,
                  id="daily_refresh", max_instances=1, coalesce=True)

    # Treasury publishes Debt to the Penny once a day, after the refresh above.
    # backfill stays False: the daily run fetches one page. Walking all ~4,000
    # pages nightly would work, show nothing unusual, and hammer a free public
    # endpoint for data that has not changed.
    sched.add_job(_collect_fiscal, "cron", hour=23, minute=30,
                  id="fiscal_debt", kwargs={"backfill": False},
                  max_instances=1, coalesce=True)


def _collect_fiscal(*, backfill: bool = False) -> None:
    """Scheduler entry point. Never lets one collector take down the worker."""
    try:
        from src.data_backbone.fiscal_jobs import collect_debt_to_penny

        written = collect_debt_to_penny(db.get_engine(), backfill=backfill)
        log.info("fiscal: %d point(s) written", written)
    except Exception as exc:  # noqa: BLE001
        log.warning("fiscal collection failed: %s", exc)


def main() -> None:
    db.init_db()
    log.info("db initialised")
    refresh_all()  # warm the store immediately

    sched = BlockingScheduler(timezone="UTC")
    schedule_jobs(sched)
    log.info("scheduler started — daily refresh weekdays 22:00 UTC, "
             "fiscal 23:30 UTC")
    sched.start()


if __name__ == "__main__":
    main()
