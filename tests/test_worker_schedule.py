"""What the data-backbone worker actually schedules.

`worker.main()` builds its scheduler inline and calls `sched.start()`, which
blocks — so the job wiring could never be asserted without either starting a
daemon or reading the source and hoping. `schedule_jobs(sched)` exists to make
that one decision testable.

The risk this covers is specific: a collector that is written, imported and
working but **never registered** does nothing at all, silently, and looks
healthy. That is the same failure as the page with no `NavEntry` — the code is
fine and the wiring is missing, so nothing errors.
"""
from __future__ import annotations

import pytest

from src.data_backbone import worker

pytest.importorskip("apscheduler")


def _scheduler():
    from apscheduler.schedulers.background import BackgroundScheduler

    # Never started: constructing it is enough to inspect the job table, and
    # starting one in a test run would fire real collectors against live APIs.
    return BackgroundScheduler(timezone="UTC")


def _jobs():
    sched = _scheduler()
    worker.schedule_jobs(sched)
    return {job.id: job for job in sched.get_jobs()}


def test_the_fiscal_collector_is_registered():
    assert "fiscal_debt" in _jobs(), (
        "collect_debt_to_penny is never scheduled - it would import cleanly, "
        "pass its tests, and collect nothing")


def test_the_existing_daily_refresh_is_not_displaced():
    # Adding a job must not disturb the one that was already there.
    assert "daily_refresh" in _jobs()


def test_the_fiscal_collector_runs_once_a_day():
    # Treasury appends one row a day. Polling faster buys nothing and multiplies
    # requests against a public endpoint with no key behind it.
    trigger = str(_jobs()["fiscal_debt"].trigger)
    assert "cron" in trigger
    assert "hour='23'" in trigger or "hour=23" in trigger


def test_the_fiscal_collector_does_not_backfill_on_a_schedule():
    """The daily run must fetch one page, not walk 4,000.

    `backfill=True` on a cron job would re-pull the entire history every night —
    working, invisible, and a standing hammering of a free public API.
    """
    job = _jobs()["fiscal_debt"]
    assert job.kwargs.get("backfill", False) is False


def test_jobs_do_not_pile_up_if_one_run_overruns():
    job = _jobs()["fiscal_debt"]
    assert job.max_instances == 1
    assert job.coalesce is True
