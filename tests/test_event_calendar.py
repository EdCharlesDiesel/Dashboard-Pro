"""The shared macro release calendar.

Lifted out of ``pages/event_week_vol_tab.py`` so the Event Reaction Map and
the Busy-Week study read one calendar instead of two that drift. Pure data and
date arithmetic — no Streamlit, no network, no DB.
"""
from datetime import date

import pandas as pd
import pytest

from src.core.event_calendar import (
    EVENT_RELEVANCE,
    FOMC_DATES,
    US_CPI_DATES,
    US_PPI_DATES,
    build_event_calendar,
    next_release,
    nfp_dates,
)


class TestNFPDates:
    def test_every_generated_date_is_a_first_friday(self):
        for ts in nfp_dates(date(2026, 1, 1), date(2026, 12, 31)):
            assert ts.weekday() == 4 and ts.day <= 7

    def test_it_yields_one_per_month(self):
        assert len(nfp_dates(date(2026, 1, 1), date(2026, 12, 31))) == 12


class TestSeededCalendars:
    @pytest.mark.parametrize("dates", [FOMC_DATES, US_CPI_DATES, US_PPI_DATES])
    def test_dates_are_iso_sorted_and_unique(self, dates):
        assert dates == sorted(dates), "keep seed lists sorted"
        assert len(dates) == len(set(dates))
        for d in dates:
            date.fromisoformat(d)          # raises if malformed

    def test_cpi_and_ppi_reach_the_current_year(self):
        # The whole reason the calendar was extracted: the old copy stopped
        # at 2025-12-10, so a 2026 default was impossible.
        assert any(d.startswith("2026") for d in US_CPI_DATES)
        assert any(d.startswith("2026") for d in US_PPI_DATES)


class TestNextRelease:
    def test_nfp_is_rule_computed_not_seeded(self):
        assert next_release("NFP", date(2026, 8, 20)) == date(2026, 9, 4)

    def test_a_seeded_event_returns_the_next_listed_date(self):
        nxt = next_release("FOMC", date(2026, 8, 20))
        assert nxt is not None and nxt >= date(2026, 8, 20)
        assert nxt.isoformat() in FOMC_DATES

    def test_the_release_day_itself_does_not_roll(self):
        first = date.fromisoformat(FOMC_DATES[0])
        assert next_release("FOMC", first) == first

    def test_an_exhausted_calendar_returns_none_rather_than_guessing(self):
        assert next_release("FOMC", date(2099, 1, 1)) is None

    def test_an_unknown_event_returns_none(self):
        assert next_release("NOT_AN_EVENT", date(2026, 8, 20)) is None


class TestBuildEventCalendar:
    def test_it_returns_long_format_within_the_window(self):
        cal = build_event_calendar(date(2026, 1, 1), date(2026, 6, 30), None)
        assert list(cal.columns) == ["date", "event"]
        assert cal["date"].is_monotonic_increasing
        assert (cal["date"].dt.date >= date(2026, 1, 1)).all()
        assert (cal["date"].dt.date <= date(2026, 6, 30)).all()

    def test_it_carries_every_seeded_event_type(self):
        cal = build_event_calendar(date(2026, 1, 1), date(2026, 12, 31), None)
        assert {"NFP", "FOMC", "US_CPI", "US_PPI"} <= set(cal["event"])

    def test_extra_rows_are_merged_and_deduped(self):
        extra = pd.DataFrame({"date": ["2026-02-02", "2026-02-02"],
                              "event": ["CUSTOM", "CUSTOM"]})
        cal = build_event_calendar(date(2026, 1, 1), date(2026, 3, 1), extra)
        assert (cal["event"] == "CUSTOM").sum() == 1


class TestEventRelevance:
    def test_jpy_pulls_in_boj_and_zar_pulls_in_sarb(self):
        assert "BOJ" in EVENT_RELEVANCE["USD/JPY"]
        assert "SARB" in EVENT_RELEVANCE["USD/ZAR"]

    def test_every_instrument_gets_the_us_macro_trio(self):
        for events in EVENT_RELEVANCE.values():
            assert {"NFP", "US_CPI", "FOMC"} <= set(events)
