"""Unit tests for the Setup Ranker's stale-book warning.

Why this exists: on 2026-08-18 the MT5 terminal went unreachable at 05:57 and
the machine slept until 16:06, so this page served a ten-hour-old book while
looking entirely normal. Every stack and exposure check on the page was
computed against positions that no longer reflected the account, and nothing on
screen said so — the only clue was a grey "Book as of" caption in a footnote.

The contract: the page must be loud when the book is old, and it must be loud
*especially* when the stale book is empty, since that is the case where every
guard silently passes.

Only the pure helpers are exercised — no Streamlit runtime — so these stay fast
and do not need a live DB.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.pages_lib.setup_ranker import SetupRankerPage, _STALE_BOOK_AFTER_MIN


def _iso(minutes_ago: float, tz=timezone.utc) -> str:
    return (datetime.now(tz) - timedelta(minutes=minutes_ago)).isoformat()


class TestBookAge:
    def test_reads_a_recent_stamp(self):
        age = SetupRankerPage._book_age_minutes(_iso(7))
        assert age == pytest.approx(7, abs=0.5)

    def test_naive_stamps_are_read_as_utc(self):
        # open_positions.save writes UTC. Reading a naive stamp as local time
        # would under-report the age by the offset — on a UTC+2 machine a
        # two-hour-old book would read as fresh, which is the exact failure
        # this warning exists to prevent.
        naive = (datetime.now(timezone.utc) - timedelta(minutes=45)).replace(
            tzinfo=None).isoformat()
        assert SetupRankerPage._book_age_minutes(naive) == pytest.approx(45, abs=0.5)

    def test_offset_stamps_are_respected(self):
        plus_two = timezone(timedelta(hours=2))
        assert SetupRankerPage._book_age_minutes(_iso(30, plus_two)) == pytest.approx(30, abs=0.5)

    @pytest.mark.parametrize("bad", [None, "", "not-a-date", 12345, "2026-13-45"])
    def test_unreadable_stamps_are_none_not_crashes(self, bad):
        assert SetupRankerPage._book_age_minutes(bad) is None

    def test_a_future_stamp_clamps_to_zero(self):
        # Clock skew between the syncing host and the container must not render
        # as a negative age.
        assert SetupRankerPage._book_age_minutes(_iso(-30)) == 0.0


class TestAgeText:
    @pytest.mark.parametrize("minutes,expected", [
        (3, "3 min"),
        (59, "59 min"),
        (120, "2.0 hours"),
        (610, "10.2 hours"),
        (4320, "3.0 days"),
    ])
    def test_reads_in_human_units(self, minutes, expected):
        assert SetupRankerPage._age_text(minutes) == expected


class TestStaleBanner:
    def test_names_the_age_and_the_risk(self):
        html = SetupRankerPage._stale_banner(610, "2026-08-18T05:57:00+00:00")
        assert "BOOK IS STALE" in html
        assert "10.2 hours" in html
        assert "may be wrong" in html
        assert "logs/mt5_sync.log" in html

    def test_escapes_the_stamp(self):
        # The stamp reaches this from the store, so it is not trusted markup.
        html = SetupRankerPage._stale_banner(20, "<script>alert(1)</script>")
        assert "<script>" not in html
        assert "&lt;script&gt;" in html


class TestThreshold:
    def test_threshold_is_three_missed_sync_cycles(self):
        # The loop writes every 300s; the warning must not fire on one slow one.
        assert _STALE_BOOK_AFTER_MIN == 15

    def test_a_fresh_book_is_below_the_threshold(self):
        assert SetupRankerPage._book_age_minutes(_iso(5)) < _STALE_BOOK_AFTER_MIN

    def test_the_2026_08_18_outage_would_have_fired(self):
        assert SetupRankerPage._book_age_minutes(_iso(610)) > _STALE_BOOK_AFTER_MIN
