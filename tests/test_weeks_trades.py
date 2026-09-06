"""Weekly aggregation of persisted signals.

`src/core/weeks_trades.py` is pure — no Streamlit, no DB — so it is measured and
can be tested exactly.

The design point worth defending in tests: **"this week" is Monday-anchored, not
a rolling 7 days.** A rolling window moves its boundary every day, so the same
page opened on Tuesday and Thursday reports different "weeks" and no two reviews
are comparable. Monday 00:00 UTC is fixed until the week turns over.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.core.weeks_trades import (
    PairWeek,
    by_day,
    in_week,
    pair_activity,
    week_start,
)

# Wed 2026-09-02 12:00 UTC. The week runs Mon 2026-08-31 00:00 -> now.
NOW = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)
MONDAY = datetime(2026, 8, 31, 0, 0, tzinfo=timezone.utc)


_UNSET = object()


def _row(pair="EUR/USD", direction="Long", source="ranker", when=_UNSET):
    # A sentinel, not `when or NOW`: the default-argument form silently turned
    # an explicit `when=None` into NOW and made the no-timestamp test vacuous.
    return {"instrument": pair, "direction": direction, "source": source,
            "logged_at": NOW if when is _UNSET else when, "checks_detail": None}


class TestWeekStart:
    def test_midweek_anchors_to_monday_midnight(self):
        assert week_start(NOW) == MONDAY

    def test_monday_is_its_own_week_start(self):
        monday_noon = MONDAY + timedelta(hours=12)
        assert week_start(monday_noon) == MONDAY

    def test_sunday_night_belongs_to_the_week_that_is_ending(self):
        # The boundary that a rolling-7-day window gets wrong: Sunday 23:00 is
        # the *end* of this week, not the start of the next.
        sunday = MONDAY + timedelta(days=6, hours=23)
        assert week_start(sunday) == MONDAY

    def test_the_following_monday_starts_a_new_week(self):
        assert week_start(MONDAY + timedelta(days=7)) == MONDAY + timedelta(days=7)


class TestInWeek:
    def test_a_row_from_this_week_is_included(self):
        assert in_week(_row(when=MONDAY + timedelta(days=1)), NOW)

    def test_last_weeks_row_is_excluded(self):
        assert not in_week(_row(when=MONDAY - timedelta(hours=1)), NOW)

    def test_a_naive_timestamp_is_read_as_utc_not_dropped(self):
        # The DB hands back naive datetimes. Discarding them would silently
        # empty the page against a perfectly healthy store.
        naive = (MONDAY + timedelta(days=1)).replace(tzinfo=None)
        assert in_week(_row(when=naive), NOW)

    def test_a_row_with_no_timestamp_is_excluded_rather_than_crashing(self):
        assert not in_week(_row(when=None), NOW)


class TestByDay:
    def test_rows_group_under_their_own_day(self):
        rows = [_row(when=MONDAY + timedelta(hours=9)),
                _row(when=MONDAY + timedelta(days=1, hours=9))]
        grouped = by_day(rows, NOW)
        assert sorted(grouped) == [MONDAY.date(), (MONDAY + timedelta(days=1)).date()]

    def test_days_come_back_in_order(self):
        rows = [_row(when=MONDAY + timedelta(days=2)),
                _row(when=MONDAY)]
        assert list(by_day(rows, NOW)) == sorted(by_day(rows, NOW))

    def test_rows_outside_the_week_do_not_appear(self):
        assert by_day([_row(when=MONDAY - timedelta(days=3))], NOW) == {}


class TestPairActivity:
    def test_a_pair_seen_on_three_days_is_one_entry_with_three_days(self):
        rows = [_row(when=MONDAY + timedelta(days=d, hours=9)) for d in (0, 1, 2)]
        act = pair_activity(rows, NOW)
        assert len(act) == 1
        assert act[0].pair == "EUR/USD" and act[0].days_seen == 3

    def test_repeats_on_one_day_count_as_one_day(self):
        # Three sources firing on Monday is one day of conviction, not three.
        rows = [_row(source=s, when=MONDAY + timedelta(hours=h))
                for s, h in (("ranker", 9), ("house", 10), ("fib", 11))]
        act = pair_activity(rows, NOW)
        assert act[0].days_seen == 1
        assert act[0].sources == {"ranker", "house", "fib"}

    def test_a_pair_holding_one_side_is_not_flipped(self):
        rows = [_row(direction="Long", when=MONDAY + timedelta(days=d))
                for d in (0, 1)]
        assert pair_activity(rows, NOW)[0].flipped is False

    def test_two_sources_disagreeing_is_not_a_flip(self):
        """The defect this replaced: 22 of 27 pairs were reported as "changed
        side" when they had simply been read differently by different
        indicators at the same moment. Eighteen independent sources disagreeing
        is the normal state of the system and the thing consensus() resolves —
        flagging it invents an alarm out of routine behaviour."""
        rows = [_row(direction="Long", source="daily_trend", when=MONDAY),
                _row(direction="Short", source="daily_macd", when=MONDAY)]
        entry = pair_activity(rows, NOW)[0]
        assert entry.flipped is False
        assert entry.reversing_sources == set()
        # The counts survive, because the split is the informative part:
        # 10/3 is conviction with a dissenter, 3/3 is a coin toss.
        assert entry.longs == 1 and entry.shorts == 1

    def test_one_source_taking_both_sides_is_a_flip_and_is_named(self):
        rows = [_row(direction="Long", source="biased_pivots", when=MONDAY),
                _row(direction="Short", source="biased_pivots",
                     when=MONDAY + timedelta(days=2))]
        entry = pair_activity(rows, NOW)[0]
        assert entry.flipped is True
        assert entry.reversing_sources == {"biased_pivots"}

    def test_several_reversing_sources_are_all_named(self):
        rows = [_row(direction="Long",  source="a", when=MONDAY),
                _row(direction="Short", source="a", when=MONDAY + timedelta(days=1)),
                _row(direction="Short", source="b", when=MONDAY),
                _row(direction="Long",  source="b", when=MONDAY + timedelta(days=1))]
        assert pair_activity(rows, NOW)[0].reversing_sources == {"a", "b"}

    def test_disagreement_plus_one_reversal_flags_only_the_reversal(self):
        rows = [_row(direction="Long",  source="steady", when=MONDAY),
                _row(direction="Short", source="other",  when=MONDAY),
                _row(direction="Long",  source="mind_changer", when=MONDAY),
                _row(direction="Short", source="mind_changer",
                     when=MONDAY + timedelta(days=1))]
        entry = pair_activity(rows, NOW)[0]
        assert entry.reversing_sources == {"mind_changer"}

    def test_first_and_last_seen_bracket_the_activity(self):
        # Days 0 and 1 only: NOW is Wednesday, and in_week rightly excludes
        # rows stamped in the future, so day 3 would never reach the aggregate.
        rows = [_row(when=MONDAY + timedelta(days=d)) for d in (0, 1)]
        entry = pair_activity(rows, NOW)[0]
        assert entry.first_seen < entry.last_seen

    def test_the_most_persistent_pair_sorts_first(self):
        rows = ([_row(pair="EUR/USD", when=MONDAY + timedelta(days=d)) for d in (0, 1, 2)]
                + [_row(pair="GBP/USD", when=MONDAY)])
        assert [a.pair for a in pair_activity(rows, NOW)] == ["EUR/USD", "GBP/USD"]

    def test_an_unreadable_direction_does_not_invent_a_side(self):
        rows = [_row(direction=None, when=MONDAY)]
        entry = pair_activity(rows, NOW)[0]
        assert entry.longs == 0 and entry.shorts == 0 and entry.flipped is False

    def test_an_empty_week_is_an_empty_list(self):
        assert pair_activity([], NOW) == []

    def test_a_row_outside_the_week_is_skipped_by_the_aggregate_too(self):
        # in_week guards the aggregate as well as the day grouping; without it
        # last week's signals would inflate this week's day counts.
        assert pair_activity([_row(when=MONDAY - timedelta(days=2))], NOW) == []

    def test_a_row_with_no_instrument_is_skipped(self):
        # trade_setups.instrument is nullable. A None key would collapse every
        # such row into one phantom "pair" in the table.
        assert pair_activity([_row(pair=None, when=MONDAY)], NOW) == []


class TestPairWeek:
    def test_it_carries_the_fields_the_page_renders(self):
        p = PairWeek(pair="EUR/USD", longs=2, shorts=0, days_seen=2,
                     sources={"ranker"}, first_seen=MONDAY, last_seen=NOW,
                     flipped=False, reversing_sources=set())
        assert p.pair == "EUR/USD" and p.days_seen == 2
