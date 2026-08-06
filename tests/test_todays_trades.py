"""Unit tests for src/core/todays_trades.py — pure, no DB, no network.

Each class guards one of the three ways yesterday's board went wrong: votes
counted per row instead of per source, correlated positions counted as
diversification, and size fixed instead of derived from the stop.
"""
from __future__ import annotations

import pytest

from src.core import todays_trades as tt


def _row(pair, direction, source):
    return {"instrument": pair, "direction": direction, "source": source}


class TestConsensus:
    def test_counts_each_source_once_per_side(self):
        # daily_macd fired three times today; it still gets one vote.
        rows = [_row("EUR/USD", "Long", "daily_macd")] * 3
        rows.append(_row("EUR/USD", "Long", "daily_trend"))
        ideas = tt.consensus(rows)
        assert len(ideas) == 1
        assert ideas[0].agree == ["daily_macd", "daily_trend"]
        assert ideas[0].score == 2

    def test_dissent_reduces_the_score(self):
        rows = [_row("EUR/USD", "Long", "a"), _row("EUR/USD", "Long", "b"),
                _row("EUR/USD", "Long", "c"), _row("EUR/USD", "Short", "d")]
        idea = tt.consensus(rows)[0]
        assert idea.direction == tt.LONG
        assert idea.agree == ["a", "b", "c"] and idea.against == ["d"]
        assert idea.score == 2          # 3 for, 1 against

    def test_an_even_split_offers_no_read(self):
        rows = [_row("EUR/USD", "Long", "a"), _row("EUR/USD", "Short", "b")]
        assert tt.consensus(rows) == []

    def test_neutral_rows_are_ignored(self):
        rows = [_row("EUR/USD", "Neutral", "confluence_checker"),
                _row("EUR/USD", "Long", "daily_macd")]
        idea = tt.consensus(rows)[0]
        assert idea.agree == ["daily_macd"] and idea.against == []

    @pytest.mark.parametrize("d,expected", [
        ("Long", tt.LONG), ("LONG", tt.LONG), ("BUY", tt.LONG),
        ("STRONG_BUY", tt.LONG), ("Bullish", tt.LONG),
        ("Short", tt.SHORT), ("SELL", tt.SHORT), ("BEARISH", tt.SHORT),
    ])
    def test_every_stored_dialect_is_understood(self, d, expected):
        assert tt._side(d) == expected

    @pytest.mark.parametrize("d", ["Neutral", "", None, "WAIT"])
    def test_directionless_values_have_no_side(self, d):
        assert tt._side(d) is None

    def test_ranked_by_net_agreement(self):
        rows = ([_row("A/B", "Long", s) for s in "ab"] +
                [_row("C/D", "Long", s) for s in "abcd"])
        assert [i.pair for i in tt.consensus(rows)] == ["C/D", "A/B"]

    def test_junk_rows_are_skipped_not_fatal(self):
        rows = [{"instrument": "", "direction": "Long", "source": "x"},
                {"instrument": "EUR/USD", "direction": "Long", "source": ""},
                _row("EUR/USD", "Long", "ok")]
        assert tt.consensus(rows)[0].agree == ["ok"]


class TestConflict:
    def test_same_pair_same_direction_is_blocked(self):
        c = tt.find_conflict("EUR/USD", tt.LONG,
                             [{"pair": "EUR/USD", "direction": "Long"}])
        assert c and "already open" in c

    def test_same_pair_opposite_direction_is_allowed(self):
        # Deliberate hedge, not a stack.
        assert tt.find_conflict("EUR/USD", tt.SHORT,
                                [{"pair": "EUR/USD", "direction": "Long"}]) is None

    def test_correlated_pair_same_direction_is_blocked(self):
        # GBP/USD and AUD/USD share the "USD vs majors" group — long both is one
        # short-USD bet on two tickets, which is what reached 54.6% of account.
        c = tt.find_conflict("GBP/USD", tt.LONG,
                             [{"pair": "AUD/USD", "direction": "Long"}])
        assert c and "stacks with" in c and "AUD/USD" in c

    def test_correlated_pair_opposite_direction_is_allowed(self):
        assert tt.find_conflict("GBP/USD", tt.SHORT,
                                [{"pair": "AUD/USD", "direction": "Long"}]) is None

    def test_uncorrelated_pair_is_clean(self):
        assert tt.find_conflict("EUR/CAD", tt.LONG,
                                [{"pair": "AUD/JPY", "direction": "Long"}]) is None

    def test_empty_book_is_always_clean(self):
        assert tt.find_conflict("EUR/USD", tt.LONG, []) is None


class TestPositionSize:
    def test_size_is_derived_from_the_stop(self):
        # Same budget, twice the stop -> half the size.
        wide = tt.position_size(20.0, 100.0, 10.0)
        tight = tt.position_size(20.0, 50.0, 10.0)
        assert tight == pytest.approx(wide * 2)

    def test_rounds_down_never_up(self):
        # 20 / (75 * 10) = 0.0266... -> 0.02, not 0.03.
        assert tt.position_size(20.0, 75.0, 10.0) == pytest.approx(0.02)

    def test_clamped_to_broker_minimum(self):
        assert tt.position_size(1.0, 5000.0, 10.0) == pytest.approx(0.01)

    def test_risk_is_reported_honestly_when_the_minimum_overshoots(self):
        # A wide stop on a small budget: size clamps up, so the real risk exceeds
        # the budget. That must be visible, not silently rounded away.
        lots = tt.position_size(20.0, 5000.0, 10.0)
        assert tt.risk_of(lots, 5000.0, 10.0) == pytest.approx(500.0)

    @pytest.mark.parametrize("bad", [(0, 100, 10), (20, 0, 10), (20, 100, 0)])
    def test_degenerate_inputs_return_zero(self, bad):
        assert tt.position_size(*bad) == 0.0


class TestSizeIdea:
    def _idea(self, direction=tt.LONG):
        return tt.Idea(pair="EUR/USD", direction=direction, agree=["a"])

    def test_long_levels_bracket_the_entry(self):
        i = tt.size_idea(self._idea(), price=1.1000, atr=0.0050, balance=2000.0)
        assert i.stop < i.entry < i.target
        assert i.sl_pips == pytest.approx(75.0)      # 1.5 * 0.0050 / 0.0001

    def test_short_levels_are_mirrored(self):
        i = tt.size_idea(self._idea(tt.SHORT), price=1.1000, atr=0.0050, balance=2000.0)
        assert i.target < i.entry < i.stop

    def test_target_is_2r_from_the_stop(self):
        i = tt.size_idea(self._idea(), price=1.1000, atr=0.0050, balance=2000.0)
        assert (i.target - i.entry) == pytest.approx(2 * (i.entry - i.stop))

    def test_risk_lands_near_one_percent(self):
        i = tt.size_idea(self._idea(), price=1.1000, atr=0.0050, balance=2000.0)
        assert i.risk_pct <= 1.0        # rounds down, so never over

    def test_unknown_instrument_is_left_unsized(self):
        i = tt.size_idea(tt.Idea(pair="ZZZ/ZZZ", direction=tt.LONG),
                         price=1.0, atr=0.01, balance=2000.0)
        assert i.lots is None and i.entry is None
