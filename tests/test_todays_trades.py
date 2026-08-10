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


# ── currency exposure ─────────────────────────────────────────────────────────
# The fourth way the board went wrong, and the one CORR_GROUPS structurally
# cannot catch: pair-name matching sees collisions someone listed, not the
# currency arithmetic underneath. The book on 2026-08-10 is the fixture.
def _pos(pair, direction, volume, price):
    return {"pair": pair, "direction": direction, "volume": volume, "price": price}


LIVE_BOOK = [
    _pos("AUD/JPY", "sell", 0.02, 111.569),
    _pos("AUD/USD", "buy", 0.01, 0.70659),
    _pos("GBP/USD", "buy", 0.01, 1.34896),
    _pos("USD/ZAR", "sell", 0.01, 16.18429),
    _pos("USD/ZAR", "sell", 0.01, 16.18429),
]
BALANCE = 1939.24


class TestUsdRates:
    def test_usd_is_unity(self):
        assert tt.usd_rates([])["USD"] == 1.0

    def test_usd_quoted_pair_fixes_its_base(self):
        r = tt.usd_rates([_pos("AUD/USD", "buy", 0.01, 0.70659)])
        assert r["AUD"] == pytest.approx(0.70659)

    def test_usd_base_pair_fixes_its_quote(self):
        r = tt.usd_rates([_pos("USD/ZAR", "sell", 0.01, 16.18429)])
        assert r["ZAR"] == pytest.approx(1 / 16.18429)

    def test_cross_resolves_off_an_already_known_currency(self):
        # AUD/JPY alone cannot be priced; with AUD/USD present it can. This is
        # why the resolver iterates instead of doing a single pass.
        r = tt.usd_rates(LIVE_BOOK)
        assert r["JPY"] == pytest.approx(0.70659 / 111.569, rel=1e-6)

    def test_unresolvable_cross_is_omitted_not_guessed(self):
        r = tt.usd_rates([_pos("EUR/CAD", "buy", 0.01, 1.61131)])
        assert "EUR" not in r and "CAD" not in r


class TestNetExposure:
    def test_reproduces_the_live_book(self):
        exp = tt.net_exposure(LIVE_BOOK)
        assert exp["USD"] == pytest.approx(-4056, abs=2)
        assert exp["ZAR"] == pytest.approx(+2000, abs=2)
        assert exp["JPY"] == pytest.approx(+1413, abs=2)
        assert exp["GBP"] == pytest.approx(+1349, abs=2)
        assert exp["AUD"] == pytest.approx(-707, abs=2)

    def test_four_tickets_sum_to_more_than_the_account(self):
        # The whole point: each ticket passed the group guard on its own.
        assert abs(tt.net_exposure(LIVE_BOOK)["USD"]) > 2 * BALANCE

    def test_long_is_plus_base_minus_quote(self):
        legs = tt.currency_legs("EUR/USD", "buy", 0.01, 1.15563, {"EUR": 1.15563, "USD": 1.0})
        assert legs["EUR"] > 0 and legs["USD"] < 0

    def test_commodity_base_leg_is_not_a_currency(self):
        legs = tt.currency_legs("XAU/USD", "buy", 0.01, 4340.145, {"USD": 1.0})
        assert "XAU" not in legs
        # ...but the dollar leg still counts, at the metal's contract size.
        assert legs["USD"] == pytest.approx(-0.01 * 100 * 4340.145)

    def test_empty_book_is_empty(self):
        assert tt.net_exposure([]) == {}


class TestExposureConflict:
    def test_buying_metal_deepens_a_short_dollar_book(self):
        # The group guard calls XAG/USD long clean — it shares no group with
        # anything open. It is still one more way to be short the dollar.
        why = tt.exposure_conflict("XAG/USD", "Long", LIVE_BOOK, BALANCE)
        assert why and "USD" in why and "2.1x" in why

    def test_group_guard_misses_what_this_catches(self):
        assert tt.find_conflict("XAG/USD", "Long", LIVE_BOOK) is None
        assert tt.exposure_conflict("XAG/USD", "Long", LIVE_BOOK, BALANCE) is not None

    def test_offsetting_trade_is_allowed(self):
        # USD/CHF long is long dollar against a short-dollar book — it reduces
        # the concentration, so refusing it would be backwards.
        assert tt.exposure_conflict("USD/CHF", "Long", LIVE_BOOK, BALANCE) is None

    def test_pair_with_no_overweight_leg_is_allowed(self):
        assert tt.exposure_conflict("EUR/CAD", "Long", LIVE_BOOK, BALANCE) is None

    def test_below_the_limit_is_allowed(self):
        # GBP is +1,349 on a 1,939 balance = 0.70x — under the 1.0x limit.
        assert tt.exposure_conflict("GBP/CAD", "Long", LIVE_BOOK, BALANCE) is None

    def test_limit_ratio_is_tunable(self):
        assert tt.exposure_conflict("GBP/CAD", "Long", LIVE_BOOK, BALANCE,
                                    limit_ratio=0.5) is not None

    def test_empty_book_never_conflicts(self):
        assert tt.exposure_conflict("EUR/USD", "Long", [], BALANCE) is None

    def test_missing_balance_degrades_quietly(self):
        assert tt.exposure_conflict("EUR/USD", "Long", LIVE_BOOK, 0) is None

    def test_unparseable_direction_is_not_a_conflict(self):
        assert tt.exposure_conflict("EUR/USD", "Neutral", LIVE_BOOK, BALANCE) is None


class TestOffsettingLegs:
    def test_finds_the_aud_contradiction(self):
        # AUD/USD long and AUD/JPY short are in different CORR_GROUPS, so the
        # group guard never compares them. They are both AUD trades.
        found = tt.offsetting_legs(LIVE_BOOK)
        auds = [o for o in found if o["currency"] == "AUD"]
        assert len(auds) == 1
        assert auds[0]["long"] == ["AUD/USD"] and auds[0]["short"] == ["AUD/JPY"]

    def test_a_one_sided_book_has_none(self):
        assert tt.offsetting_legs([_pos("EUR/USD", "buy", 0.01, 1.15563)]) == []

    def test_deduplicates_repeated_pairs(self):
        book = [_pos("USD/ZAR", "sell", 0.01, 16.18), _pos("USD/ZAR", "sell", 0.01, 16.18),
                _pos("USD/ZAR", "buy", 0.01, 16.18)]
        found = [o for o in tt.offsetting_legs(book) if o["currency"] == "USD"]
        assert found and found[0]["short"] == ["USD/ZAR"]
