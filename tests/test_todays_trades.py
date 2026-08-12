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


# ── horizon window, source stability, direction casing ────────────────────────
# Why these exist: on 2026-08-11 the board recommended XPT/USD *short*, and the
# next day read it *long* from three sources. Two causes, both here: consensus
# judged a 10-day thesis on a rolling 24-hour window, so votes reversed as old
# rows aged out; and `biased_pivots` had emitted both directions on XPT/USD on
# the same day with both counted. 12 of 27 instruments carried both directions
# inside a week.
from datetime import datetime, timedelta


def _sig(pair, direction, source, days_ago=0.0, horizon=None):
    row = {"instrument": pair, "direction": direction, "source": source,
           "logged_at": datetime.utcnow() - timedelta(days=days_ago)}
    if horizon is not None:
        row["checks_detail"] = {"horizon_days": horizon}
    return row


class TestSignalHorizon:
    def test_reads_the_rows_own_horizon(self):
        assert tt.signal_horizon_days(_sig("EUR/USD", "Long", "s", horizon=20)) == 20

    def test_falls_back_to_the_default(self):
        assert tt.signal_horizon_days(_sig("EUR/USD", "Long", "s")) == tt.DEFAULT_HORIZON_DAYS

    def test_json_string_detail_is_parsed(self):
        row = _sig("EUR/USD", "Long", "s")
        row["checks_detail"] = '{"horizon_days": 5}'
        assert tt.signal_horizon_days(row) == 5

    def test_garbage_detail_does_not_raise(self):
        row = _sig("EUR/USD", "Long", "s")
        row["checks_detail"] = "not json"
        assert tt.signal_horizon_days(row) == tt.DEFAULT_HORIZON_DAYS

    def test_nonpositive_horizon_is_ignored(self):
        assert tt.signal_horizon_days(
            _sig("EUR/USD", "Long", "s", horizon=0)) == tt.DEFAULT_HORIZON_DAYS


class TestIsLive:
    def test_fresh_signal_is_live(self):
        assert tt.is_live(_sig("EUR/USD", "Long", "s", days_ago=1)) is True

    def test_signal_past_its_horizon_is_not(self):
        assert tt.is_live(_sig("EUR/USD", "Long", "s", days_ago=40, horizon=5)) is False

    def test_trading_days_are_converted_to_calendar_days(self):
        # 10 trading days is ~14 calendar days: a signal 12 days old is still
        # inside its thesis, and expiring it on day 10 would cut a fortnight
        # short by a weekend.
        assert tt.is_live(_sig("EUR/USD", "Long", "s", days_ago=12, horizon=10)) is True
        assert tt.is_live(_sig("EUR/USD", "Long", "s", days_ago=15, horizon=10)) is False

    def test_a_long_horizon_outlives_a_short_one(self):
        old = 18
        assert tt.is_live(_sig("X/Y", "Long", "s", days_ago=old, horizon=20)) is True
        assert tt.is_live(_sig("X/Y", "Long", "s", days_ago=old, horizon=5)) is False

    def test_missing_timestamp_counts_as_live(self):
        # A clock we cannot read is a data problem, not an expiry — silently
        # shrinking the board would be worse than counting the row.
        assert tt.is_live({"instrument": "EUR/USD", "direction": "Long"}) is True


class TestUnstableSources:
    def test_detects_a_source_holding_both_directions(self):
        rows = [_sig("XPT/USD", "Long", "biased_pivots"),
                _sig("XPT/USD", "Short", "biased_pivots")]
        assert tt.unstable_sources(rows) == {"XPT/USD": {"biased_pivots"}}

    def test_two_sources_disagreeing_is_not_instability(self):
        # daily_macd long and weekly_ema short is a genuine disagreement between
        # timeframes — that is what `against` is for, not a contradiction.
        rows = [_sig("XPT/USD", "Long", "daily_macd"),
                _sig("XPT/USD", "Short", "weekly_ema")]
        assert tt.unstable_sources(rows) == {}

    def test_is_scoped_per_instrument(self):
        rows = [_sig("EUR/USD", "Long", "s"), _sig("GBP/USD", "Short", "s")]
        assert tt.unstable_sources(rows) == {}


class TestConsensusRespectsHorizon:
    def test_expired_signals_do_not_vote(self):
        rows = [_sig("EUR/USD", "Long", "a", days_ago=0.1, horizon=10),
                _sig("EUR/USD", "Short", "b", days_ago=40, horizon=5)]
        ideas = {i.pair: i for i in tt.consensus(rows)}
        assert ideas["EUR/USD"].direction == tt.LONG
        assert ideas["EUR/USD"].against == []      # the stale short is gone

    def test_yesterdays_vote_still_counts_today(self):
        # The XPT case: a 10-day view must not evaporate overnight.
        rows = [_sig("XPT/USD", "Short", "weekly_ema", days_ago=1, horizon=10),
                _sig("XPT/USD", "Short", "daily_macd", days_ago=0.1, horizon=10)]
        ideas = {i.pair: i for i in tt.consensus(rows)}
        assert sorted(ideas["XPT/USD"].agree) == ["daily_macd", "weekly_ema"]

    def test_opting_out_restores_the_old_behaviour(self):
        rows = [_sig("EUR/USD", "Short", "b", days_ago=99, horizon=5)]
        assert tt.consensus(rows) == []
        assert tt.consensus(rows, respect_horizon=False) != []

    def test_a_self_contradicting_source_is_excluded_and_reported(self):
        rows = [_sig("XPT/USD", "Long", "biased_pivots"),
                _sig("XPT/USD", "Short", "biased_pivots"),
                _sig("XPT/USD", "Long", "daily_macd")]
        ideas = {i.pair: i for i in tt.consensus(rows)}
        idea = ideas["XPT/USD"]
        assert idea.agree == ["daily_macd"]          # pivots does not vote
        assert idea.unstable == ["biased_pivots"]    # ...but is surfaced
        assert idea.direction == tt.LONG

    def test_excluding_an_unstable_source_can_empty_a_pair(self):
        rows = [_sig("XPT/USD", "Long", "biased_pivots"),
                _sig("XPT/USD", "Short", "biased_pivots")]
        assert tt.consensus(rows) == []

    def test_mixed_case_directions_are_one_direction(self):
        # LONG/Long/Short were three strings for two directions in the column.
        rows = [_sig("EUR/USD", "LONG", "a"), _sig("EUR/USD", "Long", "b")]
        ideas = {i.pair: i for i in tt.consensus(rows)}
        assert ideas["EUR/USD"].direction == tt.LONG
        assert sorted(ideas["EUR/USD"].agree) == ["a", "b"]
