"""Unit tests for src/services/exposure.py — currency-leg exposure netting.

The anchor case throughout is the real book that motivated this module:

    sell GBP/AUD  ->  short GBP, LONG AUD
    buy  AUD/JPY  ->  LONG AUD,  short JPY
    sell EUR/AUD  ->  short EUR, LONG AUD

Three pairs, two different direction words, no shared CORR_GROUPS entry (there
is no AUD group at all) — and `CorrelationService.check_exposure` reports
nothing, because it matches pair names within a group *and* requires identical
direction strings. It is one 3× bet on AUD. These tests pin that this module
sees it.
"""
from __future__ import annotations

import pytest

from src.instruments.registry import CORR_GROUPS
from src.services.correlation_service import CorrelationService
from src.services.exposure import (
    LONG,
    SHORT,
    Position,
    check_stack,
    direction_side,
    net_currency_exposure,
    pair_legs,
    split_pair,
    worst_stack,
)

# The book from the MT4 screenshot, in the shape the store hands over.
BOOK = [
    {"pair": "GBP/AUD", "direction": "SHORT", "lot_size": 0.60,
     "label": "MT4 #262764269"},
    {"pair": "AUD/JPY", "direction": "LONG", "lot_size": 0.60,
     "label": "MT4 #262764346"},
]


class TestDirectionSide:
    @pytest.mark.parametrize("word", ["LONG", "long", "BUY", "buy", "Bullish",
                                      "STRONG_BUY"])
    def test_long_vocabularies(self, word):
        assert direction_side(word) == LONG

    @pytest.mark.parametrize("word", ["SHORT", "short", "SELL", "sell",
                                      "Bearish", "STRONG_SELL"])
    def test_short_vocabularies(self, word):
        assert direction_side(word) == SHORT

    @pytest.mark.parametrize("word", [None, "", "Neutral", "flat", "wat"])
    def test_unreadable_is_zero(self, word):
        assert direction_side(word) == 0

    def test_mt4_lowercase_is_what_makes_imports_work(self):
        """MT4 writes buy/sell, the ranker writes LONG/SHORT. Both must land
        on the same side or the guard silently ignores imported positions."""
        assert direction_side("buy") == direction_side("LONG")
        assert direction_side("sell") == direction_side("SHORT")


class TestSplitPair:
    def test_splits_and_uppercases(self):
        assert split_pair(" eur/aud ") == ("EUR", "AUD")

    @pytest.mark.parametrize("bad", [None, "", "EURAUD", "/USD", "EUR/", "XAUUSD"])
    def test_unparseable_is_none(self, bad):
        assert split_pair(bad) is None


class TestPairLegs:
    def test_long_is_long_base_short_quote(self):
        legs = pair_legs("EUR/AUD", "LONG")
        assert [(x.currency, x.side) for x in legs] == [("EUR", LONG), ("AUD", SHORT)]

    def test_short_is_the_mirror(self):
        legs = pair_legs("EUR/AUD", "SHORT")
        assert [(x.currency, x.side) for x in legs] == [("EUR", SHORT), ("AUD", LONG)]

    def test_weight_is_carried_to_both_legs(self):
        legs = pair_legs("GBP/AUD", "SHORT", weight=0.6)
        assert all(x.weight == 0.6 for x in legs)

    def test_unparseable_pair_yields_no_legs(self):
        assert pair_legs("XAUUSD", "LONG") == []

    def test_unreadable_direction_yields_no_legs(self):
        assert pair_legs("EUR/AUD", "Neutral") == []


class TestNetCurrencyExposure:
    def test_the_book_is_net_long_aud(self):
        net = net_currency_exposure(BOOK)
        assert net["AUD"] == pytest.approx(1.2)      # 0.6 + 0.6, same leg
        assert net["GBP"] == pytest.approx(-0.6)
        assert net["JPY"] == pytest.approx(-0.6)

    def test_opposing_pairs_net_out(self):
        book = [
            {"pair": "EUR/USD", "direction": "LONG", "lot_size": 1.0},
            {"pair": "EUR/USD", "direction": "SHORT", "lot_size": 1.0},
        ]
        assert net_currency_exposure(book) == pytest.approx(
            {"EUR": 0.0, "USD": 0.0})

    def test_accepts_position_objects(self):
        net = net_currency_exposure([Position("EUR/AUD", "SHORT", 0.15)])
        assert net["AUD"] == pytest.approx(0.15)

    def test_accepts_instrument_and_type_keys(self):
        """DB rows use instrument/direction; raw MT4 dicts use item/type."""
        net = net_currency_exposure([{"instrument": "AUD/JPY", "type": "buy",
                                      "size": 0.2}])
        assert net["AUD"] == pytest.approx(0.2)

    def test_unparseable_rows_contribute_nothing(self):
        net = net_currency_exposure(BOOK + [{"pair": "???", "direction": "LONG"},
                                            {"pair": "EUR/USD"}, "garbage", None])
        assert net["AUD"] == pytest.approx(1.2)

    def test_empty_book_is_empty(self):
        assert net_currency_exposure([]) == {}
        assert net_currency_exposure(None) == {}

    def test_bad_weight_falls_back_to_one(self):
        net = net_currency_exposure([{"pair": "EUR/AUD", "direction": "SHORT",
                                      "lot_size": "not-a-number"}])
        assert net["AUD"] == pytest.approx(1.0)


class TestCheckStack:
    def test_catches_the_third_long_aud_position(self):
        """The case that cost real money: EUR/AUD SHORT on top of a GBP/AUD
        SHORT and an AUD/JPY LONG is a third helping of long AUD."""
        warnings = check_stack("EUR/AUD", "SHORT", BOOK)
        by_ccy = {w.currency: w for w in warnings}
        assert "AUD" in by_ccy
        aud = by_ccy["AUD"]
        assert aud.side == LONG
        assert aud.count == 3
        assert {p.pair for p in aud.existing} == {"GBP/AUD", "AUD/JPY"}

    def test_corr_groups_misses_what_this_catches(self):
        """Documents *why* this module exists rather than an extra CORR_GROUPS
        entry: the legacy check sees nothing here, on two counts."""
        legacy = CorrelationService.check_exposure(
            [{"instrument": "GBP/AUD", "direction": "SHORT"},
             {"instrument": "AUD/JPY", "direction": "LONG"}],
            direction="SHORT", instrument="EUR/AUD",
        )
        assert legacy == []
        assert not any("AUD" in name for name in CORR_GROUPS)
        assert check_stack("EUR/AUD", "SHORT", BOOK)

    def test_clean_candidate_gives_no_warning(self):
        assert check_stack("USD/CAD", "LONG", BOOK) == []

    def test_opposite_side_of_a_held_currency_is_not_a_stack(self):
        """Short AUD against a long-AUD book reduces exposure; warning would be
        noise, and a guard that cries wolf gets switched off."""
        assert check_stack("AUD/USD", "SHORT", BOOK) == []

    def test_same_pair_same_direction_is_a_stack(self):
        warnings = check_stack("AUD/JPY", "LONG", BOOK)
        assert {w.currency for w in warnings} == {"AUD", "JPY"}

    def test_same_pair_opposite_direction_is_not(self):
        assert check_stack("AUD/JPY", "SHORT", BOOK) == []

    def test_reports_every_stacked_currency(self):
        book = [{"pair": "EUR/USD", "direction": "LONG", "lot_size": 1.0}]
        warnings = check_stack("EUR/USD", "LONG", book)
        assert {w.currency for w in warnings} == {"EUR", "USD"}

    def test_empty_book_never_warns(self):
        assert check_stack("EUR/AUD", "SHORT", []) == []

    def test_unparseable_candidate_never_warns(self):
        assert check_stack("XAUUSD", "LONG", BOOK) == []
        assert check_stack("EUR/AUD", "Neutral", BOOK) == []

    def test_max_per_currency_raises_the_bar(self):
        """Threshold 2 means 'warn from the third position on a currency'."""
        one = [{"pair": "GBP/AUD", "direction": "SHORT", "lot_size": 0.6}]
        assert check_stack("EUR/AUD", "SHORT", one, max_per_currency=2) == []
        assert check_stack("EUR/AUD", "SHORT", BOOK, max_per_currency=2)

    def test_held_weight_sums_the_matching_legs(self):
        aud = next(w for w in check_stack("EUR/AUD", "SHORT", BOOK)
                   if w.currency == "AUD")
        assert aud.held_weight == pytest.approx(1.2)

    def test_worst_offender_is_first(self):
        book = BOOK + [{"pair": "GBP/USD", "direction": "SHORT", "lot_size": 0.1}]
        warnings = check_stack("EUR/AUD", "SHORT", book)
        assert warnings[0].currency == "AUD"      # 2 matches beats GBP's 1

    def test_summary_names_the_open_positions(self):
        aud = next(w for w in check_stack("EUR/AUD", "SHORT", BOOK)
                   if w.currency == "AUD")
        text = aud.summary()
        assert "AUD long" in text
        assert "GBP/AUD SHORT" in text and "AUD/JPY LONG" in text
        assert "MT4 #262764269" in text

    def test_as_dict_is_json_safe(self):
        import json
        aud = next(w for w in check_stack("EUR/AUD", "SHORT", BOOK)
                   if w.currency == "AUD")
        json.dumps(aud.as_dict())
        assert aud.as_dict()["count"] == 3
        assert aud.as_dict()["side_word"] == "long"


class TestWorstStack:
    def test_returns_the_first_warning(self):
        warnings = check_stack("EUR/AUD", "SHORT", BOOK)
        assert worst_stack(warnings) is warnings[0]

    def test_none_when_clean(self):
        assert worst_stack([]) is None


class TestPositionDescribe:
    def test_includes_label_when_present(self):
        assert Position("EUR/AUD", "sell", 0.15, "MT4 #1").describe() == \
            "EUR/AUD SHORT (MT4 #1)"

    def test_omits_empty_label(self):
        assert Position("EUR/AUD", "buy").describe() == "EUR/AUD LONG"

    def test_unreadable_direction_shows_question_mark(self):
        assert "?" in Position("EUR/AUD", "sideways").describe()
