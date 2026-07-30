"""Unit tests for src/services/confluence_alert.py.

The whole value of this alert is that it is HARD to trigger — three
independent reads must agree. These tests pin every way it must refuse, since
a false positive here lands in the user's inbox and invites a real trade.
"""
from __future__ import annotations

import pytest

from src.services import confluence_alert as ca


def _ranker(direction="SHORT", pct=88.0, grade="A"):
    return {"direction": direction, "pct": pct, "grade": grade}


def _fib(status="ENTRY_FIRED"):
    return {"status": status, "entry": 1.6470, "sl": 1.6520,
            "tp1": 1.6400, "tp2": 1.6350, "rr1": 1.4}


def _eval(**kw):
    args = dict(pair="EUR/AUD", ranker=_ranker(), house_direction="BEARISH",
                house_score=-0.83, fib=_fib(), fib_bias="SHORT")
    args.update(kw)
    return ca.evaluate(**args)


class TestAgreement:
    def test_all_three_agree_short(self):
        c = _eval()
        assert c is not None
        assert c.pair == "EUR/AUD" and c.direction == "SHORT"
        assert c.is_fired is True
        assert c.entry == pytest.approx(1.6470)

    def test_all_three_agree_long(self):
        c = _eval(ranker=_ranker("LONG"), house_direction="BULLISH",
                  house_score=0.9, fib_bias="LONG")
        assert c is not None and c.direction == "LONG"

    def test_in_zone_also_qualifies(self):
        c = _eval(fib=_fib("IN_ZONE"))
        assert c is not None and c.is_fired is False


class TestRefusals:
    def test_ranker_below_grade_a(self):
        assert _eval(ranker=_ranker(pct=70.0, grade="B")) is None

    def test_house_view_opposes(self):
        assert _eval(house_direction="BULLISH") is None

    def test_house_view_neutral_is_not_agreement(self):
        # A flat board is an absence of conviction, not a green light.
        assert _eval(house_direction="NEUTRAL") is None
        assert _eval(house_direction=None) is None

    @pytest.mark.parametrize("status", ["RETRACING", "EXTENDED", "INVALIDATED"])
    def test_fib_not_live(self, status):
        assert _eval(fib=_fib(status)) is None

    def test_fib_computed_for_the_other_side(self):
        # A LONG fib read says nothing about a SHORT ranker call.
        assert _eval(fib_bias="LONG") is None

    def test_missing_inputs(self):
        assert _eval(ranker=None) is None
        assert _eval(fib=None) is None

    def test_malformed_direction(self):
        assert _eval(ranker=_ranker(direction="NEUTRAL")) is None
        assert _eval(ranker=_ranker(direction="")) is None

    def test_non_numeric_pct(self):
        assert _eval(ranker={"direction": "SHORT", "pct": "high", "grade": "A"}) is None

    def test_threshold_is_inclusive(self):
        assert _eval(ranker=_ranker(pct=80.0)) is not None
        assert _eval(ranker=_ranker(pct=79.9)) is None


class TestHouseMapping:
    def test_maps_both_directions(self):
        assert ca.house_direction_to_trade("BULLISH") == "LONG"
        assert ca.house_direction_to_trade("BEARISH") == "SHORT"
        assert ca.house_direction_to_trade("bullish") == "LONG"   # case-tolerant

    def test_neutral_and_junk_map_to_none(self):
        for v in ("NEUTRAL", "", None, "sideways"):
            assert ca.house_direction_to_trade(v) is None


class TestDedupeKey:
    def test_key_includes_status(self):
        # An IN_ZONE alert must not suppress the later, more actionable
        # ENTRY_FIRED on the same pair.
        a = _eval(fib=_fib("IN_ZONE")).dedupe_key()
        b = _eval(fib=_fib("ENTRY_FIRED")).dedupe_key()
        assert a != b

    def test_key_is_stable(self):
        assert _eval().dedupe_key() == _eval().dedupe_key()

    def test_direction_changes_key(self):
        short = _eval().dedupe_key()
        long_ = _eval(ranker=_ranker("LONG"), house_direction="BULLISH",
                      fib_bias="LONG").dedupe_key()
        assert short != long_


class TestEmail:
    def test_body_names_all_three_legs(self):
        html, plain = ca.build_email([_eval()])
        for token in ("Setup Ranker", "House view", "15M Fib"):
            assert token in html
        assert "EUR/AUD" in html and "SHORT" in html
        assert "ENTRY FIRED" in html
        # the plan must be present and formatted to FX precision
        assert "1.64700" in html and "1.65200" in html
        assert "EUR/AUD" in plain and "ranker A 88%" in plain

    def test_subject_flags_fired_over_in_zone(self):
        assert "ENTRY FIRED" in ca.subject_for([_eval()])
        assert "IN ZONE" in ca.subject_for([_eval(fib=_fib("IN_ZONE"))])

    def test_subject_truncates_long_lists(self):
        rows = [_eval() for _ in range(5)]
        assert ca.subject_for(rows).endswith("(all 3 agree)")
        assert "…" in ca.subject_for(rows)

    def test_empty_is_safe(self):
        assert ca.build_email([]) == ("", "")
        assert ca.subject_for([]) == "No confluence"

    def test_metals_use_two_decimals(self):
        c = _eval(ranker=_ranker("LONG"), house_direction="BULLISH",
                  fib_bias="LONG",
                  fib={"status": "IN_ZONE", "entry": 4040.25, "sl": 4010.0,
                       "tp1": 4100.0, "tp2": 4150.0, "rr1": 2.0})
        html, _ = ca.build_email([c])
        assert "4040.25" in html and "4040.25000" not in html
