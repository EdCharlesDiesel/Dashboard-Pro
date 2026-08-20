"""Unit tests for src/core/threat_sentry_hook.py.

This module is measured by coverage and had no tests at all before 2026-08-20.

No live database and no Telegram: a fake engine yields a fake connection, and
`_send_telegram` is stubbed in every test that can reach an alert path, so a
unit run can never message a real chat.
"""
from __future__ import annotations

import pytest

from src.core import threat_core as tc
from src.core import threat_sentry_hook as hook


BOOK_ROW = {"pair": "USD/ZAR", "direction": "SHORT", "lot_size": 0.2,
            "entry_price": 16.12873, "stop_loss": 16.34195, "has_stop": True}

BASE_DETAIL = {
    "usdjpy_last": 159.1, "usdjpy_roc5_pct": -0.12,
    "worst_cluster_usd": 6080.0, "worst_cluster_pct_equity": 170.3,
    "worst_cluster_ccy": "ZAR", "headline_hits": [], "red_events": [],
}


class FakeConn:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeEngine:
    def connect(self):
        return FakeConn()


def _report(state="red", score=30.0, components=None, detail=None):
    return tc.ThreatReport(
        components if components is not None else {"concentration": 100.0},
        {**BASE_DETAIL, **(detail or {})}, score, state)


@pytest.fixture(autouse=True)
def _no_side_effects(monkeypatch):
    """Nothing in this file may touch Telegram, the DB, or the network."""
    monkeypatch.setattr(hook, "_send_telegram", lambda msg: None)
    monkeypatch.setattr(hook.tc, "ensure_tables", lambda conn: None)
    monkeypatch.setattr(hook.tc, "journal", lambda conn, rep: None)
    monkeypatch.setattr(hook.tc, "last_state", lambda conn: None)
    monkeypatch.setattr(hook.tc, "build_report",
                        lambda pos, eq, zone: _report())
    monkeypatch.setattr(hook.open_positions, "load", lambda: [BOOK_ROW])
    monkeypatch.setattr(hook.open_positions, "account_snapshot",
                        lambda: {"equity": 3506.37, "balance": 3844.15})
    monkeypatch.setattr(hook.open_positions, "age_minutes", lambda: 3.0)


class TestPositionSource:
    def test_it_reads_the_mt5_book_not_the_hand_typed_table(self, monkeypatch):
        # The page moved to the book on 2026-08-20 and this did not, so the
        # hook evaluated an empty table and returned None - a sentry that is
        # silent by construction, where silence reads as "all clear".
        monkeypatch.setattr(hook.tc, "load_positions",
                            lambda conn: pytest.fail("read the abandoned table"))
        assert hook.run_threat_check(FakeEngine(), equity=3500.0) is not None

    def test_equity_defaults_to_the_stored_snapshot(self, monkeypatch):
        seen = {}

        def spy(pos, eq, zone):
            seen["eq"] = eq
            return _report()

        monkeypatch.setattr(hook.tc, "build_report", spy)
        hook.run_threat_check(FakeEngine())            # no equity argument
        assert seen["eq"] == pytest.approx(3506.37)

    def test_an_explicit_equity_still_wins(self, monkeypatch):
        seen = {}

        def spy(pos, eq, zone):
            seen["eq"] = eq
            return _report()

        monkeypatch.setattr(hook.tc, "build_report", spy)
        hook.run_threat_check(FakeEngine(), equity=1234.0)
        assert seen["eq"] == 1234.0

    def test_no_equity_anywhere_returns_none_rather_than_guessing(self, monkeypatch):
        # Never substitute a constant - that is exactly the $935 bug.
        monkeypatch.setattr(hook.open_positions, "account_snapshot", lambda: None)
        assert hook.run_threat_check(FakeEngine()) is None

    def test_an_empty_book_returns_none(self, monkeypatch):
        monkeypatch.setattr(hook.open_positions, "load", lambda: [])
        assert hook.run_threat_check(FakeEngine(), equity=3500.0) is None


class TestStaleBook:
    def test_a_stale_book_is_not_evaluated(self, monkeypatch):
        monkeypatch.setattr(hook.open_positions, "age_minutes", lambda: 620.0)
        monkeypatch.setattr(hook.tc, "build_report",
                            lambda *a: pytest.fail("judged a 10-hour-old book"))
        assert hook.run_threat_check(FakeEngine(), equity=3500.0) is None

    def test_a_stale_book_alerts_that_the_feed_is_dead(self, monkeypatch):
        sent = []
        monkeypatch.setattr(hook, "_send_telegram", sent.append)
        monkeypatch.setattr(hook.open_positions, "age_minutes", lambda: 620.0)
        hook.run_threat_check(FakeEngine(), equity=3500.0)
        assert sent, "a dead feed must not be silent"
        assert "620" in sent[0] and "sync" in sent[0].lower()

    def test_the_dead_feed_alert_does_not_repeat_every_run(self, monkeypatch):
        # A 15-minute loop must not raise the same alarm four times an hour.
        sent = []
        monkeypatch.setattr(hook, "_send_telegram", sent.append)
        monkeypatch.setattr(hook.open_positions, "age_minutes", lambda: 620.0)
        monkeypatch.setattr(hook.tc, "last_state", lambda conn: "stale")
        hook.run_threat_check(FakeEngine(), equity=3500.0)
        assert sent == []

    def test_a_fresh_book_is_evaluated_normally(self):
        assert hook.run_threat_check(FakeEngine(), equity=3500.0) is not None

    def test_an_unknown_age_is_treated_as_fresh(self, monkeypatch):
        # age_minutes() is None when nothing is stored, which is the empty-book
        # case the position check already covers. Do not report it twice.
        monkeypatch.setattr(hook.open_positions, "age_minutes", lambda: None)
        assert hook.run_threat_check(FakeEngine(), equity=3500.0) is not None


class TestAlertText:
    def test_the_driver_comes_from_state_driver_not_the_top_raw_score(self):
        # Since 1.10.26 the headline follows the worst component, and
        # detail["state_driver"] records what set it. Two sources for one fact
        # drift apart; use the authoritative one.
        rep = _report(components={"concentration": 40.0, "squeeze": 95.0},
                      detail={"state_driver": ["concentration"]})
        assert "Concentration" in hook._format_alert("green", rep)

    def test_it_falls_back_to_the_top_score_on_an_older_report(self):
        # Reports journaled before 1.10.26 carry no state_driver.
        rep = _report(components={"concentration": 40.0, "squeeze": 95.0})
        assert "squeeze" in hook._format_alert("green", rep).lower()

    def test_unstopped_positions_are_named(self):
        rep = _report(detail={"unstopped": ["EUR/ZAR"]})
        text = hook._format_alert("green", rep)
        assert "NO STOP" in text.upper() and "EUR/ZAR" in text

    def test_no_unstopped_line_when_every_position_has_a_stop(self):
        assert "NO STOP" not in hook._format_alert("green", _report()).upper()
