"""Unit tests for src/services/background_scanner.py.

The worker's whole value is *not* inventing a second implementation — these
tests pin the contract: pytest-guarded startup, page-identical dedupe keys,
persist/email fan-out, and fail-soft behavior. All collaborators are
monkeypatched; nothing touches the network, SMTP, or a DB.
"""
from __future__ import annotations

from typing import List

import pandas as pd
import pytest

from src.services import background_scanner as bg


def _fake_result(pair: str, direction: str = "LONG", grade: str = "A",
                 close: float = 1.1000) -> dict:
    return {"pair": pair, "direction": direction, "grade": grade,
            "close": close, "pct": 85 if grade == "A" else 50,
            "score": 6, "max_score": 7, "scores": {}, "sl_pips": 30}


_EMPTY = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])


class _FakeCache:
    def __init__(self, seen=None):
        self.seen = set(seen or [])
        self.marked: List[str] = []

    def load(self):
        return set(self.seen)

    def filter_new(self, keys):
        self.marked.extend(keys)
        return keys


@pytest.fixture
def quiet_universe(monkeypatch):
    """Shrink the universe to 2 pairs and stub every impure collaborator."""
    from src.pages_lib import setup_ranker as sr

    universe = {"EUR/USD": {"ticker": "EURUSD=X"}, "XAU/USD": {"ticker": "GC=F"}}
    monkeypatch.setattr("src.instruments.INSTRUMENTS", universe)

    def fake_score(pair, info, direction):
        # EUR/USD LONG is the only Grade A in the stub universe.
        grade = "A" if (pair == "EUR/USD" and direction == "LONG") else "C"
        return _fake_result(pair, direction, grade)

    monkeypatch.setattr(sr._SetupRankerDataFeed, "score",
                        staticmethod(fake_score))

    persisted: List[list] = []
    monkeypatch.setattr(
        sr.SetupRankerPage, "_persist_signals",
        staticmethod(lambda results, rr: persisted.append(list(results))))
    monkeypatch.setattr(
        sr.SetupRankerPage, "_build_alert_email",
        staticmethod(lambda *a, **k: ("<html>", "plain")))

    from src.services import account_state
    monkeypatch.setattr(account_state, "get_balance", lambda default: 10000.0)

    # Stub the two impure seams the cycle adds: the ingest fetch (no network)
    # and the board store (capture instead of writing). Zero the rate-limit
    # delay so the test isn't paced by the real sweep cadence.
    monkeypatch.setattr(bg, "_pair_frames",
                        lambda ticker: (_EMPTY, _EMPTY, _EMPTY))
    stored: List[dict] = []
    monkeypatch.setattr(bg, "store_board", lambda board: stored.append(board))
    monkeypatch.setattr(bg, "_PER_PAIR_DELAY_S", 0)
    # No confluence by default — the real one fetches live 15M bars. Tests that
    # care about emailing stub this with a Confluence of their own.
    monkeypatch.setattr(bg, "_find_confluences", lambda grade_a, board: [])
    return {"persisted": persisted, "stored": stored}


def _confluence(pair="EUR/USD", direction="LONG", status="ENTRY_FIRED"):
    from src.services.confluence_alert import Confluence
    return Confluence(pair=pair, direction=direction, ranker_pct=88.0,
                      ranker_grade="A", house_direction="BULLISH",
                      house_score=0.9, fib_status=status, entry=1.10,
                      sl=1.095, tp1=1.11, tp2=1.12, rr1=2.0)


class TestEnsureStarted:
    def test_refuses_under_pytest(self):
        # PYTEST_CURRENT_TEST is set right now — the guard must trip, so no
        # daemon thread ever starts during a test run (live email/DB safety).
        assert bg.ensure_started() is False
        assert bg._thread is None


class TestScanOnce:
    def test_grade_a_persists_but_does_not_email_on_its_own(
            self, monkeypatch, quiet_universe):
        # The behaviour change that matters: Grade A alone is journalled but is
        # NOT worth an interruption. Only triple confluence emails.
        from src.services import alert_service
        monkeypatch.setattr(alert_service, "email_configured", lambda: True)
        monkeypatch.setattr(alert_service, "send_email",
                            lambda *a, **k: pytest.fail("Grade A alone must not email"))
        monkeypatch.setattr(alert_service, "NotifyCache", lambda ns: _FakeCache())

        stats = bg.scan_once()

        assert stats["scored"] == 4            # 2 pairs × 2 directions
        assert stats["pairs"] == 2             # both pairs on the board
        assert stats["grade_a"] == 1
        assert stats["saved"] == 1 and quiet_universe["persisted"]
        assert stats["emailed"] == 0

    def test_emails_on_triple_confluence(self, monkeypatch, quiet_universe):
        from src.services import alert_service
        monkeypatch.setattr(alert_service, "email_configured", lambda: True)
        sent = []
        monkeypatch.setattr(
            alert_service, "send_email",
            lambda subject, html, plain, source=None: (sent.append((subject, source)) or (True, "")))
        cache = _FakeCache()
        monkeypatch.setattr(alert_service, "NotifyCache", lambda ns: cache)
        c = _confluence()
        monkeypatch.setattr(bg, "_find_confluences", lambda grade_a, board: [c])

        stats = bg.scan_once()

        assert stats["emailed"] == 1 and len(sent) == 1
        assert "ENTRY FIRED" in sent[0][0]          # subject leads with the trigger
        assert sent[0][1] == "confluence_bg"        # audit-log source tag
        assert cache.marked == [c.dedupe_key()]

    def test_confluence_already_alerted_is_not_resent(
            self, monkeypatch, quiet_universe):
        from src.services import alert_service
        monkeypatch.setattr(alert_service, "email_configured", lambda: True)
        monkeypatch.setattr(alert_service, "send_email",
                            lambda *a, **k: pytest.fail("must not resend"))
        c = _confluence()
        cache = _FakeCache(seen=[c.dedupe_key()])
        monkeypatch.setattr(alert_service, "NotifyCache", lambda ns: cache)
        monkeypatch.setattr(bg, "_find_confluences", lambda grade_a, board: [c])

        assert bg.scan_once()["emailed"] == 0
        assert cache.marked == []

    def test_stores_a_board_for_every_pair(self, monkeypatch, quiet_universe):
        from src.services import alert_service
        monkeypatch.setattr(alert_service, "email_configured", lambda: False)

        bg.scan_once()

        stored = quiet_universe["stored"]
        assert len(stored) == 1                # one board written per cycle
        board = stored[0]
        assert "computed_at" in board
        assert set(board["pairs"]) == {"EUR/USD", "XAU/USD"}
        # Each pair carries a serialized house view + a best-setup summary.
        eur = board["pairs"]["EUR/USD"]
        assert "hv" in eur and eur["hv"]["pair"] == "EUR/USD"
        assert eur["setup"]["grade"] == "A"    # best of LONG/SHORT kept
        assert eur["setup"]["direction"] == "LONG"

    def test_failed_send_leaves_ledger_unmarked(self, monkeypatch, quiet_universe):
        # A transient SMTP failure must not consume the ledger, or the alert is
        # lost forever instead of retrying next cycle.
        from src.services import alert_service
        monkeypatch.setattr(alert_service, "email_configured", lambda: True)
        monkeypatch.setattr(alert_service, "send_email",
                            lambda *a, **k: (False, "SMTP down"))
        cache = _FakeCache()
        monkeypatch.setattr(alert_service, "NotifyCache", lambda ns: cache)
        monkeypatch.setattr(bg, "_find_confluences",
                            lambda grade_a, board: [_confluence()])

        stats = bg.scan_once()
        assert stats["emailed"] == 0
        assert cache.marked == []              # retries next cycle

    def test_confluence_failure_does_not_break_the_cycle(
            self, monkeypatch, quiet_universe):
        # The 15M leg hits the network; if it throws, ingest/score/store must
        # still complete — the board is the more important product.
        from src.services import alert_service
        monkeypatch.setattr(alert_service, "email_configured", lambda: True)

        def boom(grade_a, board):
            raise RuntimeError("yahoo 15m failed")
        monkeypatch.setattr(bg, "_find_confluences", boom)

        stats = bg.scan_once()
        assert stats["pairs"] == 2 and stats["saved"] == 1
        assert stats["emailed"] == 0

    def test_no_email_config_still_persists(self, monkeypatch, quiet_universe):
        from src.services import alert_service
        monkeypatch.setattr(alert_service, "email_configured", lambda: False)

        stats = bg.scan_once()
        assert stats["saved"] == 1             # DB path independent of email
        assert stats["emailed"] == 0

    def test_one_bad_ticker_does_not_kill_sweep(self, monkeypatch, quiet_universe):
        from src.pages_lib import setup_ranker as sr
        from src.services import alert_service
        monkeypatch.setattr(alert_service, "email_configured", lambda: False)

        def flaky_score(pair, info, direction):
            if pair == "XAU/USD":
                raise RuntimeError("yahoo hiccup")
            return _fake_result(pair, direction, "C")

        monkeypatch.setattr(sr._SetupRankerDataFeed, "score",
                            staticmethod(flaky_score))
        stats = bg.scan_once()
        assert stats["scored"] == 2            # EUR/USD both directions survived
        assert stats["grade_a"] == 0


class TestConfluenceChannels:
    """Email and Telegram are independent delivery channels.

    Until 2026-08-20 the whole alert block sat behind
    `alert_service.email_configured()`, so an owner with Telegram configured
    and no SMTP received nothing at all — the entry signal was suppressed by a
    check about a different delivery mechanism.
    """

    @staticmethod
    def _wire(monkeypatch, *, email_ok=None, tg_ok=True):
        """Stub both channels and capture what each was asked to do."""
        from src.core import secrets as core_secrets
        from src.services import alert_service

        calls = {"email": [], "telegram": [], "marked": []}

        monkeypatch.setattr(bg, "_find_confluences",
                            lambda grade_a, board: [_confluence()])
        monkeypatch.setattr(alert_service, "email_configured",
                            lambda: email_ok is not None)
        monkeypatch.setattr(
            alert_service, "send_email",
            lambda *a, **k: (calls["email"].append(a) or (bool(email_ok), "x")))
        monkeypatch.setattr(
            core_secrets, "send_telegram_message",
            lambda text: (calls["telegram"].append(text) or (tg_ok, "x")))
        monkeypatch.setattr(alert_service.NotifyCache, "load", lambda self: set())
        monkeypatch.setattr(alert_service.NotifyCache, "filter_new",
                            lambda self, keys: calls["marked"].extend(keys) or list(keys))
        return calls

    def test_telegram_is_sent_even_when_email_is_unconfigured(
            self, quiet_universe, monkeypatch):
        calls = self._wire(monkeypatch, email_ok=None)
        bg.scan_once()
        assert calls["telegram"], "no Telegram sent with email unconfigured"
        assert calls["email"] == []

    def test_both_channels_fire_when_both_are_configured(
            self, quiet_universe, monkeypatch):
        calls = self._wire(monkeypatch, email_ok=True)
        bg.scan_once()
        assert calls["email"] and calls["telegram"]

    def test_delivery_by_telegram_alone_still_dedupes(
            self, quiet_universe, monkeypatch):
        # filter_new() records keys as seen. Marking only on email success
        # would re-alert the same setup on every cycle, forever.
        calls = self._wire(monkeypatch, email_ok=False, tg_ok=True)
        bg.scan_once()
        assert calls["marked"], "delivered but never marked seen"

    def test_nothing_is_marked_seen_when_every_channel_fails(
            self, quiet_universe, monkeypatch):
        # The opposite error: silently swallowing an alert nobody received.
        calls = self._wire(monkeypatch, email_ok=False, tg_ok=False)
        bg.scan_once()
        assert calls["marked"] == []

    def test_the_message_carries_the_pair_and_direction(
            self, quiet_universe, monkeypatch):
        calls = self._wire(monkeypatch, email_ok=None)
        bg.scan_once()
        assert "EUR/USD" in calls["telegram"][0] and "LONG" in calls["telegram"][0]
