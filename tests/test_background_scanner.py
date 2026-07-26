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
    return {"persisted": persisted, "stored": stored}


class TestEnsureStarted:
    def test_refuses_under_pytest(self):
        # PYTEST_CURRENT_TEST is set right now — the guard must trip, so no
        # daemon thread ever starts during a test run (live email/DB safety).
        assert bg.ensure_started() is False
        assert bg._thread is None


class TestScanOnce:
    def test_persists_and_emails_grade_a(self, monkeypatch, quiet_universe):
        from src.services import alert_service
        monkeypatch.setattr(alert_service, "email_configured", lambda: True)
        sent = []
        monkeypatch.setattr(
            alert_service, "send_email",
            lambda subject, html, plain, source=None: (sent.append((subject, source)) or (True, "")))
        cache = _FakeCache()
        monkeypatch.setattr(alert_service, "NotifyCache", lambda ns: cache)

        stats = bg.scan_once()

        assert stats["scored"] == 4            # 2 pairs × 2 directions
        assert stats["pairs"] == 2             # both pairs on the board
        assert stats["grade_a"] == 1
        assert stats["saved"] == 1 and quiet_universe["persisted"]
        assert stats["emailed"] == 1 and len(sent) == 1
        assert sent[0][1] == "setup_ranker_bg"  # audit-log source tag
        # Dedupe key is page-identical: pair|direction|alert_price_bucket
        from src.pages_lib.setup_ranker import alert_price_bucket
        assert cache.marked == [f"EUR/USD|LONG|{alert_price_bucket(1.1000)}"]

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

    def test_already_seen_setups_do_not_reemail(self, monkeypatch, quiet_universe):
        from src.pages_lib.setup_ranker import alert_price_bucket
        from src.services import alert_service
        monkeypatch.setattr(alert_service, "email_configured", lambda: True)
        monkeypatch.setattr(alert_service, "send_email",
                            lambda *a, **k: pytest.fail("must not send"))
        cache = _FakeCache(seen=[f"EUR/USD|LONG|{alert_price_bucket(1.1000)}"])
        monkeypatch.setattr(alert_service, "NotifyCache", lambda ns: cache)

        stats = bg.scan_once()
        assert stats["emailed"] == 0
        assert cache.marked == []              # ledger untouched

    def test_failed_send_leaves_ledger_unmarked(self, monkeypatch, quiet_universe):
        from src.services import alert_service
        monkeypatch.setattr(alert_service, "email_configured", lambda: True)
        monkeypatch.setattr(alert_service, "send_email",
                            lambda *a, **k: (False, "SMTP down"))
        cache = _FakeCache()
        monkeypatch.setattr(alert_service, "NotifyCache", lambda ns: cache)

        stats = bg.scan_once()
        assert stats["emailed"] == 0
        assert cache.marked == []              # retries next cycle

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
