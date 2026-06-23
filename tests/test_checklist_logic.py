"""Save-path logic tests for the Daily Trading checklist.

Drives ChecklistController._save_setup directly with a fake Streamlit module so
the row-building, daily-loss block, and cache-invalidation logic are tested
without a Streamlit runtime or a live Postgres.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.pages_lib.daily_trading import checklist as cl
from src.pages_lib.daily_trading.state import CHECKS_TOTAL
from src.db.trade_repository import DBConfig


class FakeSessionState(dict):
    """Supports both attribute (st.session_state.x) and item (st.session_state['x']) access."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


class FakeSt:
    def __init__(self, session_state):
        self.session_state = session_state
        self.errors = []
        self.successes = []

    def error(self, msg):
        self.errors.append(msg)

    def success(self, msg):
        self.successes.append(msg)


def _session_state(db_ok=True):
    ss = FakeSessionState()
    ss.db_ok = db_ok
    ss.session = "London Kill Zone"
    ss.account_bal = 10_000.0
    ss.risk_pct = 1.0
    ss.notes = "clean setup"
    for i in range(1, CHECKS_TOTAL + 1):
        ss[f"check_{i}"] = True
    return ss


@pytest.fixture
def wired(monkeypatch):
    """Patch the checklist module's st / repo / cache and capture what's saved."""
    saved = {}
    cleared = {"called": False}

    class FakeRepo:
        def save_setup(self, row):
            saved["row"] = row

    ss = _session_state()
    fake_st = FakeSt(ss)
    monkeypatch.setattr(cl, "st", fake_st)
    monkeypatch.setattr(cl, "pooled_repository", lambda cfg: FakeRepo())
    monkeypatch.setattr(cl, "clear_read_caches",
                        lambda: cleared.__setitem__("called", True))
    return SimpleNamespace(st=fake_st, ss=ss, saved=saved, cleared=cleared)


def _call_save(verdict="GO", dll=None):
    risk = SimpleNamespace(lot_size=0.55, risk_amount=100.0, rr_tp1=2.0, rr_tp2=3.0)
    dll = dll or {"losses_today": 0, "limit": 2, "blocked": False}
    cl.ChecklistController._save_setup(
        DBConfig(), "EUR/USD", "EUR/USD", "Long", verdict, "go",
        17, 0.0012, 0.0010, 18.0, 36.0, 54.0, risk, dll,
    )


class TestSavePath:
    def test_happy_path_saves_complete_row(self, wired):
        _call_save()
        row = wired.saved["row"]
        # Every persisted field is present and correct.
        assert row["instrument"] == "EUR/USD"
        assert row["ticker"] == "EURUSD=X"
        assert row["direction"] == "Long"
        assert row["session"] == "London Kill Zone"
        assert row["score"] == f"17/{CHECKS_TOTAL}"
        assert row["verdict"] == "GO"
        assert row["atr14"] == 0.0012 and row["atr20"] == 0.0010
        assert row["sl_pips"] == 18.0 and row["tp1_pips"] == 36.0 and row["tp2_pips"] == 54.0
        assert row["lot_size"] == 0.55
        assert row["risk_amount"] == 100.0
        assert row["rr_tp1"] == 2.0 and row["rr_tp2"] == 3.0
        assert row["account_bal"] == 10_000.0
        assert row["risk_pct"] == 1.0
        assert row["checks_passed"] == 17 and row["checks_total"] == CHECKS_TOTAL
        assert "logged_at" in row and "notes" in row

    def test_all_checks_persisted_as_json(self, wired):
        _call_save()
        detail = json.loads(wired.saved["row"]["checks_detail"])
        assert len(detail) == CHECKS_TOTAL
        assert all(detail[f"check_{i}"] is True for i in range(1, CHECKS_TOTAL + 1))

    def test_cache_invalidated_after_save(self, wired):
        _call_save()
        assert wired.cleared["called"] is True
        assert wired.st.successes  # success message shown

    def test_not_connected_blocks_save(self, monkeypatch):
        saved = {}

        class FakeRepo:
            def save_setup(self, row):
                saved["row"] = row

        ss = _session_state(db_ok=False)
        fake_st = FakeSt(ss)
        monkeypatch.setattr(cl, "st", fake_st)
        monkeypatch.setattr(cl, "pooled_repository", lambda cfg: FakeRepo())
        _call_save()
        assert "row" not in saved
        assert fake_st.errors and "Not connected" in fake_st.errors[0]

    def test_daily_loss_limit_blocks_save(self, wired):
        _call_save(dll={"losses_today": 2, "limit": 2, "blocked": True})
        assert "row" not in wired.saved
        assert wired.st.errors and "Daily loss limit" in wired.st.errors[0]
        assert wired.cleared["called"] is False
