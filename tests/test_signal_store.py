"""Unit tests for src/services/signal_store.py — the shared signal persistence
service (dedupe, source tagging, graceful no-op). No live DB or Streamlit."""
from __future__ import annotations

import pytest

from src.services import signal_store
from src.db.trade_repository import DBConfig


class FakeRepo:
    def __init__(self):
        self.saved = []  # list of (row, source)

    def save_setup(self, row, source=None):
        self.saved.append((row, source))


@pytest.fixture
def signal():
    return {
        "pair": "EUR/USD", "bias": "Long", "conviction": "High",
        "strength_score": 9, "entry": 1.10000, "stop_loss": 1.09700,
        "take_profit_1": 1.10500, "take_profit_2": 1.11000,
        "stop_loss_pips": 30.0, "risk_reward_1": 1.67, "risk_reward_2": 3.33,
        "thesis": "MTF uptrend", "atr": 0.0015,
    }


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """Wire the service to a fake repo + tmp dedupe dir, DB 'configured'."""
    repo = FakeRepo()
    monkeypatch.setattr(signal_store, "_resolve_cfg", lambda: DBConfig())
    monkeypatch.setattr(signal_store, "pooled_repository", lambda cfg: repo)
    monkeypatch.setattr(signal_store, "clear_read_caches", lambda: None)
    monkeypatch.setattr(signal_store.SessionService, "current",
                        classmethod(lambda cls, now=None: {"window": "NY Kill Zone"}))
    # Keep dedupe JSON files out of the repo root.
    monkeypatch.chdir(tmp_path)
    return repo


class TestPersistSignals:
    def test_saves_and_tags_source(self, wired, signal):
        n = signal_store.persist_signals("setup_ranker", [signal])
        assert n == 1
        row, source = wired.saved[0]
        assert source == "setup_ranker"
        assert row["instrument"] == "EUR/USD"
        assert row["session"] == "NY Kill Zone"

    def test_dedupes_across_calls(self, wired, signal):
        assert signal_store.persist_signals("setup_ranker", [signal]) == 1
        # Same signal again → already seen → not saved a second time.
        assert signal_store.persist_signals("setup_ranker", [signal]) == 0
        assert len(wired.saved) == 1

    def test_new_price_level_saves_again(self, wired, signal):
        assert signal_store.persist_signals("setup_ranker", [signal]) == 1
        moved = {**signal, "entry": 1.12000}  # new rounded entry → new dedupe key
        assert signal_store.persist_signals("setup_ranker", [moved]) == 1
        assert len(wired.saved) == 2

    def test_per_source_namespaces_are_independent(self, wired, signal):
        assert signal_store.persist_signals("setup_ranker", [signal]) == 1
        # Same signal, different source → different dedupe ledger → saved.
        assert signal_store.persist_signals("daily_trend", [signal]) == 1
        assert {s for _, s in wired.saved} == {"setup_ranker", "daily_trend"}

    def test_empty_is_noop(self, wired):
        assert signal_store.persist_signals("x", []) == 0
        assert wired.saved == []

    def test_no_db_does_not_consume_dedupe(self, tmp_path, monkeypatch, signal):
        # DB unconfigured → return 0 AND leave the ledger untouched, so once the
        # DB is configured the same signal still saves.
        monkeypatch.setattr(signal_store, "_resolve_cfg", lambda: None)
        monkeypatch.chdir(tmp_path)
        assert signal_store.persist_signals("setup_ranker", [signal]) == 0

        repo = FakeRepo()
        monkeypatch.setattr(signal_store, "_resolve_cfg", lambda: DBConfig())
        monkeypatch.setattr(signal_store, "pooled_repository", lambda cfg: repo)
        monkeypatch.setattr(signal_store, "clear_read_caches", lambda: None)
        monkeypatch.setattr(signal_store.SessionService, "current",
                            classmethod(lambda cls, now=None: {"window": ""}))
        assert signal_store.persist_signals("setup_ranker", [signal]) == 1

    def test_save_failure_is_swallowed_and_not_marked_seen(self, wired, signal, monkeypatch):
        def boom(row, source=None):
            raise RuntimeError("db down mid-write")
        monkeypatch.setattr(wired, "save_setup", boom)
        # One signal that fails to save → 0 saved, no crash.
        assert signal_store.persist_signals("setup_ranker", [signal]) == 0


class TestDefaultDedupeKey:
    def test_includes_pair_bias_and_rounded_entry(self):
        k = signal_store.default_dedupe_key(
            {"pair": "EUR/USD", "bias": "Long", "entry": 1.123456})
        assert k == "EUR/USD_Long_1.1235"

    def test_tolerates_missing_or_bad_entry(self):
        assert signal_store.default_dedupe_key({"pair": "X", "bias": "Long"}) == "X_Long_na"
        assert signal_store.default_dedupe_key(
            {"pair": "X", "bias": "Long", "entry": "oops"}) == "X_Long_na"
