"""Unit tests for src/services/signal_store.py — the shared signal persistence
service (dedupe, source tagging, graceful no-op). No live DB or Streamlit."""
from __future__ import annotations

from datetime import date, datetime, timezone

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

    def test_a_price_move_no_longer_saves_the_same_view_twice(self, wired, signal):
        # Inverted 2026-08-04. This used to assert the opposite — that a new
        # rounded price level saves again — which is what produced up to 2.9 rows
        # per unchanged pair+direction view per day and made per-source
        # expectancy a function of sweep frequency. Price moving is not new
        # information about a read that hasn't changed.
        assert signal_store.persist_signals("setup_ranker", [signal]) == 1
        moved = {**signal, "entry": 1.12000}
        assert signal_store.persist_signals("setup_ranker", [moved]) == 0
        assert len(wired.saved) == 1

    def test_a_new_bar_saves_again(self, wired, signal):
        # The flip side: genuinely new information must still get through.
        assert signal_store.persist_signals(
            "setup_ranker", [{**signal, "bar_time": "2026-07-27"}]) == 1
        assert signal_store.persist_signals(
            "setup_ranker", [{**signal, "bar_time": "2026-08-03"}]) == 1
        assert len(wired.saved) == 2

    def test_a_bias_flip_saves_again_same_day(self, wired, signal):
        assert signal_store.persist_signals("setup_ranker", [signal]) == 1
        flipped = {**signal, "bias": "Short"}
        assert signal_store.persist_signals("setup_ranker", [flipped]) == 1
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
    """Keyed on the period, not the price (changed 2026-08-04).

    Keying on entry rounded to 4dp meant ~1 pip on a 5-decimal FX pair, so an
    unchanged read produced a fresh row on every tick. Measured: two sweeps in a
    day gave up to 2.9 rows per distinct pair+direction view, scaling with sweep
    frequency — which corrupts the Source Scorecard, since it ranks sources by
    expectancy over stored rows.
    """

    def test_price_no_longer_splits_an_unchanged_view(self):
        a = signal_store.default_dedupe_key(
            {"pair": "EUR/USD", "bias": "Long", "entry": 1.123456})
        b = signal_store.default_dedupe_key(
            {"pair": "EUR/USD", "bias": "Long", "entry": 1.987654})
        assert a == b, "a price move must not manufacture a second signal"

    def test_key_is_pair_bias_and_fx_day(self):
        # Compare against the FX session day, not the UTC calendar day — this
        # test would otherwise fail whenever it ran after 21:00 UTC.
        day = signal_store._fx_session_day(datetime.now(timezone.utc))
        assert signal_store.default_dedupe_key(
            {"pair": "EUR/USD", "bias": "Long"}) == "EUR/USD_Long_" + day

    def test_a_bias_flip_is_still_a_new_signal(self):
        long_k = signal_store.default_dedupe_key({"pair": "EUR/USD", "bias": "Long"})
        short_k = signal_store.default_dedupe_key({"pair": "EUR/USD", "bias": "Short"})
        assert long_k != short_k

    def test_bar_time_overrides_the_day_so_weekly_reads_dont_refire(self):
        # A weekly read swept daily should stay one row until a new weekly bar.
        key = signal_store.default_dedupe_key(
            {"pair": "EUR/USD", "bias": "Long", "bar_time": datetime(2026, 7, 27)})
        assert key == "EUR/USD_Long_2026-07-27"

    @pytest.mark.parametrize("stamp,expected", [
        (date(2026, 7, 27), "2026-07-27"),
        (datetime(2026, 7, 27, 13, 45), "2026-07-27"),
        ("2026-07-27T13:45:00+00:00", "2026-07-27"),
        ("2026-07-27", "2026-07-27"),
    ])
    def test_bar_time_accepts_datetime_date_and_iso_string(self, stamp, expected):
        assert signal_store.default_dedupe_key(
            {"pair": "X", "bias": "Long", "bar_time": stamp}) == "X_Long_" + expected

    def test_unusable_bar_time_falls_back_to_the_current_fx_day(self):
        today = signal_store._fx_session_day(datetime.now(timezone.utc))
        for bad in (None, "", "oops", 12345):
            assert signal_store.default_dedupe_key(
                {"pair": "X", "bias": "Long", "bar_time": bad}) == "X_Long_" + today


class TestFxSessionDayFallback:
    """The no-bar fallback keys on the FX trading day, not the calendar day.

    The FX day rolls at 21:00 UTC (FX_DAY_BREAK_HOUR, the same break
    volume_profile anchors sessions to). Keying on the calendar day would put
    21:00-24:00 UTC in the previous trading day — and a 23:15 SAST cron is
    21:15 UTC, squarely inside that window.
    """

    def _key_at(self, monkeypatch, moment):
        class _Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                return moment if tz else moment.replace(tzinfo=None)
        monkeypatch.setattr(signal_store, "datetime", _Clock)
        return signal_store.default_dedupe_key({"pair": "EUR/USD", "bias": "Long"})

    def test_before_the_break_is_the_same_day(self, monkeypatch):
        moment = datetime(2026, 8, 4, 20, 59, tzinfo=timezone.utc)
        assert self._key_at(monkeypatch, moment).endswith("2026-08-04")

    def test_after_the_break_rolls_to_the_next_trading_day(self, monkeypatch):
        moment = datetime(2026, 8, 4, 21, 15, tzinfo=timezone.utc)  # 23:15 SAST cron
        assert self._key_at(monkeypatch, moment).endswith("2026-08-05")

    def test_two_sweeps_either_side_of_the_break_do_not_collide(self, monkeypatch):
        before = self._key_at(monkeypatch, datetime(2026, 8, 4, 20, 0, tzinfo=timezone.utc))
        after = self._key_at(monkeypatch, datetime(2026, 8, 4, 22, 0, tzinfo=timezone.utc))
        assert before != after, "a new FX day must not reuse the old day's key"

    def test_an_explicit_bar_label_is_never_shifted(self, monkeypatch):
        # Only the clock fallback gets the session rule — a bar label already
        # identifies its bar, and shifting it would merge two distinct bars.
        monkeypatch.setattr(signal_store, "datetime", datetime)
        key = signal_store.default_dedupe_key(
            {"pair": "X", "bias": "Long", "bar_time": datetime(2026, 8, 4, 22, 0)})
        assert key == "X_Long_2026-08-04"
