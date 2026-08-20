"""Unit tests for the single-instance lock in deploy/mt5_sync.py.

The bug this prevents: on 2026-08-07 two sync loops launched in the same second.
The old `tasklist` pre-check could not catch it — each looked first and saw
nothing — so both started. One won `mt5.initialize()` and did every sync; the
other wedged on that call at 0 CPU for two and a half days, alive enough that
"is pythonw running?" answered yes while it did nothing at all.

These run on both platforms: the lock uses msvcrt on Windows and fcntl in the
Linux container, and the contract has to hold identically in both.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_module():
    """Import deploy/mt5_sync.py by path — `deploy/` is not a package."""
    path = os.path.join(_REPO, "deploy", "mt5_sync.py")
    spec = importlib.util.spec_from_file_location("mt5_sync_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sync(tmp_path):
    module = _load_module()
    module.LOCK = str(tmp_path / "mt5_sync.lock")
    # LOG too, not just LOCK: `main()` flips _LOGGING on, so a test that drives
    # the loop guard appends its "another sync loop already holds ..." line to
    # the *real* logs/mt5_sync.log. Three such lines landed in production output
    # before this was pinned — a log the desk reads to decide whether the sync
    # is healthy must not carry test noise.
    module.LOG = str(tmp_path / "mt5_sync.log")
    yield module
    module.release_lock()


class TestAcquireLock:
    def test_first_caller_gets_the_lock(self, sync):
        handle = sync.acquire_lock()
        assert handle is not None and handle is not sync._UNLOCKED

    def test_lock_file_records_the_holding_pid(self, sync):
        sync._LOCK_HANDLE = sync.acquire_lock()
        with open(sync.LOCK, "r") as fh:
            assert fh.read().strip() == str(os.getpid())

    def test_release_lets_the_next_caller_in(self, sync):
        sync._LOCK_HANDLE = sync.acquire_lock()
        assert sync._LOCK_HANDLE is not None
        sync.release_lock()
        second = sync.acquire_lock()
        assert second is not None and second is not sync._UNLOCKED
        sync._LOCK_HANDLE = second

    def test_unwritable_path_runs_unlocked_rather_than_refusing(self, sync, tmp_path):
        # A lock file we cannot create is not evidence of another loop. Refusing
        # to sync because of it would be an outage we inflicted on ourselves.
        sync.LOCK = str(tmp_path / "no-such-dir" / "x" / "\x00bad")
        assert sync.acquire_lock() is sync._UNLOCKED

    def test_release_is_safe_when_nothing_is_held(self, sync):
        sync._LOCK_HANDLE = None
        sync.release_lock()          # must not raise

    def test_release_is_safe_for_the_unlocked_sentinel(self, sync):
        sync._LOCK_HANDLE = sync._UNLOCKED
        sync.release_lock()          # must not raise


class TestSecondProcessIsRefused:
    """The real contract: a *separate process* must be turned away.

    An in-process second `acquire_lock` is not a valid test — on POSIX, flock is
    granted per open file description, and Windows byte-range locks are held per
    handle, so the same process can behave differently from a true second
    instance. Only a subprocess proves the guard.
    """

    def test_a_second_process_cannot_take_a_held_lock(self, sync):
        sync._LOCK_HANDLE = sync.acquire_lock()
        assert sync._LOCK_HANDLE is not None

        probe = textwrap.dedent("""
            import importlib.util, sys
            spec = importlib.util.spec_from_file_location("probe", sys.argv[1])
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            m.LOCK = sys.argv[2]
            got = m.acquire_lock()
            print("ACQUIRED" if (got is not None and got is not m._UNLOCKED) else "REFUSED")
        """)
        result = subprocess.run(
            [sys.executable, "-c", probe,
             os.path.join(_REPO, "deploy", "mt5_sync.py"), sync.LOCK],
            capture_output=True, text=True, timeout=60)
        assert "REFUSED" in result.stdout, result.stdout + result.stderr

    def test_the_lock_frees_when_the_holder_dies(self, sync):
        # No stale-lock problem: this is why an OS lock beats a PID file. A
        # crashed loop must not block every future start.
        probe = textwrap.dedent("""
            import importlib.util, sys
            spec = importlib.util.spec_from_file_location("probe", sys.argv[1])
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            m.LOCK = sys.argv[2]
            m.acquire_lock()
            # exit while still holding it - the OS must clean up
        """)
        subprocess.run(
            [sys.executable, "-c", probe,
             os.path.join(_REPO, "deploy", "mt5_sync.py"), sync.LOCK],
            capture_output=True, text=True, timeout=60)
        handle = sync.acquire_lock()
        assert handle is not None and handle is not sync._UNLOCKED
        sync._LOCK_HANDLE = handle


class TestLoopGuard:
    def test_loop_exits_cleanly_when_another_holds_the_lock(self, sync, monkeypatch):
        monkeypatch.setattr(sync, "acquire_lock", lambda *a, **k: None)
        slept = []
        monkeypatch.setattr(sync.time, "sleep", lambda s: slept.append(s))
        assert sync.main(["--loop", "300"]) == 0
        assert not slept            # exited before entering the loop

    def test_single_run_is_not_locked(self, sync, monkeypatch):
        # Forcing a manual sync while the loop runs must still work.
        called = {"n": 0, "lock": 0}
        monkeypatch.setattr(sync, "sync_once", lambda: called.__setitem__("n", 1) or 0)
        monkeypatch.setattr(sync, "acquire_lock",
                            lambda *a, **k: called.__setitem__("lock", 1))
        sync.main([])
        assert called["n"] == 1 and called["lock"] == 0


class TestClosedTradeImport:
    """The loop also keeps the journal current.

    Before this, closed deals reached the journal only when somebody ran
    deploy/mt5_trade_import.py by hand — on 2026-08-20 the journal's last
    trade was six days old.
    """

    @staticmethod
    def _ok_link(monkeypatch):
        """Make mt5_link.sync() succeed without touching a terminal."""
        import types
        fake = types.ModuleType("src.services.mt5_link")
        fake.sync = lambda: {"ok": True, "saved": 0, "message": "fake",
                             "account": {"balance": 1.0, "equity": 1.0}}
        import src.services
        monkeypatch.setattr(src.services, "mt5_link", fake, raising=False)
        monkeypatch.setitem(__import__("sys").modules,
                            "src.services.mt5_link", fake)

    def test_a_successful_sync_also_imports_closed_trades(self, sync, monkeypatch):
        calls = []
        # Signature matters: sync_once calls _import_closed(days=7). A stub
        # taking no arguments raises TypeError, which the production guard
        # swallows by design - so a mismatched stub reads as "never ran".
        monkeypatch.setattr(sync, "_import_closed",
                            lambda days=7: calls.append(days) or {"imported": 2})
        self._ok_link(monkeypatch)
        assert sync.sync_once() == 0
        assert calls == [7], "the journal import never ran"

    def test_a_failing_import_cannot_take_the_loop_down(self, sync, monkeypatch):
        # The book and the balance are the loop's job; the journal is a bonus.
        # A wrong position size is a trading loss, a stale journal is an
        # inconvenience - they must not share a failure path, or the watchdog
        # starts bouncing a healthy terminal over a database hiccup.
        def boom(days=7):
            raise RuntimeError("database unreachable")

        monkeypatch.setattr(sync, "_import_closed", boom)
        self._ok_link(monkeypatch)
        assert sync.sync_once() == 0, "a journal failure must not fail the sync"
