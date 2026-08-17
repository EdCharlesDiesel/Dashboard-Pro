"""Unit tests for deploy/mt5_watchdog.py.

The failure this supervises is not death, it is wedging: on 2026-08-07 a sync
loop sat at 0 CPU for two and a half days, alive enough that "is pythonw
running?" answered yes the whole time. So the contract under test is that
liveness is judged by the sync log growing, never by a process existing - and
that a loop which is alive but merely failing (closed terminal) is left alone.

Nothing here may touch the real logs/ directory: a test that appended to
logs/mt5_sync.log would forge the production heartbeat, and one that killed a
lock holder would kill the desk's running loop.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_module():
    """Import deploy/mt5_watchdog.py by path — `deploy/` is not a package."""
    path = os.path.join(_REPO, "deploy", "mt5_watchdog.py")
    spec = importlib.util.spec_from_file_location("mt5_watchdog_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def wd(tmp_path):
    """The watchdog, with every path it can write pinned inside tmp_path."""
    module = _load_module()
    module.SYNC_LOG = str(tmp_path / "mt5_sync.log")
    module.SYNC_LOCK = str(tmp_path / "mt5_sync.lock")
    module.LOG = str(tmp_path / "mt5_watchdog.log")
    module.LOCK = str(tmp_path / "mt5_watchdog.lock")
    return module


@pytest.fixture
def state(wd):
    return {"backoff": wd.BACKOFF_START, "next_try": 0.0, "warned_at": 0.0,
            "terminal_restarted_at": 0.0}


class _FakeSync:
    """Stands in for the loaded mt5_sync module. `check` only needs it to be
    passed through to `restart`, which the restart tests stub out."""
    LOCK = "unused"
    _UNLOCKED = object()
    _LOCK_HANDLE = None

    def acquire_lock(self, *a, **k):
        return object()

    def release_lock(self):
        pass


@pytest.fixture
def fake_sync():
    return _FakeSync()


def _write_log(path, age_seconds, mtime_now):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("[2026-08-12 14:46:49 UTC] ok - balance=2971.88\n")
    stamp = mtime_now - age_seconds
    os.utime(path, (stamp, stamp))


class TestHeartbeat:
    def test_missing_log_reads_as_unknown(self, wd):
        assert wd.heartbeat_age() is None

    def test_age_tracks_the_logs_mtime(self, wd):
        import time
        now = time.time()
        _write_log(wd.SYNC_LOG, 400, now)
        assert 395 <= wd.heartbeat_age(now) <= 405

    def test_last_ok_age_ignores_failure_lines(self, wd):
        # A loop logging NOT SYNCED every cycle is alive but not syncing. The
        # heartbeat must stay fresh while the *ok* age keeps growing.
        with open(wd.SYNC_LOG, "w", encoding="utf-8") as fh:
            fh.write("[2026-08-12 10:00:00 UTC] ok - balance=1.0\n")
            fh.write("[2026-08-12 12:00:00 UTC] NOT SYNCED: terminal closed\n")
        import calendar, time
        now = calendar.timegm(time.strptime("2026-08-12 13:00:00", "%Y-%m-%d %H:%M:%S"))
        assert wd.last_ok_age(now) == pytest.approx(3 * 3600, abs=2)

    def test_last_ok_age_is_unknown_when_never_succeeded(self, wd):
        with open(wd.SYNC_LOG, "w", encoding="utf-8") as fh:
            fh.write("[2026-08-12 12:00:00 UTC] NOT SYNCED: terminal closed\n")
        assert wd.last_ok_age() is None


class TestRestartDecision:
    def test_fresh_log_is_left_alone(self, wd, fake_sync, state, monkeypatch):
        import time
        _write_log(wd.SYNC_LOG, 60, time.time())
        monkeypatch.setattr(wd, "restart", lambda *a: pytest.fail("restarted a healthy loop"))
        assert wd.check(fake_sync, state) is False

    def test_quiet_log_triggers_a_restart(self, wd, fake_sync, state, monkeypatch):
        import time
        _write_log(wd.SYNC_LOG, wd.STALE_AFTER + 60, time.time())
        calls = []
        monkeypatch.setattr(wd, "restart", lambda s, reason: calls.append(reason) or True)
        assert wd.check(fake_sync, state) is True
        assert "quiet" in calls[0]

    def test_missing_log_triggers_a_restart(self, wd, fake_sync, state, monkeypatch):
        calls = []
        monkeypatch.setattr(wd, "restart", lambda s, reason: calls.append(reason) or True)
        assert wd.check(fake_sync, state) is True

    def test_alive_but_not_syncing_warns_instead_of_restarting(
            self, wd, fake_sync, state, monkeypatch):
        # The closed-terminal case. Restarting fixes nothing here, and doing it
        # every cycle would be a restart storm against a healthy process.
        import time
        now = time.time()
        with open(wd.SYNC_LOG, "w", encoding="utf-8") as fh:
            fh.write("[2026-08-12 00:00:00 UTC] ok - balance=1.0\n")
        os.utime(wd.SYNC_LOG, (now, now))
        monkeypatch.setattr(wd, "restart", lambda *a: pytest.fail("restarted a live loop"))
        said = []
        monkeypatch.setattr(wd, "emit", lambda line: said.append(line))
        assert wd.check(fake_sync, state) is False
        assert any("has not synced" in line for line in said)


class TestBackoff:
    def test_second_failure_in_a_row_backs_off(self, wd, fake_sync, state, monkeypatch):
        monkeypatch.setattr(wd, "restart", lambda *a: True)
        assert wd.check(fake_sync, state) is True
        first = state["backoff"]
        # Immediately again: still stale, but inside the backoff window.
        assert wd.check(fake_sync, state) is False
        assert first == wd.BACKOFF_START * 2

    def test_backoff_is_capped(self, wd, fake_sync, state, monkeypatch):
        monkeypatch.setattr(wd, "restart", lambda *a: True)
        state["backoff"] = wd.BACKOFF_MAX
        wd.check(fake_sync, state)
        assert state["backoff"] == wd.BACKOFF_MAX

    def test_recovery_resets_the_backoff(self, wd, fake_sync, state, monkeypatch):
        import time
        state["backoff"] = wd.BACKOFF_MAX
        _write_log(wd.SYNC_LOG, 30, time.time())
        wd.check(fake_sync, state)
        assert state["backoff"] == wd.BACKOFF_START


class TestRestartSafety:
    def test_will_not_launch_while_the_lock_is_still_held(self, wd, fake_sync, monkeypatch):
        # Launching into a held lock would hit the loop's own guard: the new
        # process logs one line and exits, leaving the wedge in place while the
        # watchdog reports success.
        monkeypatch.setattr(wd, "holder_pid", lambda: 4321)
        monkeypatch.setattr(wd, "kill", lambda pid: True)
        monkeypatch.setattr(wd, "lock_is_free", lambda s, **k: False)
        monkeypatch.setattr(wd, "start_loop", lambda: pytest.fail("launched into a held lock"))
        assert wd.restart(fake_sync, "test") is False

    def test_kills_the_holder_before_relaunching(self, wd, fake_sync, monkeypatch):
        killed, started = [], []
        monkeypatch.setattr(wd, "holder_pid", lambda: 4321)
        monkeypatch.setattr(wd, "kill", lambda pid: killed.append(pid) or True)
        monkeypatch.setattr(wd, "lock_is_free", lambda s, **k: True)
        monkeypatch.setattr(wd, "start_loop", lambda: started.append(1) or True)
        assert wd.restart(fake_sync, "test") is True
        assert killed == [4321] and started == [1]

    def test_no_holder_just_starts(self, wd, fake_sync, monkeypatch):
        monkeypatch.setattr(wd, "holder_pid", lambda: None)
        monkeypatch.setattr(wd, "kill", lambda pid: pytest.fail("killed nothing"))
        monkeypatch.setattr(wd, "lock_is_free", lambda s, **k: True)
        monkeypatch.setattr(wd, "start_loop", lambda: True)
        assert wd.restart(fake_sync, "test") is True

    def test_holder_pid_is_readable_while_the_lock_is_held(self, wd):
        # Not incidental: mt5_sync.acquire_lock claims byte 4096 precisely so
        # the PID at byte 0 stays readable by this.
        sync_path = os.path.join(_REPO, "deploy", "mt5_sync.py")
        spec = importlib.util.spec_from_file_location("mt5_sync_for_pid", sync_path)
        sync = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sync)
        sync.LOCK = wd.SYNC_LOCK
        sync._LOCK_HANDLE = sync.acquire_lock()
        try:
            assert wd.holder_pid() == os.getpid()
        finally:
            sync.release_lock()


class TestFailedStreak:
    def test_counts_only_back_to_back_failures(self, wd):
        with open(wd.SYNC_LOG, "w", encoding="utf-8") as fh:
            fh.write("[2026-08-18 05:00:00 UTC] NOT SYNCED: IPC timeout (-10005)\n")
            fh.write("[2026-08-18 05:05:00 UTC] ok - balance=1.0\n")
            fh.write("[2026-08-18 05:10:00 UTC] NOT SYNCED: IPC timeout (-10005)\n")
            fh.write("[2026-08-18 05:15:00 UTC] NOT SYNCED: IPC timeout (-10005)\n")
        assert wd.failed_streak() == 2          # the pre-success one does not count

    def test_loop_banners_neither_break_nor_pad_the_streak(self, wd):
        # A restart writes "[loop] syncing every 300s" between cycles. That is
        # not a cycle: counting it would fake a failure, and breaking on it
        # would reset a streak that is genuinely continuing.
        with open(wd.SYNC_LOG, "w", encoding="utf-8") as fh:
            fh.write("[2026-08-18 05:00:00 UTC] NOT SYNCED: IPC timeout (-10005)\n")
            fh.write("[loop] syncing every 300s - session-resident\n")
            fh.write("[2026-08-18 05:05:00 UTC] NOT SYNCED: IPC timeout (-10005)\n")
        assert wd.failed_streak() == 2

    def test_no_log_is_no_streak(self, wd):
        assert wd.failed_streak() == 0


class TestTerminalRestart:
    def _failing(self, wd, count):
        """A live loop whose last `count` cycles all failed."""
        import time
        with open(wd.SYNC_LOG, "w", encoding="utf-8") as fh:
            fh.write("[2026-08-18 04:00:00 UTC] ok - balance=1.0\n")
            for _ in range(count):
                fh.write("[2026-08-18 05:00:00 UTC] NOT SYNCED: IPC timeout (-10005)\n")
        now = time.time()
        os.utime(wd.SYNC_LOG, (now, now))       # heartbeat fresh: loop is alive

    def test_hung_terminal_is_bounced(self, wd, fake_sync, state, monkeypatch):
        self._failing(wd, wd.TERMINAL_RESTART_AFTER_FAILS)
        bounced = []
        monkeypatch.setattr(wd, "restart_terminal", lambda: bounced.append(1) or True)
        monkeypatch.setattr(wd, "restart", lambda *a: pytest.fail("bounced the loop"))
        assert wd.check(fake_sync, state) is False
        assert bounced == [1]

    def test_short_streak_is_left_alone(self, wd, fake_sync, state, monkeypatch):
        # A blip must never bounce a healthy terminal.
        self._failing(wd, wd.TERMINAL_RESTART_AFTER_FAILS - 1)
        monkeypatch.setattr(wd, "restart_terminal", lambda: pytest.fail("bounced on a blip"))
        assert wd.check(fake_sync, state) is False

    def test_cooldown_blocks_a_second_bounce(self, wd, fake_sync, state, monkeypatch):
        self._failing(wd, wd.TERMINAL_RESTART_AFTER_FAILS + 3)
        calls = []
        monkeypatch.setattr(wd, "restart_terminal", lambda: calls.append(1) or True)
        assert wd.check(fake_sync, state) is False
        assert wd.check(fake_sync, state) is False      # immediately again
        assert calls == [1], "restarted the terminal twice inside the cooldown"

    def test_a_dead_loop_still_takes_priority(self, wd, fake_sync, state, monkeypatch):
        # Failures plus a quiet log means the loop itself is gone. Restart the
        # loop; bouncing the terminal would not bring the loop back.
        with open(wd.SYNC_LOG, "w", encoding="utf-8") as fh:
            for _ in range(wd.TERMINAL_RESTART_AFTER_FAILS + 2):
                fh.write("[2026-08-18 05:00:00 UTC] NOT SYNCED: IPC timeout (-10005)\n")
        import time
        old = time.time() - (wd.STALE_AFTER + 60)
        os.utime(wd.SYNC_LOG, (old, old))
        monkeypatch.setattr(wd, "restart_terminal", lambda: pytest.fail("bounced a dead loop's terminal"))
        monkeypatch.setattr(wd, "restart", lambda s, reason: True)
        assert wd.check(fake_sync, state) is True

    def test_missing_executable_is_reported_not_raised(self, wd, monkeypatch):
        monkeypatch.setattr(wd, "TERMINAL_EXE", r"C:\no\such\terminal64.exe")
        said = []
        monkeypatch.setattr(wd, "emit", lambda line: said.append(line))
        assert wd.restart_terminal() is False
        assert any("not found" in line for line in said)


class TestLaunchIsDetached:
    """The loop must not end up a child of the watchdog.

    On 2026-08-18, restarting the watchdog to load new code killed the sync
    loop with it: the loop was a direct child, so `taskkill /T` walked into it.
    Bouncing the supervisor must never cause the outage it exists to prevent.
    """

    def test_windows_launch_goes_through_start_so_the_loop_is_orphaned(self, wd, monkeypatch):
        monkeypatch.setattr(wd.os, "name", "nt")
        argv = wd.launch_argv()
        assert argv[:3] == ["cmd", "/c", "start"]
        assert argv[3] == "", "start would steal the executable path as a window title"
        assert wd.SYNC_PY in argv and "--loop" in argv

    def test_posix_launch_is_direct(self, wd, monkeypatch):
        monkeypatch.setattr(wd.os, "name", "posix")
        assert wd.launch_argv()[0] == wd.PYTHONW

    def test_interval_is_carried_through(self, wd, monkeypatch):
        monkeypatch.setattr(wd.os, "name", "posix")
        argv = wd.launch_argv()
        assert argv[argv.index("--loop") + 1] == str(wd.LOOP_INTERVAL)


class TestIsolation:
    def test_loading_the_watchdog_does_not_pull_in_the_broker(self):
        """The real contract, checked in a subprocess.

        `sys.modules` is process-global, so an in-process assertion passes or
        fails on whether some *other* test in the suite imported MetaTrader5
        first — nothing to do with the watchdog. This test asserted exactly
        that and duly passed alone while failing the full suite. A subprocess
        is the only honest way to ask the question, and it also covers the
        transitive case: `load_sync()` is the one call that could drag the
        broker in behind our back.
        """
        import subprocess
        import sys
        import textwrap

        probe = textwrap.dedent("""
            import importlib.util, sys
            spec = importlib.util.spec_from_file_location("probe", sys.argv[1])
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            m.load_sync()          # the one call that could import the broker
            print("LEAKED" if "MetaTrader5" in sys.modules else "CLEAN")
        """)
        result = subprocess.run(
            [sys.executable, "-c", probe,
             os.path.join(_REPO, "deploy", "mt5_watchdog.py")],
            capture_output=True, text=True, timeout=60)
        assert "CLEAN" in result.stdout, result.stdout + result.stderr

    def test_watchdog_declares_no_broker_import(self, wd):
        # The whole point: a supervisor that can wedge inside mt5.initialize()
        # the way its patient does is not a supervisor.
        #
        # Parsed, not grepped: the module *docstring* names mt5_link while
        # explaining why it does not import it, so a text search reports a
        # violation that is really the documentation of the rule.
        import ast

        source = open(os.path.join(_REPO, "deploy", "mt5_watchdog.py"),
                      encoding="utf-8").read()
        imported = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
                imported.update("{0}.{1}".format(node.module, a.name)
                                for a in node.names)

        banned = [name for name in imported
                  if "MetaTrader5" in name or "mt5_link" in name]
        assert not banned, "watchdog imports the broker: {0}".format(banned)
