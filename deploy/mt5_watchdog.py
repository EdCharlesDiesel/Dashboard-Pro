"""Keep the MT5 sync loop alive, in the logged-on desktop session.

Run as a session-resident loop from the user's Startup folder, via
``deploy/mt5_watchdog_startup.cmd``, alongside the sync loop itself.

**Why a watchdog process and not Task Scheduler.** Same reason the sync loop is
a startup item: `mt5.initialize()` needs the interactive desktop session. The
registered task is Interactive *and still wedges* - on 2026-08-07 it hung on its
first run and Task Scheduler skipped 1428 consecutive runs behind it while
reporting ``Last Result: 0``. Do not re-enable it.

**Why this cannot wedge the way its patient does.** This module never imports
`MetaTrader5` or `mt5_link` - `mt5_sync.py` is loaded by path only for its lock
primitives, and its own broker imports are lazy, inside `sync_once()`. The
watchdog therefore never touches the call that hangs. A supervisor that can fail
the same way as the thing it supervises is not a supervisor.

**Why it watches the log and not the process list.** The dangerous failure here
is not death, it is wedging: on 2026-08-07 a loop sat at 0 CPU for two and a
half days, alive enough that "is pythonw running?" answered yes the whole time.
A liveness check that only looks for the process would have been fooled for two
and a half days. `sync_once()` appends exactly one line per cycle whether the
sync succeeds or fails, so *growth of the log* is the honest heartbeat - and a
wedged loop, hung inside `mt5.initialize()`, writes nothing at all.

**Why a stalled sync is not a restart.** A closed MetaTrader terminal makes the
loop log ``NOT SYNCED`` every cycle - the loop is perfectly healthy and
restarting it fixes nothing, so recent lines of *any* kind count as alive. That
case gets a warning in this log instead, and the exponential backoff means even
a misjudgement cannot become a restart storm.

**Why it writes its own log.** Appending to ``logs/mt5_sync.log`` would forge
the very heartbeat this reads, so the watchdog could keep itself convinced a
dead loop was alive. It writes ``logs/mt5_watchdog.log``.

Usage::

    python  deploy/mt5_watchdog.py --once      # one check, prints to stdout
    pythonw deploy/mt5_watchdog.py --loop 120  # forever, writes its own log
"""
from __future__ import annotations

import atexit
import importlib.util
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

SYNC_LOG = os.path.join(_REPO, "logs", "mt5_sync.log")
SYNC_LOCK = os.path.join(_REPO, "logs", "mt5_sync.lock")
SYNC_PY = os.path.join(_REPO, "deploy", "mt5_sync.py")
PYTHONW = os.path.join(_REPO, ".venv", "Scripts", "pythonw.exe")

LOG = os.path.join(_REPO, "logs", "mt5_watchdog.log")
LOCK = os.path.join(_REPO, "logs", "mt5_watchdog.lock")

# The interval the supervised loop runs at, and how long its log may go quiet
# before we call it dead. Three missed cycles: one slow broker read must never
# look like a wedge, or the watchdog becomes the outage.
LOOP_INTERVAL = 300
STALE_AFTER = 900
CHECK_EVERY = 120

# A restart that does not take (terminal missing, venv broken) must not be
# retried every two minutes forever.
BACKOFF_START = 300
BACKOFF_MAX = 3600

# How long a successful sync may be absent before we say so. Not a *loop*
# restart trigger - see the module docstring.
WARN_NO_SYNC_AFTER = 1800

# The terminal itself, for the one failure the loop cannot recover from.
#
# `mt5_link` re-initializes on every cycle, so a *closed* terminal needs no
# help - the next cycle reconnects the moment it reopens. A *hung* one is
# different: on 2026-08-18 it answered `IPC timeout (-10005)` for 33 straight
# cycles over 3h16m, and no amount of reconnecting from this side would have
# cleared it. Only bouncing the terminal does.
#
# Five consecutive failures (~25 min at a 300s loop) rather than one, because a
# broker-side blip or a weekend gap must never bounce a healthy terminal.
TERMINAL_EXE = os.environ.get(
    "MT5_TERMINAL_EXE", r"C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe")
TERMINAL_RESTART_AFTER_FAILS = 5
TERMINAL_RESTART_COOLDOWN = 3600

_LOGGING = False
_LOCK_HANDLE = None


def emit(line: str) -> None:
    """Print, and in loop mode also append to this module's own log.

    Startup launches us under ``pythonw.exe``, which has no stdout, so anything
    merely printed would vanish - the same reason `mt5_sync.py` owns its log.
    """
    try:
        print(line, flush=True)
    except Exception:                             # noqa: BLE001
        pass
    if not _LOGGING:
        return
    try:
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:                             # noqa: BLE001
        pass


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def load_sync():
    """Import ``deploy/mt5_sync.py`` by path - ``deploy/`` is not a package.

    Safe despite what that module supervises: its `MetaTrader5` imports live
    inside `sync_once()`, so importing it here pulls in nothing but stdlib.
    """
    spec = importlib.util.spec_from_file_location("mt5_sync_supervised", SYNC_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── liveness ───────────────────────────────────────────────────────────────────
def heartbeat_age(now=None):
    """Seconds since the sync log last grew, or ``None`` if it does not exist.

    ``None`` is not an error: a machine that has never synced needs the loop
    started just as much as one whose loop died.
    """
    try:
        mtime = os.path.getmtime(SYNC_LOG)
    except OSError:
        return None
    return max(0.0, (now if now is not None else time.time()) - mtime)


def _tail_text(window: int = 16384) -> str:
    """The end of the sync log. Read the tail rather than the whole file: it is
    append-only, already ~90KB, and consulted every cycle forever."""
    try:
        with open(SYNC_LOG, "rb") as fh:
            try:
                fh.seek(-window, os.SEEK_END)
            except OSError:
                fh.seek(0)                        # file shorter than the window
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return ""


def failed_streak() -> int:
    """How many sync cycles have failed back-to-back since the last success.

    Only ``ok``/``NOT SYNCED`` lines count - the loop's own ``[loop] ...``
    banners are not cycles and must not break or pad the streak.
    """
    streak = 0
    for line in reversed(_tail_text().splitlines()):
        if "] ok - " in line:
            break
        if "NOT SYNCED" in line:
            streak += 1
    return streak


def last_ok_age(now=None):
    """Seconds since the last *successful* sync line, or ``None`` if none found.

    Diagnostic only.
    """
    tail = _tail_text()
    if not tail:
        return None

    for line in reversed(tail.splitlines()):
        if "] ok - " not in line or not line.startswith("["):
            continue
        try:
            when = datetime.strptime(line[1:line.index("]")],
                                     "%Y-%m-%d %H:%M:%S UTC")
        except (ValueError, IndexError):
            continue
        when = when.replace(tzinfo=timezone.utc)
        ref = datetime.fromtimestamp(now, timezone.utc) if now is not None \
            else datetime.now(timezone.utc)
        return max(0.0, (ref - when).total_seconds())
    return None


def holder_pid():
    """PID recorded in the sync lock file, or ``None``.

    Readable even while the lock is held: `mt5_sync.acquire_lock` deliberately
    claims byte 4096, past the PID text, precisely so this stays legible.

    That intent needs help to survive, though - hence the unbuffered 64-byte
    read. A plain ``open(...).readline()`` asks for an 8192-byte buffer fill,
    which runs straight into the locked byte and fails the whole read with
    "another process has locked a portion of the file". `Get-Content` on this
    file fails for exactly that reason, so the PID was in practice unreadable
    while held - the case it exists for.
    """
    try:
        with open(SYNC_LOCK, "rb", buffering=0) as fh:
            head = fh.read(64)
    except OSError:
        return None
    try:
        return int(head.split(b"\n", 1)[0].strip())
    except ValueError:
        return None


def kill(pid: int) -> bool:
    """Stop a wedged loop. ``/T`` because the venv's `pythonw.exe` is a stub that
    launches the real interpreter as a child, so the tree is what must go."""
    try:
        if os.name == "nt":
            done = subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True, text=True, timeout=30)
            return done.returncode == 0
        os.kill(pid, 9)
        return True
    except Exception as exc:                      # noqa: BLE001
        emit("[{0}] kill {1} failed: {2}".format(_stamp(), pid, exc))
        return False


def lock_is_free(sync, timeout: float = 15.0) -> bool:
    """Whether the sync lock can be taken, polling up to ``timeout``.

    The authoritative "is it really gone?" check. A killed process releases its
    OS lock, so taking the lock proves the old loop is dead in a way that
    ``taskkill`` reporting success does not.
    """
    original = sync.LOCK
    sync.LOCK = SYNC_LOCK
    try:
        deadline = time.time() + timeout
        while True:
            handle = sync.acquire_lock()
            if handle is not None:
                if handle is not sync._UNLOCKED:
                    sync._LOCK_HANDLE = handle
                    sync.release_lock()
                return True
            if time.time() >= deadline:
                return False
            time.sleep(1.0)
    finally:
        sync.LOCK = original


def launch_argv():
    """Argv that starts the loop *without* leaving it a child of this process.

    On Windows the loop goes through ``cmd /c start``, which exits immediately
    and leaves the loop orphaned. ``DETACHED_PROCESS`` alone is not enough:
    it detaches the console but the parent link survives, so any
    ``taskkill /T`` aimed at the watchdog walks straight down into the sync.

    Learned the hard way on 2026-08-18 — restarting the watchdog to pick up new
    code killed the loop it was supervising, so bouncing the supervisor caused
    exactly the outage the supervisor exists to prevent.
    """
    tail = [PYTHONW, SYNC_PY, "--loop", str(LOOP_INTERVAL)]
    if os.name == "nt":
        # The empty "" is the window title `start` would otherwise steal from
        # the first quoted argument - without it, PYTHONW becomes the title and
        # nothing launches.
        return ["cmd", "/c", "start", "", "/B"] + tail
    return tail


def start_loop() -> bool:
    """Launch a detached sync loop from the repo root."""
    if not os.path.exists(PYTHONW):
        emit("[{0}] cannot start: {1} missing".format(_stamp(), PYTHONW))
        return False
    flags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
    try:
        subprocess.Popen(
            launch_argv(), cwd=_REPO, creationflags=flags, close_fds=True,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
        return True
    except Exception as exc:                      # noqa: BLE001
        emit("[{0}] launch failed: {1}".format(_stamp(), exc))
        return False


def restart_terminal() -> bool:
    """Bounce the MetaTrader terminal itself.

    The heavier hammer, and deliberately rate-limited: this closes the app the
    desk trades through. It is only reached when the terminal has refused a
    read for `TERMINAL_RESTART_AFTER_FAILS` consecutive cycles, which means it
    is already useless to us.

    The relaunched terminal reconnects on its saved credentials; if it is set
    to prompt instead, this gets it as far as the login dialog and the next
    cycles keep logging NOT SYNCED, which is still louder than silence.
    """
    if not os.path.exists(TERMINAL_EXE):
        emit("[{0}] cannot restart terminal: {1} not found (set MT5_TERMINAL_EXE)"
             .format(_stamp(), TERMINAL_EXE))
        return False

    emit("[{0}] restarting MetaTrader: {1}".format(_stamp(), TERMINAL_EXE))
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/IM", os.path.basename(TERMINAL_EXE)],
                           capture_output=True, text=True, timeout=30)
        time.sleep(5)                             # let the process actually go
        flags = 0x00000008 if os.name == "nt" else 0   # DETACHED_PROCESS
        subprocess.Popen([TERMINAL_EXE], cwd=os.path.dirname(TERMINAL_EXE),
                         creationflags=flags, close_fds=True)
    except Exception as exc:                      # noqa: BLE001
        emit("[{0}] terminal restart failed: {1}".format(_stamp(), exc))
        return False
    emit("[{0}] terminal relaunched - next sync cycle will retry".format(_stamp()))
    return True


def restart(sync, reason: str) -> bool:
    """Kill whatever holds the lock, confirm it let go, and start a fresh loop."""
    emit("[{0}] RESTARTING: {1}".format(_stamp(), reason))

    pid = holder_pid()
    if pid is not None:
        emit("[{0}] killing lock holder pid {1}".format(_stamp(), pid))
        kill(pid)

    if not lock_is_free(sync):
        # Never launch into a held lock: the new loop would take the guard's
        # exit path, log one line and quit, leaving the wedge in place while
        # this reported a successful restart.
        emit("[{0}] lock still held - not starting a second loop".format(_stamp()))
        return False

    if not start_loop():
        return False
    emit("[{0}] started deploy/mt5_sync.py --loop {1}".format(_stamp(), LOOP_INTERVAL))
    return True


def check(sync, state: dict) -> bool:
    """One health check. Returns True if a restart was performed."""
    now = time.time()
    age = heartbeat_age(now)

    if age is not None and age <= STALE_AFTER:
        if state.get("backoff", BACKOFF_START) != BACKOFF_START:
            emit("[{0}] healthy again - backoff reset".format(_stamp()))
        state["backoff"] = BACKOFF_START

        # Alive but not actually syncing. Restarting the *loop* is still wrong
        # here - it is doing its job - but the terminal it reads may be hung,
        # and that the loop cannot fix from its side.
        streak = failed_streak()
        if streak >= TERMINAL_RESTART_AFTER_FAILS:
            since = now - state.get("terminal_restarted_at", 0)
            if since > TERMINAL_RESTART_COOLDOWN:
                emit("[{0}] {1} consecutive failed cycles - terminal looks hung"
                     .format(_stamp(), streak))
                state["terminal_restarted_at"] = now
                restart_terminal()
            elif now - state.get("warned_at", 0) > WARN_NO_SYNC_AFTER:
                state["warned_at"] = now
                emit("[{0}] still failing ({1} cycles) but terminal was bounced "
                     "{2:.0f} min ago - waiting out the cooldown"
                     .format(_stamp(), streak, since / 60))
            return False

        ok_age = last_ok_age(now)
        if ok_age is not None and ok_age > WARN_NO_SYNC_AFTER \
                and now - state.get("warned_at", 0) > WARN_NO_SYNC_AFTER:
            state["warned_at"] = now
            emit("[{0}] loop is alive but has not synced for {1:.0f} min - "
                 "is the MetaTrader terminal open?".format(_stamp(), ok_age / 60))
        return False

    reason = ("no sync log yet" if age is None
              else "sync log quiet for {0:.0f}s (limit {1}s)".format(age, STALE_AFTER))

    if now < state.get("next_try", 0):
        emit("[{0}] {1} - backing off, next attempt in {2:.0f}s".format(
            _stamp(), reason, state["next_try"] - now))
        return False

    ok = restart(sync, reason)
    backoff = state.get("backoff", BACKOFF_START)
    state["next_try"] = now + backoff
    state["backoff"] = min(backoff * 2, BACKOFF_MAX)
    return ok


def main(argv=None) -> int:
    global _LOGGING, _LOCK_HANDLE
    argv = sys.argv[1:] if argv is None else argv

    sync = load_sync()
    state = {"backoff": BACKOFF_START, "next_try": 0.0, "warned_at": 0.0,
             "terminal_restarted_at": 0.0}

    if "--once" in argv or "--loop" not in argv:
        check(sync, state)
        return 0

    i = argv.index("--loop")
    interval = int(argv[i + 1]) if len(argv) > i + 1 else CHECK_EVERY

    _LOGGING = True

    # Single-instance for the same reason the loop is: two watchdogs would race
    # to kill and relaunch the same patient. Reuses the loop's lock primitives
    # against a different file.
    original = sync.LOCK
    sync.LOCK = LOCK
    _LOCK_HANDLE = sync.acquire_lock()
    sync.LOCK = original
    if _LOCK_HANDLE is None:
        emit("[watchdog] another watchdog already holds {0} - exiting. "
             "This is the guard working, not an error.".format(LOCK))
        return 0
    sync._LOCK_HANDLE = _LOCK_HANDLE
    atexit.register(sync.release_lock)

    emit("[watchdog] checking every {0}s - restarts the sync loop if its log "
         "goes quiet for {1}s".format(interval, STALE_AFTER))
    while True:
        try:
            check(sync, state)
        except Exception as exc:                  # noqa: BLE001 - must outlive a bad check
            emit("[{0}] check failed: {1}".format(_stamp(), exc))
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
