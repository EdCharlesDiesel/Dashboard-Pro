"""Push the live MT5 terminal's balance and open positions into the database.

Run as a session-resident loop from the user's Startup folder, via
``deploy/mt5_sync_startup.cmd``.

**Why a startup item and not a scheduled task.** `mt5.initialize()` attaches to
the MetaTrader terminal, which requires the *interactive desktop session*. Task
Scheduler launches in a different window station, so the call never returns
cleanly: the task wedges in "Queued" forever while reporting ``Last Result: 0``,
which looks like success and produces nothing. A scheduled-task version is
registered but **disabled** for exactly this reason - do not re-enable it.

**Why this must exist at all.** The ``MetaTrader5`` package is Windows-only and
needs a terminal on the same machine, so the Linux containers physically cannot
read it - `signal_sweep.sync_broker_state()` no-ops there. Without this, the
stored balance goes stale the moment you trade and every page sizes against a
wrong number. Not hypothetical: a stale $5,182 reading against a real $1,989
produced position sizes 2.6x too large, and a $10,000 default is what the
container showed before `app_state` was migrated.

**Read-only against the broker.** `mt5_link` deliberately exposes no order
function, so this can never place, modify or close anything.

Writes go wherever ``.streamlit/secrets.toml`` points - currently 127.0.0.1:5433,
the Docker container's Postgres, the single system of record. Port 5432 is a
*different*, native PostgreSQL.

Usage::

    python  deploy/mt5_sync.py                # one sync, prints to stdout
    pythonw deploy/mt5_sync.py --loop 300     # forever, writes logs/mt5_sync.log
"""
from __future__ import annotations

import atexit
import os
import sys
import time
from datetime import datetime, timezone

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

LOG = os.path.join(_REPO, "logs", "mt5_sync.log")
LOCK = os.path.join(_REPO, "logs", "mt5_sync.lock")

# Set once the loop starts; single runs just print.
_LOGGING = False

# The live lock handle. Must stay referenced for the whole process: the OS lock
# is held by the open file object, so letting it be garbage-collected releases
# the lock silently while the loop keeps running.
_LOCK_HANDLE = None

# Returned when the lock file itself cannot be created (read-only disk, missing
# logs/). That is not evidence another loop is running, so the sync should still
# start — refusing to sync because we could not write a lock file would be a
# self-inflicted outage.
_UNLOCKED = object()

# Which byte the Windows lock claims. Deliberately past the PID text at the top
# of the file: a msvcrt byte-range lock blocks *other* handles from reading that
# byte, so locking byte 0 would make the PID unreadable — `type mt5_sync.lock`
# would fail — precisely while the loop is running and you want to know who
# holds it. POSIX flock is whole-file and advisory, so the offset is harmless.
_LOCK_BYTE = 4096


def acquire_lock(path=None):
    """Hold an exclusive OS lock, or return ``None`` if another loop has it.

    ``path`` resolves to :data:`LOCK` at call time, not as a default argument —
    a default would bind the module value once at import and silently ignore any
    later reassignment of ``LOCK``.

    **Why an OS lock rather than the previous ``tasklist`` check.** That check
    only *warned*, and it could not have worked anyway: on 2026-08-07 two loops
    launched in the same second, so each looked first and saw nothing. Both
    started. One won `mt5.initialize()` and did all 447 syncs; the other wedged
    on that call and sat at 0 CPU for two and a half days, alive enough that
    "is pythonw running?" answered yes while doing nothing at all.

    An OS lock has no race — the kernel arbitrates — and, unlike a PID file, it
    is released automatically when the process dies, so a crash cannot leave a
    stale lock that blocks every future start.
    """
    path = path or LOCK
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        handle = open(path, "a+")
    except Exception as exc:                      # noqa: BLE001
        emit("[lock] cannot create {0} ({1}) - running without a lock".format(path, exc))
        return _UNLOCKED

    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(_LOCK_BYTE)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:                                     # the Linux container path
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None                               # someone else holds it

    try:
        handle.seek(0)
        handle.truncate()
        handle.write("{0}\n".format(os.getpid()))
        handle.flush()
    except Exception:                             # noqa: BLE001 - diagnostics only
        pass
    return handle


def release_lock():
    """Drop the lock. The OS would do this at exit anyway; this makes it prompt."""
    global _LOCK_HANDLE
    handle = _LOCK_HANDLE
    _LOCK_HANDLE = None
    if handle is None or handle is _UNLOCKED:
        return
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(_LOCK_BYTE)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except Exception:                             # noqa: BLE001
        pass
    try:
        handle.close()
    except Exception:                             # noqa: BLE001
        pass


def emit(line: str) -> None:
    """Print, and in loop mode also append to the log.

    The Startup launcher uses ``pythonw.exe`` so no console window appears at
    login - and pythonw has **no stdout**, so anything merely printed would
    vanish. The loop has to own its log or it runs invisibly forever.
    """
    try:
        print(line, flush=True)
    except Exception:
        pass
    if not _LOGGING:
        return
    try:
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def sync_once() -> int:
    """One sync. Returns 0 on success, 1 if the terminal could not be read."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    try:
        from src.services import mt5_link
    except Exception as exc:                      # noqa: BLE001
        emit("[{0}] mt5_link unavailable: {1}".format(stamp, exc))
        return 1

    try:
        res = mt5_link.sync()
    except Exception as exc:                      # noqa: BLE001 - never traceback into a log
        emit("[{0}] sync raised: {1}".format(stamp, exc))
        return 1

    account = res.get("account") or {}
    if not res.get("ok"):
        emit("[{0}] NOT SYNCED: {1}".format(stamp, res.get("message")))
        return 1

    emit("[{0}] ok - balance={1} equity={2} positions={3} | {4}".format(
        stamp, account.get("balance"), account.get("equity"),
        res.get("saved"), res.get("message")))
    return 0


def main(argv=None) -> int:
    global _LOGGING
    argv = sys.argv[1:] if argv is None else argv

    interval = 0
    if "--loop" in argv:
        i = argv.index("--loop")
        interval = int(argv[i + 1]) if len(argv) > i + 1 else 300

    if not interval:
        # A one-off sync is deliberately NOT locked: forcing a manual refresh
        # while the loop runs costs one extra write and is exactly what you want
        # when debugging. Only the perpetual loop is single-instance.
        return sync_once()

    _LOGGING = True

    global _LOCK_HANDLE
    _LOCK_HANDLE = acquire_lock()
    if _LOCK_HANDLE is None:
        emit("[loop] another sync loop already holds {0} - exiting. "
             "This is the guard working, not an error.".format(LOCK))
        return 0
    atexit.register(release_lock)

    emit("[loop] syncing every {0}s - session-resident, terminal must be open"
         .format(interval))
    while True:
        try:
            sync_once()
        except Exception as exc:                  # noqa: BLE001 - must outlive a bad run
            emit("[loop] run failed: {0}".format(exc))
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
