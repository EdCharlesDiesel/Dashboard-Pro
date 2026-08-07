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

import os
import sys
import time
from datetime import datetime, timezone

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

LOG = os.path.join(_REPO, "logs", "mt5_sync.log")

# Set once the loop starts; single runs just print.
_LOGGING = False


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
        return sync_once()

    _LOGGING = True
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
