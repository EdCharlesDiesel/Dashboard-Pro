"""Scheduled entry point: import closed MT5 trades into ``trade_setups``.

Runs on the Windows host, not in a container — the MetaTrader5 package is
Windows-only and talks to the session's running terminal, which is also why
this cannot live alongside the compose services.

The default 10-day window against a weekly cadence is deliberate overlap: a
run that is skipped (machine asleep, terminal closed) would otherwise leave a
permanent hole in the record, and the importer dedupes on position id, so
re-reading days already stored costs nothing but a few queries.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.mt5_trade_import import import_closed_trades  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=10,
                    help="How far back to look for closed trades (default 10).")
    ap.add_argument("--broker-utc-offset", type=float, default=None,
                    help="Broker timezone, hours east of UTC. Defaults to "
                         "$MT5_BROKER_UTC_OFFSET, else 0. Find it with "
                         "--probe-offset while the market is open.")
    ap.add_argument("--probe-offset", action="store_true",
                    help="Measure the broker clock against UTC and exit. Only "
                         "meaningful during trading hours.")
    args = ap.parse_args()

    if args.probe_offset:
        from src.services.mt5_link import detect_server_utc_offset
        probe = detect_server_utc_offset()
        print("{0} :: {1}".format("OK" if probe["ok"] else "UNAVAILABLE",
                                  probe["message"]))
        return 0 if probe["ok"] else 1

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    result = import_closed_trades(days=args.days,
                                  utc_offset=args.broker_utc_offset)
    print("[{0}] seen={1} imported={2} ok={3} :: {4}".format(
        stamp, result.get("seen"), result.get("imported"), result.get("ok"),
        result.get("message")))
    if result.get("skipped"):
        print("    unmapped symbols: {0}".format(result["skipped"]))

    # Trades deliberately held back are printed even when zero-valued would be
    # noise: a partially-closed position is a *pending* journal entry, and the
    # week's stored P/L will not match the broker until it closes in full.
    held = {k: v for k, v in (result.get("held") or {}).items() if v}
    if held:
        print("    held back: {0}".format(held))

    # Non-zero on failure so Task Scheduler's Last Result column is meaningful
    # rather than a permanent 0 that hides a broker link that has been down
    # for weeks.
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
