"""Dukascopy historical archive backfill — feeds the local debug mode's
``source="duka"`` replay/time-machine (see src/debug/).

Standalone CLI, not imported by the running app in prod. Pulls bars for every
registry instrument Dukascopy carries via the ``dukascopy-python`` package and
upserts into the ``dukascopy_bars`` table (src/db/market_data_repository.py),
so src/db/market_cache.py's ``source="duka"`` read path has something to read
once it's wired (see that module's ``_ohlc_impl``).

Usage:
    python -m src.data.dukascopy_backfill --start 2024-01-01 --end 2026-07-01
    python -m src.data.dukascopy_backfill --symbols EUR/USD XAU/USD --interval 1h
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger("ForexDashboard")

# Registry pair name -> dukascopy_python.instruments constant NAME (looked up
# via getattr so this module doesn't need dukascopy_python imported at load
# time — only when a backfill actually runs). Built by hand against the
# installed package (dukascopy-python==4.0.1, verified 2026-07) rather than
# pattern-matched: Dukascopy's naming isn't systematic (metals/energy use a
# ".CMD" suffix; WTI is named "E_Light" for "Light Crude", no "WTI"/"oil"
# string anywhere in it). GBP/ZAR has no Dukascopy coverage at all (checked
# empirically — only EUR/ZAR, USD/ZAR, ZAR/JPY exist) and is intentionally
# absent from this map; every other registry pair (21 of 22) is covered.
_INSTRUMENT_MAP: Dict[str, str] = {
    "EUR/USD": "INSTRUMENT_FX_MAJORS_EUR_USD",
    "GBP/USD": "INSTRUMENT_FX_MAJORS_GBP_USD",
    "AUD/USD": "INSTRUMENT_FX_MAJORS_AUD_USD",
    "NZD/USD": "INSTRUMENT_FX_MAJORS_NZD_USD",
    "USD/JPY": "INSTRUMENT_FX_MAJORS_USD_JPY",
    "USD/CHF": "INSTRUMENT_FX_MAJORS_USD_CHF",
    "USD/CAD": "INSTRUMENT_FX_MAJORS_USD_CAD",
    "EUR/GBP": "INSTRUMENT_FX_CROSSES_EUR_GBP",
    "EUR/JPY": "INSTRUMENT_FX_CROSSES_EUR_JPY",
    "GBP/JPY": "INSTRUMENT_FX_CROSSES_GBP_JPY",
    "AUD/JPY": "INSTRUMENT_FX_CROSSES_AUD_JPY",
    "EUR/AUD": "INSTRUMENT_FX_CROSSES_EUR_AUD",
    "GBP/AUD": "INSTRUMENT_FX_CROSSES_GBP_AUD",
    "EUR/CAD": "INSTRUMENT_FX_CROSSES_EUR_CAD",
    "GBP/CAD": "INSTRUMENT_FX_CROSSES_GBP_CAD",
    "USD/ZAR": "INSTRUMENT_FX_CROSSES_USD_ZAR",
    "EUR/ZAR": "INSTRUMENT_FX_CROSSES_EUR_ZAR",
    "XAU/USD": "INSTRUMENT_FX_METALS_XAU_USD",
    "XAG/USD": "INSTRUMENT_FX_METALS_XAG_USD",
    "XPT/USD": "INSTRUMENT_CMD_METALS_XPT_CMD_USD",
    "WTI/USD": "INSTRUMENT_CMD_ENERGY_E_LIGHT",  # "E_Light" = Light Crude = WTI
}

# App interval string -> dukascopy_python constant name. Matches the interval
# strings already used throughout market_bars/cached_ohlc.
_INTERVAL_MAP: Dict[str, str] = {
    "1d": "INTERVAL_DAY_1", "1h": "INTERVAL_HOUR_1", "4h": "INTERVAL_HOUR_4",
}


def _fetch_one(pair: str, interval: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    """One instrument's Dukascopy pull, mapped to the app's Open/High/Low/
    Close/Volume convention (tz-naive index, matching market_bars). Returns
    ``None`` (not an exception) when the pair has no Dukascopy mapping — the
    caller treats that as a skip, not a failure."""
    import dukascopy_python
    from dukascopy_python import instruments as di

    const_name = _INSTRUMENT_MAP.get(pair)
    if const_name is None:
        logger.warning("[dukascopy_backfill] %s has no Dukascopy mapping — skipped", pair)
        return None
    instrument = getattr(di, const_name)
    interval_const = getattr(dukascopy_python, _INTERVAL_MAP[interval])

    df = dukascopy_python.fetch(instrument, interval_const, dukascopy_python.OFFER_SIDE_BID, start, end)
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume",
    })
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df[["Open", "High", "Low", "Close", "Volume"]]


def backfill(pairs: List[str], interval: str, start: datetime, end: datetime, repo) -> Dict[str, Optional[int]]:
    """Fetch + upsert every pair in ``pairs`` via ``repo`` (a
    ``MarketDataRepository``). Returns ``{pair: rows_written}``, with
    ``None`` for any pair skipped (no mapping or a fetch error) — per-pair
    failures never abort the run.

    Archived under the pair's **yfinance ticker** (``INSTRUMENTS[pair].ticker``,
    e.g. "EUR/USD" -> "EURUSD=X"), not the registry pair name — every
    ``cached_ohlc``/``_ohlc_impl`` call site passes the ticker as ``symbol``
    (never the pair name), so storing under anything else would make the
    ``source="duka"`` read path miss on every lookup.
    """
    from src.instruments.registry import INSTRUMENTS

    results: Dict[str, Optional[int]] = {}
    for pair in pairs:
        ticker = INSTRUMENTS[pair]["ticker"] if pair in INSTRUMENTS else pair
        try:
            df = _fetch_one(pair, interval, start, end)
        except Exception as exc:
            logger.warning("[dukascopy_backfill] %s failed: %s — skipped", pair, exc)
            results[pair] = None
            continue
        if df is None:
            results[pair] = None
            continue
        n = repo.upsert_dukascopy_bars(ticker, interval, df)
        results[pair] = n
        logger.info("[dukascopy_backfill] %s (%s) %s: %d bars", pair, ticker, interval, n)
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="*", default=None,
                        help="Registry pair names to backfill (default: every mapped pair).")
    parser.add_argument("--interval", default="1d", choices=sorted(_INTERVAL_MAP),
                        help="Bar interval (default: 1d).")
    parser.add_argument("--start", default="2020-01-01", help="YYYY-MM-DD (default: 2020-01-01).")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD (default: today).")
    args = parser.parse_args()

    from src.core import secrets
    from src.db.market_data_repository import MarketDataRepository
    from src.db.trade_repository import DBConfig

    cfg = DBConfig.from_mapping(secrets.db_config())
    if not cfg.password:
        logger.error("[dukascopy_backfill] no DB credentials found (secrets.toml "
                     "[database] or DB_* env vars) — aborting.")
        sys.exit(1)
    repo = MarketDataRepository(cfg)
    ok, msg = repo.init_schema()
    if not ok:
        logger.error("[dukascopy_backfill] schema init failed: %s", msg)
        sys.exit(1)

    pairs = args.symbols or list(_INSTRUMENT_MAP.keys())
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.utcnow()

    results = backfill(pairs, args.interval, start, end, repo)
    ok_n = sum(1 for v in results.values() if v)
    skip_n = sum(1 for v in results.values() if v is None)
    logger.info("[dukascopy_backfill] done — %d/%d pairs backfilled, %d skipped",
               ok_n, len(results), skip_n)


if __name__ == "__main__":
    main()
