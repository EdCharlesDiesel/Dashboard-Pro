"""
Platinum collector job.

Writes the platinum factor set into the existing `ohlc` table so the tab can
stay read-only. Register with the same APScheduler instance that runs your
other collectors.

Lives at: src/data_backbone/platinum_jobs.py

    from src.data_backbone.platinum_jobs import collect_platinum_complex
    scheduler.add_job(collect_platinum_complex, "cron", hour=22, minute=15,
                      args=[engine], id="platinum_complex",
                      replace_existing=True, misfire_grace_time=3600)

Schedule it after the NY close (22:15 SAST is comfortably clear) so you are
writing settled daily bars, not a partial session.
"""

from __future__ import annotations

import logging
import time

import pandas as pd
from sqlalchemy import text

log = logging.getLogger(__name__)

# yfinance ticker -> the symbol name the tab expects in `ohlc`.
# PL=F / PA=F are front-month futures, not spot. They track spot closely enough
# for factor work but will show roll gaps — if you need true spot, point these
# at your MT5 sidecar feed instead.
TICKERS = {
    "PL=F": "XPTUSD",
    "PA=F": "XPDUSD",
    "GC=F": "XAUUSD",
    "DX-Y.NYB": "DXY",
    "ZAR=X": "USDZAR",
}

_UPSERT = """
    INSERT INTO ohlc (symbol, ts, open, high, low, close, volume)
    VALUES (:symbol, :ts, :open, :high, :low, :close, :volume)
    ON CONFLICT (symbol, ts) DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume
"""


def _fetch(ticker: str, period: str, retries: int = 3) -> pd.DataFrame:
    """Fetch with backoff — yfinance rate-limits aggressively on bursts."""
    import yfinance as yf

    for attempt in range(retries):
        try:
            df = yf.Ticker(ticker).history(period=period, interval="1d",
                                           auto_adjust=False)
            if not df.empty:
                return df
            log.warning("empty frame for %s (attempt %d)", ticker, attempt + 1)
        except Exception as exc:  # noqa: BLE001
            log.warning("fetch failed %s (attempt %d): %s", ticker, attempt + 1, exc)
        time.sleep(2 ** attempt * 2)
    return pd.DataFrame()


def collect_platinum_complex(engine, period: str = "3y") -> dict:
    """Fetch every ticker in TICKERS and upsert into ohlc. Returns row counts."""
    results: dict[str, int] = {}

    for ticker, symbol in TICKERS.items():
        df = _fetch(ticker, period)
        if df.empty:
            log.error("no data for %s (%s) — skipping", symbol, ticker)
            results[symbol] = 0
            continue

        df = df.reset_index()
        ts_col = "Date" if "Date" in df.columns else df.columns[0]
        df[ts_col] = pd.to_datetime(df[ts_col]).dt.tz_localize(None)

        rows = [
            {
                "symbol": symbol,
                "ts": r[ts_col].to_pydatetime(),
                "open": float(r["Open"]),
                "high": float(r["High"]),
                "low": float(r["Low"]),
                "close": float(r["Close"]),
                "volume": float(r.get("Volume") or 0.0),
            }
            for _, r in df.iterrows()
            if pd.notna(r.get("Close")) and float(r["Close"]) > 0
        ]
        if not rows:
            results[symbol] = 0
            continue

        try:
            with engine.begin() as conn:
                conn.execute(text(_UPSERT), rows)
            results[symbol] = len(rows)
            log.info("upserted %d rows for %s", len(rows), symbol)
        except Exception as exc:  # noqa: BLE001
            log.exception("upsert failed for %s: %s", symbol, exc)
            results[symbol] = 0

        time.sleep(1.5)  # be polite between tickers

    return results


if __name__ == "__main__":
    import os

    from sqlalchemy import create_engine

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    url = os.environ["DATABASE_URL"]  # env only — never hardcode
    print(collect_platinum_complex(create_engine(url)))
