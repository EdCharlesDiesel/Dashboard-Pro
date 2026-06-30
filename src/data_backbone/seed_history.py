"""
seed_history.py — one-shot historical backfill + coverage report.

Pulls the deepest history yfinance will give (period="max") for every ticker in
your watchlist (or ones you pass on the command line), upserts it into Postgres,
and prints how far back each instrument actually goes — plus whether it carries
real volume (spot FX does not).

Usage:
    python -m src.data_backbone.seed_history                  # whole WATCH_TICKERS
    python -m src.data_backbone.seed_history EURUSD=X GBPUSD=X # specific tickers
    python -m src.data_backbone.seed_history --fred           # also backfill WATCH_FRED
    python -m src.data_backbone.seed_history --no-save        # report coverage only
    python -m src.data_backbone.seed_history --period 10y     # cap the lookback
"""
from __future__ import annotations

import argparse

import pandas as pd

from . import data_access as da
from . import db
from .config import WATCH_TICKERS, WATCH_FRED


# ---------- pure / testable ----------

def coverage_from_df(ticker: str, df: pd.DataFrame) -> dict:
    """Summarise how much usable history a fetched OHLCV frame contains."""
    if df is None or df.empty:
        return dict(item=ticker, kind="price", bars=0, first=None, last=None,
                    years=0.0, has_volume=False)
    first, last = df.index.min(), df.index.max()
    years = (last - first).days / 365.25
    has_vol = False
    if "Volume" in df.columns:
        v = pd.to_numeric(df["Volume"], errors="coerce").fillna(0).abs().sum()
        has_vol = bool(v > 0)
    return dict(item=ticker, kind="price", bars=int(len(df)),
                first=first.date().isoformat(), last=last.date().isoformat(),
                years=round(years, 1), has_volume=has_vol)


def coverage_from_series(series_id: str, s: pd.Series) -> dict:
    if s is None or s.empty:
        return dict(item=series_id, kind="fred", bars=0, first=None, last=None,
                    years=0.0, has_volume=False)
    first, last = s.index.min(), s.index.max()
    years = (last - first).days / 365.25
    return dict(item=series_id, kind="fred", bars=int(len(s)),
                first=first.date().isoformat(), last=last.date().isoformat(),
                years=round(years, 1), has_volume=False)


# ---------- seeding ----------

def seed_ticker(ticker: str, period: str, save: bool) -> dict:
    df = da.fetch_yf(ticker, period)
    cov = coverage_from_df(ticker, df)
    cov["rows_saved"] = 0
    if save and not df.empty:
        try:
            cov["rows_saved"] = db.upsert_price_bars(ticker, df)
        except Exception as e:
            cov["error"] = str(e)[:90]
    return cov


def seed_fred(series_id: str, save: bool) -> dict:
    s = da.fetch_fred(series_id)
    cov = coverage_from_series(series_id, s)
    cov["rows_saved"] = 0
    if save and not s.empty:
        try:
            cov["rows_saved"] = db.upsert_fred(series_id, s)
        except Exception as e:
            cov["error"] = str(e)[:90]
    return cov


def print_report(rows: list[dict]) -> None:
    if not rows:
        print("No data fetched.")
        return
    df = pd.DataFrame(rows)
    show = df[["item", "first", "last", "years", "bars", "has_volume",
               "rows_saved"]].copy()
    show.columns = ["Instrument", "First", "Last", "Years", "Bars",
                    "Volume?", "Saved"]
    print("\n" + show.to_string(index=False))

    ok = df[df["bars"] > 0]
    if not ok.empty:
        deepest = ok.loc[ok["years"].idxmax()]
        print(f"\nDeepest history: {deepest['item']} "
              f"({deepest['years']} yrs back to {deepest['first']})")
        novol = ok[(ok["kind"] == "price") & (~ok["has_volume"])]["item"].tolist()
        if novol:
            print(f"No real volume (spot FX — POC/Smart-Money won't apply): "
                  f"{', '.join(novol)}")
    failed = df[df["bars"] == 0]["item"].tolist()
    if failed:
        print(f"No data returned for: {', '.join(failed)}")
    if "error" in df.columns:
        errs = df[df["error"].notna()][["item", "error"]]
        if not errs.empty:
            print("\nDB errors:")
            print(errs.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill price/FRED history.")
    ap.add_argument("tickers", nargs="*", help="tickers (default: WATCH_TICKERS)")
    ap.add_argument("--period", default="max", help="lookback (default: max)")
    ap.add_argument("--fred", action="store_true", help="also backfill WATCH_FRED")
    ap.add_argument("--no-save", action="store_true", help="report only, no DB write")
    args = ap.parse_args()

    save = not args.no_save
    if save:
        try:
            db.init_db()
            print("DB ready.")
        except Exception as e:
            print(f"WARNING: DB not reachable ({e}); reporting coverage only.")
            save = False

    tickers = args.tickers or WATCH_TICKERS
    rows = []
    for tk in tickers:
        print(f"Fetching {tk} ({args.period}) ...")
        rows.append(seed_ticker(tk, args.period, save))
    if args.fred:
        for sid in WATCH_FRED:
            print(f"Fetching FRED {sid} ...")
            rows.append(seed_fred(sid, save))

    print_report(rows)


if __name__ == "__main__":
    main()
