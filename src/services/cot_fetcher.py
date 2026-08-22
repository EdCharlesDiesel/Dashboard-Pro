"""
cot_fetcher.py

Fetches CFTC Commitment of Traders (COT) data, caches it locally as CSV,
and computes derived positioning analytics (net positioning, z-scores,
percentile extremes) for forex majors, gold, and DXY futures.

Data source: cftc.gov via the `cot_reports` package (pip install cot_reports).

Two CFTC report types are used because they cover different instruments:
- "traders_in_financial_futures_fut" (TFF report): best for FX majors.
  Breaks speculative positioning into Asset Managers / Leveraged Funds,
  which is closer to "institutional" than the older Legacy report's
  single "Non-Commercial" bucket.
- "legacy_fut" (Legacy report): needed for Gold and the USD Index, which
  aren't included in the TFF report.
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime

import pandas as pd
import cot_reports as cot

logger = logging.getLogger("ForexDashboard")

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cot_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Friendly name -> (report_type, market_name_substring, spec_group)
# spec_group selects which long/short columns count as "speculative" /
# institutional-proxy positioning for that report type.
INSTRUMENTS = {
    "EUR":   ("traders_in_financial_futures_fut", "EURO FX",              "leveraged_funds"),
    "GBP":   ("traders_in_financial_futures_fut", "BRITISH POUND",        "leveraged_funds"),
    "JPY":   ("traders_in_financial_futures_fut", "JAPANESE YEN",         "leveraged_funds"),
    "CHF":   ("traders_in_financial_futures_fut", "SWISS FRANC",          "leveraged_funds"),
    "AUD":   ("traders_in_financial_futures_fut", "AUSTRALIAN DOLLAR",    "leveraged_funds"),
    "NZD":   ("traders_in_financial_futures_fut", "NZ DOLLAR",            "leveraged_funds"),
    "CAD":   ("traders_in_financial_futures_fut", "CANADIAN DOLLAR",      "leveraged_funds"),
    "GOLD":  ("legacy_fut",                        "GOLD",                "noncommercial"),
    "DXY":   ("legacy_fut",                        "USD INDEX",           "noncommercial"),
}

# Column name candidates per spec_group, since exact CFTC header text differs
# between report types (and has drifted slightly across years / library
# versions): "traders_in_financial_futures_fut" uses underscore_case, but
# "legacy_fut" (needed for Gold/DXY) uses "Spaced Case (All)". First match wins.
COLUMN_CANDIDATES = {
    "leveraged_funds": {
        "long":  ["Lev_Money_Positions_Long_All", "Leveraged_Funds_Positions_Long_All"],
        "short": ["Lev_Money_Positions_Short_All", "Leveraged_Funds_Positions_Short_All"],
    },
    "noncommercial": {
        "long":  ["NonComm_Positions_Long_All", "Noncommercial_Positions_Long_All",
                  "Noncommercial Positions-Long (All)"],
        "short": ["NonComm_Positions_Short_All", "Noncommercial_Positions_Short_All",
                  "Noncommercial Positions-Short (All)"],
    },
}

DATE_COL_CANDIDATES = ["Report_Date_as_YYYY-MM-DD", "As_of_Date_In_Form_YYYY-MM-DD",
                       "As of Date in Form YYYY-MM-DD"]
MARKET_COL_CANDIDATES = ["Market_and_Exchange_Names", "Market and Exchange Names"]
OI_COL_CANDIDATES = ["Open_Interest_All", "Open Interest (All)"]


def normalize_datetime64(s: pd.Series, unit: str = "ns") -> pd.Series:
    """Coerce a datetime64 Series to a fixed resolution.

    pandas 2.x infers datetime64 resolution from the source: CFTC's date-only
    strings parsed via ``pd.to_datetime`` land on ``[us]``, while a
    ``DatetimeIndex`` read back through the Postgres-backed OHLC cache lands on
    ``[s]``. Two genuinely-datetime64 columns at different resolutions make
    ``pd.merge_asof``'s ``on=`` column raise "incompatible merge keys ... must
    be the same type" even though a plain ``merge`` would tolerate it. Call this
    on both sides of a merge_asof to make the resolution explicit.
    """
    return s.astype(f"datetime64[{unit}]")


def _first_matching_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"None of the expected columns {candidates} were found. "
        f"Available columns: {list(df.columns)}"
    )


def _cache_path(report_type: str, year: int) -> str:
    return os.path.join(CACHE_DIR, f"{report_type}_{year}.csv")


def _fetch_year(report_type: str, year: int, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch one year of a COT report, using local CSV cache when available.

    Only the current year is ever re-downloaded on repeated calls (COT data
    updates weekly); prior years are cached permanently.
    """
    path = _cache_path(report_type, year)
    current_year = datetime.now().year
    is_current_year = year == current_year

    if os.path.exists(path) and not force_refresh and not is_current_year:
        return pd.read_csv(path, low_memory=False)

    # is_current_year: re-download unless cache is <12 hours old (COT
    # updates once a week, no need to hit cftc.gov on every rerun)
    if os.path.exists(path) and is_current_year and not force_refresh:
        age = datetime.now().timestamp() - os.path.getmtime(path)
        if age < 12 * 3600:
            return pd.read_csv(path, low_memory=False)

    df = cot.cot_year(year=year, cot_report_type=report_type, store_txt=False, verbose=False)
    df.to_csv(path, index=False)
    return df


def fetch_report(report_type: str, years_back: int = 4, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch and concatenate multiple years of a given COT report type."""
    current_year = datetime.now().year
    frames = []
    for y in range(current_year - years_back, current_year + 1):
        try:
            frames.append(_fetch_year(report_type, y, force_refresh=force_refresh))
        except Exception as e:
            # Older years occasionally 404 on CFTC's server for brand-new
            # report types; skip rather than fail the whole fetch.
            logger.warning("[cot_fetcher] Skipping %s %s: %s", report_type, y, e)
    if not frames:
        raise RuntimeError(f"Could not fetch any data for report type '{report_type}'")
    return pd.concat(frames, ignore_index=True)


def get_instrument_series(name: str, years_back: int = 4, force_refresh: bool = False) -> pd.DataFrame:
    """Return a tidy weekly positioning DataFrame for one instrument.

    Columns: date, open_interest, spec_long, spec_short, net_spec,
             net_pct_oi, net_change_wow
    """
    if name not in INSTRUMENTS:
        raise ValueError(f"Unknown instrument '{name}'. Choose from: {list(INSTRUMENTS)}")

    report_type, market_substr, spec_group = INSTRUMENTS[name]
    raw = fetch_report(report_type, years_back=years_back, force_refresh=force_refresh)

    market_col = _first_matching_column(raw, MARKET_COL_CANDIDATES)
    # Match the PRIMARY contract only: anchored at the start of the name and
    # immediately followed by "- <EXCHANGE>". A plain substring search also
    # catches cross-rate contracts (e.g. "EURO FX/BRITISH POUND XRATE") and
    # micro/alt-venue listings (e.g. "MICRO GOLD", "GOLD -1 TROY OUNCE -
    # COINBASE"), which would silently blend a different, much smaller
    # product's open interest into the headline instrument's series.
    pattern = r"^" + re.escape(market_substr) + r"\s*-\s+[A-Z]"
    mask = raw[market_col].str.contains(pattern, case=False, na=False, regex=True)
    df = raw.loc[mask].copy()
    if df.empty:
        raise RuntimeError(
            f"No rows matched market name substring '{market_substr}' in report "
            f"'{report_type}'. Sample market names: "
            f"{raw[market_col].dropna().unique()[:10]}"
        )

    date_col = _first_matching_column(df, DATE_COL_CANDIDATES)
    long_col = _first_matching_column(df, COLUMN_CANDIDATES[spec_group]["long"])
    short_col = _first_matching_column(df, COLUMN_CANDIDATES[spec_group]["short"])
    oi_col = _first_matching_column(df, OI_COL_CANDIDATES)

    df["date"] = pd.to_datetime(df[date_col])
    df = df.sort_values("date").drop_duplicates("date")

    out = pd.DataFrame({
        "date": df["date"],
        "open_interest": pd.to_numeric(df[oi_col], errors="coerce"),
        "spec_long": pd.to_numeric(df[long_col], errors="coerce"),
        "spec_short": pd.to_numeric(df[short_col], errors="coerce"),
    })
    out["net_spec"] = out["spec_long"] - out["spec_short"]
    out["net_pct_oi"] = (out["net_spec"] / out["open_interest"]) * 100
    out["net_change_wow"] = out["net_spec"].diff()
    return out.reset_index(drop=True)


def compute_extremeness(df: pd.DataFrame, lookback_weeks: int = 156) -> dict:
    """Percentile rank + z-score of the latest net positioning vs its own
    trailing history (~3 years of weekly data by default).

    Returns dict with latest value, z-score, percentile (0-100), and a
    plain-language label. This is a positioning-crowdedness read, not a
    trade signal.
    """
    window = df["net_spec"].tail(lookback_weeks).dropna()
    if len(window) < 20:
        return {"error": "Not enough history for a meaningful extremeness read."}

    latest = window.iloc[-1]
    mean, std = window.mean(), window.std()
    z = (latest - mean) / std if std > 0 else 0.0
    percentile = (window <= latest).mean() * 100

    if percentile >= 90:
        label = "Extreme net-long (crowded long)"
    elif percentile <= 10:
        label = "Extreme net-short (crowded short)"
    elif percentile >= 75:
        label = "Stretched net-long"
    elif percentile <= 25:
        label = "Stretched net-short"
    else:
        label = "Neutral / no extreme"

    return {
        "latest_net": round(float(latest), 0),
        "z_score": round(float(z), 2),
        "percentile": round(float(percentile), 1),
        "label": label,
        "lookback_weeks": len(window),
    }


def get_all_snapshot(years_back: int = 4) -> pd.DataFrame:
    """Latest-week snapshot across every configured instrument, for a
    quick cross-market positioning table.
    """
    rows = []
    for name in INSTRUMENTS:
        try:
            series = get_instrument_series(name, years_back=years_back)
            ext = compute_extremeness(series)
            latest_row = series.iloc[-1]
            rows.append({
                "instrument": name,
                "as_of": latest_row["date"].date(),
                "net_spec": latest_row["net_spec"],
                "net_change_wow": latest_row["net_change_wow"],
                "net_pct_oi": round(latest_row["net_pct_oi"], 1),
                "percentile_3y": ext.get("percentile"),
                "read": ext.get("label"),
            })
        except Exception as e:
            logger.warning("[cot_fetcher] Skipping %s in snapshot: %s", name, e)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Quick manual smoke test: python -m src.services.cot_fetcher
    snap = get_all_snapshot(years_back=2)
    print(snap.to_string(index=False))
