"""
cot_disaggregated.py
====================
Disaggregated COT (futures-only) fetcher + signal engine for commodities
(gold, WTI crude). Companion to the existing TFF-based cot_fetcher.py /
cot_signals.py, but pointed at the Disaggregated report, which is where
physical commodities live.

Key category mapping vs the TFF report you already use:
    TFF "Leveraged Funds / Asset Managers"  ->  Disagg "Managed Money"  (specs)
    TFF "Dealer/Intermediary"               ->  Disagg "Swap Dealers"
    (new) "Producer/Merchant"               ->  the true commercial hedgers

Signals produced per report week:
    mm_net            Managed Money net contracts
    mm_net_pct_oi     net as % of open interest (comparable across time)
    prod_net_pct_oi   Producer/Merchant net as % of OI
    z_156w / pctl_3y  rolling 3-year z-score and percentile of mm_net_pct_oi
    signal            CROWDED_LONG / CROWDED_SHORT / NEUTRAL
                      (percentile-based, same convention as cot_signals.py)

Data source: CFTC Socrata API, dataset 72hh-3qpy
(Disaggregated Report, Futures Only).
"""

from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

SOCRATA_URL = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"

# CFTC contract market codes (futures-only, disaggregated)
CONTRACTS = {
    "GOLD": {
        "code": "088691",
        "name": "Gold (COMEX)",
        "price_ticker": "GC=F",     # yfinance front-month future
        "spot_ticker": "XAUUSD",    # what you trade on MT4/MT5
    },
    "WTI": {
        "code": "067651",
        "name": "WTI Crude Oil (NYMEX)",
        "price_ticker": "CL=F",
        "spot_ticker": "USOIL / XTIUSD",
    },
}

# Socrata column names. The swap-dealer short column really does have a
# double underscore in the CFTC dataset; we normalise defensively below.
RAW_COLS = {
    "report_date_as_yyyy_mm_dd": "date",
    "open_interest_all": "oi",
    "prod_merc_positions_long": "prod_long",
    "prod_merc_positions_short": "prod_short",
    "swap_positions_long_all": "swap_long",
    "swap__positions_short_all": "swap_short",
    "swap_positions_short_all": "swap_short",   # fallback spelling
    "m_money_positions_long_all": "mm_long",
    "m_money_positions_short_all": "mm_short",
    "other_rept_positions_long": "other_long",
    "other_rept_positions_short": "other_short",
}

ROLL_WEEKS = 156  # 3 years of weekly reports
UPPER_PCTL = 90   # crowded-long threshold, matches cot_signals.py convention
LOWER_PCTL = 10


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_disaggregated(commodity: str, years: int = 6) -> pd.DataFrame:
    """Pull weekly disaggregated futures-only data for one contract."""
    meta = CONTRACTS[commodity]
    since = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    params = {
        "$where": (
            f"cftc_contract_market_code='{meta['code']}' "
            f"AND report_date_as_yyyy_mm_dd>='{since}'"
        ),
        "$order": "report_date_as_yyyy_mm_dd ASC",
        "$limit": 5000,
    }
    resp = requests.get(SOCRATA_URL, params=params, timeout=30)
    resp.raise_for_status()
    raw = pd.DataFrame(resp.json())
    if raw.empty:
        return raw

    keep = {src: dst for src, dst in RAW_COLS.items() if src in raw.columns}
    df = raw[list(keep)].rename(columns=keep)
    # drop duplicate-renamed columns (swap_short fallback)
    df = df.loc[:, ~df.columns.duplicated()]

    df["date"] = pd.to_datetime(df["date"])
    num_cols = [c for c in df.columns if c != "date"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Net positioning, %OI normalisation, rolling z/percentile, signal."""
    out = df.copy()
    out["mm_net"] = out["mm_long"] - out["mm_short"]
    out["prod_net"] = out["prod_long"] - out["prod_short"]
    out["mm_net_pct_oi"] = 100.0 * out["mm_net"] / out["oi"]
    out["prod_net_pct_oi"] = 100.0 * out["prod_net"] / out["oi"]

    roll = out["mm_net_pct_oi"].rolling(ROLL_WEEKS, min_periods=52)
    out["z_156w"] = (out["mm_net_pct_oi"] - roll.mean()) / roll.std()
    out["pctl_3y"] = roll.apply(
        lambda w: 100.0 * (w.iloc[:-1] < w.iloc[-1]).mean(), raw=False
    )

    def classify(p: float) -> str:
        if pd.isna(p):
            return "WARMUP"
        if p >= UPPER_PCTL:
            return "CROWDED_LONG"    # specs stretched long -> fade/mean-revert bias
        if p <= LOWER_PCTL:
            return "CROWDED_SHORT"
        return "NEUTRAL"

    out["signal"] = out["pctl_3y"].apply(classify)

    # week-over-week flow: are specs adding or unwinding?
    out["mm_net_chg"] = out["mm_net"].diff()
    return out


def latest_snapshot(sig: pd.DataFrame) -> dict:
    """One-row summary for the dashboard header / daily cockpit."""
    last = sig.iloc[-1]
    return {
        "report_date": last["date"].date().isoformat(),
        "mm_net": int(last["mm_net"]),
        "mm_net_pct_oi": round(float(last["mm_net_pct_oi"]), 2),
        "z_156w": round(float(last["z_156w"]), 2) if pd.notna(last["z_156w"]) else None,
        "pctl_3y": round(float(last["pctl_3y"]), 1) if pd.notna(last["pctl_3y"]) else None,
        "signal": last["signal"],
        "mm_net_chg_wow": int(last["mm_net_chg"]) if pd.notna(last["mm_net_chg"]) else 0,
    }