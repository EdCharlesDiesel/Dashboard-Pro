import pandas as pd
import yfinance as yf
import requests
import streamlit as st
import logging
from typing import Dict, Optional, Tuple
from fredapi import Fred
from tenacity import retry, stop_after_attempt, wait_fixed
import numpy as np
from src.core.config import default_config

logger = logging.getLogger("ForexDashboard")


class MarketDataProvider:
    def get_data(self, symbol: str, interval: str, period: str):
        raise NotImplementedError


class QuantConnectProvider(MarketDataProvider):
    def __init__(self):
        self.base_url = default_config.quantconnect_url

    def get_data(self, symbol, interval, period):
        try:
            url = f"{self.base_url}/{symbol}?interval={interval}&period={period}"
            r = requests.get(url, timeout=5)
            r.raise_for_status()
            df = pd.DataFrame(r.json())
            if not df.empty and 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
            return df
        except Exception:
            return pd.DataFrame()


class FallbackProvider(MarketDataProvider):
    def get_data(self, symbol, interval, period):
        try:
            # Handle different intervals for yfinance
            yf_interval = interval
            needs_resample = False
            resample_rule = None

            if interval == "4h":
                yf_interval = "1h"
                needs_resample = True
                resample_rule = "4h"
            elif interval == "15m":
                yf_interval = "5m"
                needs_resample = True
                resample_rule = "15min"

            df = yf.Ticker(symbol).history(period=period, interval=yf_interval)

            if df.empty:
                return df

            if needs_resample:
                df = df.resample(resample_rule).agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()

            return df
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            raise  # Re-raise so @retry(tenacity) can handle retries properly


def get_provider():
    try:
        provider = QuantConnectProvider()
        test = provider.get_data("EURUSD=X", "1d", "1mo")
        if test is not None and not test.empty:
            return provider
    except:
        pass
    return FallbackProvider()


# Lazy singleton — the HTTP probe runs only on the first actual fetch call,
# not at import time. This prevents startup crashes when the local
# QuantConnect server isn't running.
_provider_instance: Optional[MarketDataProvider] = None


def _get_provider_lazy() -> MarketDataProvider:
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = get_provider()
    return _provider_instance


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_data(symbol: str, interval: str, period: str) -> pd.DataFrame:
    return _get_provider_lazy().get_data(symbol, interval, period)


# --- MACRO DATA ---

FRED_SERIES: Dict[str, Dict[str, str]] = {
    "USD": {"GDP": "A191RL1Q225SBEA", "CPI": "CPIAUCSL", "Rates": "FEDFUNDS", "Unemployment": "UNRATE"},
    "EUR": {"GDP": "CLVMNACSCAB1GQEA19", "CPI": "CP0000EZ19M086NEST", "Rates": "ECBDFR",
            "Unemployment": "LRHUTTTTEZM156S"},
    "GBP": {"GDP": "CLVMNACSCAB1GQGB", "CPI": "GBRCPIALLMINMEI", "Rates": "BOERUKM", "Unemployment": "LRHUTTTTGBM156S"},
    "JPY": {"GDP": "JPNRGDPEXP", "CPI": "JPNCPIALLMINMEI", "Rates": "IRSTCI01JPM156N",
            "Unemployment": "LRHUTTTTJPM156S"},
    "ZAR": {"GDP": "ZAFGDPRQPSMEI", "CPI": "ZAFCPIALLMINMEI", "Rates": "IRSTCI01ZAM156N",
            "Unemployment": "LRHUTTTTZAM156S"},
    "AUD": {"GDP": "AUSGDPRQPSMEI", "CPI": "AUSCPIALLMINMEI", "Rates": "IRSTCI01AUM156N",
            "Unemployment": "LRHUTTTTAUM156S"},
    "NZD": {"GDP": "NZLGDPRQPSMEI", "CPI": "NZLCPIALLMINMEI", "Rates": "IRSTCI01NZM156N",
            "Unemployment": "LRHUTTTTNZM156S"},
    "CAD": {"GDP": "CANGDPRQPSMEI", "CPI": "CANCPIALLMINMEI", "Rates": "IRSTCI01CAM156N",
            "Unemployment": "LRHUTTTTCAM156S"},
    "CHF": {"GDP": "CHEGDPRQPSMEI", "CPI": "CHECPIALLMINMEI", "Rates": "IRSTCI01CHM156N",
            "Unemployment": "LRHUTTTTCHM156S"},
}

MACRO_FALLBACKS_DATE = "2024-Q1"  # Update this date whenever fallback values are refreshed

MACRO_FALLBACKS: Dict[str, Dict[str, float]] = {
    "USD": {"GDP": 2.5, "Inflation": 3.2, "Rates": 5.50, "Unemployment": 3.8},
    "ZAR": {"GDP": 1.2, "Inflation": 5.0, "Rates": 8.25, "Unemployment": 32.1},
    "JPY": {"GDP": 1.1, "Inflation": 2.8, "Rates": -0.10, "Unemployment": 2.6},
    "NZD": {"GDP": 2.2, "Inflation": 3.8, "Rates": 5.50, "Unemployment": 3.9},
    "AUD": {"GDP": 2.0, "Inflation": 4.1, "Rates": 4.35, "Unemployment": 3.9},
    "CAD": {"GDP": 1.5, "Inflation": 3.4, "Rates": 5.00, "Unemployment": 5.1},
    "EUR": {"GDP": 0.8, "Inflation": 2.9, "Rates": 4.50, "Unemployment": 6.5},
    "GBP": {"GDP": 0.6, "Inflation": 3.4, "Rates": 5.25, "Unemployment": 4.2},
    "CHF": {"GDP": 0.9, "Inflation": 2.1, "Rates": 1.75, "Unemployment": 2.0},
}


def _latest_value(series: pd.Series) -> Optional[float]:
    if series is None or series.empty:
        return None
    clean = series.dropna()
    return float(clean.iloc[-1]) if not clean.empty else None


def _yoy_pct(series: pd.Series) -> Optional[float]:
    if series is None or series.empty:
        return None
    clean = series.dropna()
    if len(clean) < 13:
        return None
    vals = (clean.pct_change(12) * 100).dropna()
    return float(vals.iloc[-1]) if not vals.empty else None


@st.cache_data(ttl=3600)
def get_macro_data(api_key: str) -> Tuple[Dict[str, Dict[str, float]], bool]:
    if not api_key:
        return MACRO_FALLBACKS.copy(), False

    try:
        fred = Fred(api_key=api_key)
    except Exception:
        return MACRO_FALLBACKS.copy(), False

    result: Dict[str, Dict[str, float]] = {}
    any_success = False

    for currency, series_map in FRED_SERIES.items():
        fb = MACRO_FALLBACKS.get(currency, {})
        entry: Dict[str, float] = {}

        try:
            raw = fred.get_series(series_map["GDP"])
            if raw is not None and not raw.empty:
                if currency == "USD":
                    val = _latest_value(raw)
                else:
                    clean = raw.dropna()
                    val = float((clean.pct_change(1) * 100 * 4).dropna().iloc[-1]) if len(clean) >= 5 else None
                entry["GDP"] = val if val is not None else fb.get("GDP", 0.0)
                any_success = True
            else:
                entry["GDP"] = fb.get("GDP", 0.0)
        except Exception:
            entry["GDP"] = fb.get("GDP", 0.0)

        try:
            raw = fred.get_series(series_map["CPI"])
            val = _yoy_pct(raw)
            entry["Inflation"] = val if val is not None else fb.get("Inflation", 0.0)
        except Exception:
            entry["Inflation"] = fb.get("Inflation", 0.0)

        try:
            raw = fred.get_series(series_map["Rates"])
            val = _latest_value(raw)
            entry["Rates"] = val if val is not None else fb.get("Rates", 0.0)
        except Exception:
            entry["Rates"] = fb.get("Rates", 0.0)

        try:
            raw = fred.get_series(series_map["Unemployment"])
            val = _latest_value(raw)
            entry["Unemployment"] = val if val is not None else fb.get("Unemployment", 0.0)
        except Exception:
            entry["Unemployment"] = fb.get("Unemployment", 0.0)

        result[currency] = entry

    return (result if result else MACRO_FALLBACKS.copy()), any_success


@st.cache_data(ttl=3600)
def fetch_fred_series(series_id, fred_key, limit=36):
    if not fred_key:
        return None
    url = (f"https://api.stlouisfed.org/fred/series/observations"
           f"?series_id={series_id}&api_key={fred_key}&file_type=json&sort_order=desc&limit={limit}")
    try:
        r = requests.get(url, timeout=8)
        data = r.json()
        obs = data.get("observations", [])
        df = pd.DataFrame(obs)[["date", "value"]]
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        return df.dropna().sort_values("date")
    except Exception:
        return None