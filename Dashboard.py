import os
import json
import logging
import traceback
from datetime import datetime
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
import warnings

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from fredapi import Fred

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Macro Dashboard Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# NOTIFICATION PERSISTENCE
# Notified keys are written to disk so they survive the JS-triggered page
# reloads that power Auto-Monitor.
# ============================================================================
NOTIFY_FILE = "/tmp/forex_notify_cache.json"


def load_notified_keys() -> set:
    try:
        with open(NOTIFY_FILE) as f:
            return set(json.load(f).get("keys", []))
    except Exception:
        return set()


def save_notified_keys(keys: set):
    try:
        with open(NOTIFY_FILE, "w") as f:
            json.dump({"keys": list(keys), "updated": datetime.now().isoformat()}, f)
    except Exception:
        pass


# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class AppConfig:
    """Application configuration — all tuneable parameters in one place."""

    assets: Dict[str, str] = field(default_factory=lambda: {
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "USD/JPY": "JPY=X",
        "USD/ZAR": "ZAR=X",
        "AUD/USD": "AUDUSD=X",
        "NZD/USD": "NZDUSD=X",
        "USD/CAD": "CAD=X",
        "USD/CHF": "CHF=X",
        "XAU/USD": "GC=F",
        "BTC/USD": "BTC-USD",          # ← Bitcoin added
    })

    timeframes: Dict[str, Dict] = field(default_factory=lambda: {
        "Weekly":    {"interval": "1wk", "period": "3mo"},
        "Daily":     {"interval": "1d",  "period": "3mo"},
        "4 Hour":    {"interval": "4h",  "period": "1mo"},
        "Hourly":    {"interval": "1h",  "period": "1mo"},
        "15 Minute": {"interval": "15m", "period": "5d"},
    })

    risk_per_trade: float = 0.02
    atr_sl_mult:    float = 1.5

    # FIX: TP1 was 1.5× and stop was also 1.5× → R:R = 1.0 (below min_rr=2.0).
    # Now TP1 = 3.0× ATR → R:R ≥ 2.0, TP2 = 5.0× ATR → R:R ≥ 3.33.
    tp1_atr_mult:   float = 3.0
    tp2_atr_mult:   float = 5.0

    min_rr:         float = 2.0
    adx_trend_min:  float = 20.0
    rsi_os:         float = 40.0
    rsi_ob:         float = 60.0
    stoch_os:       float = 25.0
    stoch_ob:       float = 75.0

    pair_atr_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "EUR/USD": 1.5, "GBP/USD": 1.8, "USD/JPY": 1.5, "USD/ZAR": 2.5,
        "AUD/USD": 1.5, "NZD/USD": 1.6, "USD/CAD": 1.5, "USD/CHF": 1.5,
        "XAU/USD": 2.0, "BTC/USD": 2.0,
    })

    pair_min_stop: Dict[str, float] = field(default_factory=lambda: {
        "EUR/USD": 0.0010, "GBP/USD": 0.0015, "USD/JPY": 0.10,  "USD/ZAR": 0.05,
        "AUD/USD": 0.0010, "NZD/USD": 0.0010, "USD/CAD": 0.0010, "USD/CHF": 0.0010,
        "XAU/USD": 2.00,   "BTC/USD": 500.0,
    })

    london_start: int = 8
    london_end:   int = 16
    ny_start:     int = 13
    ny_end:       int = 21

    dxy_symbol:            str = "DX-Y.NYB"
    cache_ttl:             int = 300   # seconds
    auto_refresh_interval: int = 300   # seconds — matches cache_ttl


# ============================================================================
# FRED SERIES REGISTRY & FALLBACKS
# ============================================================================
FRED_SERIES: Dict[str, Dict[str, str]] = {
    "USD": {"GDP": "A191RL1Q225SBEA",      "CPI": "CPIAUCSL",              "Rates": "FEDFUNDS",          "Unemployment": "UNRATE"},
    "EUR": {"GDP": "CLVMNACSCAB1GQEA19",   "CPI": "CP0000EZ19M086NEST",    "Rates": "ECBDFR",            "Unemployment": "LRHUTTTTEZM156S"},
    "GBP": {"GDP": "CLVMNACSCAB1GQGB",     "CPI": "GBRCPIALLMINMEI",       "Rates": "BOERUKM",           "Unemployment": "LRHUTTTTGBM156S"},
    "JPY": {"GDP": "JPNRGDPEXP",            "CPI": "JPNCPIALLMINMEI",       "Rates": "IRSTCI01JPM156N",   "Unemployment": "LRHUTTTTJPM156S"},
    "ZAR": {"GDP": "ZAFGDPRQPSMEI",         "CPI": "ZAFCPIALLMINMEI",       "Rates": "IRSTCI01ZAM156N",   "Unemployment": "LRHUTTTTZAM156S"},
    "AUD": {"GDP": "AUSGDPRQPSMEI",          "CPI": "AUSCPIALLMINMEI",       "Rates": "IRSTCI01AUM156N",   "Unemployment": "LRHUTTTTAUM156S"},
    "NZD": {"GDP": "NZLGDPRQPSMEI",          "CPI": "NZLCPIALLMINMEI",       "Rates": "IRSTCI01NZM156N",   "Unemployment": "LRHUTTTTNZM156S"},
    "CAD": {"GDP": "CANGDPRQPSMEI",          "CPI": "CANCPIALLMINMEI",       "Rates": "IRSTCI01CAM156N",   "Unemployment": "LRHUTTTTCAM156S"},
    "CHF": {"GDP": "CHEGDPRQPSMEI",          "CPI": "CHECPIALLMINMEI",       "Rates": "IRSTCI01CHM156N",   "Unemployment": "LRHUTTTTCHM156S"},
}
# BTC/XAU are not in FRED_SERIES — they have no FRED macro equivalent.

MACRO_FALLBACKS: Dict[str, Dict[str, float]] = {
    "USD": {"GDP": 2.5,  "Inflation": 3.2,  "Rates": 5.50,  "Unemployment": 3.8},
    "ZAR": {"GDP": 1.2,  "Inflation": 5.0,  "Rates": 8.25,  "Unemployment": 32.1},
    "JPY": {"GDP": 1.1,  "Inflation": 2.8,  "Rates": -0.10, "Unemployment": 2.6},
    "NZD": {"GDP": 2.2,  "Inflation": 3.8,  "Rates": 5.50,  "Unemployment": 3.9},
    "AUD": {"GDP": 2.0,  "Inflation": 4.1,  "Rates": 4.35,  "Unemployment": 3.9},
    "CAD": {"GDP": 1.5,  "Inflation": 3.4,  "Rates": 5.00,  "Unemployment": 5.1},
    "EUR": {"GDP": 0.8,  "Inflation": 2.9,  "Rates": 4.50,  "Unemployment": 6.5},
    "GBP": {"GDP": 0.6,  "Inflation": 3.4,  "Rates": 5.25,  "Unemployment": 4.2},
    "CHF": {"GDP": 0.9,  "Inflation": 2.1,  "Rates": 1.75,  "Unemployment": 2.0},
}


# ============================================================================
# LOGGING
# ============================================================================
def setup_logging() -> logging.Logger:
    logger = logging.getLogger("ForexDashboard")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(h)
    return logger


logger = setup_logging()


# ============================================================================
# CONFIG SINGLETON
# ============================================================================
@st.cache_resource
def get_config() -> AppConfig:
    return AppConfig()


config = get_config()


# ============================================================================
# SAFE VALUE HELPER
# FIX: pd.Series.get() is deprecated and silently drops fallbacks on NaN.
# This helper replaces every .get() call on Series rows throughout the file.
# ============================================================================
def safe_get(row: pd.Series, col: str, default: float = 0.0) -> float:
    """Safely extract a scalar float from a pandas Series row."""
    try:
        if col not in row.index:
            return default
        val = row[col]
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return float(val)
    except Exception:
        return default


# ============================================================================
# NOTIFICATION SYSTEM
# ============================================================================
def init_notification_state():
    """Initialise all notification-related session state keys."""
    if 'notified_keys' not in st.session_state:
        # Load keys that survived the last page reload (file-backed)
        st.session_state.notified_keys = load_notified_keys()
    if 'notification_log' not in st.session_state:
        st.session_state.notification_log = []
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()


def check_and_notify(ideas: List[Dict]) -> List[Dict]:
    """
    Compare new ideas against the seen-key set.
    Fire st.toast() for every NEW high-conviction idea.
    Returns the list of newly alerted ideas.
    """
    init_notification_state()
    new_alerts: List[Dict] = []

    for idea in ideas:
        if idea['conviction'] != 'High':
            continue
        key = f"{idea['pair']}_{idea['bias']}"
        if key not in st.session_state.notified_keys:
            st.session_state.notified_keys.add(key)
            new_alerts.append(idea)

    # Persist so reloads don't re-fire the same alerts
    save_notified_keys(st.session_state.notified_keys)

    for idea in new_alerts:
        direction = "📈 LONG" if idea['bias'] == 'Long' else "📉 SHORT"
        st.toast(
            f"🚨 HIGH CONVICTION\n{idea['pair']} {direction}\n"
            f"Entry {idea['entry']:.5f} | R:R 1:{idea['risk_reward_1']:.2f}",
            icon="🔔",
        )
        st.session_state.notification_log.append({
            "time":  datetime.now().strftime("%H:%M:%S"),
            "pair":  idea['pair'],
            "bias":  idea['bias'],
            "entry": idea['entry'],
            "rr":    idea['risk_reward_1'],
        })

    return new_alerts


# ============================================================================
# FRED CLIENT
# ============================================================================
@st.cache_resource
def get_fred_client(api_key: str) -> Optional[Fred]:
    if not api_key:
        return None
    try:
        return Fred(api_key=api_key)
    except Exception as e:
        logger.error(f"FRED client init failed: {e}")
        return None


def _latest_value(series: pd.Series) -> Optional[float]:
    if series is None or series.empty:
        return None
    clean = series.dropna()
    return float(clean.iloc[-1]) if not clean.empty else None


def _yoy_pct(series: pd.Series) -> Optional[float]:
    """Year-over-year % change from a monthly series."""
    if series is None or series.empty:
        return None
    clean = series.dropna()
    if len(clean) < 13:
        return None
    yoy = clean.pct_change(12) * 100
    vals = yoy.dropna()
    return float(vals.iloc[-1]) if not vals.empty else None


# ============================================================================
# MACRO DATA  (FRED-backed; returns is_live flag so UI can warn on fallback)
# ============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_macro_data(api_key: str) -> Tuple[Dict[str, Dict[str, float]], bool]:
    """
    Returns (data, is_live).
    is_live=False means static fallback data is being used — UI shows a warning.
    FIX: previously returned only data; callers had no way to know it was stale.
    """
    if not api_key:
        logger.warning("No FRED API key — using fallback macro data.")
        return MACRO_FALLBACKS.copy(), False

    fred = get_fred_client(api_key)
    if fred is None:
        return MACRO_FALLBACKS.copy(), False

    result: Dict[str, Dict[str, float]] = {}
    any_success = False

    for currency, series_map in FRED_SERIES.items():
        fb    = MACRO_FALLBACKS.get(currency, {})
        entry: Dict[str, float] = {}

        # GDP
        try:
            raw = fred.get_series(series_map["GDP"])
            if raw is not None and not raw.empty:
                if currency == "USD":
                    val = _latest_value(raw)
                else:
                    clean = raw.dropna()
                    val = float((clean.pct_change(1) * 100 * 4).dropna().iloc[-1]) \
                          if len(clean) >= 5 else None
                entry["GDP"] = val if val is not None else fb.get("GDP", 0.0)
                any_success = True
            else:
                entry["GDP"] = fb.get("GDP", 0.0)
        except Exception as e:
            logger.warning(f"FRED GDP {currency}: {e}")
            entry["GDP"] = fb.get("GDP", 0.0)

        # Inflation (CPI YoY)
        try:
            raw = fred.get_series(series_map["CPI"])
            val = _yoy_pct(raw)
            entry["Inflation"] = val if val is not None else fb.get("Inflation", 0.0)
        except Exception as e:
            logger.warning(f"FRED CPI {currency}: {e}")
            entry["Inflation"] = fb.get("Inflation", 0.0)

        # Policy Rate
        try:
            raw = fred.get_series(series_map["Rates"])
            val = _latest_value(raw)
            entry["Rates"] = val if val is not None else fb.get("Rates", 0.0)
        except Exception as e:
            logger.warning(f"FRED Rates {currency}: {e}")
            entry["Rates"] = fb.get("Rates", 0.0)

        # Unemployment
        try:
            raw = fred.get_series(series_map["Unemployment"])
            val = _latest_value(raw)
            entry["Unemployment"] = val if val is not None else fb.get("Unemployment", 0.0)
        except Exception as e:
            logger.warning(f"FRED Unemployment {currency}: {e}")
            entry["Unemployment"] = fb.get("Unemployment", 0.0)

        result[currency] = entry

    return (result if result else MACRO_FALLBACKS.copy()), any_success


# ============================================================================
# DATA FETCHING
# ============================================================================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(symbol: str, interval: str, period: str) -> pd.DataFrame:
    try:
        df = yf.Ticker(symbol).history(period=period, interval=interval)
        if df.empty:
            logger.warning(f"No data for {symbol} ({interval})")
        return df
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()


# ============================================================================
# TECHNICAL ANALYZER
# ============================================================================
class TechnicalAnalyzer:
    @staticmethod
    def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or len(df) < 20:
            return df
        df = df.copy()
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(c in df.columns for c in required):
            logger.warning("Missing required columns for indicator calculation")
            return df
        try:
            df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()

            macd = ta.trend.MACD(df['Close'])
            df['MACD']           = macd.macd()
            df['MACD_Signal']    = macd.macd_signal()
            df['MACD_Histogram'] = macd.macd_diff()

            df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
            df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
            df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
            df['EMA_50'] = ta.trend.ema_indicator(df['Close'], window=50)

            bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
            df['BB_Upper']  = bb.bollinger_hband()
            df['BB_Middle'] = bb.bollinger_mavg()
            df['BB_Lower']  = bb.bollinger_lband()

            atr_ind = ta.volatility.AverageTrueRange(
                df['High'], df['Low'], df['Close'], window=14)
            df['ATR'] = atr_ind.average_true_range()

            stoch = ta.momentum.StochasticOscillator(
                df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
            df['Stoch_K'] = stoch.stoch()
            df['Stoch_D'] = stoch.stoch_signal()

            adx_ind = ta.trend.ADXIndicator(
                df['High'], df['Low'], df['Close'], window=14)
            df['ADX']     = adx_ind.adx()
            df['ADX_Pos'] = adx_ind.adx_pos()
            df['ADX_Neg'] = adx_ind.adx_neg()

            df['Resistance_20'] = df['High'].rolling(window=20).max()
            df['Support_20']    = df['Low'].rolling(window=20).min()

        except Exception as e:
            logger.error(f"Indicator calculation error: {e}")

        return df


analyzer = TechnicalAnalyzer()


# ============================================================================
# ENTRY SIGNAL GENERATOR
# FIX: Neutral bias previously returned empty reasons — now returns explanation.
# FIX: All pd.Series.get() calls replaced with safe_get().
# ============================================================================
class EntrySignalGenerator:
    def get_entry_signal(self, df_15m: pd.DataFrame, bias: str) -> Dict:
        if df_15m.empty or len(df_15m) < 5:
            return {'signal': 0, 'confidence': 0, 'reasons': ['Insufficient 15-min data']}

        df = analyzer.add_indicators(df_15m)
        if df.empty or len(df) < 2:
            return {'signal': 0, 'confidence': 0, 'reasons': ['Indicator calculation failed']}

        last = df.iloc[-1]
        prev = df.iloc[-2]

        k      = safe_get(last, 'Stoch_K', 50.0)
        d      = safe_get(last, 'Stoch_D', 50.0)
        prev_k = safe_get(prev, 'Stoch_K', 50.0)
        prev_d = safe_get(prev, 'Stoch_D', 50.0)
        rsi    = safe_get(last, 'RSI', 50.0)
        price  = safe_get(last, 'Close', 0.0)

        if price <= 0.0:
            return {'signal': 0, 'confidence': 0, 'reasons': ['Invalid price data']}

        bb_lower = safe_get(last, 'BB_Lower', price * 0.99)
        bb_upper = safe_get(last, 'BB_Upper', price * 1.01)

        signal, confidence, reasons = 0, 0, []

        if bias == 'Long':
            if prev_k <= prev_d and k > d and k < config.stoch_os:
                signal = 1
                confidence += 2
                reasons.append(f"Stochastic bullish crossover (K={k:.1f})")
            if rsi < config.rsi_os:
                confidence += 1
                reasons.append(f"RSI oversold ({rsi:.1f})")
            if price <= bb_lower * 1.002:
                confidence += 1
                reasons.append("Price at lower Bollinger Band")
            if not reasons:
                reasons.append(
                    f"Awaiting Long trigger — K={k:.1f}, RSI={rsi:.1f} "
                    f"(need K<{config.stoch_os} or RSI<{config.rsi_os})"
                )

        elif bias == 'Short':
            if prev_k >= prev_d and k < d and k > config.stoch_ob:
                signal = -1
                confidence += 2
                reasons.append(f"Stochastic bearish crossover (K={k:.1f})")
            if rsi > config.rsi_ob:
                confidence += 1
                reasons.append(f"RSI overbought ({rsi:.1f})")
            if price >= bb_upper * 0.998:
                confidence += 1
                reasons.append("Price at upper Bollinger Band")
            if not reasons:
                reasons.append(
                    f"Awaiting Short trigger — K={k:.1f}, RSI={rsi:.1f} "
                    f"(need K>{config.stoch_ob} or RSI>{config.rsi_ob})"
                )

        else:
            # FIX: was silently returning empty reasons for Neutral bias
            reasons.append(
                f"Trend bias is Neutral (ADX < {config.adx_trend_min:.0f}) — "
                "no directional entry; wait for trend to establish."
            )

        return {
            'signal':     signal,
            'confidence': min(confidence, 5),
            'reasons':    reasons,
            'stoch_k':    k,
            'stoch_d':    d,
            'rsi':        rsi,
            'price':      price,
        }


entry_generator = EntrySignalGenerator()


# ============================================================================
# STOP LOSS CALCULATOR
# FIX: Structure priority logic clarified; BTC pip size added.
# FIX: Method label now accurately reflects which rule won.
# ============================================================================
class StopLossCalculator:
    def pip_size(self, pair: str) -> float:
        if "JPY" in pair:     return 0.01
        if pair == "XAU/USD": return 0.10
        if pair == "BTC/USD": return 1.0    # $1 = 1 "pip" for BTC display
        if "ZAR" in pair:     return 0.001
        return 0.0001

    def price_to_pips(self, pair: str, distance: float) -> float:
        ps = self.pip_size(pair)
        return round(distance / ps, 1) if ps > 0 else 0.0

    def get_swing_stop(self, df: pd.DataFrame, bias: str,
                       lookback: int = 20) -> Optional[float]:
        if df.empty or len(df) < lookback:
            return None
        recent = df.tail(lookback)
        if bias == 'Long' and 'Low' in df.columns:
            return float(recent['Low'].min())
        if bias == 'Short' and 'High' in df.columns:
            return float(recent['High'].max())
        return None

    def calculate(self, df: pd.DataFrame, pair: str, bias: str,
                  current_price: float, atr: float,
                  lookback: int = 20) -> Dict:
        atr_mult = config.pair_atr_multipliers.get(pair, config.atr_sl_mult)
        min_dist = config.pair_min_stop.get(pair, 0.0010)
        buffer   = atr * 0.25

        atr_stop = (current_price - atr * atr_mult) if bias == 'Long' \
                   else (current_price + atr * atr_mult)

        swing  = self.get_swing_stop(df, bias, lookback)
        stop   = atr_stop
        method = "ATR"

        if swing is not None:
            if bias == 'Long':
                struct_stop = swing - buffer
                # Structure is valid only if it places the stop below entry
                if struct_stop < current_price:
                    # Use the WIDER of the two to give structure room;
                    # if structural stop is somehow tighter than ATR, fall back to ATR.
                    if struct_stop <= atr_stop:
                        stop, method = struct_stop, "Swing Low"
                    else:
                        stop, method = atr_stop, "ATR (struct too tight)"
            else:  # Short
                struct_stop = swing + buffer
                if struct_stop > current_price:
                    if struct_stop >= atr_stop:
                        stop, method = struct_stop, "Swing High"
                    else:
                        stop, method = atr_stop, "ATR (struct too tight)"

        # Enforce minimum stop distance
        raw_dist = abs(current_price - stop)
        if raw_dist < min_dist:
            stop   = (current_price - min_dist) if bias == 'Long' \
                     else (current_price + min_dist)
            method += " + min-dist enforced"

        return {
            "stop":          stop,
            "method":        method,
            "distance_pips": self.price_to_pips(pair, abs(current_price - stop)),
        }


sl_calculator = StopLossCalculator()


# ============================================================================
# TAKE PROFIT CALCULATOR
# FIX: TP1 multiplier raised from 1.5× to 3.0× so R:R >= min_rr (2.0).
# FIX: TP2 now also considers swing structure (was always pure-ATR).
# ============================================================================
class TakeProfitCalculator:
    def get_swing_target(self, df: pd.DataFrame, bias: str,
                         lookback: int = 20) -> Optional[float]:
        if df.empty or len(df) < lookback:
            return None
        recent = df.tail(lookback)
        if bias == 'Long' and 'High' in df.columns:
            return float(recent['High'].max())
        if bias == 'Short' and 'Low' in df.columns:
            return float(recent['Low'].min())
        return None

    def calculate(self, df: pd.DataFrame, pair: str, bias: str,
                  current_price: float, atr: float, stop_loss: float,
                  lookback: int = 20) -> Dict:
        stop_dist = abs(current_price - stop_loss) or atr
        swing     = self.get_swing_target(df, bias, lookback)

        if bias == 'Long':
            tp1_atr = current_price + atr * config.tp1_atr_mult
            tp2_atr = current_price + atr * config.tp2_atr_mult

            # TP1: use swing resistance if it sits between price and tp1_atr
            if swing is not None and current_price < swing < tp1_atr:
                tp1, m1 = swing, "Swing High"
            else:
                tp1, m1 = tp1_atr, f"ATR ×{config.tp1_atr_mult}"

            # TP2: FIX — now checks swing beyond TP1 (was pure ATR only)
            if swing is not None and tp1 < swing < tp2_atr:
                tp2, m2 = swing, "Swing High (ext)"
            else:
                tp2, m2 = tp2_atr, f"ATR ×{config.tp2_atr_mult}"

            rr1 = (tp1 - current_price) / stop_dist
            rr2 = (tp2 - current_price) / stop_dist

        else:  # Short
            tp1_atr = current_price - atr * config.tp1_atr_mult
            tp2_atr = current_price - atr * config.tp2_atr_mult

            if swing is not None and tp1_atr < swing < current_price:
                tp1, m1 = swing, "Swing Low"
            else:
                tp1, m1 = tp1_atr, f"ATR ×{config.tp1_atr_mult}"

            # FIX — TP2 swing for shorts
            if swing is not None and tp2_atr < swing < tp1:
                tp2, m2 = swing, "Swing Low (ext)"
            else:
                tp2, m2 = tp2_atr, f"ATR ×{config.tp2_atr_mult}"

            rr1 = (current_price - tp1) / stop_dist
            rr2 = (current_price - tp2) / stop_dist

        return {
            "tp1": tp1, "tp2": tp2,
            "method_tp1": m1, "method_tp2": m2,
            "rr1": round(rr1, 2), "rr2": round(rr2, 2),
            "tp1_valid": rr1 >= config.min_rr,
            "tp2_valid": rr2 >= config.min_rr,
        }


tp_calculator = TakeProfitCalculator()


# ============================================================================
# MULTI-TIMEFRAME ANALYSIS
# FIX: All row.get() calls replaced with safe_get().
# ============================================================================
def analyze_multi_timeframe(
    df_daily: pd.DataFrame, df_4h: pd.DataFrame,
    df_1h: pd.DataFrame, df_15m: pd.DataFrame,
    pair_name: str,
) -> Optional[Dict]:

    df_daily = analyzer.add_indicators(df_daily)
    df_4h    = analyzer.add_indicators(df_4h)
    df_1h    = analyzer.add_indicators(df_1h)
    df_15m   = analyzer.add_indicators(df_15m)

    if any(df.empty for df in [df_daily, df_4h, df_1h, df_15m]):
        return None

    daily     = df_daily.iloc[-1]
    four_hour = df_4h.iloc[-1]
    one_hour  = df_1h.iloc[-1]
    fifteen_m = df_15m.iloc[-1]

    if 'Close' not in daily.index or 'Close' not in four_hour.index:
        return None

    # ── Daily signals ──────────────────────────────────────────────────────
    d_close = safe_get(daily, 'Close')
    d_ema20 = safe_get(daily, 'EMA_20', d_close)
    d_trend = 'Long' if d_close > d_ema20 else 'Short'
    d_rsi   = safe_get(daily, 'RSI', 50.0)
    d_adx   = safe_get(daily, 'ADX', 0.0)

    # ── 4H signals ─────────────────────────────────────────────────────────
    h4_close    = safe_get(four_hour, 'Close')
    h4_ema20    = safe_get(four_hour, 'EMA_20', h4_close)
    h4_ema50    = safe_get(four_hour, 'EMA_50', h4_close)
    h4_trend    = 'Long' if h4_ema20 > h4_ema50 else 'Short'
    h4_macd     = safe_get(four_hour, 'MACD', 0.0)
    h4_sig      = safe_get(four_hour, 'MACD_Signal', 0.0)
    h4_macd_bull = h4_macd > h4_sig

    # ── 1H signals ─────────────────────────────────────────────────────────
    h1_close = safe_get(one_hour, 'Close')
    h1_ema20 = safe_get(one_hour, 'EMA_20', h1_close)
    h1_ema50 = safe_get(one_hour, 'EMA_50', h1_close)
    h1_trend = 'Long' if h1_ema20 > h1_ema50 else 'Short'
    h1_rsi   = safe_get(one_hour, 'RSI', 50.0)

    long_s = short_s = 0
    reasons: List[str] = []

    # Daily
    if d_trend == 'Long':
        long_s += 2; reasons.append("Daily: Bullish EMA alignment")
    else:
        short_s += 2; reasons.append("Daily: Bearish EMA alignment")

    if d_rsi < 40:
        long_s += 1;  reasons.append(f"Daily RSI oversold ({d_rsi:.1f})")
    elif d_rsi > 60:
        short_s += 1; reasons.append(f"Daily RSI overbought ({d_rsi:.1f})")

    if d_adx > config.adx_trend_min:
        if d_trend == 'Long': long_s += 1
        else:                 short_s += 1
        reasons.append(f"Strong trend (ADX={d_adx:.1f})")

    # 4H
    if h4_trend == 'Long':
        long_s += 1;  reasons.append("4H: EMA20 > EMA50")
    else:
        short_s += 1; reasons.append("4H: EMA20 < EMA50")

    if h4_macd_bull:
        long_s += 1;  reasons.append("4H: MACD bullish")
    else:
        short_s += 1; reasons.append("4H: MACD bearish")

    # 1H
    if h1_trend == 'Long':
        long_s += 1;  reasons.append("1H: Bullish EMA alignment")
    else:
        short_s += 1; reasons.append("1H: Bearish EMA alignment")

    if h1_rsi < 45:
        long_s += 1;  reasons.append(f"1H RSI supportive ({h1_rsi:.1f})")
    elif h1_rsi > 55:
        short_s += 1; reasons.append(f"1H RSI resistive ({h1_rsi:.1f})")

    if long_s > short_s:
        final_bias, strength = 'Long', long_s
    elif short_s > long_s:
        final_bias, strength = 'Short', short_s
    else:
        return None   # Tied → no clear bias

    conviction   = "High" if strength >= 6 else ("Medium" if strength >= 3 else "Low")
    entry_signal = entry_generator.get_entry_signal(df_15m, final_bias)

    atr = safe_get(one_hour, 'ATR', 0.0)
    if atr <= 0:
        atr = h1_close * 0.005 if h1_close > 0 else 0.001

    current_price = safe_get(fifteen_m, 'Close', 0.0)
    if current_price <= 0.0:
        return None

    sl_result = sl_calculator.calculate(df_1h, pair_name, final_bias, current_price, atr)
    tp_result = tp_calculator.calculate(
        df_4h, pair_name, final_bias, current_price, atr, sl_result["stop"]
    )

    thesis = " | ".join(reasons)
    if entry_signal and entry_signal['signal'] != 0:
        thesis += f" | Entry: {', '.join(entry_signal['reasons'][:2])}"

    return {
        "pair":             pair_name,
        "bias":             final_bias,
        "conviction":       conviction,
        "strength_score":   strength,
        "thesis":           thesis,
        "entry":            current_price,
        "take_profit_1":    tp_result["tp1"],
        "take_profit_2":    tp_result["tp2"],
        "tp1_method":       tp_result["method_tp1"],
        "tp2_method":       tp_result["method_tp2"],
        "tp1_valid":        tp_result["tp1_valid"],
        "tp2_valid":        tp_result["tp2_valid"],
        "stop_loss":        sl_result["stop"],
        "stop_loss_method": sl_result["method"],
        "stop_loss_pips":   sl_result["distance_pips"],
        "risk_reward_1":    tp_result["rr1"],
        "risk_reward_2":    tp_result["rr2"],
        "atr":              atr,
        "entry_signal":     entry_signal,
    }


def generate_trading_ideas(
    data_by_timeframe: Dict,
) -> Tuple[List[Dict], List[str]]:
    """
    Returns (ideas, skipped_reasons).
    FIX: skipped pairs are now reported to the UI instead of silently dropped.
    """
    ideas:   List[Dict] = []
    skipped: List[str]  = []

    for pair_name in config.assets:
        frames = {
            tf: data_by_timeframe.get(tf, {}).get(pair_name, pd.DataFrame())
            for tf in ['Daily', '4 Hour', 'Hourly', '15 Minute']
        }
        thin = [tf for tf, df in frames.items() if df.empty or len(df) < 20]
        if thin:
            skipped.append(f"{pair_name} — insufficient bars in: {', '.join(thin)}")
            continue

        idea = analyze_multi_timeframe(
            frames['Daily'], frames['4 Hour'],
            frames['Hourly'], frames['15 Minute'],
            pair_name,
        )
        if idea and idea['bias'] != 'Neutral':
            ideas.append(idea)

    ideas.sort(
        key=lambda x: (x['conviction'] == 'High', x['strength_score']),
        reverse=True,
    )
    return ideas, skipped


# ============================================================================
# DATA LOADING
# FIX: DXY removed — it was fetched across 5 timeframes but never used or
#      displayed (dead weight adding 5 extra network calls per load).
# FIX: clear_data_cache() lets a Refresh button or auto-monitor invalidate
#      both the cache and the session-state loaded flag.
# ============================================================================
@st.cache_data(ttl=300, show_spinner=False)
def _fetch_all_timeframes() -> Dict:
    data: Dict[str, Dict] = {tf: {} for tf in config.timeframes}
    for tf_name, tf_cfg in config.timeframes.items():
        for pair_name, symbol in config.assets.items():
            try:
                df = fetch_data(symbol, tf_cfg["interval"], tf_cfg["period"])
                if not df.empty:
                    data[tf_name][pair_name] = df
            except Exception as e:
                logger.warning(f"Failed {pair_name} ({tf_name}): {e}")
    return data


def load_all_timeframes() -> Dict:
    bar = st.progress(0, text="Fetching market data…")
    bar.progress(20, text="Loading all pairs…")
    data = _fetch_all_timeframes()
    bar.progress(100, text="Done ✓")
    bar.empty()
    return data


def clear_data_cache():
    """Bust cached data and force a fresh fetch on next render."""
    _fetch_all_timeframes.clear()
    fetch_data.clear()
    st.session_state.data_loaded   = False
    st.session_state.last_refresh  = datetime.now()


# ============================================================================
# UI — SIDEBAR
# FIX: Now returns auto_monitor flag.
# FIX: Refresh button + data-age indicator added.
# FIX: Notification log displayed in sidebar.
# Auto-monitor state is persisted in URL query-params so it survives the
# JS-triggered page reload (st.session_state is cleared on reload).
# ============================================================================
def render_sidebar(fred_key_default: str) -> Tuple[str, str, bool]:
    with st.sidebar:
        st.header("⚙️ Dashboard Settings")

        # FRED key
        st.subheader("🔑 FRED API Key")
        fred_api_key = st.text_input(
            "API Key", value=fred_key_default, type="password",
            help="Free key at https://fred.stlouisfed.org/docs/api/api_key.html",
        )
        st.success("✅ FRED key loaded") if fred_api_key else \
            st.warning("⚠️ No key — using static fallback data")

        st.divider()

        selected_tf = st.selectbox(
            "Default Chart Timeframe",
            ["Daily", "4 Hour", "Hourly", "15 Minute"],
        )

        st.divider()

        # ── Refresh controls ────────────────────────────────────────────────
        st.subheader("🔄 Data Refresh")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("↺ Refresh Now", use_container_width=True):
                clear_data_cache()
                st.rerun()
        with col_b:
            elapsed = int(
                (datetime.now() - st.session_state.get('last_refresh', datetime.now()))
                .total_seconds()
            )
            st.caption(f"Age: {elapsed}s / {config.cache_ttl}s")

        # Auto-monitor persisted via query params so the JS reload restores it
        qp = st.query_params
        default_am = qp.get("am", "false") == "true"
        auto_monitor = st.toggle("🔔 Auto-Monitor (5 min)", value=default_am)
        qp["am"] = "true" if auto_monitor else "false"

        if auto_monitor:
            st.info(
                "Page auto-refreshes every 5 min. "
                "High-conviction ideas trigger a toast notification."
            )

        st.divider()

        # ── Notification log ─────────────────────────────────────────────────
        st.subheader("🔔 Alert Log")
        log = st.session_state.get('notification_log', [])
        if log:
            for entry in reversed(log[-10:]):
                icon = "📈" if entry['bias'] == 'Long' else "📉"
                st.markdown(
                    f"**{entry['time']}** {icon} **{entry['pair']}** "
                    f"{entry['bias']} — R:R {entry['rr']:.2f}"
                )
            if st.button("🗑️ Clear Alerts"):
                st.session_state.notification_log = []
                st.session_state.notified_keys    = set()
                save_notified_keys(set())
                st.rerun()
        else:
            st.caption("No alerts yet.")

        st.divider()
        st.header("🎯 Strategy")
        st.info(
            f"**Entry:** Stoch crossover · RSI · BB touch\n\n"
            f"**Stop:** Structure + ATR fallback\n\n"
            f"**TP1:** ATR ×{config.tp1_atr_mult} (swing-adjusted)\n\n"
            f"**TP2:** ATR ×{config.tp2_atr_mult} (swing-adjusted)\n\n"
            f"**Risk:** {config.risk_per_trade*100:.0f}% per trade · "
            f"Min R:R {config.min_rr}:1"
        )
        st.caption(f"Last render: {datetime.now().strftime('%H:%M:%S')}")

    return selected_tf, fred_api_key, auto_monitor


# ============================================================================
# UI — KPIs  (now includes BTC and XAU in the strip)
# ============================================================================
def render_kpis(daily_data: Dict):
    kpi_pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "XAU/USD", "BTC/USD"]
    cols      = st.columns(len(kpi_pairs))
    for i, pair in enumerate(kpi_pairs):
        df = daily_data.get(pair)
        with cols[i]:
            if df is not None and not df.empty and 'Close' in df.columns:
                price  = df['Close'].iloc[-1]
                change = df['Close'].pct_change().iloc[-1] * 100 if len(df) > 1 else 0.0
                # BTC and Gold don't need 4dp — use 2dp with thousands separator
                fmt = f"{price:,.2f}" if pair in ("BTC/USD", "XAU/USD") else f"{price:.4f}"
                st.metric(pair, fmt, f"{change:+.2f}%")
            else:
                st.metric(pair, "N/A", "—")


# ============================================================================
# UI — MACRO TABLE
# FIX: is_live flag displayed so users always know if they're seeing fallback.
# ============================================================================
def render_macro_table(macro: Dict, is_live: bool):
    if is_live:
        st.success("✅ Live FRED data")
    else:
        st.warning(
            "⚠️ **Static fallback data** — these figures may be months out of date. "
            "Enter a FRED API key in the sidebar to fetch live values."
        )

    rows = [
        {
            "Currency":     ccy,
            "GDP %":        round(vals.get("GDP", 0), 2),
            "Inflation %":  round(vals.get("Inflation", 0), 2),
            "Rate %":       round(vals.get("Rates", 0), 2),
            "Unemployment": round(vals.get("Unemployment", 0), 2),
        }
        for ccy, vals in macro.items()
    ]
    if not rows:
        st.warning("No macro data available")
        return

    df = pd.DataFrame(rows).set_index("Currency")
    try:
        styled = (
            df.style
            .background_gradient(subset=["GDP %"],        cmap="RdYlGn")
            .background_gradient(subset=["Inflation %"],  cmap="RdYlGn_r")
            .background_gradient(subset=["Rate %"],       cmap="Blues")
            .background_gradient(subset=["Unemployment"], cmap="RdYlGn_r")
        )
        st.dataframe(styled, use_container_width=True)
    except Exception:
        st.dataframe(df, use_container_width=True)


# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    st.title("💹 Macro Dashboard Pro")
    st.caption("Multi-Timeframe · FRED Fundamentals · 15-Min Entry Signals · High-Conviction Alerts")

    init_notification_state()

    # ── Default FRED key ─────────────────────────────────────────────────────
    default_key = ""
    try:
        default_key = (
            st.secrets.get("FRED_API_KEY", "")
            if hasattr(st, "secrets")
            else os.environ.get("FRED_API_KEY", "")
        )
    except Exception:
        default_key = os.environ.get("FRED_API_KEY", "")

    # ── Sidebar ──────────────────────────────────────────────────────────────
    selected_tf, fred_api_key, auto_monitor = render_sidebar(default_key)

    # ── Auto-monitor: page-level refresh via JS meta-refresh ─────────────────
    # When auto_monitor is on, a hidden meta-refresh causes the browser to
    # reload the page every cache_ttl seconds.  Because auto_monitor=True is
    # stored in ?am=true, the toggle is restored after reload.
    if auto_monitor:
        # Belt-and-suspenders: also trigger st.rerun() if enough time has passed
        elapsed = (datetime.now() - st.session_state.last_refresh).total_seconds()
        if elapsed >= config.auto_refresh_interval:
            clear_data_cache()
            st.rerun()

        components.html(
            f'<meta http-equiv="refresh" content="{config.auto_refresh_interval}">',
            height=0,
        )

    # ── Load market data ──────────────────────────────────────────────────────
    # FIX: session state flag now also gets cleared by clear_data_cache() and
    #      the Refresh button, so data actually updates within a session.
    if not st.session_state.get('data_loaded', False):
        with st.spinner("Loading market data…"):
            data_by_timeframe = load_all_timeframes()
            st.session_state.data_by_timeframe = data_by_timeframe
            st.session_state.data_loaded       = True
            st.session_state.last_refresh      = datetime.now()

        with st.spinner("Fetching macro fundamentals from FRED…"):
            macro, is_live = get_macro_data(fred_api_key)
            st.session_state.macro_data = macro
            st.session_state.macro_live = is_live

        # On auto-monitor reloads, silently regenerate ideas and check alerts
        if auto_monitor:
            ideas, _ = generate_trading_ideas(data_by_timeframe)
            st.session_state.latest_ideas = ideas
            check_and_notify(ideas)

    else:
        data_by_timeframe = st.session_state.data_by_timeframe
        macro             = st.session_state.macro_data
        is_live           = st.session_state.get('macro_live', False)

    daily_data = data_by_timeframe.get('Daily', {})

    # ── KPI strip ────────────────────────────────────────────────────────────
    if daily_data:
        render_kpis(daily_data)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🌍 Macro Fundamentals",
        "📈 Technical Chart",
        "⏱️ 15-Min Entry",
        "🎯 Trading Ideas",
    ])

    # ── Tab 1: Overview ───────────────────────────────────────────────────────
    with tab1:
        st.subheader("Market Overview")
        if daily_data:
            rows = []
            for pair, df in daily_data.items():
                if not df.empty and 'Close' in df.columns:
                    price  = df['Close'].iloc[-1]
                    change = df['Close'].pct_change().iloc[-1] * 100 if len(df) > 1 else 0.0
                    dp     = 2 if pair in ("BTC/USD", "XAU/USD") else 5
                    rows.append({
                        "Pair":      pair,
                        "Price":     round(price, dp),
                        "Change %":  round(change, 3),
                        "Bars":      len(df),
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.warning("No valid data for any pair")
        else:
            st.error("No data loaded — check your internet connection.")

    # ── Tab 2: Macro ──────────────────────────────────────────────────────────
    with tab2:
        st.subheader("🌍 Macro Fundamentals (FRED)")
        if macro:
            render_macro_table(macro, is_live)
        else:
            st.warning("No macro data available")

        with st.expander("ℹ️ FRED series IDs used"):
            rows = [
                {"Currency": c, "Metric": m, "Series ID": s}
                for c, sm in FRED_SERIES.items()
                for m, s in sm.items()
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ── Tab 3: Technical Chart ────────────────────────────────────────────────
    with tab3:
        st.subheader("Technical Analysis Chart")
        avail = [p for p, d in daily_data.items() if not d.empty and 'Close' in d.columns]

        if avail:
            col1, col2 = st.columns(2)
            pair = col1.selectbox("Pair",      avail,                          key="chart_pair")
            tf   = col2.selectbox("Timeframe", list(config.timeframes.keys()), key="chart_tf")

            df_c = data_by_timeframe.get(tf, {}).get(pair, pd.DataFrame())
            if not df_c.empty and 'Close' in df_c.columns:
                df_c = analyzer.add_indicators(df_c)

                fig = make_subplots(
                    rows=2, cols=1, shared_xaxes=True,
                    vertical_spacing=0.05, row_heights=[0.7, 0.3],
                    subplot_titles=(f"{pair} — {tf}", "RSI"),
                )
                fig.add_trace(go.Candlestick(
                    x=df_c.index, open=df_c['Open'], high=df_c['High'],
                    low=df_c['Low'], close=df_c['Close'], name="Price",
                ), row=1, col=1)

                for col_name, colour in [('EMA_20', 'orange'), ('EMA_50', 'royalblue')]:
                    if col_name in df_c.columns:
                        fig.add_trace(go.Scatter(
                            x=df_c.index, y=df_c[col_name],
                            name=col_name, line=dict(color=colour, width=1),
                        ), row=1, col=1)

                for bb_col in ['BB_Upper', 'BB_Lower']:
                    if bb_col in df_c.columns:
                        fig.add_trace(go.Scatter(
                            x=df_c.index, y=df_c[bb_col], name=bb_col,
                            line=dict(color='gray', dash='dash'),
                        ), row=1, col=1)

                if 'RSI' in df_c.columns:
                    fig.add_trace(go.Scatter(
                        x=df_c.index, y=df_c['RSI'],
                        name="RSI", line=dict(color='purple'),
                    ), row=2, col=1)
                    fig.add_hline(y=70, line_dash="dash", line_color="red",   row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

                fig.update_layout(height=600, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)

                last = df_c.iloc[-1]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("RSI",     f"{safe_get(last, 'RSI',     0):.1f}")
                m2.metric("ADX",     f"{safe_get(last, 'ADX',     0):.1f}")
                m3.metric("ATR",     f"{safe_get(last, 'ATR',     0):.5f}")
                m4.metric("Stoch K", f"{safe_get(last, 'Stoch_K', 0):.1f}")
            else:
                st.warning(f"No data available for {pair} on {tf}")
        else:
            st.warning("No data available")

    # ── Tab 4: 15-Min Entry ───────────────────────────────────────────────────
    with tab4:
        st.subheader("⏱️ 15-Minute Entry Signal")
        avail = [p for p in daily_data if not daily_data[p].empty]

        if avail:
            pair_e = st.selectbox("Pair", avail, key="entry_pair")
            df_15m = data_by_timeframe.get('15 Minute', {}).get(pair_e, pd.DataFrame())
            df_d   = data_by_timeframe.get('Daily',     {}).get(pair_e, pd.DataFrame())

            if not df_15m.empty and not df_d.empty and 'Close' in df_d.columns:
                df_d_ind = analyzer.add_indicators(df_d)
                if not df_d_ind.empty:
                    di      = df_d_ind.iloc[-1]
                    adx_v   = safe_get(di, 'ADX',   0.0)
                    close_v = safe_get(di, 'Close',  0.0)
                    ema20_v = safe_get(di, 'EMA_20', close_v)

                    bias_v = ('Long' if close_v > ema20_v else 'Short') \
                             if adx_v > config.adx_trend_min else 'Neutral'

                    st.write(f"**Daily Trend Bias:** `{bias_v}` | ADX = {adx_v:.1f}")
                    sig = entry_generator.get_entry_signal(df_15m, bias_v)

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        if   sig['signal'] ==  1: st.success("### 🟢 LONG")
                        elif sig['signal'] == -1: st.error("### 🔴 SHORT")
                        else:                     st.info("### ⚪ NO SIGNAL")
                    c2.metric("Confidence", f"{sig['confidence']}/5")
                    c3.metric("Price",      f"{sig.get('price', 0):.5f}")

                    for r in sig.get('reasons', []):
                        st.info(f"ℹ️ {r}")
                else:
                    st.warning("Could not calculate daily indicators")
            else:
                st.warning("Insufficient data for 15-min analysis on this pair")
        else:
            st.warning("No data available")

    # ── Tab 5: Trading Ideas ──────────────────────────────────────────────────
    with tab5:
        st.subheader("🎯 Trading Ideas")
        st.caption("Multi-timeframe confluence · Structure-based stops · Swing/ATR take-profits")

        if st.button("🔄 Generate Trading Ideas", type="primary"):
            with st.spinner("Analysing all pairs across timeframes…"):
                ideas, skipped = generate_trading_ideas(data_by_timeframe)
                st.session_state.latest_ideas = ideas

            # Fire toast notifications for any new high-conviction ideas
            new_alerts = check_and_notify(ideas)
            if new_alerts:
                st.success(
                    f"🔔 {len(new_alerts)} new high-conviction alert(s) — "
                    "see the sidebar Alert Log"
                )

            # Report skipped pairs
            if skipped:
                with st.expander(f"⚠️ {len(skipped)} pair(s) skipped"):
                    for s in skipped:
                        st.caption(f"• {s}")

            if ideas:
                st.success(f"✅ {len(ideas)} idea(s) generated")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total",           len(ideas))
                c2.metric("Long",            sum(1 for i in ideas if i['bias'] == 'Long'))
                c3.metric("Short",           sum(1 for i in ideas if i['bias'] == 'Short'))
                c4.metric("High Conviction", sum(1 for i in ideas if i['conviction'] == 'High'))

                st.divider()

                for idx, idea in enumerate(ideas):
                    direction = "📈" if idea['bias'] == 'Long' else "📉"
                    header = (
                        f"### {idx+1}. {idea['pair']} — "
                        f"{idea['bias'].upper()} {direction}"
                    )
                    if idea['conviction'] == 'High':
                        st.success(header + " 🔔 HIGH CONVICTION")
                    elif idea['conviction'] == 'Medium':
                        st.warning(header)
                    else:
                        st.info(header)

                    ma, mb, mc = st.columns(3)
                    ma.caption(f"**Conviction:** {idea['conviction']}")
                    mb.caption(f"**Strength:**   {idea['strength_score']}/8")
                    mc.caption(f"**ATR (1H):**   {idea['atr']:.5f}")

                    st.markdown(f"**📝 Thesis:** {idea['thesis']}")
                    st.markdown("**💰 Price Levels:**")

                    p1, p2, p3, p4, p5 = st.columns(5)
                    p1.metric("Entry", f"{idea['entry']:.5f}")

                    tp1_lbl = "TP1" if idea['tp1_valid'] else "TP1 ⚠️"
                    p2.metric(
                        tp1_lbl,
                        f"{idea['take_profit_1']:.5f}",
                        delta=f"R:R 1:{idea['risk_reward_1']:.2f} ({idea['tp1_method']})",
                    )

                    tp2_lbl = "TP2" if idea['tp2_valid'] else "TP2 ⚠️"
                    p3.metric(
                        tp2_lbl,
                        f"{idea['take_profit_2']:.5f}",
                        delta=f"R:R 1:{idea['risk_reward_2']:.2f} ({idea['tp2_method']})",
                    )

                    p4.metric("Stop Loss", f"{idea['stop_loss']:.5f}")
                    risk_pct = (abs(idea['entry'] - idea['stop_loss']) / idea['entry']) * 100
                    p5.metric("Risk %",    f"{risk_pct:.2f}%")

                    st.caption(
                        f"🛡️ Stop method: **{idea['stop_loss_method']}** | "
                        f"Distance: **{idea['stop_loss_pips']} pips**"
                    )

                    if idea.get('entry_signal') and idea['entry_signal'].get('signal') != 0:
                        with st.expander("📊 Entry Signal Details"):
                            es = idea['entry_signal']
                            st.write(f"**Confidence:** {es['confidence']}/5")
                            st.write(
                                f"**Stoch K/D:** "
                                f"{es.get('stoch_k', 0):.1f} / {es.get('stoch_d', 0):.1f}"
                            )
                            st.write(f"**RSI:** {es.get('rsi', 0):.1f}")
                            for r in es.get('reasons', []):
                                st.write(f"  • {r}")

                    st.divider()

                # CSV export
                export_df = pd.DataFrame([{
                    "Pair":        i["pair"],
                    "Bias":        i["bias"],
                    "Conviction":  i["conviction"],
                    "Strength":    i["strength_score"],
                    "Entry":       i["entry"],
                    "TP1":         i["take_profit_1"],
                    "TP1 Method":  i["tp1_method"],
                    "TP1 Valid":   i["tp1_valid"],
                    "TP2":         i["take_profit_2"],
                    "TP2 Method":  i["tp2_method"],
                    "TP2 Valid":   i["tp2_valid"],
                    "Stop Loss":   i["stop_loss"],
                    "Stop Method": i["stop_loss_method"],
                    "R:R TP1":     i["risk_reward_1"],
                    "R:R TP2":     i["risk_reward_2"],
                    "Stop Pips":   i["stop_loss_pips"],
                    "Thesis":      i["thesis"],
                } for i in ideas])

                st.download_button(
                    "📥 Download Trading Ideas (CSV)",
                    data=export_df.to_csv(index=False),
                    file_name=f"ideas_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                )

            else:
                st.warning("⚠️ No trading ideas generated.")
                st.info(
                    "Possible causes:\n"
                    "- Multi-timeframe signals are split (no clear confluence)\n"
                    "- Market is ranging (ADX < threshold on most pairs)\n"
                    "- Some pairs had insufficient intraday bars\n\n"
                    "Try clicking **↺ Refresh Now** in the sidebar then re-running."
                )


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {e}\n{traceback.format_exc()}")
        st.error(f"An error occurred: {e}")
        st.stop()