import os
import sys
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass, field
import warnings
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
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
# CONFIGURATION
# ============================================================================

@dataclass
class AppConfig:
    """Application configuration"""
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
    })

    timeframes: Dict[str, Dict] = field(default_factory=lambda: {
        "Weekly":     {"interval": "1wk", "period": "3mo"},
        "Daily":      {"interval": "1d",  "period": "3mo"},
        # FIX #3: was "1h" — must be "4h" to actually fetch 4-hour candles
        "4 Hour":     {"interval": "4h",  "period": "1mo"},
        "Hourly":     {"interval": "1h",  "period": "1mo"},
        "15 Minute":  {"interval": "15m", "period": "5d"},
    })

    risk_per_trade:  float = 0.02
    atr_sl_mult:     float = 1.5
    min_rr:          float = 2.0
    adx_trend_min:   float = 20
    rsi_os:          float = 40
    rsi_ob:          float = 60
    stoch_os:        float = 25
    stoch_ob:        float = 75

    pair_atr_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "EUR/USD": 1.5, "GBP/USD": 1.8, "USD/JPY": 1.5, "USD/ZAR": 2.5,
        "AUD/USD": 1.5, "NZD/USD": 1.6, "USD/CAD": 1.5, "USD/CHF": 1.5,
        "XAU/USD": 2.0,
    })

    pair_min_stop: Dict[str, float] = field(default_factory=lambda: {
        "EUR/USD": 0.0010, "GBP/USD": 0.0015, "USD/JPY": 0.10, "USD/ZAR": 0.05,
        "AUD/USD": 0.0010, "NZD/USD": 0.0010, "USD/CAD": 0.0010, "USD/CHF": 0.0010,
        "XAU/USD": 2.00,
    })

    london_start: int = 8
    london_end:   int = 16
    ny_start:     int = 13
    ny_end:       int = 21

    dxy_symbol: str = "DX-Y.NYB"
    notification_check_interval:  int = 300000


# ============================================================================
# FRED SERIES REGISTRY
# ============================================================================

FRED_SERIES: Dict[str, Dict[str, str]] = {
    "USD": {
        "GDP":          "A191RL1Q225SBEA",
        "CPI":          "CPIAUCSL",
        "Rates":        "FEDFUNDS",
        "Unemployment": "UNRATE",
    },
    "EUR": {
        "GDP":          "CLVMNACSCAB1GQEA19",
        "CPI":          "CP0000EZ19M086NEST",
        "Rates":        "ECBDFR",
        "Unemployment": "LRHUTTTTEZM156S",
    },
    "GBP": {
        "GDP":          "CLVMNACSCAB1GQGB",
        "CPI":          "GBRCPIALLMINMEI",
        "Rates":        "BOERUKM",
        "Unemployment": "LRHUTTTTGBM156S",
    },
    "JPY": {
        "GDP":          "JPNRGDPEXP",
        "CPI":          "JPNCPIALLMINMEI",
        "Rates":        "IRSTCI01JPM156N",
        "Unemployment": "LRHUTTTTJPM156S",
    },
    "ZAR": {
        "GDP":          "ZAFGDPRQPSMEI",
        "CPI":          "ZAFCPIALLMINMEI",
        "Rates":        "IRSTCI01ZAM156N",
        "Unemployment": "LRHUTTTTZAM156S",
    },
    "AUD": {
        "GDP":          "AUSGDPRQPSMEI",
        "CPI":          "AUSCPIALLMINMEI",
        "Rates":        "IRSTCI01AUM156N",
        "Unemployment": "LRHUTTTTAUM156S",
    },
    "NZD": {
        "GDP":          "NZLGDPRQPSMEI",
        "CPI":          "NZLCPIALLMINMEI",
        "Rates":        "IRSTCI01NZM156N",
        # FIX #1: was "LRHUTTTTНЗМ156S" — Н, З, М were Cyrillic characters
        "Unemployment": "LRHUTTTTNZM156S",
    },
    "CAD": {
        "GDP":          "CANGDPRQPSMEI",
        "CPI":          "CANCPIALLMINMEI",
        "Rates":        "IRSTCI01CAM156N",
        "Unemployment": "LRHUTTTTCAM156S",
    },
    "CHF": {
        "GDP":          "CHEGDPRQPSMEI",
        "CPI":          "CHECPIALLMINMEI",
        "Rates":        "IRSTCI01CHM156N",
        "Unemployment": "LRHUTTTTCHM156S",
    },
}

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
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(handler)
    return logger

logger = setup_logging()


# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

@st.cache_resource
def get_config() -> AppConfig:
    return AppConfig()

config = get_config()


# ============================================================================
# FRED CLIENT
# ============================================================================

@st.cache_resource
def get_fred_client(api_key: str) -> Fred:
    return Fred(api_key=api_key)


def _latest_value(series: pd.Series) -> Optional[float]:
    clean = series.dropna()
    return float(clean.iloc[-1]) if not clean.empty else None


def _yoy_pct(series: pd.Series) -> Optional[float]:
    clean = series.dropna()
    if len(clean) < 13:
        return None
    yoy = clean.pct_change(12) * 100
    return float(yoy.dropna().iloc[-1])


# ============================================================================
# MACRO DATA  (FRED-backed, with fallbacks)
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_macro_data(api_key: str) -> Dict[str, Dict[str, float]]:
    if not api_key:
        logger.warning("No FRED API key provided – using fallback macro data.")
        return MACRO_FALLBACKS.copy()

    try:
        fred = get_fred_client(api_key)
    except Exception as e:
        logger.error(f"Failed to initialise FRED client: {e}")
        return MACRO_FALLBACKS.copy()

    result: Dict[str, Dict[str, float]] = {}

    for currency, series_map in FRED_SERIES.items():
        fb = MACRO_FALLBACKS.get(currency, {})
        entry: Dict[str, float] = {}

        # ── GDP ───────────────────────────────────────────────────────────────
        try:
            raw = fred.get_series(series_map["GDP"])
            if currency == "USD":
                val = _latest_value(raw)
            else:
                clean = raw.dropna()
                if len(clean) >= 5:
                    qoq = clean.pct_change(1) * 100 * 4
                    val = float(qoq.dropna().iloc[-1])
                else:
                    val = None
            entry["GDP"] = val if val is not None else fb.get("GDP", 0.0)
        except Exception as e:
            logger.warning(f"FRED GDP fetch failed for {currency}: {e}")
            entry["GDP"] = fb.get("GDP", 0.0)

        # ── Inflation (CPI YoY) ───────────────────────────────────────────────
        try:
            raw = fred.get_series(series_map["CPI"])
            val = _yoy_pct(raw)
            entry["Inflation"] = val if val is not None else fb.get("Inflation", 0.0)
        except Exception as e:
            logger.warning(f"FRED CPI fetch failed for {currency}: {e}")
            entry["Inflation"] = fb.get("Inflation", 0.0)

        # ── Policy Rate ───────────────────────────────────────────────────────
        try:
            raw = fred.get_series(series_map["Rates"])
            val = _latest_value(raw)
            entry["Rates"] = val if val is not None else fb.get("Rates", 0.0)
        except Exception as e:
            logger.warning(f"FRED Rates fetch failed for {currency}: {e}")
            entry["Rates"] = fb.get("Rates", 0.0)

        # ── Unemployment ──────────────────────────────────────────────────────
        try:
            raw = fred.get_series(series_map["Unemployment"])
            val = _latest_value(raw)
            entry["Unemployment"] = val if val is not None else fb.get("Unemployment", 0.0)
        except Exception as e:
            logger.warning(f"FRED Unemployment fetch failed for {currency}: {e}")
            entry["Unemployment"] = fb.get("Unemployment", 0.0)

        result[currency] = entry

    return result


# ============================================================================
# DATA FETCHING
# ============================================================================

@st.cache_data(ttl=config.cache_ttl, show_spinner=False)
def fetch_data(symbol: str, interval: str, period: str) -> pd.DataFrame:
    try:
        df = yf.Ticker(symbol).history(period=period, interval=interval)
        return df
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()


# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================

class TechnicalAnalyzer:
    @staticmethod
    def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or len(df) < 20:
            return df
        df = df.copy()
        try:
            df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
            macd = ta.trend.MACD(df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            df['MACD_Histogram'] = macd.macd_diff()
            df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
            df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
            df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
            df['EMA_50'] = ta.trend.ema_indicator(df['Close'], window=50)
            bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
            df['BB_Upper']  = bb.bollinger_hband()
            df['BB_Middle'] = bb.bollinger_mavg()
            df['BB_Lower']  = bb.bollinger_lband()
            atr = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14)
            df['ATR'] = atr.average_true_range()
            stoch = ta.momentum.StochasticOscillator(
                df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
            df['Stoch_K'] = stoch.stoch()
            df['Stoch_D'] = stoch.stoch_signal()
            adx_ind = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
            df['ADX']     = adx_ind.adx()
            df['ADX_Pos'] = adx_ind.adx_pos()
            df['ADX_Neg'] = adx_ind.adx_neg()
            df['Resistance_20'] = df['High'].rolling(window=20).max()
            df['Support_20']    = df['Low'].rolling(window=20).min()
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
        return df


analyzer = TechnicalAnalyzer()


# ============================================================================
# ENTRY SIGNAL GENERATOR
# ============================================================================

class EntrySignalGenerator:
    def __init__(self):
        self.config = config

    def get_entry_signal(self, df_15m: pd.DataFrame, bias: str) -> Dict:
        if df_15m.empty or len(df_15m) < 5:
            return {'signal': 0, 'confidence': 0, 'reasons': ['Insufficient data']}

        df   = analyzer.add_indicators(df_15m)
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last

        def safe(row, col, default=50):
            v = row.get(col, default)
            return default if pd.isna(v) else v

        k      = safe(last, 'Stoch_K')
        d      = safe(last, 'Stoch_D')
        prev_k = safe(prev, 'Stoch_K')
        prev_d = safe(prev, 'Stoch_D')
        rsi    = safe(last, 'RSI')
        price  = last['Close']
        bb_lower = last.get('BB_Lower', price * 0.99)
        bb_upper = last.get('BB_Upper', price * 1.01)

        signal, confidence, reasons = 0, 0, []

        if bias == 'Long':
            if prev_k <= prev_d and k > d and k < self.config.stoch_os:
                signal = 1; confidence += 2
                reasons.append(f"Stochastic bullish crossover (K={k:.1f})")
            if rsi < self.config.rsi_os:
                confidence += 1; reasons.append(f"RSI oversold ({rsi:.1f})")
            if price <= bb_lower * 1.002:
                confidence += 1; reasons.append("Price at lower Bollinger Band")

        elif bias == 'Short':
            if prev_k >= prev_d and k < d and k > self.config.stoch_ob:
                signal = -1; confidence += 2
                reasons.append(f"Stochastic bearish crossover (K={k:.1f})")
            if rsi > self.config.rsi_ob:
                confidence += 1; reasons.append(f"RSI overbought ({rsi:.1f})")
            if price >= bb_upper * 0.998:
                confidence += 1; reasons.append("Price at upper Bollinger Band")

        return {
            'signal': signal, 'confidence': min(confidence, 5),
            'reasons': reasons, 'stoch_k': k, 'stoch_d': d,
            'rsi': rsi, 'price': price,
        }


entry_generator = EntrySignalGenerator()


# ============================================================================
# STOP LOSS CALCULATOR
# ============================================================================

class StopLossCalculator:
    def __init__(self, cfg: AppConfig):
        self.config = cfg

    def pip_size(self, pair: str) -> float:
        if "JPY" in pair:     return 0.01
        if pair == "XAU/USD": return 0.10
        if "ZAR" in pair:     return 0.001
        return 0.0001

    def price_to_pips(self, pair: str, distance: float) -> float:
        ps = self.pip_size(pair)
        return round(distance / ps, 1) if ps > 0 else 0.0

    def get_swing_stop(self, df: pd.DataFrame, bias: str, lookback: int = 20) -> Optional[float]:
        if df.empty or len(df) < lookback:
            return None
        recent = df.tail(lookback)
        return float(recent['Low'].min()) if bias == 'Long' else float(recent['High'].max())

    def calculate(self, df: pd.DataFrame, pair: str, bias: str,
                  current_price: float, atr: float, lookback: int = 20) -> Dict:
        atr_mult = self.config.pair_atr_multipliers.get(pair, self.config.atr_sl_mult)
        min_dist = self.config.pair_min_stop.get(pair, 0.0010)

        atr_stop = (current_price - atr * atr_mult) if bias == 'Long' \
                   else (current_price + atr * atr_mult)

        swing  = self.get_swing_stop(df, bias, lookback)
        method = []
        buffer = atr * 0.25

        if swing is not None:
            if bias == 'Long':
                struct_stop = swing - buffer
                stop = struct_stop if struct_stop < atr_stop else atr_stop
                method.append("Swing Low" if struct_stop < atr_stop else "ATR")
            else:
                struct_stop = swing + buffer
                stop = struct_stop if struct_stop > atr_stop else atr_stop
                method.append("Swing High" if struct_stop > atr_stop else "ATR")
        else:
            stop = atr_stop
            method.append("ATR")

        if abs(current_price - stop) < min_dist:
            stop = (current_price - min_dist) if bias == 'Long' \
                   else (current_price + min_dist)

        return {
            "stop":          stop,
            "method":        " | ".join(method),
            "distance_pips": self.price_to_pips(pair, abs(current_price - stop)),
        }


sl_calculator = StopLossCalculator(config)


# ============================================================================
# TRADING IDEAS ENGINE
# ============================================================================

def generate_trading_ideas(data_by_timeframe, macro, dxy_by_timeframe):
    ideas = []
    for pair_name in config.assets.keys():
        df_daily = data_by_timeframe.get('Daily',     {}).get(pair_name, pd.DataFrame())
        df_4h    = data_by_timeframe.get('4 Hour',    {}).get(pair_name, pd.DataFrame())
        df_1h    = data_by_timeframe.get('Hourly',    {}).get(pair_name, pd.DataFrame())
        df_15m   = data_by_timeframe.get('15 Minute', {}).get(pair_name, pd.DataFrame())

        if not all(not df.empty and len(df) >= 20
                   for df in [df_daily, df_4h, df_1h, df_15m]):
            continue

        idea = analyze_multi_timeframe(df_daily, df_4h, df_1h, df_15m, pair_name)
        if idea and idea['bias'] != 'Neutral':
            ideas.append(idea)

    ideas.sort(key=lambda x: (x['conviction'] == 'High', x['strength_score']), reverse=True)
    return ideas


# FIX #4: actually use df_4h and df_1h for intermediate trend confirmation
def analyze_multi_timeframe(df_daily, df_4h, df_1h, df_15m, pair_name):
    df_daily = analyzer.add_indicators(df_daily)
    df_4h    = analyzer.add_indicators(df_4h)
    df_1h    = analyzer.add_indicators(df_1h)
    df_15m   = analyzer.add_indicators(df_15m)

    daily       = df_daily.iloc[-1]
    four_hour   = df_4h.iloc[-1]
    one_hour    = df_1h.iloc[-1]
    fifteen_min = df_15m.iloc[-1]

    # ── Daily bias ────────────────────────────────────────────────────────────
    daily_trend = 'Long' if daily['Close'] > daily.get('EMA_20', daily['Close']) else 'Short'
    daily_rsi   = daily.get('RSI', 50)
    daily_adx   = daily.get('ADX', 0)

    # ── 4H confirmation ───────────────────────────────────────────────────────
    h4_ema20    = four_hour.get('EMA_20', four_hour['Close'])
    h4_ema50    = four_hour.get('EMA_50', four_hour['Close'])
    h4_trend    = 'Long' if h4_ema20 > h4_ema50 else 'Short'
    h4_macd     = four_hour.get('MACD', 0)
    h4_signal   = four_hour.get('MACD_Signal', 0)
    h4_macd_bull = (not pd.isna(h4_macd) and not pd.isna(h4_signal) and h4_macd > h4_signal)

    # ── 1H refinement ─────────────────────────────────────────────────────────
    h1_ema20    = one_hour.get('EMA_20', one_hour['Close'])
    h1_ema50    = one_hour.get('EMA_50', one_hour['Close'])
    h1_trend    = 'Long' if h1_ema20 > h1_ema50 else 'Short'
    h1_rsi      = one_hour.get('RSI', 50)

    long_signals = short_signals = 0
    reasons = []

    # Daily
    if daily_trend == 'Long':
        long_signals += 2; reasons.append("Daily: Bullish EMA alignment")
    else:
        short_signals += 2; reasons.append("Daily: Bearish EMA alignment")

    if not pd.isna(daily_rsi):
        if daily_rsi < 40:
            long_signals += 1; reasons.append(f"Daily RSI oversold ({daily_rsi:.1f})")
        elif daily_rsi > 60:
            short_signals += 1; reasons.append(f"Daily RSI overbought ({daily_rsi:.1f})")

    if not pd.isna(daily_adx) and daily_adx > config.adx_trend_min:
        if daily_trend == 'Long': long_signals += 1
        else: short_signals += 1
        reasons.append(f"Strong trend (ADX={daily_adx:.1f})")

    # 4H
    if h4_trend == 'Long':
        long_signals += 1; reasons.append("4H: EMA20 > EMA50")
    else:
        short_signals += 1; reasons.append("4H: EMA20 < EMA50")

    if h4_macd_bull:
        long_signals += 1;  reasons.append("4H: MACD bullish")
    else:
        short_signals += 1; reasons.append("4H: MACD bearish")

    # 1H
    if h1_trend == 'Long':
        long_signals += 1; reasons.append("1H: Bullish EMA alignment")
    else:
        short_signals += 1; reasons.append("1H: Bearish EMA alignment")

    if not pd.isna(h1_rsi):
        if h1_rsi < 45:
            long_signals += 1;  reasons.append(f"1H RSI supportive ({h1_rsi:.1f})")
        elif h1_rsi > 55:
            short_signals += 1; reasons.append(f"1H RSI resistive ({h1_rsi:.1f})")

    if long_signals > short_signals:
        final_bias, strength = 'Long', long_signals
    elif short_signals > long_signals:
        final_bias, strength = 'Short', short_signals
    else:
        return None

    entry_signal = entry_generator.get_entry_signal(df_15m, final_bias)
    conviction   = "High" if strength >= 6 else ("Medium" if strength >= 3 else "Low")

    # ── ATR: use 1H ATR for stop sizing (same frame as sl_calculator input)
    atr = one_hour.get('ATR', one_hour['Close'] * 0.005)
    if pd.isna(atr) or atr <= 0:
        atr = one_hour['Close'] * 0.005

    current_price = fifteen_min['Close']

    # FIX #6: use 4H support/resistance for TP levels — same timeframe as atr
    resistance = four_hour.get('Resistance_20', current_price * 1.02)
    support    = four_hour.get('Support_20',    current_price * 0.98)

    sl_result = sl_calculator.calculate(df_1h, pair_name, final_bias,
                                        current_price, atr, lookback=20)
    stop_loss = sl_result["stop"]

    if final_bias == 'Long':
        entry = current_price
        tp1   = min(current_price + atr * 1.5, resistance) \
                if resistance > current_price else current_price + atr * 1.5
        tp2   = current_price + atr * 3.0
        denom = entry - stop_loss
        rr1   = (tp1 - entry) / denom if denom > 0 else 0
        rr2   = (tp2 - entry) / denom if denom > 0 else 0
    else:
        entry = current_price
        tp1   = max(current_price - atr * 1.5, support) \
                if support < current_price else current_price - atr * 1.5
        tp2   = current_price - atr * 3.0
        denom = stop_loss - entry
        rr1   = (entry - tp1) / denom if denom > 0 else 0
        rr2   = (entry - tp2) / denom if denom > 0 else 0

    thesis = " | ".join(reasons)
    if entry_signal and entry_signal['signal'] != 0:
        thesis += f" | Entry: {', '.join(entry_signal['reasons'][:2])}"

    return {
        "pair":              pair_name,
        "bias":              final_bias,
        "conviction":        conviction,
        "strength_score":    strength,
        "thesis":            thesis,
        "entry":             entry,
        "take_profit_1":     tp1,
        "take_profit_2":     tp2,
        "stop_loss":         stop_loss,
        "stop_loss_method":  sl_result["method"],
        "stop_loss_pips":    sl_result["distance_pips"],
        "risk_reward_1":     rr1,
        "risk_reward_2":     rr2,
        "atr":               atr,
        "entry_signal":      entry_signal,
    }


# ============================================================================
# DATA LOADING
# FIX #2: st.progress() moved outside the cached function so it only runs
#         during actual fetches and not on cache replays.
# ============================================================================

@st.cache_data(ttl=config.cache_ttl)
def _fetch_all_timeframes() -> Tuple[Dict, Dict]:
    """Pure data fetch — no Streamlit UI calls inside."""
    data_by_timeframe = {tf: {} for tf in config.timeframes.keys()}
    dxy_by_timeframe  = {}

    for tf_name, tf_cfg in config.timeframes.items():
        try:
            dxy_df = fetch_data(config.dxy_symbol, tf_cfg["interval"], tf_cfg["period"])
            if not dxy_df.empty:
                dxy_by_timeframe[tf_name] = dxy_df
        except Exception as e:
            logger.warning(f"Failed to load DXY ({tf_name}): {e}")

    for tf_name, tf_cfg in config.timeframes.items():
        for pair_name, symbol in config.assets.items():
            try:
                df = fetch_data(symbol, tf_cfg["interval"], tf_cfg["period"])
                if not df.empty:
                    data_by_timeframe[tf_name][pair_name] = df
            except Exception as e:
                logger.warning(f"Failed to load {pair_name} ({tf_name}): {e}")

    return data_by_timeframe, dxy_by_timeframe


def load_all_timeframes() -> Tuple[Dict, Dict]:
    """Wrapper that shows progress UI then delegates to the cached fetch."""
    total   = len(config.assets) * len(config.timeframes)
    bar     = st.progress(0)
    bar.progress(10)                         # show activity immediately
    result  = _fetch_all_timeframes()
    bar.progress(100)
    bar.empty()
    return result


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_sidebar() -> Tuple[str, str]:
    with st.sidebar:
        st.header("⚙️ Dashboard Settings")

        st.subheader("🔑 FRED API Key")
        default_key = (
            st.secrets.get("FRED_API_KEY", "")
            if hasattr(st, "secrets") and "FRED_API_KEY" in st.secrets
            else os.environ.get("FRED_API_KEY", "")
        )
        fred_api_key = st.text_input(
            "API Key",
            value=default_key,
            type="password",
            help="Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html",
        )
        if fred_api_key:
            st.success("✅ FRED key loaded")
        else:
            st.warning("⚠️ No key – using static fallback data")

        st.divider()

        selected_timeframe = st.selectbox(
            "Default Chart Timeframe",
            ["Daily", "4 Hour", "Hourly", "15 Minute"],
        )
        st.divider()

        st.header("🎯 Strategy Parameters")
        st.info(f"""
        **Entry Conditions:**
        - Stochastic crossover
        - RSI confirmation
        - Bollinger Band touch

        **Stop Loss Method:**
        - Structure + ATR validation

        **Risk:**
        - Risk per trade: {config.risk_per_trade * 100}%
        - Min R/R: {config.min_rr}:1
        """)
        st.divider()
        st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return selected_timeframe, fred_api_key


def render_kpis(daily_data: Dict):
    cols  = st.columns(4)
    pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"]
    for i, pair in enumerate(pairs):
        if i < len(cols):
            with cols[i]:
                df = daily_data.get(pair)
                if df is not None and not df.empty:
                    price  = df['Close'].iloc[-1]
                    change = df['Close'].pct_change().iloc[-1] * 100 if len(df) > 1 else 0
                    st.metric(pair, f"{price:.4f}", f"{change:+.2f}%")


def render_macro_table(macro: Dict):
    rows = []
    for ccy, vals in macro.items():
        rows.append({
            "Currency":     ccy,
            "GDP %":        round(vals.get("GDP", 0), 2),
            "Inflation %":  round(vals.get("Inflation", 0), 2),
            "Rate %":       round(vals.get("Rates", 0), 2),
            "Unemployment": round(vals.get("Unemployment", 0), 2),
        })
    df = pd.DataFrame(rows).set_index("Currency")
    st.dataframe(
        df.style
          .background_gradient(subset=["GDP %"],        cmap="RdYlGn")
          .background_gradient(subset=["Inflation %"],  cmap="RdYlGn_r")
          .background_gradient(subset=["Rate %"],       cmap="Blues")
          .background_gradient(subset=["Unemployment"], cmap="RdYlGn_r"),
        use_container_width=True,
    )


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.title("💹 Dashboard Pro")
    st.caption("Multi-Timeframe Analysis · FRED Macro Data · 15-Minute Entry Signals")

    selected_timeframe, fred_api_key = render_sidebar()

    with st.spinner("Loading market data..."):
        data_by_timeframe, dxy_by_timeframe = load_all_timeframes()

    with st.spinner("Fetching macro fundamentals from FRED..."):
        macro = get_macro_data(fred_api_key)

    daily_data = data_by_timeframe.get('Daily', {})

    if daily_data:
        render_kpis(daily_data)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🌍 Macro Fundamentals",
        "📈 Technical Chart",
        "⏱️ 15-Minute Entry",
        "🎯 Trading Ideas",
    ])

    # ── Overview ──────────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Market Overview")
        if daily_data:
            available = [
                {"Pair": pair, "Price": df['Close'].iloc[-1], "Data Points": len(df)}
                for pair, df in daily_data.items()
                if not df.empty
            ]
            if available:
                st.dataframe(pd.DataFrame(available), use_container_width=True)
        else:
            st.warning("No data available. Please check your internet connection.")

    # ── Macro Fundamentals ────────────────────────────────────────────────────
    with tab2:
        st.subheader("🌍 Macro Fundamentals (FRED)")

        if not fred_api_key:
            st.info("Enter your FRED API key in the sidebar to fetch live data. "
                    "Showing static fallback values below.")

        render_macro_table(macro)

        with st.expander("ℹ️ Series sources"):
            rows = []
            for ccy, series_map in FRED_SERIES.items():
                for metric, sid in series_map.items():
                    rows.append({"Currency": ccy, "Metric": metric, "FRED Series ID": sid})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ── Technical Chart ───────────────────────────────────────────────────────
    with tab3:
        st.subheader("Technical Analysis Chart")
        available_pairs = [p for p in daily_data if not daily_data[p].empty]

        if available_pairs:
            # FIX #7: renamed inner loop variable to `indicator_name` to avoid
            # shadowing the outer `col1` columns variable.
            col1, col2 = st.columns(2)
            with col1:
                pair = st.selectbox("Select Pair", available_pairs, key="chart_pair")
            with col2:
                tf = st.selectbox("Timeframe", list(config.timeframes.keys()), key="chart_tf")

            df = data_by_timeframe[tf].get(pair, pd.DataFrame())
            if not df.empty:
                df = analyzer.add_indicators(df)
                fig = make_subplots(
                    rows=2, cols=1, shared_xaxes=True,
                    vertical_spacing=0.05, row_heights=[0.7, 0.3],
                    subplot_titles=(f'{pair} – {tf}', 'RSI'),
                )
                fig.add_trace(go.Candlestick(
                    x=df.index, open=df['Open'], high=df['High'],
                    low=df['Low'], close=df['Close'], name="Price",
                ), row=1, col=1)
                # FIX #7: renamed loop variable from `col_name` to `indicator_name`
                for indicator_name, colour in [('EMA_20', 'orange'), ('EMA_50', 'blue')]:
                    if indicator_name in df.columns:
                        fig.add_trace(go.Scatter(x=df.index, y=df[indicator_name],
                                                 name=indicator_name,
                                                 line=dict(color=colour, width=1)), row=1, col=1)
                for bb_col in ['BB_Upper', 'BB_Lower']:
                    if bb_col in df.columns:
                        fig.add_trace(go.Scatter(x=df.index, y=df[bb_col], name=bb_col,
                                                 line=dict(color='gray', dash='dash')), row=1, col=1)
                if 'RSI' in df.columns:
                    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI",
                                             line=dict(color='purple')), row=2, col=1)
                    fig.add_hline(y=70, line_dash="dash", line_color="red",   row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

                fig.update_layout(height=600, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)

                last = df.iloc[-1]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("RSI",     f"{last.get('RSI',    0):.1f}")
                c2.metric("ADX",     f"{last.get('ADX',    0):.1f}")
                c3.metric("ATR",     f"{last.get('ATR',    0):.5f}")
                c4.metric("Stoch K", f"{last.get('Stoch_K',0):.1f}")
            else:
                st.warning(f"No data available for {pair}")
        else:
            st.warning("No data available.")

    # ── 15-Minute Entry ───────────────────────────────────────────────────────
    with tab4:
        st.subheader("⏱️ 15-Minute Entry Signals")
        available_pairs = [p for p in daily_data if not daily_data[p].empty]

        if available_pairs:
            pair_entry = st.selectbox("Select Pair", available_pairs, key="entry_pair")
            df_15m = data_by_timeframe.get('15 Minute', {}).get(pair_entry, pd.DataFrame())
            df_d   = data_by_timeframe.get('Daily',     {}).get(pair_entry, pd.DataFrame())

            if not df_15m.empty and not df_d.empty:
                daily_ind  = analyzer.add_indicators(df_d).iloc[-1]
                adx_val    = daily_ind.get('ADX', 0)
                trend_bias = ('Long' if daily_ind['Close'] > daily_ind.get('EMA_20', daily_ind['Close'])
                              else 'Short') if adx_val > config.adx_trend_min else 'Neutral'

                st.write(f"**Trend Bias:** {trend_bias}")
                entry_signal = entry_generator.get_entry_signal(df_15m, trend_bias)

                c1, c2, c3 = st.columns(3)
                with c1:
                    if   entry_signal['signal'] ==  1: st.success("### 🟢 LONG SIGNAL")
                    elif entry_signal['signal'] == -1: st.error("### 🔴 SHORT SIGNAL")
                    else:                              st.info("### ⚪ NO SIGNAL")
                c2.metric("Confidence", f"{entry_signal['confidence']}/5")
                c3.metric("Price",      f"{entry_signal['price']:.5f}")

                for reason in entry_signal['reasons']:
                    st.success(f"✅ {reason}")
            else:
                st.warning("Insufficient data for 15-minute analysis")

    # ── Trading Ideas ─────────────────────────────────────────────────────────
    with tab5:
        st.subheader("🎯 Trading Ideas")
        st.caption("Multi-timeframe analysis with entry signals and take-profit levels")

        if st.button("🔄 Generate Trading Ideas", type="primary", key="gen_ideas"):
            with st.spinner("Analysing all pairs across multiple timeframes..."):
                ideas = generate_trading_ideas(data_by_timeframe, macro, dxy_by_timeframe)

            if ideas:
                st.success(f"✅ Generated {len(ideas)} trading ideas")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Ideas",     len(ideas))
                c2.metric("Long",            sum(1 for i in ideas if i["bias"] == "Long"))
                c3.metric("Short",           sum(1 for i in ideas if i["bias"] == "Short"))
                c4.metric("High Conviction", sum(1 for i in ideas if i["conviction"] == "High"))

                st.divider()

                for idx, idea in enumerate(ideas):
                    if idea["bias"] == "Long":
                        st.success(f"### {idx+1}. {idea['pair']} – LONG 📈")
                    else:
                        st.error(f"### {idx+1}. {idea['pair']} – SHORT 📉")

                    mc1, mc2, mc3 = st.columns(3)
                    mc1.caption(f"**Conviction:** {idea['conviction']}")
                    mc2.caption(f"**Strength:** {idea['strength_score']}/8")
                    mc3.caption(f"**ATR (1H):** {idea['atr']:.5f}")

                    st.markdown(f"**📝 Thesis:** {idea['thesis']}")
                    st.markdown("**💰 Price Levels:**")

                    p1, p2, p3, p4, p5 = st.columns(5)
                    p1.metric("Entry",     f"{idea['entry']:.5f}")
                    p2.metric("TP1",       f"{idea['take_profit_1']:.5f}",
                              delta=f"R:R 1:{idea['risk_reward_1']:.2f}")
                    p3.metric("TP2",       f"{idea['take_profit_2']:.5f}",
                              delta=f"R:R 1:{idea['risk_reward_2']:.2f}")
                    p4.metric("Stop Loss", f"{idea['stop_loss']:.5f}")

                    risk_pct = (abs(idea['entry'] - idea['stop_loss']) / idea['entry']) * 100
                    p5.metric("Risk %", f"{risk_pct:.2f}%")

                    st.caption(f"🛡️ **Stop Method:** {idea['stop_loss_method']} "
                               f"| **Distance:** {idea['stop_loss_pips']} pips")

                    if idea['entry_signal'] and idea['entry_signal']['signal'] != 0:
                        with st.expander("📊 Entry Signal Details"):
                            es = idea['entry_signal']
                            st.write(f"**Confidence:** {es['confidence']}/5")
                            st.write(f"**Stochastic K:** {es['stoch_k']:.1f}")
                            st.write(f"**Stochastic D:** {es['stoch_d']:.1f}")
                            st.write(f"**RSI:** {es['rsi']:.1f}")
                            for r in es['reasons']:
                                st.write(f"  • {r}")

                    st.divider()

                export_df = pd.DataFrame([{
                    "Pair":       i["pair"],
                    "Bias":       i["bias"],
                    "Conviction": i["conviction"],
                    "Entry":      i["entry"],
                    "TP1":        i["take_profit_1"],
                    "TP2":        i["take_profit_2"],
                    "Stop Loss":  i["stop_loss"],
                    "R:R (TP1)":  i["risk_reward_1"],
                    "R:R (TP2)":  i["risk_reward_2"],
                    "Stop Pips":  i["stop_loss_pips"],
                    "Thesis":     i["thesis"],
                } for i in ideas])

                st.download_button(
                    label="📥 Download Trading Ideas (CSV)",
                    data=export_df.to_csv(index=False),
                    file_name=f"trading_ideas_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                )
            else:
                st.warning("⚠️ No trading ideas generated.")
                st.markdown("""
                This could be due to:
                - Insufficient data for some pairs
                - No clear trends detected
                - Market conditions not meeting criteria

                Try refreshing the page or checking individual charts.
                """)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {e}\n{traceback.format_exc()}")
        st.error(f"An error occurred: {str(e)}")
        st.stop()
