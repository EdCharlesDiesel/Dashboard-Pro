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
    page_title="Weekly Swing Trading Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SWING TRADING CONFIGURATION
# ============================================================================

@dataclass
class SwingTradingConfig:
    """Configuration optimized for weekly swing trading"""
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
        # Adding major indices for swing trading
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC",
        "Dow Jones": "^DJI",
    })

    # Focused on higher timeframes for swing trading
    timeframes: Dict[str, Dict] = field(default_factory=lambda: {
        "Monthly":    {"interval": "1mo", "period": "2y"},
        "Weekly":     {"interval": "1wk", "period": "1y"},
        "Daily":      {"interval": "1d",  "period": "6mo"},
        "4 Hour":     {"interval": "4h",  "period": "2mo"},
    })

    # Swing trading specific parameters
    risk_per_trade:  float = 0.015  # 1.5% risk per swing trade
    atr_sl_mult:     float = 2.0    # Wider stops for swing trades
    min_rr:          float = 2.5    # Higher R:R requirement for swings
    adx_trend_min:   float = 25     # Stronger trend requirement
    
    # Swing-specific indicator settings
    swing_lookback:  int = 50       # Longer lookback for swing analysis
    rsi_oversold:    float = 30     # More extreme RSI for swing entries
    rsi_overbought:  float = 70
    stoch_os:        float = 20
    stoch_ob:        float = 80
    
    # ATR multipliers for swing stops (wider)
    pair_atr_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "EUR/USD": 2.0, "GBP/USD": 2.2, "USD/JPY": 2.0, "USD/ZAR": 3.0,
        "AUD/USD": 2.0, "NZD/USD": 2.1, "USD/CAD": 2.0, "USD/CHF": 2.0,
        "XAU/USD": 2.5, "S&P 500": 2.0, "NASDAQ": 2.2, "Dow Jones": 2.0,
    })

    # Minimum stop distances (wider for swings)
    pair_min_stop: Dict[str, float] = field(default_factory=lambda: {
        "EUR/USD": 0.0030, "GBP/USD": 0.0040, "USD/JPY": 0.30, "USD/ZAR": 0.15,
        "AUD/USD": 0.0030, "NZD/USD": 0.0030, "USD/CAD": 0.0030, "USD/CHF": 0.0030,
        "XAU/USD": 5.00, "S&P 500": 50, "NASDAQ": 200, "Dow Jones": 500,
    })

    dxy_symbol: str = "DX-Y.NYB"
    cache_ttl: int = 3600  # 1 hour cache for swing trading


# ============================================================================
# FRED SERIES REGISTRY (Same as original)
# ============================================================================

FRED_SERIES: Dict[str, Dict[str, str]] = {
    "USD": {"GDP": "A191RL1Q225SBEA", "CPI": "CPIAUCSL", 
            "Rates": "FEDFUNDS", "Unemployment": "UNRATE"},
    "EUR": {"GDP": "CLVMNACSCAB1GQEA19", "CPI": "CP0000EZ19M086NEST", 
            "Rates": "ECBDFR", "Unemployment": "LRHUTTTTEZM156S"},
    "GBP": {"GDP": "CLVMNACSCAB1GQGB", "CPI": "GBRCPIALLMINMEI", 
            "Rates": "BOERUKM", "Unemployment": "LRHUTTTTGBM156S"},
    "JPY": {"GDP": "JPNRGDPEXP", "CPI": "JPNCPIALLMINMEI", 
            "Rates": "IRSTCI01JPM156N", "Unemployment": "LRHUTTTTJPM156S"},
    "ZAR": {"GDP": "ZAFGDPRQPSMEI", "CPI": "ZAFCPIALLMINMEI", 
            "Rates": "IRSTCI01ZAM156N", "Unemployment": "LRHUTTTTZAM156S"},
    "AUD": {"GDP": "AUSGDPRQPSMEI", "CPI": "AUSCPIALLMINMEI", 
            "Rates": "IRSTCI01AUM156N", "Unemployment": "LRHUTTTTAUM156S"},
    "NZD": {"GDP": "NZLGDPRQPSMEI", "CPI": "NZLCPIALLMINMEI", 
            "Rates": "IRSTCI01NZM156N", "Unemployment": "LRHUTTTTNZM156S"},
    "CAD": {"GDP": "CANGDPRQPSMEI", "CPI": "CANCPIALLMINMEI", 
            "Rates": "IRSTCI01CAM156N", "Unemployment": "LRHUTTTTCAM156S"},
    "CHF": {"GDP": "CHEGDPRQPSMEI", "CPI": "CHECPIALLMINMEI", 
            "Rates": "IRSTCI01CHM156N", "Unemployment": "LRHUTTTTCHM156S"},
}

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


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("SwingTradingApp")
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
def get_config() -> SwingTradingConfig:
    return SwingTradingConfig()

config = get_config()


# ============================================================================
# FRED CLIENT (Same as original)
# ============================================================================

@st.cache_resource
def get_fred_client(api_key: str) -> Optional[Fred]:
    if not api_key:
        return None
    try:
        return Fred(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize FRED client: {e}")
        return None


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
    yoy = clean.pct_change(12) * 100
    return float(yoy.dropna().iloc[-1]) if not yoy.dropna().empty else None


# ============================================================================
# MACRO DATA (Same as original)
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_macro_data(api_key: str) -> Dict[str, Dict[str, float]]:
    if not api_key:
        logger.warning("No FRED API key provided – using fallback macro data.")
        return MACRO_FALLBACKS.copy()

    fred = get_fred_client(api_key)
    if fred is None:
        logger.warning("FRED client initialization failed – using fallback data.")
        return MACRO_FALLBACKS.copy()

    result: Dict[str, Dict[str, float]] = {}

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
                    if len(clean) >= 5:
                        qoq = clean.pct_change(1) * 100 * 4
                        val = float(qoq.dropna().iloc[-1]) if not qoq.dropna().empty else None
                    else:
                        val = None
                entry["GDP"] = val if val is not None else fb.get("GDP", 0.0)
            else:
                entry["GDP"] = fb.get("GDP", 0.0)
        except Exception as e:
            logger.warning(f"FRED GDP fetch failed for {currency}: {e}")
            entry["GDP"] = fb.get("GDP", 0.0)

        try:
            raw = fred.get_series(series_map["CPI"])
            val = _yoy_pct(raw)
            entry["Inflation"] = val if val is not None else fb.get("Inflation", 0.0)
        except Exception as e:
            logger.warning(f"FRED CPI fetch failed for {currency}: {e}")
            entry["Inflation"] = fb.get("Inflation", 0.0)

        try:
            raw = fred.get_series(series_map["Rates"])
            val = _latest_value(raw)
            entry["Rates"] = val if val is not None else fb.get("Rates", 0.0)
        except Exception as e:
            logger.warning(f"FRED Rates fetch failed for {currency}: {e}")
            entry["Rates"] = fb.get("Rates", 0.0)

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

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """Fetch data from yfinance with error handling."""
    try:
        df = yf.Ticker(symbol).history(period=period, interval=interval)
        if df.empty:
            logger.warning(f"No data returned for {symbol} ({interval})")
        return df
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()


# ============================================================================
# SWING TRADING TECHNICAL ANALYZER
# ============================================================================

class SwingTechnicalAnalyzer:
    @staticmethod
    def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add swing trading specific indicators."""
        if df.empty or len(df) < 20:
            return df
        
        df = df.copy()
        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            logger.warning("Missing required columns for indicator calculation")
            return df
        
        try:
            # RSI with 14 period
            df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
            
            # MACD for swing trading
            macd = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            df['MACD_Histogram'] = macd.macd_diff()
            
            # Moving Averages for swing trading
            df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
            df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
            df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
            df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
            df['EMA_50'] = ta.trend.ema_indicator(df['Close'], window=50)
            
            # Bollinger Bands (20,2)
            bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
            df['BB_Upper'] = bb.bollinger_hband()
            df['BB_Middle'] = bb.bollinger_mavg()
            df['BB_Lower'] = bb.bollinger_lband()
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
            
            # ATR for swing stops (14 period)
            atr = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14)
            df['ATR'] = atr.average_true_range()
            df['ATR_Percent'] = (df['ATR'] / df['Close']) * 100
            
            # Stochastic (14,3,3)
            stoch = ta.momentum.StochasticOscillator(
                df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
            df['Stoch_K'] = stoch.stoch()
            df['Stoch_D'] = stoch.stoch_signal()
            
            # ADX for trend strength
            adx_ind = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
            df['ADX'] = adx_ind.adx()
            df['ADX_Pos'] = adx_ind.adx_pos()
            df['ADX_Neg'] = adx_ind.adx_neg()
            
            # Swing-specific indicators
            # Support/Resistance levels (50-period)
            df['Resistance_50'] = df['High'].rolling(window=50).max()
            df['Support_50'] = df['Low'].rolling(window=50).min()
            
            # Volume indicators
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            
            # Ichimoku Cloud for swing trading
            high_9 = df['High'].rolling(window=9).max()
            low_9 = df['Low'].rolling(window=9).min()
            df['Tenkan_Sen'] = (high_9 + low_9) / 2
            
            high_26 = df['High'].rolling(window=26).max()
            low_26 = df['Low'].rolling(window=26).min()
            df['Kijun_Sen'] = (high_26 + low_26) / 2
            
            df['Senkou_Span_A'] = ((df['Tenkan_Sen'] + df['Kijun_Sen']) / 2).shift(26)
            high_52 = df['High'].rolling(window=52).max()
            low_52 = df['Low'].rolling(window=52).min()
            df['Senkou_Span_B'] = ((high_52 + low_52) / 2).shift(26)
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
        
        return df


analyzer = SwingTechnicalAnalyzer()


# ============================================================================
# SWING TRADE SETUP DETECTOR
# ============================================================================

class SwingSetupDetector:
    def __init__(self, cfg: SwingTradingConfig):
        self.config = cfg

    def detect_weekly_setup(self, df_weekly: pd.DataFrame, df_daily: pd.DataFrame) -> Dict:
        """Detect swing trading setups on weekly timeframe."""
        if df_weekly.empty or df_daily.empty:
            return {'setup': False, 'bias': 'Neutral', 'reasons': []}
        
        df_w = analyzer.add_indicators(df_weekly)
        df_d = analyzer.add_indicators(df_daily)
        
        if df_w.empty or df_d.empty or len(df_w) < 10:
            return {'setup': False, 'bias': 'Neutral', 'reasons': []}
        
        weekly = df_w.iloc[-1]
        prev_weekly = df_w.iloc[-2] if len(df_w) > 1 else weekly
        daily = df_d.iloc[-1]
        
        long_signals = short_signals = 0
        reasons = []
        
        # ── Weekly Trend Analysis ──────────────────────────────────────────
        # Price vs EMA alignment
        if 'EMA_20' in weekly.index and 'EMA_50' in weekly.index:
            if weekly['Close'] > weekly['EMA_20'] > weekly['EMA_50']:
                long_signals += 3
                reasons.append("Weekly: Bullish EMA stack (Price > EMA20 > EMA50)")
            elif weekly['Close'] < weekly['EMA_20'] < weekly['EMA_50']:
                short_signals += 3
                reasons.append("Weekly: Bearish EMA stack (Price < EMA20 < EMA50)")
        
        # Weekly ADX for trend strength
        if 'ADX' in weekly.index and weekly['ADX'] > self.config.adx_trend_min:
            if 'ADX_Pos' in weekly.index and 'ADX_Neg' in weekly.index:
                if weekly['ADX_Pos'] > weekly['ADX_Neg']:
                    long_signals += 2
                    reasons.append(f"Weekly: Strong bullish trend (ADX={weekly['ADX']:.1f})")
                else:
                    short_signals += 2
                    reasons.append(f"Weekly: Strong bearish trend (ADX={weekly['ADX']:.1f})")
        
        # Weekly RSI
        if 'RSI' in weekly.index:
            if weekly['RSI'] < self.config.rsi_oversold:
                long_signals += 2
                reasons.append(f"Weekly: Oversold RSI ({weekly['RSI']:.1f})")
            elif weekly['RSI'] > self.config.rsi_overbought:
                short_signals += 2
                reasons.append(f"Weekly: Overbought RSI ({weekly['RSI']:.1f})")
            elif 40 < weekly['RSI'] < 60:
                # Trending RSI
                if weekly['RSI'] > prev_weekly.get('RSI', 50):
                    long_signals += 1
                else:
                    short_signals += 1
        
        # Ichimoku Cloud
        if 'Senkou_Span_A' in weekly.index and 'Senkou_Span_B' in weekly.index:
            price_above_cloud = weekly['Close'] > max(weekly['Senkou_Span_A'], weekly['Senkou_Span_B'])
            price_below_cloud = weekly['Close'] < min(weekly['Senkou_Span_A'], weekly['Senkou_Span_B'])
            
            if price_above_cloud:
                long_signals += 2
                reasons.append("Weekly: Price above Ichimoku Cloud")
            elif price_below_cloud:
                short_signals += 2
                reasons.append("Weekly: Price below Ichimoku Cloud")
        
        # ── Daily Confirmation ─────────────────────────────────────────────
        if 'MACD' in daily.index and 'MACD_Signal' in daily.index:
            if daily['MACD'] > daily['MACD_Signal']:
                long_signals += 2
                reasons.append("Daily: MACD bullish")
            else:
                short_signals += 2
                reasons.append("Daily: MACD bearish")
        
        # Daily Bollinger Bands
        if 'BB_Lower' in daily.index and 'BB_Upper' in daily.index:
            if daily['Close'] <= daily['BB_Lower']:
                long_signals += 2
                reasons.append("Daily: Price at lower Bollinger Band")
            elif daily['Close'] >= daily['BB_Upper']:
                short_signals += 2
                reasons.append("Daily: Price at upper Bollinger Band")
        
        # Volume confirmation
        if 'Volume_Ratio' in daily.index and daily['Volume_Ratio'] > 1.2:
            reasons.append(f"Daily: High volume confirmation ({daily['Volume_Ratio']:.1f}x)")
            if long_signals > short_signals:
                long_signals += 1
            else:
                short_signals += 1
        
        # Determine bias
        if long_signals >= 6 and long_signals > short_signals:
            bias = 'Long'
            strength = long_signals
        elif short_signals >= 6 and short_signals > long_signals:
            bias = 'Short'
            strength = short_signals
        else:
            return {'setup': False, 'bias': 'Neutral', 'reasons': reasons}
        
        return {
            'setup': True,
            'bias': bias,
            'strength': strength,
            'reasons': reasons,
            'weekly_rsi': weekly.get('RSI', 50),
            'weekly_adx': weekly.get('ADX', 0),
            'daily_macd': daily.get('MACD', 0),
            'daily_volume_ratio': daily.get('Volume_Ratio', 1.0),
        }


# ============================================================================
# SWING STOP LOSS CALCULATOR
# ============================================================================

class SwingStopLossCalculator:
    def __init__(self, cfg: SwingTradingConfig):
        self.config = cfg

    def pip_size(self, pair: str) -> float:
        if "JPY" in pair:
            return 0.01
        if pair == "XAU/USD":
            return 0.10
        if "ZAR" in pair:
            return 0.001
        if any(idx in pair for idx in ["S&P", "NASDAQ", "Dow"]):
            return 1.0
        return 0.0001

    def price_to_pips(self, pair: str, distance: float) -> float:
        ps = self.pip_size(pair)
        return round(distance / ps, 1) if ps > 0 else 0.0

    def calculate_swing_stop(self, df_weekly: pd.DataFrame, pair: str, 
                            bias: str, current_price: float) -> Dict:
        """Calculate swing stop loss based on weekly structure."""
        if df_weekly.empty or len(df_weekly) < 10:
            return self._atr_only_stop(df_weekly, pair, bias, current_price)
        
        df_w = analyzer.add_indicators(df_weekly)
        atr = df_w['ATR'].iloc[-1] if 'ATR' in df_w.columns else current_price * 0.02
        
        # Find major swing levels
        lookback = self.config.swing_lookback
        recent = df_w.tail(lookback)
        
        atr_mult = self.config.pair_atr_multipliers.get(pair, self.config.atr_sl_mult)
        min_dist = self.config.pair_min_stop.get(pair, current_price * 0.005)
        
        # ATR-based stop
        atr_stop = (current_price - atr * atr_mult) if bias == 'Long' \
                   else (current_price + atr * atr_mult)
        
        # Structure-based stop
        if bias == 'Long':
            swing_low = recent['Low'].min()
            struct_stop = swing_low - atr * 0.5  # Buffer below swing low
            stop = min(struct_stop, atr_stop)  # Tighter stop
            method = "Swing Low" if struct_stop < atr_stop else "ATR"
        else:
            swing_high = recent['High'].max()
            struct_stop = swing_high + atr * 0.5  # Buffer above swing high
            stop = max(struct_stop, atr_stop)  # Tighter stop
            method = "Swing High" if struct_stop > atr_stop else "ATR"
        
        # Ensure minimum stop distance
        if abs(current_price - stop) < min_dist:
            stop = (current_price - min_dist) if bias == 'Long' \
                   else (current_price + min_dist)
        
        return {
            "stop": stop,
            "method": method,
            "atr": atr,
            "distance_pips": self.price_to_pips(pair, abs(current_price - stop)),
            "risk_percent": (abs(current_price - stop) / current_price) * 100,
        }
    
    def _atr_only_stop(self, df: pd.DataFrame, pair: str, 
                       bias: str, current_price: float) -> Dict:
        """Fallback to ATR-only stop."""
        if not df.empty and len(df) > 14:
            df = analyzer.add_indicators(df)
            atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else current_price * 0.02
        else:
            atr = current_price * 0.02
        
        atr_mult = self.config.pair_atr_multipliers.get(pair, self.config.atr_sl_mult)
        stop = (current_price - atr * atr_mult) if bias == 'Long' \
               else (current_price + atr * atr_mult)
        
        return {
            "stop": stop,
            "method": "ATR",
            "atr": atr,
            "distance_pips": self.price_to_pips(pair, abs(current_price - stop)),
            "risk_percent": (abs(current_price - stop) / current_price) * 100,
        }


# ============================================================================
# SWING TAKE PROFIT CALCULATOR
# ============================================================================

class SwingTakeProfitCalculator:
    def __init__(self, cfg: SwingTradingConfig):
        self.config = cfg

    def calculate_swing_targets(self, df_weekly: pd.DataFrame, df_daily: pd.DataFrame,
                               pair: str, bias: str, current_price: float,
                               stop_loss: float) -> Dict:
        """Calculate swing trading take profit levels."""
        if df_weekly.empty or df_daily.empty:
            return self._atr_only_targets(pair, bias, current_price, stop_loss)
        
        df_w = analyzer.add_indicators(df_weekly)
        df_d = analyzer.add_indicators(df_daily)
        
        atr = df_w['ATR'].iloc[-1] if 'ATR' in df_w.columns else current_price * 0.02
        stop_dist = abs(current_price - stop_loss)
        
        # Find key resistance/support levels
        recent_weekly = df_w.tail(50)
        recent_daily = df_d.tail(20)
        
        if bias == 'Long':
            # TP1: First resistance (recent high or 2x ATR)
            recent_high = recent_daily['High'].max()
            tp1_atr = current_price + atr * 2.0
            
            if recent_high > current_price and recent_high < tp1_atr:
                tp1 = recent_high
                tp1_method = "Recent High"
            else:
                tp1 = tp1_atr
                tp1_method = "ATR 2x"
            
            # TP2: Major resistance (weekly high or Fibonacci extension)
            weekly_high = recent_weekly['High'].max()
            tp2_atr = current_price + atr * 3.5
            
            if weekly_high > tp1 and weekly_high < tp2_atr:
                tp2 = weekly_high
                tp2_method = "Weekly High"
            else:
                tp2 = tp2_atr
                tp2_method = "ATR 3.5x"
            
            # TP3: Extended target (4-5x ATR for trend continuation)
            tp3 = current_price + atr * 5.0
            tp3_method = "ATR 5x"
            
            rr1 = (tp1 - current_price) / stop_dist
            rr2 = (tp2 - current_price) / stop_dist
            rr3 = (tp3 - current_price) / stop_dist
            
        else:  # Short
            recent_low = recent_daily['Low'].min()
            tp1_atr = current_price - atr * 2.0
            
            if recent_low < current_price and recent_low > tp1_atr:
                tp1 = recent_low
                tp1_method = "Recent Low"
            else:
                tp1 = tp1_atr
                tp1_method = "ATR 2x"
            
            weekly_low = recent_weekly['Low'].min()
            tp2_atr = current_price - atr * 3.5
            
            if weekly_low < tp1 and weekly_low > tp2_atr:
                tp2 = weekly_low
                tp2_method = "Weekly Low"
            else:
                tp2 = tp2_atr
                tp2_method = "ATR 3.5x"
            
            tp3 = current_price - atr * 5.0
            tp3_method = "ATR 5x"
            
            rr1 = (current_price - tp1) / stop_dist
            rr2 = (current_price - tp2) / stop_dist
            rr3 = (current_price - tp3) / stop_dist
        
        return {
            "tp1": tp1, "tp1_method": tp1_method, "rr1": round(rr1, 2),
            "tp2": tp2, "tp2_method": tp2_method, "rr2": round(rr2, 2),
            "tp3": tp3, "tp3_method": tp3_method, "rr3": round(rr3, 2),
            "tp1_valid": rr1 >= self.config.min_rr,
            "tp2_valid": rr2 >= self.config.min_rr,
            "tp3_valid": rr3 >= self.config.min_rr,
        }
    
    def _atr_only_targets(self, pair: str, bias: str, 
                          current_price: float, stop_loss: float) -> Dict:
        """Fallback to ATR-only targets."""
        atr = current_price * 0.02
        stop_dist = abs(current_price - stop_loss)
        
        if bias == 'Long':
            tp1 = current_price + atr * 2.0
            tp2 = current_price + atr * 3.5
            tp3 = current_price + atr * 5.0
            rr1 = (tp1 - current_price) / stop_dist
            rr2 = (tp2 - current_price) / stop_dist
            rr3 = (tp3 - current_price) / stop_dist
        else:
            tp1 = current_price - atr * 2.0
            tp2 = current_price - atr * 3.5
            tp3 = current_price - atr * 5.0
            rr1 = (current_price - tp1) / stop_dist
            rr2 = (current_price - tp2) / stop_dist
            rr3 = (current_price - tp3) / stop_dist
        
        return {
            "tp1": tp1, "tp1_method": "ATR 2x", "rr1": round(rr1, 2),
            "tp2": tp2, "tp2_method": "ATR 3.5x", "rr2": round(rr2, 2),
            "tp3": tp3, "tp3_method": "ATR 5x", "rr3": round(rr3, 2),
            "tp1_valid": rr1 >= self.config.min_rr,
            "tp2_valid": rr2 >= self.config.min_rr,
            "tp3_valid": rr3 >= self.config.min_rr,
        }


# ============================================================================
# WEEKLY SWING IDEAS GENERATOR
# ============================================================================

def generate_swing_ideas(data_by_timeframe: Dict) -> List[Dict]:
    """Generate weekly swing trading ideas."""
    ideas = []
    setup_detector = SwingSetupDetector(config)
    sl_calc = SwingStopLossCalculator(config)
    tp_calc = SwingTakeProfitCalculator(config)
    
    for pair_name in config.assets.keys():
        df_monthly = data_by_timeframe.get('Monthly', {}).get(pair_name, pd.DataFrame())
        df_weekly = data_by_timeframe.get('Weekly', {}).get(pair_name, pd.DataFrame())
        df_daily = data_by_timeframe.get('Daily', {}).get(pair_name, pd.DataFrame())
        df_4h = data_by_timeframe.get('4 Hour', {}).get(pair_name, pd.DataFrame())

        # Need at least weekly and daily data
        if df_weekly.empty or df_daily.empty:
            continue
        if len(df_weekly) < 10 or len(df_daily) < 20:
            continue

        # Detect setup
        setup = setup_detector.detect_weekly_setup(df_weekly, df_daily)
        
        if not setup['setup'] or setup['bias'] == 'Neutral':
            continue
        
        current_price = df_daily['Close'].iloc[-1] if 'Close' in df_daily.columns else 0
        if current_price == 0:
            continue
        
        # Calculate stop loss
        sl_result = sl_calc.calculate_swing_stop(df_weekly, pair_name, 
                                                  setup['bias'], current_price)
        
        # Calculate take profit levels
        tp_result = tp_calc.calculate_swing_targets(df_weekly, df_daily, pair_name,
                                                     setup['bias'], current_price,
                                                     sl_result['stop'])
        
        # Determine conviction based on multiple factors
        conviction_score = setup['strength']
        if tp_result['tp1_valid'] and tp_result['tp2_valid']:
            conviction_score += 2
        if sl_result['risk_percent'] < config.risk_per_trade * 100:
            conviction_score += 1
        
        if conviction_score >= 10:
            conviction = "High"
        elif conviction_score >= 7:
            conviction = "Medium"
        else:
            conviction = "Low"
        
        # Build the idea
        idea = {
            "pair": pair_name,
            "bias": setup['bias'],
            "conviction": conviction,
            "strength_score": conviction_score,
            "thesis": " | ".join(setup['reasons']),
            "entry": current_price,
            "stop_loss": sl_result['stop'],
            "stop_method": sl_result['method'],
            "stop_pips": sl_result['distance_pips'],
            "risk_percent": sl_result['risk_percent'],
            "atr": sl_result['atr'],
            
            "tp1": tp_result['tp1'],
            "tp1_method": tp_result['tp1_method'],
            "tp1_valid": tp_result['tp1_valid'],
            "rr1": tp_result['rr1'],
            
            "tp2": tp_result['tp2'],
            "tp2_method": tp_result['tp2_method'],
            "tp2_valid": tp_result['tp2_valid'],
            "rr2": tp_result['rr2'],
            
            "tp3": tp_result['tp3'],
            "tp3_method": tp_result['tp3_method'],
            "tp3_valid": tp_result['tp3_valid'],
            "rr3": tp_result['rr3'],
            
            "weekly_rsi": setup['weekly_rsi'],
            "weekly_adx": setup['weekly_adx'],
        }
        
        ideas.append(idea)
    
    # Sort by conviction and strength
    ideas.sort(key=lambda x: (x['conviction'] == 'High', x['strength_score']), reverse=True)
    return ideas


# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_all_timeframes() -> Dict:
    """Fetch data for swing trading timeframes."""
    data_by_timeframe = {tf: {} for tf in config.timeframes.keys()}
    
    for tf_name, tf_cfg in config.timeframes.items():
        for pair_name, symbol in config.assets.items():
            try:
                df = fetch_data(symbol, tf_cfg["interval"], tf_cfg["period"])
                if not df.empty:
                    data_by_timeframe[tf_name][pair_name] = df
            except Exception as e:
                logger.warning(f"Failed to load {pair_name} ({tf_name}): {e}")
    
    return data_by_timeframe


def load_all_timeframes() -> Dict:
    """Wrapper that shows progress UI."""
    progress_bar = st.progress(0, text="Loading swing trading data...")
    
    progress_bar.progress(30, text="Fetching weekly data...")
    data_by_timeframe = _fetch_all_timeframes()
    
    progress_bar.progress(100, text="Data loaded!")
    progress_bar.empty()
    
    return data_by_timeframe


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_sidebar() -> Tuple[str, str]:
    """Render sidebar for swing trading app."""
    with st.sidebar:
        st.header("📈 Swing Trading Settings")
        
        st.subheader("🔑 FRED API Key")
        
        default_key = ""
        try:
            if hasattr(st, "secrets") and "FRED_API_KEY" in st.secrets:
                default_key = st.secrets["FRED_API_KEY"]
            else:
                default_key = os.environ.get("FRED_API_KEY", "")
        except Exception:
            default_key = os.environ.get("FRED_API_KEY", "")
            
        fred_api_key = st.text_input(
            "API Key",
            value=default_key,
            type="password",
            help="Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html",
        )
        
        st.divider()
        st.header("🎯 Swing Strategy Parameters")
        st.info(f"""
        **Timeframe Focus:**
        - Primary: Weekly
        - Confirmation: Daily
        - Entry Refinement: 4H
        
        **Entry Criteria:**
        - EMA stack alignment
        - Strong ADX (>25)
        - Ichimoku Cloud confirmation
        - MACD alignment
        
        **Risk Management:**
        - Risk per trade: {config.risk_per_trade * 100}%
        - Min R:R: {config.min_rr}:1
        - ATR stop multiplier: {config.atr_sl_mult}x
        
        **Holding Period:**
        - 2-8 weeks typical
        - Multiple TP levels
        """)
        
        st.divider()
        st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return fred_api_key


def render_swing_kpis(data_by_timeframe: Dict):
    """Render KPI metrics for swing trading."""
    daily_data = data_by_timeframe.get('Daily', {})
    weekly_data = data_by_timeframe.get('Weekly', {})
    
    cols = st.columns(4)
    major_pairs = ["EUR/USD", "S&P 500", "XAU/USD", "GBP/USD"]
    
    for i, pair in enumerate(major_pairs):
        if i < len(cols):
            with cols[i]:
                df_d = daily_data.get(pair)
                df_w = weekly_data.get(pair)
                
                if df_d is not None and not df_d.empty and 'Close' in df_d.columns:
                    price = df_d['Close'].iloc[-1]
                    
                    # Weekly change
                    if df_w is not None and not df_w.empty and len(df_w) > 1:
                        week_change = ((df_w['Close'].iloc[-1] / df_w['Close'].iloc[-2]) - 1) * 100
                    else:
                        week_change = 0
                    
                    st.metric(
                        pair, 
                        f"{price:.4f}" if "USD" in pair or "XAU" in pair else f"{price:.2f}",
                        f"{week_change:+.2f}% (Week)"
                    )
                else:
                    st.metric(pair, "N/A", "No data")


def render_macro_table(macro: Dict):
    """Render macro data table."""
    rows = []
    for ccy, vals in macro.items():
        rows.append({
            "Currency": ccy,
            "GDP %": round(vals.get("GDP", 0), 2),
            "Inflation %": round(vals.get("Inflation", 0), 2),
            "Rate %": round(vals.get("Rates", 0), 2),
            "Unemployment": round(vals.get("Unemployment", 0), 2),
        })
    
    if not rows:
        st.warning("No macro data available")
        return
        
    df = pd.DataFrame(rows).set_index("Currency")
    
    try:
        styled_df = df.style \
            .background_gradient(subset=["GDP %"], cmap="RdYlGn") \
            .background_gradient(subset=["Inflation %"], cmap="RdYlGn_r") \
            .background_gradient(subset=["Rate %"], cmap="Blues") \
            .background_gradient(subset=["Unemployment"], cmap="RdYlGn_r")
        st.dataframe(styled_df, use_container_width=True)
    except Exception as e:
        logger.error(f"Error styling dataframe: {e}")
        st.dataframe(df, use_container_width=True)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application for weekly swing trading."""
    st.title("📈 Weekly Swing Trading Pro")
    st.caption("Multi-Week Swing Trading · Weekly/Monthly Analysis · FRED Macro Data")

    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.data_by_timeframe = {}
        st.session_state.macro_data = {}
        st.session_state.swing_ideas = []

    fred_api_key = render_sidebar()

    # Load data
    if not st.session_state.data_loaded:
        with st.spinner("Loading swing trading data..."):
            data_by_timeframe = load_all_timeframes()
            st.session_state.data_by_timeframe = data_by_timeframe
            st.session_state.data_loaded = True

        with st.spinner("Fetching macro fundamentals..."):
            macro = get_macro_data(fred_api_key)
            st.session_state.macro_data = macro
    else:
        data_by_timeframe = st.session_state.data_by_timeframe
        macro = st.session_state.macro_data

    # Render KPIs
    render_swing_kpis(data_by_timeframe)

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🌍 Macro Fundamentals",
        "📈 Swing Charts",
        "🎯 Swing Ideas",
        "📋 Watchlist",
    ])

    # ── Overview Tab ──────────────────────────────────────────────────────
    with tab1:
        st.subheader("Market Overview - Swing Trading")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Swing Trading Strategy")
            st.markdown("""
            **Multi-Week Swing Trading Approach:**
            
            1. **Trend Identification (Weekly)**
               - EMA stack alignment
               - ADX > 25 for strong trends
               - Ichimoku Cloud position
            
            2. **Entry Timing (Daily)**
               - MACD confirmation
               - RSI extremes for reversal
               - Volume confirmation
            
            3. **Risk Management**
               - 1.5% risk per trade
               - Minimum 2.5:1 R:R
               - 3-tier take profit levels
            """)
        
        with col2:
            st.markdown("### 📊 Current Market Conditions")
            weekly_data = data_by_timeframe.get('Weekly', {})
            
            trending_pairs = []
            for pair, df in weekly_data.items():
                if not df.empty and len(df) > 20:
                    df_ind = analyzer.add_indicators(df)
                    if not df_ind.empty:
                        adx = df_ind['ADX'].iloc[-1] if 'ADX' in df_ind.columns else 0
                        if adx > config.adx_trend_min:
                            close = df_ind['Close'].iloc[-1]
                            ema20 = df_ind['EMA_20'].iloc[-1] if 'EMA_20' in df_ind.columns else close
                            trend = "Bullish" if close > ema20 else "Bearish"
                            trending_pairs.append({
                                "Pair": pair,
                                "Trend": trend,
                                "ADX": f"{adx:.1f}",
                                "Strength": "Strong" if adx > 30 else "Moderate"
                            })
            
            if trending_pairs:
                st.dataframe(pd.DataFrame(trending_pairs), use_container_width=True)
            else:
                st.info("No strong trends detected currently")

    # ── Macro Fundamentals Tab ─────────────────────────────────────────────
    with tab2:
        st.subheader("🌍 Macro Fundamentals")
        
        if not fred_api_key:
            st.info("Enter FRED API key in sidebar for live data")
        
        if macro:
            render_macro_table(macro)
        
        st.markdown("### 💡 Macro Impact on Swing Trades")
        st.markdown("""
        - **Rate Differentials**: Drive long-term currency trends
        - **GDP Growth**: Stronger economies attract capital flows
        - **Inflation**: Affects central bank policy expectations
        - **Unemployment**: Key indicator for economic health
        """)

    # ── Swing Charts Tab ──────────────────────────────────────────────────
    with tab3:
        st.subheader("📈 Swing Trading Charts")
        
        available_pairs = [p for p in config.assets.keys() 
                          if p in data_by_timeframe.get('Weekly', {})]
        
        if available_pairs:
            col1, col2 = st.columns(2)
            with col1:
                pair = st.selectbox("Select Pair", available_pairs, key="chart_pair")
            with col2:
                tf = st.selectbox("Timeframe", ["Weekly", "Daily", "4 Hour"], key="chart_tf")
            
            df = data_by_timeframe[tf].get(pair, pd.DataFrame())
            if not df.empty:
                df = analyzer.add_indicators(df)
                
                fig = make_subplots(
                    rows=3, cols=1, shared_xaxes=True,
                    vertical_spacing=0.03, row_heights=[0.5, 0.25, 0.25],
                    subplot_titles=(f'{pair} - {tf}', 'RSI & Volume', 'MACD'),
                )
                
                # Main chart with Ichimoku
                fig.add_trace(go.Candlestick(
                    x=df.index, open=df['Open'], high=df['High'],
                    low=df['Low'], close=df['Close'], name="Price",
                ), row=1, col=1)
                
                # Ichimoku Cloud
                if 'Senkou_Span_A' in df.columns and 'Senkou_Span_B' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df['Senkou_Span_A'],
                        name="Senkou A", line=dict(color='green', width=1),
                        fill=None, mode='lines',
                    ), row=1, col=1)
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df['Senkou_Span_B'],
                        name="Senkou B", line=dict(color='red', width=1),
                        fill='tonexty', mode='lines',
                    ), row=1, col=1)
                
                # EMAs
                for ma, color in [('EMA_20', 'orange'), ('EMA_50', 'blue')]:
                    if ma in df.columns:
                        fig.add_trace(go.Scatter(
                            x=df.index, y=df[ma], name=ma,
                            line=dict(color=color, width=1.5)
                        ), row=1, col=1)
                
                # RSI
                if 'RSI' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df['RSI'], name="RSI",
                        line=dict(color='purple')
                    ), row=2, col=1)
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                
                # MACD
                if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df['MACD'], name="MACD",
                        line=dict(color='blue')
                    ), row=3, col=1)
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df['MACD_Signal'], name="Signal",
                        line=dict(color='red')
                    ), row=3, col=1)
                    
                    # MACD histogram
                    colors = ['green' if val >= 0 else 'red' for val in df['MACD_Histogram']]
                    fig.add_trace(go.Bar(
                        x=df.index, y=df['MACD_Histogram'], name="Histogram",
                        marker_color=colors
                    ), row=3, col=1)
                
                fig.update_layout(height=800, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
                
                # Key levels
                last = df.iloc[-1]
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("RSI", f"{last.get('RSI', 0):.1f}")
                c2.metric("ADX", f"{last.get('ADX', 0):.1f}")
                c3.metric("ATR%", f"{last.get('ATR_Percent', 0):.2f}%")
                c4.metric("Volume Ratio", f"{last.get('Volume_Ratio', 1):.2f}x")
                c5.metric("BB Width", f"{last.get('BB_Width', 0):.3f}")
            else:
                st.warning(f"No data available for {pair}")
        else:
            st.warning("No data available")

    # ── Swing Ideas Tab ───────────────────────────────────────────────────
    with tab4:
        st.subheader("🎯 Weekly Swing Trading Ideas")
        st.caption("Multi-week swing setups based on weekly and daily analysis")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            if st.button("🔄 Generate Swing Ideas", type="primary", key="gen_swing"):
                with st.spinner("Analyzing weekly setups..."):
                    ideas = generate_swing_ideas(data_by_timeframe)
                    st.session_state.swing_ideas = ideas
        
        if st.session_state.swing_ideas:
            ideas = st.session_state.swing_ideas
            st.success(f"✅ Found {len(ideas)} swing trading opportunities")
            
            # Summary metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Setups", len(ideas))
            c2.metric("Long Setups", sum(1 for i in ideas if i["bias"] == "Long"))
            c3.metric("Short Setups", sum(1 for i in ideas if i["bias"] == "Short"))
            c4.metric("High Conviction", sum(1 for i in ideas if i["conviction"] == "High"))
            
            st.divider()
            
            for idx, idea in enumerate(ideas):
                # Color based on bias
                if idea["bias"] == "Long":
                    st.success(f"### {idx+1}. {idea['pair']} - LONG SWING 📈")
                else:
                    st.error(f"### {idx+1}. {idea['pair']} - SHORT SWING 📉")
                
                # Metrics row
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.caption(f"**Conviction:** {idea['conviction']}")
                mc2.caption(f"**Strength:** {idea['strength_score']}/13")
                mc3.caption(f"**Weekly RSI:** {idea['weekly_rsi']:.1f}")
                mc4.caption(f"**Weekly ADX:** {idea['weekly_adx']:.1f}")
                
                st.markdown(f"**📝 Thesis:** {idea['thesis']}")
                
                # Price levels
                st.markdown("**💰 Trade Levels:**")
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Entry", f"{idea['entry']:.5f}" if "USD" in idea['pair'] else f"{idea['entry']:.2f}")
                p2.metric("Stop Loss", f"{idea['stop_loss']:.5f}" if "USD" in idea['pair'] else f"{idea['stop_loss']:.2f}",
                         delta=f"Risk: {idea['risk_percent']:.2f}%")
                
                # Take profit levels
                st.markdown("**🎯 Take Profit Levels:**")
                tp1_col, tp2_col, tp3_col = st.columns(3)
                
                tp1_label = "TP1 ✅" if idea['tp1_valid'] else "TP1 ⚠️"
                tp1_col.metric(
                    tp1_label,
                    f"{idea['tp1']:.5f}" if "USD" in idea['pair'] else f"{idea['tp1']:.2f}",
                    delta=f"R:R 1:{idea['rr1']} ({idea['tp1_method']})"
                )
                
                tp2_label = "TP2 ✅" if idea['tp2_valid'] else "TP2 ⚠️"
                tp2_col.metric(
                    tp2_label,
                    f"{idea['tp2']:.5f}" if "USD" in idea['pair'] else f"{idea['tp2']:.2f}",
                    delta=f"R:R 1:{idea['rr2']} ({idea['tp2_method']})"
                )
                
                tp3_label = "TP3 ✅" if idea['tp3_valid'] else "TP3 ⚠️"
                tp3_col.metric(
                    tp3_label,
                    f"{idea['tp3']:.5f}" if "USD" in idea['pair'] else f"{idea['tp3']:.2f}",
                    delta=f"R:R 1:{idea['rr3']} ({idea['tp3_method']})"
                )
                
                st.caption(f"🛡️ **Stop Method:** {idea['stop_method']} | "
                          f"**Distance:** {idea['stop_pips']} pips | "
                          f"**ATR:** {idea['atr']:.5f}")
                
                with st.expander("📊 Detailed Analysis"):
                    st.json({
                        "Pair": idea['pair'],
                        "Bias": idea['bias'],
                        "Conviction": idea['conviction'],
                        "Entry Price": idea['entry'],
                        "Stop Loss": idea['stop_loss'],
                        "TP1": idea['tp1'],
                        "TP2": idea['tp2'],
                        "TP3": idea['tp3'],
                        "Risk %": f"{idea['risk_percent']:.2f}%",
                        "R:R TP1": idea['rr1'],
                        "R:R TP2": idea['rr2'],
                        "R:R TP3": idea['rr3'],
                        "Weekly RSI": idea['weekly_rsi'],
                        "Weekly ADX": idea['weekly_adx'],
                    })
                
                st.divider()
            
            # Export functionality
            export_df = pd.DataFrame([{
                "Pair": i["pair"],
                "Bias": i["bias"],
                "Conviction": i["conviction"],
                "Entry": i["entry"],
                "Stop Loss": i["stop_loss"],
                "TP1": i["tp1"],
                "TP1 R:R": i["rr1"],
                "TP2": i["tp2"],
                "TP2 R:R": i["rr2"],
                "TP3": i["tp3"],
                "TP3 R:R": i["rr3"],
                "Risk %": f"{i['risk_percent']:.2f}%",
                "Thesis": i["thesis"],
            } for i in ideas])
            
            st.download_button(
                label="📥 Download Swing Ideas (CSV)",
                data=export_df.to_csv(index=False),
                file_name=f"swing_ideas_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
        else:
            st.info("Click 'Generate Swing Ideas' to analyze the market for weekly swing setups")
            st.markdown("""
            **The scanner looks for:**
            - Strong weekly trends (ADX > 25)
            - EMA stack alignment
            - Ichimoku Cloud confirmation
            - Favorable RSI levels
            - Volume confirmation
            - Minimum 2.5:1 R:R setups
            """)

    # ── Watchlist Tab ─────────────────────────────────────────────────────
    with tab5:
        st.subheader("📋 Swing Trading Watchlist")
        
        st.markdown("### 🔍 Pairs to Monitor")
        
        monitor_data = []
        weekly_data = data_by_timeframe.get('Weekly', {})
        daily_data = data_by_timeframe.get('Daily', {})
        
        for pair in config.assets.keys():
            df_w = weekly_data.get(pair)
            df_d = daily_data.get(pair)
            
            if df_w is not None and not df_w.empty and df_d is not None and not df_d.empty:
                df_w_ind = analyzer.add_indicators(df_w)
                df_d_ind = analyzer.add_indicators(df_d)
                
                if not df_w_ind.empty and not df_d_ind.empty:
                    last_w = df_w_ind.iloc[-1]
                    last_d = df_d_ind.iloc[-1]
                    
                    # Determine status
                    adx = last_w.get('ADX', 0)
                    rsi = last_w.get('RSI', 50)
                    
                    if adx > 30:
                        status = "🔥 Strong Trend"
                    elif adx > 25:
                        status = "📈 Trending"
                    elif rsi < 35:
                        status = "👀 Oversold"
                    elif rsi > 65:
                        status = "⚠️ Overbought"
                    else:
                        status = "⏳ Ranging"
                    
                    monitor_data.append({
                        "Pair": pair,
                        "Status": status,
                        "Price": f"{last_d['Close']:.4f}" if "USD" in pair else f"{last_d['Close']:.2f}",
                        "Weekly ADX": f"{adx:.1f}",
                        "Weekly RSI": f"{rsi:.1f}",
                        "ATR%": f"{last_w.get('ATR_Percent', 0):.2f}%",
                    })
        
        if monitor_data:
            df_monitor = pd.DataFrame(monitor_data)
            st.dataframe(df_monitor, use_container_width=True, hide_index=True)
        
        st.markdown("### 📝 Trade Management Notes")
        st.info("""
        **Swing Trade Management Guidelines:**
        
        1. **Entry**: Scale in over 2-3 days
        2. **Stop Management**: Trail stop to breakeven after TP1 hit
        3. **Position Sizing**: Risk max 1.5% per trade
        4. **Hold Time**: 2-8 weeks typical
        5. **Exit**: Take partial profits at TP1, TP2, let remainder run to TP3
        
        **Red Flags (Exit Early):**
        - Weekly ADX drops below 20
        - Price closes beyond stop loss
        - Fundamental shift in macro outlook
        - Volume dries up significantly
        """)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {e}\n{traceback.format_exc()}")
        st.error(f"An error occurred: {str(e)}")
        st.stop()