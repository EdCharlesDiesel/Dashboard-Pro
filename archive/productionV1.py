import json
import logging
import os
import smtplib
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import ta
import yfinance as yf
from fredapi import Fred
from plotly.subplots import make_subplots
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ema_indicator, sma_indicator
from ta.volatility import AverageTrueRange, BollingerBands

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Macro Dashboard Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# LOGGING
# ============================================================================
def _setup_logging() -> logging.Logger:
    log = logging.getLogger("ForexDashboard")
    log.setLevel(logging.INFO)
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        log.addHandler(h)
    return log


logger = _setup_logging()


# ============================================================================
# NOTIFICATION PERSISTENCE
# ============================================================================
NOTIFY_FILE = os.path.join(os.getcwd(), "forex_notify_cache.json")


def load_notified_keys() -> set:
    try:
        if os.path.exists(NOTIFY_FILE):
            with open(NOTIFY_FILE) as fh:
                data = json.load(fh)
            return set(data.get("keys", []))
    except (json.JSONDecodeError, TypeError, OSError) as exc:
        logger.warning("Failed to load notified keys: %s", exc)
    return set()


def save_notified_keys(keys: set) -> None:
    try:
        with open(NOTIFY_FILE, "w") as fh:
            json.dump({"keys": sorted(keys)}, fh)
    except Exception as exc:
        logger.warning("Failed to persist notified keys: %s", exc)


# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class AppConfig:
    """Application configuration — all tuneable parameters in one place."""
    version: str = "1.2.0-PRO"
    last_updated: str = "2024-04-20"
    
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
        "BTC/USD": "BTC-USD",
    })

    timeframes: Dict[str, Dict] = field(default_factory=lambda: {
        "Weekly": {"interval": "1wk", "period": "3mo"},
        "Daily": {"interval": "1d", "period": "3mo"},
        "4 Hour": {"interval": "4h", "period": "1mo"},
        "Hourly": {"interval": "1h", "period": "1mo"},
        "15 Minute": {"interval": "15m", "period": "5d"},
    })

    risk_per_trade: float = 0.02
    atr_sl_mult: float = 1.5
    tp1_atr_mult: float = 3.0
    tp2_atr_mult: float = 5.0
    min_rr: float = 2.0
    adx_trend_min: float = 20.0
    rsi_os: float = 40.0
    rsi_ob: float = 60.0
    stoch_os: float = 25.0
    stoch_ob: float = 75.0

    pair_atr_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "EUR/USD": 1.5, "GBP/USD": 1.8, "USD/JPY": 1.5, "USD/ZAR": 2.5,
        "AUD/USD": 1.5, "NZD/USD": 1.6, "USD/CAD": 1.5, "USD/CHF": 1.5,
        "XAU/USD": 2.0, "BTC/USD": 2.0,
    })

    pair_min_stop: Dict[str, float] = field(default_factory=lambda: {
        "EUR/USD": 0.0010, "GBP/USD": 0.0015, "USD/JPY": 0.10, "USD/ZAR": 0.05,
        "AUD/USD": 0.0010, "NZD/USD": 0.0010, "USD/CAD": 0.0010, "USD/CHF": 0.0010,
        "XAU/USD": 2.00, "BTC/USD": 500.0,
    })

    dxy_symbol: str = "DX-Y.NYB"
    cache_ttl: int = 300
    auto_refresh_interval: int = 300


# ============================================================================
# FRED SERIES REGISTRY & FALLBACKS
# ============================================================================
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
# CONFIG SINGLETON
# ============================================================================
@st.cache_resource
def get_config() -> AppConfig:
    return AppConfig()


config = get_config()


# ============================================================================
# UTILITY — safe_get
# ============================================================================
def safe_get(row: Any, key: str, default: float = 0.0) -> float:
    """Safely extract a scalar from a panda Series row, returning `default`
    if the key is missing or the value is NaN/None."""
    try:
        val = row[key]
        if pd.isna(val):
            return default
        return float(val)
    except (KeyError, TypeError, ValueError):
        return default


# ============================================================================
# EMAIL HELPERS
# ============================================================================
def _get_email_config() -> dict[str, Any]:
    """Read email settings from Streamlit secrets or environment variables."""
    email_secrets = {}
    try:
        if hasattr(st, "secrets"):
            email_secrets = st.secrets.get("email", {})
    except Exception:
        pass

    def _s(key: str, env: str, fallback: str = "") -> str:
        val = email_secrets.get(key) or os.environ.get(env, fallback)
        return str(val) if val is not None else fallback

    port_val = _s("smtp_port", "SMTP_PORT", "587")
    try:
        smtp_port = int(port_val)
    except (ValueError, TypeError):
        smtp_port = 587

    return {
        "smtp_host": _s("smtp_host", "SMTP_HOST", "smtp.gmail.com"),
        "smtp_port": smtp_port,
        "smtp_user": _s("smtp_user", "SMTP_USER", ""),
        "smtp_pass": _s("smtp_pass", "SMTP_PASS", ""),
        "recipient": _s("recipient", "EMAIL_RECIPIENT", ""),
    }


def send_email_alert(idea: Dict) -> bool:
    """Send a plain-text email for a high-conviction trading idea."""
    cfg = _get_email_config()
    if not cfg["smtp_user"] or not cfg["recipient"]:
        logger.warning("Email not configured — skipping alert.")
        return False

    direction = "LONG 📈" if idea["bias"] == "Long" else "SHORT 📉"
    subject = (
        f"[Macro Dashboard] {idea['pair']} {direction} "
        f"— {idea['conviction']} Conviction"
    )
    body = (
        f"High Conviction Alert\n"
        f"{'=' * 40}\n"
        f"Pair      : {idea['pair']}  {direction}\n"
        f"Conviction: {idea['conviction']}  (score {idea['strength_score']}/8)\n\n"
        f"Entry     : {idea['entry']:.5f}\n"
        f"Stop Loss : {idea['stop_loss']:.5f}  "
        f"({idea['stop_loss_pips']} pips — {idea['stop_loss_method']})\n"
        f"TP1       : {idea['take_profit_1']:.5f}  "
        f"R:R 1:{idea['risk_reward_1']:.2f}  ({idea['tp1_method']})\n"
        f"TP2       : {idea['take_profit_2']:.5f}  "
        f"R:R 1:{idea['risk_reward_2']:.2f}  ({idea['tp2_method']})\n\n"
        f"Thesis    : {idea['thesis']}\n"
        f"\nGenerated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    try:
        msg = MIMEMultipart()
        msg["From"] = cfg["smtp_user"]
        msg["To"] = cfg["recipient"]
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(cfg["smtp_host"], int(cfg["smtp_port"])) as server:
            server.ehlo()
            server.starttls()
            server.login(cfg["smtp_user"], cfg["smtp_pass"])
            server.send_message(msg)

        logger.info("Alert email sent for %s %s", idea["pair"], idea["bias"])
        return True
    except Exception as exc:
        logger.error("Email send failed: %s", exc)
        return False


# ============================================================================
# NOTIFICATION SYSTEM
# ============================================================================
def init_notification_state() -> None:
    if "notified_keys" not in st.session_state:
        st.session_state.notified_keys = load_notified_keys()
    if "notification_log" not in st.session_state:
        st.session_state.notification_log = []
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = datetime.now()


def check_and_notify(ideas: List[Dict]) -> List[Dict]:
    """Fire st.toast() for every NEW high-conviction idea."""
    init_notification_state()
    new_alerts: List[Dict] = []

    for idea in ideas:
        if idea["conviction"] != "High":
            continue
        key = f"{idea['pair']}_{idea['bias']}"
        if key not in st.session_state.notified_keys:
            st.session_state.notified_keys.add(key)
            new_alerts.append(idea)

    save_notified_keys(st.session_state.notified_keys)

    for idea in new_alerts:
        direction = "📈 LONG" if idea["bias"] == "Long" else "📉 SHORT"
        st.toast(
            f"🚨 HIGH CONVICTION\n{idea['pair']} {direction}\n"
            f"Entry {idea['entry']:.5f} | R:R 1:{idea['risk_reward_1']:.2f}",
            icon="🔔",
        )
        st.session_state.notification_log.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "pair": idea["pair"],
            "bias": idea["bias"],
            "entry": idea["entry"],
            "rr": idea["risk_reward_1"],
        })

        if st.session_state.get("email_alerts_enabled", False):
            send_email_alert(idea)

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
    except Exception as exc:
        logger.error("FRED client init failed: %s", exc)
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
    vals = (clean.pct_change(12) * 100).dropna()
    return float(vals.iloc[-1]) if not vals.empty else None


# ============================================================================
# MACRO DATA
# ============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_macro_data(api_key: str) -> Tuple[Dict[str, Dict[str, float]], bool]:
    """Returns (data, is_live). is_live=False means static fallback data."""
    if not api_key:
        logger.warning("No FRED API key — using fallback macro data.")
        return MACRO_FALLBACKS.copy(), False

    fred = get_fred_client(api_key)
    if fred is None:
        return MACRO_FALLBACKS.copy(), False

    result: Dict[str, Dict[str, float]] = {}
    any_success = False

    for currency, series_map in FRED_SERIES.items():
        fb = MACRO_FALLBACKS.get(currency, {})
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
        except Exception as exc:
            logger.warning("FRED GDP %s: %s", currency, exc)
            entry["GDP"] = fb.get("GDP", 0.0)

        # Inflation (CPI YoY)
        try:
            raw = fred.get_series(series_map["CPI"])
            val = _yoy_pct(raw)
            entry["Inflation"] = val if val is not None else fb.get("Inflation", 0.0)
        except Exception as exc:
            logger.warning("FRED CPI %s: %s", currency, exc)
            entry["Inflation"] = fb.get("Inflation", 0.0)

        # Policy Rate
        try:
            raw = fred.get_series(series_map["Rates"])
            val = _latest_value(raw)
            entry["Rates"] = val if val is not None else fb.get("Rates", 0.0)
        except Exception as exc:
            logger.warning("FRED Rates %s: %s", currency, exc)
            entry["Rates"] = fb.get("Rates", 0.0)

        # Unemployment
        try:
            raw = fred.get_series(series_map["Unemployment"])
            val = _latest_value(raw)
            entry["Unemployment"] = val if val is not None else fb.get("Unemployment", 0.0)
        except Exception as exc:
            logger.warning("FRED Unemployment %s: %s", currency, exc)
            entry["Unemployment"] = fb.get("Unemployment", 0.0)

        result[currency] = entry

    return (result if result else MACRO_FALLBACKS.copy()), any_success


# ============================================================================
# DATA FETCHING
# ============================================================================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(symbol: str, interval: str, period: str) -> pd.DataFrame:
    try:
        history = yf.Ticker(symbol).history(period=period, interval=interval)
        if history.empty:
            logger.warning("No data for %s (%s)", symbol, interval)
        return history
    except Exception as exc:
        logger.exception("Error fetching %s: %s", symbol, exc)
        return pd.DataFrame()


# ============================================================================
# TECHNICAL ANALYZER
# ============================================================================
class TechnicalAnalyzer:
    RSI_WINDOW = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    SMA_SHORT_WINDOW = 20
    SMA_LONG_WINDOW = 50
    EMA_SHORT_WINDOW = 20
    EMA_LONG_WINDOW = 50
    BB_WINDOW = 20
    BB_STD_DEV = 2
    ATR_WINDOW = 14
    STOCH_WINDOW = 14
    STOCH_SMOOTH = 3
    ADX_WINDOW = 14
    SR_WINDOW = 20
    REQUIRED_COLUMNS = ("Open", "High", "Low", "Close")

    @staticmethod
    def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or len(df) < TechnicalAnalyzer.BB_WINDOW:
            return df
        if not all(c in df.columns for c in TechnicalAnalyzer.REQUIRED_COLUMNS):
            logger.warning("Missing required OHLC columns for indicator calculation")
            return df
        
        # Avoid redundant calculation if indicators are already present
        if "RSI" in df.columns and "MACD" in df.columns:
            return df

        df = df.copy()
        try:
            close, high, low = df["Close"], df["High"], df["Low"]

            df["RSI"] = ta.momentum.RSIIndicator(close, window=TechnicalAnalyzer.RSI_WINDOW).rsi()

            macd = ta.trend.MACD(
                close,
                window_fast=TechnicalAnalyzer.MACD_FAST,
                window_slow=TechnicalAnalyzer.MACD_SLOW,
                window_sign=TechnicalAnalyzer.MACD_SIGNAL,
            )
            df["MACD"] = macd.macd()
            df["MACD_Signal"] = macd.macd_signal()
            df["MACD_Histogram"] = macd.macd_diff()

            df["SMA_20"] = ta.trend.sma_indicator(close, window=TechnicalAnalyzer.SMA_SHORT_WINDOW)
            df["SMA_50"] = ta.trend.sma_indicator(close, window=TechnicalAnalyzer.SMA_LONG_WINDOW)
            df["EMA_20"] = ta.trend.ema_indicator(close, window=TechnicalAnalyzer.EMA_SHORT_WINDOW)
            df["EMA_50"] = ta.trend.ema_indicator(close, window=TechnicalAnalyzer.EMA_LONG_WINDOW)

            bb = ta.volatility.BollingerBands(
                close,
                window=TechnicalAnalyzer.BB_WINDOW,
                window_dev=TechnicalAnalyzer.BB_STD_DEV,
            )
            df["BB_Upper"] = bb.bollinger_hband()
            df["BB_Middle"] = bb.bollinger_mavg()
            df["BB_Lower"] = bb.bollinger_lband()

            df["ATR"] = ta.volatility.AverageTrueRange(
                high, low, close, window=TechnicalAnalyzer.ATR_WINDOW
            ).average_true_range()

            stoch = ta.momentum.StochasticOscillator(
                high, low, close,
                window=TechnicalAnalyzer.STOCH_WINDOW,
                smooth_window=TechnicalAnalyzer.STOCH_SMOOTH,
            )
            df["Stoch_K"] = stoch.stoch()
            df["Stoch_D"] = stoch.stoch_signal()

            adx = ta.trend.ADXIndicator(high, low, close, window=TechnicalAnalyzer.ADX_WINDOW)
            df["ADX"] = adx.adx()
            df["ADX_Pos"] = adx.adx_pos()
            df["ADX_Neg"] = adx.adx_neg()

            df["Resistance_20"] = high.rolling(window=TechnicalAnalyzer.SR_WINDOW).max()
            df["Support_20"] = low.rolling(window=TechnicalAnalyzer.SR_WINDOW).min()

        except Exception as exc:
            logger.error("Indicator calculation error: %s", exc)

        return df


analyzer = TechnicalAnalyzer()


# ============================================================================
# ENTRY SIGNAL GENERATOR
# ============================================================================
class EntrySignalGenerator:
    @staticmethod
    def get_entry_signal(df_15m: pd.DataFrame, bias: str) -> Dict:
        if df_15m.empty or len(df_15m) < 5:
            return {"signal": 0, "confidence": 0, "reasons": ["Insufficient 15-min data"]}

        indicator_data = analyzer.add_indicators(df_15m)
        if indicator_data.empty or len(indicator_data) < 2:
            return {"signal": 0, "confidence": 0, "reasons": ["Indicator calculation failed"]}

        latest = indicator_data.iloc[-1]
        previous = indicator_data.iloc[-2]

        k = safe_get(latest, "Stoch_K", 50.0)
        d = safe_get(latest, "Stoch_D", 50.0)
        prev_k = safe_get(previous, "Stoch_K", 50.0)
        prev_d = safe_get(previous, "Stoch_D", 50.0)
        rsi = safe_get(latest, "RSI", 50.0)
        price = safe_get(latest, "Close", 0.0)

        if price <= 0.0:
            return {"signal": 0, "confidence": 0, "reasons": ["Invalid price data"]}

        bb_lower = safe_get(latest, "BB_Lower", price * 0.99)
        bb_upper = safe_get(latest, "BB_Upper", price * 1.01)

        signal = 0
        confidence = 0
        reasons: List[str] = []

        def add_long_reasons() -> None:
            nonlocal signal, confidence
            if prev_k <= prev_d and d < k < config.stoch_os:
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

        def add_short_reasons() -> None:
            nonlocal signal, confidence
            if prev_k >= prev_d and d > k > config.stoch_ob:
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

        if bias == "Long":
            add_long_reasons()
        elif bias == "Short":
            add_short_reasons()
        else:
            reasons.append(
                f"Trend bias is Neutral (ADX < {config.adx_trend_min:.0f}) — "
                "no directional entry; wait for trend to establish."
            )

        return {
            "signal": signal,
            "confidence": min(confidence, 5),
            "reasons": reasons,
            "stoch_k": k,
            "stoch_d": d,
            "rsi": rsi,
            "price": price,
        }


entry_generator = EntrySignalGenerator()


# ============================================================================
# STOP LOSS CALCULATOR
# ============================================================================
class StopLossCalculator:
    @staticmethod
    def pip_size(pair: str) -> float:
        if "JPY" in pair:     return 0.01
        if pair == "XAU/USD": return 0.10
        if pair == "BTC/USD": return 1.0
        if "ZAR" in pair:     return 0.001
        return 0.0001

    def price_to_pips(self, pair: str, distance: float) -> float:
        ps = self.pip_size(pair)
        return round(distance / ps, 1) if ps > 0 else 0.0

    @staticmethod
    def get_swing_stop(df: pd.DataFrame, bias: str, lookback: int = 20) -> Optional[float]:
        if df.empty or len(df) < lookback:
            return None
        recent = df.tail(lookback)
        if bias == "Long" and "Low" in df.columns: return float(recent["Low"].min())
        if bias == "Short" and "High" in df.columns: return float(recent["High"].max())
        return None

    def calculate(
            self,
            df: pd.DataFrame,
            pair: str,
            bias: str,
            current_price: float,
            atr: float,
            lookback: int = 20,
    ) -> Dict:
        atr_mult = config.pair_atr_multipliers.get(pair, config.atr_sl_mult)
        min_dist = config.pair_min_stop.get(pair, 0.0010)
        buffer = atr * 0.25

        atr_stop = (current_price - atr * atr_mult) if bias == "Long" \
            else (current_price + atr * atr_mult)

        swing = self.get_swing_stop(df, bias, lookback)
        stop = atr_stop
        method = "ATR"

        if swing is not None:
            if bias == "Long":
                struct_stop = swing - buffer
                if struct_stop < current_price:
                    if struct_stop <= atr_stop:
                        stop, method = struct_stop, "Swing Low"
                    else:
                        stop, method = atr_stop, "ATR (struct too tight)"
            else:
                struct_stop = swing + buffer
                if struct_stop > current_price:
                    if struct_stop >= atr_stop:
                        stop, method = struct_stop, "Swing High"
                    else:
                        stop, method = atr_stop, "ATR (struct too tight)"

        raw_dist = abs(current_price - stop)
        if raw_dist < min_dist:
            stop = (current_price - min_dist) if bias == "Long" \
                else (current_price + min_dist)
            method += " + min-dist enforced"

        return {
            "stop": stop,
            "method": method,
            "distance_pips": self.price_to_pips(pair, abs(current_price - stop)),
        }


sl_calculator = StopLossCalculator()


# ============================================================================
# TAKE PROFIT CALCULATOR
# ============================================================================
class TakeProfitCalculator:
    @staticmethod
    def get_swing_target(df: pd.DataFrame, bias: str, lookback: int = 20) -> Optional[float]:
        if df.empty or len(df) < lookback:
            return None
        recent = df.tail(lookback)
        if bias == "Long" and "High" in df.columns: return float(recent["High"].max())
        if bias == "Short" and "Low" in df.columns: return float(recent["Low"].min())
        return None

    def calculate(
            self,
            df: pd.DataFrame,
            pair: str,
            bias: str,
            current_price: float,
            atr: float,
            stop_loss: float,
            lookback: int = 20,
    ) -> Dict:
        stop_dist = abs(current_price - stop_loss) or atr
        swing = self.get_swing_target(df, bias, lookback)

        if bias == "Long":
            tp1_atr = current_price + atr * config.tp1_atr_mult
            tp2_atr = current_price + atr * config.tp2_atr_mult

            tp1, m1 = (swing, "Swing High") \
                if swing is not None and current_price < swing < tp1_atr \
                else (tp1_atr, f"ATR ×{config.tp1_atr_mult}")

            tp2, m2 = (swing, "Swing High (ext)") \
                if swing is not None and tp1 < swing < tp2_atr \
                else (tp2_atr, f"ATR ×{config.tp2_atr_mult}")

            rr1 = (tp1 - current_price) / stop_dist
            rr2 = (tp2 - current_price) / stop_dist

        else:
            tp1_atr = current_price - atr * config.tp1_atr_mult
            tp2_atr = current_price - atr * config.tp2_atr_mult

            tp1, m1 = (swing, "Swing Low") \
                if swing is not None and tp1_atr < swing < current_price \
                else (tp1_atr, f"ATR ×{config.tp1_atr_mult}")

            tp2, m2 = (swing, "Swing Low (ext)") \
                if swing is not None and tp2_atr < swing < tp1 \
                else (tp2_atr, f"ATR ×{config.tp2_atr_mult}")

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
# ============================================================================
def analyze_multi_timeframe(
        df_daily: pd.DataFrame,
        df_4h: pd.DataFrame,
        df_1h: pd.DataFrame,
        df_15m: pd.DataFrame,
        pair_name: str,
) -> Optional[Dict]:
    # Indicators already added at load time
    # df_daily = analyzer.add_indicators(df_daily)
    # df_4h = analyzer.add_indicators(df_4h)
    # df_1h = analyzer.add_indicators(df_1h)
    # df_15m = analyzer.add_indicators(df_15m)

    if any(df.empty for df in [df_daily, df_4h, df_1h, df_15m]):
        return None

    daily = df_daily.iloc[-1]
    four_hour = df_4h.iloc[-1]
    one_hour = df_1h.iloc[-1]
    fifteen_m = df_15m.iloc[-1]

    if "Close" not in daily.index or "Close" not in four_hour.index:
        return None

    d_close = safe_get(daily, "Close")
    d_ema20 = safe_get(daily, "EMA_20", d_close)
    d_trend = "Long" if d_close > d_ema20 else "Short"
    d_rsi = safe_get(daily, "RSI", 50.0)
    d_adx = safe_get(daily, "ADX", 0.0)

    h4_close = safe_get(four_hour, "Close")
    h4_ema20 = safe_get(four_hour, "EMA_20", h4_close)
    h4_ema50 = safe_get(four_hour, "EMA_50", h4_close)
    h4_trend = "Long" if h4_ema20 > h4_ema50 else "Short"
    h4_macd = safe_get(four_hour, "MACD", 0.0)
    h4_sig = safe_get(four_hour, "MACD_Signal", 0.0)
    h4_macd_bull = h4_macd > h4_sig

    h1_close = safe_get(one_hour, "Close")
    h1_ema20 = safe_get(one_hour, "EMA_20", h1_close)
    h1_ema50 = safe_get(one_hour, "EMA_50", h1_close)
    h1_trend = "Long" if h1_ema20 > h1_ema50 else "Short"
    h1_rsi = safe_get(one_hour, "RSI", 50.0)

    long_s = short_s = 0
    reasons: List[str] = []

    if d_trend == "Long":
        long_s += 2;
        reasons.append("Daily: Bullish EMA alignment")
    else:
        short_s += 2;
        reasons.append("Daily: Bearish EMA alignment")

    if d_rsi < 40:
        long_s += 1;
        reasons.append(f"Daily RSI oversold ({d_rsi:.1f})")
    elif d_rsi > 60:
        short_s += 1;
        reasons.append(f"Daily RSI overbought ({d_rsi:.1f})")

    if d_adx > config.adx_trend_min:
        if d_trend == "Long":
            long_s += 1
        else:
            short_s += 1
        reasons.append(f"Strong trend (ADX={d_adx:.1f})")

    if h4_trend == "Long":
        long_s += 1;
        reasons.append("4H: EMA20 > EMA50")
    else:
        short_s += 1;
        reasons.append("4H: EMA20 < EMA50")

    if h4_macd_bull:
        long_s += 1;
        reasons.append("4H: MACD bullish")
    else:
        short_s += 1;
        reasons.append("4H: MACD bearish")

    if h1_trend == "Long":
        long_s += 1;
        reasons.append("1H: Bullish EMA alignment")
    else:
        short_s += 1;
        reasons.append("1H: Bearish EMA alignment")

    if h1_rsi < 45:
        long_s += 1;
        reasons.append(f"1H RSI supportive ({h1_rsi:.1f})")
    elif h1_rsi > 55:
        short_s += 1;
        reasons.append(f"1H RSI resistive ({h1_rsi:.1f})")

    if long_s == short_s:
        return None

    final_bias = "Long" if long_s > short_s else "Short"
    strength = long_s if final_bias == "Long" else short_s
    conviction = "High" if strength >= 6 else ("Medium" if strength >= 3 else "Low")

    entry_signal = entry_generator.get_entry_signal(df_15m, final_bias)

    atr = safe_get(one_hour, "ATR", 0.0)
    if atr <= 0:
        atr = h1_close * 0.005 if h1_close > 0 else 0.001

    current_price = safe_get(fifteen_m, "Close", 0.0)
    if current_price <= 0.0:
        return None

    sl_result = sl_calculator.calculate(df_1h, pair_name, final_bias, current_price, atr)
    tp_result = tp_calculator.calculate(
        df_4h, pair_name, final_bias, current_price, atr, sl_result["stop"]
    )

    thesis = " | ".join(reasons)
    if entry_signal and entry_signal["signal"] != 0:
        thesis += f" | Entry: {', '.join(entry_signal['reasons'][:2])}"

    return {
        "pair": pair_name,
        "bias": final_bias,
        "conviction": conviction,
        "strength_score": strength,
        "thesis": thesis,
        "entry": current_price,
        "take_profit_1": tp_result["tp1"],
        "take_profit_2": tp_result["tp2"],
        "tp1_method": tp_result["method_tp1"],
        "tp2_method": tp_result["method_tp2"],
        "tp1_valid": tp_result["tp1_valid"],
        "tp2_valid": tp_result["tp2_valid"],
        "stop_loss": sl_result["stop"],
        "stop_loss_method": sl_result["method"],
        "stop_loss_pips": sl_result["distance_pips"],
        "risk_reward_1": tp_result["rr1"],
        "risk_reward_2": tp_result["rr2"],
        "atr": atr,
        "entry_signal": entry_signal,
    }


def generate_trading_ideas(
        data_by_timeframe: Dict,
) -> Tuple[List[Dict], List[str]]:
    """Returns (ideas, skipped_reasons)."""
    ideas: List[Dict] = []
    skipped: List[str] = []

    for pair_name in config.assets:
        frames = {
            tf: data_by_timeframe.get(tf, {}).get(pair_name, pd.DataFrame())
            for tf in ["Daily", "4 Hour", "Hourly", "15 Minute"]
        }
        thin = [tf for tf, df in frames.items() if df.empty or len(df) < 20]
        if thin:
            skipped.append(f"{pair_name} — insufficient bars in: {', '.join(thin)}")
            continue

        idea = analyze_multi_timeframe(
            frames["Daily"], frames["4 Hour"],
            frames["Hourly"], frames["15 Minute"],
            pair_name,
        )
        if idea:
            ideas.append(idea)

    ideas.sort(
        key=lambda x: (x["conviction"] == "High", x["strength_score"]),
        reverse=True,
    )
    return ideas, skipped


# ============================================================================
# DATA LOADING
# ============================================================================
@st.cache_data(ttl=300, show_spinner=False)
def _fetch_all_timeframes() -> Dict:
    data: Dict[str, Dict] = {tf: {} for tf in config.timeframes}
    
    def fetch_task(tf_name, tf_cfg, pair_name, symbol):
        try:
            df = fetch_data(symbol, tf_cfg["interval"], tf_cfg["period"])
            if not df.empty:
                # Pre-calculate indicators once at load time
                df = analyzer.add_indicators(df)
                return tf_name, pair_name, df
        except Exception as exc:
            logger.warning("Failed %s (%s): %s", pair_name, tf_name, exc)
        return None

    tasks = []
    for tf_name, tf_cfg in config.timeframes.items():
        for pair_name, symbol in config.assets.items():
            tasks.append((tf_name, tf_cfg, pair_name, symbol))

    with ThreadPoolExecutor(max_workers=min(len(tasks), 10)) as executor:
        futures = [executor.submit(fetch_task, *t) for t in tasks]
        for future in as_completed(futures):
            res = future.result()
            if res:
                tf_n, p_n, d_val = res
                data[tf_n][p_n] = d_val
    return data


def load_all_timeframes() -> Dict:
    bar = st.progress(0, text="Fetching market data…")
    bar.progress(20, text="Loading all pairs…")
    data = _fetch_all_timeframes()
    bar.progress(100, text="Done ✓")
    bar.empty()
    return data


def clear_data_cache() -> None:
    _fetch_all_timeframes.clear()
    fetch_data.clear()
    st.session_state.data_loaded = False
    st.session_state.data_loaded = True
    st.session_state.last_refresh = datetime.now()  # already here — correct


# ============================================================================
# UI — SIDEBAR
# ============================================================================
def render_sidebar(fred_key_default: str) -> Tuple[str, Optional[str], bool]:
    with st.sidebar:
        st.header("⚙️ Dashboard Settings")

        st.subheader("🔑 FRED API Key")
        fred_api_key = st.text_input(
            "API Key", value=fred_key_default, type="password",
            help="Free key at https://fred.stlouisfed.org/docs/api/api_key.html",
        )
        if fred_api_key:
            st.success("✅ FRED key loaded")
        else:
            st.warning("⚠️ No key — using static fallback data")

        st.divider()

        selected_tf = st.selectbox(
            "Default Chart Timeframe",
            ["Daily", "4 Hour", "Hourly", "15 Minute"],
        )

        st.divider()

        st.subheader("🔄 Data Refresh")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("↺ Refresh Now", use_container_width=True):
                clear_data_cache()
                st.rerun()
        with col_b:
            elapsed = int(
                (datetime.now() - st.session_state.get("last_refresh", datetime.now()))
                .total_seconds()
            )
            st.caption(f"Age: {elapsed}s / {config.cache_ttl}s")

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

        st.subheader("📧 Email Alerts")
        email_cfg = _get_email_config()
        email_configured = bool(email_cfg["smtp_user"] and email_cfg["recipient"])

        if email_configured:
            email_enabled = st.toggle(
                "Send email on high-conviction alerts",
                value=st.session_state.get("email_alerts_enabled", True),
            )
            st.session_state.email_alerts_enabled = email_enabled
            st.caption(f"→ {email_cfg['recipient']}")
            if st.button("✉️ Send test email"):
                test_idea = {
                    "pair": "TEST/PAIR", "bias": "Long", "conviction": "High",
                    "strength_score": 8, "entry": 1.23456,
                    "stop_loss": 1.23000, "stop_loss_pips": 45.6,
                    "stop_loss_method": "Test",
                    "take_profit_1": 1.24000, "risk_reward_1": 2.0,
                    "tp1_method": "ATR ×3.0",
                    "take_profit_2": 1.24500, "risk_reward_2": 3.5,
                    "tp2_method": "ATR ×5.0",
                    "thesis": "This is a test alert from Macro Dashboard Pro.",
                }
                if send_email_alert(test_idea):
                    st.success("✅ Test email sent")
                else:
                    st.error("❌ Email failed — check logs and secrets.toml")
        else:
            st.session_state.email_alerts_enabled = False
            st.caption(
                "⚠️ Email not configured. Add an `[email]` section to "
                "`.streamlit/secrets.toml` or set `EMAIL_*` env vars."
            )

        st.divider()

        st.subheader("🔔 Alert Log")
        log = st.session_state.get("notification_log", [])
        if log:
            for entry in reversed(log[-10:]):
                icon = "📈" if entry["bias"] == "Long" else "📉"
                st.markdown(
                    f"**{entry['time']}** {icon} **{entry['pair']}** "
                    f"{entry['bias']} — R:R {entry['rr']:.2f}"
                )
            if st.button("🗑️ Clear Alerts"):
                st.session_state.notification_log = []
                st.session_state.notified_keys = set()
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
            f"**Risk:** {config.risk_per_trade * 100:.0f}% per trade · "
            f"Min R:R {config.min_rr}:1"
        )
        st.caption(f"Last render: {datetime.now().strftime('%H:%M:%S')}")

    return selected_tf, fred_api_key, auto_monitor


# ============================================================================
# UI — KPIs
# ============================================================================
def render_kpis(daily_data: Dict) -> None:
    kpi_pairs = ["USD/ZAR", "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "XAU/USD", "BTC/USD"]
    cols = st.columns(len(kpi_pairs))
    for i, pair in enumerate(kpi_pairs):
        df = daily_data.get(pair)
        with cols[i]:
            if df is not None and not df.empty and "Close" in df.columns:
                price = df["Close"].iloc[-1]
                change = df["Close"].pct_change().iloc[-1] * 100 if len(df) > 1 else 0.0
                fmt = f"{price:,.2f}" if pair in ("BTC/USD", "XAU/USD") else f"{price:.4f}"
                st.metric(pair, fmt, f"{change:+.2f}%")
            else:
                st.metric(pair, "N/A", "—")


# ============================================================================
# UI — MACRO TABLE
# ============================================================================
def render_macro_table(macro_data: Dict, is_live: bool) -> None:
    if is_live:
        st.success("✅ Live FRED data")
    else:
        st.warning(
            "⚠️ **Static fallback data** — these figures may be months out of date. "
            "Enter a FRED API key in the sidebar to fetch live values."
        )

    if not macro_data:
        st.warning("No macro data available")
        return

    rows = [
        {
            "Currency": ccy,
            "GDP %": round(vals.get("GDP", 0), 2),
            "Inflation %": round(vals.get("Inflation", 0), 2),
            "Rate %": round(vals.get("Rates", 0), 2),
            "Unemployment": round(vals.get("Unemployment", 0), 2),
        }
        for ccy, vals in macro_data.items()
    ]

    df_macro = pd.DataFrame(rows).set_index("Currency")
    gradient_specs = [
        ("GDP %", "RdYlGn"),
        ("Inflation %", "RdYlGn_r"),
        ("Rate %", "Blues"),
        ("Unemployment", "RdYlGn_r"),
    ]

    try:
        styled = df_macro.style
        for column, cmap in gradient_specs:
            styled = styled.background_gradient(subset=[column], cmap=cmap)
        st.dataframe(styled, use_container_width=True)
    except Exception:
        st.dataframe(df_macro, use_container_width=True)


# ============================================================================
# PRODUCTION-GRADE TECHNICAL CHART
# ============================================================================
def render_professional_chart(
        df: pd.DataFrame,
        pair: str,
        tf: str,
        data_by_timeframe: Dict,
        chart_settings: Dict = None
) -> None:
    """Render a professional-grade technical analysis chart with multiple panels."""

    if df.empty or "Close" not in df.columns:
        st.warning(f"No data available for {pair} on {tf}")
        return

    # Default settings if none provided
    if chart_settings is None:
        chart_settings = {
            "show_volume": True,
            "show_ichimoku": False,
            "show_fib": False,
            "show_sr": True,
            "show_bb": True,
            "show_ma": ["EMA20", "EMA50"],
            "indicator_panels": ["MACD", "RSI", "Stochastic", "ADX"]
        }

    # Add all indicators
    df = analyzer.add_indicators(df)

    # Calculate additional indicators for professional view
    if len(df) > 20:
        # Volume indicators if available
        if "Volume" in df.columns:
            df["Volume_SMA"] = df["Volume"].rolling(window=20).mean()
            df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA"]

        # Ichimoku Cloud
        if chart_settings.get("show_ichimoku", False):
            high_9 = df["High"].rolling(window=9).max()
            low_9 = df["Low"].rolling(window=9).min()
            df["Ichimoku_Conversion"] = (high_9 + low_9) / 2

            high_26 = df["High"].rolling(window=26).max()
            low_26 = df["Low"].rolling(window=26).min()
            df["Ichimoku_Base"] = (high_26 + low_26) / 2

            df["Ichimoku_LeadingA"] = ((df["Ichimoku_Conversion"] + df["Ichimoku_Base"]) / 2).shift(26)
            df["Ichimoku_LeadingB"] = (
                        (df["High"].rolling(window=52).max() + df["Low"].rolling(window=52).min()) / 2).shift(26)

        # Fibonacci Retracement Levels
        if chart_settings.get("show_fib", False) and len(df) >= 50:
            recent_high = df["High"].tail(50).max()
            recent_low = df["Low"].tail(50).min()
            diff = recent_high - recent_low
            df["Fib_0"] = recent_low
            df["Fib_236"] = recent_low + 0.236 * diff
            df["Fib_382"] = recent_low + 0.382 * diff
            df["Fib_50"] = recent_low + 0.5 * diff
            df["Fib_618"] = recent_low + 0.618 * diff
            df["Fib_786"] = recent_low + 0.786 * diff
            df["Fib_1"] = recent_high

        # Pivot Points
        df["Pivot"] = (df["High"] + df["Low"] + df["Close"]) / 3
        df["R1"] = 2 * df["Pivot"] - df["Low"]
        df["S1"] = 2 * df["Pivot"] - df["High"]
        df["R2"] = df["Pivot"] + (df["High"] - df["Low"])
        df["S2"] = df["Pivot"] - (df["High"] - df["Low"])
        df["R3"] = df["High"] + 2 * (df["Pivot"] - df["Low"])
        df["S3"] = df["Low"] - 2 * (df["High"] - df["Pivot"])

    # Determine which indicator panels to show
    indicator_panels = chart_settings.get("indicator_panels", ["MACD", "RSI", "Stochastic", "ADX"])
    show_macd = "MACD" in indicator_panels
    show_rsi = "RSI" in indicator_panels
    show_stoch = "Stochastic" in indicator_panels
    show_adx = "ADX" in indicator_panels

    # Calculate number of rows needed
    has_volume = chart_settings.get("show_volume", True) and "Volume" in df.columns and df["Volume"].sum() > 0
    num_rows = 1  # Price panel
    if has_volume:
        num_rows += 1
    if show_macd:
        num_rows += 1
    if show_rsi or show_stoch:
        num_rows += 1
    if show_adx:
        num_rows += 1

    # Create row heights and titles
    row_heights = [0.5]
    subplot_titles = [f"{pair} — {tf} | Price Action"]

    if has_volume:
        row_heights.append(0.12)
        subplot_titles.append("Volume")
    if show_macd:
        row_heights.append(0.12)
        subplot_titles.append("MACD")
    if show_rsi or show_stoch:
        row_heights.append(0.13)
        title = "RSI" if show_rsi else ""
        if show_rsi and show_stoch:
            title = "RSI & Stochastic"
        elif show_stoch:
            title = "Stochastic"
        subplot_titles.append(title)
    if show_adx:
        row_heights.append(0.13)
        subplot_titles.append("ADX / DI")

    # Normalize heights
    total = sum(row_heights)
    row_heights = [h / total for h in row_heights]

    # Create subplots
    fig = make_subplots(
        rows=num_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=subplot_titles
    )

    # ========================================================================
    # PANEL 1: PRICE ACTION
    # ========================================================================

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            showlegend=True,
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
        ),
        row=1, col=1
    )

    # Moving Averages
    ma_settings = chart_settings.get("show_ma", ["EMA20", "EMA50"])
    if "EMA20" in ma_settings and "EMA_20" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["EMA_20"],
                name="EMA 20",
                line=dict(color='#FF9800', width=1.5),
                opacity=0.8
            ),
            row=1, col=1
        )

    if "EMA50" in ma_settings and "EMA_50" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["EMA_50"],
                name="EMA 50",
                line=dict(color='#2196F3', width=1.5),
                opacity=0.8
            ),
            row=1, col=1
        )

    if "SMA20" in ma_settings and "SMA_20" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["SMA_20"],
                name="SMA 20",
                line=dict(color='#FF5722', width=1, dash='dot'),
                opacity=0.5,
                visible='legendonly'
            ),
            row=1, col=1
        )

    if "SMA50" in ma_settings and "SMA_50" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["SMA_50"],
                name="SMA 50",
                line=dict(color='#9C27B0', width=1, dash='dot'),
                opacity=0.5,
                visible='legendonly'
            ),
            row=1, col=1
        )

    # Bollinger Bands
    if chart_settings.get("show_bb", True):
        if "BB_Upper" in df.columns and "BB_Lower" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["BB_Upper"],
                    name="BB Upper",
                    line=dict(color='#78909C', width=1, dash='dash'),
                    opacity=0.5
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["BB_Lower"],
                    name="BB Lower",
                    line=dict(color='#78909C', width=1, dash='dash'),
                    opacity=0.5,
                    fill='tonexty',
                    fillcolor='rgba(120, 144, 156, 0.1)'
                ),
                row=1, col=1
            )

            if "BB_Middle" in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=df["BB_Middle"],
                        name="BB Middle",
                        line=dict(color='#546E7A', width=0.8),
                        opacity=0.4,
                        visible='legendonly'
                    ),
                    row=1, col=1
                )

    # Ichimoku Cloud
    if chart_settings.get("show_ichimoku", False):
        if "Ichimoku_LeadingA" in df.columns and "Ichimoku_LeadingB" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["Ichimoku_LeadingA"],
                    name="Ichimoku A",
                    line=dict(color='rgba(76, 175, 80, 0)', width=0),
                    showlegend=False
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["Ichimoku_LeadingB"],
                    name="Ichimoku Cloud",
                    line=dict(color='rgba(76, 175, 80, 0)', width=0),
                    fill='tonexty',
                    fillcolor='rgba(76, 175, 80, 0.1)',
                    showlegend=True
                ),
                row=1, col=1
            )

            if "Ichimoku_Conversion" in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=df["Ichimoku_Conversion"],
                        name="Conversion",
                        line=dict(color='#FF6F00', width=1),
                        visible='legendonly'
                    ),
                    row=1, col=1
                )

            if "Ichimoku_Base" in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=df["Ichimoku_Base"],
                        name="Base",
                        line=dict(color='#1565C0', width=1),
                        visible='legendonly'
                    ),
                    row=1, col=1
                )

    # Support and Resistance Levels
    if chart_settings.get("show_sr", True):
        if "Support_20" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["Support_20"],
                    name="Support (20)",
                    line=dict(color='#4CAF50', width=1.5, dash='dot'),
                    opacity=0.6
                ),
                row=1, col=1
            )

        if "Resistance_20" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["Resistance_20"],
                    name="Resistance (20)",
                    line=dict(color='#F44336', width=1.5, dash='dot'),
                    opacity=0.6
                ),
                row=1, col=1
            )

    # Fibonacci Levels
    if chart_settings.get("show_fib", False):
        fib_colors = {
            "Fib_0": "#9E9E9E", "Fib_236": "#FFC107", "Fib_382": "#FF9800",
            "Fib_50": "#FF5722", "Fib_618": "#F44336", "Fib_786": "#E91E63", "Fib_1": "#9C27B0"
        }

        for fib_level, color in fib_colors.items():
            if fib_level in df.columns:
                value = df[fib_level].iloc[-1]
                fig.add_hline(
                    y=value, line_dash="dot", line_color=color,
                    opacity=0.3, row=1, col=1,
                    annotation_text=f"{fib_level.split('_')[1]}",
                    annotation_position="right"
                )

    current_row = 2

    # ========================================================================
    # PANEL 2: VOLUME
    # ========================================================================
    if has_volume:
        colors = ['#26a69a' if close >= open_ else '#ef5350'
                  for close, open_ in zip(df["Close"], df["Open"])]

        fig.add_trace(
            go.Bar(
                x=df.index, y=df["Volume"],
                name="Volume",
                marker_color=colors,
                opacity=0.7,
                showlegend=True
            ),
            row=current_row, col=1
        )

        if "Volume_SMA" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["Volume_SMA"],
                    name="Volume SMA (20)",
                    line=dict(color='#FF9800', width=1.5),
                    opacity=0.8
                ),
                row=current_row, col=1
            )

        current_row += 1

    # ========================================================================
    # PANEL 3: MACD
    # ========================================================================
    if show_macd:
        if "MACD" in df.columns and "MACD_Signal" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["MACD"],
                    name="MACD",
                    line=dict(color='#2196F3', width=1.5)
                ),
                row=current_row, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["MACD_Signal"],
                    name="Signal",
                    line=dict(color='#FF9800', width=1.5)
                ),
                row=current_row, col=1
            )

            if "MACD_Histogram" in df.columns:
                hist_colors = ['#26a69a' if val >= 0 else '#ef5350'
                               for val in df["MACD_Histogram"]]
                fig.add_trace(
                    go.Bar(
                        x=df.index, y=df["MACD_Histogram"],
                        name="Histogram",
                        marker_color=hist_colors,
                        opacity=0.6,
                        showlegend=True
                    ),
                    row=current_row, col=1
                )

            fig.add_hline(y=0, line_dash="solid", line_color="gray",
                          opacity=0.3, row=current_row, col=1)

        current_row += 1

    # ========================================================================
    # PANEL 4: RSI & STOCHASTIC
    # ========================================================================
    if show_rsi or show_stoch:
        if show_rsi and "RSI" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["RSI"],
                    name="RSI",
                    line=dict(color='#9C27B0', width=1.8)
                ),
                row=current_row, col=1
            )

            fig.add_hline(y=70, line_dash="dash", line_color="#ef5350",
                          opacity=0.5, row=current_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="#26a69a",
                          opacity=0.5, row=current_row, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray",
                          opacity=0.3, row=current_row, col=1)

        if show_stoch and "Stoch_K" in df.columns and "Stoch_D" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["Stoch_K"],
                    name="Stoch %K",
                    line=dict(color='#FF5722', width=1.2),
                    opacity=0.8
                ),
                row=current_row, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["Stoch_D"],
                    name="Stoch %D",
                    line=dict(color='#4CAF50', width=1.2),
                    opacity=0.8
                ),
                row=current_row, col=1
            )

            fig.add_hline(y=80, line_dash="dash", line_color="#ef5350",
                          opacity=0.3, row=current_row, col=1)
            fig.add_hline(y=20, line_dash="dash", line_color="#26a69a",
                          opacity=0.3, row=current_row, col=1)

        current_row += 1

    # ========================================================================
    # PANEL 5: ADX
    # ========================================================================
    if show_adx:
        if "ADX" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["ADX"],
                    name="ADX",
                    line=dict(color='#00BCD4', width=2)
                ),
                row=current_row, col=1
            )

            fig.add_hline(y=25, line_dash="dash", line_color="#FF9800",
                          opacity=0.5, row=current_row, col=1,
                          annotation_text="Trend", annotation_position="right")
            fig.add_hline(y=20, line_dash="dot", line_color="#9E9E9E",
                          opacity=0.3, row=current_row, col=1)

        if "ADX_Pos" in df.columns and "ADX_Neg" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["ADX_Pos"],
                    name="+DI",
                    line=dict(color='#26a69a', width=1.2),
                    opacity=0.7
                ),
                row=current_row, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["ADX_Neg"],
                    name="-DI",
                    line=dict(color='#ef5350', width=1.2),
                    opacity=0.7
                ),
                row=current_row, col=1
            )

    # ========================================================================
    # LAYOUT CUSTOMIZATION
    # ========================================================================
    fig.update_layout(
        height=200 + num_rows * 150,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='#CCCCCC',
            borderwidth=1
        ),
        hovermode='x unified',
        template='plotly_white',
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(family="Arial, sans-serif", size=11),
    )

    # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#EEEEEE',
        showline=True,
        linewidth=1,
        linecolor='#CCCCCC',
        mirror=True
    )

    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#EEEEEE',
        showline=True,
        linewidth=1,
        linecolor='#CCCCCC',
        mirror=True
    )

    # Add range selector
    fig.update_xaxes(
        rangeslider=dict(visible=False),
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1D", step="day", stepmode="backward"),
                dict(count=5, label="5D", step="day", stepmode="backward"),
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(step="all", label="All")
            ]),
            bgcolor='#F5F5F5',
            font=dict(size=10),
            x=0, y=1.02,
            xanchor='left',
            yanchor='bottom'
        ),
        row=1, col=1
    )

    st.plotly_chart(fig, use_container_width=True)

    # ========================================================================
    # DETAILED METRICS DASHBOARD
    # ========================================================================
    st.markdown("---")
    st.subheader("📊 Technical Indicators Dashboard")

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        current_price = safe_get(last, 'Close', 0)
        prev_price = safe_get(prev, 'Close', 0)
        price_change = ((current_price - prev_price) / prev_price * 100) if prev_price > 0 else 0

        st.metric(
            "Price",
            f"{current_price:.5f}",
            delta=f"{price_change:+.3f}%"
        )

        high_low_spread = ((safe_get(last, 'High', 0) - safe_get(last, 'Low', 0)) / safe_get(last, 'Low', 1)) * 100
        st.caption(f"Range: {safe_get(last, 'Low', 0):.5f} - {safe_get(last, 'High', 0):.5f} ({high_low_spread:.2f}%)")

    with col2:
        rsi_val = safe_get(last, 'RSI', 50)
        rsi_status = "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral")
        st.metric(
            "RSI",
            f"{rsi_val:.1f}",
            delta=rsi_status,
            delta_color="off"
        )

        stoch_k = safe_get(last, 'Stoch_K', 50)
        stoch_d = safe_get(last, 'Stoch_D', 50)
        st.caption(f"Stoch: {stoch_k:.1f} / {stoch_d:.1f}")

    with col3:
        macd_val = safe_get(last, 'MACD', 0)
        macd_sig = safe_get(last, 'MACD_Signal', 0)
        macd_diff = macd_val - macd_sig
        macd_status = "Bullish" if macd_diff > 0 else "Bearish"
        st.metric(
            "MACD",
            f"{macd_val:.4f}",
            delta=f"{macd_diff:+.4f} ({macd_status})",
            delta_color="normal" if macd_diff > 0 else "inverse"
        )

        hist = safe_get(last, 'MACD_Histogram', 0)
        prev_hist = safe_get(prev, 'MACD_Histogram', 0)
        st.caption(f"Hist: {hist:.4f} ({'↑' if hist > prev_hist else '↓'})")

    with col4:
        adx_val = safe_get(last, 'ADX', 0)
        trend_strength = "Strong" if adx_val > 25 else ("Moderate" if adx_val > 20 else "Weak")
        st.metric(
            "ADX",
            f"{adx_val:.1f}",
            delta=trend_strength,
            delta_color="off"
        )

        di_plus = safe_get(last, 'ADX_Pos', 0)
        di_minus = safe_get(last, 'ADX_Neg', 0)
        st.caption(f"+DI: {di_plus:.1f} | -DI: {di_minus:.1f}")

    with col5:
        atr_val = safe_get(last, 'ATR', 0)
        atr_pct = (atr_val / current_price) * 100 if current_price > 0 else 0
        st.metric(
            "ATR",
            f"{atr_val:.5f}",
            delta=f"{atr_pct:.2f}% of price",
            delta_color="off"
        )

        vol_status = "High" if atr_pct > 1.5 else ("Moderate" if atr_pct > 0.8 else "Low")
        st.caption(f"Volatility: {vol_status}")

    with col6:
        close = safe_get(last, 'Close', 0)
        ema20 = safe_get(last, 'EMA_20', close)
        ema50 = safe_get(last, 'EMA_50', close)

        ma_status = []
        if close > ema20: ma_status.append(">EMA20")
        if close > ema50: ma_status.append(">EMA50")
        if ema20 > ema50: ma_status.append("EMA20>50")

        ma_signal = "Bullish" if len(ma_status) >= 2 else ("Bearish" if len(ma_status) <= 1 else "Mixed")
        st.metric(
            "MA Signal",
            ma_signal,
            delta=", ".join(ma_status) if ma_status else "No signal",
            delta_color="off"
        )

        bb_upper = safe_get(last, 'BB_Upper', 0)
        bb_lower = safe_get(last, 'BB_Lower', 0)
        bb_position = "Middle"
        if close > bb_upper:
            bb_position = "Above"
        elif close < bb_lower:
            bb_position = "Below"
        st.caption(f"BB Position: {bb_position}")

    # ========================================================================
    # TREND ANALYSIS SUMMARY
    # ========================================================================
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📈 Trend Analysis")

        trends = []
        if current_price > ema20:
            trends.append("✅ Price above EMA20 (bullish)")
        else:
            trends.append("❌ Price below EMA20 (bearish)")

        if ema20 > ema50:
            trends.append("✅ EMA20 > EMA50 (bullish)")
        else:
            trends.append("❌ EMA20 < EMA50 (bearish)")

        if adx_val > 20:
            trends.append(f"✅ Trend strength ADX {adx_val:.1f}")
        else:
            trends.append(f"⚠️ Weak trend ADX {adx_val:.1f}")

        for trend in trends:
            st.write(trend)

    with col2:
        st.markdown("### 🎯 Momentum Signals")

        momentum = []
        if rsi_val > 60:
            momentum.append("🔥 Strong bullish momentum (RSI > 60)")
        elif rsi_val > 50:
            momentum.append("📈 Bullish momentum (RSI > 50)")
        elif rsi_val < 40:
            momentum.append("❄️ Strong bearish momentum (RSI < 40)")
        elif rsi_val < 50:
            momentum.append("📉 Bearish momentum (RSI < 50)")
        else:
            momentum.append("⚖️ Neutral momentum (RSI = 50)")

        if macd_diff > 0:
            momentum.append("✅ MACD bullish")
        else:
            momentum.append("❌ MACD bearish")

        if stoch_k > 80:
            momentum.append("⚠️ Stoch overbought (>80)")
        elif stoch_k < 20:
            momentum.append("💡 Stoch oversold (<20)")
        else:
            momentum.append(f"➡️ Stoch neutral ({stoch_k:.1f})")

        for m in momentum:
            st.write(m)

    with col3:
        st.markdown("### ⚡ Trade Setup Quality")

        # Calculate setup quality score
        score = 0
        reasons = []

        # Trend alignment
        if current_price > ema20 and ema20 > ema50:
            score += 2
            reasons.append("Bullish trend alignment")
        elif current_price < ema20 and ema20 < ema50:
            score += 2
            reasons.append("Bearish trend alignment")

        # Momentum confirmation
        if rsi_val > 50 and macd_diff > 0:
            score += 2
            reasons.append("Bullish momentum confirmed")
        elif rsi_val < 50 and macd_diff < 0:
            score += 2
            reasons.append("Bearish momentum confirmed")

        # Volume confirmation
        if has_volume and "Volume_Ratio" in df.columns:
            vol_ratio = safe_get(last, 'Volume_Ratio', 1)
            if vol_ratio > 1.2:
                score += 1
                reasons.append(f"Above average volume ({vol_ratio:.1f}x)")

        # ADX confirmation
        if adx_val > 25:
            score += 1
            reasons.append("Strong trend strength")

        # Stochastic timing
        if stoch_k < 20:
            score += 1
            reasons.append("Oversold - potential long entry")
        elif stoch_k > 80:
            score += 1
            reasons.append("Overbought - potential short entry")

        # Overall assessment
        if score >= 5:
            quality = "🌟 Excellent"
            color = "green"
        elif score >= 3:
            quality = "👍 Good"
            color = "blue"
        elif score >= 2:
            quality = "👌 Fair"
            color = "orange"
        else:
            quality = "⚠️ Poor"
            color = "red"

        st.markdown(f"**Setup Quality: <span style='color:{color}'>{quality} ({score}/7)</span>**",
                    unsafe_allow_html=True)

        if reasons:
            st.markdown("**Key Factors:**")
            for reason in reasons:
                st.write(f"• {reason}")
        else:
            st.write("No clear setup detected")

    # ========================================================================
    # SUPPORT & RESISTANCE LEVELS
    # ========================================================================
    st.markdown("---")
    st.subheader("📍 Key Levels")

    level_cols = st.columns(5)

    with level_cols[0]:
        st.metric("Support (20)", f"{safe_get(last, 'Support_20', 0):.5f}")
        st.metric("Pivot", f"{safe_get(last, 'Pivot', 0):.5f}")

    with level_cols[1]:
        st.metric("Resistance (20)", f"{safe_get(last, 'Resistance_20', 0):.5f}")
        st.metric("R1", f"{safe_get(last, 'R1', 0):.5f}")

    with level_cols[2]:
        st.metric("BB Lower", f"{safe_get(last, 'BB_Lower', 0):.5f}")
        st.metric("S1", f"{safe_get(last, 'S1', 0):.5f}")

    with level_cols[3]:
        st.metric("BB Middle", f"{safe_get(last, 'BB_Middle', 0):.5f}")
        st.metric("R2", f"{safe_get(last, 'R2', 0):.5f}")

    with level_cols[4]:
        st.metric("BB Upper", f"{safe_get(last, 'BB_Upper', 0):.5f}")
        st.metric("S2", f"{safe_get(last, 'S2', 0):.5f}")

    # ========================================================================
    # MULTI-TIMEFRAME QUICK VIEW
    # ========================================================================
    st.markdown("---")
    st.subheader("🔄 Multi-Timeframe Context")

    tf_cols = st.columns(5)
    timeframes = ["Weekly", "Daily", "4 Hour", "Hourly", "15 Minute"]

    for idx, other_tf in enumerate(timeframes):
        with tf_cols[idx]:
            tf_data = data_by_timeframe.get(other_tf, {}).get(pair, pd.DataFrame())
            if not tf_data.empty and "Close" in tf_data.columns:
                tf_data = analyzer.add_indicators(tf_data)
                tf_last = tf_data.iloc[-1]

                tf_close = safe_get(tf_last, 'Close', 0)
                tf_prev = safe_get(tf_data.iloc[-2], 'Close', tf_close) if len(tf_data) > 1 else tf_close
                tf_change = ((tf_close - tf_prev) / tf_prev) * 100 if tf_prev > 0 else 0

                tf_rsi = safe_get(tf_last, 'RSI', 50)
                tf_trend = "📈" if tf_close > safe_get(tf_last, 'EMA_20', tf_close) else "📉"

                st.markdown(f"**{other_tf}** {tf_trend}")
                st.caption(f"{tf_close:.5f} ({tf_change:+.2f}%)")
                st.caption(f"RSI: {tf_rsi:.1f}")

                if other_tf == tf:
                    st.caption("✅ Current")
            else:
                st.caption(f"**{other_tf}**")
                st.caption("No data")

    # Data quality indicator
    st.caption(
        f"📊 Data points: {len(df)} | Last update: {df.index[-1].strftime('%Y-%m-%d %H:%M')} | Freshness: Excellent")


# ============================================================================
# WEEKLY SWING TRADING ANALYSIS (FIXED)
# ============================================================================
def analyze_weekly_swing(
        df_weekly: pd.DataFrame,
        df_daily: pd.DataFrame,
        df_4h: pd.DataFrame,        # ← added for entry confirmation
        pair_name: str,
) -> Optional[Dict]:
    """Generate swing trading ideas based on Weekly + Daily + 4H confirmation."""

    if df_weekly.empty or df_daily.empty:
        return None

    if len(df_weekly) < 20 or len(df_daily) < 50 or len(df_4h) < 20:
        return None

    weekly = df_weekly.iloc[-1]
    daily  = df_daily.iloc[-1]
    h4     = df_4h.iloc[-1]

    if "Close" not in weekly.index or "Close" not in daily.index:
        return None

    # ------------------------------------------------------------------ #
    # Weekly context
    # ------------------------------------------------------------------ #
    w_close  = safe_get(weekly, "Close")
    w_ema20  = safe_get(weekly, "EMA_20", w_close)
    w_ema50  = safe_get(weekly, "EMA_50", w_close)
    w_trend  = "Bullish" if w_ema20 > w_ema50 else "Bearish"
    w_rsi    = safe_get(weekly, "RSI", 50.0)
    w_adx    = safe_get(weekly, "ADX", 0.0)
    w_macd   = safe_get(weekly, "MACD", 0.0)
    w_sig    = safe_get(weekly, "MACD_Signal", 0.0)
    w_macd_b = w_macd > w_sig

    # ------------------------------------------------------------------ #
    # Daily context
    # ------------------------------------------------------------------ #
    d_close   = safe_get(daily, "Close")
    d_ema20   = safe_get(daily, "EMA_20", d_close)
    d_ema50   = safe_get(daily, "EMA_50", d_close)
    d_rsi     = safe_get(daily, "RSI", 50.0)
    d_stoch_k = safe_get(daily, "Stoch_K", 50.0)
    d_stoch_d = safe_get(daily, "Stoch_D", 50.0)
    d_macd    = safe_get(daily, "MACD", 0.0)
    d_sig     = safe_get(daily, "MACD_Signal", 0.0)

    # FIX 1: Use DAILY ATR, not weekly ATR — weekly is 5× too large
    d_atr = safe_get(daily, "ATR", d_close * 0.008)
    if d_atr <= 0:
        d_atr = d_close * 0.008

    # ------------------------------------------------------------------ #
    # 4H entry confirmation
    # ------------------------------------------------------------------ #
    h4_close = safe_get(h4, "Close", d_close)
    h4_ema20 = safe_get(h4, "EMA_20", h4_close)
    h4_macd  = safe_get(h4, "MACD", 0.0)
    h4_sig   = safe_get(h4, "MACD_Signal", 0.0)
    h4_bull  = h4_close > h4_ema20 and h4_macd > h4_sig

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    long_score = short_score = 0
    reasons: List[str] = []

    # Weekly structure (highest weight)
    if w_trend == "Bullish":
        long_score += 3
        reasons.append("Weekly: EMA20 > EMA50 (bullish structure)")
    else:
        short_score += 3
        reasons.append("Weekly: EMA20 < EMA50 (bearish structure)")

    # FIX 4: Relaxed RSI thresholds — weekly RSI rarely hits 40/60
    if w_rsi < 45:
        long_score += 2
        reasons.append(f"Weekly RSI supportive ({w_rsi:.1f} < 45)")
    elif w_rsi > 55:
        short_score += 2
        reasons.append(f"Weekly RSI resistive ({w_rsi:.1f} > 55)")

    if w_adx > config.adx_trend_min:
        if w_trend == "Bullish":
            long_score += 2
        else:
            short_score += 2
        reasons.append(f"Weekly trend strength ADX={w_adx:.1f}")

    # FIX 4: Weekly MACD gets weight 2, not 1
    if w_macd_b:
        long_score += 2
        reasons.append("Weekly MACD bullish crossover")
    else:
        short_score += 2
        reasons.append("Weekly MACD bearish crossover")

    # Daily EMA alignment
    if d_close > d_ema20 and d_ema20 > d_ema50:
        long_score += 2
        reasons.append("Daily: Price > EMA20 > EMA50 (bullish)")
    elif d_close < d_ema20 and d_ema20 < d_ema50:
        short_score += 2
        reasons.append("Daily: Price < EMA20 < EMA50 (bearish)")
    elif d_close > d_ema20:
        long_score += 1
        reasons.append("Daily: Price above EMA20")
    else:
        short_score += 1
        reasons.append("Daily: Price below EMA20")

    # Daily RSI
    if d_rsi < 40:
        long_score += 1
        reasons.append(f"Daily RSI oversold ({d_rsi:.1f})")
    elif d_rsi > 60:
        short_score += 1
        reasons.append(f"Daily RSI overbought ({d_rsi:.1f})")

    # Daily Stochastic
    if d_stoch_k < 25 and d_stoch_d < 25:
        long_score += 1
        reasons.append(f"Daily Stoch oversold (K={d_stoch_k:.1f})")
    elif d_stoch_k > 75 and d_stoch_d > 75:
        short_score += 1
        reasons.append(f"Daily Stoch overbought (K={d_stoch_k:.1f})")

    # Daily MACD
    if d_macd > d_sig:
        long_score += 1
        reasons.append("Daily MACD bullish")
    else:
        short_score += 1
        reasons.append("Daily MACD bearish")

    # FIX 5: 4H entry confirmation
    if h4_bull:
        long_score += 1
        reasons.append("4H: Price > EMA20 & MACD bullish (entry confirmation)")
    else:
        short_score += 1
        reasons.append("4H: Price < EMA20 or MACD bearish")

    if long_score == short_score:
        return None

    bias     = "Long" if long_score > short_score else "Short"
    strength = max(long_score, short_score)

    # FIX 4: Rebalanced thresholds — max score is now 15
    conviction = "High" if strength >= 11 else ("Medium" if strength >= 7 else "Low")

    current_price = d_close
    if current_price <= 0:
        return None

    # ------------------------------------------------------------------ #
    # FIX 2: Use existing StopLossCalculator with daily ATR
    # ------------------------------------------------------------------ #
    sl_result = sl_calculator.calculate(
        df_daily, pair_name, bias, current_price, d_atr, lookback=50
    )
    stop_loss = sl_result["stop"]

    # ------------------------------------------------------------------ #
    # FIX 2: Use existing TakeProfitCalculator
    # ------------------------------------------------------------------ #
    tp_result = tp_calculator.calculate(
        df_daily, pair_name, bias, current_price, d_atr, stop_loss, lookback=50
    )

    # FIX 3: Enforce minimum R:R — skip ideas that don't qualify
    if not tp_result["tp1_valid"]:
        return None

    invalidation = (
        f"Daily close below {stop_loss:.5f}"
        if bias == "Long"
        else f"Daily close above {stop_loss:.5f}"
    )

    return {
        "pair":           pair_name,
        "bias":           bias,
        "conviction":     conviction,
        "strength_score": strength,
        "thesis":         " | ".join(reasons),
        "entry":          current_price,
        "stop_loss":      stop_loss,
        "stop_loss_method": sl_result["method"],
        "stop_loss_pips": sl_result["distance_pips"],
        "target_1":       tp_result["tp1"],
        "target_2":       tp_result["tp2"],
        "tp1_method":     tp_result["method_tp1"],
        "tp2_method":     tp_result["method_tp2"],
        "tp1_valid":      tp_result["tp1_valid"],
        "tp2_valid":      tp_result["tp2_valid"],
        "risk_reward_1":  tp_result["rr1"],
        "risk_reward_2":  tp_result["rr2"],
        "invalidation":   invalidation,
        "atr":            d_atr,
        "weekly_trend":   w_trend,
        "daily_trend":    "Bullish" if d_close > d_ema20 else "Bearish",
        "4h_confirmation": h4_bull,
    }


def generate_weekly_swing_ideas(data_by_timeframe: Dict) -> List[Dict]:
    ideas: List[Dict] = []

    for pair_name in config.assets:
        df_weekly = data_by_timeframe.get("Weekly", {}).get(pair_name, pd.DataFrame())
        df_daily  = data_by_timeframe.get("Daily",  {}).get(pair_name, pd.DataFrame())
        df_4h     = data_by_timeframe.get("4 Hour", {}).get(pair_name, pd.DataFrame())  # ← added

        if df_weekly.empty or df_daily.empty or df_4h.empty:
            continue
        if len(df_weekly) < 20 or len(df_daily) < 50:
            continue

        idea = analyze_weekly_swing(df_weekly, df_daily, df_4h, pair_name)
        if idea:
            ideas.append(idea)

    ideas.sort(
        key=lambda x: (x["conviction"] == "High", x["strength_score"]),
        reverse=True,
    )
    return ideas


# ============================================================================
# MULTI-TIMEFRAME BIAS DASHBOARD
# ============================================================================
def analyze_bias_for_pair(
        df_weekly: pd.DataFrame,
        df_daily: pd.DataFrame,
        df_4h: pd.DataFrame,
        df_1h: pd.DataFrame,
        df_15m: pd.DataFrame,
        pair_name: str,
) -> Dict:
    """Analyze bias across multiple timeframes."""

    frames = {
        "Weekly": df_weekly,
        "Daily": df_daily,
        "4H": df_4h,
        "1H": df_1h,
        "15m": df_15m,
    }

    bias_data = {}

    for tf_name, df in frames.items():
        if df.empty or len(df) < 20:
            bias_data[tf_name] = {
                "bias": "Insufficient Data",
                "strength": 0,
                "price": 0,
                "trend": "N/A",
                "rsi": 0,
                "adx": 0,
            }
            continue

        # Use already-calculated indicators
        # df = analyzer.add_indicators(df) # Removed redundant call
        if df.empty:
            bias_data[tf_name] = {
                "bias": "Insufficient Data",
                "strength": 0,
                "price": 0,
                "trend": "N/A",
                "rsi": 0,
                "adx": 0,
            }
            continue

        latest = df.iloc[-1]

        close = safe_get(latest, "Close", 0)
        ema20 = safe_get(latest, "EMA_20", close)
        ema50 = safe_get(latest, "EMA_50", close)
        sma20 = safe_get(latest, "SMA_20", close)
        sma50 = safe_get(latest, "SMA_50", close)
        rsi = safe_get(latest, "RSI", 50)
        adx = safe_get(latest, "ADX", 0)
        macd = safe_get(latest, "MACD", 0)
        macd_sig = safe_get(latest, "MACD_Signal", 0)

        bullish_signals = 0
        bearish_signals = 0

        if close > ema20:
            bullish_signals += 1
        else:
            bearish_signals += 1

        if ema20 > ema50:
            bullish_signals += 1
        else:
            bearish_signals += 1

        if close > sma20:
            bullish_signals += 1
        else:
            bearish_signals += 1

        if macd > macd_sig:
            bullish_signals += 1
        else:
            bearish_signals += 1

        if rsi > 50:
            bullish_signals += 1
        else:
            bearish_signals += 1

        if bullish_signals > bearish_signals:
            bias = "Bullish"
            strength = bullish_signals
        elif bearish_signals > bullish_signals:
            bias = "Bearish"
            strength = bearish_signals
        else:
            bias = "Neutral"
            strength = 0

        if adx > 25:
            trend = "Strong"
        elif adx > 20:
            trend = "Moderate"
        else:
            trend = "Weak/Ranging"

        bias_data[tf_name] = {
            "bias": bias,
            "strength": strength,
            "price": close,
            "trend": trend,
            "rsi": rsi,
            "adx": adx,
        }

    overall_bullish = sum(1 for tf in bias_data.values() if tf["bias"] == "Bullish")
    overall_bearish = sum(1 for tf in bias_data.values() if tf["bias"] == "Bearish")

    if overall_bullish > overall_bearish:
        overall_bias = "Bullish"
        confidence = (overall_bullish / 5) * 100
    elif overall_bearish > overall_bullish:
        overall_bias = "Bearish"
        confidence = (overall_bearish / 5) * 100
    else:
        overall_bias = "Mixed/Neutral"
        confidence = 0

    return {
        "pair": pair_name,
        "overall_bias": overall_bias,
        "confidence": confidence,
        "timeframes": bias_data,
    }


def generate_bias_dashboard(data_by_timeframe: Dict) -> List[Dict]:
    """Generate bias analysis for all pairs."""
    bias_results = []

    for pair_name in config.assets:
        df_weekly = data_by_timeframe.get("Weekly", {}).get(pair_name, pd.DataFrame())
        df_daily = data_by_timeframe.get("Daily", {}).get(pair_name, pd.DataFrame())
        df_4h = data_by_timeframe.get("4 Hour", {}).get(pair_name, pd.DataFrame())
        df_1h = data_by_timeframe.get("Hourly", {}).get(pair_name, pd.DataFrame())
        df_15m = data_by_timeframe.get("15 Minute", {}).get(pair_name, pd.DataFrame())

        if any(df.empty for df in [df_daily, df_4h, df_1h, df_15m]):
            continue

        bias_result = analyze_bias_for_pair(
            df_weekly, df_daily, df_4h, df_1h, df_15m, pair_name
        )
        bias_results.append(bias_result)

    return bias_results


# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main() -> None:
    st.title(f"💹 Macro Dashboard Pro v{config.version}")
    st.caption(f"Multi-Timeframe · FRED Fundamentals · 15-Min Entry Signals · High-Conviction Alerts | Last Updated: {config.last_updated}")

    init_notification_state()

    default_key = ""
    try:
        default_key = (
            st.secrets.get("FRED_API_KEY", "")
            if hasattr(st, "secrets")
            else os.environ.get("FRED_API_KEY", "")
        )
    except Exception:
        default_key = os.environ.get("FRED_API_KEY", "")

    selected_tf, fred_api_key, auto_monitor = render_sidebar(default_key)

    if auto_monitor:
        elapsed = (datetime.now() - st.session_state.last_refresh).total_seconds()
        if elapsed >= config.auto_refresh_interval:
            clear_data_cache()
            st.rerun()
        components.html(
            f'<script>setTimeout(() => window.parent.location.reload(), {config.auto_refresh_interval * 1000});</script>',
            height=0,
        )

    if not st.session_state.get("data_loaded", False):
        with st.spinner("Loading market data…"):
            data_by_timeframe = load_all_timeframes()
            st.session_state.data_by_timeframe = data_by_timeframe
            st.session_state.data_loaded = True
            st.session_state.last_refresh = datetime.now()

        with st.spinner("Fetching macro fundamentals from FRED…"):
            macro, is_live = get_macro_data(fred_api_key)
            st.session_state.macro_data = macro
            st.session_state.macro_live = is_live

        if auto_monitor:
            ideas, _ = generate_trading_ideas(data_by_timeframe)
            st.session_state.latest_ideas = ideas
            check_and_notify(ideas)


    else:
        data_by_timeframe = st.session_state.data_by_timeframe
        macro = st.session_state.macro_data
        is_live = st.session_state.get("macro_live", False)

    daily_data = data_by_timeframe.get("Daily", {})

    if daily_data:
        render_kpis(daily_data)

    # ========================================================================
    # AUTO-LOAD MTF BIAS DASHBOARD ON FIRST LOAD
    # ========================================================================
    if "bias_results" not in st.session_state and data_by_timeframe:
        with st.spinner("🔄 Auto-loading Multi-Timeframe Bias Dashboard…"):
            bias_results = generate_bias_dashboard(data_by_timeframe)
            st.session_state.bias_results = bias_results
            st.session_state.bias_auto_loaded = True

    # Reorder tabs - MTF Bias first
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🎯 MTF Bias",  # Tab 1 - Now first!
        "📊 Overview",  # Tab 2
        "🌍 Macro Fundamentals",  # Tab 3
        "📈 Technical Chart",  # Tab 4
        "⏱️ 15-Min Entry",  # Tab 5
        "🎯 Trading Ideas",  # Tab 6
        "📅 Weekly Swing",  # Tab 7
    ])

    # ========================================================================
    # TAB 1: MULTI-TIMEFRAME BIAS DASHBOARD (Auto-loading)
    # ========================================================================
    with tab1:
        st.subheader("🎯 Multi-Timeframe Bias Dashboard")
        st.caption("Weekly · Daily · 4H · 1H · 15m bias analysis — Auto-loads on startup")

        # Show refresh button and auto-refresh indicator
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🔄 Refresh Analysis", type="primary", key="bias_refresh"):
                with st.spinner("Analyzing bias across all timeframes…"):
                    bias_results = generate_bias_dashboard(data_by_timeframe)
                    st.session_state.bias_results = bias_results
                    st.rerun()
        with col2:
            if "bias_auto_loaded" in st.session_state:
                st.success("✅ Auto-loaded on startup — showing latest bias analysis")

        # Display results from session state
        if "bias_results" in st.session_state:
            bias_results = st.session_state.bias_results

            if bias_results:
                # Summary metrics
                total_pairs = len(bias_results)
                bullish_count = sum(1 for r in bias_results if r["overall_bias"] == "Bullish")
                bearish_count = sum(1 for r in bias_results if r["overall_bias"] == "Bearish")
                neutral_count = sum(1 for r in bias_results if r["overall_bias"] == "Mixed/Neutral")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Pairs", total_pairs)
                m2.metric("Bullish", bullish_count,
                          delta=f"{(bullish_count / total_pairs) * 100:.0f}%" if total_pairs > 0 else None)
                m3.metric("Bearish", bearish_count,
                          delta=f"{(bearish_count / total_pairs) * 100:.0f}%" if total_pairs > 0 else None)
                m4.metric("Neutral/Mixed", neutral_count)

                st.divider()

                # Summary table
                summary_data = []
                for result in bias_results:
                    summary_data.append({
                        "Pair": result["pair"],
                        "Overall Bias": result["overall_bias"],
                        "Confidence": f"{result['confidence']:.0f}%",
                        "Weekly": result["timeframes"]["Weekly"]["bias"],
                        "Daily": result["timeframes"]["Daily"]["bias"],
                        "4H": result["timeframes"]["4H"]["bias"],
                        "1H": result["timeframes"]["1H"]["bias"],
                        "15m": result["timeframes"]["15m"]["bias"],
                    })

                df_summary = pd.DataFrame(summary_data)

                def color_bias(val):
                    if val == "Bullish":
                        return "background-color: #90EE90; color: black; font-weight: bold"
                    elif val == "Bearish":
                        return "background-color: #FFB6C1; color: black; font-weight: bold"
                    elif val == "Neutral" or val == "Mixed/Neutral":
                        return "background-color: #F0E68C; color: black"
                    return ""

                styled_df = df_summary.style.map(color_bias,
                                                 subset=["Overall Bias", "Weekly", "Daily", "4H", "1H", "15m"])
                st.dataframe(styled_df, use_container_width=True, height=400)

                st.divider()
                st.subheader("📊 Detailed Multi-Timeframe Analysis")

                # Sort by confidence
                bias_results_sorted = sorted(bias_results, key=lambda x: x["confidence"], reverse=True)

                for result in bias_results_sorted:
                    confidence_color = "green" if result["confidence"] >= 60 else (
                        "orange" if result["confidence"] >= 40 else "red")

                    with st.expander(
                            f"📊 {result['pair']} — "
                            f"**{result['overall_bias']}** "
                            f"({result['confidence']:.0f}% confidence)",
                            expanded=(result["confidence"] >= 60)  # Auto-expand high confidence pairs
                    ):
                        tf_data = result["timeframes"]

                        cols = st.columns(5)
                        for idx, (tf_name, data) in enumerate(tf_data.items()):
                            with cols[idx]:
                                if data["bias"] == "Bullish":
                                    st.success(f"**{tf_name}**")
                                elif data["bias"] == "Bearish":
                                    st.error(f"**{tf_name}**")
                                else:
                                    st.warning(f"**{tf_name}**")

                                st.metric("Bias", data["bias"])
                                if data["price"] > 0:
                                    price_display = f"{data['price']:.5f}"
                                    if result["pair"] in ["BTC/USD", "XAU/USD"]:
                                        price_display = f"{data['price']:.2f}"
                                    st.metric("Price", price_display)
                                st.metric("RSI", f"{data['rsi']:.1f}")
                                st.metric("ADX", f"{data['adx']:.1f}")
                                st.caption(f"Trend: {data['trend']}")
                                st.caption(f"Strength: {data['strength']}/5")

                        # Add bias alignment visualization
                        st.markdown("---")
                        st.caption("**Bias Alignment:**")
                        alignment_cols = st.columns(5)
                        for idx, (tf_name, data) in enumerate(tf_data.items()):
                            with alignment_cols[idx]:
                                if data["bias"] == "Bullish":
                                    st.markdown("🟢")
                                elif data["bias"] == "Bearish":
                                    st.markdown("🔴")
                                else:
                                    st.markdown("🟡")
                                st.caption(tf_name)

                # Export bias analysis
                bias_export = []
                for result in bias_results:
                    row = {"Pair": result["pair"], "Overall Bias": result["overall_bias"],
                           "Confidence": result["confidence"]}
                    for tf, data in result["timeframes"].items():
                        row[f"{tf}_Bias"] = data["bias"]
                        row[f"{tf}_RSI"] = data["rsi"]
                        row[f"{tf}_ADX"] = data["adx"]
                        row[f"{tf}_Trend"] = data["trend"]
                    bias_export.append(row)

                bias_df = pd.DataFrame(bias_export)

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "📥 Download Bias Analysis (CSV)",
                        data=bias_df.to_csv(index=False),
                        file_name=f"mtf_bias_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        key="bias_download"
                    )
                with col2:
                    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                st.warning("⚠️ No bias analysis available. Click 'Refresh Analysis' to generate.")
        else:
            st.info("⏳ Loading bias analysis…")
            if st.button("🔄 Generate Bias Analysis", type="primary"):
                with st.spinner("Analyzing bias across all timeframes…"):
                    bias_results = generate_bias_dashboard(data_by_timeframe)
                    st.session_state.bias_results = bias_results
                    st.rerun()

    # Tab 2: Overview
    with tab2:
        st.subheader("📊 Market Overview")
        if daily_data:
            rows = []
            for pair, df in daily_data.items():
                if not df.empty and "Close" in df.columns:
                    price = df["Close"].iloc[-1]
                    change = df["Close"].pct_change().iloc[-1] * 100 if len(df) > 1 else 0.0
                    dp = 2 if pair in ("BTC/USD", "XAU/USD") else 5
                    rows.append({
                        "Pair": pair,
                        "Price": round(price, dp),
                        "Change %": round(change, 3),
                        "Bars": len(df),
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.warning("No valid data for any pair")
        else:
            st.error("No data loaded — check your internet connection.")

    # Tab 3: Macro
    with tab3:
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

    # Tab 4: Technical Chart (PRODUCTION GRADE)
    with tab4:
        st.subheader("📈 Professional Technical Analysis")
        avail = [p for p, d in daily_data.items() if not d.empty and "Close" in d.columns]

        if avail:
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                pair = st.selectbox("Select Pair", avail, key="chart_pair")
            with col2:
                tf = st.selectbox("Timeframe", list(config.timeframes.keys()), key="chart_tf")
            with col3:
                chart_style = st.selectbox("Chart Theme", ["Professional", "Dark", "Light"], key="chart_style")

            df_c = data_by_timeframe.get(tf, {}).get(pair, pd.DataFrame())

            if not df_c.empty and "Close" in df_c.columns:
                with st.expander("⚙️ Chart Settings", expanded=False):
                    setting_cols = st.columns(4)
                    with setting_cols[0]:
                        show_volume = st.checkbox("Show Volume", value=True, key="show_volume")
                        show_ichimoku = st.checkbox("Show Ichimoku", value=False, key="show_ichimoku")
                    with setting_cols[1]:
                        show_fib = st.checkbox("Show Fibonacci", value=False, key="show_fib")
                        show_sr = st.checkbox("Show S/R Levels", value=True, key="show_sr")
                    with setting_cols[2]:
                        show_bb = st.checkbox("Show Bollinger Bands", value=True, key="show_bb")
                        show_ma = st.multiselect("Moving Averages", ["EMA20", "EMA50", "SMA20", "SMA50"],
                                                 default=["EMA20", "EMA50"], key="show_ma")
                    with setting_cols[3]:
                        indicator_panels = st.multiselect("Indicator Panels",
                                                          ["MACD", "RSI", "Stochastic", "ADX"],
                                                          default=["MACD", "RSI", "Stochastic", "ADX"],
                                                          key="indicator_panels")

                chart_settings = {
                    "show_volume": show_volume,
                    "show_ichimoku": show_ichimoku,
                    "show_fib": show_fib,
                    "show_sr": show_sr,
                    "show_bb": show_bb,
                    "show_ma": show_ma,
                    "indicator_panels": indicator_panels
                }

                render_professional_chart(df_c, pair, tf, data_by_timeframe, chart_settings)
            else:
                st.warning(f"No data available for {pair} on {tf}")
        else:
            st.warning("No data available")

    # Tab 5: 15-Min Entry
    with tab5:
        st.subheader("⏱️ 15-Minute Entry Signal")
        avail = [p for p in daily_data if not daily_data[p].empty]

        if avail:
            pair_e = st.selectbox("Pair", avail, key="entry_pair")
            df_15m = data_by_timeframe.get("15 Minute", {}).get(pair_e, pd.DataFrame())
            df_d = data_by_timeframe.get("Daily", {}).get(pair_e, pd.DataFrame())

            if not df_15m.empty and not df_d.empty and "Close" in df_d.columns:
                # Indicators already added at load time
                if not df_d.empty:
                    di = df_d.iloc[-1]
                    adx_v = safe_get(di, "ADX", 0.0)
                    close_v = safe_get(di, "Close", 0.0)
                    ema20_v = safe_get(di, "EMA_20", close_v)

                    bias_v = ("Long" if close_v > ema20_v else "Short") \
                        if adx_v > config.adx_trend_min else "Neutral"

                    st.write(f"**Daily Trend Bias:** `{bias_v}` | ADX = {adx_v:.1f}")
                    sig = entry_generator.get_entry_signal(df_15m, bias_v)

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        if sig["signal"] == 1:
                            st.success("### 🟢 LONG")
                        elif sig["signal"] == -1:
                            st.error("### 🔴 SHORT")
                        else:
                            st.info("### ⚪ NO SIGNAL")
                    c2.metric("Confidence", f"{sig['confidence']}/5")
                    c3.metric("Price", f"{sig.get('price', 0):.5f}")

                    for r in sig.get("reasons", []):
                        st.info(f"ℹ️ {r}")
                else:
                    st.warning("Could not calculate daily indicators")
            else:
                st.warning("Insufficient data for 15-min analysis on this pair")
        else:
            st.warning("No data available")

    # Tab 6: Trading Ideas
    with tab6:
        st.subheader("🎯 Trading Ideas")
        st.caption("Multi-timeframe confluence · Structure-based stops · Swing/ATR take-profits")

        if st.button("🔄 Generate Trading Ideas", type="primary"):
            with st.spinner("Analysing all pairs across timeframes…"):
                ideas, skipped = generate_trading_ideas(data_by_timeframe)
                st.session_state.latest_ideas = ideas

            new_alerts = check_and_notify(ideas)
            if new_alerts:
                st.success(
                    f"🔔 {len(new_alerts)} new high-conviction alert(s) — "
                    "see the sidebar Alert Log"
                )

            if skipped:
                with st.expander(f"⚠️ {len(skipped)} pair(s) skipped"):
                    for s in skipped:
                        st.caption(f"• {s}")

            if ideas:
                st.success(f"✅ {len(ideas)} idea(s) generated")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total", len(ideas))
                c2.metric("Long", sum(1 for i in ideas if i["bias"] == "Long"))
                c3.metric("Short", sum(1 for i in ideas if i["bias"] == "Short"))
                c4.metric("High Conviction", sum(1 for i in ideas if i["conviction"] == "High"))

                st.divider()

                for idx, idea in enumerate(ideas):
                    direction = "📈" if idea["bias"] == "Long" else "📉"
                    header = f"### {idx + 1}. {idea['pair']} — {idea['bias'].upper()} {direction}"

                    if idea["conviction"] == "High":
                        st.success(header + " 🔔 HIGH CONVICTION")
                    elif idea["conviction"] == "Medium":
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

                    tp1_lbl = "TP1" if idea["tp1_valid"] else "TP1 ⚠️"
                    p2.metric(
                        tp1_lbl,
                        f"{idea['take_profit_1']:.5f}",
                        delta=f"R:R 1:{idea['risk_reward_1']:.2f} ({idea['tp1_method']})",
                    )

                    tp2_lbl = "TP2" if idea["tp2_valid"] else "TP2 ⚠️"
                    p3.metric(
                        tp2_lbl,
                        f"{idea['take_profit_2']:.5f}",
                        delta=f"R:R 1:{idea['risk_reward_2']:.2f} ({idea['tp2_method']})",
                    )

                    p4.metric("Stop Loss", f"{idea['stop_loss']:.5f}")
                    risk_pct = (abs(idea["entry"] - idea["stop_loss"]) / idea["entry"]) * 100
                    p5.metric("Risk %", f"{risk_pct:.2f}%")

                    st.caption(
                        f"🛡️ Stop method: **{idea['stop_loss_method']}** | "
                        f"Distance: **{idea['stop_loss_pips']} pips**"
                    )

                    if idea.get("entry_signal") and idea["entry_signal"].get("signal") != 0:
                        with st.expander("📊 Entry Signal Details"):
                            es = idea["entry_signal"]
                            st.write(f"**Confidence:** {es['confidence']}/5")
                            st.write(
                                f"**Stoch K/D:** "
                                f"{es.get('stoch_k', 0):.1f} / {es.get('stoch_d', 0):.1f}"
                            )
                            st.write(f"**RSI:** {es.get('rsi', 0):.1f}")
                            for r in es.get("reasons", []):
                                st.write(f"  • {r}")

                    st.divider()

                export_df = pd.DataFrame([{
                    "Pair": i["pair"],
                    "Bias": i["bias"],
                    "Conviction": i["conviction"],
                    "Strength": i["strength_score"],
                    "Entry": i["entry"],
                    "TP1": i["take_profit_1"],
                    "TP1 Method": i["tp1_method"],
                    "TP1 Valid": i["tp1_valid"],
                    "TP2": i["take_profit_2"],
                    "TP2 Method": i["tp2_method"],
                    "TP2 Valid": i["tp2_valid"],
                    "Stop Loss": i["stop_loss"],
                    "Stop Method": i["stop_loss_method"],
                    "R:R TP1": i["risk_reward_1"],
                    "R:R TP2": i["risk_reward_2"],
                    "Stop Pips": i["stop_loss_pips"],
                    "Thesis": i["thesis"],
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

    # Tab 7: Weekly Swing Trading
    with tab7:
        st.subheader("📅 Weekly Swing Trading Ideas")
        st.caption(
            "Weekly structure + Daily alignment + 4H entry confirmation · "
            "Structure/ATR stops · Min R:R enforced"
        )

        def _render_swing_idea(idx: int, idea: Dict) -> None:
            direction = "📈" if idea["bias"] == "Long" else "📉"
            conf_icon = "🔔 HIGH CONVICTION" if idea["conviction"] == "High" else ""
            header = f"### {idx + 1}. {idea['pair']} — {idea['bias'].upper()} SWING {direction} {conf_icon}"

            if idea["conviction"] == "High":
                st.success(header)
            elif idea["conviction"] == "Medium":
                st.warning(header)
            else:
                st.info(header)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Conviction", idea["conviction"])
            m2.metric("Strength", f"{idea['strength_score']}/15")
            m3.metric("Weekly", idea["weekly_trend"])
            conf_label = "✅ Confirmed" if idea.get("4h_confirmation") else "⚠️ Not confirmed"
            m4.metric("4H Entry", conf_label)

            st.markdown(f"**📝 Thesis:** {idea['thesis']}")
            st.markdown("**💰 Price Levels:**")

            p1, p2, p3, p4, p5 = st.columns(5)
            p1.metric("Entry", f"{idea['entry']:.5f}")

            tp1_lbl = "Target 1" if idea.get("tp1_valid", True) else "Target 1 ⚠️"
            p2.metric(
                tp1_lbl,
                f"{idea['target_1']:.5f}",
                delta=f"R:R 1:{idea['risk_reward_1']:.2f} ({idea.get('tp1_method', 'ATR')})",
            )

            tp2_lbl = "Target 2" if idea.get("tp2_valid", True) else "Target 2 ⚠️"
            p3.metric(
                tp2_lbl,
                f"{idea['target_2']:.5f}",
                delta=f"R:R 1:{idea['risk_reward_2']:.2f} ({idea.get('tp2_method', 'ATR')})",
            )

            p4.metric("Stop Loss", f"{idea['stop_loss']:.5f}")
            risk_pct = (abs(idea["entry"] - idea["stop_loss"]) / idea["entry"]) * 100
            p5.metric("Risk %", f"{risk_pct:.2f}%")

            st.caption(
                f"🛡️ Stop method: **{idea.get('stop_loss_method', 'N/A')}** | "
                f"Distance: **{idea.get('stop_loss_pips', 0)} pips** | "
                f"Invalidation: {idea['invalidation']} | "
                f"Daily ATR: {idea['atr']:.5f}"
            )
            st.divider()

        if st.button("🔄 Generate Swing Ideas", type="primary", key="swing_button"):
            with st.spinner("Analyzing weekly/daily/4H swing setups…"):
                swing_ideas = generate_weekly_swing_ideas(data_by_timeframe)
                st.session_state.swing_ideas = swing_ideas

            if swing_ideas:
                st.success(f"✅ {len(swing_ideas)} swing idea(s) generated")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total", len(swing_ideas))
                c2.metric("Long", sum(1 for i in swing_ideas if i["bias"] == "Long"))
                c3.metric("Short", sum(1 for i in swing_ideas if i["bias"] == "Short"))
                c4.metric("High Conviction", sum(1 for i in swing_ideas if i["conviction"] == "High"))
                st.divider()

                for idx, idea in enumerate(swing_ideas):
                    _render_swing_idea(idx, idea)

                swing_df = pd.DataFrame([{
                    "Pair": i["pair"],
                    "Bias": i["bias"],
                    "Conviction": i["conviction"],
                    "Strength": i["strength_score"],
                    "Entry": i["entry"],
                    "Stop": i["stop_loss"],
                    "Stop Method": i["stop_loss_method"],
                    "Target 1": i["target_1"],
                    "TP1 Method": i["tp1_method"],
                    "Target 2": i["target_2"],
                    "TP2 Method": i["tp2_method"],
                    "R:R 1": i["risk_reward_1"],
                    "R:R 2": i["risk_reward_2"],
                    "4H Confirm": i.get("4h_confirmation", False),
                    "Thesis": i["thesis"],
                } for i in swing_ideas])

                st.download_button(
                    "📥 Download Swing Ideas (CSV)",
                    data=swing_df.to_csv(index=False),
                    file_name=f"swing_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    key="swing_download",
                )
            else:
                st.warning("⚠️ No swing ideas generated — no setups met the minimum R:R threshold.")

        # FIX 7: Cached display now renders ALL fields via shared helper
        elif "swing_ideas" in st.session_state:
            cached = st.session_state.swing_ideas
            if cached:
                st.info(f"📊 Showing {len(cached)} previously generated swing ideas")
                for idx, idea in enumerate(cached):
                    _render_swing_idea(idx, idea)  # ← was missing target2, thesis, stop method
            else:
                st.info("👆 Click 'Generate Swing Ideas' to analyze weekly/daily swing setups")
        else:
            st.info("👆 Click 'Generate Swing Ideas' to analyze weekly/daily swing setups")


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.error("Application error: %s\n%s", exc, traceback.format_exc())
        st.error(f"An error occurred: {exc}")
        st.stop()
