import pandas as pd
import logging
import ta

logger = logging.getLogger("ForexDashboard")

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

    @staticmethod
    def calculate_pivots(df: pd.DataFrame) -> dict:
        """Calculates standard pivot points based on the previous period's OHLC."""
        if len(df) < 2:
            if df.empty: return {}
            ref = df.iloc[-1]
        else:
            # Use previous completed candle for current pivots
            ref = df.iloc[-2]

        h, l, c = ref["High"], ref["Low"], ref["Close"]
        p = (h + l + c) / 3
        return {
            "Pivot": p,
            "R1": (2 * p) - l,
            "S1": (2 * p) - h,
            "R2": p + (h - l),
            "S2": p - (h - l),
            "R3": h + 2 * (p - l),
            "S3": l - 2 * (h - p),
        }

    @staticmethod
    def calculate_fibonacci(df: pd.DataFrame) -> dict:
        """Calculates Fibonacci levels based on the high and low of the provided dataframe."""
        if df.empty:
            return {}
        high = df["High"].max()
        low = df["Low"].min()
        diff = high - low
        return {
            "0.0%": high,
            "23.6%": high - 0.236 * diff,
            "38.2%": high - 0.382 * diff,
            "50.0%": high - 0.5 * diff,
            "61.8%": high - 0.618 * diff,
            "78.6%": high - 0.786 * diff,
            "100.0%": low,
        }

    @staticmethod
    def get_sentiment(df: pd.DataFrame) -> str:
        """Returns 'Bullish', 'Bearish', or 'Neutral' sentiment based on EMA and RSI."""
        if df.empty or "EMA_20" not in df.columns:
            return "Neutral"

        last = df.iloc[-1]
        close = last["Close"]
        ema20 = last["EMA_20"]
        rsi = last.get("RSI", 50)

        if close > ema20 and rsi > 50:
            return "Bullish"
        elif close < ema20 and rsi < 50:
            return "Bearish"
        else:
            return "Neutral"

    @staticmethod
    def get_mtf_sentiment(data_by_tf: dict, pair: str) -> dict:
        """Aggregates sentiment across all available timeframes for a pair."""
        results = {}
        for tf, pairs in data_by_tf.items():
            df = pairs.get(pair)
            if df is not None and not df.empty:
                # Ensure indicators are present
                df = TechnicalAnalyzer.add_indicators(df)
                results[tf] = TechnicalAnalyzer.get_sentiment(df)
            else:
                results[tf] = "N/A"
        return results
