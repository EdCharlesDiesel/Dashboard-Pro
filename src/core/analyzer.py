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
    # Full set of columns produced by add_indicators — used for idempotency check
    INDICATOR_COLS = frozenset({
        "RSI", "MACD", "MACD_Signal", "MACD_Histogram",
        "SMA_20", "SMA_50", "EMA_20", "EMA_50",
        "BB_Upper", "BB_Middle", "BB_Lower",
        "ATR", "Stoch_K", "Stoch_D",
        "ADX", "ADX_Pos", "ADX_Neg",
        "Resistance_20", "Support_20",
    })

    @staticmethod
    def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        # EMA_50 is the slowest indicator — it needs at least EMA_LONG_WINDOW bars to produce
        # a non-NaN value. BB_WINDOW (20) was previously used here but would silently produce
        # all-NaN EMA_50 columns on timeframes with limited history (e.g. Weekly at "3mo").
        if df.empty or len(df) < TechnicalAnalyzer.EMA_LONG_WINDOW:
            return df
        if not all(c in df.columns for c in TechnicalAnalyzer.REQUIRED_COLUMNS):
            logger.warning("Missing required OHLC columns for indicator calculation")
            return df

        # Only skip recalculation when ALL expected indicator columns are present.
        # Checking just RSI+MACD would return a partially-computed DataFrame if a
        # previous run failed partway through.
        if TechnicalAnalyzer.INDICATOR_COLS.issubset(df.columns):
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
    def calculate_expanded_pivots(df: pd.DataFrame) -> dict:
        """Calculates expanded pivot points (R1-3, S1-3, and Midpoints)."""
        if len(df) < 2:
            if df.empty: return {}
            ref = df.iloc[-1]
        else:
            ref = df.iloc[-2]

        h, l, c = ref["High"], ref["Low"], ref["Close"]
        pp = (h + l + c) / 3
        r1 = 2 * pp - l
        r2 = pp + (h - l)
        r3 = h + 2 * (pp - l)
        s1 = 2 * pp - h
        s2 = pp - (h - l)
        s3 = l - 2 * (h - pp)

        return {
            "R3": r3, "M4": (r2 + r3) / 2, "R2": r2, "M3": (r1 + r2) / 2, "R1": r1,
            "M2": (pp + r1) / 2, "PP": pp, "M1": (s1 + pp) / 2, "S1": s1,
            "M0": (s1 + s2) / 2, "S2": s2, "S3": s3,
        }

    @staticmethod
    def get_sessions(date):
        import pandas as pd
        base = pd.Timestamp(date)
        sessions = {
            "Asian": (base.replace(hour=0, minute=0), base.replace(hour=8, minute=59)),
            "London": (base.replace(hour=7, minute=0), base.replace(hour=15, minute=59)),
            "NY": (base.replace(hour=13, minute=0), base.replace(hour=21, minute=59)),
        }
        return sessions

    @staticmethod
    def session_analysis(df_day) -> dict:
        results = {}
        if df_day.empty:
            return results
        for name, (s, e) in TechnicalAnalyzer.get_sessions(df_day["datetime"].dt.date.iloc[0]).items():
            seg = df_day[(df_day["datetime"] >= s) & (df_day["datetime"] <= e)]
            if len(seg) == 0:
                results[name] = None
                continue
            o = seg["Open"].iloc[0]
            c = seg["Close"].iloc[-1]
            h = seg["High"].max()
            l = seg["Low"].min()
            direction = "🟢 Bull" if c > o else ("🔴 Bear" if c < o else "⚪ Flat")
            results[name] = {
                "open": o, "close": c, "high": h, "low": l,
                "range_pips": round((h - l) * 10000, 1),
                "direction": direction,
            }
        return results

    @staticmethod
    def compute_daily_bias(df_day) -> tuple:
        if len(df_day) < 5:
            return "Neutral", 50.0

        opens = df_day["Open"].iloc[0]
        closes = df_day["Close"].iloc[-1]
        highs = df_day["High"].max()
        lows = df_day["Low"].min()
        mid = (highs + lows) / 2
        pct_change = (closes - opens) / opens * 100

        # Check higher highs / lower lows structure
        h4 = df_day.set_index("datetime").resample("4h").agg(
            Open=("Open", "first"), High=("High", "max"),
            Low=("Low", "min"), Close=("Close", "last")
        ).dropna()

        bull_score = 0
        if closes > opens: bull_score += 2
        if closes > mid: bull_score += 1
        if len(h4) >= 2:
            if h4["Close"].iloc[-1] > h4["Close"].iloc[0]: bull_score += 2
            if h4["High"].iloc[-1] > h4["High"].iloc[-2]: bull_score += 1
            if h4["Low"].iloc[-1] > h4["Low"].iloc[-2]: bull_score += 1
        if pct_change > 0.1: bull_score += 1

        total = 8
        bull_pct = (bull_score / total) * 100
        if bull_pct >= 62:
            return "Bullish", bull_pct
        elif bull_pct <= 38:
            return "Bearish", 100 - bull_pct
        else:
            return "Neutral", 50.0

    @staticmethod
    def generate_pro_signals(df: pd.DataFrame, pivots: dict) -> pd.DataFrame:
        """Generates QuantConnect-style signals based on pivots, MAs, and RSI."""
        if df.empty: return df
        df = df.copy()

        # Ensure all indicators are present via the canonical method (avoids duplicate work)
        df = TechnicalAnalyzer.add_indicators(df)

        # Add fast MAs that add_indicators doesn't produce (MA5, MA10)
        for name, period in [("MA5", 5), ("MA10", 10)]:
            if name not in df.columns:
                df[name] = df["Close"].rolling(period).mean()

        # Map EMA_20/EMA_50 to the MA20/MA50 names the signal logic expects
        if "MA20" not in df.columns and "EMA_20" in df.columns:
            df["MA20"] = df["EMA_20"]
        if "MA50" not in df.columns and "EMA_50" in df.columns:
            df["MA50"] = df["EMA_50"]

        df["Signal"] = "HOLD"
        df["Signal_Score"] = 0

        if pivots and "PP" in pivots:
            pp, r1, s1 = pivots["PP"], pivots["R1"], pivots["S1"]
            close = df["Close"]

            # Logic from provided snippet
            buy = (close > pp) & (close < r1) & (df["MA5"] > df["MA10"]) & (df["MA10"] > df["MA20"]) & (
                        df["RSI"] < 70) & (close > df["MA20"])
            sell = (close < pp) & (close > s1) & (df["MA5"] < df["MA10"]) & (df["MA10"] < df["MA20"]) & (
                        df["RSI"] > 30) & (close < df["MA20"])

            sbuy = buy & (close > df["MA50"]) & (df["RSI"] < 60)
            ssell = sell & (close < df["MA50"]) & (df["RSI"] > 40)

            df.loc[buy, "Signal"] = "BUY"
            df.loc[sbuy, "Signal"] = "STRONG BUY"
            df.loc[sell, "Signal"] = "SELL"
            df.loc[ssell, "Signal"] = "STRONG SELL"

            scores = {"STRONG BUY": 2, "BUY": 1, "HOLD": 0, "SELL": -1, "STRONG SELL": -2}
            df["Signal_Score"] = df["Signal"].map(scores)

        return df

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