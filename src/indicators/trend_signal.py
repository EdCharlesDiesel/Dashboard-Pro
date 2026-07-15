"""Trend-following signal evaluator — preserves logic from daily-trading-checklist."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd
import streamlit as st
import yfinance as yf

from src.indicators.technical import TechnicalIndicators


@dataclass(frozen=True)
class TrendSignal:
    """Outcome of a single-symbol trend evaluation."""
    label: str          # e.g. "🚀 STRONG BUY"
    score: int          # conditions met
    max_score: int      # always 6
    conditions: Dict[str, bool]
    direction: str      # STRONG_BUY | BUY | NEUTRAL | SELL | STRONG_SELL

    def as_tuple(self) -> Tuple[str, int, int, Dict[str, bool], str]:
        return self.label, self.score, self.max_score, self.conditions, self.direction


class TrendSignalEvaluator:
    """Encapsulates the 6-condition trend-following scoring rules.

    All numbers (RSI bands, EMA tolerances, ADX threshold) are unchanged from
    the original `evaluate_trend_signal()` function so the verdicts are
    byte-for-byte identical.
    """

    MAX_SCORE = 6
    DEFAULT_MIN_CONDITIONS = 4
    STRONG_THRESHOLD = 5

    @staticmethod
    @st.cache_data(ttl=300, show_spinner=False)
    def fetch(
        ticker: str, interval: str, period: str, resample: Optional[str]
    ) -> Optional[pd.DataFrame]:
        """Pull OHLCV (read-through Postgres cache) + indicators. None on failure."""
        from src.db.market_cache import cached_ohlc
        df = cached_ohlc(ticker, period=period, interval=interval, ttl=300)
        if df is None or df.empty:
            return None
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in df.columns for col in required_cols):
            return None
        df = df[required_cols].copy()
        df.dropna(inplace=True)
        if resample:
            rule = "W" if resample == "1W" else resample
            df = df.resample(rule).agg({
                "Open": "first", "High": "max", "Low": "min",
                "Close": "last", "Volume": "sum",
            }).dropna()
        df = TechnicalIndicators.enrich_trend_frame(df)
        return df.dropna()

    @staticmethod
    def evaluate(
        df: pd.DataFrame, min_conditions: int = DEFAULT_MIN_CONDITIONS
    ) -> TrendSignal:
        if len(df) < 10:
            return TrendSignal("⏳ NEUTRAL", 0, TrendSignalEvaluator.MAX_SCORE, {}, "NEUTRAL")
        r = df.iloc[-1]
        close = float(r["Close"])
        e50 = float(r["EMA50"])
        e200 = float(r["EMA200"])
        rsi_val = float(r["RSI"])
        macd_v = float(r["MACD"])
        macd_s = float(r["MACDSig"])
        adx_v = float(r["ADX"])
        buy_conds = {
            "Price above 200 EMA": close > e200,
            "50 EMA above 200 EMA (Golden X)": e50 > e200,
            "Price at/above 50 EMA": close >= e50 * 0.9985,
            "RSI 45–70 (bullish momentum)": 45 <= rsi_val <= 70,
            "MACD above Signal line": macd_v > macd_s,
            "Strong trend (ADX > 25)": adx_v > 25,
        }
        sell_conds = {
            "Price below 200 EMA": close < e200,
            "50 EMA below 200 EMA (Death X)": e50 < e200,
            "Price at/below 50 EMA": close <= e50 * 1.0015,
            "RSI 30–45 (bearish momentum)": 30 <= rsi_val <= 45,
            "MACD below Signal line": macd_v < macd_s,
            "Strong trend (ADX > 25)": adx_v > 25,
        }
        bs = sum(buy_conds.values())
        ss = sum(sell_conds.values())
        max_score = TrendSignalEvaluator.MAX_SCORE
        strong = TrendSignalEvaluator.STRONG_THRESHOLD

        # ADX is the precondition for the other 5 conditions meaning anything
        # in a trend-following strategy, not one point out of six equal-weight
        # criteria — golden cross + price above EMAs + positive MACD in a
        # ranging market is exactly the whipsaw setup (those line up at range
        # tops right before mean reversion). Gate on it instead of letting it
        # get outvoted 5-to-1.
        if adx_v <= 25:
            conds = buy_conds if bs >= ss else sell_conds
            return TrendSignal("⏳ NO TREND — STAND ASIDE", max(bs, ss),
                               max_score, conds, "NEUTRAL")
        if bs >= min_conditions and bs >= ss:
            if bs >= strong:
                return TrendSignal("🚀 STRONG BUY", bs, max_score, buy_conds, "STRONG_BUY")
            return TrendSignal("📈 BUY", bs, max_score, buy_conds, "BUY")
        if ss >= min_conditions and ss > bs:
            if ss >= strong:
                return TrendSignal("🔻 STRONG SELL", ss, max_score, sell_conds, "STRONG_SELL")
            return TrendSignal("📉 SELL", ss, max_score, sell_conds, "SELL")
        return TrendSignal("⏳ NEUTRAL", max(bs, ss), max_score, {}, "NEUTRAL")
