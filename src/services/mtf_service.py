"""Multi-Timeframe alignment helper — Weekly / Daily / 4H EMA stance."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import streamlit as st

from src.indicators.trend_signal import TrendSignalEvaluator


@dataclass(frozen=True)
class MTFAlignment:
    weekly: Optional[str]
    daily: Optional[str]
    h4: Optional[str]
    aligned: int
    total: int
    target: str

    def as_dict(self) -> dict:
        return {
            "weekly": self.weekly, "daily": self.daily, "4h": self.h4,
            "aligned": self.aligned, "total": self.total, "target": self.target,
        }


class MTFService:
    """Checks W/D/4H EMA alignment using the same data pipeline as TrendSignalEvaluator."""

    @staticmethod
    @st.cache_data(ttl=300, show_spinner=False)
    def alignment(ticker: str, direction: str) -> dict:
        res: dict = {}
        # Weekly resample
        try:
            df_w = TrendSignalEvaluator.fetch(ticker, "1d", "2y", "1W")
            if df_w is not None and len(df_w) > 10:
                r = df_w.iloc[-1]
                c, e50, e200 = float(r["Close"]), float(r["EMA50"]), float(r["EMA200"])
                res["weekly"] = "BULL" if c > e50 > e200 else ("BEAR" if c < e50 < e200 else "NEUTRAL")
            else:
                res["weekly"] = None
        except Exception:
            res["weekly"] = None
        # Daily
        try:
            df_d = TrendSignalEvaluator.fetch(ticker, "1d", "2y", None)
            if df_d is not None and len(df_d) > 10:
                r = df_d.iloc[-1]
                c, e50 = float(r["Close"]), float(r["EMA50"])
                res["daily"] = "BULL" if c > e50 else "BEAR"
            else:
                res["daily"] = None
        except Exception:
            res["daily"] = None
        # 4H (60m resampled)
        try:
            df_4h = TrendSignalEvaluator.fetch(ticker, "60m", "59d", "4h")
            if df_4h is not None and len(df_4h) > 10:
                r = df_4h.iloc[-1]
                c, e50 = float(r["Close"]), float(r["EMA50"])
                res["4h"] = "BULL" if c > e50 else "BEAR"
            else:
                res["4h"] = None
        except Exception:
            res["4h"] = None
        target = "BULL" if direction == "LONG" else "BEAR"
        aligned = sum(1 for v in res.values() if v == target)
        return {**res, "aligned": aligned, "total": 3, "target": target}
