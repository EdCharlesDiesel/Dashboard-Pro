import json
from datetime import datetime

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from src.core.analyzer import TechnicalAnalyzer as analyzer
from src.core.config import default_config as config


def safe_get(row: pd.Series, col: str, default: float = 0.0) -> float:

    try:
        if col not in row.index:
            return default
        val = row[col]
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return float(val)
    except Exception:
        return default


class EntrySignalGenerator:
    @staticmethod
    def get_entry_signal(df_15m: pd.DataFrame, bias: str) -> Dict:
        if df_15m.empty or len(df_15m) < 5:
            return {'signal': 0, 'confidence': 0, 'reasons': ['Insufficient 15-min data']}

        df = analyzer.add_indicators(df_15m)
        if df.empty or len(df) < 2:
            return {'signal': 0, 'confidence': 0, 'reasons': ['Indicator calculation failed']}

        last = df.iloc[-1]
        prev = df.iloc[-2]

        k = safe_get(last, 'Stoch_K', 50.0)
        d = safe_get(last, 'Stoch_D', 50.0)
        prev_k = safe_get(prev, 'Stoch_K', 50.0)
        prev_d = safe_get(prev, 'Stoch_D', 50.0)
        rsi = safe_get(last, 'RSI', 50.0)
        price = safe_get(last, 'Close', 0.0)

        if price <= 0.0:
            return {'signal': 0, 'confidence': 0, 'reasons': ['Invalid price data']}

        bb_lower = safe_get(last, 'BB_Lower', price * 0.99)
        bb_upper = safe_get(last, 'BB_Upper', price * 1.01)

        signal, confidence, reasons = 0, 0, []

        if bias == 'Long':
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
                reasons.append(f"Awaiting Long trigger — K={k:.1f}, RSI={rsi:.1f}")

        elif bias == 'Short':
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
                reasons.append(f"Awaiting Short trigger — K={k:.1f}, RSI={rsi:.1f}")
        else:
            reasons.append(f"Trend bias is Neutral (ADX < {config.adx_trend_min:.0f})")

        return {
            'signal': signal,
            'confidence': min(confidence, 5),
            'reasons': reasons,
            'stoch_k': k,
            'stoch_d': d,
            'rsi': rsi,
            'price': price,
        }


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

    def get_swing_stop(self, df: pd.DataFrame, bias: str, lookback: int = 20) -> Optional[float]:
        if df.empty or len(df) < lookback:
            return None
        recent = df.tail(lookback)
        if bias == 'Long' and 'Low' in df.columns:
            return float(recent['Low'].min())
        if bias == 'Short' and 'High' in df.columns:
            return float(recent['High'].max())
        return None

    def calculate(self, df: pd.DataFrame, pair: str, bias: str, current_price: float, atr: float, pivots: dict = None,
                  lookback: int = 20) -> Dict:
        atr_mult = config.pair_atr_multipliers.get(pair, config.atr_sl_mult)
        min_dist = config.pair_min_stop.get(pair, 0.0010)
        buffer = atr * 0.25

        atr_stop = (current_price - atr * atr_mult) if bias == 'Long' else (current_price + atr * atr_mult)
        swing = self.get_swing_stop(df, bias, lookback)

        # Start with ATR as default
        stop = atr_stop
        method = "ATR"

        # Pivot-based stops
        if pivots:
            if bias == 'Long':
                # Use S1 as a conservative stop if it's below current price
                s1 = pivots.get("S1")
                if s1 and s1 < current_price:
                    stop, method = s1, "Pivot S1"
            else:
                # Use R1 as a conservative stop if it's above current price
                r1 = pivots.get("R1")
                if r1 and r1 > current_price:
                    stop, method = r1, "Pivot R1"

        if swing is not None:
            if bias == 'Long':
                struct_stop = swing - buffer
                if struct_stop < current_price:
                    # Choose the WIDEST stop (furthest below price) to give the trade
                    # maximum room — avoids getting stopped out on normal pullbacks.
                    if struct_stop < stop:
                        stop, method = struct_stop, "Swing Low"
            else:
                struct_stop = swing + buffer
                if struct_stop > current_price:
                    # Choose the WIDEST stop (furthest above price) for the same reason.
                    if struct_stop > stop:
                        stop, method = struct_stop, "Swing High"

        raw_dist = abs(current_price - stop)
        if raw_dist < min_dist:
            stop = (current_price - min_dist) if bias == 'Long' else (current_price + min_dist)
            method += " + min-dist enforced"

        return {
            "stop": stop,
            "method": method,
            "distance_pips": self.price_to_pips(pair, abs(current_price - stop)),
        }


class TakeProfitCalculator:
    def get_swing_target(self, df: pd.DataFrame, bias: str, lookback: int = 20) -> Optional[float]:
        if df.empty or len(df) < lookback:
            return None
        recent = df.tail(lookback)
        if bias == 'Long' and 'High' in df.columns:
            return float(recent['High'].max())
        if bias == 'Short' and 'Low' in df.columns:
            return float(recent['Low'].min())
        return None

    def calculate(self, df: pd.DataFrame, pair: str, bias: str, current_price: float, atr: float, stop_loss: float,
                  pivots: dict = None, lookback: int = 20) -> Dict:
        stop_dist = abs(current_price - stop_loss) or atr
        swing = self.get_swing_target(df, bias, lookback)

        if bias == 'Long':
            tp1 = current_price + atr * config.tp1_atr_mult
            m1 = f"ATR ×{config.tp1_atr_mult}"
            tp2 = current_price + atr * config.tp2_atr_mult
            m2 = f"ATR ×{config.tp2_atr_mult}"

            if pivots:
                r1, r2, r3 = pivots.get("R1"), pivots.get("R2"), pivots.get("R3")
                if r1 and r1 > current_price:
                    tp1, m1 = r1, "Pivot R1"
                if r2 and r2 > tp1:
                    tp2, m2 = r2, "Pivot R2"
                elif r3 and r3 > tp1:
                    tp2, m2 = r3, "Pivot R3"

            if swing is not None:
                if tp1 < swing < tp2:
                    tp2, m2 = swing, "Swing High"
                elif swing > tp2:
                    tp2, m2 = swing, "Swing High (ext)"

            rr1 = (tp1 - current_price) / stop_dist
            rr2 = (tp2 - current_price) / stop_dist
        else:
            tp1 = current_price - atr * config.tp1_atr_mult
            m1 = f"ATR ×{config.tp1_atr_mult}"
            tp2 = current_price - atr * config.tp2_atr_mult
            m2 = f"ATR ×{config.tp2_atr_mult}"

            if pivots:
                s1, s2, s3 = pivots.get("S1"), pivots.get("S2"), pivots.get("S3")
                if s1 and s1 < current_price:
                    tp1, m1 = s1, "Pivot S1"
                if s2 and s2 < tp1:
                    tp2, m2 = s2, "Pivot S2"
                elif s3 and s3 < tp1:
                    tp2, m2 = s3, "Pivot S3"

            if swing is not None:
                if tp2 < swing < tp1:
                    tp2, m2 = swing, "Swing Low"
                elif swing < tp2:
                    tp2, m2 = swing, "Swing Low (ext)"

            rr1 = (current_price - tp1) / stop_dist
            rr2 = (current_price - tp2) / stop_dist

        return {
            "tp1": tp1, "tp2": tp2,
            "method_tp1": m1, "method_tp2": m2,
            "rr1": round(rr1, 2), "rr2": round(rr2, 2),
            "tp1_valid": rr1 >= config.min_rr,
            "tp2_valid": rr2 >= config.min_rr,
        }


entry_generator = EntrySignalGenerator()
sl_calculator = StopLossCalculator()
tp_calculator = TakeProfitCalculator()


# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED SETUP SCORING
# Single source of truth for the 10-point multi-timeframe checklist used by
# BOTH the Setup Ranker page (pages/setup-ranker.py) and the Trading Ideas tab
# (app.py). Keeping the math here guarantees the two surfaces agree for a given
# pair + direction. The helpers below intentionally recompute indicators from
# raw OHLC so the score is identical regardless of which caller supplied the
# data (and whether it already had indicator columns attached).
# ═══════════════════════════════════════════════════════════════════════════

# Typical spreads (in pips) for the app's instrument universe. The Setup Ranker
# page keeps its own wider list (incl. metals/crosses); both feed score_setup.
TYPICAL_SPREADS: Dict[str, float] = {
    "EUR/USD": 1.2, "GBP/USD": 1.5, "USD/JPY": 1.2, "USD/ZAR": 80.0,
    "AUD/USD": 1.5, "NZD/USD": 2.0, "USD/CAD": 2.0, "USD/CHF": 2.0,
    "XAU/USD": 3.0, "BTC/USD": 20.0,
}


def typical_spread(pair: str) -> float:
    return TYPICAL_SPREADS.get(pair, 0.0)


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi_series(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def macd_cross(s: pd.Series) -> bool:
    m = ema(s, 12) - ema(s, 26)
    sig = ema(m, 9)
    return bool(m.iloc[-1] > sig.iloc[-1])


def atr14(df: pd.DataFrame) -> float:
    prev = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - prev).abs(),
                    (df["Low"] - prev).abs()], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean().iloc[-1])


def swing_structure(df: pd.DataFrame, n: int = 3) -> str:
    """Return BULLISH, BEARISH, or NEUTRAL based on last 2 swing HH/HL."""
    if len(df) < 20:
        return "NEUTRAL"
    highs = df["High"].values
    lows = df["Low"].values
    sh, sl = [], []
    for i in range(n, len(df) - n):
        if highs[i] == max(highs[i - n: i + n + 1]):
            sh.append(highs[i])
        if lows[i] == min(lows[i - n: i + n + 1]):
            sl.append(lows[i])
    if len(sh) >= 2 and len(sl) >= 2:
        hh = sh[-1] > sh[-2]
        hl = sl[-1] > sl[-2]
        lh = sh[-1] < sh[-2]
        ll = sl[-1] < sl[-2]
        if hh and hl:
            return "BULLISH"
        if lh and ll:
            return "BEARISH"
    return "NEUTRAL"


def score_setup(df_weekly: pd.DataFrame, df_daily: pd.DataFrame, df_4h: pd.DataFrame,
                direction: str, pip_size: float, spread_pips: float = 0.0) -> Dict:
    """Score a pair 0–10 against the 10-point multi-timeframe checklist.

    direction: "LONG" or "SHORT". Returns a dict with the per-criterion scores,
    a total, a letter grade, and presentation helpers (price, SL pips, spread%).
    Data-source agnostic — pass whatever Weekly/Daily/4H frames the caller has.
    """
    df_w = df_weekly if df_weekly is not None else pd.DataFrame()
    df_d = df_daily if df_daily is not None else pd.DataFrame()
    df_4h = df_4h if df_4h is not None else pd.DataFrame()

    scores: Dict[str, int] = {}
    details: Dict[str, str] = {}

    # ── 1. Weekly EMA alignment ──────────────────────────────────────────────
    if not df_w.empty and len(df_w) > 50:
        e50_w = float(ema(df_w["Close"], 50).iloc[-1])
        e200_w = float(ema(df_w["Close"], 200).iloc[-1])
        close_w = float(df_w["Close"].iloc[-1])
        ok = close_w > e50_w > e200_w if direction == "LONG" else close_w < e50_w < e200_w
        scores["Weekly EMA"] = 1 if ok else 0
        details["Weekly EMA"] = "✅" if ok else "❌"
    else:
        scores["Weekly EMA"] = 0
        details["Weekly EMA"] = "—"

    # ── 2. Weekly RSI has room ───────────────────────────────────────────────
    if not df_w.empty and len(df_w) > 20:
        rsi_w = float(rsi_series(df_w["Close"]).iloc[-1])
        ok = rsi_w < 70 if direction == "LONG" else rsi_w > 30
        scores["Weekly RSI"] = 1 if ok else 0
        details["Weekly RSI"] = f"{'✅' if ok else '❌'} {rsi_w:.1f}"
    else:
        scores["Weekly RSI"] = 0
        details["Weekly RSI"] = "—"

    # ── 3. Weekly structure ──────────────────────────────────────────────────
    if not df_w.empty and len(df_w) > 20:
        ws = swing_structure(df_w, 3)
        ok = (ws == "BULLISH" and direction == "LONG") or (ws == "BEARISH" and direction == "SHORT")
        scores["Weekly Structure"] = 1 if ok else 0
        details["Weekly Structure"] = f"{'✅' if ok else '❌'} {ws}"
    else:
        scores["Weekly Structure"] = 0
        details["Weekly Structure"] = "—"

    # ── 4. Daily trend ───────────────────────────────────────────────────────
    daily_struct = "NEUTRAL"
    if not df_d.empty and len(df_d) > 50:
        e20_d = float(ema(df_d["Close"], 20).iloc[-1])
        e50_d = float(ema(df_d["Close"], 50).iloc[-1])
        ok = e20_d > e50_d if direction == "LONG" else e20_d < e50_d
        scores["Daily Trend"] = 1 if ok else 0
        details["Daily Trend"] = "✅ EMA20>50" if ok else "❌ EMA20<50"
        daily_struct = swing_structure(df_d, 3)
    else:
        scores["Daily Trend"] = 0
        details["Daily Trend"] = "—"

    # ── 5. Daily structure (HH/HL or LH/LL) ──────────────────────────────────
    ok_ds = (daily_struct == "BULLISH" and direction == "LONG") or \
            (daily_struct == "BEARISH" and direction == "SHORT")
    scores["Daily Structure"] = 1 if ok_ds else 0
    details["Daily Structure"] = f"{'✅' if ok_ds else '❌'} {daily_struct}"

    # ── 6. Daily MACD ────────────────────────────────────────────────────────
    if not df_d.empty and len(df_d) > 50:
        mc = macd_cross(df_d["Close"])
        ok = (mc and direction == "LONG") or (not mc and direction == "SHORT")
        scores["Daily MACD"] = 1 if ok else 0
        details["Daily MACD"] = "✅ Above sig" if ok else "❌ Below sig"
    else:
        scores["Daily MACD"] = 0
        details["Daily MACD"] = "—"

    # ── 7. ATR volatility ────────────────────────────────────────────────────
    a14p = 0.0
    if not df_d.empty and len(df_d) > 25 and pip_size > 0:
        prev = df_d["Close"].shift(1)
        tr = pd.concat([df_d["High"] - df_d["Low"],
                        (df_d["High"] - prev).abs(),
                        (df_d["Low"] - prev).abs()], axis=1).max(axis=1)
        a14 = float(tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean().iloc[-1])
        a20 = float(tr.ewm(alpha=1 / 20, min_periods=20, adjust=False).mean().iloc[-1])
        ok = a14 > a20
        a14p = round(a14 / pip_size, 1)
        scores["ATR Volatile"] = 1 if ok else 0
        details["ATR Volatile"] = f"{'✅' if ok else '❌'} {a14p}p"
    else:
        scores["ATR Volatile"] = 0
        details["ATR Volatile"] = "—"

    # ── 8. 4H confluence zone ────────────────────────────────────────────────
    if not df_4h.empty and len(df_4h) > 30:
        close_4h = float(df_4h["Close"].iloc[-1])
        e20_4h = float(ema(df_4h["Close"], 20).iloc[-1])
        e50_4h = float(ema(df_4h["Close"], 50).iloc[-1])
        tol = atr14(df_4h) * 0.5 if len(df_4h) > 14 else close_4h * 0.005
        near_ema = abs(close_4h - e20_4h) <= tol or abs(close_4h - e50_4h) <= tol
        prev_candle = df_4h.iloc[-2]
        H, L, C = float(prev_candle["High"]), float(prev_candle["Low"]), float(prev_candle["Close"])
        pp = (H + L + C) / 3
        near_pivot = abs(close_4h - pp) <= tol * 2
        ok = near_ema or near_pivot
        scores["4H Zone"] = 1 if ok else 0
        details["4H Zone"] = "✅ At zone" if ok else "❌ Not at zone"
    else:
        scores["4H Zone"] = 0
        details["4H Zone"] = "—"

    # ── 9. 4H structure aligned ──────────────────────────────────────────────
    if not df_4h.empty and len(df_4h) > 20:
        struct_4h = swing_structure(df_4h, 3)
        ok = (struct_4h == "BULLISH" and direction == "LONG") or \
             (struct_4h == "BEARISH" and direction == "SHORT")
        scores["4H Structure"] = 1 if ok else 0
        details["4H Structure"] = f"{'✅' if ok else '❌'} {struct_4h}"
    else:
        scores["4H Structure"] = 0
        details["4H Structure"] = "—"

    # ── 10. Spread/ATR ratio ─────────────────────────────────────────────────
    atr_pips = a14p if a14p > 0 else 1
    spread_pct = spread_pips / atr_pips * 100
    ok = spread_pct <= 5
    scores["Spread/ATR"] = 1 if ok else 0
    details["Spread/ATR"] = f"{'✅' if ok else '❌'} {spread_pct:.1f}%"

    total = sum(scores.values())
    grade = "A" if total >= 8 else "B" if total >= 6 else "C" if total >= 4 else "D"
    close_price = float(df_d["Close"].iloc[-1]) if not df_d.empty else 0.0
    sl_pips = round(a14p * 1.5, 1) if a14p else 0.0

    return {
        "direction": direction,
        "score": total,
        "max_score": 10,
        "pct": int(total / 10 * 100),
        "grade": grade,
        "scores": scores,
        "details": details,
        "close": close_price,
        "sl_pips": sl_pips,
        "spread_pct": spread_pct,
        "atr_pips": a14p,
    }


def analyze_multi_timeframe(df_weekly: pd.DataFrame, df_daily: pd.DataFrame, df_4h: pd.DataFrame,
                            df_1h: pd.DataFrame, df_15m: pd.DataFrame,
                            pair_name: str) -> Optional[Dict]:
    if any(df.empty for df in [df_daily, df_4h, df_1h, df_15m]):
        return None

    one_hour = df_1h.iloc[-1]
    fifteen_m = df_15m.iloc[-1]

    if 'Close' not in df_daily.iloc[-1].index or 'Close' not in df_4h.iloc[-1].index:
        return None

    # ── Directional bias from the unified 10-point checklist ─────────────────
    # Score both directions with the SAME function the Setup Ranker uses, then
    # take whichever side scores higher. This is what makes the Trading Ideas
    # tab and the Setup Ranker agree for a given pair + direction.
    pip_size = sl_calculator.pip_size(pair_name)
    spread = typical_spread(pair_name)
    long_res = score_setup(df_weekly, df_daily, df_4h, "LONG", pip_size, spread)
    short_res = score_setup(df_weekly, df_daily, df_4h, "SHORT", pip_size, spread)

    if long_res["score"] > short_res["score"]:
        final_bias, res = 'Long', long_res
    elif short_res["score"] > long_res["score"]:
        final_bias, res = 'Short', short_res
    else:
        return None  # no clear directional edge

    # strength_score IS the unified checklist score (0-10). Conviction is keyed
    # to the same A/B/C/D grade bands as the Setup Ranker (A→High, B→Medium).
    strength_score = res["score"]
    conviction = "High" if strength_score >= 8 else ("Medium" if strength_score >= 6 else "Low")

    entry_signal = entry_generator.get_entry_signal(df_15m, final_bias)

    # confidence = 15-min entry trigger quality (0-5 from entry_signal, scaled to 0-10).
    # This is deliberately separate from strength_score: a trade can have strong MTF
    # alignment but a poor entry setup (low confidence), or vice versa.
    entry_conf = entry_signal.get('confidence', 0) if entry_signal else 0
    confidence = min(entry_conf * 2, 10)
    h1_close = safe_get(one_hour, 'Close')
    atr = safe_get(one_hour, 'ATR', 0.0)
    if atr <= 0:
        atr = h1_close * 0.005 if h1_close > 0 else 0.001

    current_price = safe_get(fifteen_m, 'Close', 0.0)
    if current_price <= 0.0:
        return None

    # Calculate Daily Pivots for SL/TP
    pivots = analyzer.calculate_expanded_pivots(df_daily)

    sl_result = sl_calculator.calculate(df_1h, pair_name, final_bias, current_price, atr, pivots=pivots)
    tp_result = tp_calculator.calculate(df_4h, pair_name, final_bias, current_price, atr, sl_result["stop"],
                                        pivots=pivots)

    passed = [k for k, v in res["scores"].items() if v]
    thesis = f"{final_bias} setup {strength_score}/10 (Grade {res['grade']})"
    if passed:
        thesis += " — " + ", ".join(passed)
    if entry_signal and entry_signal['signal'] != 0:
        thesis += f" | Entry: {', '.join(entry_signal['reasons'][:2])}"

    return {
        "pair": pair_name,
        "bias": final_bias,
        "conviction": conviction,
        "strength_score": strength_score,
        "grade": res["grade"],
        "setup_scores": res["scores"],
        "setup_details": res["details"],
        "confidence": confidence,
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


def generate_trading_ideas(data_by_timeframe: Dict) -> Tuple[List[Dict], List[str]]:
    ideas: List[Dict] = []
    skipped: List[str] = []

    for pair_name in config.assets:
        # Weekly feeds the unified setup score but is not required for a trade
        # plan, so it is excluded from the "thin bars" gate below.
        frames = {
            tf: data_by_timeframe.get(tf, {}).get(pair_name, pd.DataFrame())
            for tf in ['Weekly', 'Daily', '4 Hour', 'Hourly', '15 Minute']
        }
        thin = [tf for tf in ['Daily', '4 Hour', 'Hourly', '15 Minute']
                if frames[tf].empty or len(frames[tf]) < 20]
        if thin:
            skipped.append(f"{pair_name} — insufficient bars in: {', '.join(thin)}")
            continue

        frames = {tf: analyzer.add_indicators(df) for tf, df in frames.items()}

        idea = analyze_multi_timeframe(frames['Weekly'], frames['Daily'], frames['4 Hour'], frames['Hourly'],
                                       frames['15 Minute'], pair_name)
        if idea and idea['bias'] != 'Neutral':
            ideas.append(idea)

    ideas.sort(key=lambda x: (x['conviction'] == 'High', x['strength_score']), reverse=True)
    return ideas, skipped


def signal_to_setup_row(idea: Dict, session: str) -> Dict:
    """Map a :func:`generate_trading_ideas` idea onto a ``trade_setups`` row dict
    — the exact schema :meth:`TradeRepository.save_setup` writes — so Market
    Overview signals can be persisted alongside checklist trades.

    Pure: no DB, no Streamlit, no network. Ticker/pip size come from the
    instrument registry (the single source of truth); price-distance targets are
    converted to pips via ``pip_size``. Sizing fields the page can't know
    (lot/account/risk) are left ``None``; the full signal context is preserved in
    ``checks_detail`` JSON so nothing is lost.
    """
    from src.instruments.registry import INSTRUMENTS

    pair = idea.get("pair", "")
    inst = INSTRUMENTS.get(pair)
    ticker = inst.ticker if inst else None
    pip_size = (inst.pip_size if inst else 0.0001) or 0.0001

    entry = float(idea.get("entry", 0.0) or 0.0)
    tp1 = idea.get("take_profit_1")
    tp2 = idea.get("take_profit_2")
    score = idea.get("strength_score")

    def _pips(level) -> Optional[float]:
        if level is None or not entry:
            return None
        return round(abs(float(level) - entry) / pip_size, 1)

    detail = {
        "entry": entry,
        "take_profit_1": tp1,
        "take_profit_2": tp2,
        "stop_loss": idea.get("stop_loss"),
        "strength_score": score,
        "confidence": idea.get("confidence"),
        "conviction": idea.get("conviction"),
        "thesis": idea.get("thesis"),
        "stop_loss_method": idea.get("stop_loss_method"),
    }

    return {
        "logged_at": datetime.now(),
        "instrument": pair,
        "ticker": ticker,
        "direction": idea.get("bias"),
        "session": session,
        "score": f"{score}/10" if score is not None else None,
        # verdict column is VARCHAR(20) — cap so a long conviction label can't
        # overflow the insert (the full text lives in `thesis` / `checks_detail`).
        "verdict": (str(idea["conviction"])[:20]
                    if idea.get("conviction") is not None else None),
        "atr14": idea.get("atr"),
        "atr20": None,
        "sl_pips": idea.get("stop_loss_pips"),
        "tp1_pips": _pips(tp1),
        "tp2_pips": _pips(tp2),
        "lot_size": None,
        "risk_amount": None,
        "rr_tp1": idea.get("risk_reward_1"),
        "rr_tp2": idea.get("risk_reward_2"),
        "account_bal": None,
        "risk_pct": None,
        "checks_passed": None,
        "checks_total": None,
        "checks_detail": json.dumps(detail),
        "notes": (
            f"Market Overview signal · {idea.get('conviction')} · "
            f"score {score}/10 · {str(idea.get('thesis', ''))[:160]}"
        ),
    }


def generate_weekly_swing_ideas(data_by_timeframe: Dict) -> List[Dict]:
    weekly_data = data_by_timeframe.get("Weekly", {})
    daily_data = data_by_timeframe.get("Daily", {})

    if not weekly_data:
        return []

    ideas = []

    for pair in config.assets:
        df_w = weekly_data.get(pair, pd.DataFrame())
        df_d = daily_data.get(pair, pd.DataFrame())

        if df_w.empty or len(df_w) < 20:
            continue

        df_w = analyzer.add_indicators(df_w.copy())
        last = df_w.iloc[-1]

        price = float(last["Close"])
        ema20 = float(last.get("EMA_20", price))
        ema50 = float(last.get("EMA_50", price))
        rsi = float(last.get("RSI", 50.0))
        adx = float(last.get("ADX", 0.0))
        atr = float(last.get("ATR", price * 0.01))

        # ── Weekly pivots (based on previous completed weekly candle) ─────────
        pivots = analyzer.calculate_pivots(df_w)
        if not pivots:
            continue
        pp = pivots["Pivot"]
        r1 = pivots["R1"]
        r2 = pivots["R2"]
        r3 = pivots["R3"]
        s1 = pivots["S1"]
        s2 = pivots["S2"]
        s3 = pivots["S3"]

        # ── Fibonacci levels over the last 12 weeks ───────────────────────────
        fibs = analyzer.calculate_fibonacci(df_w.tail(12))

        # ── Bias scoring ──────────────────────────────────────────────────────
        bull_s = bear_s = 0
        reasons: List[str] = []

        if price > ema20:
            bull_s += 2
        else:
            bear_s += 2

        if ema20 > ema50:
            bull_s += 2
            reasons.append("EMA20 > EMA50 — weekly uptrend")
        else:
            bear_s += 2
            reasons.append("EMA20 < EMA50 — weekly downtrend")

        if rsi > 55:
            bull_s += 1
            reasons.append(f"RSI {rsi:.0f} — bullish momentum")
        elif rsi < 45:
            bear_s += 1
            reasons.append(f"RSI {rsi:.0f} — bearish momentum")

        if price > pp:
            bull_s += 1
            reasons.append(f"Price above weekly pivot ({pp:.5f})")
        else:
            bear_s += 1
            reasons.append(f"Price below weekly pivot ({pp:.5f})")

        if adx > 20:
            if bull_s > bear_s:
                bull_s += 1
                reasons.append(f"ADX {adx:.0f} — trend is strong")
            else:
                bear_s += 1
                reasons.append(f"ADX {adx:.0f} — trend is strong")

        if bull_s == bear_s or adx < 15:
            continue  # skip flat / trendless markets

        bias = "Long" if bull_s > bear_s else "Short"

        # ── Entry / SL / TP ───────────────────────────────────────────────────
        fib_382 = fibs.get("38.2%", price)
        fib_500 = fibs.get("50.0%", price)
        fib_618 = fibs.get("61.8%", price)

        if bias == "Long":
            # Entry: current price (or tighten to S1 pullback zone)
            entry = price
            # SL: below S1 or below 61.8% fib — whichever is lower
            sl = min(s1, fib_618) - atr * 0.3
            tp1 = r1
            tp2 = r2
            tp3 = r3
        else:
            entry = price
            # SL: above R1 or above 38.2% fib — whichever is higher
            sl = max(r1, fib_382) + atr * 0.3
            tp1 = s1
            tp2 = s2
            tp3 = s3

        stop_dist = abs(entry - sl)
        if stop_dist == 0:
            continue

        rr1 = round(abs(tp1 - entry) / stop_dist, 2)
        rr2 = round(abs(tp2 - entry) / stop_dist, 2)
        rr3 = round(abs(tp3 - entry) / stop_dist, 2)

        if rr1 < 1.5:
            continue  # reject low-quality setups

        # ── Daily confirmation ────────────────────────────────────────────────
        daily_conf = "—"
        if not df_d.empty:
            df_d_ind = analyzer.add_indicators(df_d.copy())
            daily_sent = analyzer.get_sentiment(df_d_ind)
            if (bias == "Long" and daily_sent == "Bullish") or \
                    (bias == "Short" and daily_sent == "Bearish"):
                daily_conf = "✅ Aligned"
            elif daily_sent == "Neutral":
                daily_conf = "⚪ Neutral"
            else:
                daily_conf = "⚠️ Conflicting"

        ideas.append({
            "pair": pair, "bias": bias, "price": price,
            "entry": entry, "sl": sl,
            "tp1": tp1, "tp2": tp2, "tp3": tp3,
            "rr1": rr1, "rr2": rr2, "rr3": rr3,
            "rsi": rsi, "adx": adx, "atr": atr,
            "pivots": pivots, "fibs": fibs,
            "reasons": reasons, "daily_conf": daily_conf,
            "score": bull_s if bias == "Long" else bear_s,
            "df_w": df_w,
        })

    # Best R:R first, then highest score
    ideas.sort(key=lambda x: (x["rr1"], x["score"]), reverse=True)
    return ideas


def generate_supertrend_signals(df: pd.DataFrame, dema_period: int = 200,
                                st_period1: int = 10, st_mult1: float = 3.0,
                                st_period2: int = 12, st_mult2: float = 3.5,
                                qqe_rsi: int = 14, qqe_smooth: int = 5, qqe_factor: float = 2.618) -> pd.Series:
    """
    Consolidated Supertrend + QQE + DEMA strategy logic.
    - Both Supertrends must agree
    - Price must be on correct side of DEMA
    - QQE trend must confirm
    Returns a series with 1 (Buy), -1 (Sell), or 0 (Hold).
    """
    if df.empty:
        return pd.Series()

    st1 = analyzer.calculate_supertrend(df, st_period1, st_mult1)
    st2 = analyzer.calculate_supertrend(df, st_period2, st_mult2)
    dema = analyzer.calculate_dema(df["Close"], dema_period)
    qqe = analyzer.calculate_qqe(df["Close"], qqe_rsi, qqe_smooth, qqe_factor)

    signals = pd.Series(0, index=df.index)

    st_bullish = (st1["Trend"] == 1) & (st2["Trend"] == 1)
    st_bearish = (st1["Trend"] == -1) & (st2["Trend"] == -1)
    price_above_dema = df['Close'] > dema
    price_below_dema = df['Close'] < dema
    qqe_bullish = qqe["Trend"] == 1
    qqe_bearish = qqe["Trend"] == -1

    signals[st_bullish & price_above_dema & qqe_bullish] = 1
    signals[st_bearish & price_below_dema & qqe_bearish] = -1

    return signals


def evaluate_trend_following_signal(df: pd.DataFrame, min_conditions: int = 4) -> Tuple[str, int, int, Dict[str, bool], str]:
    """
    Evaluates trend following signal based on 50/200 EMA, RSI, MACD, and ADX.
    Returns (signal_label, score, max_score, conditions_dict, direction)
    """
    if len(df) < 200:
        return "⏳ NEUTRAL", 0, 6, {}, "NEUTRAL"

    df = analyzer.add_indicators(df)
    r = df.iloc[-1]
    close = float(r["Close"])

    # add_indicators supplies EMA_20/EMA_50 but not the 200 EMA this signal is
    # built around, so compute it here from the close. The 200-bar guard above
    # guarantees a meaningful (fully-seeded) value.
    e50 = float(r["EMA_50"])
    e200 = float(ema(df["Close"], 200).iloc[-1])
    rsi_val = float(r["RSI"])
    macd_v = float(r["MACD"])
    macd_s = float(r["MACD_Signal"])
    adx_v = float(r["ADX"])

    buy_conds = {
        "Price above 200 EMA": close > e200,
        "50 EMA above 200 EMA (golden)": e50 > e200,
        "Price at/above 50 EMA": close >= e50 * 0.9985,
        "RSI 45–70 (bullish momentum)": 45 <= rsi_val <= 70,
        "MACD above Signal line": macd_v > macd_s,
        "Strong trend (ADX > 25)": adx_v > 25,
    }
    sell_conds = {
        "Price below 200 EMA": close < e200,
        "50 EMA below 200 EMA (death)": e50 < e200,
        "Price at/below 50 EMA": close <= e50 * 1.0015,
        "RSI 30–55 (bearish momentum)": 30 <= rsi_val <= 55,
        "MACD below Signal line": macd_v < macd_s,
        "Strong trend (ADX > 25)": adx_v > 25,
    }

    bs = sum(buy_conds.values())
    ss = sum(sell_conds.values())

    if bs >= min_conditions and bs >= ss:
        direction = "STRONG_BUY" if bs >= 5 else "BUY"
        label = "🚀 STRONG BUY" if bs >= 5 else "📈 BUY"
        return label, bs, 6, buy_conds, direction
    elif ss >= min_conditions and ss > bs:
        direction = "STRONG_SELL" if ss >= 5 else "SELL"
        label = "🔻 STRONG SELL" if ss >= 5 else "📉 SELL"
        return label, ss, 6, sell_conds, direction
    else:
        return "⏳ NEUTRAL", max(bs, ss), 6, {}, "NEUTRAL"