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


def analyze_multi_timeframe(df_daily: pd.DataFrame, df_4h: pd.DataFrame, df_1h: pd.DataFrame, df_15m: pd.DataFrame,
                            pair_name: str) -> Optional[Dict]:
    if any(df.empty for df in [df_daily, df_4h, df_1h, df_15m]):
        return None

    daily = df_daily.iloc[-1]
    four_hour = df_4h.iloc[-1]
    one_hour = df_1h.iloc[-1]
    fifteen_m = df_15m.iloc[-1]

    if 'Close' not in daily.index or 'Close' not in four_hour.index:
        return None

    d_close = safe_get(daily, 'Close')
    d_ema20 = safe_get(daily, 'EMA_20', d_close)
    d_trend = 'Long' if d_close > d_ema20 else 'Short'
    d_rsi = safe_get(daily, 'RSI', 50.0)
    d_adx = safe_get(daily, 'ADX', 0.0)

    h4_close = safe_get(four_hour, 'Close')
    h4_ema20 = safe_get(four_hour, 'EMA_20', h4_close)
    h4_ema50 = safe_get(four_hour, 'EMA_50', h4_close)
    h4_trend = 'Long' if h4_ema20 > h4_ema50 else 'Short'
    h4_macd = safe_get(four_hour, 'MACD', 0.0)
    h4_sig = safe_get(four_hour, 'MACD_Signal', 0.0)
    h4_macd_bull = h4_macd > h4_sig

    h1_close = safe_get(one_hour, 'Close')
    h1_ema20 = safe_get(one_hour, 'EMA_20', h1_close)
    h1_ema50 = safe_get(one_hour, 'EMA_50', h1_close)
    h1_trend = 'Long' if h1_ema20 > h1_ema50 else 'Short'
    h1_rsi = safe_get(one_hour, 'RSI', 50.0)

    long_s = short_s = 0
    reasons: List[str] = []

    if d_trend == 'Long':
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
        if d_trend == 'Long':
            long_s += 1
        else:
            short_s += 1
        reasons.append(f"Strong trend (ADX={d_adx:.1f})")

    if h4_trend == 'Long':
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

    if h1_trend == 'Long':
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

    if long_s > short_s:
        final_bias, strength = 'Long', long_s
    elif short_s > long_s:
        final_bias, strength = 'Short', short_s
    else:
        return None

    # strength_score = MTF alignment strength (how many timeframes agree), scaled 1-10
    normalized_score = min(int(strength * 1.25), 10)

    conviction = "High" if strength >= 6 else ("Medium" if strength >= 3 else "Low")
    entry_signal = entry_generator.get_entry_signal(df_15m, final_bias)

    # confidence = 15-min entry trigger quality (0-5 from entry_signal, scaled to 0-10).
    # This is deliberately separate from strength_score: a trade can have strong MTF
    # alignment but a poor entry setup (low confidence), or vice versa.
    entry_conf = entry_signal.get('confidence', 0) if entry_signal else 0
    confidence = min(entry_conf * 2, 10)
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

    thesis = " | ".join(reasons)
    if entry_signal and entry_signal['signal'] != 0:
        thesis += f" | Entry: {', '.join(entry_signal['reasons'][:2])}"

    return {
        "pair": pair_name,
        "bias": final_bias,
        "conviction": conviction,
        "strength_score": normalized_score,
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
        frames = {
            tf: data_by_timeframe.get(tf, {}).get(pair_name, pd.DataFrame())
            for tf in ['Daily', '4 Hour', 'Hourly', '15 Minute']
        }
        thin = [tf for tf, df in frames.items() if df.empty or len(df) < 20]
        if thin:
            skipped.append(f"{pair_name} — insufficient bars in: {', '.join(thin)}")
            continue

        idea = analyze_multi_timeframe(frames['Daily'], frames['4 Hour'], frames['Hourly'], frames['15 Minute'],
                                       pair_name)
        if idea and idea['bias'] != 'Neutral':
            ideas.append(idea)

    ideas.sort(key=lambda x: (x['conviction'] == 'High', x['strength_score']), reverse=True)
    return ideas, skipped


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