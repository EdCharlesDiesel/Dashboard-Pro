import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from src.core.analyzer import TechnicalAnalyzer as analyzer
from src.core.config import default_config as config

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
            'signal':     signal,
            'confidence': min(confidence, 5),
            'reasons':    reasons,
            'stoch_k':    k,
            'stoch_d':    d,
            'rsi':        rsi,
            'price':      price,
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

    def calculate(self, df: pd.DataFrame, pair: str, bias: str, current_price: float, atr: float, lookback: int = 20) -> Dict:
        atr_mult = config.pair_atr_multipliers.get(pair, config.atr_sl_mult)
        min_dist = config.pair_min_stop.get(pair, 0.0010)
        buffer   = atr * 0.25

        atr_stop = (current_price - atr * atr_mult) if bias == 'Long' else (current_price + atr * atr_mult)
        swing  = self.get_swing_stop(df, bias, lookback)
        stop   = atr_stop
        method = "ATR"

        if swing is not None:
            if bias == 'Long':
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
            stop   = (current_price - min_dist) if bias == 'Long' else (current_price + min_dist)
            method += " + min-dist enforced"

        return {
            "stop":          stop,
            "method":        method,
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

    def calculate(self, df: pd.DataFrame, pair: str, bias: str, current_price: float, atr: float, stop_loss: float, lookback: int = 20) -> Dict:
        stop_dist = abs(current_price - stop_loss) or atr
        swing     = self.get_swing_target(df, bias, lookback)

        if bias == 'Long':
            tp1_atr = current_price + atr * config.tp1_atr_mult
            tp2_atr = current_price + atr * config.tp2_atr_mult
            if swing is not None and current_price < swing < tp1_atr:
                tp1, m1 = swing, "Swing High"
            else:
                tp1, m1 = tp1_atr, f"ATR ×{config.tp1_atr_mult}"
            if swing is not None and tp1 < swing < tp2_atr:
                tp2, m2 = swing, "Swing High (ext)"
            else:
                tp2, m2 = tp2_atr, f"ATR ×{config.tp2_atr_mult}"
            rr1 = (tp1 - current_price) / stop_dist
            rr2 = (tp2 - current_price) / stop_dist
        else:
            tp1_atr = current_price - atr * config.tp1_atr_mult
            tp2_atr = current_price - atr * config.tp2_atr_mult
            if swing is not None and tp1_atr < swing < current_price:
                tp1, m1 = swing, "Swing Low"
            else:
                tp1, m1 = tp1_atr, f"ATR ×{config.tp1_atr_mult}"
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

entry_generator = EntrySignalGenerator()
sl_calculator = StopLossCalculator()
tp_calculator = TakeProfitCalculator()

def analyze_multi_timeframe(df_daily: pd.DataFrame, df_4h: pd.DataFrame, df_1h: pd.DataFrame, df_15m: pd.DataFrame, pair_name: str) -> Optional[Dict]:
    if any(df.empty for df in [df_daily, df_4h, df_1h, df_15m]):
        return None

    daily     = df_daily.iloc[-1]
    four_hour = df_4h.iloc[-1]
    one_hour  = df_1h.iloc[-1]
    fifteen_m = df_15m.iloc[-1]

    if 'Close' not in daily.index or 'Close' not in four_hour.index:
        return None

    d_close = safe_get(daily, 'Close')
    d_ema20 = safe_get(daily, 'EMA_20', d_close)
    d_trend = 'Long' if d_close > d_ema20 else 'Short'
    d_rsi   = safe_get(daily, 'RSI', 50.0)
    d_adx   = safe_get(daily, 'ADX', 0.0)

    h4_close    = safe_get(four_hour, 'Close')
    h4_ema20    = safe_get(four_hour, 'EMA_20', h4_close)
    h4_ema50    = safe_get(four_hour, 'EMA_50', h4_close)
    h4_trend    = 'Long' if h4_ema20 > h4_ema50 else 'Short'
    h4_macd     = safe_get(four_hour, 'MACD', 0.0)
    h4_sig      = safe_get(four_hour, 'MACD_Signal', 0.0)
    h4_macd_bull = h4_macd > h4_sig

    h1_close = safe_get(one_hour, 'Close')
    h1_ema20 = safe_get(one_hour, 'EMA_20', h1_close)
    h1_ema50 = safe_get(one_hour, 'EMA_50', h1_close)
    h1_trend = 'Long' if h1_ema20 > h1_ema50 else 'Short'
    h1_rsi   = safe_get(one_hour, 'RSI', 50.0)

    long_s = short_s = 0
    reasons: List[str] = []

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

    if h4_trend == 'Long':
        long_s += 1;  reasons.append("4H: EMA20 > EMA50")
    else:
        short_s += 1; reasons.append("4H: EMA20 < EMA50")

    if h4_macd_bull:
        long_s += 1;  reasons.append("4H: MACD bullish")
    else:
        short_s += 1; reasons.append("4H: MACD bearish")

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
        return None

    conviction   = "High" if strength >= 6 else ("Medium" if strength >= 3 else "Low")
    entry_signal = entry_generator.get_entry_signal(df_15m, final_bias)

    atr = safe_get(one_hour, 'ATR', 0.0)
    if atr <= 0:
        atr = h1_close * 0.005 if h1_close > 0 else 0.001

    current_price = safe_get(fifteen_m, 'Close', 0.0)
    if current_price <= 0.0:
        return None

    sl_result = sl_calculator.calculate(df_1h, pair_name, final_bias, current_price, atr)
    tp_result = tp_calculator.calculate(df_4h, pair_name, final_bias, current_price, atr, sl_result["stop"])

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

def generate_trading_ideas(data_by_timeframe: Dict) -> Tuple[List[Dict], List[str]]:
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

        idea = analyze_multi_timeframe(frames['Daily'], frames['4 Hour'], frames['Hourly'], frames['15 Minute'], pair_name)
        if idea and idea['bias'] != 'Neutral':
            ideas.append(idea)

    ideas.sort(key=lambda x: (x['conviction'] == 'High', x['strength_score']), reverse=True)
    return ideas, skipped
