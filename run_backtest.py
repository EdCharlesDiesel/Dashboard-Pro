import sys, warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, date

def calc_ema(s, p): return s.ewm(span=p, adjust=False).mean()
def calc_rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))
def calc_macd(s, fast=12, slow=26, signal=9):
    h = calc_ema(s, fast) - calc_ema(s, slow)
    return h - calc_ema(h, signal)
def calc_atr(df, p=14):
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift()).abs()
    lc = (df['Low']  - df['Close'].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).ewm(alpha=1/p, adjust=False).mean()
def calc_stochastic(df, k=14, d=3):
    lk = df['Low'].rolling(k).min()
    hk = df['High'].rolling(k).max()
    pk = 100 * (df['Close'] - lk) / (hk - lk + 1e-9)
    return pk, pk.rolling(d).mean()
def swing_highs(s, strength=3):
    idx = []
    for i in range(strength, len(s) - strength):
        if s.iloc[i] == s.iloc[i-strength:i+strength+1].max():
            idx.append(i)
    return idx
def swing_lows(s, strength=3):
    idx = []
    for i in range(strength, len(s) - strength):
        if s.iloc[i] == s.iloc[i-strength:i+strength+1].min():
            idx.append(i)
    return idx

CHECK_META = [
    ('macro_bias',       '01. Macro Bias Confirmed',    'auto', True),
    ('news_filter',      '02. News Filter Clear',       'auto', True),
    ('correlations',     '03. Correlation Exposure OK', 'auto', False),
    ('atr_volatility',   '04. ATR Volatility OK',       'calc', False),
    ('weekly_ema',       '05. Weekly EMA Aligned',      'calc', True),   # hard block
    ('weekly_rsi',       '06. Weekly RSI has Room',     'calc', False),
    ('weekly_swing',     '07. Weekly Swing Structure',  'calc', False),
    ('daily_trend',      '08. Daily Trend Intact',      'calc', True),   # hard block
    ('daily_macd',       '09. Daily MACD Momentum',     'calc', False),
    ('4h_confluence',    '10. 4H Confluence Zone',      'calc', False),
    ('confluence_check', '11. 2/3 Confluences Met',     'calc', False),
    ('rejection',        '12. 15M Rejection Candle',    'calc', False),
    ('entry_signal',     '13. 15M Entry Signal',        'calc', False),
    ('stop_structure',   '14. Stop Below Structure',    'calc', True),   # hard block
    ('rr_check',         '15. R:R >= 2:1',              'calc', True),   # hard block
    ('loss_limit',       '16. Daily Loss Limit OK',     'auto', False),
    ('mkt_structure',    '17. Market Structure Intact', 'calc', False),
    ('setup_rank',       '18. Setup Score >= 14/18',    'calc', False),
]

def run_checks(d, w, direction):
    res = {}
    for key in ('macro_bias', 'news_filter', 'correlations', 'loss_limit'):
        res[key] = (True, 'Auto-pass')

    if len(d) >= 20:
        atr = calc_atr(d, 14).iloc[-1]; price = d['Close'].iloc[-1]; ratio = atr / price
        res['atr_volatility'] = (0.001 <= ratio <= 0.05, f'ATR/Price={ratio*100:.2f}%')
    else:
        res['atr_volatility'] = (False, 'Insufficient data')

    if len(w) >= 55:
        e20 = calc_ema(w['Close'], 20).iloc[-1]; e50 = calc_ema(w['Close'], 50).iloc[-1]; wp = w['Close'].iloc[-1]
        ok = (wp > e20 > e50) if direction == 'Long' else (wp < e20 < e50)
        res['weekly_ema'] = (ok, f'EMA20={e20:.5f} EMA50={e50:.5f}')
    else:
        res['weekly_ema'] = (False, 'Insufficient weekly data')

    if len(w) >= 18:
        wr = calc_rsi(w['Close'], 14).iloc[-1]
        ok = (wr < 65) if direction == 'Long' else (wr > 35)
        res['weekly_rsi'] = (ok, f'Weekly RSI={wr:.1f}')
    else:
        res['weekly_rsi'] = (False, 'Insufficient weekly data')

    sh = swing_highs(w['High'], 2); sl_w = swing_lows(w['Low'], 2)
    if len(sh) >= 2 and len(sl_w) >= 2:
        hh = w['High'].iloc[sh[-1]] > w['High'].iloc[sh[-2]]
        hl = w['Low'].iloc[sl_w[-1]] > w['Low'].iloc[sl_w[-2]]
        ll = w['Low'].iloc[sl_w[-1]] < w['Low'].iloc[sl_w[-2]]
        lh = w['High'].iloc[sh[-1]] < w['High'].iloc[sh[-2]]
        if direction == 'Long':
            ok = hh and hl; res['weekly_swing'] = (ok, f'HH={hh} HL={hl}')
        else:
            ok = ll and lh; res['weekly_swing'] = (ok, f'LL={ll} LH={lh}')
    else:
        res['weekly_swing'] = (False, 'Not enough weekly swings')

    if len(d) >= 52:
        e20 = calc_ema(d['Close'], 20).iloc[-1]; e50 = calc_ema(d['Close'], 50).iloc[-1]; dp = d['Close'].iloc[-1]
        ok = (e20 > e50 and dp > e20) if direction == 'Long' else (e20 < e50 and dp < e20)
        res['daily_trend'] = (ok, f'EMA20={e20:.5f} EMA50={e50:.5f}')
    else:
        res['daily_trend'] = (False, 'Insufficient data')

    if len(d) >= 35:
        hist = calc_macd(d['Close']); h = hist.iloc[-1]; hp = hist.iloc[-2]
        ok = (h > 0 or (h > hp and h > -0.001)) if direction == 'Long' else (h < 0 or (h < hp and h < 0.001))
        res['daily_macd'] = (ok, f'MACD={h:.6f}')
    else:
        res['daily_macd'] = (False, 'Insufficient data')

    if len(d) >= 20:
        atr = calc_atr(d, 14).iloc[-1]; price = d['Close'].iloc[-1]; tol = atr * 0.6
        rh = d['High'].rolling(20).max().iloc[-1]; rl = d['Low'].rolling(20).min().iloc[-1]
        fr = rh - rl; fibs = [rh - r * fr for r in (0.236, 0.382, 0.5, 0.618, 0.786)]
        nf = any(abs(price - f) <= tol for f in fibs)
        prev = d.iloc[-2]; pp = (prev['High'] + prev['Low'] + prev['Close']) / 3
        s1, r1 = 2*pp - prev['High'], 2*pp - prev['Low']
        np2 = any(abs(price - lvl) <= tol for lvl in (pp, s1, r1))
        ne = abs(price - calc_ema(d['Close'], 20).iloc[-1]) <= tol
        cnt = sum([nf, np2, ne])
        res['4h_confluence']    = (cnt >= 1, f'Fib={nf} Pivot={np2} EMA={ne} ({cnt}/3)')
        res['confluence_check'] = (cnt >= 2, f'{cnt}/3 confluences')
    else:
        res['4h_confluence']    = (False, 'Insufficient data')
        res['confluence_check'] = (False, 'Insufficient data')

    if len(d) >= 3:
        bar = d.iloc[-1]; rng = bar['High'] - bar['Low']
        if rng > 0:
            pos = (bar['Close'] - bar['Low']) / rng
            ok = (pos >= 0.60) if direction == 'Long' else (pos <= 0.40)
            res['rejection'] = (ok, f'Close pos={pos*100:.0f}%')
        else:
            res['rejection'] = (False, 'Doji')
    else:
        res['rejection'] = (False, 'Insufficient data')

    if len(d) >= 20:
        k, kd = calc_stochastic(d)
        kn, kp = k.iloc[-1], k.iloc[-2]; dn2, dp2 = kd.iloc[-1], kd.iloc[-2]
        if direction == 'Long':
            cu = kp < dp2 and kn >= dn2; ok = cu and kp < 50
        else:
            cd = kp > dp2 and kn <= dn2; ok = cd and kp > 50
        res['entry_signal'] = (ok, f'K={kn:.1f} D={dn2:.1f}')
    else:
        res['entry_signal'] = (False, 'Insufficient data')

    if len(d) >= 12:
        price = d['Close'].iloc[-1]; atr = calc_atr(d, 14).iloc[-1]
        shi = swing_highs(d['High'], 3); sli = swing_lows(d['Low'], 3)
        if direction == 'Long' and sli:
            lvl = d['Low'].iloc[sli[-1]]; dist = price - lvl; ok = 0 < dist < atr * 3
            res['stop_structure'] = (ok, f'SwingLow={lvl:.5f} dist={dist:.5f}')
        elif direction == 'Short' and shi:
            lvl = d['High'].iloc[shi[-1]]; dist = lvl - price; ok = 0 < dist < atr * 3
            res['stop_structure'] = (ok, f'SwingHigh={lvl:.5f} dist={dist:.5f}')
        else:
            res['stop_structure'] = (False, 'No swing point')
    else:
        res['stop_structure'] = (False, 'Insufficient data')

    if len(d) >= 14:
        atr = calc_atr(d, 14).iloc[-1]; sl_d = atr * 1.5; tp1_d = sl_d * 2.0
        res['rr_check'] = (True, f'R:R={tp1_d/sl_d:.1f}:1')
    else:
        res['rr_check'] = (False, 'Insufficient data')

    shi = swing_highs(d['High'], 3); sli = swing_lows(d['Low'], 3)
    if len(shi) >= 2 and len(sli) >= 2:
        hh = d['High'].iloc[shi[-1]] > d['High'].iloc[shi[-2]]
        hl = d['Low'].iloc[sli[-1]]  > d['Low'].iloc[sli[-2]]
        ll = d['Low'].iloc[sli[-1]]  < d['Low'].iloc[sli[-2]]
        lh = d['High'].iloc[shi[-1]] < d['High'].iloc[shi[-2]]
        if direction == 'Long':
            ok = hh and hl; res['mkt_structure'] = (ok, f'HH={hh} HL={hl}')
        else:
            ok = ll and lh; res['mkt_structure'] = (ok, f'LL={ll} LH={lh}')
    else:
        res['mkt_structure'] = (False, 'Not enough swing history')

    score = sum(v for v, _ in res.values())
    res['setup_rank'] = (score >= 14, f'Score={score}/18')
    return res, score

def simulate_trade(df, entry_idx, direction, sl_mult=1.5, rr=2.0, max_bars=20):
    entry = df['Open'].iloc[entry_idx]
    atr   = calc_atr(df.iloc[:entry_idx], 14).iloc[-1]
    sl_d  = atr * sl_mult
    if direction == 'Long':
        sl, tp1, tp2 = entry - sl_d, entry + sl_d * rr, entry + sl_d * (rr + 1)
    else:
        sl, tp1, tp2 = entry + sl_d, entry - sl_d * rr, entry - sl_d * (rr + 1)
    outcome = 'Timeout'; exit_price = df['Close'].iloc[min(entry_idx + max_bars, len(df) - 1)]
    tp1_hit = False; active_sl = sl
    for i in range(entry_idx, min(entry_idx + max_bars, len(df))):
        bar = df.iloc[i]
        if direction == 'Long':
            if bar['Low'] <= active_sl:
                outcome = 'TP1->BE' if tp1_hit else 'SL'; exit_price = active_sl; break
            if not tp1_hit and bar['High'] >= tp1:
                tp1_hit = True; active_sl = entry
            if tp1_hit and bar['High'] >= tp2:
                outcome = 'TP2'; exit_price = tp2; break
        else:
            if bar['High'] >= active_sl:
                outcome = 'TP1->BE' if tp1_hit else 'SL'; exit_price = active_sl; break
            if not tp1_hit and bar['Low'] <= tp1:
                tp1_hit = True; active_sl = entry
            if tp1_hit and bar['Low'] <= tp2:
                outcome = 'TP2'; exit_price = tp2; break
    r = ((exit_price - entry) / sl_d) if direction == 'Long' else ((entry - exit_price) / sl_d)
    return dict(entry=entry, exit=exit_price, sl=sl, tp1=tp1, outcome=outcome, r=round(r, 2), sl_d=sl_d)

def run_backtest(daily, weekly, direction, min_score, sl_mult, rr, start_dt, end_dt):
    trades = []; warmup = 55; in_trade = False
    for i in range(warmup, len(daily)):
        bd = daily.index[i].date()
        if bd < start_dt or bd > end_dt: continue
        if in_trade: in_trade = False; continue
        d_s = daily.iloc[:i+1]; w_s = weekly[weekly.index <= daily.index[i]]
        if len(w_s) < 20: continue
        res, score = run_checks(d_s, w_s, direction)
        crit = [k for k, _, mode, c in CHECK_META if mode == 'calc' and c]
        cp = all(res.get(k, (False,))[0] for k in crit)
        if score >= min_score and cp:
            if i + 1 >= len(daily): continue
            t = simulate_trade(daily, i+1, direction, sl_mult, rr)
            trades.append({'date': bd, 'score': score, **t})
            in_trade = True
    return trades

# ── Instruments to test ──────────────────────────────────────────
RUNS = [
    ('GBP/JPY', 'GBPJPY=X'),
    ('Gold',    'GC=F'),
]

end_dt   = date.today()
start_dt = end_dt - timedelta(days=180)

def print_results(name, trades, direction):
    if not trades:
        print(f'  {name} {direction:<6} : NO TRADES in 180-day window')
        return
    df_t   = pd.DataFrame(trades)
    wins   = df_t[df_t['r'] > 0]
    losses = df_t[df_t['r'] <= 0]
    net_r  = df_t['r'].sum()
    avg_r  = df_t['r'].mean()
    win_r  = len(wins) / len(df_t) * 100
    gp     = wins['r'].sum() if len(wins) else 0
    gl     = abs(losses['r'].sum()) if len(losses) else 1
    pf     = gp / gl if gl else 0
    cumr   = df_t['r'].cumsum()
    max_dd = (cumr - cumr.cummax()).min()

    print('=' * 60)
    print(f'  {name} {direction}  |  180-day backtest results')
    print('=' * 60)
    print(f'  Total trades   : {len(df_t)}')
    print(f'  Win rate       : {win_r:.0f}%  ({len(wins)} wins / {len(losses)} losses)')
    print(f'  Net R          : {net_r:+.1f}R')
    print(f'  Avg R / trade  : {avg_r:+.2f}R')
    print(f'  Profit factor  : {pf:.2f}')
    print(f'  Max drawdown   : {max_dd:.1f}R')
    print(f'  P&L (1% risk, $10k) : ${net_r*(10000*0.01):+,.0f}')
    print('=' * 60)
    print(f'  {"DATE":<12} {"SCORE":>5} {"ENTRY":>10} {"EXIT":>10} {"R":>7}  OUTCOME')
    print('  ' + '-' * 60)
    for t in trades:
        print(f"  {str(t['date']):<12} {t['score']:>5} {float(t['entry']):>10.3f} {float(t['exit']):>10.3f} {t['r']:>+7.2f}R  {t['outcome']}")
    print()

for name, ticker in RUNS:
    print(f'\nDownloading {name} ({ticker})...')
    daily  = yf.download(ticker, period='1y',  interval='1d',  auto_adjust=True, progress=False)
    weekly = yf.download(ticker, period='3y',  interval='1wk', auto_adjust=True, progress=False)
    if hasattr(daily.columns, 'levels'):  daily.columns  = daily.columns.get_level_values(0)
    if hasattr(weekly.columns, 'levels'): weekly.columns = weekly.columns.get_level_values(0)
    daily.index  = pd.to_datetime(daily.index).tz_localize(None)
    weekly.index = pd.to_datetime(weekly.index).tz_localize(None)
    daily = daily.dropna(); weekly = weekly.dropna()
    print(f'  Daily bars: {len(daily)}  |  Weekly bars: {len(weekly)}')
    print(f'  Window: {start_dt} to {end_dt}  |  Min score 14/18')

    for direction in ('Long', 'Short'):
        trades = run_backtest(daily, weekly, direction, 14, 1.5, 2.0, start_dt, end_dt)
        print_results(name, trades, direction)
