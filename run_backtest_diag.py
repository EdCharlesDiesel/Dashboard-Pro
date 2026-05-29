import warnings; warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta

def calc_ema(s, p): return s.ewm(span=p, adjust=False).mean()
def calc_rsi(s, p=14):
    d = s.diff(); g = d.clip(lower=0).rolling(p).mean(); l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))
def calc_macd(s, fast=12, slow=26, signal=9):
    h = calc_ema(s, fast) - calc_ema(s, slow); return h - calc_ema(h, signal)
def calc_atr(df, p=14):
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift()).abs()
    lc = (df['Low']  - df['Close'].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).ewm(alpha=1/p, adjust=False).mean()
def calc_stochastic(df, k=14, d=3):
    lk = df['Low'].rolling(k).min(); hk = df['High'].rolling(k).max()
    pk = 100 * (df['Close'] - lk) / (hk - lk + 1e-9); return pk, pk.rolling(d).mean()
def swing_highs(s, strength=3):
    idx = []
    for i in range(strength, len(s) - strength):
        if s.iloc[i] == s.iloc[i-strength:i+strength+1].max(): idx.append(i)
    return idx
def swing_lows(s, strength=3):
    idx = []
    for i in range(strength, len(s) - strength):
        if s.iloc[i] == s.iloc[i-strength:i+strength+1].min(): idx.append(i)
    return idx

CHECK_META = [
    ('macro_bias',       '01. Macro Bias',        'auto', True),
    ('news_filter',      '02. News Filter',        'auto', True),
    ('correlations',     '03. Correlations',       'auto', False),
    ('atr_volatility',   '04. ATR Volatility',     'calc', False),
    ('weekly_ema',       '05. Weekly EMA',         'calc', True),
    ('weekly_rsi',       '06. Weekly RSI',         'calc', True),
    ('weekly_swing',     '07. Weekly Swing',       'calc', True),
    ('daily_trend',      '08. Daily Trend',        'calc', True),
    ('daily_macd',       '09. Daily MACD',         'calc', True),
    ('4h_confluence',    '10. 4H Confluence',      'calc', True),
    ('confluence_check', '11. 2/3 Confluence',     'calc', True),
    ('rejection',        '12. 15M Rejection',      'calc', True),
    ('entry_signal',     '13. 15M Entry Signal',   'calc', True),
    ('stop_structure',   '14. Stop Structure',     'calc', True),
    ('rr_check',         '15. R:R',                'calc', True),
    ('loss_limit',       '16. Loss Limit',         'auto', False),
    ('mkt_structure',    '17. Mkt Structure',      'calc', True),
    ('setup_rank',       '18. Setup Score',        'calc', False),
]

def run_checks(d, w, direction):
    res = {}
    for key in ('macro_bias', 'news_filter', 'correlations', 'loss_limit'):
        res[key] = (True, 'Auto-pass')
    if len(d) >= 20:
        atr = calc_atr(d, 14).iloc[-1]; price = d['Close'].iloc[-1]; ratio = atr / price
        res['atr_volatility'] = (0.001 <= ratio <= 0.05, f'ATR/Price={ratio*100:.2f}%')
    else: res['atr_volatility'] = (False, 'n/a')
    if len(w) >= 55:
        e20 = calc_ema(w['Close'], 20).iloc[-1]; e50 = calc_ema(w['Close'], 50).iloc[-1]; wp = w['Close'].iloc[-1]
        ok = (wp > e20 > e50) if direction == 'Long' else (wp < e20 < e50)
        res['weekly_ema'] = (ok, f'price={wp:.5f} EMA20={e20:.5f} EMA50={e50:.5f}')
    else: res['weekly_ema'] = (False, 'n/a')
    if len(w) >= 18:
        wr = calc_rsi(w['Close'], 14).iloc[-1]
        ok = (wr < 65) if direction == 'Long' else (wr > 35)
        res['weekly_rsi'] = (ok, f'RSI={wr:.1f}')
    else: res['weekly_rsi'] = (False, 'n/a')
    sh = swing_highs(w['High'], 2); sl_w = swing_lows(w['Low'], 2)
    if len(sh) >= 2 and len(sl_w) >= 2:
        hh = w['High'].iloc[sh[-1]] > w['High'].iloc[sh[-2]]
        hl = w['Low'].iloc[sl_w[-1]] > w['Low'].iloc[sl_w[-2]]
        ll = w['Low'].iloc[sl_w[-1]] < w['Low'].iloc[sl_w[-2]]
        lh = w['High'].iloc[sh[-1]] < w['High'].iloc[sh[-2]]
        if direction == 'Long': ok = hh and hl; res['weekly_swing'] = (ok, f'HH={hh} HL={hl}')
        else: ok = ll and lh; res['weekly_swing'] = (ok, f'LL={ll} LH={lh}')
    else: res['weekly_swing'] = (False, 'n/a')
    if len(d) >= 52:
        e20 = calc_ema(d['Close'], 20).iloc[-1]; e50 = calc_ema(d['Close'], 50).iloc[-1]; dp = d['Close'].iloc[-1]
        ok = (e20 > e50 and dp > e20) if direction == 'Long' else (e20 < e50 and dp < e20)
        res['daily_trend'] = (ok, f'price={dp:.5f} EMA20={e20:.5f} EMA50={e50:.5f}')
    else: res['daily_trend'] = (False, 'n/a')
    if len(d) >= 35:
        hist = calc_macd(d['Close']); h = hist.iloc[-1]; hp = hist.iloc[-2]
        ok = (h > 0 or (h > hp and h > -0.001)) if direction == 'Long' else (h < 0 or (h < hp and h < 0.001))
        res['daily_macd'] = (ok, f'hist={h:.6f}')
    else: res['daily_macd'] = (False, 'n/a')
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
        res['4h_confluence']    = (cnt >= 1, f'{cnt}/3 (fib={nf},piv={np2},ema={ne})')
        res['confluence_check'] = (cnt >= 2, f'{cnt}/3 confluences')
    else:
        res['4h_confluence'] = (False, 'n/a'); res['confluence_check'] = (False, 'n/a')
    if len(d) >= 3:
        bar = d.iloc[-1]; rng = bar['High'] - bar['Low']
        if rng > 0:
            pos = (bar['Close'] - bar['Low']) / rng
            ok = (pos >= 0.60) if direction == 'Long' else (pos <= 0.40)
            res['rejection'] = (ok, f'pos={pos*100:.0f}%')
        else: res['rejection'] = (False, 'doji')
    else: res['rejection'] = (False, 'n/a')
    if len(d) >= 20:
        k, kd = calc_stochastic(d); kn, kp = k.iloc[-1], k.iloc[-2]; dn2, dp2 = kd.iloc[-1], kd.iloc[-2]
        if direction == 'Long':
            cu = kp < dp2 and kn >= dn2; ok = cu and kp < 50
        else:
            cd = kp > dp2 and kn <= dn2; ok = cd and kp > 50
        res['entry_signal'] = (ok, f'K={kn:.1f} D={dn2:.1f} cross={cu if direction=="Long" else cd}')
    else: res['entry_signal'] = (False, 'n/a')
    if len(d) >= 12:
        price = d['Close'].iloc[-1]; atr = calc_atr(d, 14).iloc[-1]
        shi = swing_highs(d['High'], 3); sli = swing_lows(d['Low'], 3)
        if direction == 'Long' and sli:
            lvl = d['Low'].iloc[sli[-1]]; dist = price - lvl; ok = 0 < dist < atr * 3
            res['stop_structure'] = (ok, f'dist={dist:.5f} ({dist/atr:.1f}xATR)')
        elif direction == 'Short' and shi:
            lvl = d['High'].iloc[shi[-1]]; dist = lvl - price; ok = 0 < dist < atr * 3
            res['stop_structure'] = (ok, f'dist={dist:.5f} ({dist/atr:.1f}xATR)')
        else: res['stop_structure'] = (False, 'no swing point')
    else: res['stop_structure'] = (False, 'n/a')
    if len(d) >= 14:
        atr = calc_atr(d, 14).iloc[-1]; sl_d = atr * 1.5
        res['rr_check'] = (True, f'R:R={sl_d*2/sl_d:.1f}:1')
    else: res['rr_check'] = (False, 'n/a')
    shi = swing_highs(d['High'], 3); sli = swing_lows(d['Low'], 3)
    if len(shi) >= 2 and len(sli) >= 2:
        hh = d['High'].iloc[shi[-1]] > d['High'].iloc[shi[-2]]
        hl = d['Low'].iloc[sli[-1]]  > d['Low'].iloc[sli[-2]]
        ll = d['Low'].iloc[sli[-1]]  < d['Low'].iloc[sli[-2]]
        lh = d['High'].iloc[shi[-1]] < d['High'].iloc[shi[-2]]
        if direction == 'Long': ok = hh and hl; res['mkt_structure'] = (ok, f'HH={hh} HL={hl}')
        else: ok = ll and lh; res['mkt_structure'] = (ok, f'LL={ll} LH={lh}')
    else: res['mkt_structure'] = (False, 'n/a')
    score = sum(v for v, _ in res.values())
    res['setup_rank'] = (score >= 14, f'Score={score}/18')
    return res, score

print('Downloading EUR/USD...')
daily  = yf.download('EURUSD=X', period='1y',  interval='1d',  auto_adjust=True, progress=False)
weekly = yf.download('EURUSD=X', period='3y',  interval='1wk', auto_adjust=True, progress=False)
if hasattr(daily.columns, 'levels'):  daily.columns  = daily.columns.get_level_values(0)
if hasattr(weekly.columns, 'levels'): weekly.columns = weekly.columns.get_level_values(0)
daily.index  = pd.to_datetime(daily.index).tz_localize(None)
weekly.index = pd.to_datetime(weekly.index).tz_localize(None)
daily = daily.dropna(); weekly = weekly.dropna()

end_dt   = date.today()
start_dt = end_dt - timedelta(days=180)

# ── Per-check pass rate over every day in the 180-day window ──
print(f'\nCheck pass rates  (EUR/USD, Long & Short, {start_dt} to {end_dt})\n')
warmup = 55
pass_long  = {k: 0 for k, *_ in CHECK_META}
pass_short = {k: 0 for k, *_ in CHECK_META}
total = 0
scores_long = []; scores_short = []
all_pass_long = []; all_pass_short = []

for i in range(warmup, len(daily)):
    bd = daily.index[i].date()
    if bd < start_dt or bd > end_dt: continue
    d_s = daily.iloc[:i+1]; w_s = weekly[weekly.index <= daily.index[i]]
    if len(w_s) < 20: continue
    total += 1
    for direction, pass_d, scores_list, all_pass_list in [
        ('Long',  pass_long,  scores_long,  all_pass_long),
        ('Short', pass_short, scores_short, all_pass_short),
    ]:
        res, score = run_checks(d_s, w_s, direction)
        scores_list.append(score)
        crit = [k for k, _, mode, c in CHECK_META if mode == 'calc' and c]
        cp = all(res.get(k, (False,))[0] for k in crit)
        all_pass_list.append(score >= 14 and cp)
        for k, _ in res.items():
            if k in pass_d: pass_d[k] += int(res[k][0])

print(f'Days evaluated: {total}')
print()
print(f'{"CHECK":<28} {"LONG%":>7} {"SHORT%":>8}   CRITICAL?')
print('-' * 58)
for key, label, mode, crit in CHECK_META:
    lp = pass_long.get(key, 0) / total * 100 if total else 0
    sp = pass_short.get(key, 0) / total * 100 if total else 0
    crit_tag = ' <-- CRITICAL' if crit and mode == 'calc' else ''
    bar_l = '#' * int(lp / 5)
    bar_s = '#' * int(sp / 5)
    print(f'  {label:<26} {lp:>5.0f}%  {sp:>6.0f}%{crit_tag}')

print()
avg_l = sum(scores_long) / len(scores_long) if scores_long else 0
avg_s = sum(scores_short) / len(scores_short) if scores_short else 0
print(f'Avg score  Long={avg_l:.1f}/18   Short={avg_s:.1f}/18')
print(f'Days all-critical-pass+score>=14:  Long={sum(all_pass_long)}  Short={sum(all_pass_short)}')
print()

# ── Show today's check details ──
print('--- Latest bar check details (today) ---')
d_s = daily; w_s = weekly
for direction in ('Long', 'Short'):
    res, score = run_checks(d_s, w_s, direction)
    crit = [k for k, _, mode, c in CHECK_META if mode == 'calc' and c]
    cp = all(res.get(k, (False,))[0] for k in crit)
    print(f'\n{direction} — score={score}/18  all_critical={cp}')
    for key, label, mode, crit_flag in CHECK_META:
        if key in res:
            v, detail = res[key]
            sym = 'PASS' if v else 'FAIL'
            crit_tag = '*' if crit_flag and mode == 'calc' else ' '
            print(f'  [{sym}]{crit_tag} {label:<28} {detail}')
