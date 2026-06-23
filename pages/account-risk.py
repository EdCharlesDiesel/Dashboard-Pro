"""Account Risk & Position Sizer.

Given an account balance and a risk tolerance, work out the two numbers a
trader actually needs before pulling the trigger:

  • how much money is on the line (risk in account currency), and
  • the lot size that puts exactly that much at risk for a given stop.

Honours the desk convention of taking **two trades per signal** — the sizing
splits across the two entries (or doubles, depending on the mode selected) so
the combined exposure matches the chosen risk budget.
"""
import streamlit as st
from src.ui.theme import BloombergTheme
from src.pages_lib.navigation import render_sidebar_nav
from src.instruments.registry import INSTRUMENTS, TREND_COMMODITIES
import pandas as pd
import yfinance as yf
from datetime import datetime

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Account Risk · Position Sizer",
    page_icon="💵",
    layout="wide",
    initial_sidebar_state="expanded",
)
BloombergTheme.apply()

# ── CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
      --border: color-mix(in srgb, var(--text-color) 12%, transparent);
      --muted:  color-mix(in srgb, var(--text-color) 55%, transparent);
    }
    html,body,[class*="css"]{ font-family:'JetBrains Mono','Fira Code',monospace; }
    .stApp{ background:var(--background-color); }
    section[data-testid="stSidebar"]{ background:var(--secondary-background-color)!important; border-right:1px solid var(--border,#2a2a2a); }
    #MainMenu,footer{ visibility:hidden; }
    [data-testid="stSidebarCollapsedControl"]{visibility:visible !important;}
    [data-testid="stSidebarNav"]{ display:none; }
    .block-container{ padding-top:1.5rem; max-width:1380px; }

    .card{ background:var(--secondary-background-color); border:1px solid var(--border,#2a2a2a);
           padding:20px; margin-bottom:16px; }
    .card-header{ font-size:13px; font-weight:600; letter-spacing:.08em;
                  text-transform:uppercase; color:#9a9a9a; margin-bottom:14px; }
    .metric-box{ background:var(--background-color); border:1px solid var(--border,#2a2a2a);
                 padding:14px; text-align:center; }
    .metric-value{ font-size:22px; font-weight:700; color:var(--text-color); }
    .metric-label{ font-size:11px; color:var(--muted,#9a9a9a); margin-top:2px; font-weight:500;
                   letter-spacing:.04em; text-transform:uppercase; }
    .section-title{ font-size:16px; font-weight:700; color:var(--text-color);
                    margin:24px 0 12px 0; padding-left:4px; border-left:3px solid #00ff41; }
    .lvl-row{ display:flex; justify-content:space-between; align-items:center;
              padding:10px 16px; border-bottom:1px solid #2a2a2a; font-size:13px; }
    .lvl-row:last-child{ border-bottom:none; }
    .lvl-label{ color:#9a9a9a; }
    .lvl-val  { font-family:'JetBrains Mono',monospace; font-weight:600; color:#e6e6e6; }
    .explainer{ background:var(--background-color); border:1px solid #1e3a5f;
                border-left:3px solid #00ff41;
                padding:14px 18px; font-size:13px; color:var(--muted,#9a9a9a); line-height:1.7; }
    .formula-box{ background:#000000; border:1px solid #2a2a2a;
                  padding:14px 18px; font-family:'JetBrains Mono',monospace;
                  font-size:13px; color:#e6e6e6; margin:10px 0; line-height:2.2; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# DATA FETCH
# ══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300, show_spinner=False)
def fetch_price(ticker: str) -> float | None:
    try:
        df = yf.download(ticker, interval="1d", period="5d",
                         progress=False, auto_adjust=True)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return float(df["Close"].iloc[-1])
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_atr14_pips(ticker: str, pip_size: float) -> float | None:
    try:
        df = yf.download(ticker, interval="1d", period="60d",
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 15:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        prev = df["Close"].shift(1)
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - prev).abs(),
            (df["Low"] - prev).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        return float(atr.iloc[-1]) / pip_size
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════
# CALCULATIONS
# ══════════════════════════════════════════════════════════════════
def fmt_price(p: float) -> str:
    return f"{p:.5f}" if abs(p) < 100 else f"{p:.3f}"


def size_position(balance: float, risk_pct: float, sl_pips: float,
                  pip_val: float, n_trades: int, split_mode: str) -> dict:
    """Translate a risk budget into money-at-risk and a lot size.

    n_trades is the desk's trades-per-signal (2 here). split_mode decides
    whether the risk % is the budget for the whole signal (split across the
    trades) or applied per trade (so total exposure scales by n_trades).
    """
    risk_signal = balance * (risk_pct / 100)

    if split_mode == "Split across trades":
        risk_per_trade = risk_signal / n_trades if n_trades else risk_signal
        risk_total = risk_signal
    else:  # "Per trade"
        risk_per_trade = risk_signal
        risk_total = risk_signal * n_trades

    lots_per_trade = (risk_per_trade / (sl_pips * pip_val)) if sl_pips and pip_val else 0.0
    lots_total = lots_per_trade * n_trades

    return {
        "risk_signal": risk_signal,
        "risk_per_trade": risk_per_trade,
        "risk_total": risk_total,
        "lots_per_trade": lots_per_trade,
        "lots_total": lots_total,
        "units_per_trade": lots_per_trade * 100_000,
        "micro_per_trade": lots_per_trade * 100,
        "risk_pct_total": (risk_total / balance * 100) if balance else 0.0,
    }


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 💵 Account Risk")

    inst_keys = INSTRUMENTS.keys()
    default_inst = st.session_state.get("selected_instrument", "EUR/USD")
    if default_inst not in inst_keys:
        default_inst = inst_keys[0]
    selected_pair = st.selectbox("Instrument", inst_keys,
                                 index=inst_keys.index(default_inst))
    st.session_state["selected_instrument"] = selected_pair

    inst = INSTRUMENTS[selected_pair]
    pip_size = inst["pip_size"]
    pip_val = inst["pip"]
    ticker = inst["ticker"]

    st.divider()
    st.markdown("**🏦 Account**")
    balance = st.number_input(
        "Account Balance ($)",
        value=float(st.session_state.get("account_bal", 10000.0)),
        min_value=0.0, step=500.0, format="%.2f")
    st.session_state["account_bal"] = balance

    risk_pct = st.slider(
        "Risk per signal (%)", 0.25, 5.0,
        float(st.session_state.get("risk_pct", 1.0)), 0.25)
    st.session_state["risk_pct"] = risk_pct

    leverage = st.selectbox("Account Leverage",
                            [30, 50, 100, 200, 400, 500], index=2)

    st.divider()
    st.markdown("**🎯 Two Trades / Signal**")
    two_trades = st.checkbox("Take 2 trades per signal", value=True,
                             help="Desk convention — every signal is entered "
                                  "as two positions.")
    n_trades = 2 if two_trades else 1
    split_mode = st.radio(
        "Risk % applies as",
        ["Split across trades", "Per trade"],
        index=0,
        help="Split = the risk % is the budget for the whole signal. "
             "Per trade = each of the two trades risks the full %, so total "
             "exposure doubles.")

    st.divider()
    st.markdown("**🛡️ Stop Loss**")
    if st.button("⚡ Fetch Live Price + ATR", width="stretch", type="primary"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner("Fetching…"):
        live_price = fetch_price(ticker)
        live_atr_pips = fetch_atr14_pips(ticker, pip_size)

    sl_mode = st.radio("Stop method", ["Manual (pips)", "ATR × 1.5"],
                       index=1 if live_atr_pips else 0, horizontal=True)
    if sl_mode == "ATR × 1.5" and live_atr_pips:
        sl_pips = round(live_atr_pips * 1.5, 1)
        st.caption(f"ATR14 = {live_atr_pips:.1f} pips → SL = {sl_pips:.1f} pips")
    else:
        sl_pips = st.number_input("Stop Loss (pips)", value=30.0,
                                  min_value=1.0, step=0.5, format="%.1f")

    st.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')} local · pip ${pip_val:.2f}/lot")

    st.divider()
    render_sidebar_nav()


# ══════════════════════════════════════════════════════════════════
# COMPUTE
# ══════════════════════════════════════════════════════════════════
r = size_position(balance, risk_pct, sl_pips, pip_val, n_trades, split_mode)

price = live_price if live_price else 0.0
notional_per_trade = r["units_per_trade"] * price
margin_per_trade = notional_per_trade / leverage if leverage else 0.0
margin_total = margin_per_trade * n_trades
margin_pct = (margin_total / balance * 100) if balance else 0.0

# Heavy-risk warning band
if r["risk_pct_total"] > 4:
    band_color, band_txt, band_cls = "#ff3344", "⚠️ OVER-LEVERAGED — total risk above 4%", "fail"
elif r["risk_pct_total"] > 2:
    band_color, band_txt, band_cls = "#ffcc00", "⚠️ ELEVATED — total risk above 2%", "warn"
else:
    band_color, band_txt, band_cls = "#00ff66", "✅ RISK WITHIN LIMITS", "ok"


# ── Header ─────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:linear-gradient(135deg,#000000 0%,#0a0a0a 50%,#000000 100%);
            border:1px solid #2a2a2a; padding:24px 28px; margin-bottom:20px;">
  <div style="font-size:24px; font-weight:700; color:#e6e6e6;">💵 Account Risk &amp; Position Sizer</div>
  <div style="color:#9a9a9a; font-size:13px; margin-top:4px;">
    {selected_pair} · risk {risk_pct:.2f}% of ${balance:,.0f} · SL {sl_pips:.1f} pips ·
    {n_trades} trade{'s' if n_trades > 1 else ''}/signal ({split_mode.lower()})
  </div>
  <div style="font-size:12px; color:{band_color}; margin-top:6px;">
    {band_txt} · {datetime.now().strftime('%A %d %B %Y  |  %H:%M')}
  </div>
</div>
""", unsafe_allow_html=True)


# ── KPI strip ──────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
for col, val, lbl, color in [
    (k1, f"${r['risk_total']:,.2f}",     "Total $ at Risk",      band_color),
    (k2, f"{r['lots_total']:.2f}",       "Total Lot Size",       "#00ff41"),
    (k3, f"{r['lots_per_trade']:.2f}",   "Lots / Trade",         "#00ff66"),
    (k4, f"${r['risk_per_trade']:,.2f}", "$ Risk / Trade",       "#56d364"),
    (k5, f"{r['risk_pct_total']:.2f}%",  "Total Risk %",         band_color),
]:
    with col:
        st.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-value" style="color:{color};font-size:20px;">{val}</div>'
            f'<div class="metric-label">{lbl}</div></div>',
            unsafe_allow_html=True)

st.markdown("---")

# ── Breakdown + margin ─────────────────────────────────────────────
col_left, col_right = st.columns([1, 1])


def lvl_row(label, val, extra="", color="#e6e6e6"):
    return (f'<div class="lvl-row"><span class="lvl-label">{label}</span>'
            f'<span class="lvl-val" style="color:{color};">{val}'
            f'<span style="color:#9a9a9a;font-size:11px;font-weight:400;">'
            f'&nbsp;&nbsp;{extra}</span></span></div>')


with col_left:
    st.markdown('<div class="section-title">📐 Position Breakdown</div>', unsafe_allow_html=True)
    units_note = "units (1 lot = 100k)" if selected_pair not in TREND_COMMODITIES else "≈ contract units"
    st.markdown(f"""
    <div class="card">
      <div class="card-header">{selected_pair} — {n_trades} trade{'s' if n_trades > 1 else ''}/signal</div>
      {lvl_row("Account balance", f"${balance:,.2f}")}
      {lvl_row("Risk budget / signal", f"${r['risk_signal']:,.2f}", f"{risk_pct:.2f}%", band_color)}
      {lvl_row("$ risk per trade", f"${r['risk_per_trade']:,.2f}", "", "#56d364")}
      {lvl_row("Total $ at risk", f"${r['risk_total']:,.2f}", f"{r['risk_pct_total']:.2f}% of acct", band_color)}
      {lvl_row("Lot size per trade", f"{r['lots_per_trade']:.2f} lots", f"{r['micro_per_trade']:.0f} micro", "#00ff41")}
      {lvl_row("Total lot size", f"{r['lots_total']:.2f} lots", "", "#00ff41")}
      {lvl_row("Units per trade", f"{r['units_per_trade']:,.0f}", units_note, "#a29bfe")}
      {lvl_row("Stop loss", f"{sl_pips:.1f} pips", f"${pip_val:.2f}/pip/lot", "#ff3344")}
    </div>
    """, unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="section-title">💰 Margin &amp; Exposure</div>', unsafe_allow_html=True)
    if price <= 0:
        st.markdown(
            '<div class="card"><div class="card-header">Margin</div>'
            '<div style="color:#9a9a9a;font-size:13px;">Fetch the live price '
            '(⚡ button in the sidebar) to estimate notional &amp; margin.</div></div>',
            unsafe_allow_html=True)
    else:
        margin_color = "#ff3344" if margin_pct > 50 else "#ffcc00" if margin_pct > 25 else "#00ff66"
        st.markdown(f"""
        <div class="card">
          <div class="card-header">Live price {fmt_price(price)} · {leverage}:1 leverage</div>
          {lvl_row("Notional per trade", f"${notional_per_trade:,.0f}", "", "#e6e6e6")}
          {lvl_row("Total notional", f"${notional_per_trade * n_trades:,.0f}", "", "#e6e6e6")}
          {lvl_row("Margin per trade", f"${margin_per_trade:,.2f}", "", "#a29bfe")}
          {lvl_row("Total margin used", f"${margin_total:,.2f}", f"{margin_pct:.1f}% of acct", margin_color)}
          {lvl_row("Free margin after", f"${max(balance - margin_total, 0):,.2f}", "", "#56d364")}
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ── Risk-ladder table ──────────────────────────────────────────────
st.markdown('<div class="section-title">📊 Risk Ladder — $ &amp; Lots by Risk %</div>',
            unsafe_allow_html=True)
st.caption(f"Same SL ({sl_pips:.1f} pips) on {selected_pair}, {n_trades} trade(s)/signal, "
           f"{split_mode.lower()}.")

ladder_rows = []
for rp in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    rr = size_position(balance, rp, sl_pips, pip_val, n_trades, split_mode)
    ladder_rows.append({
        "Risk %": f"{rp:.1f}%",
        "$ / Signal": round(rr["risk_signal"], 2),
        "$ / Trade": round(rr["risk_per_trade"], 2),
        "Total $ Risk": round(rr["risk_total"], 2),
        "Lots / Trade": round(rr["lots_per_trade"], 2),
        "Total Lots": round(rr["lots_total"], 2),
    })
st.dataframe(
    pd.DataFrame(ladder_rows), width="stretch", hide_index=True,
    column_config={
        "$ / Signal":   st.column_config.NumberColumn(format="$%.2f"),
        "$ / Trade":    st.column_config.NumberColumn(format="$%.2f"),
        "Total $ Risk": st.column_config.NumberColumn(format="$%.2f"),
    })

# ── All-pairs sizer ────────────────────────────────────────────────
st.markdown('<div class="section-title">🌐 All-Pairs Lot Sizer</div>', unsafe_allow_html=True)
st.caption(f"Lots needed to risk {risk_pct:.2f}% (${r['risk_per_trade']:,.2f}/trade) "
           f"with each pair's live ATR×1.5 stop.")

if st.button("📡 Scan all pairs (live ATR)", width="content"):
    st.cache_data.clear()
    st.rerun()

with st.expander("📋 All-Pairs Sizing Table", expanded=False):
    prog = st.progress(0, text="Scanning…")
    pairs = INSTRUMENTS.items()
    scan_rows = []
    for i, (name, info) in enumerate(pairs):
        prog.progress((i + 1) / len(pairs), text=f"Fetching {name}…")
        atr_pips = fetch_atr14_pips(info.ticker, info.pip_size)
        if atr_pips is None:
            scan_rows.append({"Pair": name, "ATR Stop (pips)": None,
                              "Lots / Trade": None, "Total Lots": None,
                              "$ / Trade": None})
            continue
        sl = round(atr_pips * 1.5, 1)
        rr = size_position(balance, risk_pct, sl, info.pip, n_trades, split_mode)
        scan_rows.append({
            "Pair": name,
            "ATR Stop (pips)": round(sl, 1),
            "Lots / Trade": round(rr["lots_per_trade"], 2),
            "Total Lots": round(rr["lots_total"], 2),
            "$ / Trade": round(rr["risk_per_trade"], 2),
        })
    prog.empty()
    st.dataframe(
        pd.DataFrame(scan_rows), width="stretch", hide_index=True,
        column_config={"$ / Trade": st.column_config.NumberColumn(format="$%.2f")})

st.markdown("---")

# ── Explainer + formula ────────────────────────────────────────────
col_a, col_b = st.columns(2)
with col_a:
    st.markdown(f"""
    <div class="explainer">
    <b style="color:#00ff41;">📏 How the lot size is derived</b><br><br>
    Money at risk is fixed first — <b style="color:#e6e6e6;">{risk_pct:.2f}%</b> of your
    ${balance:,.0f} balance = <b style="color:{band_color};">${r['risk_signal']:,.2f}</b>
    per signal.<br><br>
    Lot size then falls out of the stop distance: a wider stop forces a smaller
    lot so the dollar loss stays constant. This is why position size is an
    <em>output</em> of risk, never an input.
    </div>
    """, unsafe_allow_html=True)
with col_b:
    st.markdown(f"""
    <div class="explainer">
    <b style="color:#00ff41;">🎯 Two trades per signal</b><br><br>
    Every signal is entered as <b style="color:#e6e6e6;">two positions</b>. In
    <b>{split_mode.lower()}</b> mode each trade carries
    <b style="color:#56d364;">{r['lots_per_trade']:.2f} lots</b>
    (${r['risk_per_trade']:,.2f} risk), for a combined
    <b style="color:{band_color};">{r['lots_total']:.2f} lots</b> /
    ${r['risk_total']:,.2f} on the signal.<br><br>
    Switch to <em>per-trade</em> mode in the sidebar if each entry should carry
    the full risk % independently.
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"""
<div class="formula-box">
$ at risk &nbsp;= &nbsp;balance × risk%
&nbsp;= &nbsp;${balance:,.0f} × {risk_pct:.2f}%
&nbsp;= &nbsp;<b style="color:{band_color};">${r['risk_signal']:,.2f}</b><br>
Lots / trade &nbsp;= &nbsp;$ risk per trade ÷ (SL pips × pip value)
&nbsp;= &nbsp;${r['risk_per_trade']:,.2f} ÷ ({sl_pips:.1f} × ${pip_val:.2f})
&nbsp;= &nbsp;<b style="color:#00ff41;">{r['lots_per_trade']:.2f} lots</b><br>
Total exposure &nbsp;= &nbsp;{r['lots_per_trade']:.2f} × {n_trades} trades
&nbsp;= &nbsp;<b style="color:#00ff41;">{r['lots_total']:.2f} lots</b>
&nbsp;(${r['risk_total']:,.2f} total risk)
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;color:#555555;font-size:11px;margin-top:32px;
            padding-top:16px;border-top:1px solid #2a2a2a;">
  💵 Account Risk &amp; Position Sizer · risk → lots · For educational purposes only
</div>
""", unsafe_allow_html=True)
