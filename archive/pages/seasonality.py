"""Seasonality — historical calendar bias.

Shows how an instrument has behaved by *time of year* and *day of week* over
many years: average monthly return, the share of years each month closed
green, a year × month heatmap, and a day-of-week breakdown. Used as a bias
filter — a month that is green 75% of the time is a tailwind, not a signal.
"""
import streamlit as st
from src.ui.theme import BloombergTheme
from src.pages_lib.navigation import render_sidebar_nav
from src.instruments.registry import INSTRUMENTS
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime

st.set_page_config(
    page_title="Seasonality · Trading System",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="expanded",
)
BloombergTheme.apply()

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
    .block-container{ padding-top:1.5rem; max-width:1400px; }

    .card{ background:var(--secondary-background-color); border:1px solid var(--border,#2a2a2a);
           padding:18px 20px; margin-bottom:14px; }
    .metric-box{ background:var(--background-color); border:1px solid var(--border,#2a2a2a);
                 padding:12px 14px; text-align:center; }
    .metric-value{ font-size:20px; font-weight:700; color:var(--text-color); }
    .metric-label{ font-size:10px; color:var(--muted,#9a9a9a); margin-top:2px; font-weight:500;
                   letter-spacing:.05em; text-transform:uppercase; }
    .section-title{ font-size:16px; font-weight:700; color:var(--text-color);
                    margin:24px 0 12px 0; padding-left:4px; border-left:3px solid #00ff41; }
    .explainer{ background:var(--background-color); border:1px solid #1e3a5f;
                border-left:3px solid #00ff41;
                padding:14px 18px; font-size:13px; color:var(--muted,#9a9a9a); line-height:1.7; }
</style>
""", unsafe_allow_html=True)

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri"]


# ══════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_history(ticker: str, period: str) -> pd.DataFrame | None:
    from src.db.market_cache import cached_ohlc
    try:
        df = cached_ohlc(ticker, period=period, interval="1d", ttl=3600)
        if df.empty or len(df) < 60:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[["Close"]].dropna()
    except Exception:
        return None


def monthly_table(df: pd.DataFrame) -> pd.DataFrame:
    """Year × month grid of percentage returns (month-over-month close)."""
    monthly = df["Close"].resample("ME").last()
    rets = monthly.pct_change() * 100
    rets = rets.dropna()
    frame = pd.DataFrame({
        "year": rets.index.year,
        "month": rets.index.month,
        "ret": rets.values,
    })
    pivot = frame.pivot_table(index="year", columns="month", values="ret")
    pivot = pivot.reindex(columns=range(1, 13))
    pivot.columns = MONTHS
    return pivot


def dow_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Average daily % return by weekday (Mon–Fri)."""
    daily = df["Close"].pct_change() * 100
    frame = pd.DataFrame({"dow": df.index.dayofweek, "ret": daily.values}).dropna()
    frame = frame[frame["dow"] < 5]
    agg = frame.groupby("dow")["ret"].agg(
        avg="mean",
        winrate=lambda s: (s > 0).mean() * 100,
        n="count",
    )
    agg.index = [DOW[i] for i in agg.index]
    return agg


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📅 Seasonality")

    inst_keys = INSTRUMENTS.keys()
    default_inst = st.session_state.get("selected_instrument", "EUR/USD")
    if default_inst not in inst_keys:
        default_inst = inst_keys[0]
    selected_pair = st.selectbox("Instrument", inst_keys,
                                 index=inst_keys.index(default_inst))
    st.session_state["selected_instrument"] = selected_pair
    ticker = INSTRUMENTS[selected_pair]["ticker"]

    lookback = st.selectbox("History window",
                            ["5y", "10y", "15y", "max"], index=1)

    if st.button("🔄 Refresh Data", width="stretch", type="primary"):
        st.cache_data.clear()
        st.rerun()

    st.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')} local")

    st.divider()
    render_sidebar_nav()


# ══════════════════════════════════════════════════════════════════
# COMPUTE
# ══════════════════════════════════════════════════════════════════
with st.spinner(f"Loading {selected_pair} history…"):
    hist = fetch_history(ticker, lookback)

now = datetime.now()
cur_month_idx = now.month - 1
cur_month_name = MONTHS[cur_month_idx]

if hist is None:
    st.error(f"⚠️ Could not load history for {selected_pair}. "
             f"Try a shorter window or refresh.")
    st.stop()

pivot = monthly_table(hist)
avg_by_month = pivot.mean(axis=0)
winrate_by_month = (pivot > 0).sum(axis=0) / pivot.notna().sum(axis=0) * 100
years_covered = int(pivot.notna().any(axis=1).sum())

cur_avg = float(avg_by_month.get(cur_month_name, np.nan))
cur_wr = float(winrate_by_month.get(cur_month_name, np.nan))

best_month = avg_by_month.idxmax()
worst_month = avg_by_month.idxmin()


# ── Header ─────────────────────────────────────────────────────────
if np.isnan(cur_avg):
    band_color, band_txt = "#9a9a9a", "no data for current month"
elif cur_avg > 0.3 and cur_wr >= 60:
    band_color, band_txt = "#00ff66", f"🟢 BULLISH SEASON — {cur_month_name} avg +{cur_avg:.2f}%, green {cur_wr:.0f}% of years"
elif cur_avg < -0.3 and cur_wr <= 40:
    band_color, band_txt = "#ff3344", f"🔴 BEARISH SEASON — {cur_month_name} avg {cur_avg:.2f}%, green {cur_wr:.0f}% of years"
else:
    band_color, band_txt = "#ffcc00", f"🟡 NEUTRAL SEASON — {cur_month_name} avg {cur_avg:+.2f}%, green {cur_wr:.0f}% of years"

# Persist a BULLISH/BEARISH seasonal bias for the current month (source='seasonality').
# Neutral / no-data months are skipped. Dedupe (pair+bias+price) keeps it to once
# per price level — a seasonal bias is a slow-moving tailwind, not a repeat trigger.
if band_color in ("#00ff66", "#ff3344"):
    try:
        from src.services.signal_store import persist_signals
        _entry = float(hist["Close"].iloc[-1]) if "Close" in getattr(hist, "columns", []) else None
        persist_signals("seasonality", [{
            "pair": selected_pair,
            "bias": "Long" if band_color == "#00ff66" else "Short",
            "entry": _entry,
            "conviction": f"{cur_month_name} seasonal",
            "thesis": (f"Seasonality — {cur_month_name} avg {cur_avg:+.2f}%, "
                       f"green {cur_wr:.0f}% of {years_covered} years"),
        }])
    except Exception:
        pass

st.markdown(f"""
<div style="background:linear-gradient(135deg,#000000 0%,#0a0a0a 50%,#000000 100%);
            border:1px solid #2a2a2a; padding:24px 28px; margin-bottom:20px;">
  <div style="font-size:24px; font-weight:700; color:#e6e6e6;">📅 Seasonality — {selected_pair}</div>
  <div style="color:#9a9a9a; font-size:13px; margin-top:4px;">
    {years_covered} years of history · current month: <b style="color:#e6e6e6;">{cur_month_name}</b>
  </div>
  <div style="font-size:13px; color:{band_color}; margin-top:6px;">
    {band_txt} · {now.strftime('%A %d %B %Y')}
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI strip ──────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
for col, val, lbl, color in [
    (k1, f"{cur_avg:+.2f}%" if not np.isnan(cur_avg) else "—", f"{cur_month_name} Avg Return", band_color),
    (k2, f"{cur_wr:.0f}%" if not np.isnan(cur_wr) else "—",     f"{cur_month_name} Win Rate",   band_color),
    (k3, f"{best_month}",  f"Best Month (+{avg_by_month.max():.2f}%)", "#00ff66"),
    (k4, f"{worst_month}", f"Worst Month ({avg_by_month.min():.2f}%)", "#ff3344"),
    (k5, f"{years_covered}", "Years Sampled", "#e6e6e6"),
]:
    with col:
        st.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-value" style="color:{color};font-size:18px;">{val}</div>'
            f'<div class="metric-label">{lbl}</div></div>',
            unsafe_allow_html=True)

st.markdown("---")

tab_month, tab_heat, tab_dow = st.tabs(
    ["📊 Monthly Averages", "🗓️ Year × Month Heatmap", "📆 Day of Week"])

# ── Monthly average bar ────────────────────────────────────────────
with tab_month:
    st.markdown('<div class="section-title">📊 Average Return by Month</div>',
                unsafe_allow_html=True)
    bar_colors = []
    for m in MONTHS:
        v = avg_by_month.get(m, 0)
        if m == cur_month_name:
            bar_colors.append("#00e0ff")
        else:
            bar_colors.append("#00ff66" if v >= 0 else "#ff3344")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=MONTHS, y=[avg_by_month.get(m, 0) for m in MONTHS],
        marker_color=bar_colors,
        text=[f"{avg_by_month.get(m, 0):+.2f}%" for m in MONTHS],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Avg: %{y:.2f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#9a9a9a", line_width=1)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#000000",
        font=dict(family="JetBrains Mono, monospace", size=11, color="#9a9a9a"),
        margin=dict(l=10, r=10, t=20, b=10), height=380,
        xaxis=dict(showgrid=False, linecolor="#2a2a2a"),
        yaxis=dict(showgrid=True, gridcolor="#2a2a2a", ticksuffix="%"),
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch", config=dict(displayModeBar=False))

    # Win-rate companion table
    st.markdown('<div class="section-title">✅ Win Rate by Month (% of years green)</div>',
                unsafe_allow_html=True)
    wr_rows = []
    for m in MONTHS:
        wr_rows.append({
            "Month": m + (" ◀" if m == cur_month_name else ""),
            "Avg Return %": round(float(avg_by_month.get(m, np.nan)), 2),
            "Win Rate %": round(float(winrate_by_month.get(m, np.nan)), 1),
            "Years": int(pivot[m].notna().sum()),
        })
    st.dataframe(
        pd.DataFrame(wr_rows), width="stretch", hide_index=True,
        column_config={
            "Avg Return %": st.column_config.NumberColumn(format="%.2f"),
            "Win Rate %":   st.column_config.ProgressColumn(
                format="%.0f%%", min_value=0, max_value=100),
        })

# ── Heatmap ────────────────────────────────────────────────────────
with tab_heat:
    st.markdown('<div class="section-title">🗓️ Monthly Returns — Year × Month (%)</div>',
                unsafe_allow_html=True)
    z = pivot.values.astype(float)
    fig2 = go.Figure(go.Heatmap(
        z=z, x=MONTHS, y=[str(y) for y in pivot.index],
        colorscale=[[0, "#ff3344"], [0.5, "#0a0a0a"], [1, "#00ff66"]],
        zmid=0,
        text=[[f"{v:+.1f}" if not np.isnan(v) else "" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=9),
        hovertemplate="%{y} %{x}: %{z:.2f}%<extra></extra>",
        colorbar=dict(title="%", tickfont=dict(color="#9a9a9a")),
    ))
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#000000",
        font=dict(family="JetBrains Mono, monospace", size=10, color="#9a9a9a"),
        margin=dict(l=10, r=10, t=20, b=10),
        height=max(360, 22 * len(pivot.index)),
        xaxis=dict(side="top", showgrid=False),
        yaxis=dict(autorange="reversed", showgrid=False),
    )
    st.plotly_chart(fig2, width="stretch", config=dict(displayModeBar=False))
    st.caption("Green = month closed up vs prior month · red = down. "
               "Each cell is the month-over-month percentage change of the close.")

# ── Day of week ────────────────────────────────────────────────────
with tab_dow:
    st.markdown('<div class="section-title">📆 Average Return by Weekday</div>',
                unsafe_allow_html=True)
    dow = dow_returns(hist)
    cur_dow = DOW[now.weekday()] if now.weekday() < 5 else None

    colors = []
    for d in dow.index:
        if d == cur_dow:
            colors.append("#00e0ff")
        else:
            colors.append("#00ff66" if dow.loc[d, "avg"] >= 0 else "#ff3344")

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=list(dow.index), y=dow["avg"].values,
        marker_color=colors,
        text=[f"{v:+.3f}%" for v in dow["avg"].values],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Avg: %{y:.3f}%<extra></extra>",
    ))
    fig3.add_hline(y=0, line_color="#9a9a9a", line_width=1)
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#000000",
        font=dict(family="JetBrains Mono, monospace", size=11, color="#9a9a9a"),
        margin=dict(l=10, r=10, t=20, b=10), height=340,
        xaxis=dict(showgrid=False, linecolor="#2a2a2a"),
        yaxis=dict(showgrid=True, gridcolor="#2a2a2a", ticksuffix="%"),
        showlegend=False,
    )
    st.plotly_chart(fig3, width="stretch", config=dict(displayModeBar=False))

    dow_disp = dow.reset_index().rename(columns={
        "index": "Weekday", "avg": "Avg Return %",
        "winrate": "Win Rate %", "n": "Sample Days"})
    st.dataframe(
        dow_disp, width="stretch", hide_index=True,
        column_config={
            "Avg Return %": st.column_config.NumberColumn(format="%.3f"),
            "Win Rate %":   st.column_config.ProgressColumn(
                format="%.0f%%", min_value=0, max_value=100),
        })

st.markdown("---")
st.markdown(f"""
<div class="explainer">
<b style="color:#00ff41;">📅 How to use seasonality</b><br><br>
Seasonality is a <b>bias filter, not a trigger</b>. A month that closes green in
{cur_wr:.0f}% of years (like the current month for {selected_pair}) tilts the odds
toward longs — but you still need the daily/4H/15M confirmation stack before
entering. Treat a strong seasonal tailwind as a reason to favour setups in that
direction and to be stricter on counter-seasonal trades.<br><br>
Sample = {years_covered} years. Thinner history (FX pairs, exotic crosses) makes
the averages noisier — weight them less.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;color:#555555;font-size:11px;margin-top:32px;
            padding-top:16px;border-top:1px solid #2a2a2a;">
  📅 Seasonality · calendar bias · For educational purposes only
</div>
""", unsafe_allow_html=True)
