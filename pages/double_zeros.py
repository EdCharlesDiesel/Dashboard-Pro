import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

from src.ui.theme import BloombergTheme
from src.pages_lib.navigation import render_sidebar_nav

# ---------- helper functions ----------
def get_pip_size(pair):
    """Return pip value for a forex pair."""
    if "JPY" in pair:
        return 0.01
    return 0.0001

def double_zero_levels(low, high, pip):
    """Generate all double-zero price levels within the data range."""
    step = 100 * pip  # 100 pips = double zero
    start = np.floor(low / step) * step
    end = np.ceil(high / step) * step
    levels = np.arange(start, end + step, step)
    return levels

def simulate_trades(df, pip, offset_pips):
    """
    Backtest the strategy on a DataFrame.
    Returns a list of trade records and a DataFrame with exit markers.
    """
    risk_pips = offset_pips + 20
    risk_price = risk_pips * pip
    offset_price = offset_pips * pip

    # 20-period SMA
    df['SMA20'] = df['Close'].rolling(20).mean()

    # Generate all relevant double zero levels
    all_levels = double_zero_levels(df['Low'].min(), df['High'].max(), pip)

    trades = []
    active_trades = []

    for i in range(20, len(df)):
        bar = df.iloc[i]
        close = bar['Close']
        high = bar['High']
        low = bar['Low']
        sma = bar['SMA20']

        if pd.isna(sma):
            continue

        # ---- Entry logic ----
        # Check for long setup: close < SMA, and a double zero level's entry price falls within the bar
        if close < sma:
            for level in all_levels:
                entry_price = level + offset_price
                stop_loss = level - 20 * pip
                if low <= entry_price <= high and low > level - 2*pip:  # avoid deep penetrations
                    # Open a new long trade
                    trade = {
                        'type': 'long',
                        'entry_time': bar.name,
                        'entry_price': entry_price,
                        'figure': level,
                        'stop_loss': stop_loss,
                        'target1': entry_price + risk_price,
                        'half_closed': False,
                        'remaining_stop': None,
                        'exit_time_half': None,
                        'exit_price_half': None,
                        'exit_time_rest': None,
                        'exit_price_rest': None,
                        'entry_idx': i
                    }
                    active_trades.append(trade)
                    trades.append(trade)

        # Check for short setup: close > SMA, entry below the figure
        if close > sma:
            for level in all_levels:
                entry_price = level - offset_price
                stop_loss = level + 20 * pip
                if low <= entry_price <= high and high < level + 2*pip:
                    trade = {
                        'type': 'short',
                        'entry_time': bar.name,
                        'entry_price': entry_price,
                        'figure': level,
                        'stop_loss': stop_loss,
                        'target1': entry_price - risk_price,
                        'half_closed': False,
                        'remaining_stop': None,
                        'exit_time_half': None,
                        'exit_price_half': None,
                        'exit_time_rest': None,
                        'exit_price_rest': None,
                        'entry_idx': i
                    }
                    active_trades.append(trade)
                    trades.append(trade)

        # ---- Exit management (process active trades) ----
        for trade in active_trades[:]:  # iterate over a copy
            idx = trade['entry_idx']
            # only check bars after entry
            if i <= idx:
                continue

            # 1) check original stop loss
            if trade['type'] == 'long':
                if low <= trade['stop_loss']:
                    _close_remaining(trade, i, trade['stop_loss'], df, 'stop_loss')
                    active_trades.remove(trade)
                    continue
                # 2) check target1
                if not trade['half_closed'] and high >= trade['target1']:
                    trade['exit_time_half'] = df.index[i]
                    trade['exit_price_half'] = trade['target1']
                    trade['half_closed'] = True
                    trade['remaining_stop'] = trade['entry_price']  # move to breakeven
                # 3) trailing stop after half is closed
                if trade['half_closed']:
                    # two-bar low trail
                    if i - idx >= 2:
                        trail_low = min(df['Low'].iloc[i-1], df['Low'].iloc[i-2])
                    else:
                        trail_low = df['Low'].iloc[i-1] if i-1 > idx else trade['entry_price']
                    trail_stop = max(trade['entry_price'], trail_low)
                    if low <= trail_stop:
                        _close_remaining(trade, i, trail_stop, df, 'trailing')
                        active_trades.remove(trade)
                        continue

            else:  # short trade
                if high >= trade['stop_loss']:
                    _close_remaining(trade, i, trade['stop_loss'], df, 'stop_loss')
                    active_trades.remove(trade)
                    continue
                if not trade['half_closed'] and low <= trade['target1']:
                    trade['exit_time_half'] = df.index[i]
                    trade['exit_price_half'] = trade['target1']
                    trade['half_closed'] = True
                    trade['remaining_stop'] = trade['entry_price']
                if trade['half_closed']:
                    # two-bar high trail
                    if i - idx >= 2:
                        trail_high = max(df['High'].iloc[i-1], df['High'].iloc[i-2])
                    else:
                        trail_high = df['High'].iloc[i-1] if i-1 > idx else trade['entry_price']
                    trail_stop = min(trade['entry_price'], trail_high)
                    if high >= trail_stop:
                        _close_remaining(trade, i, trail_stop, df, 'trailing')
                        active_trades.remove(trade)
                        continue

    # close any trade still open at the end of data
    for trade in active_trades:
        _close_remaining(trade, len(df)-1, df['Close'].iloc[-1], df, 'end_of_data')
    return trades

def _close_remaining(trade, bar_idx, exit_price, df, reason):
    """Close the remaining position (the half that wasn't already closed)."""
    trade['exit_time_rest'] = df.index[bar_idx]
    trade['exit_price_rest'] = exit_price
    if not trade['half_closed']:
        # both halves are still open, close everything at this exit
        trade['exit_time_half'] = df.index[bar_idx]
        trade['exit_price_half'] = exit_price
        trade['half_closed'] = True

def calc_trade_pips(trade, pip):
    """Calculate total pips gained from a trade."""
    if trade['type'] == 'long':
        pips_half = (trade['exit_price_half'] - trade['entry_price']) / pip
        pips_rest = (trade['exit_price_rest'] - trade['entry_price']) / pip
    else:
        pips_half = (trade['entry_price'] - trade['exit_price_half']) / pip
        pips_rest = (trade['entry_price'] - trade['exit_price_rest']) / pip
    return pips_half, pips_rest

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Double Zeros · Trading System", page_icon="🎯",
                   layout="wide", initial_sidebar_state="expanded")
BloombergTheme.apply()
st.title("Fading the Double Zeros – Intraday Trading Strategy")
st.markdown("""
This app demonstrates the **Fading the Double Zeros** technique for forex.
Select a currency pair, a date range, and the entry offset (pips).
The backtest follows the official rules:
- **Long**: price below the 20‑period SMA (15m), enter 10‑15 pips *above* the round number, initial stop 20 pips below the figure.
- **Short**: price above the 20‑period SMA, enter 10‑15 pips *below* the round number, initial stop 20 pips above the figure.
- When the trade is profitable by the amount risked, **close half** and move stop to breakeven. Then trail the remaining half using a **two‑bar low/high**.
""")

# Sidebar inputs
st.sidebar.markdown("### 🎯 Double Zeros")
pair = st.sidebar.selectbox("Currency Pair",
                            ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCAD=X",
                             "AUDUSD=X", "NZDUSD=X", "EURJPY=X", "GBPJPY=X"])
start_date = st.sidebar.date_input("Start date", datetime.today() - timedelta(days=30))
end_date = st.sidebar.date_input("End date", datetime.today())
offset_pips = st.sidebar.slider("Entry offset (pips)", 10, 15, 15)
run = st.sidebar.button("Run Backtest")

with st.sidebar:
    st.divider()
    render_sidebar_nav()

if run:
    with st.spinner("Downloading data..."):
        # Yahoo Finance limit: 15m data available for ~60 days
        df = yf.download(pair, start=start_date, end=end_date, interval="15m")
        if df.empty:
            st.error("No data found. Try a different date range or pair.")
            st.stop()

        # Flatten MultiIndex columns if necessary
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    pip = get_pip_size(pair)
    trades = simulate_trades(df, pip, offset_pips)

    if not trades:
        st.warning("No trades triggered with the selected parameters.")
        st.stop()

    # Create results table
    results = []
    total_pips = 0
    for t in trades:
        ph, pr = calc_trade_pips(t, pip)
        total_pips += (ph + pr)
        results.append({
            "Entry time": t['entry_time'],
            "Type": t['type'].capitalize(),
            "Figure": t['figure'],
            "Entry": round(t['entry_price'], 5),
            "Exit half": round(t['exit_price_half'], 5) if t['exit_price_half'] else "Open",
            "Exit rest": round(t['exit_price_rest'], 5) if t['exit_price_rest'] else "Open",
            "Pips half": round(ph, 1),
            "Pips rest": round(pr, 1),
            "Total pips": round(ph+pr, 1)
        })

    st.subheader(f"Backtest Results – {len(trades)} trades")
    st.dataframe(pd.DataFrame(results), width="stretch")
    st.metric("Total Pips", round(total_pips, 1))

    # ---- Chart ----
    fig = go.Figure()

    # Price and SMA
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], mode='lines', name='SMA20', line=dict(color='orange')))

    # Double zero levels that were traded
    traded_levels = set(t['figure'] for t in trades)
    for lvl in traded_levels:
        fig.add_hline(y=lvl, line_dash="dot", line_color="grey", annotation_text=str(lvl))

    # Trade markers
    for t in trades:
        # Entry
        fig.add_annotation(x=t['entry_time'], y=t['entry_price'],
                           text="E", showarrow=True, arrowhead=1, bgcolor="yellow")
        # Half exit
        if t['exit_time_half']:
            fig.add_annotation(x=t['exit_time_half'], y=t['exit_price_half'],
                               text="½", showarrow=True, arrowhead=1, bgcolor="lightgreen")
        # Rest exit
        if t['exit_time_rest']:
            color = "red" if (t['exit_price_rest'] < t['entry_price'] and t['type']=='long') or \
                             (t['exit_price_rest'] > t['entry_price'] and t['type']=='short') else "green"
            fig.add_annotation(x=t['exit_time_rest'], y=t['exit_price_rest'],
                               text="R", showarrow=True, arrowhead=1, bgcolor=color)

    fig.update_layout(title=f"{pair} – 15m with trades", xaxis_title="Time", yaxis_title="Price",
                      hovermode="x unified", height=600)
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Legend:** E = Entry, ½ = first half closed, R = remaining closed. Green R = profitable, red R = losing.")