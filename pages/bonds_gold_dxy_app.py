"""
Bonds → Gold → DXY : Interactive learning module
------------------------------------------------
Tickers used (all free via yfinance):
  ^TNX     = US 10-Year Treasury yield (x10, i.e. 42.5 -> 4.25%)
  DX-Y.NYB = US Dollar Index (DXY)
  GC=F     = Gold futures
  TIP      = iShares TIPS ETF (price ~ inverse proxy for REAL yields)

Educational tool only. Nothing here is financial advice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.ui.theme import BloombergTheme
from src.pages_lib.navigation import render_sidebar_nav
from src.services.market_data import daily_ohlc

st.set_page_config(
    page_title="Bonds → Gold → DXY · Trading System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)
BloombergTheme.apply()

T = BloombergTheme  # short alias for palette tokens

st.title("🏦 Bonds → Gold → DXY")
st.caption(
    "How Treasury yields drive gold and the dollar — the seesaw explained, "
    "then tested against live data. Educational tool, **not financial advice.**"
)

# ── Plotly terminal styling ──────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=T.BG_PANEL,
    font=dict(family=T.FONT_MONO, color=T.GREY, size=11),
    legend=dict(font=dict(color=T.GREY, size=11), bgcolor=T.BG_ELEV,
                bordercolor=T.BORDER, orientation="h", x=0, y=1.08),
    margin=dict(l=10, r=10, t=30, b=10),
)


def style_axes(fig: go.Figure, **kwargs) -> None:
    fig.update_xaxes(showgrid=False, tickfont=dict(color=T.GREY, size=9),
                     linecolor=T.BORDER, **kwargs)
    fig.update_yaxes(showgrid=True, gridcolor=T.BORDER,
                     tickfont=dict(color=T.GREY, size=9), **kwargs)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
TICKERS = {
    "yield10": "^TNX",      # 10Y yield * 10
    "dxy": "DX-Y.NYB",
    "gold": "GC=F",
    "tips": "TIP",
}


@st.cache_data(ttl=3600, show_spinner="Fetching market data...")
def load_data(period: str = "5y") -> pd.DataFrame:
    # One canonical fetch per ticker instead of a multi-ticker download. The
    # spine's `cached_ohlc` already defaults to auto_adjust=True, which is what
    # this page asked for — TIP is a distribution-paying ETF, so unadjusted
    # closes would step on every ex-date.
    closes = {}
    for key, tic in TICKERS.items():
        frame = daily_ohlc(tic, period=period)
        if frame.empty or "Close" not in frame.columns:
            return pd.DataFrame()
        closes[tic] = frame["Close"].rename(tic)
    raw = pd.concat(closes.values(), axis=1, sort=True)
    df = pd.DataFrame({
        "yield10": raw[TICKERS["yield10"]] / 10.0,  # ^TNX quotes 42.5 = 4.25%
        "dxy": raw[TICKERS["dxy"]],
        "gold": raw[TICKERS["gold"]],
        "tips": raw[TICKERS["tips"]],
    })
    return df.dropna(how="all").ffill().dropna()


# ----------------------------------------------------------------------
# Sidebar — page controls first, then shared nav (sidebar contract)
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("Data settings")
    period = st.selectbox("History", ["1y", "2y", "5y", "10y"], index=2)
    corr_window = st.slider("Rolling correlation window (days)", 20, 120, 60, 5)

    st.divider()
    render_sidebar_nav()

try:
    df = load_data(period)
    data_ok = not df.empty
except Exception as e:
    data_ok = False
    st.sidebar.error(f"Data fetch failed: {e}")

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
tab_learn, tab_charts, tab_corr, tab_scatter = st.tabs(
    ["📚 Learn the basics", "📈 Live relationships", "🔗 Rolling correlation", "🎯 Scatter test"]
)

# ======================================================================
# TAB 1 — LEARN
# ======================================================================
with tab_learn:
    st.header("Bonds, explained with pocket money")

    st.markdown(
        """
**A bond = an IOU.** You lend the US government \\$100 for 10 years.
Every year they pay you *pocket money* (the **coupon**). At the end you get your \\$100 back.

**The yield** is your pocket money as a % of what you *paid* for the bond.
"""
    )

    st.subheader("The seesaw: price vs yield")
    st.caption(
        "You own an old bond paying $4/year. Move the slider to change what NEW bonds pay, "
        "and watch what happens to the price someone will pay for YOUR old bond."
    )

    new_rate = st.slider("What new bonds pay per year ($ on a $100 bond)", 1.0, 8.0, 4.0, 0.25)
    old_coupon = 4.0
    # Simplified perpetuity-style intuition: fair price makes yields equal
    fair_price = round(old_coupon / new_rate * 100, 2)
    your_yield = round(old_coupon / fair_price * 100, 2)

    c1, c2, c3 = st.columns(3)
    c1.metric("New bonds pay", f"{new_rate:.2f}%")
    c2.metric("Your old bond's price", f"${fair_price:,.2f}",
              delta=f"{fair_price - 100:+.2f} vs $100", delta_color="normal")
    c3.metric("Buyer's yield on your bond", f"{your_yield:.2f}%")

    if new_rate > old_coupon:
        st.info("Rates went **up** → nobody wants your low-paying bond at full price → its **price falls** until its yield matches. **Yield up = price down.**")
    elif new_rate < old_coupon:
        st.info("Rates went **down** → your bond pays more than new ones → people pay a **premium**. **Yield down = price up.**")
    else:
        st.info("Rates unchanged → your bond is worth exactly $100.")

    st.divider()

    colA, colB = st.columns(2)
    with colA:
        st.subheader("🥇 Why gold cares")
        st.markdown(
            """
Gold pays **zero** pocket money. It just sits there being shiny.

So gold competes with bonds for your savings:

- Bonds paying a lot (**high real yields**) → holding gold has a big *opportunity cost* → **gold tends to fall**
- Bonds paying little (or less than inflation) → gold's "pays nothing" flaw doesn't matter → **gold tends to rise**

**Real yield** = bond yield − inflation. This is the number gold actually watches.
A 5% yield with 6% inflation is a *negative* real yield → great for gold.
"""
        )
    with colB:
        st.subheader("💵 Why DXY cares")
        st.markdown(
            """
The dollar index (DXY) measures USD vs a basket (mostly EUR).

- US yields **rise** relative to other countries → global money wants US bonds → must **buy dollars** first → **DXY up**
- US yields **fall** → money leaves → **DXY down**

It's the *differential* that matters: US 10Y vs German/Japanese 10Y, not the US number alone.
"""
        )

    st.divider()
    st.subheader("The usual chain (and when it breaks)")
    st.markdown(
        """
**Normal regime:** yields ↑ → DXY ↑ and gold ↓ (and the reverse).

**When it breaks** — gold rising *while* yields rise usually means:
war/crisis fear, central banks buying gold, or markets doubting the dollar itself.
That divergence is a signal in its own right. The Correlation tab shows you when the
relationship is holding and when it's broken.
"""
    )

# ======================================================================
# TAB 2 — LIVE CHARTS
# ======================================================================
with tab_charts:
    st.header("See the relationship in real data")
    if not data_ok:
        st.warning("No data available. Check your connection / yfinance.")
    else:
        latest = df.iloc[-1]
        m1, m2, m3 = st.columns(3)
        m1.metric("US 10Y yield", f"{latest['yield10']:.2f}%")
        m2.metric("DXY", f"{latest['dxy']:.2f}")
        m3.metric("Gold", f"${latest['gold']:,.0f}")

        # --- Gold vs 10Y yield (inverted axis to show the inverse relation)
        st.subheader("Gold vs 10Y yield (yield axis inverted)")
        st.caption("With the yield axis flipped, the two lines should roughly move *together* when the classic relationship holds.")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=df.index, y=df["gold"], name="Gold",
                                 line=dict(color=T.YELLOW, width=1.6)), secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index, y=df["yield10"], name="10Y yield (inverted)",
                                 line=dict(color=T.PURPLE, width=1.4)), secondary_y=True)
        fig.update_yaxes(title_text="Gold ($)", secondary_y=False)
        fig.update_yaxes(title_text="10Y yield (%)", secondary_y=True, autorange="reversed")
        style_axes(fig)
        fig.update_layout(**PLOTLY_LAYOUT, height=420)
        st.plotly_chart(fig, width="stretch")

        # --- DXY vs 10Y yield
        st.subheader("DXY vs 10Y yield")
        st.caption("These usually move in the *same* direction: higher US yields attract dollar buying.")
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Scatter(x=df.index, y=df["dxy"], name="DXY",
                                  line=dict(color=T.CYAN, width=1.6)), secondary_y=False)
        fig2.add_trace(go.Scatter(x=df.index, y=df["yield10"], name="10Y yield",
                                  line=dict(color=T.PURPLE, width=1.4)), secondary_y=True)
        fig2.update_yaxes(title_text="DXY", secondary_y=False)
        fig2.update_yaxes(title_text="10Y yield (%)", secondary_y=True)
        style_axes(fig2)
        fig2.update_layout(**PLOTLY_LAYOUT, height=420)
        st.plotly_chart(fig2, width="stretch")

        # --- Gold vs TIP (real-yield proxy)
        st.subheader("Gold vs TIP ETF (real-yield proxy)")
        st.caption(
            "TIP's *price* moves opposite to real yields. So gold and TIP price should move "
            "*together* — this is usually a tighter fit than gold vs nominal yields."
        )
        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        fig3.add_trace(go.Scatter(x=df.index, y=df["gold"], name="Gold",
                                  line=dict(color=T.YELLOW, width=1.6)), secondary_y=False)
        fig3.add_trace(go.Scatter(x=df.index, y=df["tips"], name="TIP price",
                                  line=dict(color=T.GREEN, width=1.4)), secondary_y=True)
        fig3.update_yaxes(title_text="Gold ($)", secondary_y=False)
        fig3.update_yaxes(title_text="TIP ($)", secondary_y=True)
        style_axes(fig3)
        fig3.update_layout(**PLOTLY_LAYOUT, height=420)
        st.plotly_chart(fig3, width="stretch")

# ======================================================================
# TAB 3 — ROLLING CORRELATION
# ======================================================================
with tab_corr:
    st.header("Is the relationship holding right now?")
    if not data_ok:
        st.warning("No data available.")
    else:
        # yield10 is already in % points; use diff for yields, pct for prices
        chg = pd.DataFrame({
            "gold": df["gold"].pct_change(),
            "dxy": df["dxy"].pct_change(),
            "yield10": df["yield10"].diff(),
        }).dropna()

        corr_gold_yield = chg["gold"].rolling(corr_window).corr(chg["yield10"])
        corr_gold_dxy = chg["gold"].rolling(corr_window).corr(chg["dxy"])
        corr_dxy_yield = chg["dxy"].rolling(corr_window).corr(chg["yield10"])

        st.caption(
            f"{corr_window}-day rolling correlation of daily changes. "
            "Textbook: Gold↔Yield **negative**, Gold↔DXY **negative**, DXY↔Yield **positive**. "
            "When a line crosses zero, the regime has flipped — pay attention."
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=corr_gold_yield.index, y=corr_gold_yield,
                                 name="Gold ↔ 10Y yield", line=dict(color=T.RED)))
        fig.add_trace(go.Scatter(x=corr_gold_dxy.index, y=corr_gold_dxy,
                                 name="Gold ↔ DXY", line=dict(color=T.YELLOW)))
        fig.add_trace(go.Scatter(x=corr_dxy_yield.index, y=corr_dxy_yield,
                                 name="DXY ↔ 10Y yield", line=dict(color=T.CYAN)))
        fig.add_hline(y=0, line_dash="dot", line_color=T.GREY)
        style_axes(fig)
        fig.update_layout(**PLOTLY_LAYOUT, height=480,
                          yaxis=dict(range=[-1, 1], title="Correlation",
                                     gridcolor=T.BORDER, tickfont=dict(color=T.GREY, size=9)))
        st.plotly_chart(fig, width="stretch")

        # Current regime readout
        cur_gy = corr_gold_yield.dropna().iloc[-1]
        cur_gd = corr_gold_dxy.dropna().iloc[-1]
        cur_dy = corr_dxy_yield.dropna().iloc[-1]

        c1, c2, c3 = st.columns(3)
        c1.metric("Gold ↔ Yield (now)", f"{cur_gy:+.2f}",
                  "textbook" if cur_gy < -0.1 else "broken/weak")
        c2.metric("Gold ↔ DXY (now)", f"{cur_gd:+.2f}",
                  "textbook" if cur_gd < -0.1 else "broken/weak")
        c3.metric("DXY ↔ Yield (now)", f"{cur_dy:+.2f}",
                  "textbook" if cur_dy > 0.1 else "broken/weak")

        broken = sum([cur_gy > -0.1, cur_gd > -0.1, cur_dy < 0.1])
        if broken == 0:
            st.success("All three relationships are in textbook mode. Yield moves should be tradeable signals for gold/DXY right now.")
        elif broken <= 1:
            st.info("Mostly textbook, one relationship is weak. Trade the strong ones, be careful with the weak one.")
        else:
            st.warning("Regime is broken — something else (fear, central-bank gold buying, fiscal worries) is driving prices. Don't lean on the yield chain right now.")

# ======================================================================
# TAB 4 — SCATTER TEST
# ======================================================================
with tab_scatter:
    st.header("Test it yourself: weekly changes")
    if not data_ok:
        st.warning("No data available.")
    else:
        wk = df.resample("W-FRI").last()
        wchg = pd.DataFrame({
            "gold_ret": wk["gold"].pct_change() * 100,
            "dxy_ret": wk["dxy"].pct_change() * 100,
            "yield_chg": wk["yield10"].diff() * 100,  # basis points
        }).dropna()

        pair = st.radio("Pick a pair", ["Yield change → Gold return", "Yield change → DXY return", "DXY return → Gold return"],
                        horizontal=True)

        if pair == "Yield change → Gold return":
            x, y, xl, yl = wchg["yield_chg"], wchg["gold_ret"], "10Y yield change (bp/week)", "Gold return (%/week)"
        elif pair == "Yield change → DXY return":
            x, y, xl, yl = wchg["yield_chg"], wchg["dxy_ret"], "10Y yield change (bp/week)", "DXY return (%/week)"
        else:
            x, y, xl, yl = wchg["dxy_ret"], wchg["gold_ret"], "DXY return (%/week)", "Gold return (%/week)"

        slope, intercept = np.polyfit(x, y, 1)
        r = np.corrcoef(x, y)[0, 1]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="weeks",
                                 marker=dict(size=6, color=T.CYAN, opacity=0.6)))
        xs = np.linspace(x.min(), x.max(), 50)
        fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, name="fit",
                                 line=dict(color=T.RED, dash="dash")))
        style_axes(fig)
        fig.update_layout(**PLOTLY_LAYOUT, height=460, xaxis_title=xl, yaxis_title=yl)
        st.plotly_chart(fig, width="stretch")

        c1, c2 = st.columns(2)
        c1.metric("Correlation (r)", f"{r:+.2f}")
        c2.metric("Slope", f"{slope:+.3f} per unit")
        st.caption(
            "Each dot is one week. A downward-sloping cloud for gold-vs-yield means the "
            "textbook relationship shows up in the data — but notice how noisy it is. "
            "That noise is why this is a *bias*, not a mechanical rule."
        )