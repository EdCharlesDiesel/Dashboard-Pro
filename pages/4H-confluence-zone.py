import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import pytz

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="4H Confluence Zone",
    page_icon="🎯",
    layout="wide",
)

# ─────────────────────────────────────────────
#  Custom CSS — dark terminal / trading aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: #0a0e17;
    color: #c9d1e0;
}

.stApp { background-color: #0a0e17; }

h1, h2, h3 {
    font-family: 'Syne', sans-serif;
    letter-spacing: -0.03em;
}

.metric-card {
    background: linear-gradient(135deg, #111827 0%, #1a2235 100%);
    border: 1px solid #1e3a5f;
    border-left: 3px solid #00d4ff;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 10px;
}

.confluence-badge {
    display: inline-block;
    background: linear-gradient(90deg, rgba(0, 212, 255, 0.13), rgba(255, 107, 53, 0.13));
    border: 1px solid rgba(0, 212, 255, 0.33);
    border-radius: 4px;
    padding: 4px 10px;
    font-size: 0.78em;
    color: #00d4ff;
    font-weight: 600;
    letter-spacing: 0.05em;
}

.strong-badge {
    background: linear-gradient(90deg, rgba(255, 107, 53, 0.13), rgba(255, 107, 53, 0.27));
    border-color: rgba(255, 107, 53, 0.53);
    color: #ff6b35;
}

.page-header {
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 16px;
    margin-bottom: 24px;
}

div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.4rem !important;
    color: #00d4ff !important;
}

div[data-testid="stMetricLabel"] {
    color: #6b7a99 !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

.stSelectbox > div, .stNumberInput > div {
    background-color: #111827 !important;
    border-color: #1e3a5f !important;
}

.stButton > button {
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.13), rgba(0, 119, 255, 0.13));
    border: 1px solid rgba(0, 212, 255, 0.33);
    color: #00d4ff;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    letter-spacing: 0.05em;
    border-radius: 6px;
    transition: all 0.2s;
}

.stButton > button:hover {
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.27), rgba(0, 119, 255, 0.27));
    border-color: #00d4ff;
    color: #fff;
}
    [data-testid="stSidebarNav"]{display:none;}
hr { border-color: #1e3a5f; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Helper functions
# ─────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_4h_data(ticker: str, days: int = 90) -> pd.DataFrame:
    end = datetime.now(pytz.utc)
    start = end - timedelta(days=days)

    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1h",
        progress=False,
        auto_adjust=True,
    )

    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required_columns = {"Open", "High", "Low", "Close", "Volume"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        return pd.DataFrame()

    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    df_4h = (
        df.resample("4h")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna(subset=["Open", "High", "Low", "Close"])
    )

    return df_4h


def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_fibonacci_levels(high: float, low: float) -> dict[str, float]:
    diff = high - low

    return {
        "0.0%": high,
        "23.6%": high - 0.236 * diff,
        "38.2%": high - 0.382 * diff,
        "50.0%": high - 0.500 * diff,
        "61.8%": high - 0.618 * diff,
        "78.6%": high - 0.786 * diff,
        "100%": low,
    }


def calc_pivot_points(df: pd.DataFrame) -> dict[str, float]:
    if len(df) < 2:
        last = df.iloc[-1]
    else:
        last = df.iloc[-2]

    high = float(last["High"])
    low = float(last["Low"])
    close = float(last["Close"])

    pp = (high + low + close) / 3

    return {
        "PP": pp,
        "R1": 2 * pp - low,
        "R2": pp + (high - low),
        "R3": high + 2 * (pp - low),
        "S1": 2 * pp - high,
        "S2": pp - (high - low),
        "S3": low - 2 * (high - pp),
    }


def find_confluence_zones(
        price: float,
        fib_levels: dict[str, float],
        pivots: dict[str, float],
        ema_values: dict[str, float],
        tolerance_pct: float = 0.005,
) -> list[dict]:
    tolerance = price * tolerance_pct
    all_levels = []

    for label, value in fib_levels.items():
        all_levels.append({"source": "Fib", "label": label, "value": float(value)})

    for label, value in pivots.items():
        all_levels.append({"source": "Pivot", "label": label, "value": float(value)})

    for label, value in ema_values.items():
        all_levels.append({"source": "EMA", "label": label, "value": float(value)})

    zones = []
    used = set()

    for i, level_a in enumerate(all_levels):
        if i in used:
            continue

        cluster = [level_a]

        for j, level_b in enumerate(all_levels):
            if j <= i or j in used:
                continue

            if abs(level_a["value"] - level_b["value"]) <= tolerance:
                cluster.append(level_b)
                used.add(j)

        if len(cluster) >= 2:
            used.add(i)

            avg_value = float(np.mean([item["value"] for item in cluster]))
            sources = sorted({item["source"] for item in cluster})
            labels = [f"{item['source']}:{item['label']}" for item in cluster]
            distance_pct = (avg_value - price) / price * 100

            zones.append(
                {
                    "value": avg_value,
                    "sources": sources,
                    "labels": labels,
                    "strength": len(cluster),
                    "distance_pct": distance_pct,
                }
            )

    zones.sort(key=lambda item: abs(item["distance_pct"]))

    return zones


def safe_rgba(hex_color: str, opacity: float) -> str:
    value = hex_color.strip().lstrip("#")

    if len(value) != 6:
        raise ValueError("Expected 6-digit hex color like #00d4ff")

    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)

    opacity = max(0.0, min(1.0, opacity))

    return f"rgba({r}, {g}, {b}, {opacity})"


# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    ticker = st.text_input("Ticker Symbol", value="BTC-USD").upper().strip()
    lookback_days = st.slider("Lookback (days)", 30, 180, 90, step=15)

    st.markdown("---")
    st.markdown("**EMA Periods**")

    ema_short = st.number_input("EMA Short", value=20, min_value=5, max_value=200, step=1)
    ema_mid = st.number_input("EMA Mid", value=50, min_value=10, max_value=500, step=1)
    ema_long = st.number_input("EMA Long", value=200, min_value=20, max_value=800, step=5)

    st.markdown("---")

    confluence_tol = st.slider(
        "Confluence Tolerance (%)",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
    ) / 100

    refresh = st.button("🔄  Refresh Data", use_container_width=True)


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
st.markdown('<div class="page-header">', unsafe_allow_html=True)
st.markdown("# 🎯 4H Confluence Zone")
st.markdown(f"**{ticker}** · Fibonacci + Pivot Points + EMA overlap on 4H chart")
st.markdown("</div>", unsafe_allow_html=True)

if refresh:
    st.cache_data.clear()

if not ticker:
    st.error("Please enter a valid ticker symbol.")
    st.stop()

with st.spinner(f"Fetching 4H data for {ticker} …"):
    df = fetch_4h_data(ticker, lookback_days)

if df.empty or len(df) < 10:
    st.error(f"Could not fetch enough 4H data for **{ticker}**. Try a different symbol.")
    st.stop()

# ─────────────────────────────────────────────
#  Indicators
# ─────────────────────────────────────────────
df["EMA_short"] = calc_ema(df["Close"], int(ema_short))
df["EMA_mid"] = calc_ema(df["Close"], int(ema_mid))
df["EMA_long"] = calc_ema(df["Close"], int(ema_long))

current_price = float(df["Close"].iloc[-1])
swing_high = float(df["High"].rolling(50, min_periods=1).max().iloc[-1])
swing_low = float(df["Low"].rolling(50, min_periods=1).min().iloc[-1])

fib_levels = calc_fibonacci_levels(swing_high, swing_low)
pivots = calc_pivot_points(df)

ema_values = {
    f"EMA{int(ema_short)}": float(df["EMA_short"].iloc[-1]),
    f"EMA{int(ema_mid)}": float(df["EMA_mid"].iloc[-1]),
    f"EMA{int(ema_long)}": float(df["EMA_long"].iloc[-1]),
}

zones = find_confluence_zones(
    price=current_price,
    fib_levels=fib_levels,
    pivots=pivots,
    ema_values=ema_values,
    tolerance_pct=confluence_tol,
)

# ─────────────────────────────────────────────
#  Top metrics
# ─────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Current Price", f"{current_price:,.4f}")

with col2:
    st.metric("Swing High (50)", f"{swing_high:,.4f}")

with col3:
    st.metric("Swing Low (50)", f"{swing_low:,.4f}")

with col4:
    st.metric("Confluence Zones Found", len(zones))

st.markdown("---")

# ─────────────────────────────────────────────
#  Confluence zone cards
# ─────────────────────────────────────────────
if zones:
    st.markdown("### 🔥 Active Confluence Zones")

    for zone in zones[:6]:
        direction = "▲" if zone["distance_pct"] > 0 else "▼"
        badge_class = "strong-badge" if zone["strength"] >= 3 else "confluence-badge"
        strength_label = "STRONG" if zone["strength"] >= 3 else "MODERATE"
        labels_text = "  ·  ".join(zone["labels"])

        st.markdown(
            f"""
            <div class="metric-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="font-size:1.2em; font-weight:700; color:#e2e8f0;">
                            {zone["value"]:,.4f}
                        </span>
                        &nbsp;&nbsp;
                        <span style="color:#6b7a99; font-size:0.85em;">
                            {direction} {abs(zone["distance_pct"]):.2f}% from price
                        </span>
                    </div>
                    <span class="confluence-badge {badge_class}">
                        {strength_label} · {len(zone["sources"])} sources
                    </span>
                </div>
                <div style="margin-top:8px; font-size:0.78em; color:#4a6080;">
                    {labels_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.info(
        "No confluence zones detected within the current tolerance. "
        "Try increasing the tolerance in the sidebar."
    )

st.markdown("---")

# ─────────────────────────────────────────────
#  Plotly chart
# ─────────────────────────────────────────────
st.markdown("### 📊 4H Chart — Fibonacci · Pivot · EMA")

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    row_heights=[0.78, 0.22],
    vertical_spacing=0.03,
)

fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        increasing=dict(
            line=dict(color="#00d4ff"),
            fillcolor=safe_rgba("#00d4ff", 0.13),
        ),
        decreasing=dict(
            line=dict(color="#ff4d6d"),
            fillcolor=safe_rgba("#ff4d6d", 0.13),
        ),
        name="4H Candles",
        showlegend=False,
    ),
    row=1,
    col=1,
)

ema_colors = {
    "EMA_short": "#f59e0b",
    "EMA_mid": "#a78bfa",
    "EMA_long": "#34d399",
}

ema_names = {
    "EMA_short": f"EMA {int(ema_short)}",
    "EMA_mid": f"EMA {int(ema_mid)}",
    "EMA_long": f"EMA {int(ema_long)}",
}

for column_name, color in ema_colors.items():
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[column_name],
            mode="lines",
            line=dict(color=color, width=1.4),
            name=ema_names[column_name],
        ),
        row=1,
        col=1,
    )

fib_colors = {
    "0.0%": "#ffffff",
    "23.6%": "#00d4ff",
    "38.2%": "#a78bfa",
    "50.0%": "#f59e0b",
    "61.8%": "#ff6b35",
    "78.6%": "#ff4560",
    "100%": "#ffffff",
}

for label, value in fib_levels.items():
    color = fib_colors.get(label, "#888888")

    fig.add_hline(
        y=value,
        line_dash="dot",
        line_color=color,
        line_width=1,
        opacity=0.55,
        annotation_text=f"Fib {label}",
        annotation_position="right",
        annotation_font=dict(size=9, color=color),
        row=1,
        col=1,
    )

pivot_colors = {
    "PP": "#ffffff",
    "R1": "#00d4ff",
    "R2": "#0077ff",
    "R3": "#003f8a",
    "S1": "#ff6b35",
    "S2": "#ff3300",
    "S3": "#8a1a00",
}

for label, value in pivots.items():
    color = pivot_colors.get(label, "#888888")

    fig.add_hline(
        y=value,
        line_dash="dash",
        line_color=color,
        line_width=1,
        opacity=0.6,
        annotation_text=f"  {label}",
        annotation_position="left",
        annotation_font=dict(size=9, color=color),
        row=1,
        col=1,
    )

for zone in zones:
    band = zone["value"] * confluence_tol
    color = "#ff6b35" if zone["strength"] >= 3 else "#00d4ff"

    fig.add_hrect(
        y0=zone["value"] - band,
        y1=zone["value"] + band,
        fillcolor=safe_rgba(color, 0.08),
        line_width=0,
        row=1,
        col=1,
    )

    fig.add_hline(
        y=zone["value"],
        line_dash="longdash",
        line_color=color,
        line_width=1.5,
        opacity=0.8,
        row=1,
        col=1,
    )

fig.add_hline(
    y=current_price,
    line_dash="solid",
    line_color="#ffffff",
    line_width=1,
    opacity=0.9,
    annotation_text=f"  Price: {current_price:,.4f}",
    annotation_position="right",
    annotation_font=dict(size=10, color="#ffffff"),
    row=1,
    col=1,
)

colors_vol = [
    safe_rgba("#ff4560", 0.53) if close < open_ else safe_rgba("#00d4ff", 0.53)
    for close, open_ in zip(df["Close"], df["Open"])
]

fig.add_trace(
    go.Bar(
        x=df.index,
        y=df["Volume"],
        marker_color=colors_vol,
        name="Volume",
        showlegend=False,
    ),
    row=2,
    col=1,
)

fig.update_layout(
    paper_bgcolor="#0a0e17",
    plot_bgcolor="#0d1220",
    font=dict(
        family="JetBrains Mono, monospace",
        size=11,
        color="#6b7a99",
    ),
    xaxis_rangeslider_visible=False,
    legend=dict(
        bgcolor=safe_rgba("#0a0e17", 0.60),
        bordercolor="#1e3a5f",
        borderwidth=1,
        font=dict(size=10),
        orientation="h",
        yanchor="bottom",
        y=1.01,
        xanchor="left",
        x=0,
    ),
    margin=dict(l=10, r=120, t=10, b=10),
    height=680,
    xaxis2=dict(showgrid=True, gridcolor="#1a2235"),
    yaxis=dict(showgrid=True, gridcolor="#1a2235", side="right"),
    yaxis2=dict(showgrid=False, side="right"),
)

st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
#  Raw level table
# ─────────────────────────────────────────────
with st.expander("📋 All Levels Reference Table"):
    rows = []

    for label, value in fib_levels.items():
        rows.append(
            {
                "Type": "Fibonacci",
                "Label": label,
                "Level": round(float(value), 6),
                "Δ% from Price": round((float(value) - current_price) / current_price * 100, 3),
            }
        )

    for label, value in pivots.items():
        rows.append(
            {
                "Type": "Pivot",
                "Label": label,
                "Level": round(float(value), 6),
                "Δ% from Price": round((float(value) - current_price) / current_price * 100, 3),
            }
        )

    for label, value in ema_values.items():
        rows.append(
            {
                "Type": "EMA",
                "Label": label,
                "Level": round(float(value), 6),
                "Δ% from Price": round((float(value) - current_price) / current_price * 100, 3),
            }
        )

    ref_df = pd.DataFrame(rows).sort_values("Level", ascending=False)

    st.dataframe(
        ref_df,
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    f"Data: yfinance · 4H candles · Last refreshed "
    f"{datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M')} UTC"
)