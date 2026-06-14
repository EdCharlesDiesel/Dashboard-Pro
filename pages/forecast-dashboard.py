"""
FX Forecast Dashboard
----------------------
A Streamlit app that:
  - Pulls live & historical FX rates from Yahoo Finance (yfinance)
  - Pulls macro context series from FRED (pandas_datareader)
  - Produces a short-term forecast (Exponential Smoothing or ARIMA)
  - Displays a Trading-Economics-style chart: historical area + dashed forecast line

Run with:
    streamlit run app.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from io import StringIO
import requests
import warnings

from src.ui.theme import BloombergTheme as T
from src.pages_lib.navigation import render_sidebar_nav
from src.instruments.registry import INSTRUMENTS, TREND_COMMODITIES
from src.pages_lib.setup_ranker import fmt_price

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Forecast Dashboard · Trading System",
    page_icon="\U0001F4C8",
    layout="wide",
    initial_sidebar_state="expanded",
)
T.apply()

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

# Every tradable instrument comes from the single source of truth
# (src/instruments/registry.py) — all 18 forex pairs plus the three metals
# (XAU/USD, XAG/USD, XPT/USD). Maps display name -> Yahoo ticker.
CURRENCY_PAIRS = {name: INSTRUMENTS[name]["ticker"] for name in INSTRUMENTS}

# Metals carry forex display symbols but are categorised separately so the
# picker can offer a Forex / Metals filter.
METALS = sorted(TREND_COMMODITIES)
FOREX = [n for n in CURRENCY_PAIRS if n not in TREND_COMMODITIES]

# FRED series that give useful macro context for FX (rate differentials, etc.)
FRED_SERIES = {
    "US 10Y Treasury Yield (%)": "DGS10",
    "US Effective Fed Funds Rate (%)": "DFF",
    "Germany 10Y Govt Bond Yield (%)": "IRLTLT01DEM156N",
    "US Dollar Index (Broad, DXY proxy)": "DTWEXBGS",
}

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600, show_spinner=False)
def load_fx_history(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Download historical FX data from Yahoo Finance."""
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Close"])
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_fred_series(series_id: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Download a single series from FRED. Returns empty DataFrame on failure.

    Uses FRED's public keyless CSV endpoint (fredgraph.csv) instead of
    pandas_datareader, which is unmaintained and broken on Python 3.12+.
    Output is unchanged: a DataFrame indexed by date with one numeric column,
    filtered to [start, end].
    """
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.set_index(date_col)
        df.columns = [series_id]
        df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
        df = df.dropna()
        df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def load_quote_table(pairs: dict) -> pd.DataFrame:
    """Build a quick quote table (price / chg / %chg) for a dict of pairs."""
    rows = []
    for label, tk in pairs.items():
        try:
            d = yf.download(tk, period="5d", interval="1d", progress=False, auto_adjust=False)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            d = d.dropna(subset=["Close"])
            if len(d) >= 2:
                last = float(d["Close"].iloc[-1])
                prev = float(d["Close"].iloc[-2])
                chg = last - prev
                pct = chg / prev * 100
                rows.append(
                    {
                        "Instrument": label,
                        "Actual": fmt_price(last),
                        "Chg": f"{chg:+.4f}" if abs(last) < 100 else f"{chg:+.2f}",
                        "% Chg": pct,
                    }
                )
        except Exception:
            continue
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------


def run_forecast(series: pd.Series, periods: int, model_type: str):
    """Return a forecast Series of length `periods` (business-day index)."""
    series = series.dropna().astype(float)

    if model_type == "Exponential Smoothing":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        model = ExponentialSmoothing(series, trend="add", damped_trend=True, seasonal=None)
        fit = model.fit()
        fc_values = fit.forecast(periods)
    else:  # ARIMA
        from statsmodels.tsa.arima.model import ARIMA

        model = ARIMA(series, order=(1, 1, 1))
        fit = model.fit()
        fc_values = fit.forecast(periods)

    future_dates = pd.bdate_range(series.index[-1] + pd.Timedelta(days=1), periods=periods)
    fc_values = pd.Series(np.asarray(fc_values), index=future_dates, name="Forecast")
    return fc_values


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.title("Settings")

asset_class = st.sidebar.radio(
    "Asset class", ["All", "Forex", "Metals"], index=0, horizontal=True,
    help="Forecast any currency pair or a precious metal (Gold/Silver/Platinum).",
)
if asset_class == "Forex":
    _choices = FOREX
elif asset_class == "Metals":
    _choices = METALS
else:
    _choices = list(CURRENCY_PAIRS.keys())

pair_label = st.sidebar.selectbox("Instrument", _choices, index=0)
ticker = CURRENCY_PAIRS[pair_label]

period = st.sidebar.selectbox(
    "History window", ["6mo", "1y", "2y", "5y", "10y"], index=3
)

interval = st.sidebar.selectbox("Data interval", ["1d", "1wk"], index=0)


# Always project a full trading week ahead (5 business days).
forecast_horizon = 5
st.sidebar.caption(
    f"📅 Forecast horizon: **one week ahead** ({forecast_horizon} business days)"
)

model_choice = st.sidebar.radio(
    "Forecast model",
    ["Exponential Smoothing", "ARIMA"],
    help="Both are simple statistical models for illustration -- not financial advice.",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Data sources: **Yahoo Finance** (FX prices) and **FRED** (macro indicators). "
    "Forecasts are statistical extrapolations of historical data, not predictions "
    "of real market behaviour."
)

with st.sidebar:
    st.divider()
    render_sidebar_nav()

# ---------------------------------------------------------------------------
# Header & summary
# ---------------------------------------------------------------------------

_kind = "Metal" if pair_label in TREND_COMMODITIES else "Exchange Rate"
st.title(f"{pair_label} {_kind} \u2014 One-Week Forecast")

with st.spinner("Loading market data..."):
    df = load_fx_history(ticker, period=period, interval=interval)

if df.empty:
    st.error("Could not retrieve data from Yahoo Finance for this pair. Try again later.")
    st.stop()

close = df["Close"]

last_price = float(close.iloc[-1])
prev_price = float(close.iloc[-2]) if len(close) > 1 else last_price
day_chg = last_price - prev_price
day_pct = (day_chg / prev_price * 100) if prev_price else 0.0

# Approximate 1M / 1Y comparisons (trading days)
def pct_change_back(series: pd.Series, days: int) -> float:
    if len(series) > days:
        ref = float(series.iloc[-days - 1])
        return (float(series.iloc[-1]) - ref) / ref * 100
    return float("nan")

month_pct = pct_change_back(close, 21)
year_pct = pct_change_back(close, 252)

last_date = close.index[-1].strftime("%B %d, %Y")

summary = (
    f"The **{pair_label}** rate is at **{fmt_price(last_price)}** as of {last_date}, "
    f"{'down' if day_chg < 0 else 'up'} {fmt_price(abs(day_chg))} ({day_pct:+.2f}%) from the previous session. "
)
if not np.isnan(month_pct):
    summary += (
        f"Over the past month it has {'weakened' if month_pct < 0 else 'strengthened'} "
        f"{abs(month_pct):.2f}%"
    )
if not np.isnan(year_pct):
    summary += (
        f", and it is {'down' if year_pct < 0 else 'up'} {abs(year_pct):.2f}% over the last 12 months."
    )
else:
    summary += "."

st.markdown(summary)

# ---------------------------------------------------------------------------
# Forecast computation
# ---------------------------------------------------------------------------

with st.spinner("Fitting forecast model..."):
    try:
        forecast = run_forecast(close, forecast_horizon, model_choice)
        forecast_ok = True
    except Exception as e:
        forecast_ok = False
        st.warning(f"Forecast model failed to fit: {e}")
        forecast = pd.Series(dtype=float)

# ---------------------------------------------------------------------------
# Chart (Trading-Economics style: blue area + dashed gray forecast)
# ---------------------------------------------------------------------------

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=close.index,
        y=close.values,
        name="Historical",
        mode="lines",
        line=dict(color=T.CYAN, width=1.5),
        fill="tozeroy",
        fillcolor="rgba(0, 224, 255, 0.12)",
    )
)

if forecast_ok and not forecast.empty:
    # Connect last historical point to forecast for a continuous line
    bridge_x = [close.index[-1]] + list(forecast.index)
    bridge_y = [last_price] + list(forecast.values)
    fig.add_trace(
        go.Scatter(
            x=bridge_x,
            y=bridge_y,
            name=f"Forecast ({model_choice})",
            mode="lines",
            line=dict(color=T.YELLOW, width=2, dash="dash"),
        )
    )

# Keep the area-fill (fill="tozeroy") from dragging the y-axis down to 0,
# which would flatten high-priced instruments (Gold ~2600, JPY crosses ~150)
# into an unreadable sliver. Pin the range to the data with a small pad; the
# fill still renders and simply clips at the axis floor.
_yvals = list(close.values) + (list(forecast.values) if forecast_ok and not forecast.empty else [])
_ymin, _ymax = float(np.nanmin(_yvals)), float(np.nanmax(_yvals))
_pad = (_ymax - _ymin) * 0.08 or abs(_ymax) * 0.001

fig.update_layout(
    title=f"{pair_label} \u2014 Historical Price & This-Week Forecast ({forecast_horizon}d)",
    yaxis_title="Price",
    xaxis_title=None,
    height=520,
    hovermode="x unified",
    template="plotly_dark",
    paper_bgcolor=T.BG, plot_bgcolor=T.BG_PANEL,
    font=dict(color=T.WHITE, family=T.FONT_MONO, size=11),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                bgcolor=T.BG, bordercolor=T.AMBER, borderwidth=1,
                font=dict(color=T.AMBER, family=T.FONT_MONO)),
    hoverlabel=dict(bgcolor=T.BG, bordercolor=T.AMBER,
                    font=dict(color=T.AMBER, family=T.FONT_MONO)),
    margin=dict(l=40, r=40, t=60, b=40),
)
fig.update_xaxes(gridcolor=T.BORDER, linecolor=T.BORDER)
fig.update_yaxes(gridcolor=T.BORDER, linecolor=T.BORDER, range=[_ymin - _pad, _ymax + _pad])

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Key metrics row
# ---------------------------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)
_chg_str = f"{day_chg:+.4f}" if abs(last_price) < 100 else f"{day_chg:+.2f}"
c1.metric("Current price", fmt_price(last_price), f"{_chg_str} ({day_pct:+.2f}%)")
c2.metric("1-month change", f"{month_pct:+.2f}%" if not np.isnan(month_pct) else "n/a")
c3.metric("1-year change", f"{year_pct:+.2f}%" if not np.isnan(year_pct) else "n/a")
if forecast_ok and not forecast.empty:
    end_val = float(forecast.iloc[-1])
    end_pct = (end_val - last_price) / last_price * 100
    c4.metric(
        "End-of-week forecast",
        fmt_price(end_val),
        f"{end_pct:+.2f}% vs current",
    )
else:
    c4.metric("End-of-week forecast", "n/a")

st.caption(
    "Forecasts are produced with a simple statistical time-series model fitted on "
    "historical Yahoo Finance data and **do not** constitute investment advice."
)

# ---------------------------------------------------------------------------
# Macro context from FRED
# ---------------------------------------------------------------------------

st.header("Macro Context (FRED)")

fred_start = datetime.today() - timedelta(days=365 * 5)
fred_end = datetime.today()

fred_cols = st.columns(2)
any_fred_data = False

for i, (name, series_id) in enumerate(FRED_SERIES.items()):
    fred_df = load_fred_series(series_id, fred_start, fred_end)
    with fred_cols[i % 2]:
        if not fred_df.empty:
            any_fred_data = True
            latest_val = fred_df.iloc[-1, 0]
            latest_date = fred_df.index[-1].strftime("%b %Y")
            st.markdown(f"**{name}**  \nLatest: {latest_val:.2f} ({latest_date})")
            st.line_chart(fred_df, height=180, use_container_width=True)
        else:
            st.markdown(f"**{name}**")
            st.caption("Data unavailable from FRED right now.")

if not any_fred_data:
    st.info(
        "FRED data could not be loaded. If running locally, ensure you have an "
        "internet connection (no API key is required for these public series)."
    )

# ---------------------------------------------------------------------------
# Other currency pairs quote table
# ---------------------------------------------------------------------------

st.header("Markets Snapshot — Forex & Metals")

quotes_df = load_quote_table(CURRENCY_PAIRS)

if not quotes_df.empty:
    def fmt_pct(x):
        color = f"color: {T.RED}" if x < 0 else f"color: {T.GREEN}"
        return f"<span style='{color}'>{x:+.2f}%</span>"

    display_df = quotes_df.copy()
    display_df["% Chg"] = display_df["% Chg"].apply(fmt_pct)
    st.markdown(
        display_df.to_html(escape=False, index=False),
        unsafe_allow_html=True,
    )
else:
    st.info("Quote table unavailable right now.")

st.markdown("---")
st.caption(
    "Built with Streamlit, yfinance and FRED (via pandas-datareader). "
    "Data may be delayed and is provided for informational purposes only."
)