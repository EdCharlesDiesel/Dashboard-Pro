"""
AMD Market-Phase Scanner — Streamlit page
=========================================
Identifies candidate Accumulation / Manipulation / Distribution (AMD) phases on
price data and explains how each phase is traded in the Wyckoff / smart-money
framework.

Educational tool only. Nothing here is financial advice. Heuristic phase labels
are descriptive, not predictive — markets do not move in tidy A→M→D loops.
"""

from __future__ import annotations
import smtplib
from email.message import EmailMessage
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def send_gmail(sender: str, app_password: str, recipient: str,
               subject: str, body: str) -> tuple[bool, str]:
    """Send an email via Gmail SMTP using an App Password."""
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.set_content(body)
    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=15) as s:
            s.starttls()
            s.login(sender, app_password)
            s.send_message(msg)
        return True, "sent"
    except Exception as e:  # noqa: BLE001
        return False, str(e)

# --------------------------------------------------------------------------- #
# Instrument registry (mirrors daily-trading-checklist.py)
# --------------------------------------------------------------------------- #
INSTRUMENTS = {
    "EUR/USD":     "EURUSD=X",
    "GBP/USD":     "GBPUSD=X",
    "AUD/USD":     "AUDUSD=X",
    "NZD/USD":     "NZDUSD=X",
    "USD/JPY":     "USDJPY=X",
    "USD/CHF":     "USDCHF=X",
    "USD/CAD":     "USDCAD=X",
    "EUR/GBP":     "EURGBP=X",
    "EUR/JPY":     "EURJPY=X",
    "GBP/JPY":     "GBPJPY=X",
    "AUD/JPY":     "AUDJPY=X",
    "EUR/AUD":     "EURAUD=X",
    "GBP/AUD":     "GBPAUD=X",
    "EUR/CAD":     "EURCAD=X",
    "GBP/CAD":     "GBPCAD=X",
    "USD/ZAR":     "USDZAR=X",
    "EUR/ZAR":     "EURZAR=X",
    "GBP/ZAR":     "GBPZAR=X",
    "🥇 Gold":     "GC=F",
    "🥈 Silver":   "SI=F",
    "🪙 Platinum": "PL=F",
}


# --------------------------------------------------------------------------- #
# Detection engine (framework-free)
# --------------------------------------------------------------------------- #
def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    """Per-bar true range (no smoothing). Used as a volume-like activity proxy."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    return pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] for c in out.columns]
    rename = {}
    for c in out.columns:
        cl = str(c).strip().lower()
        if cl in ("open", "o"):
            rename[c] = "Open"
        elif cl in ("high", "h"):
            rename[c] = "High"
        elif cl in ("low", "l"):
            rename[c] = "Low"
        elif cl in ("close", "c"):
            rename[c] = "Close"
        elif cl in ("adj close", "adj_close", "adjclose"):
            rename[c] = "AdjClose"
        elif cl in ("volume", "vol", "v"):
            rename[c] = "Volume"
    out = out.rename(columns=rename)
    if "Close" not in out.columns and "AdjClose" in out.columns:
        out["Close"] = out["AdjClose"]
    missing = {"Open", "High", "Low", "Close"} - set(out.columns)
    if missing:
        raise ValueError(f"Missing required price columns: {sorted(missing)}")
    if "Volume" not in out.columns:
        out["Volume"] = np.nan
    return out[["Open", "High", "Low", "Close", "Volume"]]


def detect_phases(
        df: pd.DataFrame,
        range_window: int = 20,
        atr_period: int = 14,
        consolidation_pctile: float = 0.35,
        sweep_atr_mult: float = 0.15,
        expansion_atr_mult: float = 1.3,
) -> pd.DataFrame:
    data = normalize_columns(df).copy()
    data["ATR"] = atr(data, atr_period)
    data["range_high"] = data["High"].rolling(range_window, min_periods=2).max().shift(1)
    data["range_low"] = data["Low"].rolling(range_window, min_periods=2).min().shift(1)
    data["range_width"] = (data["range_high"] - data["range_low"]) / data["Close"]
    width_thresh = data["range_width"].quantile(consolidation_pctile)

    phases = np.array(["neutral"] * len(data), dtype=object)
    sweep_dir = np.array([""] * len(data), dtype=object)
    notes = np.array([""] * len(data), dtype=object)

    o, h, l, c = (data[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
    rh, rl, rw, a = (data[k].to_numpy() for k in ("range_high", "range_low", "range_width", "ATR"))

    for i in range(len(data)):
        if np.isnan(rh[i]) or np.isnan(a[i]) or a[i] == 0:
            continue
        body = abs(c[i] - o[i])
        swept_high = (h[i] > rh[i] + sweep_atr_mult * a[i]) and (c[i] < rh[i])
        swept_low = (l[i] < rl[i] - sweep_atr_mult * a[i]) and (c[i] > rl[i])
        if swept_high:
            phases[i], sweep_dir[i] = "manipulation", "bearish"
            notes[i] = "Swept highs (buy-side liquidity) and closed back inside"
            continue
        if swept_low:
            phases[i], sweep_dir[i] = "manipulation", "bullish"
            notes[i] = "Swept lows (sell-side liquidity) and closed back inside"
            continue
        broke_up = c[i] > rh[i] and body > expansion_atr_mult * a[i]
        broke_dn = c[i] < rl[i] and body > expansion_atr_mult * a[i]
        if broke_up:
            phases[i], sweep_dir[i] = "distribution", "up"
            notes[i] = "Expansion candle closing above range (markup)"
            continue
        if broke_dn:
            phases[i], sweep_dir[i] = "distribution", "down"
            notes[i] = "Expansion candle closing below range (markdown)"
            continue
        contained = (h[i] <= rh[i]) and (l[i] >= rl[i])
        if (not np.isnan(rw[i])) and rw[i] <= width_thresh and contained:
            phases[i] = "accumulation"
            notes[i] = "Price contained in a tight range"

    data["phase"], data["sweep_dir"], data["notes"] = phases, sweep_dir, notes
    return data


def summarize_phases(labeled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if labeled.empty:
        return pd.DataFrame(columns=["phase", "start", "end", "bars", "ret_%", "note"])
    pc = labeled["phase"].to_numpy()
    idx = labeled.index
    start = 0
    for i in range(1, len(labeled) + 1):
        if i == len(labeled) or pc[i] != pc[start]:
            ph = pc[start]
            if ph != "neutral":
                seg = labeled.iloc[start:i]
                ret = (seg["Close"].iloc[-1] / seg["Close"].iloc[0] - 1) * 100
                rows.append(
                    {
                        "phase": ph,
                        "start": idx[start],
                        "end": idx[i - 1],
                        "bars": i - start,
                        "ret_%": round(float(ret), 2),
                        "note": seg["notes"].iloc[-1],
                    }
                )
            start = i
    return pd.DataFrame(rows)


def current_assessment(labeled: pd.DataFrame, tail: int = 8) -> dict:
    if labeled.empty:
        return {"phase": "unknown", "bias": "n/a", "detail": "No data."}
    recent = labeled.tail(tail)
    last = labeled.iloc[-1]
    phase = last["phase"]
    if "manipulation" in recent["phase"].values:
        last_manip = recent[recent["phase"] == "manipulation"].iloc[-1]
        bias = last_manip["sweep_dir"]
        detail = (
            f"A recent liquidity sweep hints at positioning for a {bias} move. "
            "Watch for an expansion (distribution) leg to confirm."
        )
    elif phase == "accumulation":
        bias = "neutral / coiling"
        detail = ("Price is consolidating. Edge = patience: wait for a sweep of "
                  "one side of the range before committing.")
    elif phase == "distribution":
        bias = last["sweep_dir"]
        detail = (f"An expansion ({bias}) leg is underway. Chasing late is risky; "
                  "the higher-probability entry was after the preceding sweep.")
    else:
        bias = "neutral"
        detail = "No clean AMD structure in the most recent bars."
    return {"phase": phase, "bias": bias, "detail": detail}


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=300, show_spinner=False)
def load_yfinance(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval,
                     auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise ValueError("No data returned. Check the symbol / period / interval.")
    return df


PHASE_COLORS = {
    "accumulation": "#3b82f6",
    "manipulation": "#f59e0b",
    "distribution": "#a855f7",
    "neutral": "rgba(0,0,0,0)",
}


def _has_usable_volume(s: pd.Series) -> bool:
    """Volume is usable when it has at least one non-NaN, non-zero value.
    Yahoo Finance returns all-zero Volume for FX pairs, which renders as an
    empty subplot — treat that as "no volume" and fall back to a range-based
    activity proxy instead."""
    vol = pd.to_numeric(s, errors="coerce")
    return bool(vol.fillna(0).abs().sum() > 0)


def _period_boundaries(idx, interval: str) -> list:
    """X-positions where a major calendar period begins, used to draw separator
    lines on the price panel. The bucket size scales with the bar interval so
    the chart isn't drowned in lines (day boundaries for intraday, week for
    daily, month for weekly)."""
    if len(idx) == 0:
        return []
    try:
        dt_idx = pd.DatetimeIndex(idx)
    except Exception:  # noqa: BLE001
        return []
    if interval in ("15m", "30m", "1h"):
        bucket = dt_idx.normalize()
    elif interval == "1d":
        bucket = dt_idx.to_period("W").to_timestamp()
    elif interval == "1wk":
        bucket = dt_idx.to_period("M").to_timestamp()
    else:
        return []
    boundaries, prev = [], None
    for t, p in zip(dt_idx, bucket):
        if prev is not None and p != prev:
            boundaries.append(t)
        prev = p
    return boundaries


def _volume_profile(labeled: pd.DataFrame, y_series: pd.Series,
                    bins: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Volume-by-price: distribute each bar's volume across the price bins its
    high–low range touches. Returns (bin centers, summed activity per bin)."""
    lo = float(labeled["Low"].min())
    hi = float(labeled["High"].max())
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return np.array([]), np.array([])
    edges = np.linspace(lo, hi, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    profile = np.zeros(bins)
    highs = labeled["High"].to_numpy()
    lows = labeled["Low"].to_numpy()
    vols = y_series.to_numpy()
    for h, l, v in zip(highs, lows, vols):
        if not (np.isfinite(v) and np.isfinite(h) and np.isfinite(l)) or v <= 0:
            continue
        lo_i = int(np.clip(np.searchsorted(edges, l, side="right") - 1, 0, bins - 1))
        hi_i = int(np.clip(np.searchsorted(edges, h, side="right") - 1, 0, bins - 1))
        span = hi_i - lo_i + 1
        profile[lo_i:hi_i + 1] += v / span
    return centers, profile


def make_chart(labeled: pd.DataFrame, symbol: str, interval: str = "1d") -> go.Figure:
    # Spot FX (=X tickers) has no traded volume on Yahoo, so we substitute a
    # range-based activity proxy. Either way the lower-left panel shows it as
    # a horizontal histogram aligned with price levels (volume-by-price).
    real_volume = _has_usable_volume(labeled["Volume"])
    if real_volume:
        y_vals = pd.to_numeric(labeled["Volume"], errors="coerce").fillna(0)
        prof_title = "Volume by price"
        hover_left = "Price: %{y:,.5f}<br>Vol: %{x:,.0f}<extra></extra>"
    else:
        y_vals = true_range(labeled).fillna(0)
        prof_title = "Activity by price (range proxy)"
        hover_left = "Price: %{y:,.5f}<br>Range: %{x:,.5f}<extra></extra>"

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True,
        column_widths=[0.18, 0.82], horizontal_spacing=0.01,
        subplot_titles=(prof_title, f"{symbol} — AMD phases"),
    )

    # ---- LEFT: horizontal volume profile (volume by price level) ----
    centers, profile = _volume_profile(labeled, y_vals, bins=40)
    if len(centers) > 0 and profile.sum() > 0:
        fig.add_trace(
            go.Bar(
                x=profile, y=centers, orientation="h",
                marker=dict(color="#64748b", line=dict(width=0)),
                opacity=0.75, name=prof_title, showlegend=False,
                hovertemplate=hover_left,
            ),
            row=1, col=1,
        )
    # Mirror the x-axis so the bars grow from the price axis outward to the left.
    fig.update_xaxes(autorange="reversed", row=1, col=1,
                     showgrid=False, zeroline=False, title_text="")

    # ---- RIGHT: candles + phase markers ----
    fig.add_trace(
        go.Candlestick(
            x=labeled.index, open=labeled["Open"], high=labeled["High"],
            low=labeled["Low"], close=labeled["Close"], name="price",
            increasing_line_color="#16a34a", decreasing_line_color="#dc2626",
        ),
        row=1, col=2,
    )
    for phase, color in PHASE_COLORS.items():
        if phase == "neutral":
            continue
        pts = labeled[labeled["phase"] == phase]
        if pts.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=pts.index, y=pts["High"] * 1.002, mode="markers",
                marker=dict(symbol="triangle-down", size=9, color=color),
                name=phase.capitalize(),
                text=pts["notes"], hoverinfo="text+x",
            ),
            row=1, col=2,
        )

    # ---- Period separators on the price chart ----
    for b in _period_boundaries(labeled.index, interval):
        fig.add_vline(
            x=b, line=dict(color="#94a3b8", width=1, dash="dot"),
            row=1, col=2,
        )

    fig.update_xaxes(rangeslider_visible=False, row=1, col=2)
    fig.update_layout(
        height=680,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.2),
        margin=dict(l=10, r=10, t=60, b=10),
        template="plotly_white",
        bargap=0.05,
    )
    return fig


# --------------------------------------------------------------------------- #
# Educational content
# --------------------------------------------------------------------------- #
PLAYBOOK = {
    "Accumulation": {
        "what": "A low-volatility range where larger participants quietly build "
                "positions. Price chops sideways; ranges contract; volume often "
                "fades into the range. This is the 'spring loading' stage.",
        "spot": [
            "Tight, overlapping candles contained in a horizontal band.",
            "Falling volatility (ATR) and shrinking range width.",
            "Clear, repeatedly tested support and resistance edges (liquidity pools "
            "sitting just beyond them).",
        ],
        "trade": [
            "Do NOT trade the chop. Your job here is preparation, not entry.",
            "Mark the range high and low — those edges are where stops cluster.",
            "Decide in advance which side a sweep would have to take to interest you.",
            "Note the higher-timeframe trend; ranges usually resolve with it.",
        ],
        "risk": "The trap is over-trading the range and getting whipsawed. Sitting "
                "out is a position.",
    },
    "Manipulation": {
        "what": "A deliberate false move — a stop hunt / 'judas swing' — that pokes "
                "beyond the range to grab the liquidity resting past the obvious "
                "high or low, then snaps back inside. It engineers the fuel for the "
                "real move in the opposite direction of the sweep.",
        "spot": [
            "A wick that breaks the range high/low but the candle CLOSES back inside.",
            "A sharp spike on rising volume that immediately reverses.",
            "Failure to follow through after an apparent breakout (breakout fakeout).",
        ],
        "trade": [
            "This is the high-probability moment. A sweep of the lows that reclaims "
            "is a bullish signal; a sweep of the highs that rejects is bearish.",
            "Wait for the reclaim/close back inside — don't catch the falling knife "
            "mid-sweep.",
            "Enter on the reclaim or on a lower-timeframe shift in structure; place "
            "the stop just beyond the sweep extreme (now an invalidated level).",
            "Target the opposite side of the range and beyond, where the next "
            "liquidity pool sits.",
        ],
        "risk": "Not every wick is a manipulation; a genuine breakout also starts "
                "with a poke. Confirmation (the reclaim) is what separates them.",
    },
    "Distribution": {
        "what": "The expansion / delivery phase: the impulsive directional move that "
                "follows the manipulation. Smart money offloads (or the trend leg "
                "plays out). In Wyckoff terms this is the markup or markdown.",
        "spot": [
            "Large-bodied candles closing decisively outside the prior range.",
            "Momentum and break of market structure in one direction.",
            "Expanding volatility (rising ATR) and often a volume surge on the break.",
        ],
        "trade": [
            "If you entered on the manipulation, this is where you HOLD and manage.",
            "Trail the stop behind structure (e.g. behind each new higher low in an "
            "uptrend) rather than guessing the top.",
            "Scale out into the next liquidity pool / measured-move target.",
            "Entering fresh this late is low edge — you'd be buying the move others "
            "are distributing into. If you must, wait for a pullback to a tested "
            "level, not a chase.",
        ],
        "risk": "Chasing the expansion is the most common retail mistake. The good "
                "entry already happened during manipulation.",
    },
}


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="AMD Market-Phase Scanner", layout="wide")
st.title("📊 AMD Market-Phase Scanner")
st.caption(
    "Accumulation → Manipulation → Distribution. Heuristic, educational, "
    "**not financial advice.**"
)

with st.sidebar:
    st.header("Data")
    inst_keys = list(INSTRUMENTS.keys())
    default_idx = inst_keys.index("EUR/USD") if "EUR/USD" in inst_keys else 0
    instrument = st.selectbox("Symbol", inst_keys, index=default_idx)
    symbol = INSTRUMENTS[instrument]
    st.caption(f"📡 Ticker: `{symbol}`")
    period = st.selectbox("Period",
                          ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=2)
    interval = st.selectbox("Interval",
                            ["15m", "30m", "1h", "1d", "1wk"], index=3)
    fetch = st.button("Fetch data", type="primary")

    st.divider()
    st.header("Detection settings")
    range_window = st.slider("Range window (bars)", 5, 60, 20)
    atr_period = st.slider("ATR period", 5, 30, 14)
    consolidation_pctile = st.slider("Consolidation tightness (pctile)",
                                     0.10, 0.70, 0.35, 0.05)
    sweep_atr_mult = st.slider("Sweep depth (×ATR)", 0.05, 1.0, 0.15, 0.05)
    expansion_atr_mult = st.slider("Expansion body (×ATR)", 0.5, 3.0, 1.3, 0.1)

    st.divider()
    st.header("📧 Email alerts (Gmail)")
    # Credentials come from .streamlit/secrets.toml under [gmail].
    try:
        gmail_secrets = dict(st.secrets.get("gmail", {}))
    except Exception:  # noqa: BLE001
        gmail_secrets = {}

    gmail_sender = gmail_secrets.get("sender", "")
    gmail_app_pw = gmail_secrets.get("app_password", "")
    gmail_recipient = gmail_secrets.get("recipient", "")

    # Treat the placeholder values as "not configured".
    placeholders = {"your.address@gmail.com", "xxxxxxxxxxxxxxxx", ""}
    creds_ready = (
            gmail_sender not in placeholders
            and gmail_app_pw not in placeholders
            and gmail_recipient not in placeholders
    )

    if creds_ready:
        st.caption(
            f"🔒 Loaded from `.streamlit/secrets.toml`\n\n"
            f"**From:** `{gmail_sender}` → **To:** `{gmail_recipient}`"
        )
    else:
        st.warning(
            "Add `[gmail]` credentials to `.streamlit/secrets.toml`:\n\n"
            "```toml\n[gmail]\n"
            "sender = \"you@gmail.com\"\n"
            "app_password = \"abcd efgh ijkl mnop\"\n"
            "recipient = \"ckhotso@gmail.com\"\n```\n"
            "App password: https://myaccount.google.com/apppasswords (requires 2FA)."
        )

    enable_alerts = st.toggle(
        "Enable email alerts on new phase",
        value=st.session_state.get("amd_alerts_on", False),
        disabled=not creds_ready,
        help="Sends an email when the latest bar's phase becomes manipulation or distribution.",
    )
    st.session_state["amd_alerts_on"] = enable_alerts

    if st.button("✉️ Send test email", use_container_width=True, disabled=not creds_ready):
        ok, info = send_gmail(
            gmail_sender, gmail_app_pw, gmail_recipient,
            "AMD Scanner — test email",
            "If you can read this, Gmail SMTP is wired up correctly.",
        )
        (st.success if ok else st.error)(f"Test: {info}")

# ---- load data ----
df_raw = None
if fetch and symbol:
    try:
        df_raw = load_yfinance(symbol, period, interval)
        st.session_state["amd_df_cache"] = df_raw
        st.session_state["amd_symbol_cache"] = symbol
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not load data: {e}")
elif "amd_df_cache" in st.session_state and st.session_state.get("amd_symbol_cache") == symbol:
    df_raw = st.session_state["amd_df_cache"]

tab_chart, tab_phases, tab_play, tab_about = st.tabs(
    ["📈 Chart", "🧩 Detected phases", "📘 How to trade each phase", "ℹ️ About"]
)

if df_raw is not None and not df_raw.empty:
    try:
        labeled = detect_phases(
            df_raw, range_window=range_window, atr_period=atr_period,
            consolidation_pctile=consolidation_pctile,
            sweep_atr_mult=sweep_atr_mult, expansion_atr_mult=expansion_atr_mult,
        )
    except Exception as e:  # noqa: BLE001
        st.error(f"Detection failed: {e}")
        labeled = None

    if labeled is not None:
        assess = current_assessment(labeled)

        # --- Phase-transition email alert ---
        last_bar = labeled.iloc[-1]
        last_phase = str(last_bar["phase"])
        alert_key = f"amd_last_alerted_{symbol}_{interval}"
        prev_alerted = st.session_state.get(alert_key)
        if (
                enable_alerts
                and creds_ready
                and last_phase in ("manipulation", "distribution")
                and last_phase != prev_alerted
        ):
            subject = f"AMD: {last_phase.upper()} on {instrument} ({interval})"
            body = (
                f"Symbol:    {instrument} ({symbol})\n"
                f"Interval:  {interval}\n"
                f"Bar time:  {last_bar.name}\n"
                f"Phase:     {last_phase}\n"
                f"Sweep dir: {last_bar.get('sweep_dir', '') or '-'}\n"
                f"Close:     {float(last_bar['Close']):,.5f}\n\n"
                f"Note: {last_bar.get('notes', '') or '-'}\n\n"
                f"Read: {assess['detail']}\n"
            )
            ok, info = send_gmail(
                gmail_sender, gmail_app_pw, gmail_recipient, subject, body,
            )
            if ok:
                st.session_state[alert_key] = last_phase
                st.toast(f"📧 Email sent: {last_phase.upper()} on {instrument}", icon="✉️")
            else:
                st.error(f"Email alert failed: {info}")

        with tab_chart:
            c1, c2, c3 = st.columns(3)
            c1.metric("Last close", f"{labeled['Close'].iloc[-1]:,.2f}")
            c2.metric("Current phase", assess["phase"].capitalize())
            c3.metric("Read / bias", str(assess["bias"]))
            st.info(assess["detail"])
            if not _has_usable_volume(labeled["Volume"]):
                st.caption(
                    "ℹ️ No traded volume is available for this instrument "
                    "(spot FX trades over-the-counter, so Yahoo returns zero "
                    "volume). The lower panel shows a **range-based activity "
                    "proxy** instead. For real volume, pick a futures contract "
                    "(Gold, Silver, Platinum)."
                )
            st.plotly_chart(make_chart(labeled, symbol, interval), use_container_width=True)

        with tab_phases:
            seg = summarize_phases(labeled)
            counts = labeled["phase"].value_counts()
            cols = st.columns(3)
            for col, ph in zip(cols, ["accumulation", "manipulation", "distribution"]):
                col.metric(ph.capitalize(), int(counts.get(ph, 0)))
            st.subheader("Phase segments (most recent last)")
            if seg.empty:
                st.write("No non-neutral phases detected with current settings. "
                         "Try widening the range window or lowering the sweep depth.")
            else:
                st.dataframe(seg.iloc[::-1], use_container_width=True, hide_index=True)
            with st.expander("Per-bar labels (raw)"):
                st.dataframe(
                    labeled[["Open", "High", "Low", "Close", "Volume",
                             "phase", "sweep_dir", "notes"]],
                    use_container_width=True,
                )
                st.download_button(
                    "Download labeled data (CSV)",
                    labeled.to_csv().encode("utf-8"),
                    file_name=f"{symbol}_amd_labeled.csv", mime="text/csv",
                )
else:
    with tab_chart:
        st.write("👈 Pick a symbol in the sidebar and click **Fetch data** to begin.")

with tab_play:
    st.write("The AMD cycle in one line: smart money **builds** a position in a "
             "range (accumulation), **fakes out** retail to grab liquidity "
             "(manipulation), then **delivers** the real move (distribution).")
    for phase, info in PLAYBOOK.items():
        color = PHASE_COLORS.get(phase.lower(), "#333")
        st.markdown(f"### <span style='color:{color}'>● </span>{phase}",
                    unsafe_allow_html=True)
        st.markdown(f"**What it is** — {info['what']}")
        st.markdown("**How to spot it**")
        for s in info["spot"]:
            st.markdown(f"- {s}")
        st.markdown("**How it's traded**")
        for s in info["trade"]:
            st.markdown(f"- {s}")
        st.markdown(f"**Main risk** — {info['risk']}")
        st.divider()

with tab_about:
    st.markdown(
        """
**What this app does.** It scans OHLC price data and labels each bar with a
*candidate* AMD phase using simple, transparent rules:

- **Accumulation** — price is contained in a tight rolling range (range width
  below your chosen percentile).
- **Manipulation** — a bar wicks beyond the prior range by ≥ *sweep depth × ATR*
  but closes back inside (a liquidity sweep / stop hunt).
- **Distribution** — a large-bodied bar (> *expansion × ATR*) closes outside the
  range (the impulsive markup/markdown leg).

**A note on volume.** Spot FX pairs (`=X` tickers) trade over-the-counter, so
Yahoo Finance reports no real traded volume for them. For those instruments the
lower chart panel shows a **range-based activity proxy** (per-bar true range) so
the panel stays informative — but this is *not* traded volume. Futures contracts
(Gold `GC=F`, Silver `SI=F`, Platinum `PL=F`) carry genuine exchange volume and
display it directly.

**What it is not.** It is not predictive, not a signal service, and not financial
advice. Phase labels are descriptive geometry — real markets are noisy and do not
move in clean A→M→D loops. Past structure does not imply future structure. Tune
the sliders and judge the labels critically; use it to *study* structure, manage
your own risk, and make your own decisions.
        """
    )
    st.caption("Built with Streamlit · plotly · pandas. Data via Yahoo Finance.")