"""
AMD Market-Phase Scanner — Streamlit app
=========================================
Identifies candidate Accumulation / Manipulation / Distribution (AMD) phases on
price data and explains how each phase is traded in the Wyckoff / smart-money
framework.

Run with:
    pip install streamlit yfinance plotly pandas numpy
    streamlit run amd_app.py

Educational tool only. Nothing here is financial advice. Heuristic phase labels
are descriptive, not predictive — markets do not move in tidy A→M→D loops.
"""

from __future__ import annotations
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
@st.cache_data(show_spinner=False)
def load_yfinance(symbol: str, period: str, interval: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(symbol, period=period, interval=interval,
                     auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise ValueError("No data returned. Check the symbol / period / interval.")
    return df


PHASE_COLORS = {
    "accumulation": "#3b82f6",   # blue
    "manipulation": "#f59e0b",   # amber
    "distribution": "#a855f7",   # purple
    "neutral": "rgba(0,0,0,0)",
}


def make_chart(labeled: pd.DataFrame, symbol: str) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.03,
        subplot_titles=(f"{symbol} — AMD phases", "Volume"),
    )
    fig.add_trace(
        go.Candlestick(
            x=labeled.index, open=labeled["Open"], high=labeled["High"],
            low=labeled["Low"], close=labeled["Close"], name="price",
            increasing_line_color="#16a34a", decreasing_line_color="#dc2626",
        ),
        row=1, col=1,
    )
    # Shade detected phases as vertical bands + markers
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
            row=1, col=1,
        )
    if labeled["Volume"].notna().any():
        fig.add_trace(
            go.Bar(x=labeled.index, y=labeled["Volume"], name="volume",
                   marker_color="#94a3b8"),
            row=2, col=1,
        )
    fig.update_layout(
        height=680, xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=50, b=10),
        template="plotly_white",
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
def main():
    st.set_page_config(page_title="AMD Market-Phase Scanner", layout="wide")
    st.title("📊 AMD Market-Phase Scanner")
    st.caption(
        "Accumulation → Manipulation → Distribution. Heuristic, educational, "
        "**not financial advice.**"
    )

    with st.sidebar:
        st.header("Data")
        source = st.radio("Source", ["Yahoo Finance", "Upload CSV"], index=0)
        df_raw = None
        symbol = "—"
        if source == "Yahoo Finance":
            symbol = st.text_input("Symbol", value="BTC-USD").strip()
            period = st.selectbox("Period",
                                  ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=2)
            interval = st.selectbox("Interval",
                                    ["15m", "30m", "1h", "1d", "1wk"], index=3)
            fetch = st.button("Fetch data", type="primary")
        else:
            up = st.file_uploader("CSV with Open/High/Low/Close (+Volume)", type=["csv"])
            symbol = "uploaded.csv"
            fetch = up is not None

        st.divider()
        st.header("Detection settings")
        range_window = st.slider("Range window (bars)", 5, 60, 20)
        atr_period = st.slider("ATR period", 5, 30, 14)
        consolidation_pctile = st.slider("Consolidation tightness (pctile)",
                                          0.10, 0.70, 0.35, 0.05)
        sweep_atr_mult = st.slider("Sweep depth (×ATR)", 0.05, 1.0, 0.15, 0.05)
        expansion_atr_mult = st.slider("Expansion body (×ATR)", 0.5, 3.0, 1.3, 0.1)

    # ---- load data ----
    df_raw = None
    if source == "Yahoo Finance":
        if fetch and symbol:
            try:
                df_raw = load_yfinance(symbol, period, interval)
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not load data: {e}")
        elif "df_cache" in st.session_state:
            df_raw = st.session_state["df_cache"]
    else:
        if fetch:
            try:
                df_raw = pd.read_csv(up)
                # try to set a datetime index if a date-like column exists
                for cand in ("Date", "date", "Datetime", "timestamp", "time"):
                    if cand in df_raw.columns:
                        df_raw[cand] = pd.to_datetime(df_raw[cand])
                        df_raw = df_raw.set_index(cand)
                        break
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not read CSV: {e}")

    tab_chart, tab_phases, tab_play, tab_about = st.tabs(
        ["📈 Chart", "🧩 Detected phases", "📘 How to trade each phase", "ℹ️ About"]
    )

    if df_raw is not None and not df_raw.empty:
        st.session_state["df_cache"] = df_raw
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
            with tab_chart:
                c1, c2, c3 = st.columns(3)
                c1.metric("Last close", f"{labeled['Close'].iloc[-1]:,.2f}")
                c2.metric("Current phase", assess["phase"].capitalize())
                c3.metric("Read / bias", str(assess["bias"]))
                st.info(assess["detail"])
                st.plotly_chart(make_chart(labeled, symbol), use_container_width=True)

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
            st.write("👈 Pick a data source in the sidebar and fetch data to begin.")

    # ---- playbook always available ----
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

**What it is not.** It is not predictive, not a signal service, and not financial
advice. Phase labels are descriptive geometry — real markets are noisy and do not
move in clean A→M→D loops. Past structure does not imply future structure. Tune
the sliders and judge the labels critically; use it to *study* structure, manage
your own risk, and make your own decisions.
            """
        )
        st.caption("Built with Streamlit · plotly · pandas. Data via Yahoo Finance "
                   "(unofficial) or your own CSV.")


if __name__ == "__main__":
    main()
