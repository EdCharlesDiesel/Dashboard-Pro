"""
surprise_tab.py — Economic Surprise Awareness module (standalone dashboard
page, auto-registered from pages/ — see src/pages_lib/navigation.py's SURP
entry).
=====================================================
Three jobs:
  1. SURPRISE INDEX   — z-scored actual-vs-forecast per release (Citi-ESI style),
                        journaled to PostgreSQL.
  2. EVENT GATE       — boolean gate for the confluence engine: suppress signals
                        when a red-folder event is within a window BEFORE or AFTER now.
  3. REGIME INVERSION — rolling gold~oil correlation so you can see when
                        "war headline = buy gold" has flipped (it flipped in Feb 2026).

Follows the render(conn) / *_tab.py pattern (SQLAlchemy + Plotly), matching
src/services/evening_sentry.py and src/services/news_fetcher.py, which share
this page's data: CALENDAR_TABLE ("sentry_news") is the table news_fetcher.py
writes the ForexFactory feed to (title/currency/impact/ts_utc/actual/forecast/
previous — aliased to "event" below to match this module's own column names).
Telegram sends go through src.core.secrets.send_telegram_message ([telegram]
in secrets.toml) when run as this page; the standalone evening_sentry.py CLI
uses its own Telegram class (TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID env vars)
since it has no Streamlit runtime to read secrets.toml from.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import text

from src.core.secrets import send_telegram_message
from src.instruments.registry import INSTRUMENTS as _REGISTRY
from src.pages_lib.navigation import render_sidebar_nav
from src.ui.theme import BloombergTheme as T

CALENDAR_TABLE = "sentry_news"          # written by src/services/news_fetcher.py
SURPRISE_TABLE = "econ_surprises"
MIN_HISTORY = 6                          # releases needed before z-score is trusted
Z_ALERT = 1.5                            # |z| threshold for sentry alert

# Which currencies' red events matter per instrument — derived from the
# shared registry (was previously a hardcoded 5-symbol subset: gold, silver,
# EUR/USD, GBP/USD, USD/JPY) so every registry pair/commodity gets gated, not
# just gold. Compact symbol keys ("XAUUSD" not "XAU/USD", "WTI" not "WTI/USD")
# match this app's other pages (abr_toolkit_tab.py, forecast_tab.py).
_SYMBOL_OVERRIDES = {"WTI/USD": "WTI"}
INSTRUMENT_LEGS: dict[str, tuple[str, str]] = {}   # compact symbol -> (base, quote)
INSTRUMENT_CCY: dict[str, set] = {}
for _name in _REGISTRY.keys():
    _sym = _SYMBOL_OVERRIDES.get(_name, _name.replace("/", ""))
    _base, _quote = _name.split("/")
    INSTRUMENT_LEGS[_sym] = (_base, _quote)
    INSTRUMENT_CCY[_sym] = {_base, _quote}


def implied_direction(instrument: str, surprise_currency: str) -> int | None:
    """+1 if a positive surprise in `surprise_currency`'s data is bullish for
    `instrument`, -1 if bearish, None if that currency isn't one of the
    instrument's two legs (INSTRUMENT_LEGS) — i.e. not relevant to it.

    Standard FX rate-differential logic, generalized from the previous
    gold-only rule ("hot US inflation/jobs -> hike bets -> USD up -> gold
    DOWN"): a beat in the BASE currency's data strengthens the base (pair up,
    +1); a beat in the QUOTE currency's data strengthens the quote (pair —
    or a USD-priced commodity, whose 'base' is the commodity itself and never
    matches a real currency code — down, -1). Same regime-dependent caveat as
    the gold~oil inversion below: assumes 'stronger data -> more hawkish ->
    currency up', which can invert in easing/risk-off regimes.
    """
    legs = INSTRUMENT_LEGS.get(instrument)
    if legs is None:
        return None
    base, quote = legs
    if surprise_currency == base:
        return 1
    if surprise_currency == quote:
        return -1
    return None

EVENT_CATEGORY = [
    (re.compile(r"cpi|ppi|pce|inflation", re.I), "inflation"),
    (re.compile(r"non-?farm|nfp|payroll|unemployment|jobless|adp", re.I), "employment"),
    (re.compile(r"gdp|retail sales|ism|pmi", re.I), "growth"),
    (re.compile(r"rate|fomc|testif|minutes|press conference", re.I), "policy"),
]


# ----------------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------------

DDL = f"""
CREATE TABLE IF NOT EXISTS {SURPRISE_TABLE} (
    id            SERIAL PRIMARY KEY,
    event         TEXT NOT NULL,
    category      TEXT,
    currency      TEXT NOT NULL,
    impact        TEXT,
    ts_utc        TIMESTAMPTZ NOT NULL,
    actual        DOUBLE PRECISION,
    forecast      DOUBLE PRECISION,
    raw_surprise  DOUBLE PRECISION,
    z_score       DOUBLE PRECISION,
    UNIQUE (event, currency, ts_utc)
);
"""


def ensure_schema(conn) -> None:
    conn.execute(text(DDL))
    conn.commit()


# ----------------------------------------------------------------------------
# Parsing & scoring
# ----------------------------------------------------------------------------

_NUM = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*([%kmbt]?)\s*$", re.I)
_MULT = {"": 1.0, "%": 1.0, "k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}


def parse_ff_value(raw) -> float | None:
    """'3.8%' -> 3.8, '220K' -> 220000, '-0.1' -> -0.1, '' -> None."""
    if raw is None:
        return None
    m = _NUM.match(str(raw))
    if not m:
        return None
    return float(m.group(1)) * _MULT[m.group(2).lower()]


def categorize(event: str) -> str:
    for pat, cat in EVENT_CATEGORY:
        if pat.search(event):
            return cat
    return "other"


def compute_surprises(conn) -> int:
    """Pull released events from the calendar table, z-score against each
    event's own history, upsert into econ_surprises. Returns rows written."""
    cal = pd.read_sql(
        text(
            f"""SELECT title AS event, currency, impact, ts_utc, actual, forecast
                FROM {CALENDAR_TABLE}
                WHERE actual IS NOT NULL AND actual <> ''
                  AND forecast IS NOT NULL AND forecast <> ''
                ORDER BY ts_utc"""
        ),
        conn,
    )
    if cal.empty:
        return 0

    cal["actual_v"] = cal["actual"].map(parse_ff_value)
    cal["forecast_v"] = cal["forecast"].map(parse_ff_value)
    cal = cal.dropna(subset=["actual_v", "forecast_v"])
    cal["raw"] = cal["actual_v"] - cal["forecast_v"]

    written = 0
    for (event, ccy), grp in cal.groupby(["event", "currency"]):
        grp = grp.sort_values("ts_utc").reset_index(drop=True)
        for i, row in grp.iterrows():
            hist = grp["raw"].iloc[:i]  # strictly prior releases only
            if len(hist) >= MIN_HISTORY:
                sigma = hist.std(ddof=1)
                z = row["raw"] / sigma if sigma and sigma > 1e-12 else None
            else:
                z = None
            conn.execute(
                text(
                    f"""INSERT INTO {SURPRISE_TABLE}
                        (event, category, currency, impact, ts_utc,
                         actual, forecast, raw_surprise, z_score)
                        VALUES (:e, :c, :ccy, :imp, :ts, :a, :f, :r, :z)
                        ON CONFLICT (event, currency, ts_utc)
                        DO UPDATE SET actual = EXCLUDED.actual,
                                      raw_surprise = EXCLUDED.raw_surprise,
                                      z_score = EXCLUDED.z_score"""
                ),
                {
                    "e": event, "c": categorize(event), "ccy": ccy,
                    "imp": row.get("impact"), "ts": row["ts_utc"],
                    "a": row["actual_v"], "f": row["forecast_v"],
                    "r": row["raw"], "z": z,
                },
            )
            written += 1
    conn.commit()
    return written


# ----------------------------------------------------------------------------
# 2. Event-proximity gate — call this from the confluence engine
# ----------------------------------------------------------------------------

def event_gate(
    conn,
    instrument: str,
    now_utc: datetime | None = None,
    hours_before: float = 4.0,
    hours_after: float = 2.0,
    impacts: tuple = ("high",),   # sentry_news.impact is stored lowercase
) -> dict:
    """Returns {'blocked': bool, 'events': [...], 'reason': str}.

    blocked=True when a high-impact event for the instrument's currencies is
    UPCOMING within `hours_before` or was RELEASED within `hours_after`.
    Today's lesson: CPI 14:30 SAST + Warsh 21:00 sandwiched your whole
    17:00-20:00 window — this gate would have flagged both.
    """
    now_utc = now_utc or datetime.now(timezone.utc)
    ccys = INSTRUMENT_CCY.get(instrument, {"USD"})
    lo = now_utc - timedelta(hours=hours_after)
    hi = now_utc + timedelta(hours=hours_before)

    rows = conn.execute(
        text(
            f"""SELECT title AS event, currency, impact, ts_utc
                FROM {CALENDAR_TABLE}
                WHERE ts_utc BETWEEN :lo AND :hi
                  AND currency = ANY(:ccys)
                  AND impact = ANY(:imps)
                ORDER BY ts_utc"""
        ),
        {"lo": lo, "hi": hi, "ccys": list(ccys), "imps": list(impacts)},
    ).fetchall()

    events = [dict(r._mapping) for r in rows]
    if events:
        names = ", ".join(f"{e['event']} ({e['ts_utc']:%H:%M} UTC)" for e in events)
        return {"blocked": True, "events": events,
                "reason": f"Red-folder proximity: {names}"}
    return {"blocked": False, "events": [], "reason": "clear"}


# ----------------------------------------------------------------------------
# 3. Shock detector — realized move vs GARCH-implied sigma
# ----------------------------------------------------------------------------

def shock_check(returns: pd.Series, garch_sigma: float, k: float = 2.5) -> dict:
    """Latest return vs conditional sigma from your GARCH module.
    |r_t| > k * sigma_t  ->  shock. Sentry should say 'stand aside', not signal."""
    r = float(returns.iloc[-1])
    ratio = abs(r) / garch_sigma if garch_sigma > 1e-12 else np.nan
    return {"shock": bool(ratio > k), "return": r,
            "sigma": garch_sigma, "ratio": ratio}


# ----------------------------------------------------------------------------
# 4. Regime inversion — rolling gold~oil correlation
# ----------------------------------------------------------------------------

def gold_oil_corr(window: int = 20) -> pd.Series | None:
    """20d rolling corr of GC=F vs BZ=F daily returns.
    Positive = classic safe-haven co-move on geopolitics.
    Negative = the current oil->inflation->hike->USD chain. Sign flip = regime change."""
    try:
        import yfinance as yf
        px = yf.download(["GC=F", "BZ=F"], period="1y",
                         progress=False)["Close"].dropna()
        rets = px.pct_change().dropna()
        return rets["GC=F"].rolling(window).corr(rets["BZ=F"]).dropna()
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Sentry hook — call from evening_sentry.py after each poll cycle
# ----------------------------------------------------------------------------

def sentry_surprise_alerts(conn, send_telegram, lookback_minutes: int = 90) -> None:
    """Alert on any |z| > Z_ALERT release in the last N minutes, with the
    direction implied for every registry instrument that has this currency
    as one of its two legs (was previously gold-only)."""
    since = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
    rows = conn.execute(
        text(
            f"""SELECT event, category, currency, ts_utc, actual, forecast, z_score
                FROM {SURPRISE_TABLE}
                WHERE ts_utc >= :since AND ABS(COALESCE(z_score, 0)) >= :z"""
        ),
        {"since": since, "z": Z_ALERT},
    ).fetchall()

    for r in rows:
        implied = []
        for instrument in INSTRUMENT_LEGS:
            d = implied_direction(instrument, r.currency)
            if d is not None:
                bias = "BEARISH" if (d * np.sign(r.z_score)) < 0 else "BULLISH"
                implied.append(f"{instrument} {bias}")
        bias_line = f" | implied: {', '.join(sorted(implied))}" if implied else ""
        send_telegram(
            f"SURPRISE {r.event} ({r.currency}) z={r.z_score:+.2f} "
            f"act={r.actual} vs fcst={r.forecast}{bias_line}"
        )


def _send_telegram(msg: str) -> None:
    ok, detail = send_telegram_message(msg)
    if not ok:
        st.caption(f"Telegram send skipped: {detail}")


# ----------------------------------------------------------------------------
# Streamlit tab
# ----------------------------------------------------------------------------

def render(conn):
    import plotly.graph_objects as go

    st.title("Surprise Awareness")
    st.caption("Economic surprise index · event gate · gold~oil regime inversion")
    ensure_schema(conn)

    inst_keys = sorted(INSTRUMENT_LEGS.keys())
    with st.sidebar:
        st.subheader("Surprise Awareness")
        instrument = st.selectbox("Instrument", inst_keys,
                                  index=inst_keys.index("XAUUSD"))
        if st.button("✉️ Send Telegram test message", width="stretch"):
            ok, detail = send_telegram_message(
                "Dashboard Pro — Surprise Awareness test message.")
            (st.success if ok else st.error)("Sent — check Telegram." if ok else detail)
        st.divider()
        render_sidebar_nav()

    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("Recompute surprises"):
            n = compute_surprises(conn)
            st.success(f"{n} rows upserted")
            sentry_surprise_alerts(conn, _send_telegram)

    # --- live gate status ------------------------------------------------
    gate = event_gate(conn, instrument)
    if gate["blocked"]:
        st.error(f"{instrument} SIGNALS GATED — {gate['reason']}")
    else:
        st.success(f"No red-folder event within the gate window for {instrument}")

    # --- surprise history ----------------------------------------------------
    ccys = sorted(INSTRUMENT_CCY.get(instrument, {"USD"}))
    df = pd.read_sql(
        text(
            f"""SELECT event, category, currency, ts_utc, z_score
                FROM {SURPRISE_TABLE}
                WHERE currency = ANY(:ccys) AND z_score IS NOT NULL
                ORDER BY ts_utc DESC LIMIT 60"""
        ),
        conn, params={"ccys": ccys},
    )
    if not df.empty:
        df = df.sort_values("ts_utc")
        fig = go.Figure(
            go.Bar(
                x=df["ts_utc"], y=df["z_score"],
                marker_color=np.where(df["z_score"] >= 0, "#2ca02c", "#d62728"),
                hovertext=df["event"],
            )
        )
        fig.add_hline(y=Z_ALERT, line_dash="dot")
        fig.add_hline(y=-Z_ALERT, line_dash="dot")
        fig.update_layout(title=f"{'/'.join(ccys)} data surprises (z-scored) — {instrument}",
                          yaxis_title="z", height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No scored surprises yet — run recompute once the calendar "
                "table has actual+forecast pairs.")

    # --- regime inversion --------------------------------------------------
    corr = gold_oil_corr()
    if corr is not None and not corr.empty:
        fig2 = go.Figure(go.Scatter(x=corr.index, y=corr.values, mode="lines"))
        fig2.add_hline(y=0, line_dash="dot")
        fig2.update_layout(
            title="Gold ~ Brent 20d rolling correlation "
                  "(negative = escalation is BEARISH gold)",
            yaxis_title="corr", height=350,
        )
        st.plotly_chart(fig2, use_container_width=True)
        latest = float(corr.iloc[-1])
        st.metric("Current gold~oil correlation", f"{latest:+.2f}",
                  help="Sign flip vs your mental model = do not trade the headline.")


@st.cache_resource(show_spinner=False)
def _sentry_store():
    """src.services.evening_sentry.Store, pointed at this app's actual DB
    target (same Neon/Postgres instance as TradeRepository — see
    src.core.secrets.db_config()) instead of its default sqlite fallback.
    Instantiating Store() creates sentry_news/sentry_journal/sentry_runs/
    sentry_meta if they don't exist yet, so this page's queries against
    CALENDAR_TABLE ("sentry_news") never hit a missing-table error on first
    load even if src/services/news_fetcher.py has never been run. Cached so
    reruns reuse one connection pool instead of opening a fresh one on every
    widget interaction."""
    from src.core.secrets import db_config
    from src.services.evening_sentry import Store
    cfg = db_config()
    url = (f"postgresql+psycopg2://{cfg['user']}:{cfg['password']}"
          f"@{cfg['host']}:{cfg['port']}/{cfg['dbname']}")
    return Store(url=url)


st.set_page_config(page_title="Surprise Awareness", layout="wide")
T.apply()
with _sentry_store().engine.connect() as _conn:
    render(_conn)