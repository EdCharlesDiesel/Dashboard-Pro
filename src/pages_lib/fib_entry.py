"""15M Fibonacci Entry — Bloomberg-terminal version.

Merges the former "15M Rejection" and "15M Entry Signal" pages into a single
trigger page that:

1. Pulls the **Setup Ranker** signals (the exact same scorer/data feed used by
   ``pages/setup-ranker.py`` — reuses ``_SetupRankerDataFeed.score()`` directly,
   see :func:`src.core.signals.score_setup`) and only considers pairs that
   score at or above the chosen threshold. The Setup Ranker decides *what* and
   *which direction*; this page finds *where* to enter.
2. Maps Fibonacci retracements onto the most recent 15M impulse leg in the
   signal's direction and flags every candle that retraced into the 0.382–0.618
   "golden zone" and confirmed in the bias direction. Those are the entries.
3. Charts the 15M candles with the full Fibonacci grid, the shaded golden zone,
   and a marker on **every** entry in the loaded window.
4. Emails fresh entries (opt-in) via :mod:`src.services.alert_service`, deduped
   per pair/direction/entry so the same trigger never mails twice.

Sizing and projected growth honour the desk's house rule of two trades per
signal (see ``[[two-trades-per-signal]]``).
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

from src.instruments import INSTRUMENTS
from src.pages_lib.base import BloombergPage, PageContext
from src.pages_lib.setup_ranker import (
    _SetupRankerDataFeed, fmt_price, money_breakdown, trade_levels,
)
from src.services import alert_service, account_state
from src.services.signal_store import persist_signals
from src.ui.components import CommandBar, MetricCell, render_metric_row
from src.ui.theme import BloombergTheme as T
from src.core.config import CANDLE_STYLE

# Fibonacci retracement fractions measured from the impulse END back toward its
# START. 0.0 = leg end (prior swing extreme), 1.0 = leg origin. The 0.382–0.618
# band is the classic "golden zone" entry; 0.786 is the invalidation level.
FIB_FRACTIONS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
GOLDEN_LOW, GOLDEN_HIGH = 0.382, 0.618
INVALIDATION = 0.786
EXT_TARGET = 0.618  # 1.618 extension beyond the leg end for TP2


# ── 15M data ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=180, show_spinner=False)
def _fetch_15m(ticker: str, days: int = 5) -> pd.DataFrame:
    """15-minute OHLCV (yfinance allows 15m up to ~60 days)."""
    from src.db.market_cache import cached_ohlc
    try:
        df = cached_ohlc(ticker, interval="15m", period=f"{days}d", ttl=180)
        if df.empty:
            return pd.DataFrame()
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception:
        return pd.DataFrame()


# ── Fibonacci engine ─────────────────────────────────────────────────────────

def _impulse_leg(df: pd.DataFrame, bias: str, lookback: int) -> Optional[dict]:
    """Most recent impulse leg in the bias direction, inside the last ``lookback``
    candles. LONG → swing-low then the highest high after it; SHORT → swing-high
    then the lowest low after it. Returns ``None`` if no leg can be formed."""
    if df.empty or len(df) < 5:
        return None
    w = df.tail(lookback)
    if bias == "LONG":
        lo_idx = w["Low"].idxmin()
        after = w.loc[lo_idx:]
        if len(after) < 2:
            return None
        hi_idx = after["High"].idxmax()
        low = float(w.loc[lo_idx, "Low"])
        high = float(after.loc[hi_idx, "High"])
        start_idx, end_idx = lo_idx, hi_idx
    else:  # SHORT
        hi_idx = w["High"].idxmax()
        after = w.loc[hi_idx:]
        if len(after) < 2:
            return None
        lo_idx = after["Low"].idxmin()
        high = float(w.loc[hi_idx, "High"])
        low = float(after.loc[lo_idx, "Low"])
        start_idx, end_idx = hi_idx, lo_idx
    diff = high - low
    if diff <= 0:
        return None
    return {"high": high, "low": low, "diff": diff,
            "start_idx": start_idx, "end_idx": end_idx}


def _retrace_price(leg: dict, bias: str, frac: float) -> float:
    """Price at ``frac`` retracement of the leg (direction-aware)."""
    if bias == "LONG":
        return leg["high"] - frac * leg["diff"]      # retrace down from the high
    return leg["low"] + frac * leg["diff"]           # retrace up from the low


def fib_analysis(df: pd.DataFrame, bias: str, lookback: int) -> Optional[dict]:
    """Full Fibonacci-retracement entry analysis for one pair + bias.

    Returns the fib levels, golden-zone band, every confirmed entry candle in the
    retracement phase, a live status, and entry/SL/TP trade levels — or ``None``
    when there isn't enough data to form a leg.
    """
    leg = _impulse_leg(df, bias, lookback)
    if leg is None:
        return None

    levels = {f: _retrace_price(leg, bias, f) for f in FIB_FRACTIONS}
    z1 = _retrace_price(leg, bias, GOLDEN_LOW)
    z2 = _retrace_price(leg, bias, GOLDEN_HIGH)
    zone_lo, zone_hi = min(z1, z2), max(z1, z2)
    invalid = _retrace_price(leg, bias, INVALIDATION)

    # Retracement phase starts at the leg end (where price turns to pull back).
    ret = df.loc[leg["end_idx"]:]

    entries: List[dict] = []
    for ts, row in ret.iterrows():
        if ts == leg["end_idx"]:
            continue
        lo, hi = float(row["Low"]), float(row["High"])
        o, c = float(row["Open"]), float(row["Close"])
        overlaps = (lo <= zone_hi) and (hi >= zone_lo)   # candle range hits zone
        if not overlaps:
            continue
        confirm = (c > o) if bias == "LONG" else (c < o)
        if confirm:
            entries.append({"time": ts, "price": c, "low": lo, "high": hi})

    # Live status from the latest candle.
    last = df.iloc[-1]
    last_c = float(last["Close"])
    in_zone = zone_lo <= last_c <= zone_hi
    if entries and entries[-1]["time"] == df.index[-1]:
        status = "ENTRY_FIRED"
    elif in_zone:
        status = "IN_ZONE"
    elif bias == "LONG" and last_c > zone_hi:
        status = "RETRACING" if last_c < leg["high"] else "EXTENDED"
    elif bias == "SHORT" and last_c < zone_lo:
        status = "RETRACING" if last_c > leg["low"] else "EXTENDED"
    else:
        status = "INVALIDATED"  # price beyond the golden zone past 0.618

    # Trade levels from the latest entry (or the zone mid if none yet).
    entry_px = entries[-1]["price"] if entries else _retrace_price(leg, bias, 0.5)
    sl = invalid
    tp1 = levels[0.0]                                  # leg end (prior swing)
    tp2 = (leg["high"] + EXT_TARGET * leg["diff"]) if bias == "LONG" \
        else (leg["low"] - EXT_TARGET * leg["diff"])  # 1.618 extension
    risk = abs(entry_px - sl)
    rr1 = abs(tp1 - entry_px) / risk if risk > 0 else 0.0
    rr2 = abs(tp2 - entry_px) / risk if risk > 0 else 0.0

    return {
        "leg": leg, "levels": levels,
        "zone_lo": zone_lo, "zone_hi": zone_hi, "invalid": invalid,
        "entries": entries, "status": status,
        "entry": entry_px, "sl": sl, "tp1": tp1, "tp2": tp2,
        "rr1": round(rr1, 2), "rr2": round(rr2, 2),
        "last_close": last_c, "last_time": df.index[-1],
    }


STATUS_CFG = {
    "ENTRY_FIRED": ("⚡ ENTRY FIRED",  T.GREEN,  "Confirmed candle closed inside the golden zone this bar"),
    "IN_ZONE":     ("🎯 IN ZONE",      T.CYAN,   "Price is inside the 0.382–0.618 golden zone — watch for a confirming candle"),
    "RETRACING":   ("🟡 RETRACING",    T.YELLOW, "Price is pulling back toward the golden zone"),
    "EXTENDED":    ("⏳ EXTENDED",     T.GREY,   "Price hasn't started retracing yet"),
    "INVALIDATED": ("✖ PAST ZONE",    T.RED,    "Price retraced beyond 0.618 — setup weakening / invalidated"),
}


# ── Page ─────────────────────────────────────────────────────────────────────

class FibEntryPage(BloombergPage):
    """Fibonacci-retracement entry trigger driven by Setup Ranker signals."""

    def configure(self) -> PageContext:
        st.session_state.setdefault("fib_pairs",
                                    ["EUR/USD", "GBP/USD", "AUD/USD", "USD/JPY",
                                     "USD/CHF", "USD/CAD", "XAU/USD", "XAG/USD",
                                     "WTI/USD"])
        st.session_state.setdefault("fib_min_score", 6)
        st.session_state.setdefault("fib_lookback", 60)
        st.session_state.setdefault("fib_days", 5)
        st.session_state.setdefault("fib_show", 96)
        st.session_state.setdefault("fib_detail", "EUR/USD")
        st.session_state.setdefault("account_bal", account_state.get_balance(10000.0))
        st.session_state.setdefault("risk_pct", 1.0)
        # ON by default (user's standing preference — alerts fire without
        # having to find the toggle; the sidebar switch remains an off-ramp).
        st.session_state.setdefault("fib_email_on", True)
        st.session_state.setdefault("fib_trades_per_signal", 2)
        return PageContext(code="15FB", title="15M Fib Entry", icon="⚡")

    # ── sidebar ──────────────────────────────────────────────────────────────
    def sidebar(self, ctx: PageContext) -> None:
        st.markdown("""
        <style>
            span[data-baseweb="tag"]{
                background-color:#00ff41 !important;
            }
            span[data-baseweb="tag"] span{
                color:#000000 !important;
                font-weight:600 !important;
            }
            span[data-baseweb="tag"] svg{
                fill:#000000 !important;
            }
            span[data-baseweb="tag"] [role="button"]:hover{
                background-color:#00cc34 !important;
            }
        </style>
        """, unsafe_allow_html=True)
        st.session_state.fib_detail = st.selectbox(
            "Detail instrument", sorted(INSTRUMENTS.keys()),  # alphabetical dropdown
            index=sorted(INSTRUMENTS.keys()).index(
                st.session_state.fib_detail
                if st.session_state.fib_detail in INSTRUMENTS else "EUR/USD"),
            help="Pair shown on the detail chart. The scanner runs on the Setup "
                 "Ranker signals below regardless of this pick.",
        )
        st.divider()
        st.markdown(
            '<div style="color:#00ff41;font-weight:700;letter-spacing:0.15em;'
            'text-transform:uppercase;font-size:11px;">SETUP RANKER FILTER</div>',
            unsafe_allow_html=True,
        )
        # Grouped in a form so ticking instruments on/off doesn't fire the
        # expensive multi-pair Setup Ranker rescan on every single click —
        # only "Apply Filter" reruns the scan, so the picker itself never
        # appears to "reset" mid-selection.
        with st.form("fib_filter_form"):
            pairs = st.multiselect(
                "Instruments to scan", sorted(INSTRUMENTS.keys()),  # alphabetical dropdown
                default=[p for p in st.session_state.fib_pairs if p in INSTRUMENTS],
            )
            min_score = st.slider(
                "Min Setup Ranker score", 0, 10, int(st.session_state.fib_min_score),
                help="Only pairs scoring at/above this are analysed for a fib entry — "
                     "same multi-timeframe checklist and threshold as Setup Ranker's "
                     "Min score (out of 10, applied as a percentage since FX pairs "
                     "score /11 with Currency Strength and commodities score /10).",
            )
            if st.form_submit_button("Apply Filter", width="stretch", type="primary"):
                st.session_state.fib_pairs = pairs
                st.session_state.fib_min_score = min_score
        st.divider()
        st.markdown(
            '<div style="color:#00ff41;font-weight:700;letter-spacing:0.15em;'
            'text-transform:uppercase;font-size:11px;">FIB / 15M PARAMS</div>',
            unsafe_allow_html=True,
        )
        st.session_state.fib_lookback = st.slider(
            "Impulse lookback (candles)", 20, 120, int(st.session_state.fib_lookback),
            help="Window used to find the most recent impulse leg the retracement "
                 "is measured against.",
        )
        st.session_state.fib_days = st.slider(
            "Data lookback (days)", 1, 14, int(st.session_state.fib_days))
        st.session_state.fib_show = st.slider(
            "Candles on chart", 40, 240, int(st.session_state.fib_show), step=8)
        st.divider()
        if st.button("◆ RESCAN", width="stretch", type="primary"):
            st.cache_data.clear()
            st.rerun()
        st.caption(f"◷ {datetime.now().strftime('%H:%M')} · TTL 3min")

        # ── Email alerts ─────────────────────────────────────────────────────
        st.divider()
        st.markdown(
            '<div style="color:#00ff41;font-weight:700;letter-spacing:0.15em;'
            'text-transform:uppercase;font-size:11px;">EMAIL ALERTS</div>',
            unsafe_allow_html=True,
        )
        st.session_state.fib_email_on = st.toggle(
            "📧 Email fresh fib entries",
            value=st.session_state.fib_email_on,
            help="On each scan, email any new entry that fired on the latest few "
                 "candles. Each pair/direction/entry mails once.",
        )
        st.session_state.fib_trades_per_signal = st.number_input(
            "Trades per signal", min_value=1, max_value=10,
            value=int(st.session_state.fib_trades_per_signal), step=1,
            help="Sizing in the alert is multiplied by this. Defaults to 2 "
                 "(your house rule).",
        )
        if alert_service.email_configured():
            st.caption(f"✅ Alerts → {alert_service.email_recipient()}")
        else:
            st.caption("⚠ Set [gmail] in secrets.toml (or GMAIL_* env vars) to enable.")
        c_test, c_reset = st.columns(2)
        with c_test:
            if st.button("✉ Test", width="stretch"):
                ok, msg = alert_service.send_email(
                    "15M Fib Entry — test alert",
                    "<p style='font-family:monospace;color:#0a0;'>✅ Fib-entry email "
                    "alerts are wired up correctly.</p>",
                    "Fib-entry email alerts are wired up correctly.", source="test")
                (st.success if ok else st.error)(msg)
        with c_reset:
            if st.button("↺ Reset", width="stretch",
                         help="Clear the already-alerted memory so current entries re-fire."):
                alert_service.NotifyCache("fib_entry").reset()
                st.toast("Alert memory cleared.")

    # ── body ─────────────────────────────────────────────────────────────────
    def body(self, ctx: PageContext) -> None:
        pairs = st.session_state.fib_pairs
        min_score = st.session_state.fib_min_score
        if not pairs:
            st.warning("⚠ Select at least one instrument in the sidebar.")
            return

        CommandBar(label="15FB", cells=[
            ("FIB 0.382–0.618", "amber"),
            (f"SCAN {len(pairs)}", ""),
            (f"MIN {min_score}/10", "cyan"),
            (f"BAL ${st.session_state.account_bal:,.0f}", ""),
            (f"RISK {st.session_state.risk_pct:.2f}%", "cyan"),
            (datetime.now().strftime("%a %d %b %H:%M"), "amber"),
        ]).show()

        # Auto-refreshing fragment (same cadence as the Setup Ranker): while
        # the page is open it re-scans, re-persists, and re-checks email
        # alerts every 5 minutes without a manual rerun. Dedupe ledgers keep
        # repeat scans quiet — only genuinely new entries email/save.
        @st.fragment(run_every=300)
        def _scan_and_render():
            signals = self._scan_signals(pairs, min_score)
            results = self._analyse(signals)

            self._persist_signals(results)

            if st.session_state.get("fib_email_on"):
                self._maybe_email(results)

            n_entries = sum(len(r["fib"]["entries"]) for r in results)
            n_live = sum(1 for r in results if r["fib"]["status"] in ("ENTRY_FIRED", "IN_ZONE"))
            render_metric_row([
                MetricCell("Setup Signals", str(len(signals)), "cyan"),
                MetricCell("With Fib Leg", str(len(results)), "green"),
                MetricCell("Live Entries", str(n_live), "green"),
                MetricCell("Total Entries", str(n_entries), "yellow"),
            ])
            st.caption(f"◷ Auto-refreshes every 5 min · last scan "
                       f"{datetime.now().strftime('%H:%M:%S')}")

            tab_sig, tab_detail, tab_ref = st.tabs(
                ["◆ FIB SIGNALS", "🔍 DETAIL CHART", "📖 METHOD"])
            with tab_sig:
                self._render_signals(results)
            with tab_detail:
                self._render_detail()
            with tab_ref:
                self._render_reference()

        _scan_and_render()

    # ── scanning / analysis ──────────────────────────────────────────────────
    @staticmethod
    def _scan_signals(pairs, min_score) -> List[dict]:
        """Best-direction Setup Ranker score per pair, kept if ≥ threshold."""
        out: List[dict] = []
        prog = st.progress(0, text="Ranking setups…")
        for i, pair in enumerate(pairs, 1):
            prog.progress(i / len(pairs), text=f"Ranking {pair}…")
            if pair not in INSTRUMENTS:
                continue
            info = INSTRUMENTS[pair]
            best = None
            for d in ("LONG", "SHORT"):
                try:
                    r = _SetupRankerDataFeed.score(pair, info, d)
                except Exception:
                    continue
                if best is None or r["score"] > best["score"]:
                    best = r
            # Compare on percentage, not raw score — FX pairs score /11
            # (Currency Strength) while commodities score /10, matching the
            # same normalization Setup Ranker's Min score filter uses.
            if best and best["pct"] >= min_score / 10 * 100:
                out.append(best)
        prog.empty()
        out.sort(key=lambda x: -x["score"])
        return out

    def _analyse(self, signals) -> List[dict]:
        """Attach a fib analysis to each signal that yields a usable leg."""
        results = []
        lookback = int(st.session_state.fib_lookback)
        days = int(st.session_state.fib_days)
        for sig in signals:
            pair = sig["pair"]
            info = INSTRUMENTS[pair]
            df = _fetch_15m(info["ticker"], days)
            if df.empty or len(df) < 20:
                continue
            fib = fib_analysis(df, sig["direction"], lookback)
            if fib is None:
                continue
            results.append({"sig": sig, "fib": fib, "df": df})
        return results

    # ── DB persistence ────────────────────────────────────────────────────────
    @staticmethod
    def _persist_signals(results) -> None:
        """Persist live fib entries (ENTRY_FIRED / IN_ZONE) to trade_setups via the
        shared signal store. Deduped per pair/direction/price, source='fib_entry'."""
        signals = []
        for r in results:
            fib = r["fib"]
            if fib["status"] not in ("ENTRY_FIRED", "IN_ZONE"):
                continue
            sig = r["sig"]
            inst = INSTRUMENTS.get(sig["pair"])
            pip_size = inst.pip_size if inst else None
            sl_pips = (round(abs(fib["entry"] - fib["sl"]) / pip_size, 1)
                       if pip_size else None)
            signals.append({
                "pair": sig["pair"],
                "bias": sig["direction"],
                "entry": fib["entry"],
                "stop_loss": fib["sl"],
                "stop_loss_pips": sl_pips,
                "take_profit_1": fib["tp1"],
                "take_profit_2": fib["tp2"],
                "risk_reward_1": fib["rr1"],
                "risk_reward_2": fib["rr2"],
                "strength_score": sig["score"],
                "conviction": fib["status"],
                "thesis": (f"15M Fib Entry — golden-zone {sig['direction']}, "
                           f"setup {sig['score']}/{sig.get('max_score', 7)} ({sig['grade']})"),
            })
        persist_signals("fib_entry", signals)

    # ── email ────────────────────────────────────────────────────────────────
    def _maybe_email(self, results) -> None:
        """Email entries that fired on the latest ~3 candles and weren't sent yet."""
        fresh = []
        for r in results:
            fib = r["fib"]
            if not fib["entries"]:
                continue
            recent = r["df"].index[-3:]
            last_entry = fib["entries"][-1]
            if last_entry["time"] in recent:
                fresh.append(r)
        if not fresh:
            return

        cache = alert_service.NotifyCache("fib_entry")
        seen = cache.load()
        to_send = []
        keys = []
        for r in fresh:
            e = r["fib"]["entries"][-1]
            k = f"{r['sig']['pair']}|{r['sig']['direction']}|{e['time']}|{e['price']:.5f}"
            if k not in seen:
                to_send.append(r)
                keys.append(k)
        if not to_send:
            return

        n = int(st.session_state.get("fib_trades_per_signal", 2))
        html, plain = self._build_email(to_send, n)
        pairs_txt = ", ".join(r["sig"]["pair"] for r in to_send[:3])
        subject = (f"⚡ {len(to_send)} fib entr{'y' if len(to_send) == 1 else 'ies'} — "
                   f"{pairs_txt}{'…' if len(to_send) > 3 else ''}")
        ok, msg = alert_service.send_email(subject, html, plain, source="fib_entry")
        if ok:
            cache.filter_new(keys)
            st.toast(f"📧 Emailed {len(to_send)} fib entr"
                     f"{'y' if len(to_send) == 1 else 'ies'} to "
                     f"{alert_service.email_recipient()}", icon="📧")
        else:
            st.toast(f"⚠ Alert email failed: {msg}", icon="⚠️")

    @staticmethod
    def _build_email(results, n_trades):
        bal = float(st.session_state.account_bal)
        risk_pct = float(st.session_state.risk_pct)
        n = max(1, int(n_trades))
        rows, plain_lines = "", []
        total_win = 0.0
        for r in results:
            sig, fib = r["sig"], r["fib"]
            pair, d = sig["pair"], sig["direction"]
            inst = INSTRUMENTS.get(pair)
            pip_size = inst.pip_size if inst else None
            pip_value = inst.pip if inst else None
            risk_dist = abs(fib["entry"] - fib["sl"])
            sl_pips = round(risk_dist / pip_size, 1) if pip_size else None
            mb = money_breakdown(sl_pips, pip_value, bal, risk_pct, fib["rr1"])
            win_c = mb["win"] * n if mb["win"] is not None else None
            lot_c = mb["lot"] * n if mb["lot"] is not None else None
            if win_c is not None:
                total_win += win_c
            d_color = "#26a69a" if d == "LONG" else "#ef5350"
            arrow = "📈 LONG" if d == "LONG" else "📉 SHORT"
            lot_txt = f"{lot_c:.2f}" if lot_c is not None else "—"
            win_txt = f"+${win_c:,.0f}" if win_c is not None else "—"
            rows += f"""
            <tr style="border-bottom:1px solid #2d3148;">
              <td style="padding:9px 12px;font-weight:700;color:#e0e0e0;">{pair}</td>
              <td style="padding:9px 12px;color:{d_color};font-weight:700;">{arrow}</td>
              <td style="padding:9px 12px;color:#ffa726;font-weight:700;">{sig['score']}/{sig.get('max_score', 7)}
                  <span style="color:#8b8fa8;font-size:11px;">({sig['grade']})</span></td>
              <td style="padding:9px 12px;color:#e0e0e0;">{fmt_price(fib['entry'])}</td>
              <td style="padding:9px 12px;color:#ef5350;">{fmt_price(fib['sl'])}
                  <span style="color:#8b8fa8;font-size:11px;">{sl_pips}p</span></td>
              <td style="padding:9px 12px;color:#26a69a;">{fmt_price(fib['tp1'])}
                  <span style="color:#8b8fa8;font-size:11px;">{fib['rr1']:.1f}R</span></td>
              <td style="padding:9px 12px;color:#8b8fa8;">{n}× {lot_txt} lot</td>
              <td style="padding:9px 12px;color:#26a69a;font-weight:700;">{win_txt}</td>
            </tr>"""
            plain_lines.append(
                f"{pair} {d} | Setup {sig['score']}/{sig.get('max_score', 7)} ({sig['grade']}) | "
                f"Entry {fmt_price(fib['entry'])} (golden zone)  "
                f"SL {fmt_price(fib['sl'])} ({sl_pips}p)  "
                f"TP {fmt_price(fib['tp1'])} ({fib['rr1']:.1f}R)  | "
                f"{n}× {lot_txt} lot · win {win_txt}")

        grow_pct = (total_win / bal * 100) if bal else 0.0
        summary = (
            f"<div style='margin-top:18px;padding:14px 16px;background:#0d2f1a;"
            f"border:1px solid #00a32a;border-radius:8px;color:#e0e0e0;'>"
            f"💰 <b>If all {len(results)} entr"
            f"{'y' if len(results) == 1 else 'ies'} hit TP1 ({n} trade"
            f"{'s' if n != 1 else ''} each):</b> "
            f"<b style='color:#00ff41;'>+${total_win:,.0f} (+{grow_pct:.2f}%)</b> "
            f"→ <b style='color:#00ff41;'>${bal + total_win:,.0f}</b></div>")
        html = f"""<!DOCTYPE html><html><body style="background:#0e1117;
          font-family:'Courier New',monospace;color:#c0c0c0;margin:0;padding:20px;">
          <div style="max-width:820px;margin:0 auto;">
            <h2 style="color:#00ff41;border-bottom:1px solid #2d3148;padding-bottom:10px;">
              ⚡ 15M Fib Entry — {len(results)} New Entr{'y' if len(results) == 1 else 'ies'}
            </h2>
            <p style="color:#8b8fa8;font-size:13px;">
              Detected {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ·
              Setup Ranker–filtered · entry in the 0.382–0.618 golden zone ·
              SL at 0.786 · TP1 at prior swing · {n} trades per signal
            </p>
            <table style="width:100%;border-collapse:collapse;background:#1a1d27;border-radius:8px;overflow:hidden;">
              <thead><tr style="background:#000000;">
                {''.join(f'<th style="padding:9px 12px;text-align:left;color:#8b8fa8;font-size:11px;letter-spacing:1px;">{h}</th>' for h in ('PAIR', 'DIR', 'SETUP', 'ENTRY', 'STOP', 'TARGET', 'SIZING', 'WIN'))}
              </tr></thead><tbody>{rows}</tbody>
            </table>{summary}
            <p style="color:#8b8fa8;font-size:11px;margin-top:22px;border-top:1px solid #2d3148;padding-top:12px;">
              Fibonacci golden-zone entry on the 15M chart, filtered by Setup Ranker
              signals. Confirm the live candle before executing.
            </p>
          </div></body></html>"""
        plain = (f"15M FIB ENTRY — {len(results)} new entr"
                 f"{'y' if len(results) == 1 else 'ies'} "
                 f"({datetime.now().strftime('%Y-%m-%d %H:%M')})\n"
                 f"Sizing {risk_pct:.2f}% · {n} trades/signal\n\n"
                 + "\n".join(plain_lines)
                 + f"\n\nIF ALL HIT TP1: +${total_win:,.0f} (+{grow_pct:.2f}%) "
                   f"-> ${bal + total_win:,.0f}")
        return html, plain

    # ── rendering ────────────────────────────────────────────────────────────
    @staticmethod
    def _render_signals(results) -> None:
        if not results:
            st.info("◇ No Setup Ranker signals with a valid 15M fib leg. "
                    "Lower the min score or widen the instrument set.")
            return
        bal = float(st.session_state.account_bal)
        risk_pct = float(st.session_state.risk_pct)
        for rank, r in enumerate(results, 1):
            sig, fib = r["sig"], r["fib"]
            d = sig["direction"]
            d_color = T.GREEN if d == "LONG" else T.RED
            d_arrow = "▲" if d == "LONG" else "▼"
            s_label, s_color, s_desc = STATUS_CFG[fib["status"]]
            inst = INSTRUMENTS.get(sig["pair"])
            pip_size = inst.pip_size if inst else None
            pip_value = inst.pip if inst else None
            risk_dist = abs(fib["entry"] - fib["sl"])
            sl_pips = round(risk_dist / pip_size, 1) if pip_size else None
            mb = money_breakdown(sl_pips, pip_value, bal, risk_pct, fib["rr1"])
            lot_txt = f"{mb['lot']:.2f}" if mb["lot"] is not None else "—"
            win_txt = f"+${mb['win']:,.0f}" if mb["win"] is not None else "—"
            last_entry = fib["entries"][-1]["time"].strftime("%m-%d %H:%M") \
                if fib["entries"] else "—"

            body = (
                f'<div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px;">'
                f'<div><span style="color:{T.GREY};font-size:10px;">#{rank:02d}</span> '
                f'<span style="color:{T.WHITE};font-size:14px;font-weight:700;">{sig["pair"]}</span> '
                f'<span style="color:{d_color};font-weight:700;font-size:11px;">{d_arrow} {d}</span></div>'
                f'<div style="text-align:right;">'
                f'<span style="color:{T.YELLOW};font-weight:700;font-family:\'JetBrains Mono\',monospace;">'
                f'SETUP {sig["score"]}/{sig.get("max_score", 7)} ({sig["grade"]})</span><br>'
                f'<span style="color:{s_color};font-size:11px;font-weight:700;">{s_label}</span></div>'
                f'</div>'
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
                f'display:flex;flex-wrap:wrap;gap:14px;color:{T.GREY};margin-top:8px;">'
                f'<span>ENTRY <span style="color:{T.WHITE};">{fmt_price(fib["entry"])}</span></span>'
                f'<span>SL <span style="color:{T.RED};">{fmt_price(fib["sl"])}</span> '
                f'<span style="color:{T.GREY};">({sl_pips}p · 0.786)</span></span>'
                f'<span>TP1 <span style="color:{T.GREEN};">{fmt_price(fib["tp1"])}</span> '
                f'<span style="color:{T.GREY};">({fib["rr1"]:.1f}R)</span></span>'
                f'<span>TP2 <span style="color:{T.GREEN};">{fmt_price(fib["tp2"])}</span> '
                f'<span style="color:{T.GREY};">({fib["rr2"]:.1f}R)</span></span>'
                f'</div>'
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
                f'display:flex;flex-wrap:wrap;gap:14px;color:{T.GREY};margin-top:4px;">'
                f'<span>ZONE <span style="color:{T.CYAN};">{fmt_price(fib["zone_lo"])}–{fmt_price(fib["zone_hi"])}</span></span>'
                f'<span>ENTRIES <span style="color:{T.YELLOW};">{len(fib["entries"])}</span></span>'
                f'<span>LAST <span style="color:{T.WHITE};">{last_entry}</span></span>'
                f'<span>LOT <span style="color:{T.CYAN};">{lot_txt}</span></span>'
                f'<span>WIN <span style="color:{T.GREEN};">{win_txt}</span></span>'
                f'</div>'
                f'<div style="color:{T.GREY};font-size:10px;margin-top:6px;">{s_desc}</div>'
            )
            st.markdown(
                f'<div style="background:{T.BG_PANEL};border:1px solid {T.BORDER};'
                f'border-left:3px solid {d_color};padding:12px 14px;margin-bottom:10px;">'
                f'{body}</div>',
                unsafe_allow_html=True)

    def _render_detail(self) -> None:
        pair = st.session_state.fib_detail
        info = INSTRUMENTS[pair]
        df = _fetch_15m(info["ticker"], int(st.session_state.fib_days))
        if df.empty or len(df) < 20:
            st.error(f"⚠ Could not load 15M data for {pair}.")
            return
        # Bias for the detail chart: prefer the live Setup Ranker best direction.
        bias, score, max_score, grade = "LONG", None, 7, None
        best = None
        for d in ("LONG", "SHORT"):
            try:
                r = _SetupRankerDataFeed.score(pair, info, d)
            except Exception:
                continue
            if best is None or r["score"] > best["score"]:
                best = r
        if best:
            bias, score, max_score, grade = (
                best["direction"], best["score"], best["max_score"], best["grade"])
        fib = fib_analysis(df, bias, int(st.session_state.fib_lookback))
        if fib is None:
            st.warning(f"No clean impulse leg for {pair} in the lookback window.")
            return

        s_label, s_color, _ = STATUS_CFG[fib["status"]]
        setup_txt = f"Setup {score}/{max_score} ({grade})" if score is not None else "Setup —"
        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:13px;'
            f'color:{T.WHITE};margin-bottom:8px;">{pair} · '
            f'<span style="color:{T.GREEN if bias == "LONG" else T.RED};">{bias}</span> · '
            f'<span style="color:{T.YELLOW};">{setup_txt}</span> · '
            f'<span style="color:{s_color};">{s_label}</span> · '
            f'<span style="color:{T.GREY};">{len(fib["entries"])} entries</span></div>',
            unsafe_allow_html=True)

        fig = self._build_chart(df, pair, bias, fib, int(st.session_state.fib_show))
        st.plotly_chart(fig, width="stretch",
                        config=dict(displayModeBar=False))

        if fib["entries"]:
            rows = [{
                "Time": e["time"].strftime("%m-%d %H:%M"),
                "Dir": bias,
                "Entry": fmt_price(e["price"]),
                "Candle Low": fmt_price(e["low"]),
                "Candle High": fmt_price(e["high"]),
            } for e in reversed(fib["entries"])]
            st.markdown(
                f'<div style="color:{T.GREY};font-size:11px;margin:10px 0 4px;">'
                f'All fib golden-zone entries in the loaded window</div>',
                unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        else:
            st.info("No confirmed golden-zone entries yet for this leg.")

    @staticmethod
    def _build_chart(df, pair, bias, fib, show_n) -> go.Figure:
        plot = df.tail(show_n).copy()
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.8, 0.2], vertical_spacing=0.03)
        fig.add_trace(go.Candlestick(
            x=plot.index, open=plot["Open"], high=plot["High"],
            low=plot["Low"], close=plot["Close"],
            **CANDLE_STYLE,
            name="15M", showlegend=False), row=1, col=1)

        # Golden zone shading.
        fig.add_hrect(y0=fib["zone_lo"], y1=fib["zone_hi"],
                      fillcolor="rgba(0,212,255,0.10)", line_width=0, row=1, col=1,
                      annotation_text="GOLDEN ZONE 0.382–0.618",
                      annotation_font=dict(size=9, color=T.CYAN),
                      annotation_position="top left")

        # Fibonacci level lines.
        fib_colors = {0.0: T.GREEN, 0.236: T.GREY, 0.382: T.CYAN, 0.5: T.YELLOW,
                      0.618: T.CYAN, 0.786: T.RED, 1.0: T.GREY}
        for frac, price in fib["levels"].items():
            fig.add_hline(y=price, line_color=fib_colors.get(frac, T.GREY),
                          line_dash="dot", line_width=1, row=1, col=1,
                          annotation_text=f"{frac * 100:.1f}%",
                          annotation_font=dict(size=8, color=fib_colors.get(frac, T.GREY)),
                          annotation_position="right")

        # SL / TP lines.
        for y, c, lbl in [(fib["sl"], T.RED, "SL 0.786"),
                          (fib["tp1"], T.GREEN, "TP1"),
                          (fib["tp2"], T.GREEN, "TP2 1.618")]:
            fig.add_hline(y=y, line_color=c, line_dash="dash", line_width=1.2,
                          row=1, col=1, annotation_text=lbl,
                          annotation_font=dict(size=9, color=c),
                          annotation_position="left")

        # All entries.
        ent = [e for e in fib["entries"] if e["time"] in plot.index]
        if ent:
            xs = [e["time"] for e in ent]
            if bias == "LONG":
                ys = [e["low"] - (plot["High"].max() - plot["Low"].min()) * 0.01 for e in ent]
                sym, col, pos = "triangle-up", T.GREEN, "bottom center"
            else:
                ys = [e["high"] + (plot["High"].max() - plot["Low"].min()) * 0.01 for e in ent]
                sym, col, pos = "triangle-down", T.RED, "top center"
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers+text",
                marker=dict(symbol=sym, size=14, color=col,
                            line=dict(color="#000000", width=1)),
                text=["⚡"] * len(ent), textposition=pos,
                name=f"{bias} entry", showlegend=True), row=1, col=1)

        # Impulse leg line.
        leg = fib["leg"]
        fig.add_trace(go.Scatter(
            x=[leg["start_idx"], leg["end_idx"]],
            y=[leg["low"] if bias == "LONG" else leg["high"],
               leg["high"] if bias == "LONG" else leg["low"]],
            mode="lines", line=dict(color=T.YELLOW, width=1.5, dash="dash"),
            name="Impulse leg", showlegend=True), row=1, col=1)

        # Volume.
        vol_colors = [T.GREEN if c >= o else T.RED
                      for c, o in zip(plot["Close"], plot["Open"])]
        fig.add_trace(go.Bar(x=plot.index, y=plot["Volume"],
                             marker_color=vol_colors, opacity=0.4,
                             name="Volume", showlegend=False), row=2, col=1)

        fig.update_layout(
            height=620, paper_bgcolor=T.BG, plot_bgcolor=T.BG_PANEL,
            font=dict(family=T.FONT_MONO, size=10, color=T.GREY),
            xaxis_rangeslider_visible=False,
            legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor=T.BORDER,
                        borderwidth=1, font=dict(size=10), orientation="h",
                        yanchor="bottom", y=1.01, xanchor="left", x=0),
            margin=dict(l=10, r=60, t=10, b=10),
            title=dict(text=f"<b>{pair}</b> — 15M Fibonacci Entries · {bias}",
                       font=dict(color=T.WHITE, size=14), x=0.01))
        fig.update_yaxes(gridcolor=T.BORDER, zeroline=False)
        fig.update_xaxes(gridcolor=T.BORDER, showgrid=False)
        return fig

    @staticmethod
    def _render_reference() -> None:
        st.markdown(f"""
        <div style="background:{T.BG_PANEL};border:1px solid {T.BORDER};
                    border-left:3px solid {T.GREEN};padding:16px 20px;
                    font-family:'JetBrains Mono',monospace;font-size:13px;
                    color:{T.GREY};line-height:1.8;">
        <b style="color:{T.GREEN};">How this page works</b><br><br>
        <b style="color:{T.WHITE};">1 · Setup Ranker filter</b><br>
        Every selected pair is scored using the exact same checklist as the
        Setup Ranker (10 technical criteria, +1 Currency Strength for FX
        pairs). Only pairs at/above the min score are treated as signals —
        and the higher-scoring side sets the direction.<br><br>
        <b style="color:{T.WHITE};">2 · Impulse leg</b><br>
        On the 15M chart we find the most recent impulse leg in the signal's
        direction (swing low→high for LONG, swing high→low for SHORT) within the
        lookback window.<br><br>
        <b style="color:{T.WHITE};">3 · Golden-zone entries</b><br>
        Fibonacci retracements are drawn on the leg. Every candle that pulls back
        into the <span style="color:{T.CYAN};">0.382–0.618 golden zone</span> and
        closes in the bias direction is flagged as an entry and marked on the
        chart.<br><br>
        <b style="color:{T.WHITE};">4 · Trade levels</b><br>
        SL sits beyond the <span style="color:{T.RED};">0.786</span> invalidation
        level; TP1 targets the prior swing (0.0), TP2 the 1.618 extension.<br><br>
        <b style="color:{T.WHITE};">5 · Email alerts</b><br>
        Turn on email alerts to be notified whenever a fresh golden-zone entry
        fires on a Setup Ranker signal. Each pair/direction/entry mails once.<br><br>
        <span style="color:{T.GREY};font-size:11px;">For educational purposes
        only · entry in golden zone · confirm the live candle before executing.</span>
        </div>
        """, unsafe_allow_html=True)
