"""Setup Ranker page — Bloomberg-terminal version.

Logic for `score_pair()`, `trade_levels()`, and `fmt_price()` is unchanged
from the original `pages/setup-ranker.py`; only presentation is refactored.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd
import plotly.graph_objects as go
import pytz
import streamlit as st
import yfinance as yf

from src.core.signals import score_setup
from src.instruments import INSTRUMENTS, TYPICAL_SPREADS
from src.pages_lib.base import BloombergPage, PageContext
from src.services import RiskService, alert_service, account_state
from src.services.parallel_fetch import run_parallel
from src.services.signal_store import persist_signals
from src.ui.components import (
    CommandBar, MetricCell, Panel, ProgressBar, render_metric_row,
)
from src.ui.theme import BloombergTheme as T


# ── Pure helpers (unchanged logic) ────────────────────────────────────────

def fmt_price(p) -> str:
    if p is None:
        return "—"
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "—"
    return f"{p:.5f}" if abs(p) < 100 else f"{p:.3f}"


def trade_levels(entry, sl_pips, pip_size, direction, rr) -> dict:
    try:
        entry = float(entry); sl_pips = float(sl_pips)
        pip_size = float(pip_size); rr = float(rr)
    except (TypeError, ValueError):
        return {"sl_price": None, "tp_price": None, "tp_pips": None}
    if sl_pips <= 0 or pip_size <= 0:
        return {"sl_price": None, "tp_price": None, "tp_pips": None}
    sl_dist = sl_pips * pip_size
    tp_dist = sl_dist * rr
    if direction == "LONG":
        sl_price, tp_price = entry - sl_dist, entry + tp_dist
    else:
        sl_price, tp_price = entry + sl_dist, entry - tp_dist
    return {
        "sl_price": sl_price, "tp_price": tp_price,
        "tp_pips": round(sl_pips * rr, 1),
    }


def money_breakdown(sl_pips, pip_value, account_balance, risk_pct, rr) -> dict:
    """Lot size, $ risked (if SL hit) and $ won (if TP hit).

    Uses the project risk model (same as the checklist's RiskService):
    lot = (balance · risk%) / sl_pips / pip_value. Because TP = R:R × SL,
    the profit at target is risk_amount × R:R.
    """
    try:
        sl_pips = float(sl_pips)
        pip_value = float(pip_value)
        rr = float(rr)
        account_balance = float(account_balance)
        risk_pct = float(risk_pct)
    except (TypeError, ValueError):
        return {"lot": None, "risk_amt": None, "win": None}
    if sl_pips <= 0 or pip_value <= 0:
        return {"lot": None, "risk_amt": None, "win": None}
    tp_pips = sl_pips * rr
    rb = RiskService.compute(account_balance, risk_pct, pip_value,
                             sl_pips, tp_pips, tp_pips)
    return {"lot": rb.lot_size, "risk_amt": rb.risk_amount,
            "win": rb.risk_amount * rr}


# ── Data fetchers (preserved from original) ───────────────────────────────

class _SetupRankerDataFeed:
    """Encapsulates the 3 timeframe pulls so the page logic is testable."""

    @staticmethod
    @st.cache_data(ttl=300, show_spinner=False)
    def daily(ticker: str, days: int = 300) -> pd.DataFrame:
        from src.db.market_cache import cached_ohlc
        try:
            df = cached_ohlc(ticker, period=f"{days}d", interval="1d", ttl=300)
            if df.empty:
                return pd.DataFrame()
            return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        except Exception:
            return pd.DataFrame()

    @staticmethod
    @st.cache_data(ttl=300, show_spinner=False)
    def four_hour(ticker: str) -> pd.DataFrame:
        from src.db.market_cache import cached_ohlc
        try:
            end = datetime.now(pytz.utc)
            start = end - timedelta(days=90)
            df = cached_ohlc(ticker, start=start, end=end, interval="1h", ttl=300)
            if df.empty:
                return pd.DataFrame()
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            return df.resample("4h").agg({
                "Open": "first", "High": "max", "Low": "min",
                "Close": "last", "Volume": "sum",
            }).dropna()
        except Exception:
            return pd.DataFrame()

    @staticmethod
    @st.cache_data(ttl=300, show_spinner=False)
    def weekly(ticker: str) -> pd.DataFrame:
        from src.db.market_cache import cached_ohlc
        try:
            df = cached_ohlc(ticker, period="2y", interval="1d", ttl=300)
            if df.empty:
                return pd.DataFrame()
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            return df.resample("W").agg({
                "Open": "first", "High": "max", "Low": "min",
                "Close": "last", "Volume": "sum",
            }).dropna()
        except Exception:
            return pd.DataFrame()

    @classmethod
    def score(cls, pair: str, info: dict, direction: str) -> dict:
        ticker = info["ticker"]
        pip_size = info["pip_size"]
        df_w = cls.weekly(ticker)
        df_d = cls.daily(ticker)
        df_4h = cls.four_hour(ticker)
        spread = TYPICAL_SPREADS.get(pair, 0.0)
        result = score_setup(df_w, df_d, df_4h, direction, pip_size, spread)
        result["pair"] = pair
        return result


# ── Page implementation ───────────────────────────────────────────────────

class SetupRankerPage(BloombergPage):
    """Bloomberg-styled multi-pair scoring scanner."""

    def configure(self) -> PageContext:
        # Scan both directions by default so every pair is evaluated long AND short.
        st.session_state.setdefault("sr_direction", "Both")
        st.session_state.setdefault("sr_min_score", 5)
        st.session_state.setdefault("sr_rr_ratio", 2.0)
        # Load the full instrument universe by default.
        st.session_state.setdefault("sr_pairs", list(INSTRUMENTS.keys()))
        # Shared with the checklist so account settings carry across pages.
        # Default to the live balance from the Trade Journal (MT4 statement) when
        # one has been recorded, otherwise the historical $10k default.
        st.session_state.setdefault("account_bal", account_state.get_balance(10000.0))
        st.session_state.setdefault("risk_pct", 1.0)
        # Use the Trade Journal's live balance by default when it's available.
        st.session_state.setdefault("sr_use_live_bal", bool(account_state.get()))
        # Email alerts (opt-in — toggle it on in the sidebar to activate).
        # Threshold default 9: only the top tier (9-10/10, not all of Grade A
        # 8+) is worth an interruption; 8s still auto-save to the journal DB
        # via _persist_signals regardless of this toggle.
        st.session_state.setdefault("sr_email_on", False)
        st.session_state.setdefault("sr_email_min", 9)
        # User's house rule: every signal is taken as 2 trades.
        st.session_state.setdefault("sr_trades_per_signal", 2)
        return PageContext(code="RANK", title="Setup Ranker", icon="🎰")

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

        st.markdown(
            '<div style="color:#00ff41;font-weight:700;letter-spacing:0.15em;'
            'text-transform:uppercase;font-size:11px;">SCAN PARAMS</div>',
            unsafe_allow_html=True,
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("All", width="stretch"):
                st.session_state.sr_pairs = INSTRUMENTS.keys()
        with col2:
            if st.button("Majors", width="stretch"):
                st.session_state.sr_pairs = [
                    "EUR/USD", "GBP/USD", "AUD/USD", "NZD/USD",
                    "USD/JPY", "USD/CHF", "USD/CAD",
                ]
        with col3:
            if st.button("Metals", width="stretch"):
                st.session_state.sr_pairs = ["XAU/USD", "XAG/USD", "XPT/USD"]
        st.session_state.sr_pairs = st.multiselect(
            "Instruments", INSTRUMENTS.keys(),
            default=[p for p in st.session_state.sr_pairs if p in INSTRUMENTS],
        )
        st.session_state.sr_direction = st.radio(
            "Direction", ["LONG", "SHORT", "Both"], horizontal=True,
            index=["LONG", "SHORT", "Both"].index(st.session_state.sr_direction),
        )
        st.session_state.sr_min_score = st.slider(
            "Min score", 0, 10, st.session_state.sr_min_score,
        )
        st.session_state.sr_rr_ratio = st.select_slider(
            "TP R:R", options=[1.0, 1.5, 2.0, 2.5, 3.0],
            value=st.session_state.sr_rr_ratio,
        )
        # ── Account balance: live from Trade Journal, or manual ───────
        acct = account_state.get()
        live_bal = account_state.get_balance(0.0)
        has_live = bool(acct) and live_bal > 0
        use_live = st.checkbox(
            "🔗 Use live balance from Trade Journal",
            value=bool(st.session_state.get("sr_use_live_bal", has_live)) and has_live,
            disabled=not has_live,
            help="Reads the balance imported from your MT4 statement so sizing reflects "
                 "your real account. Uncheck to enter a balance manually.",
        )
        st.session_state.sr_use_live_bal = use_live
        if use_live and has_live:
            st.session_state.account_bal = live_bal
            st.markdown(
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;">'
                f'BAL <span style="color:#00ff41;font-weight:700;">${live_bal:,.2f}</span> '
                f'<span style="color:#9a9a9a;font-size:10px;">live · '
                f'{acct.get("source", "")} · {acct.get("updated_at", "")[-8:]}</span></div>',
                unsafe_allow_html=True,
            )
        else:
            if not has_live:
                st.caption("No live balance yet — import an MT4 statement in the Trade Journal.")
            st.session_state.account_bal = st.number_input(
                "Account balance ($)", min_value=0.0, step=500.0,
                value=float(st.session_state.account_bal),
            )
        st.session_state.risk_pct = st.slider(
            "Risk % / trade", 0.25, 5.0,
            float(st.session_state.risk_pct), 0.25,
        )
        st.divider()
        if st.button("◆ RESCAN ALL PAIRS", width="stretch", type="primary"):
            st.cache_data.clear()
            st.rerun()
        st.caption(f"◷ {datetime.now().strftime('%H:%M')} · TTL 5min")

        # ── Email alerts ──────────────────────────────────────────────
        st.divider()
        st.markdown(
            '<div style="color:#00ff41;font-weight:700;letter-spacing:0.15em;'
            'text-transform:uppercase;font-size:11px;">EMAIL ALERTS</div>',
            unsafe_allow_html=True,
        )
        st.session_state.sr_email_on = st.toggle(
            "📧 Email new high-score setups",
            value=st.session_state.sr_email_on,
            help="On each scan, email any newly-appearing setup scoring at or above "
                 "the threshold. Each unique pair/direction/price fires once.",
        )
        st.session_state.sr_email_min = st.slider(
            "Alert when score ≥", 6, 10, int(st.session_state.sr_email_min),
        )
        st.session_state.sr_trades_per_signal = st.number_input(
            "Trades per signal", min_value=1, max_value=10,
            value=int(st.session_state.sr_trades_per_signal), step=1,
            help="Position sizing and projected account growth in the alert are "
                 "multiplied by this. Defaults to 2 (your house rule).",
        )
        if alert_service.email_configured():
            st.caption(f"✅ Alerts → {alert_service.email_recipient()}")
        else:
            st.caption("⚠ Set [gmail] in secrets.toml (or GMAIL_* env vars) to enable.")
        c_test, c_reset = st.columns(2)
        with c_test:
            if st.button("✉ Test", width="stretch"):
                ok, msg = alert_service.send_email(
                    "Setup Ranker — test alert",
                    "<p style='font-family:monospace;color:#0a0;'>✅ Setup Ranker email alerts "
                    "are wired up correctly.</p>",
                    "Setup Ranker email alerts are wired up correctly.",
                )
                (st.success if ok else st.error)(msg)
        with c_reset:
            if st.button("↺ Reset", width="stretch",
                         help="Clear the 'already-alerted' memory so current setups can re-fire."):
                alert_service.NotifyCache("setup_ranker").reset()
                st.toast("Alert memory cleared.")

    def body(self, ctx: PageContext) -> None:
        direction = st.session_state.sr_direction
        min_score = st.session_state.sr_min_score
        rr_ratio = st.session_state.sr_rr_ratio
        account_bal = st.session_state.account_bal
        risk_pct = st.session_state.risk_pct
        pairs = st.session_state.sr_pairs
        if not pairs:
            st.warning("⚠ Select at least one instrument in the sidebar.")
            return

        CommandBar(label="RANK", cells=[
            (f"DIR {direction}", ""),
            (f"PAIRS {len(pairs)}", ""),
            (f"MIN {min_score}/10", "cyan"),
            (f"TP {rr_ratio:.1f}R", "green"),
            (f"BAL ${account_bal:,.0f}", ""),
            (f"RISK {risk_pct:.2f}%", "cyan"),
            (datetime.now().strftime("%a %d %b %H:%M"), "amber"),
        ]).show()

        directions = ["LONG", "SHORT"] if direction == "Both" else [direction]

        # Auto-refreshing fragment: re-scans, re-persists, and re-checks email
        # alerts every 5 min on its own (matching the underlying data's cache
        # TTL) without a manual "RESCAN ALL PAIRS" click or full-page rerun —
        # same pattern as market_overview_lib's live trading-ideas fragment.
        # Persistence + email dedupe are keyed per pair/direction/price, so an
        # unattended tick that finds nothing new is a safe no-op.
        @st.fragment(run_every=300, parallel=True)
        def _scan_and_render():
            results = self._scan(pairs, directions, min_score)

            # Auto-save Grade-A setups to the journal DB (deduped, source-tagged).
            self._persist_signals(results, rr_ratio)

            # Fire email alerts for newly-appearing high-score setups (opt-in).
            if st.session_state.get("sr_email_on"):
                self._maybe_email_alerts(results, rr_ratio, account_bal, risk_pct)

            st.caption(f"◷ Auto-refreshes every 5 min · last scan "
                      f"{datetime.now().strftime('%H:%M:%S')}")

            # KPI strip
            grade_counts = {
                g: sum(1 for r in results if r["grade"] == g)
                for g in ("A", "B", "C", "D")
            }
            # Projected profit if every Grade-A setup hit TP (one risk-unit each).
            grade_a_win = 0.0
            for r in results:
                if r["grade"] != "A":
                    continue
                inst = INSTRUMENTS.get(r["pair"])
                mb = money_breakdown(r.get("sl_pips"), inst.pip if inst else None,
                                     account_bal, risk_pct, rr_ratio)
                if mb["win"] is not None:
                    grade_a_win += mb["win"]
            render_metric_row([
                MetricCell("Grade A (8–10)", str(grade_counts["A"]), "green"),
                MetricCell("Grade B (6–7)",  str(grade_counts["B"]), "cyan"),
                MetricCell("Grade C (4–5)",  str(grade_counts["C"]), "yellow"),
                MetricCell("Grade D (<4)",   str(grade_counts["D"]), "red"),
                MetricCell("Grade-A Win $",  f"+${grade_a_win:,.0f}", "green"),
            ])

            if not results:
                st.info(f"◇ No pairs scored ≥{min_score}. Lower threshold or change direction.")
                return

            st.markdown(
                f'<div style="color:#9a9a9a;font-family:\'JetBrains Mono\',monospace;'
                f'font-size:10px;margin:8px 0;">Entry ≈ current close · Stop from scorer\'s '
                f'structural SL · TP = stop × {rr_ratio:.1f}R · WIN/RISK $ at '
                f'{risk_pct:.2f}% of ${account_bal:,.0f}. Confirm on 15M before trading.</div>',
                unsafe_allow_html=True,
            )

            tab_cards, tab_table, tab_chart = st.tabs(["◆ RANKED", "▤ TABLE", "▦ CHART"])
            with tab_cards: self._render_cards(results, rr_ratio, account_bal, risk_pct)
            with tab_table: self._render_table(results, rr_ratio, account_bal, risk_pct)
            with tab_chart: self._render_chart(results)

        _scan_and_render()

    @staticmethod
    def _scan(pairs, directions, min_score) -> list:
        """Score every pair × direction, fanned out one worker per pair.

        Each pair's weekly/daily/4H fetch is I/O-bound (yfinance/Postgres), so
        parallelizing across pairs turns the scan's wall-clock time from a sum
        of ~N round-trips into a max over a few concurrent waves. Both
        directions for a pair run in the same worker (fetched data is shared,
        scoring itself is cheap CPU), preserving the original per-direction
        exception isolation.
        """
        results = []
        pairs_list = [(p, INSTRUMENTS[p]) for p in pairs if p in INSTRUMENTS]
        total = len(pairs_list)
        done = 0
        prog = st.progress(0, text="Scoring pairs…")

        def _score_pair(item):
            pair_name, info = item
            scored = []
            for d in directions:
                try:
                    scored.append(_SetupRankerDataFeed.score(pair_name, info, d))
                except Exception:
                    pass
            return scored

        def _on_progress(item):
            nonlocal done
            done += 1
            prog.progress(done / total, text=f"Scoring {item[0]}…")

        def _on_result(item, scored):
            _on_progress(item)
            for r in scored:
                if r["score"] >= min_score:
                    results.append(r)

        def _on_error(item, exc):
            _on_progress(item)

        run_parallel(_score_pair, pairs_list, on_result=_on_result, on_error=_on_error)
        prog.empty()
        results.sort(key=lambda x: -x["score"])
        return results

    # ── DB persistence ──────────────────────────────────────────────────────
    @staticmethod
    def _persist_signals(results, rr_ratio) -> None:
        """Persist Grade-A setups (score ≥ 8) to trade_setups via the shared
        signal store. Deduped per pair/direction/price, tagged source='setup_ranker',
        silent no-op without a DB."""
        signals = []
        for r in results:
            if r.get("score", 0) < 8:  # Grade A only — high-conviction
                continue
            inst = INSTRUMENTS.get(r["pair"])
            pip_size = inst.pip_size if inst else None
            lv = trade_levels(r["close"], r.get("sl_pips"), pip_size,
                              r["direction"], rr_ratio)
            passed = [k for k, v in r.get("scores", {}).items() if v]
            signals.append({
                "pair": r["pair"],
                "bias": r["direction"],
                "entry": r["close"],
                "stop_loss": lv["sl_price"],
                "stop_loss_pips": r.get("sl_pips"),
                "take_profit_1": lv["tp_price"],
                "strength_score": r.get("score"),
                "conviction": f"Grade {r.get('grade')}",
                "risk_reward_1": rr_ratio,
                "thesis": "Setup Ranker — " + ", ".join(passed),
            })
        persist_signals("setup_ranker", signals)

    # ── Email alerts ──────────────────────────────────────────────────────
    @classmethod
    def _maybe_email_alerts(cls, results, rr_ratio, account_bal, risk_pct) -> None:
        """Email setups scoring ≥ threshold that we haven't alerted on yet. The
        'seen' memory is only updated after a successful send, so a transient
        SMTP failure retries on the next scan instead of silently dropping."""
        threshold = int(st.session_state.get("sr_email_min", 9))
        qualifying = [r for r in results if r.get("score", 0) >= threshold]
        if not qualifying:
            return

        keys = [f"{r['pair']}|{r['direction']}|{float(r['close']):.4f}" for r in qualifying]
        cache = alert_service.NotifyCache("setup_ranker")
        seen = cache.load()
        fresh = [(r, k) for r, k in zip(qualifying, keys) if k not in seen]
        if not fresh:
            return

        setups = [r for r, _ in fresh]
        n_trades = int(st.session_state.get("sr_trades_per_signal", 2))
        html, plain = cls._build_alert_email(
            setups, rr_ratio, account_bal, risk_pct, threshold, n_trades)
        subject = (
            f"🎰 {len(setups)} new setup{'s' if len(setups) != 1 else ''} ≥{threshold}/10 — "
            f"{', '.join(r['pair'] for r in setups[:3])}{'…' if len(setups) > 3 else ''}"
        )
        ok, msg = alert_service.send_email(subject, html, plain)
        if ok:
            cache.filter_new([k for _, k in fresh])  # mark as alerted only now
            st.toast(f"📧 Emailed {len(setups)} new setup(s) to {alert_service.email_recipient()}",
                     icon="📧")
        else:
            st.toast(f"⚠ Alert email failed: {msg}", icon="⚠️")

    @staticmethod
    def _build_alert_email(setups, rr_ratio, account_bal, risk_pct, threshold,
                           trades_per_signal=2):
        """Render (html, plain) bodies for a batch of new setups.

        Sizing and projected account growth assume ``trades_per_signal`` trades
        per signal (the user takes 2 per signal). 'Growth' is the account gain if
        a signal's trades all hit TP; the footer totals it across all setups.
        """
        n = max(1, int(trades_per_signal))
        rows_html, plain_lines = "", []
        total_win = 0.0   # if every alerted signal hits TP (n trades each)
        total_risk = 0.0  # if every alerted signal hits SL (n trades each)
        for r in setups:
            inst = INSTRUMENTS.get(r["pair"])
            pip_size = inst.pip_size if inst else None
            pip_value = inst.pip if inst else None
            lv = trade_levels(r["close"], r.get("sl_pips"), pip_size, r["direction"], rr_ratio)
            mb = money_breakdown(r.get("sl_pips"), pip_value, account_bal, risk_pct, rr_ratio)

            # Combined figures for n trades taken on this one signal.
            win_c = mb["win"] * n if mb["win"] is not None else None
            risk_c = mb["risk_amt"] * n if mb["risk_amt"] is not None else None
            lot_c = mb["lot"] * n if mb["lot"] is not None else None
            if win_c is not None:
                total_win += win_c
            if risk_c is not None:
                total_risk += risk_c
            grow_pct = (win_c / account_bal * 100) if (win_c is not None and account_bal) else None

            d_color = "#26a69a" if r["direction"] == "LONG" else "#ef5350"
            arrow = "📈 LONG" if r["direction"] == "LONG" else "📉 SHORT"
            win_txt = f"+${win_c:,.0f}" if win_c is not None else "—"
            risk_txt = f"${risk_c:,.0f}" if risk_c is not None else "—"
            lot_txt = f"{lot_c:.2f}" if lot_c is not None else "—"
            grow_txt = (f"+${win_c:,.0f} <span style='color:#8b8fa8;'>(+{grow_pct:.2f}%)</span>"
                        if grow_pct is not None else "—")
            grow_plain = f"+${win_c:,.0f} (+{grow_pct:.2f}%)" if grow_pct is not None else "—"
            rows_html += f"""
            <tr style="border-bottom:1px solid #2d3148;">
              <td style="padding:9px 12px;font-weight:700;color:#e0e0e0;">{r['pair']}</td>
              <td style="padding:9px 12px;color:{d_color};font-weight:700;">{arrow}</td>
              <td style="padding:9px 12px;color:#ffa726;font-weight:700;">{r['score']}/10
                  <span style="color:#8b8fa8;font-size:11px;">({r['grade']})</span></td>
              <td style="padding:9px 12px;color:#e0e0e0;">{fmt_price(r['close'])}</td>
              <td style="padding:9px 12px;color:#ef5350;">{fmt_price(lv['sl_price'])}
                  <span style="color:#8b8fa8;font-size:11px;">{r.get('sl_pips','—')}p</span></td>
              <td style="padding:9px 12px;color:#26a69a;">{fmt_price(lv['tp_price'])}
                  <span style="color:#8b8fa8;font-size:11px;">{rr_ratio:.1f}R</span></td>
              <td style="padding:9px 12px;color:#8b8fa8;">{n}× {lot_txt} lot · {risk_txt} risk</td>
              <td style="padding:9px 12px;color:#26a69a;font-weight:700;">{grow_txt}</td>
            </tr>"""
            plain_lines.append(
                f"{r['pair']} {r['direction']}  |  Score {r['score']}/10 ({r['grade']})  |  "
                f"Entry {fmt_price(r['close'])}  SL {fmt_price(lv['sl_price'])}  "
                f"TP {fmt_price(lv['tp_price'])} ({rr_ratio:.1f}R)  |  "
                f"{n}× {lot_txt} lot · risk {risk_txt} · GROWTH {grow_plain}"
            )

        new_bal = account_bal + total_win
        total_grow_pct = (total_win / account_bal * 100) if account_bal else 0.0
        summary_html = (
            f"<div style='margin-top:18px;padding:14px 16px;background:#0d2f1a;"
            f"border:1px solid #00a32a;border-radius:8px;color:#e0e0e0;'>"
            f"💰 <b>If all {len(setups)} signal{'s' if len(setups) != 1 else ''} hit TP "
            f"({n} trade{'s' if n != 1 else ''} each):</b> "
            f"account grows <b style='color:#00ff41;'>+${total_win:,.0f} (+{total_grow_pct:.2f}%)</b> "
            f"→ <b style='color:#00ff41;'>${new_bal:,.0f}</b>"
            f"<br><span style='color:#8b8fa8;font-size:12px;'>Worst case if all hit SL: "
            f"−${total_risk:,.0f} → ${account_bal - total_risk:,.0f}</span></div>"
        )
        summary_plain = (
            f"\nIF ALL HIT TP ({n} trades/signal): +${total_win:,.0f} "
            f"(+{total_grow_pct:.2f}%) -> ${new_bal:,.0f}\n"
            f"Worst case all SL: -${total_risk:,.0f} -> ${account_bal - total_risk:,.0f}"
        )

        html = f"""<!DOCTYPE html><html><body style="background:#0e1117;
          font-family:'Courier New',monospace;color:#c0c0c0;margin:0;padding:20px;">
          <div style="max-width:820px;margin:0 auto;">
            <h2 style="color:#00ff41;border-bottom:1px solid #2d3148;padding-bottom:10px;">
              🎰 Setup Ranker — {len(setups)} New Setup{'s' if len(setups) != 1 else ''} ≥ {threshold}/10
            </h2>
            <p style="color:#8b8fa8;font-size:13px;">
              Detected {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ·
              sizing at {risk_pct:.2f}% of ${account_bal:,.0f} · {n} trades per signal
            </p>
            <table style="width:100%;border-collapse:collapse;background:#1a1d27;border-radius:8px;overflow:hidden;">
              <thead><tr style="background:#000000;">
                {''.join(f'<th style="padding:9px 12px;text-align:left;color:#8b8fa8;font-size:11px;letter-spacing:1px;">{h}</th>' for h in ('PAIR','DIR','SCORE','ENTRY','STOP','TARGET','SIZING','GROWTH'))}
              </tr></thead>
              <tbody>{rows_html}</tbody>
            </table>
            {summary_html}
            <p style="color:#8b8fa8;font-size:11px;margin-top:22px;border-top:1px solid #2d3148;padding-top:12px;">
              Entry ≈ current close · confirm on the 15M trigger before executing ·
              each unique pair/direction/price alerts once.
            </p>
          </div></body></html>"""
        plain = (f"SETUP RANKER — {len(setups)} new setup(s) ≥ {threshold}/10 "
                 f"({datetime.now().strftime('%Y-%m-%d %H:%M')})\n"
                 f"Sizing {risk_pct:.2f}% · {n} trades/signal\n\n"
                 + "\n".join(plain_lines) + "\n" + summary_plain)
        return html, plain

    @staticmethod
    def _render_cards(results, rr_ratio, account_bal, risk_pct) -> None:
        grade_color = {
            "A": ("●", T.GREEN, "GO", "go"),
            "B": ("◆", T.CYAN, "STRONG", "info"),
            "C": ("◐", T.YELLOW, "PARTIAL", "wait"),
            "D": ("○", T.GREY, "WEAK", "no"),
        }
        for rank, r in enumerate(results, 1):
            icon, color, label, _ = grade_color[r["grade"]]
            d_color = T.GREEN if r["direction"] == "LONG" else T.RED
            d_arrow = "▲" if r["direction"] == "LONG" else "▼"
            price_fmt = (f"{r['close']:.5f}" if r["close"] < 100
                         else f"{r['close']:.3f}")
            inst = INSTRUMENTS.get(r["pair"])
            pip_size = inst.pip_size if inst else None
            pip_value = inst.pip if inst else None
            lv = trade_levels(r["close"], r.get("sl_pips"),
                              pip_size, r["direction"], rr_ratio)
            sl_pips_txt = (f"{r['sl_pips']}"
                           if r.get("sl_pips") not in (None, "") else "—")
            tp_pips_txt = (f"{lv['tp_pips']:g}"
                           if lv["tp_pips"] is not None else "—")
            mb = money_breakdown(r.get("sl_pips"), pip_value,
                                 account_bal, risk_pct, rr_ratio)
            lot_txt = f"{mb['lot']:.2f}" if mb["lot"] is not None else "—"
            risk_txt = f"${mb['risk_amt']:,.0f}" if mb["risk_amt"] is not None else "—"
            win_txt = f"+${mb['win']:,.0f}" if mb["win"] is not None else "—"
            dots = "".join(
                f'<span style="display:inline-block;width:8px;height:8px;'
                f'background:{color if i < r["score"] else T.BORDER};'
                f'margin-right:2px;"></span>'
                for i in range(10)
            )
            pills = "".join(
                f'<span style="display:inline-block;border:1px solid '
                f'{T.GREEN if v else T.BORDER};color:{T.GREEN if v else T.GREY};'
                f'padding:1px 6px;font-size:9px;margin:1px;'
                f'font-family:\'JetBrains Mono\',monospace;letter-spacing:0.05em;">'
                f'{k}</span>'
                for k, v in r["scores"].items()
            )
            spread_warn = ""
            if r["spread_pct"] > 5:
                spread_warn = (
                    f'<span style="border:1px solid {T.RED};color:{T.RED};'
                    f'padding:1px 6px;font-size:9px;margin-left:6px;'
                    f'font-family:\'JetBrains Mono\',monospace;">⚠ SPREAD '
                    f'{r["spread_pct"]:.1f}%</span>'
                )

            body = (
                f'<div style="display:flex;justify-content:space-between;'
                f'flex-wrap:wrap;gap:6px;align-items:flex-start;">'
                f'<div>'
                f'<span style="color:{T.GREY};font-size:10px;">#{rank:02d}</span> '
                f'<span style="color:{T.WHITE};font-size:14px;font-weight:700;'
                f'letter-spacing:0.05em;">{r["pair"]}</span> '
                f'<span style="color:{d_color};font-weight:700;font-size:11px;">'
                f'{d_arrow} {r["direction"]}</span>{spread_warn}'
                f'</div>'
                f'<div style="text-align:right;">'
                f'<span style="color:{color};font-size:14px;font-weight:700;'
                f'font-family:\'JetBrains Mono\',monospace;">{r["score"]}/10</span> '
                f'<span style="color:{color};font-size:10px;letter-spacing:0.1em;">'
                f'{icon} {label}</span>'
                f'</div></div>'
                f'<div style="margin:8px 0 6px 0;">{dots}</div>'
                f'<div style="font-family:\'JetBrains Mono\',monospace;'
                f'font-size:11px;display:flex;flex-wrap:wrap;gap:14px;color:{T.GREY};">'
                f'<span>PX <span style="color:{T.WHITE};">{price_fmt}</span></span>'
                f'<span>SL <span style="color:{T.RED};">{fmt_price(lv["sl_price"])}</span> '
                f'<span style="color:{T.GREY};">({sl_pips_txt}p)</span></span>'
                f'<span>TP <span style="color:{T.GREEN};">{fmt_price(lv["tp_price"])}</span> '
                f'<span style="color:{T.GREY};">({tp_pips_txt}p · {rr_ratio:.1f}R)</span></span>'
                f'</div>'
                f'<div style="font-family:\'JetBrains Mono\',monospace;'
                f'font-size:11px;display:flex;flex-wrap:wrap;gap:14px;'
                f'color:{T.GREY};margin-top:4px;">'
                f'<span>LOT <span style="color:{T.CYAN};">{lot_txt}</span></span>'
                f'<span>RISK <span style="color:{T.RED};">{risk_txt}</span></span>'
                f'<span>WIN <span style="color:{T.GREEN};">{win_txt}</span> '
                f'<span style="color:{T.GREY};">@ {tp_pips_txt}p</span></span>'
                f'</div>'
                f'<div style="margin-top:6px;">{pills}</div>'
            )
            Panel(
                title=f"{r['pair']} · {r['direction']}",
                tag=f"GRADE {r['grade']}",
                body_html=body,
            ).show()

    @staticmethod
    def _render_table(results, rr_ratio, account_bal, risk_pct) -> None:
        rows = []
        criteria_keys = list(results[0]["scores"].keys()) if results else []
        for r in results:
            inst = INSTRUMENTS.get(r["pair"])
            pip_size = inst.pip_size if inst else None
            pip_value = inst.pip if inst else None
            lv = trade_levels(r["close"], r.get("sl_pips"),
                              pip_size, r["direction"], rr_ratio)
            mb = money_breakdown(r.get("sl_pips"), pip_value,
                                 account_bal, risk_pct, rr_ratio)
            row = {
                "Rank": results.index(r) + 1,
                "Pair": r["pair"],
                "Dir": r["direction"],
                "Score": f"{r['score']}/10",
                "Grade": r["grade"],
                "Price": (f"{r['close']:.5f}" if r["close"] < 100
                          else f"{r['close']:.3f}"),
                "SL Price": fmt_price(lv["sl_price"]),
                "SL Pips": r["sl_pips"],
                "TP Price": fmt_price(lv["tp_price"]),
                "TP Pips": lv["tp_pips"] if lv["tp_pips"] is not None else "—",
                "R:R": f"{rr_ratio:.1f}",
                "Lot": f"{mb['lot']:.2f}" if mb["lot"] is not None else "—",
                "Risk $": f"${mb['risk_amt']:,.0f}" if mb["risk_amt"] is not None else "—",
                "Win $": f"+${mb['win']:,.0f}" if mb["win"] is not None else "—",
            }
            for k in criteria_keys:
                row[k] = "✓" if r["scores"][k] else "✗"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    @staticmethod
    def _render_chart(results) -> None:
        top = results[:20]
        labels = [f"{r['pair']} {r['direction']}" for r in top]
        values = [r["score"] for r in top]
        color_for_grade = {"A": T.GREEN, "B": T.CYAN, "C": T.YELLOW, "D": T.GREY}
        bar_colors = [color_for_grade[r["grade"]] for r in top]
        fig = go.Figure(go.Bar(
            y=labels[::-1], x=values[::-1], orientation="h",
            marker_color=bar_colors[::-1],
            text=[f"{v}/10" for v in values[::-1]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Score: %{x}/10<extra></extra>",
        ))
        fig.add_vline(x=8, line_dash="dash", line_color=T.GREEN, line_width=1.5,
                      annotation_text="Grade A", annotation_font_color=T.GREEN)
        fig.add_vline(x=6, line_dash="dot", line_color=T.CYAN, line_width=1,
                      annotation_text="Grade B", annotation_font_color=T.CYAN)
        fig.update_layout(
            paper_bgcolor=T.BG, plot_bgcolor=T.BG_PANEL,
            font=dict(color=T.WHITE, family=T.FONT_MONO, size=10),
            height=max(400, len(top) * 30),
            xaxis=dict(range=[0, 12], showgrid=True,
                       gridcolor=T.BORDER, title="Score / 10"),
            yaxis=dict(showgrid=False),
            margin=dict(l=10, r=80, t=10, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig, width="stretch",
                        config=dict(displayModeBar=False))
