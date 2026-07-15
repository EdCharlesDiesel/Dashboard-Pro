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

from src.core.signals import _QUALITY_CRITERIA, score_setup
from src.instruments import INSTRUMENTS, TYPICAL_SPREADS
from src.pages_lib.base import BloombergPage, PageContext
from src.services import RiskService, alert_service, account_state, score_history
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
    """Lot size, target $ risk, actual $ risk (after lot rounding), and $ won
    (if TP hit).

    Uses the project risk model (same as the checklist's RiskService):
    lot = (balance · risk%) / sl_pips / pip_value, rounded to 2dp. Because
    that rounding step can be large relative to the ideal lot (a wide-stop
    metals trade rounding 0.078 -> 0.08 lots, for instance), ``actual_risk``
    — lot_size × pip_value × sl_pips — is reported alongside the target
    ``risk_amt`` so a deviation is visible instead of silently assumed away.
    ``win`` is derived from the actual lot (what's really traded), not the
    target, so it and ``risk_amt``/``actual_risk`` stay internally consistent.
    """
    try:
        sl_pips = float(sl_pips)
        pip_value = float(pip_value)
        rr = float(rr)
        account_balance = float(account_balance)
        risk_pct = float(risk_pct)
    except (TypeError, ValueError):
        return {"lot": None, "risk_amt": None, "actual_risk": None, "win": None}
    if sl_pips <= 0 or pip_value <= 0:
        return {"lot": None, "risk_amt": None, "actual_risk": None, "win": None}
    tp_pips = sl_pips * rr
    rb = RiskService.compute(account_balance, risk_pct, pip_value,
                             sl_pips, tp_pips, tp_pips)
    return {"lot": rb.lot_size, "risk_amt": rb.risk_amount,
            "actual_risk": rb.actual_risk, "win": rb.actual_risk * rr}


def risk_deviates(mb: dict, tolerance: float = 0.10) -> bool:
    """True when lot-rounding pushed actual $ risk >tolerance away from the
    target — the flag that should turn the risk figure red in the UI."""
    target, actual = mb.get("risk_amt"), mb.get("actual_risk")
    if not target or actual is None:
        return False
    return abs(actual - target) / target > tolerance


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

    @staticmethod
    def currency_strength_diff(pair: str) -> Optional[float]:
        """(base 20d % return − quote 20d % return), or None for commodities /
        when currency-strength data isn't available. Same 20-day window and
        source as Instrument Predictor's Currency Strength component, so a
        pair's fundamental read agrees across both pages."""
        if INSTRUMENTS[pair].instrument.is_commodity:
            return None
        from src.pages_lib.currency_strength import _currency_returns, _fetch_pair_closes
        closes = _fetch_pair_closes()
        if closes.empty:
            return None
        strength = _currency_returns(closes)
        base, quote = pair.split("/")
        base_s = strength.get(base, {}).get(20)
        quote_s = strength.get(quote, {}).get(20)
        if base_s is None or quote_s is None:
            return None
        return base_s - quote_s

    @classmethod
    def score(cls, pair: str, info: dict, direction: str) -> dict:
        ticker = info["ticker"]
        pip_size = info["pip_size"]
        df_w = cls.weekly(ticker)
        df_d = cls.daily(ticker)
        df_4h = cls.four_hour(ticker)
        spread = TYPICAL_SPREADS.get(pair, 0.0)
        ccy_diff = cls.currency_strength_diff(pair)
        result = score_setup(df_w, df_d, df_4h, direction, pip_size, spread,
                             currency_strength_diff=ccy_diff)
        result["pair"] = pair
        return result


# ── Page implementation ───────────────────────────────────────────────────

class SetupRankerPage(BloombergPage):
    """Bloomberg-styled multi-pair scoring scanner."""

    def configure(self) -> PageContext:
        # Scan both directions by default so every pair is evaluated long AND short.
        st.session_state.setdefault("sr_direction", "Both")
        # Percentage of *direction* criteria (score_setup excludes the 3
        # quality-gate criteria from this) — not "out of 10", since the
        # denominator is 7 or 8 depending on whether Currency Strength
        # applies, and used to read "/10" on a table that showed "/11" rows.
        st.session_state.setdefault("sr_min_pct", 50)
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
        # Threshold default 90%: only the top tier (not all of Grade A ≥80%)
        # is worth an interruption; Grade A still auto-saves to the journal DB
        # via _persist_signals regardless of this toggle.
        st.session_state.setdefault("sr_email_on", False)
        st.session_state.setdefault("sr_email_min", 90)
        # Separate opt-in: email when a Grade-A setup drops out of grade.
        st.session_state.setdefault("sr_email_stale_on", False)
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
                st.session_state.sr_pairs = ["XAU/USD", "XAG/USD", "XPT/USD", "WTI/USD"]
        st.session_state.sr_pairs = st.multiselect(
            "Instruments", INSTRUMENTS.keys(),
            default=[p for p in st.session_state.sr_pairs if p in INSTRUMENTS],
        )
        st.session_state.sr_direction = st.radio(
            "Direction", ["LONG", "SHORT", "Both"], horizontal=True,
            index=["LONG", "SHORT", "Both"].index(st.session_state.sr_direction),
        )
        st.session_state.sr_min_pct = st.slider(
            "Min score %", 0, 100, st.session_state.sr_min_pct, step=5,
            help="Percentage of *direction* criteria passed (7 without "
                 "Currency Strength, 8 with — the 3 quality-gate criteria "
                 "ATR/Spread/4H-Zone don't count toward this).",
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
            "Alert when score ≥ %", 60, 100, int(st.session_state.sr_email_min),
            step=5, help="Percentage of direction criteria — same "
                 "normalization as the Min score % filter above.",
        )
        st.session_state.sr_trades_per_signal = st.number_input(
            "Trades per signal", min_value=1, max_value=10,
            value=int(st.session_state.sr_trades_per_signal), step=1,
            help="Position sizing and projected account growth in the alert are "
                 "multiplied by this. Defaults to 2 (your house rule).",
        )
        st.session_state.sr_email_stale_on = st.toggle(
            "⚠ Email invalidated (Grade A → lower) setups",
            value=st.session_state.sr_email_stale_on,
            help="On each scan, email any setup that was Grade A on the previous "
                 "scan and has since dropped below it. Each specific transition "
                 "(e.g. 90%→50%) fires once; if it recovers to A and drops again, "
                 "that fires again.",
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
                         help="Clear the 'already-alerted' memory (new-setup AND "
                              "invalidation alerts) so current setups can re-fire."):
                alert_service.NotifyCache("setup_ranker").reset()
                alert_service.NotifyCache("setup_ranker_stale").reset()
                st.toast("Alert memory cleared.")

    def body(self, ctx: PageContext) -> None:
        direction = st.session_state.sr_direction
        min_pct = st.session_state.sr_min_pct
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
            (f"MIN {min_pct}%", "cyan"),
            (f"TP {rr_ratio:.1f}R", "green"),
            (f"BAL ${account_bal:,.0f}", ""),
            (f"RISK {risk_pct:.2f}%", "cyan"),
            (datetime.now().strftime("%a %d %b %H:%M"), "amber"),
        ]).show()

        directions = ["LONG", "SHORT"] if direction == "Both" else [direction]

        # Rendered outside the fragment below — fragments can't write new
        # sidebar widgets ("Fragments cannot write widgets to outside
        # containers"), so the debug panel has to live in the outer body().
        from src.debug import debug_panel
        debug_panel("Setup Ranker")

        # Auto-refreshing fragment: re-scans, re-persists, and re-checks email
        # alerts every 5 min on its own (matching the underlying data's cache
        # TTL) without a manual "RESCAN ALL PAIRS" click or full-page rerun —
        # same pattern as market_overview_lib's live trading-ideas fragment.
        # Persistence + email dedupe are keyed per pair/direction/price, so an
        # unattended tick that finds nothing new is a safe no-op.
        @st.fragment(run_every=300, parallel=True)
        def _scan_and_render():
            all_results = self._scan(pairs, directions)

            # Diff every scored pair/direction against its last-seen score so a
            # setup that was Grade A and no longer is gets flagged rather than
            # silently vanishing from the ranked list on the next scan.
            stale = self._track_score_history(all_results)
            # Compare on percentage, not raw score — FX pairs carry an 8th
            # (Currency Strength) direction criterion that commodities don't,
            # so their raw scores aren't on the same scale.
            results = [r for r in all_results if r["pct"] >= min_pct]

            # Auto-save Grade-A setups to the journal DB (deduped, source-tagged).
            self._persist_signals(results, rr_ratio)

            # Fire email alerts for newly-appearing high-score setups (opt-in).
            if st.session_state.get("sr_email_on"):
                self._maybe_email_alerts(results, rr_ratio, account_bal, risk_pct)

            # Fire email alerts for setups that just dropped out of Grade A (opt-in).
            if st.session_state.get("sr_email_stale_on"):
                self._maybe_email_stale_alerts(stale)

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
                # Grade bands are percentage cutoffs (A>=80%, B>=60%, C>=40%)
                # of the direction score, not fixed ranges — FX pairs score
                # /8 (with Currency Strength) while commodities score /7, so
                # a raw range like "6-8" would be inaccurate for FX pairs.
                MetricCell("Grade A (≥80%)", str(grade_counts["A"]), "green"),
                MetricCell("Grade B (60-79%)",  str(grade_counts["B"]), "cyan"),
                MetricCell("Grade C (40-59%)",  str(grade_counts["C"]), "yellow"),
                MetricCell("Grade D (<40%)",   str(grade_counts["D"]), "red"),
                MetricCell("Grade-A Win $",  f"+${grade_a_win:,.0f}", "green"),
                MetricCell("Invalidated",    str(len(stale)), "red" if stale else "white"),
            ])

            self._render_stale_panel(stale)

            if not results:
                st.info(f"◇ No pairs scored ≥{min_pct}%. Lower threshold or change direction.")
                return

            self._render_exposure_strip(results, account_bal, risk_pct, rr_ratio)

            # Best-effort: Daily Cockpit's regime + rate-lean read, displayed
            # per card/row (not scored) so a Grade B that also agrees with
            # the cockpit visibly outranks a Grade B fighting it. A fetch
            # failure here just means no badge, never a broken scan.
            try:
                cockpit = self._cockpit_context(sorted({r["pair"] for r in results}))
            except Exception:
                cockpit = {"regime": {"score": None, "label": "Unknown"}, "biases": {}}

            st.markdown(
                f'<div style="color:#9a9a9a;font-family:\'JetBrains Mono\',monospace;'
                f'font-size:10px;margin:8px 0;">Entry ≈ current close · Stop = 1.5× ATR14 '
                f'(ATR SL, not structure) · TP = stop × {rr_ratio:.1f}R · WIN/RISK $ at '
                f'{risk_pct:.2f}% of ${account_bal:,.0f}. Confirm on 15M before trading.</div>',
                unsafe_allow_html=True,
            )

            tab_cards, tab_table, tab_chart = st.tabs(["◆ RANKED", "▤ TABLE", "▦ CHART"])
            with tab_cards: self._render_cards(results, rr_ratio, account_bal, risk_pct, cockpit)
            with tab_table: self._render_table(results, rr_ratio, account_bal, risk_pct, cockpit)
            with tab_chart: self._render_chart(results)

        _scan_and_render()

    @staticmethod
    def _scan(pairs, directions) -> list:
        """Score every pair × direction, fanned out one worker per pair.

        Each pair's weekly/daily/4H fetch is I/O-bound (yfinance/Postgres), so
        parallelizing across pairs turns the scan's wall-clock time from a sum
        of ~N round-trips into a max over a few concurrent waves. Both
        directions for a pair run in the same worker (fetched data is shared,
        scoring itself is cheap CPU), preserving the original per-direction
        exception isolation.

        Returns EVERY scored pair/direction, unfiltered by score — the min-score
        cutoff is applied by the caller for display. Scanning unfiltered lets
        ``_track_score_history`` see a setup that fell below the threshold (and
        flag it stale) instead of it silently disappearing mid-scan.
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
            # When both directions were scanned for this pair and tied on
            # direction score (quality-gate criteria excluded — see
            # score_setup), there's no directional edge, the same rule
            # analyze_multi_timeframe already applies. Showing both sides as
            # separate ranked ideas would overstate conviction on whichever
            # happened to score higher on a display tiebreak alone.
            if len(scored) == 2 and scored[0]["score"] == scored[1]["score"]:
                return []
            return scored

        def _on_progress(item):
            nonlocal done
            done += 1
            prog.progress(done / total, text=f"Scoring {item[0]}…")

        def _on_result(item, scored):
            _on_progress(item)
            results.extend(scored)

        def _on_error(item, exc):
            _on_progress(item)

        run_parallel(_score_pair, pairs_list, on_result=_on_result, on_error=_on_error)
        prog.empty()
        results.sort(key=lambda x: -x["score"])
        return results

    # ── Score-history tracking (validity/state) ───────────────────────────────
    @classmethod
    def _track_score_history(cls, all_results: list) -> list:
        """Diff every scored pair/direction against its last-seen score,
        annotate each result in place with ``pct``/``delta_pct``/``prev_grade``/
        ``signal_state``, persist the fresh scores for the next scan, and return
        the subset that just fell out of Grade A (``STALE`` — a setup that used
        to justify a trade and no longer does).

        ``signal_state`` is one of NEW (first time seen this pair/direction),
        STALE (was Grade A, isn't anymore), UP/DOWN (score moved, still not a
        grade-A dropout) or UNCHANGED. History survives restarts via
        ``score_history`` (Postgres ``app_state`` + local JSON fallback).
        """
        prev = score_history.load()
        now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        next_state = dict(prev)
        stale = []
        for r in all_results:
            key = f"{r['pair']}|{r['direction']}"
            # pct is already computed correctly by score_setup (direction-only,
            # /7 or /8 depending on whether Currency Strength applies) — reuse
            # it rather than recomputing against a hardcoded denominator.
            pct = r["pct"]
            p = prev.get(key)
            if p is None:
                r["delta_pct"] = None
                r["prev_grade"] = None
                r["prev_pct"] = None
                r["signal_state"] = "NEW"
            else:
                r["delta_pct"] = pct - p.get("pct", pct)
                r["prev_grade"] = p.get("grade")
                r["prev_pct"] = p.get("pct")
                if p.get("grade") == "A" and r["grade"] != "A":
                    r["signal_state"] = "STALE"
                    stale.append(r)
                elif r["delta_pct"] > 0:
                    r["signal_state"] = "UP"
                elif r["delta_pct"] < 0:
                    r["signal_state"] = "DOWN"
                else:
                    r["signal_state"] = "UNCHANGED"
            next_state[key] = {"score": r["score"], "grade": r["grade"],
                               "pct": pct, "ts": now_iso}
        score_history.save(next_state)
        return stale

    @staticmethod
    def _render_stale_panel(stale: list) -> None:
        """A pair/direction that was Grade A on the last scan and isn't now —
        the closest thing this page has to 'this signal is invalid.'"""
        if not stale:
            return
        rows_html = ""
        for r in stale:
            d_color = T.GREEN if r["direction"] == "LONG" else T.RED
            prev_pct = r.get("prev_pct")
            prev_pct_txt = f"{prev_pct}%" if prev_pct is not None else "—"
            rows_html += (
                f'<div style="display:flex;justify-content:space-between;'
                f'flex-wrap:wrap;gap:6px;padding:6px 0;border-bottom:1px solid {T.BORDER};'
                f'font-family:\'JetBrains Mono\',monospace;font-size:11px;">'
                f'<span><span style="color:{T.WHITE};font-weight:700;">{r["pair"]}</span> '
                f'<span style="color:{d_color};">{r["direction"]}</span></span>'
                f'<span style="color:{T.GREY};">was <span style="color:{T.GREEN};">'
                f'Grade A</span> ({prev_pct_txt}) &rarr; now <span style="color:{T.RED};">'
                f'Grade {r["grade"]}</span> ({r["pct"]}%) '
                f'<span style="color:{T.RED};font-weight:700;">{r["delta_pct"]:+d}%</span></span>'
                f'</div>'
            )
        Panel(
            title="⚠ INVALIDATED SINCE LAST SCAN",
            tag=f"{len(stale)}",
            body_html=rows_html,
        ).show()

    # ── Portfolio view: net currency exposure ──────────────────────────────
    @staticmethod
    def _net_exposure(results, account_bal, risk_pct, rr_ratio) -> dict:
        """Signed $ risk per currency across displayed ideas.

        Long base/quote = long base, short quote (the mirror for short) —
        same convention CorrelationService's exposure check uses elsewhere.
        Returns {ccy: {"net": float, "contributions": [(pair, direction,
        signed_risk), ...]}}; a currency whose contributions carry both signs
        means two displayed ideas are fighting each other at the currency
        level even if the net looks small.
        """
        exposure: Dict[str, dict] = {}
        for r in results:
            inst = INSTRUMENTS.get(r["pair"])
            pip_value = inst.pip if inst else None
            mb = money_breakdown(r.get("sl_pips"), pip_value,
                                 account_bal, risk_pct, rr_ratio)
            risk = mb["actual_risk"] if mb["actual_risk"] is not None else mb["risk_amt"]
            if not risk:
                continue
            base, quote = r["pair"].split("/")
            sign = 1 if r["direction"] == "LONG" else -1
            for ccy, s in ((base, sign), (quote, -sign)):
                entry = exposure.setdefault(ccy, {"net": 0.0, "contributions": []})
                signed = s * risk
                entry["net"] += signed
                entry["contributions"].append((r["pair"], r["direction"], signed))
        return exposure

    @classmethod
    def _render_exposure_strip(cls, results, account_bal, risk_pct, rr_ratio) -> None:
        """Portfolio view above the ranked cards: sum signed risk per currency
        so a board that's really 'one giant JPY trade' spread across several
        'independent' cards is visible before it's four separate $100 risks
        that are actually ~$400 short JPY."""
        exposure = cls._net_exposure(results, account_bal, risk_pct, rr_ratio)
        if not exposure:
            return
        per_trade_risk = account_bal * (risk_pct / 100)
        conflict_ccys = {
            ccy for ccy, data in exposure.items()
            if len({1 if c[2] > 0 else -1 for c in data["contributions"]}) > 1
        }
        chips = ""
        for ccy, data in sorted(exposure.items(), key=lambda kv: -abs(kv[1]["net"])):
            net = data["net"]
            color = T.GREEN if net > 0 else T.RED if net < 0 else T.GREY
            heavy = per_trade_risk > 0 and abs(net) > 2 * per_trade_risk
            conflict = ccy in conflict_ccys
            label = f"{ccy} {'+' if net >= 0 else '-'}${abs(net):,.0f}"
            border = T.RED if (heavy or conflict) else T.BORDER
            tags = "".join([
                ' <span style="color:#000;background:{0};padding:0 3px;">{1}</span>'.format(
                    T.RED, tag)
                for tag in (["2X"] if heavy else []) + (["CONFLICT"] if conflict else [])
            ])
            chips += (
                f'<span style="display:inline-block;border:1px solid {border};'
                f'color:{color};padding:2px 8px;font-size:11px;margin:2px;'
                f'font-family:\'JetBrains Mono\',monospace;font-weight:700;">'
                f'{label}{tags}</span>'
            )
        Panel(
            title="⚖ NET CURRENCY EXPOSURE",
            tag=f"{len(conflict_ccys)} conflict" if conflict_ccys else "",
            body_html=(
                f'<div>{chips}</div>'
                f'<div style="color:{T.GREY};font-size:10px;margin-top:6px;">'
                f'Signed $ risk per currency across displayed ideas · red border '
                f'= exceeds 2× per-trade risk or the same currency appears on '
                f'both sides of two ideas (CONFLICT).</div>'
            ),
        ).show()

    # ── Cockpit reconciliation (regime + rate lean, displayed not scored) ──
    @staticmethod
    def _cockpit_context(pairs: list) -> dict:
        """Pull the Daily Cockpit's regime + rate-bias reads — same pure
        functions, same FRED/yfinance-backed caches (daily_cockpit_tab.py is
        import-safe: its page bootstrap is behind ``if __name__ ==
        "__main__"``, same pattern as cot_composite_trade_signal.py) — so a
        Setup Ranker idea that also agrees with the cockpit visibly outranks
        one that fights it, without re-deriving regime/rate logic here.
        Best-effort: any fetch failure just leaves cards without a badge.
        """
        from pages.daily_cockpit_tab import (
            RISK_SIGNALS, SHORT_RATE, load_fred, load_risk_signal, oriented_z,
            rate_bias, risk_composite,
        )
        comps = {}

        def _fetch(item):
            _name, cfg = item
            return oriented_z(load_risk_signal(cfg), cfg["kind"], cfg["o"])

        run_parallel(
            _fetch, list(RISK_SIGNALS.items()),
            on_result=lambda item, z: comps.__setitem__(item[0], z),
            on_error=lambda item, exc: comps.__setitem__(item[0], None),
        )
        rc = risk_composite(comps)

        ccys = sorted({c for p in pairs if "/" in p for c in p.split("/")
                      if c in SHORT_RATE})
        rate_cache = {}
        run_parallel(
            lambda ccy: load_fred(SHORT_RATE[ccy]), ccys,
            on_result=lambda ccy, series: rate_cache.__setitem__(ccy, series),
            on_error=lambda ccy, exc: rate_cache.__setitem__(ccy, pd.Series(dtype=float)),
        )

        biases = {}
        for p in pairs:
            if "/" not in p:
                continue
            base, quote = p.split("/")
            rb_s, rq_s = rate_cache.get(base), rate_cache.get(quote)
            if rb_s is None or rq_s is None or rb_s.empty or rq_s.empty \
                    or len(rb_s) < 8 or len(rq_s) < 8:
                continue
            try:
                biases[p] = rate_bias(float(rb_s.iloc[-1]), float(rq_s.iloc[-1]),
                                      float(rb_s.iloc[-7]), float(rq_s.iloc[-7]))
            except Exception:
                pass
        return {"regime": rc, "biases": biases}

    @staticmethod
    def _cockpit_badge(r: dict, cockpit: dict) -> tuple:
        """(text, color) for one result's cockpit row — 'agrees'/'fights' by
        the cockpit's own is_aligned rule (rate lean direction + regime veto),
        or a neutral no-read badge when no rate bias is available (e.g. metals)."""
        from pages.daily_cockpit_tab import is_aligned
        rc = cockpit.get("regime", {"score": None, "label": "Unknown"})
        base, quote = (r["pair"].split("/") + ["", ""])[:2]
        rb = cockpit.get("biases", {}).get(r["pair"])
        emoji = {"Risk-On": "🟢", "Risk-Off": "🔴", "Neutral": "🟡", "Unknown": "⚪"}
        regime_txt = f"{emoji.get(rc['label'], '⚪')} {rc['label']}"
        if not rb:
            return f"🛫 {regime_txt} · no rate read", T.GREY
        agree = is_aligned(r["direction"].lower(), rb["lean"], base, quote, rc["label"])
        color = T.GREEN if agree else T.RED
        verdict = "agrees" if agree else "fights"
        return f"🛫 {regime_txt} · {rb['lean']} ({verdict})", color

    # ── DB persistence ──────────────────────────────────────────────────────
    @staticmethod
    def _persist_signals(results, rr_ratio) -> None:
        """Persist Grade-A setups (direction score ≥80%, i.e. 6/7 or 7/8) to
        trade_setups via the shared signal store. Deduped per pair/direction/
        price, tagged source='setup_ranker', silent no-op without a DB."""
        signals = []
        for r in results:
            if r.get("grade") != "A":  # Grade A only — high-conviction
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
        threshold_pct = int(st.session_state.get("sr_email_min", 90))
        qualifying = [r for r in results if r.get("pct", 0) >= threshold_pct]
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
            setups, rr_ratio, account_bal, risk_pct, threshold_pct, n_trades)
        subject = (
            f"🎰 {len(setups)} new setup{'s' if len(setups) != 1 else ''} ≥{threshold_pct}% — "
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
    def _build_alert_email(setups, rr_ratio, account_bal, risk_pct, threshold_pct,
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

            # Combined figures for n trades taken on this one signal. Risk uses
            # actual_risk (post lot-rounding), not the idealized target — a
            # wide-stop instrument (metals) can round to noticeably more $ at
            # risk than the account_bal × risk_pct target implies.
            win_c = mb["win"] * n if mb["win"] is not None else None
            risk_c = mb["actual_risk"] * n if mb["actual_risk"] is not None else None
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
            if risk_deviates(mb):
                risk_txt += " ⚠"
            lot_txt = f"{lot_c:.2f}" if lot_c is not None else "—"
            grow_txt = (f"+${win_c:,.0f} <span style='color:#8b8fa8;'>(+{grow_pct:.2f}%)</span>"
                        if grow_pct is not None else "—")
            grow_plain = f"+${win_c:,.0f} (+{grow_pct:.2f}%)" if grow_pct is not None else "—"
            rows_html += f"""
            <tr style="border-bottom:1px solid #2d3148;">
              <td style="padding:9px 12px;font-weight:700;color:#e0e0e0;">{r['pair']}</td>
              <td style="padding:9px 12px;color:{d_color};font-weight:700;">{arrow}</td>
              <td style="padding:9px 12px;color:#ffa726;font-weight:700;">{r['score']}/{r.get('max_score', 7)}
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
                f"{r['pair']} {r['direction']}  |  Score {r['score']}/{r.get('max_score', 7)} ({r['grade']})  |  "
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
              🎰 Setup Ranker — {len(setups)} New Setup{'s' if len(setups) != 1 else ''} ≥ {threshold_pct}%
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
        plain = (f"SETUP RANKER — {len(setups)} new setup(s) ≥ {threshold_pct}% "
                 f"({datetime.now().strftime('%Y-%m-%d %H:%M')})\n"
                 f"Sizing {risk_pct:.2f}% · {n} trades/signal\n\n"
                 + "\n".join(plain_lines) + "\n" + summary_plain)
        return html, plain

    # ── Invalidation email alerts ─────────────────────────────────────────
    @classmethod
    def _maybe_email_stale_alerts(cls, stale: list) -> None:
        """Email setups that just dropped out of Grade A (opt-in, separate
        toggle/dedupe ledger from the new-setup alert). Dedup key includes the
        specific prev%→now% transition, so the same drop doesn't re-fire every
        5 min, but a setup that recovers to Grade A and drops again does. The
        'seen' memory is only updated after a successful send, matching
        ``_maybe_email_alerts``."""
        if not stale:
            return

        keys = [f"{r['pair']}|{r['direction']}|{r.get('prev_pct')}->{r['pct']}"
                for r in stale]
        cache = alert_service.NotifyCache("setup_ranker_stale")
        seen = cache.load()
        fresh = [(r, k) for r, k in zip(stale, keys) if k not in seen]
        if not fresh:
            return

        setups = [r for r, _ in fresh]
        html, plain = cls._build_stale_email(setups)
        subject = (
            f"⚠ {len(setups)} setup{'s' if len(setups) != 1 else ''} invalidated — "
            f"{', '.join(r['pair'] for r in setups[:3])}{'…' if len(setups) > 3 else ''}"
        )
        ok, msg = alert_service.send_email(subject, html, plain)
        if ok:
            cache.filter_new([k for _, k in fresh])  # mark as alerted only now
            st.toast(f"📧 Emailed {len(setups)} invalidated setup(s) to "
                     f"{alert_service.email_recipient()}", icon="⚠️")
        else:
            st.toast(f"⚠ Invalidation alert email failed: {msg}", icon="⚠️")

    @staticmethod
    def _build_stale_email(stale: list):
        """Render (html, plain) bodies for a batch of just-invalidated setups."""
        rows_html, plain_lines = "", []
        for r in stale:
            d_color = "#26a69a" if r["direction"] == "LONG" else "#ef5350"
            arrow = "📈 LONG" if r["direction"] == "LONG" else "📉 SHORT"
            prev_pct = r.get("prev_pct")
            prev_pct_txt = f"{prev_pct}%" if prev_pct is not None else "—"
            rows_html += f"""
            <tr style="border-bottom:1px solid #2d3148;">
              <td style="padding:9px 12px;font-weight:700;color:#e0e0e0;">{r['pair']}</td>
              <td style="padding:9px 12px;color:{d_color};font-weight:700;">{arrow}</td>
              <td style="padding:9px 12px;color:#26a69a;">Grade A
                  <span style="color:#8b8fa8;font-size:11px;">({prev_pct_txt})</span></td>
              <td style="padding:9px 12px;color:#ef5350;font-weight:700;">Grade {r['grade']}
                  <span style="color:#8b8fa8;font-size:11px;">({r['pct']}%)</span></td>
              <td style="padding:9px 12px;color:#ef5350;font-weight:700;">{r['delta_pct']:+d}%</td>
            </tr>"""
            plain_lines.append(
                f"{r['pair']} {r['direction']}  |  was Grade A ({prev_pct_txt})  ->  "
                f"now Grade {r['grade']} ({r['pct']}%)  |  {r['delta_pct']:+d}%"
            )

        html = f"""<!DOCTYPE html><html><body style="background:#0e1117;
          font-family:'Courier New',monospace;color:#c0c0c0;margin:0;padding:20px;">
          <div style="max-width:820px;margin:0 auto;">
            <h2 style="color:#ff5252;border-bottom:1px solid #2d3148;padding-bottom:10px;">
              ⚠ Setup Ranker — {len(stale)} Setup{'s' if len(stale) != 1 else ''} Invalidated
            </h2>
            <p style="color:#8b8fa8;font-size:13px;">
              These were Grade A on the previous scan and have since dropped out, as of
              {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. Re-evaluate before acting on them.
            </p>
            <table style="width:100%;border-collapse:collapse;background:#1a1d27;border-radius:8px;overflow:hidden;">
              <thead><tr style="background:#000000;">
                {''.join(f'<th style="padding:9px 12px;text-align:left;color:#8b8fa8;font-size:11px;letter-spacing:1px;">{h}</th>' for h in ('PAIR', 'DIR', 'WAS', 'NOW', 'Δ'))}
              </tr></thead>
              <tbody>{rows_html}</tbody>
            </table>
            <p style="color:#8b8fa8;font-size:11px;margin-top:22px;border-top:1px solid #2d3148;padding-top:12px;">
              If you already took one of these, tighten or exit per your plan — it no longer
              clears the Grade-A bar.
            </p>
          </div></body></html>"""
        plain = (f"SETUP RANKER — {len(stale)} setup(s) invalidated "
                 f"({datetime.now().strftime('%Y-%m-%d %H:%M')})\n\n"
                 + "\n".join(plain_lines))
        return html, plain

    @classmethod
    def _render_cards(cls, results, rr_ratio, account_bal, risk_pct, cockpit=None) -> None:
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
            # Show actual $ risk (post lot-rounding), not the idealized target
            # — flag red when rounding pushed it >10% off target so a small
            # lot step on a wide-stop instrument (e.g. metals) doesn't hide.
            risk_deviated = risk_deviates(mb)
            risk_val = mb["actual_risk"] if mb["actual_risk"] is not None else mb["risk_amt"]
            risk_txt = f"${risk_val:,.0f}" if risk_val is not None else "—"
            if risk_deviated:
                risk_txt += f" ⚠ (target ${mb['risk_amt']:,.0f})"
            win_txt = f"+${mb['win']:,.0f}" if mb["win"] is not None else "—"
            dots = "".join(
                f'<span style="display:inline-block;width:8px;height:8px;'
                f'background:{color if i < r["score"] else T.BORDER};'
                f'margin-right:2px;"></span>'
                for i in range(r.get("max_score", 7))
            )
            # Direction pills (feed the grade) rendered separately from
            # quality-gate pills (ATR/Spread/4H-Zone — pass/fail identically
            # for LONG and SHORT, so they don't belong mixed into the same
            # row as evidence for this specific direction).
            pills = "".join(
                f'<span style="display:inline-block;border:1px solid '
                f'{T.GREEN if v else T.BORDER};color:{T.GREEN if v else T.GREY};'
                f'padding:1px 6px;font-size:9px;margin:1px;'
                f'font-family:\'JetBrains Mono\',monospace;letter-spacing:0.05em;">'
                f'{k}</span>'
                for k, v in r["scores"].items() if k not in _QUALITY_CRITERIA
            )
            quality_pills = "".join(
                f'<span style="display:inline-block;border:1px solid '
                f'{T.CYAN if v else T.BORDER};color:{T.CYAN if v else T.GREY};'
                f'padding:1px 6px;font-size:9px;margin:1px;'
                f'font-family:\'JetBrains Mono\',monospace;letter-spacing:0.05em;">'
                f'{k}</span>'
                for k, v in r["scores"].items() if k in _QUALITY_CRITERIA
            )
            quality_txt = f"{r.get('quality_score', 0)}/{r.get('quality_max', 3)}"
            spread_warn = ""
            if r["spread_pct"] > 5:
                spread_warn = (
                    f'<span style="border:1px solid {T.RED};color:{T.RED};'
                    f'padding:1px 6px;font-size:9px;margin-left:6px;'
                    f'font-family:\'JetBrains Mono\',monospace;">⚠ SPREAD '
                    f'{r["spread_pct"]:.1f}%</span>'
                )
            state = r.get("signal_state")
            delta = r.get("delta_pct")
            state_badge = ""
            if state == "NEW":
                state_badge = (f'<span style="color:{T.CYAN};font-size:9px;'
                               f'margin-left:6px;">NEW</span>')
            elif state == "STALE":
                state_badge = (f'<span style="color:{T.RED};font-size:9px;'
                               f'margin-left:6px;">⚠ WAS GRADE A</span>')
            elif delta:
                dcolor = T.GREEN if delta > 0 else T.RED
                arrow = "▲" if delta > 0 else "▼"
                state_badge = (f'<span style="color:{dcolor};font-size:9px;'
                               f'margin-left:6px;">{arrow}{delta:+d}%</span>')

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
                f'font-family:\'JetBrains Mono\',monospace;">{r["score"]}/{r.get("max_score", 7)}</span> '
                f'<span style="color:{color};font-size:10px;letter-spacing:0.1em;">'
                f'{icon} {label}</span>{state_badge}'
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
                f'<div style="margin-top:4px;display:flex;align-items:center;gap:6px;">'
                f'<span style="color:{T.GREY};font-size:9px;letter-spacing:0.05em;">'
                f'QUALITY {quality_txt}</span>{quality_pills}</div>'
            )
            if cockpit is not None:
                cockpit_txt, cockpit_color = cls._cockpit_badge(r, cockpit)
                body += (
                    f'<div style="margin-top:4px;font-size:10px;'
                    f'font-family:\'JetBrains Mono\',monospace;color:{cockpit_color};">'
                    f'{cockpit_txt}</div>'
                )
            Panel(
                title=f"{r['pair']} · {r['direction']}",
                tag=f"GRADE {r['grade']}",
                body_html=body,
            ).show()

    @classmethod
    def _render_table(cls, results, rr_ratio, account_bal, risk_pct, cockpit=None) -> None:
        rows = []
        # Union of every result's criteria, not just the first row's — FX
        # pairs carry "Currency Strength" (8 direction criteria) but
        # commodities don't (7), so the first row alone may not have every
        # column. Quality-gate criteria (ATR/Spread/4H-Zone) are excluded
        # here since they get their own "Quality" summary column instead.
        criteria_keys: list = []
        for r in results:
            for k in r["scores"]:
                if k not in criteria_keys and k not in _QUALITY_CRITERIA:
                    criteria_keys.append(k)
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
                "Score": f"{r['score']}/{r.get('max_score', 7)}",
                "Grade": r["grade"],
                "Quality": f"{r.get('quality_score', 0)}/{r.get('quality_max', 3)}",
                "State": r.get("signal_state", "—"),
                "Δ%": (f"{r['delta_pct']:+d}%"
                       if r.get("delta_pct") is not None else "—"),
                "Price": (f"{r['close']:.5f}" if r["close"] < 100
                          else f"{r['close']:.3f}"),
                "SL Price": fmt_price(lv["sl_price"]),
                "SL Pips": r["sl_pips"],
                "TP Price": fmt_price(lv["tp_price"]),
                "TP Pips": lv["tp_pips"] if lv["tp_pips"] is not None else "—",
                "R:R": f"{rr_ratio:.1f}",
                "Lot": f"{mb['lot']:.2f}" if mb["lot"] is not None else "—",
                "Risk $ (target)": f"${mb['risk_amt']:,.0f}" if mb["risk_amt"] is not None else "—",
                "Risk $ (actual)": (
                    f"${mb['actual_risk']:,.0f}" + (" ⚠" if risk_deviates(mb) else "")
                    if mb["actual_risk"] is not None else "—"
                ),
                "Win $": f"+${mb['win']:,.0f}" if mb["win"] is not None else "—",
            }
            if cockpit is not None:
                cockpit_txt, _ = cls._cockpit_badge(r, cockpit)
                row["Cockpit"] = cockpit_txt
            for k in criteria_keys:
                if k not in r["scores"]:
                    row[k] = "—"  # not applicable (e.g. Currency Strength on a commodity)
                else:
                    row[k] = "✓" if r["scores"][k] else "✗"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    @staticmethod
    def _render_chart(results) -> None:
        top = results[:20]
        labels = [f"{r['pair']} {r['direction']}" for r in top]
        # Percentage, not raw score — FX pairs score /8 (Currency Strength)
        # while commodities score /7, so raw counts aren't comparable on one
        # axis, but pct (grade cutoffs are pct-based) always is.
        values = [r["pct"] for r in top]
        color_for_grade = {"A": T.GREEN, "B": T.CYAN, "C": T.YELLOW, "D": T.GREY}
        bar_colors = [color_for_grade[r["grade"]] for r in top]
        fig = go.Figure(go.Bar(
            y=labels[::-1], x=values[::-1], orientation="h",
            marker_color=bar_colors[::-1],
            text=[f"{v}%" for v in values[::-1]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Score: %{x}%<extra></extra>",
        ))
        fig.add_vline(x=80, line_dash="dash", line_color=T.GREEN, line_width=1.5,
                      annotation_text="Grade A", annotation_font_color=T.GREEN)
        fig.add_vline(x=60, line_dash="dot", line_color=T.CYAN, line_width=1,
                      annotation_text="Grade B", annotation_font_color=T.CYAN)
        fig.update_layout(
            paper_bgcolor=T.BG, plot_bgcolor=T.BG_PANEL,
            font=dict(color=T.WHITE, family=T.FONT_MONO, size=10),
            height=max(400, len(top) * 30),
            xaxis=dict(range=[0, 110], showgrid=True,
                       gridcolor=T.BORDER, title="Score %"),
            yaxis=dict(showgrid=False),
            margin=dict(l=10, r=80, t=10, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig, width="stretch",
                        config=dict(displayModeBar=False))
