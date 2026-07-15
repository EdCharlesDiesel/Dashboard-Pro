"""Checklist mode — 18-point pre-trade confluence flow."""
from __future__ import annotations

import json
from datetime import datetime
from typing import List

import pandas as pd
import streamlit as st

from src.db import DBConfig
from src.db.cache import (
    cached_daily_losses,
    cached_load_open,
    cached_load_setups,
    cached_performance_stats,
    clear_read_caches,
    pooled_repository,
)
from src.indicators import TrendSignalEvaluator
from src.instruments import INSTRUMENTS
from src.pages_lib.daily_trading.state import CHECKS_TOTAL, CRITICAL_CHECKS
from src.services import (
    CorrelationService, MTFService, RiskService, SessionService,
)
from src.ui.charts import ChartBuilder
from src.ui.components import (
    Chip, CommandBar, DataRow, HeroHeader, MetricCell, Panel,
    ProgressBar, TickerStrip, render_data_rows, render_metric_row,
)
from src.ui.theme import BloombergTheme as T


class ChecklistController:
    """Owns all the rendering for CHECKLIST mode.

    Logic is preserved from the original procedural code — only the
    presentation is restructured.
    """

    def render(self) -> None:
        inst_name = st.session_state.selected_instrument
        inst = INSTRUMENTS[inst_name]
        direction = st.session_state.trade_direction
        pip_val = inst["pip"]
        sl_pips = st.session_state.get("trend_sl_pips", 20.0)
        tp1_pips = st.session_state.get("trend_tp1_pips", 40.0)
        tp2_pips = st.session_state.get("trend_tp2_pips", 60.0)
        atr_data = st.session_state.get(f"atr_data_{inst_name}", None)
        atr14 = atr_data["atr14"] if atr_data else 0.0
        atr20 = atr_data["atr20"] if atr_data else 0.0
        atr_ok = atr_data["atr_ok"] if atr_data else None

        checked = sum(st.session_state[f"check_{i}"] for i in range(1, CHECKS_TOTAL + 1))
        pct = int(checked / CHECKS_TOTAL * 100)
        risk = RiskService.compute(
            st.session_state.account_bal, st.session_state.risk_pct,
            pip_val, sl_pips, tp1_pips, tp2_pips,
        )
        critical_ok = all(st.session_state[f"check_{i}"] for i in CRITICAL_CHECKS)
        verdict_label, chip_kind, prog_color = self._verdict(checked, pct, critical_ok)

        # ── Ticker strip at the top ────────────────────────────────────────
        self._render_ticker_strip(inst_name)

        # ── Hero ──────────────────────────────────────────────────────────
        sess = SessionService.current()
        instrument_obj = INSTRUMENTS.get(inst_name)
        display = instrument_obj.display_name if instrument_obj else inst_name
        HeroHeader(
            title="DAILY TRADING SYSTEM",
            subtitle=(
                f"18-POINT CONFLUENCE · {sess['icon']} {sess['window'].upper()} "
                f"· {datetime.now().strftime('%a %d %b %Y %H:%M UTC')}"
            ),
            symbol=display,
            ticker=inst["ticker"],
            side=direction,
        ).show()

        # ── Command bar (account snapshot) ────────────────────────────────
        CommandBar(
            label="ACCT",
            cells=[
                (f"BAL ${st.session_state.account_bal:,.0f}", ""),
                (f"RISK {st.session_state.risk_pct:.2f}%", "cyan"),
                (f"SL {sl_pips} pips", "red"),
                (f"TP1 {tp1_pips} pips", "green"),
                (f"R:R {risk.rr_tp1:.2f}:1", "green" if risk.rr_tp1 >= 2 else "red"),
                (f"LOT {risk.lot_size:.2f}", "cyan"),
            ],
        ).show()

        # ── KPI strip ─────────────────────────────────────────────────────
        kpis = [
            MetricCell("Checks Passed", f"{checked}/{CHECKS_TOTAL}", color=self._color_for(prog_color)),
            MetricCell("Confidence", f"{pct}%", color=self._color_for(prog_color)),
            MetricCell("ATR (14)", f"{atr14:.5f}" if atr14 else "—", color="white"),
            MetricCell("ATR (20)", f"{atr20:.5f}" if atr20 else "—", color="white"),
            MetricCell("Stop Loss", f"{sl_pips}p", color="red"),
            MetricCell("R:R TP1",   f"{risk.rr_tp1:.2f}:1",
                       color="green" if risk.rr_tp1 >= 2 else "red"),
            MetricCell("Lot Size",  f"{risk.lot_size:.2f}", color="cyan"),
            MetricCell("Risk $",    f"${risk.risk_amount:.0f}", color="amber"),
        ]
        render_metric_row(kpis)

        # ── Progress strip ────────────────────────────────────────────────
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;'
            f'margin:6px 0 2px 0;font-family:\'JetBrains Mono\',monospace;font-size:10px;">'
            f'<span style="color:#9a9a9a;letter-spacing:0.15em;">CHECKLIST PROGRESS</span>'
            f'<span style="color:{prog_color};letter-spacing:0.1em;font-weight:700;">'
            f'{pct}% · {verdict_label}</span></div>',
            unsafe_allow_html=True,
        )
        ProgressBar(pct, color=prog_color).show()

        # ── MTF alignment strip ───────────────────────────────────────────
        with st.spinner("Checking MTF alignment…"):
            mtf = MTFService.alignment(inst["ticker"], direction)
        self._render_mtf_strip(mtf)

        # ── Main 2-column responsive layout (mosaic) ──────────────────────
        col_check, col_dash = st.columns([3, 2], gap="medium")
        with col_check:
            self._render_checks_panel(inst, direction, atr14, atr20, atr_ok,
                                       sl_pips, tp1_pips, risk)
        with col_dash:
            self._render_dashboard(
                inst_name, display, direction, verdict_label, chip_kind,
                prog_color, checked, pct, critical_ok,
                atr14, atr20, atr_data, sl_pips, tp1_pips, tp2_pips, risk,
            )

        # ── Performance & journal tail ────────────────────────────────────
        self._render_performance_section(risk)
        self._render_close_trade_section()
        self._render_trade_log_section()

    # ── helpers ────────────────────────────────────────────────────────────
    @staticmethod
    def _verdict(checked: int, pct: int, critical_ok: bool):
        if checked >= 16 and critical_ok:
            return "● GO", "go", T.GREEN
        if pct >= 55:
            return "◐ WAIT", "wait", T.YELLOW
        return "○ PASS", "no", T.RED

    @staticmethod
    def _color_for(hex_color: str) -> str:
        return {
            T.GREEN: "green", T.YELLOW: "yellow", T.RED: "red",
            T.AMBER: "amber", T.CYAN: "cyan",
        }.get(hex_color, "white")

    def _render_ticker_strip(self, current: str) -> None:
        """Build a synthetic ticker strip from session state — no extra API calls.

        Shows the current pick + a handful of session-state ATR readings already
        cached. We don't trigger fetches here so the page stays snappy.
        """
        items: List = []
        for name in ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD", current]:
            if name not in INSTRUMENTS:
                continue
            atr = st.session_state.get(f"atr_data_{name}", None)
            if atr:
                items.append((
                    INSTRUMENTS[name]["ticker"],
                    f"ATR14 {atr['atr14']:.5f}",
                    (atr["atr14"] - atr["atr20"]) / atr["atr20"] * 100,
                ))
            else:
                items.append((INSTRUMENTS[name]["ticker"], "—", None))
        # Dedup while preserving order
        seen = set()
        dedup = []
        for it in items:
            if it[0] in seen:
                continue
            seen.add(it[0])
            dedup.append(it)
        TickerStrip(dedup).show()

    def _render_mtf_strip(self, mtf: dict) -> None:
        target = mtf["target"]
        aligned = mtf["aligned"]
        if aligned == 3:
            badge = Chip("★ FULLY ALIGNED", "go").render()
        elif aligned == 2:
            badge = Chip("◆ 2/3 ALIGNED", "amber").render()
        else:
            badge = Chip(f"{aligned}/3 ALIGNED", "no").render()

        def cell(label, val):
            if val is None:
                return Chip(f"{label} —", "info").render()
            if val == target:
                return Chip(f"✓ {label} {val}", "go").render()
            if val == "NEUTRAL":
                return Chip(f"~ {label} {val}", "info").render()
            return Chip(f"✗ {label} {val}", "no").render()

        body = " ".join([
            cell("WEEKLY", mtf.get("weekly")),
            cell("DAILY", mtf.get("daily")),
            cell("4H", mtf.get("4h")),
        ])
        st.markdown(
            f'<div class="bb-panel"><div class="bb-panel-header">'
            f'<span>MULTI-TIMEFRAME ALIGNMENT</span><span class="bb-tag">{target}</span>'
            f'</div><div class="bb-panel-body" style="display:flex;align-items:center;'
            f'gap:8px;flex-wrap:wrap;">{body}'
            f'<span style="margin-left:auto;">{badge}</span></div></div>',
            unsafe_allow_html=True,
        )

    def _render_checks_panel(
        self, inst, direction, atr14, atr20, atr_ok,
        sl_pips, tp1_pips, risk,
    ) -> None:
        sess = SessionService.current()
        atr_hint = (
            f"✓ ATR14 {atr14:.5f} > ATR20 {atr20:.5f} — volatility supports follow-through"
            if atr_ok else
            f"✗ ATR14 {atr14:.5f} ≤ ATR20 {atr20:.5f} — low volatility caution"
            if atr_ok is False else
            "Fetch ATR data from sidebar"
        )

        # Section headers and the per-check OPEN ▸ deep-links are kept in sync
        # with the 8-step workflow in src/pages_lib/navigation.py — every check
        # that has a matching nav page links to it, and the section names mirror
        # the nav step labels. The check numbers (1–18) are unchanged: they are
        # already in workflow order, so CRITICAL_CHECKS (11–16), the radar and
        # autofill indices stay valid.
        def link(slug: str, label: str = "OPEN ▸") -> str:
            return (f" · <a href='/{slug}' target='_self' "
                    f"style='color:#00e0ff;'>{label}</a>")

        sections = [
            ("2 · FILTER THE DAY", [
                (1, "Macro bias confirmed",
                 "Rates, GDP, inflation favour direction"),
                (2, "No red-folder news",
                 "No high-impact news within 1 hour" + link("news-filter")
                 + link("surprise_tab", "GATE ▸")),
                (3, "Correlated markets align",
                 f"Expected: {inst['corr']}" + link("correlations")),
                (4, "ATR above 20-period avg", atr_hint),
            ]),
            ("3 · WEEKLY BIAS", [
                (5, "Weekly EMA aligned",
                 "Weekly EMA direction matches trade direction" + link("weekly-ema")),
                (6, "Weekly RSI has room",
                 "Not yet OB (>70) or OS (<30)"),
                (7, "Weekly Swing aligned",
                 "Daily confirmation shows aligned" + link("weekly-swing")),
            ]),
            ("4 · DAILY CONFIRM", [
                (8, "Daily trend intact",
                 "EMA20 > EMA50 for longs / inverse for shorts" + link("daily-trend")),
                (9, "Daily MACD momentum",
                 "MACD histogram turning into trade direction" + link("daily-macd")),
                (10, "Entry in kill zone",
                 f"NOW: {sess['icon']} {sess['window']} ({sess['time']}) — {sess['desc']}"),
            ]),
            ("5 · 4H ZONE", [
                (11, "Price at 4H zone",
                 "Overlap of Fib + Pivot + EMA on 4H" + link("4H-confluence-zone")),
                (12, "Min 2/3 confluence",
                 "≥2 of 3 elements confirmed" + link("confluence-checker")),
            ]),
            ("6 · 15M TRIGGER", [
                (13, "15M rejection",
                 "Rejection into the 15M fib golden zone" + link("15m-fib-entry")),
                (14, "15M entry fired",
                 "Confirmed fib golden-zone entry" + link("15m-fib-entry")),
            ]),
            ("7 · RISK & EXECUTE", [
                (15, "Stop below/above structure",
                 f"SL = {sl_pips} pips (1.5×ATR14)" + link("stop-structure")),
                (16, "R:R ≥ 2:1 to TP1",
                 f"TP1 = {tp1_pips} pips → R:R {risk.rr_tp1:.2f}:1 "
                 f"{'✓' if risk.rr_tp1 >= 2 else '✗'}" + link("rr-calculator")),
                (17, "Partial TP + BE rule set",
                 f"50% close TP1 → SL to BE; runner trails to TP2"),
                (18, "Position within limit",
                 f"Lot {risk.lot_size:.2f} · Risk ${risk.risk_amount:.2f}"),
            ]),
        ]
        # Autofill button at the start
        with Panel("PRE-TRADE CHECKS", tag=f"{CHECKS_TOTAL}-POINT").context():
            if st.button("⚡ Auto-fill from Trend Signals", key="auto_trend",
                         width="stretch"):
                self._autofill_from_trend(inst, direction)
                st.rerun()

            for title, items in sections:
                st.markdown(
                    f'<div style="color:#00ff41;font-family:\'JetBrains Mono\',monospace;'
                    f'font-size:10px;letter-spacing:0.18em;border-left:3px solid #00ff41;'
                    f'padding-left:8px;margin:10px 0 6px 0;">{title}</div>',
                    unsafe_allow_html=True,
                )
                for num, t, desc in items:
                    self._render_check(num, t, desc)

    @staticmethod
    def _autofill_from_trend(inst, direction) -> None:
        with st.spinner("Fetching trend signals…"):
            df_w = TrendSignalEvaluator.fetch(inst["ticker"], "1d", "2y", "1W")
            df_d = TrendSignalEvaluator.fetch(inst["ticker"], "1d", "2y", None)
            if df_w is not None and len(df_w) > 10:
                sig_w = TrendSignalEvaluator.evaluate(df_w)
                if (sig_w.direction in ("BUY", "STRONG_BUY") and direction == "LONG") or \
                   (sig_w.direction in ("SELL", "STRONG_SELL") and direction == "SHORT"):
                    for i in [5, 6, 7]:
                        st.session_state[f"check_{i}"] = True
            if df_d is not None and len(df_d) > 10:
                sig_d = TrendSignalEvaluator.evaluate(df_d)
                if (sig_d.direction in ("BUY", "STRONG_BUY") and direction == "LONG") or \
                   (sig_d.direction in ("SELL", "STRONG_SELL") and direction == "SHORT"):
                    for i in [8, 9]:
                        st.session_state[f"check_{i}"] = True

    @staticmethod
    def _render_check(num: int, title: str, desc: str) -> None:
        col_a, col_b = st.columns([0.06, 0.94])
        with col_a:
            val = st.checkbox(
                "", value=st.session_state[f"check_{num}"], key=f"cb_{num}",
                label_visibility="collapsed",
            )
            st.session_state[f"check_{num}"] = val
        with col_b:
            icon = "✓" if st.session_state[f"check_{num}"] else "○"
            icon_color = T.GREEN if st.session_state[f"check_{num}"] else T.GREY
            st.markdown(
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;">'
                f'<span style="color:{icon_color};font-weight:700;">{icon}</span> '
                f'<span style="color:#e6e6e6;font-weight:600;">#{num:02d}</span> '
                f'<span style="color:#ffcc00;">— {title}</span>'
                f'<div style="color:#9a9a9a;font-size:10px;margin-left:14px;'
                f'margin-top:2px;">{desc}</div></div>',
                unsafe_allow_html=True,
            )

    def _render_dashboard(
        self, inst_name, display, direction, verdict_label, chip_kind, prog_color,
        checked, pct, critical_ok, atr14, atr20, atr_data, sl_pips, tp1_pips,
        tp2_pips, risk,
    ) -> None:
        db_cfg = self._db_cfg()

        # Daily loss limit banner
        dll = {"losses_today": 0, "limit": 2, "blocked": False}
        if st.session_state.db_ok:
            try:
                dll = cached_daily_losses(
                    db_cfg.host, db_cfg.port, db_cfg.dbname,
                    db_cfg.user, db_cfg.password, max_losses=2,
                )
            except Exception:
                pass
        self._render_loss_banner(dll)

        # Correlation warnings
        if st.session_state.db_ok:
            try:
                opens = cached_load_open(
                    db_cfg.host, db_cfg.port, db_cfg.dbname,
                    db_cfg.user, db_cfg.password,
                )
                warns = CorrelationService.check_exposure(opens, direction, inst_name)
                if warns:
                    self._render_corr_warnings(warns)
            except Exception:
                pass

        # Trade verdict panel
        if checked >= 16 and critical_ok:
            verdict_msg = f"All {checked}/18 checks passed + critical clear — edge confirmed"
        elif not critical_ok:
            miss = [i for i in CRITICAL_CHECKS if not st.session_state[f"check_{i}"]]
            verdict_msg = (
                "Critical path incomplete: "
                + ", ".join(f"#{i}" for i in miss)
                + " — no entry"
            )
        elif pct >= 55:
            verdict_msg = "Building confluence — complete critical checks 11–16 before entry"
        else:
            verdict_msg = "Insufficient confirmation — stay flat, wait for setup"

        critical_badge = "● CRITICAL CLEAR" if critical_ok else "○ CRITICAL INCOMPLETE"
        verdict_body = (
            f'<div style="text-align:center;padding:6px 0 8px 0;">'
            f'<div style="font-size:28px;color:{prog_color};margin-bottom:4px;">'
            f'{verdict_label}</div>'
            f'<div style="font-size:11px;color:#9a9a9a;margin-bottom:6px;">'
            f'{checked}/18 PASSED · {critical_badge}</div>'
            f'<div style="margin-top:6px;font-size:11px;color:#e6e6e6;">'
            f'{verdict_msg}</div></div>'
        )
        Panel("TRADE VERDICT", tag="LIVE", body_html=verdict_body).show()

        # Levels card
        levels_body = render_data_rows([
            DataRow("ATR (14)",
                    f"{atr14:.5f} | {atr_data['atr14_pips'] if atr_data else '—'} pips",
                    "amber"),
            DataRow("ATR (20)",
                    f"{atr20:.5f} | {atr_data['atr20_pips'] if atr_data else '—'} pips",
                    "amber"),
            DataRow("Stop Loss (1.5×ATR14)", f"{sl_pips} pips", "red"),
            DataRow("TP1 (2:1 R:R)", f"{tp1_pips} pips", "green"),
            DataRow("TP2 (3:1 R:R)", f"{tp2_pips} pips", "green"),
            DataRow("Lot Size", f"{risk.lot_size:.2f} lots", "cyan"),
            DataRow("Risk Amount",
                    f"${risk.risk_amount:.2f} ({st.session_state.risk_pct}%)",
                    "amber"),
        ])
        Panel(f"LEVELS · {display}", tag="AUTO", body_html=levels_body).show()

        # Radar
        cats = ["Macro", "News", "Corr", "ATR", "WEMA", "WRSI", "WSwg",
                "Daily", "MACD", "Sess", "4H", "Conf", "15Cdl",
                "15Sig", "SLStr", "R:R", "PartTP", "PosSz"]
        vals = [1 if st.session_state[f"check_{i}"] else 0 for i in range(1, 19)]
        with Panel("CONFLUENCE RADAR", tag="18 PTS").context():
            st.plotly_chart(
                ChartBuilder.radar(cats, vals),
                width="stretch",
                config=dict(displayModeBar=False),
            )

        # Section breakdown
        secs = {
            "Filter the Day (1–4)": [1, 2, 3, 4],
            "Weekly Bias (5–7)": [5, 6, 7],
            "Daily Confirm (8–10)": [8, 9, 10],
            "4H Zone (11–12)": [11, 12],
            "15M Trigger (13–14)": [13, 14],
            "Risk & Execute (15–18)": [15, 16, 17, 18],
        }
        with Panel("SECTION BREAKDOWN").context():
            for name, ids in secs.items():
                n = sum(st.session_state[f"check_{i}"] for i in ids)
                t = len(ids)
                p = int(n / t * 100)
                c = T.GREEN if p == 100 else T.YELLOW if p >= 50 else T.RED
                st.markdown(
                    f'<div style="margin-bottom:6px;">'
                    f'<div style="display:flex;justify-content:space-between;'
                    f'font-family:\'JetBrains Mono\',monospace;font-size:10px;'
                    f'margin-bottom:2px;">'
                    f'<span style="color:#e6e6e6;">{name}</span>'
                    f'<span style="color:{c};font-weight:700;">{n}/{t}</span></div>'
                    f'{ProgressBar(p, color=c).render()}</div>',
                    unsafe_allow_html=True,
                )

        # Notes
        with Panel("TRADE NOTES").context():
            st.session_state.notes = st.text_area(
                "", value=st.session_state.notes,
                placeholder="Key levels, bias, news context…",
                height=80, label_visibility="collapsed",
            )

        # Save button
        if st.button("◆ SAVE SETUP TO POSTGRES", type="primary", width="stretch"):
            self._save_setup(
                db_cfg, inst_name, display, direction, verdict_label, chip_kind,
                checked, atr14, atr20, sl_pips, tp1_pips, tp2_pips, risk, dll,
            )

    @staticmethod
    def _render_loss_banner(dll: dict) -> None:
        limit = dll["limit"]
        cnt = dll["losses_today"]
        if dll["blocked"]:
            txt = f'🚫 DAILY LOSS LIMIT HIT — {cnt}/{limit} losses · no new trades'
            color = T.RED
        elif cnt == limit - 1:
            txt = f'⚠ {cnt}/{limit} LOSSES — 1 more triggers daily limit'
            color = T.YELLOW
        elif st.session_state.db_ok:
            txt = f'✓ {cnt}/{limit} LOSSES — daily limit clear'
            color = T.GREEN
        else:
            return
        st.markdown(
            f'<div style="border:1px solid {color};color:{color};padding:6px 12px;'
            f'font-family:\'JetBrains Mono\',monospace;font-size:11px;'
            f'letter-spacing:0.1em;margin-bottom:10px;text-align:center;'
            f'background:rgba(0,0,0,0.4);">{txt}</div>',
            unsafe_allow_html=True,
        )

    @staticmethod
    def _render_corr_warnings(warns: list) -> None:
        items = "".join(
            f'<div style="margin:3px 0;font-size:11px;color:#ffcc00;">'
            f'⚠ <b>{w["group"]}</b><br>'
            f'<span style="color:#9a9a9a;">Also open: {w["conflicting"]}</span></div>'
            for w in warns
        )
        Panel("CORRELATED EXPOSURE", tag="WARN", body_html=items).show()

    @staticmethod
    def _db_cfg() -> DBConfig:
        return DBConfig.from_mapping({
            "host": st.session_state.db_host,
            "port": st.session_state.db_port,
            "dbname": st.session_state.db_name,
            "user": st.session_state.db_user,
            "password": st.session_state.db_pass,
        })

    @staticmethod
    def _save_setup(
        db_cfg, inst_name, display, direction, verdict_label, chip_kind,
        checked, atr14, atr20, sl_pips, tp1_pips, tp2_pips, risk, dll,
    ) -> None:
        if not st.session_state.db_ok:
            st.error("❌ Not connected to PostgreSQL.")
            return
        if dll["blocked"]:
            st.error(
                f"🚫 Daily loss limit hit "
                f"({dll['losses_today']}/{dll['limit']}) — no new trades."
            )
            return
        checks_detail = {
            f"check_{i}": st.session_state[f"check_{i}"]
            for i in range(1, CHECKS_TOTAL + 1)
        }
        row = {
            "logged_at": datetime.now(),
            "instrument": display,
            "ticker": INSTRUMENTS[inst_name]["ticker"],
            "direction": direction,
            "session": st.session_state.session,
            "score": f"{checked}/{CHECKS_TOTAL}",
            "verdict": verdict_label,
            "atr14": atr14, "atr20": atr20,
            "sl_pips": sl_pips, "tp1_pips": tp1_pips, "tp2_pips": tp2_pips,
            "lot_size": risk.lot_size, "risk_amount": round(risk.risk_amount, 2),
            "rr_tp1": risk.rr_tp1, "rr_tp2": risk.rr_tp2,
            "account_bal": st.session_state.account_bal,
            "risk_pct": st.session_state.risk_pct,
            "checks_passed": checked, "checks_total": CHECKS_TOTAL,
            "checks_detail": json.dumps(checks_detail),
            "notes": st.session_state.notes,
        }
        try:
            pooled_repository(db_cfg).save_setup(row)
            clear_read_caches()  # journal / stats / open-trades must re-query
            st.success(
                f"✅ Saved — {display} · {verdict_label} · "
                f"{datetime.now().strftime('%H:%M:%S')}"
            )
        except Exception as e:
            st.error(f"❌ Save failed: {e}")

    # ── performance / journal sections (preserved logic, restyled) ────────
    def _render_performance_section(self, risk) -> None:
        st.markdown("")
        Panel(
            "PERFORMANCE STATS — LAST 20 CLOSED TRADES",
            tag="ANALYTICS",
            body_html="",
        ).show()
        if not st.session_state.db_ok:
            st.info("◇ Connect to PostgreSQL in the sidebar to view stats.")
            return
        try:
            cfg = self._db_cfg()
            stats = cached_performance_stats(
                cfg.host, cfg.port, cfg.dbname, cfg.user, cfg.password, n=20,
            )
        except Exception as e:
            st.error(f"❌ Could not load stats: {e}")
            return
        if not stats:
            st.info("◇ No closed trades yet.")
            return
        wr = stats["win_rate"]
        target = 66.0
        wr_color = "green" if wr >= target else "red"
        cells = [
            MetricCell("Win Rate", f"{wr:.1f}%", wr_color),
            MetricCell("W / Total", f"{stats['wins']}/{stats['total']}", "white"),
            MetricCell("Expectancy", f"{stats['expectancy']:+.2f}R",
                       "green" if stats["expectancy"] > 0 else "red"),
            MetricCell("Profit Factor", f"{stats['profit_factor']:.2f}",
                       "green" if stats["profit_factor"] >= 1.5 else "yellow"),
        ]
        st.markdown(
            '<div class="bb-mosaic bb-mosaic-4">'
            + "".join(c.render() for c in cells)
            + "</div>",
            unsafe_allow_html=True,
        )

    def _render_close_trade_section(self) -> None:
        st.markdown("")
        Panel("CLOSE TRADE — RECORD OUTCOME", tag="WRITE", body_html="").show()
        if not st.session_state.db_ok:
            st.info("◇ Connect to PostgreSQL to close trades.")
            return
        try:
            cfg = self._db_cfg()
            opens = cached_load_open(
                cfg.host, cfg.port, cfg.dbname, cfg.user, cfg.password,
            )
        except Exception as e:
            st.error(f"❌ Could not load open trades: {e}")
            return
        if not opens:
            st.info("◇ No open trades to close.")
            return
        labels = {
            f"#{t['id']} · {t['instrument']} {t['direction']} · "
            f"logged {str(t['logged_at'])[:16]}": t["id"]
            for t in opens
        }
        chosen_label = st.selectbox("Open trade", list(labels.keys()),
                                    key="close_trade_select")
        chosen_id = labels[chosen_label]
        chosen = next(t for t in opens if t["id"] == chosen_id)

        cf1, cf2, cf3 = st.columns(3)
        with cf1:
            ct_entry = st.number_input("Entry Price", value=0.0,
                                       format="%.5f", step=0.00001, key="ct_entry")
        with cf2:
            ct_close = st.number_input("Close Price", value=0.0,
                                       format="%.5f", step=0.00001, key="ct_close")
        with cf3:
            ct_pips = st.number_input("Pips Gained", value=0.0,
                                      step=0.1, key="ct_pips")
        cf4, cf5 = st.columns(2)
        with cf4:
            ct_outcome = st.selectbox("Outcome", ["WIN", "LOSS", "BE"], key="ct_outcome")
        with cf5:
            sl_p = chosen.get("sl_pips") or 1.0
            ct_r = round(ct_pips / sl_p, 2) if sl_p else 0.0
            MetricCell("R Multiple", f"{ct_r:+.2f}R",
                       color="green" if ct_r >= 0 else "red").show()

        if st.button("◆ SAVE TRADE OUTCOME", type="primary",
                     width="stretch", key="save_outcome_btn"):
            try:
                pooled_repository(self._db_cfg()).close_trade(
                    chosen_id, ct_entry, ct_close, ct_pips, ct_r, ct_outcome,
                )
                # Closing a trade changes P/L, stats, open-trades and the journal.
                clear_read_caches()
                st.success(
                    f"✅ Trade #{chosen_id} closed — {ct_outcome} · "
                    f"{ct_r:+.2f}R · {ct_pips:+.1f} pips"
                )
                st.rerun()
            except Exception as e:
                st.error(f"❌ Close failed: {e}")

    def _render_trade_log_section(self) -> None:
        st.markdown("")
        Panel("TRADE LOG — POSTGRESQL", tag="READ", body_html="").show()
        if not st.session_state.db_ok:
            st.info("◇ Connect to PostgreSQL to view saved trades.")
            return
        try:
            cfg = self._db_cfg()
            trades = cached_load_setups(
                cfg.host, cfg.port, cfg.dbname, cfg.user, cfg.password, limit=100,
            )
        except Exception as e:
            st.error(f"❌ Could not load trades: {e}")
            return
        if not trades:
            st.info("◇ No trade setups saved yet.")
            return
        df_log = pd.DataFrame(trades)
        df_log["logged_at"] = pd.to_datetime(df_log["logged_at"]).dt.strftime("%Y-%m-%d %H:%M")
        if "checks_detail" in df_log.columns:
            df_log["checks_detail"] = df_log["checks_detail"].apply(
                lambda x: str(x) if x else ""
            )
        st.dataframe(df_log, width="stretch")
