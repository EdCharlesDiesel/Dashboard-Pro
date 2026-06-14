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
from src.services import RiskService
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
        try:
            df = yf.download(ticker, period=f"{days}d", interval="1d",
                             progress=False, auto_adjust=True)
            if df.empty:
                return pd.DataFrame()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        except Exception:
            return pd.DataFrame()

    @staticmethod
    @st.cache_data(ttl=300, show_spinner=False)
    def four_hour(ticker: str) -> pd.DataFrame:
        try:
            end = datetime.now(pytz.utc)
            start = end - timedelta(days=90)
            df = yf.download(ticker, start=start, end=end, interval="1h",
                             progress=False, auto_adjust=True)
            if df.empty:
                return pd.DataFrame()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
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
        try:
            df = yf.download(ticker, period="2y", interval="1d",
                             progress=False, auto_adjust=True)
            if df.empty:
                return pd.DataFrame()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
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
        st.session_state.setdefault("sr_direction", "LONG")
        st.session_state.setdefault("sr_min_score", 5)
        st.session_state.setdefault("sr_rr_ratio", 2.0)
        st.session_state.setdefault("sr_pairs", INSTRUMENTS.keys())
        # Shared with the checklist so account settings carry across pages.
        st.session_state.setdefault("account_bal", 10000.0)
        st.session_state.setdefault("risk_pct", 1.0)
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
            if st.button("All", use_container_width=True):
                st.session_state.sr_pairs = INSTRUMENTS.keys()
        with col2:
            if st.button("Majors", use_container_width=True):
                st.session_state.sr_pairs = [
                    "EUR/USD", "GBP/USD", "AUD/USD", "NZD/USD",
                    "USD/JPY", "USD/CHF", "USD/CAD",
                ]
        with col3:
            if st.button("Metals", use_container_width=True):
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
        st.session_state.account_bal = st.number_input(
            "Account balance ($)", min_value=0.0, step=500.0,
            value=float(st.session_state.account_bal),
        )
        st.session_state.risk_pct = st.slider(
            "Risk % / trade", 0.25, 5.0,
            float(st.session_state.risk_pct), 0.25,
        )
        st.divider()
        if st.button("◆ RESCAN ALL PAIRS", use_container_width=True, type="primary"):
            st.cache_data.clear()
            st.rerun()
        st.caption(f"◷ {datetime.now().strftime('%H:%M')} · TTL 5min")

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
        results = self._scan(pairs, directions, min_score)

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

    @staticmethod
    def _scan(pairs, directions, min_score) -> list:
        results = []
        pairs_list = [(p, INSTRUMENTS[p]) for p in pairs if p in INSTRUMENTS]
        total = len(pairs_list) * len(directions)
        done = 0
        prog = st.progress(0, text="Scoring pairs…")
        for pair_name, info in pairs_list:
            for d in directions:
                done += 1
                prog.progress(done / total, text=f"Scoring {pair_name} {d}…")
                try:
                    r = _SetupRankerDataFeed.score(pair_name, info, d)
                    if r["score"] >= min_score:
                        results.append(r)
                except Exception:
                    pass
        prog.empty()
        results.sort(key=lambda x: -x["score"])
        return results

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
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

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
        st.plotly_chart(fig, use_container_width=True,
                        config=dict(displayModeBar=False))
