"""Trend-Following Signals mode — Bloomberg-styled signal scanner."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List

import pandas as pd
import streamlit as st


def _format_age(delta) -> str:
    """Human age string ('5m ago', '1h 12m ago', '2d 3h ago') for a signal's
    last-bar timestamp, so a morning signal isn't mistaken for a fresh one."""
    total_min = max(0, int(delta.total_seconds() // 60))
    days, rem_min = divmod(total_min, 1440)
    hours, mins = divmod(rem_min, 60)
    if days:
        return f"{days}d {hours}h ago"
    if hours:
        return f"{hours}h {mins}m ago"
    return f"{mins}m ago"

from src.indicators import TrendSignalEvaluator
from src.instruments import INSTRUMENTS, TREND_COMMODITIES, TREND_TIMEFRAMES
from src.services.signal_store import persist_signals
from src.ui.charts import ChartBuilder
from src.ui.components import (
    Chip, CommandBar, DataRow, MetricCell, Panel, ProgressBar,
    TickerStrip, render_data_rows, render_metric_row,
)
from src.ui.theme import BloombergTheme as T


class TrendController:
    """Owns the rendering for TREND SIGNALS mode."""

    def render(self) -> None:
        selected: List[str] = st.session_state.trend_selected_pairs
        if not selected:
            st.warning("⚠ Select at least one market in the sidebar.")
            return

        tf = st.session_state.trend_timeframe
        min_conds = st.session_state.trend_min_conds
        interval, period, resample = TREND_TIMEFRAMES[tf]

        CommandBar(
            label="TREND",
            cells=[
                (f"TF {tf.upper()}", ""),
                (f"MIN {min_conds}/6", "cyan"),
                ("50/200 EMA · RSI(14) · MACD · ADX", "amber"),
                (datetime.now().strftime("%H:%M UTC"), "green"),
            ],
        ).show()

        results = self._scan_markets(selected, interval, period, resample, min_conds)
        self._persist_signals(results)
        self._render_ticker_strip(results)
        self._render_signal_grid(results)
        self._render_detail_panel(results, selected)
        self._render_strategy_reference()
        self._record_alert_history(results)
        self._render_alert_history()

    @staticmethod
    def _scan_markets(
        selected: List[str], interval: str, period: str,
        resample, min_conds: int,
    ) -> Dict[str, dict]:
        results: Dict[str, dict] = {}
        progress = st.progress(0, text="Fetching market data…")
        for idx, pair in enumerate(selected):
            ticker = INSTRUMENTS[pair]["ticker"]
            with st.spinner(f"Loading {pair}…"):
                df = TrendSignalEvaluator.fetch(ticker, interval, period, resample)
            progress.progress((idx + 1) / len(selected), text=f"Processed {pair}")
            if df is None or df.empty:
                results[pair] = {"error": True}
                continue
            sig = TrendSignalEvaluator.evaluate(df, min_conds)
            last = df.iloc[-1]
            bar_time = df.index[-1].to_pydatetime()
            results[pair] = {
                "error": False, "df": df,
                "signal": sig.label, "score": sig.score,
                "max_score": sig.max_score, "conds": sig.conditions,
                "direction": sig.direction,
                "close": float(last["Close"]),
                "ema50": float(last["EMA50"]),
                "ema200": float(last["EMA200"]),
                "rsi": float(last["RSI"]),
                "macd": float(last["MACD"]),
                "adx": float(last["ADX"]),
                # bar_time is tz-naive UTC (market_cache normalizes it) —
                # compare against naive utcnow() so a morning 4H signal isn't
                # mistaken for a fresh one hours later.
                "bar_time": bar_time,
                "age": _format_age(datetime.utcnow() - bar_time),
            }
        progress.empty()
        return results

    def _render_ticker_strip(self, results: Dict[str, dict]) -> None:
        items = []
        for pair, r in results.items():
            if r.get("error"):
                items.append((INSTRUMENTS[pair]["ticker"], "ERR", None))
                continue
            delta = (r["close"] - r["ema50"]) / r["ema50"] * 100
            items.append((INSTRUMENTS[pair]["ticker"], f"{r['close']:.5f}", delta))
        TickerStrip(items).show()

    def _render_signal_grid(self, results: Dict[str, dict]) -> None:
        st.markdown(
            '<div style="color:#00ff41;font-family:\'JetBrains Mono\',monospace;'
            'font-size:12px;letter-spacing:0.18em;margin:14px 0 6px 0;'
            'text-transform:uppercase;">▸ LIVE SIGNALS</div>',
            unsafe_allow_html=True,
        )
        cells_html = []
        kind_map = {
            "STRONG_BUY":  ("● STRONG BUY",  "go",    T.GREEN),
            "BUY":         ("▲ BUY",         "go",    T.GREEN),
            "STRONG_SELL": ("● STRONG SELL", "no",    T.RED),
            "SELL":        ("▼ SELL",        "no",    T.RED),
            "NEUTRAL":     ("◆ NEUTRAL",     "info",  T.GREY),
        }
        for pair, r in results.items():
            if r.get("error"):
                cells_html.append(
                    f'<div class="bb-panel"><div class="bb-panel-header">'
                    f'<span>{pair}</span><span class="bb-tag">ERROR</span></div>'
                    f'<div class="bb-panel-body" style="color:#ff3344;">'
                    f'Failed to load</div></div>'
                )
                continue
            label, kind, color = kind_map.get(r["direction"], kind_map["NEUTRAL"])
            if r["direction"] == "NEUTRAL" and r["adx"] <= 25:
                # Distinguish "ADX gate failed" from a genuine flat-market
                # NEUTRAL — the whole point of the gate is visible staleness,
                # not a silently identical grey badge.
                label = "◇ NO TREND"
            tag_text = "METAL" if pair in TREND_COMMODITIES else "FOREX"
            tag_color = T.YELLOW if pair in TREND_COMMODITIES else T.CYAN
            pct_bar = int(r["score"] / r["max_score"] * 100)
            body = (
                f'<div style="font-family:\'JetBrains Mono\',monospace;'
                f'font-size:18px;font-weight:700;color:{color};'
                f'letter-spacing:0.08em;">{label}</div>'
                f'<div style="font-family:\'JetBrains Mono\',monospace;'
                f'font-size:11px;color:#e6e6e6;margin-top:4px;">'
                f'PX {r["close"]:.5f}</div>'
                f'<div style="font-family:\'JetBrains Mono\',monospace;'
                f'font-size:10px;color:#9a9a9a;margin-top:2px;">'
                f'COND {r["score"]}/{r["max_score"]} · ADX {r["adx"]:.1f} '
                f'({"TREND" if r["adx"] > 25 else "RANGE"})</div>'
                f'<div style="font-family:\'JetBrains Mono\',monospace;'
                f'font-size:9px;color:#6a6a6a;margin-top:2px;">{r["age"]}</div>'
                f'<div style="margin-top:6px;">{ProgressBar(pct_bar, color=color).render()}</div>'
            )
            cells_html.append(
                f'<div class="bb-panel"><div class="bb-panel-header">'
                f'<span>{pair}</span>'
                f'<span class="bb-tag" style="color:{tag_color};">{tag_text}</span>'
                f'</div><div class="bb-panel-body">{body}</div></div>'
            )
        st.markdown(
            f'<div class="bb-mosaic bb-mosaic-4">{"".join(cells_html)}</div>',
            unsafe_allow_html=True,
        )

    def _render_detail_panel(self, results: Dict[str, dict], selected: List[str]) -> None:
        valid_pairs = [p for p in selected if not results.get(p, {}).get("error")]
        if not valid_pairs:
            st.error("No valid data. Try refreshing.")
            return
        default = next(
            (p for p in valid_pairs if results[p]["direction"] != "NEUTRAL"),
            valid_pairs[0],
        )
        chosen = st.selectbox(
            "Inspect pair:", valid_pairs, index=valid_pairs.index(default),
        )
        r = results[chosen]

        st.caption(
            f"{r['signal']} · {r['score']}/{r['max_score']} conditions · "
            f"signal bar {r['bar_time'].strftime('%Y-%m-%d %H:%M UTC')} "
            f"({r['age']}) — confirm it's still fresh before acting on it."
        )

        render_metric_row([
            MetricCell("Close",  f"{r['close']:.5f}", "white"),
            MetricCell(
                "50 EMA", f"{r['ema50']:.5f}", "amber",
                delta=f"{((r['close'] - r['ema50']) / r['ema50'] * 100):+.2f}%",
            ),
            MetricCell(
                "200 EMA", f"{r['ema200']:.5f}", "amber",
                delta=f"{((r['close'] - r['ema200']) / r['ema200'] * 100):+.2f}%",
            ),
            MetricCell(
                "RSI (14)", f"{r['rsi']:.1f}",
                "red" if r["rsi"] > 70 else "green" if r["rsi"] < 30 else "cyan",
                delta="OB" if r["rsi"] > 70 else "OS" if r["rsi"] < 30 else "MID",
            ),
            MetricCell("MACD", f"{r['macd']:.5f}", "cyan"),
            MetricCell(
                "ADX", f"{r['adx']:.1f}",
                "green" if r["adx"] > 25 else "yellow",
                delta="TREND" if r["adx"] > 25 else "RANGE",
            ),
        ])

        with Panel(f"CHART · {chosen}", tag=st.session_state.trend_timeframe).context():
            st.plotly_chart(
                ChartBuilder.trend_panel(r["df"], chosen),
                width="stretch",
            )

        if r["conds"]:
            with Panel(f"SIGNAL CONDITIONS · {chosen}", tag="6 PTS").context():
                col_a, col_b = st.columns(2)
                items = list(r["conds"].items())
                half = (len(items) + 1) // 2
                for col, chunk in zip([col_a, col_b], [items[:half], items[half:]]):
                    with col:
                        for cond, met in chunk:
                            icon = "✓" if met else "✗"
                            color = T.GREEN if met else T.RED
                            st.markdown(
                                f'<div style="font-family:\'JetBrains Mono\',monospace;'
                                f'font-size:11px;margin:3px 0;color:#e6e6e6;">'
                                f'<span style="color:{color};font-weight:700;">{icon}</span> '
                                f'{cond}</div>',
                                unsafe_allow_html=True,
                            )

    @staticmethod
    def _render_strategy_reference() -> None:
        with st.expander("◇ STRATEGY REFERENCE — TREND FOLLOWING RULES"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
**BUY SETUP**
- Price **above** 200 EMA
- 50 EMA **above** 200 EMA (Golden Cross)
- Price pulling back to / holding 50 EMA
- RSI **45–70** (bullish zone)
- MACD **above** signal line
- ADX **> 25** (trend strength)
""")
            with col2:
                st.markdown("""
**SELL SETUP**
- Price **below** 200 EMA
- 50 EMA **below** 200 EMA (Death Cross)
- Price pulling back to / below 50 EMA
- RSI **30–45** (bearish zone)
- MACD **below** signal line
- ADX **> 25** (trend strength)
""")

    # Alert/persist floor: STRONG only. A plain BUY/SELL (4/6, ADX gate passed
    # but not all conditions) still displays on the grid, but doesn't fire an
    # alert or auto-save — otherwise the floor is asymmetric per-pair (one
    # pair alerting at 4/6 while another needs 5/6 to say the same thing) and
    # the alert history becomes noise. STRONG already implies the ADX gate
    # passed (evaluate() gates before scoring), so this is "5/6 with trend
    # confirmed," not just "5 points."
    _ALERT_DIRECTIONS = ("STRONG_BUY", "STRONG_SELL")

    @classmethod
    def _persist_signals(cls, results: Dict[str, dict]) -> None:
        """Persist STRONG BUY/SELL signals to trade_setups (deduped,
        source='trend_signals'). Weaker/NEUTRAL/error rows are skipped."""
        signals = []
        for pair, r in results.items():
            if r.get("error"):
                continue
            direction = r.get("direction")
            if direction not in cls._ALERT_DIRECTIONS:
                continue
            signals.append({
                "pair": pair,
                "bias": "Long" if direction == "STRONG_BUY" else "Short",
                "entry": r.get("close"),
                "conviction": r.get("signal"),
                "thesis": (f"Trend Signals — {r.get('signal')} · "
                           f"{r.get('score')}/{r.get('max_score')} conditions · "
                           f"RSI {r.get('rsi', 0):.0f} ADX {r.get('adx', 0):.0f}"),
            })
        persist_signals("trend_signals", signals)

    @classmethod
    def _record_alert_history(cls, results: Dict[str, dict]) -> None:
        st.session_state.setdefault("trend_alert_history", [])
        new_alerts = [
            (pair, r["signal"], r["direction"], r["close"])
            for pair, r in results.items()
            if not r.get("error")
            and r["direction"] in cls._ALERT_DIRECTIONS
        ]
        if (new_alerts and st.session_state.get("trend_last_alert_count", 0)
                != len(new_alerts)):
            for pair, sig, direction, price in new_alerts:
                st.session_state.trend_alert_history.append({
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
                    "pair": pair, "signal": sig, "price": f"{price:.5f}",
                })
            st.session_state.trend_alert_history = (
                st.session_state.trend_alert_history[-50:]
            )
            st.session_state.trend_last_alert_count = len(new_alerts)

    @staticmethod
    def _render_alert_history() -> None:
        with st.expander(
            f"◷ ALERT HISTORY "
            f"({len(st.session_state.trend_alert_history)} this session)"
        ):
            if st.session_state.trend_alert_history:
                df_h = pd.DataFrame(st.session_state.trend_alert_history[::-1])
                st.dataframe(df_h, hide_index=True, width="stretch")
                if st.button("Clear History", key="clear_trend_history"):
                    st.session_state.trend_alert_history = []
                    st.rerun()
            else:
                st.info("No alerts triggered yet this session.")
