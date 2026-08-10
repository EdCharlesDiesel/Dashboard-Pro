"""Today's Trades — the morning answer, on one screen.

A **view**, not a signal source. Every row here was already persisted to
`trade_setups` by the page that found it; this reads them back, aggregates the
consensus, filters against the live book and sizes what survives. Nothing is
written — so it is deliberately absent from `signal_sweep.PAGES`, for the same
reason MTF Matrix is: a display grid over already-persisted data is not an
independent read, and counting it as one would double-vote every signal.

What it answers, in the order the desk asks:
  1. What did the whole system find today?
  2. Which of those can I actually take, given what I already hold?
  3. At what size, against my real balance?
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from src.core.todays_trades import (LONG, Idea, consensus, exposure_conflict,
                                    find_conflict, net_exposure,
                                    offsetting_legs, size_idea)
from src.instruments.registry import INSTRUMENTS
from src.indicators.technical import TechnicalIndicators as TI
from src.pages_lib.base import BloombergPage, PageContext
from src.services.market_data import daily_ohlc
from src.ui.theme import BloombergTheme as T


class TodaysTradesPage(BloombergPage):
    """Consensus board for today, filtered by exposure and sized to balance."""

    def configure(self) -> PageContext:
        return PageContext(code="TDAY", title="Today's Trades", icon="🎯")

    # ── sidebar ─────────────────────────────────────────────────────────────
    def sidebar(self, ctx: PageContext) -> None:
        st.markdown("### 🎯 TODAY'S TRADES")
        st.caption("Consensus across every signal source that fired today, "
                   "filtered against your open book and sized to balance.")
        st.session_state.setdefault("tday_risk", 1.0)
        st.session_state.tday_risk = st.slider(
            "Risk per trade (%)", 0.25, 3.0, st.session_state.tday_risk, 0.25,
            help="Size is derived from the stop distance, so this is the real "
                 "dollar risk — not a fixed lot size.")
        st.session_state.setdefault("tday_min", 2)
        st.session_state.tday_min = st.slider(
            "Minimum agreeing sources", 1, 6, st.session_state.tday_min,
            help="How many independent pages must agree before it is listed.")
        st.session_state.setdefault("tday_show_blocked", True)
        st.session_state.tday_show_blocked = st.checkbox(
            "Show blocked ideas", value=st.session_state.tday_show_blocked,
            help="Ideas that stack onto something you already hold.")
        st.divider()

    # ── data ────────────────────────────────────────────────────────────────
    @staticmethod
    @st.cache_data(ttl=120, show_spinner=False)
    def _signals_today() -> List[Dict[str, Any]]:
        """Every signal row persisted today, or [] when the DB is unreachable."""
        try:
            from src.db.cache import pooled_repository
            from src.db.market_cache import _resolve_cfg
            from contextlib import closing
            cfg = _resolve_cfg()
            if cfg is None:
                return []
            repo = pooled_repository(cfg)
            with closing(repo._connect()) as conn, conn, conn.cursor() as cur:  # noqa: SLF001
                cur.execute(
                    "SELECT instrument, direction, source FROM trade_setups "
                    "WHERE logged_at::date = CURRENT_DATE")
                return [{"instrument": r[0], "direction": r[1], "source": r[2]}
                        for r in cur.fetchall()]
        except Exception:
            return []

    @staticmethod
    def _open_book() -> List[Dict[str, Any]]:
        """Open positions from the durable store (works without a terminal)."""
        try:
            from src.services.open_positions import load
            # entry_price rides along for the currency-exposure guard, which
            # needs a price per position to convert legs into one comparable
            # unit. Dropping it here is what limited the guard to name matching.
            return [{"pair": p.get("pair"), "direction": p.get("direction"),
                     "lot_size": p.get("lot_size"),
                     "entry_price": p.get("entry_price")} for p in (load() or [])]
        except Exception:
            return []

    @staticmethod
    def _balance() -> float:
        try:
            from src.services import account_state
            return float(account_state.get_balance() or 0.0)
        except Exception:
            return 0.0

    @staticmethod
    @st.cache_data(ttl=300, show_spinner=False)
    def _price_and_atr(pair: str) -> Optional[Dict[str, float]]:
        inst = INSTRUMENTS.get(pair)
        if inst is None:
            return None
        try:
            df = daily_ohlc(inst.ticker)
            if df is None or df.empty or len(df) < 20:
                return None
            atr = float(TI.atr(df["High"], df["Low"], df["Close"], 14).iloc[-1])
            return {"price": float(df["Close"].iloc[-1]), "atr": atr}
        except Exception:
            return None

    # ── body ────────────────────────────────────────────────────────────────
    def body(self, ctx: PageContext) -> None:
        rows = self._signals_today()
        if not rows:
            st.warning("No signals persisted today. Run the sweep "
                       "(`python -m src.services.signal_sweep`) or check the DB.")
            return

        book = self._open_book()
        balance = self._balance()
        risk_pct = float(st.session_state.get("tday_risk", 1.0))
        min_src = int(st.session_state.get("tday_min", 2))

        ideas = [i for i in consensus(rows) if len(i.agree) >= min_src]
        for idea in ideas:
            # Two independent guards, and the order matters only for which
            # reason gets shown: the group check names a specific position, so
            # it reads better when both fire. The exposure check catches what
            # the group table structurally cannot — see todays_trades.py.
            idea.conflict = (find_conflict(idea.pair, idea.direction, book)
                             or exposure_conflict(idea.pair, idea.direction,
                                                  book, balance))
            pa = self._price_and_atr(idea.pair)
            if pa:
                size_idea(idea, pa["price"], pa["atr"], balance, risk_pct=risk_pct)

        takeable = [i for i in ideas if i.tradeable and i.lots]
        blocked = [i for i in ideas if not i.tradeable]

        self._header(rows, ideas, takeable, blocked, book, balance)

        if not takeable:
            st.info("Nothing is takeable right now — everything either lacks "
                    "agreement or stacks onto a position you already hold. "
                    "That is a valid answer, not a broken scan.")
        else:
            st.markdown("### ✅ Takeable now")
            st.dataframe(self._table(takeable), width="stretch",
                         hide_index=True)
            total = sum(i.risk_usd or 0 for i in takeable)
            # Escape the dollar signs: Streamlit markdown treats $...$ as LaTeX,
            # so an unescaped pair silently swallows both and the text between.
            st.caption(r"Taking **all** of these risks **\${0:,.2f}** — {1:.1f}% of "
                       r"\${2:,.2f}. Size is derived from each stop, so the dollar "
                       r"risk is even across them regardless of instrument."
                       .format(total, (total / balance * 100) if balance else 0, balance))
            for idea in takeable[:3]:
                self._card(idea)

        self._probabilities(takeable)

        if blocked and st.session_state.get("tday_show_blocked", True):
            st.markdown("### ⛔ Blocked by your open book")
            st.caption("Real signals — they would just double a bet you already have.")
            st.dataframe(self._table(blocked, blocked_col=True),
                         width="stretch", hide_index=True)

        self._exposure(book, balance)

        st.caption("A view over signals already persisted by other pages — this "
                   "page writes nothing and is not part of the sweep.")

    # ── rendering ───────────────────────────────────────────────────────────
    @staticmethod
    @st.cache_data(ttl=900, show_spinner=False)
    def _gbm(pair: str):
        """Calibrated GBM for one pair. Cached: it refits the whole daily series."""
        from src.services.position_risk import calibrate
        return calibrate(pair)

    @classmethod
    def _probabilities(cls, takeable: List[Idea]) -> None:
        """Will the target be reached before the stop — and does that pay?

        Sits above the blocked table because it answers a different question
        from everything else on this page. Consensus says the *direction* is
        agreed; sizing says the position is affordable. Neither says the target
        is reachable before the stop, which is the only one of the three that
        decides whether the trade makes money.
        """
        if not takeable:
            return
        from src.services.position_risk import assess, position_from_idea, verdict

        st.markdown("### 🎲 Target before stop")
        rows: List[Dict[str, Any]] = []
        reasons: List[tuple] = []
        for idea in takeable:
            position = position_from_idea(idea)
            params = cls._gbm(idea.pair) if position is not None else None
            if position is None or params is None:
                rows.append({"Pair": idea.pair, "Side": idea.direction,
                             "P(target 1st)": "—", "P(stop 1st)": "—",
                             "R:R": "—", "Needs": "—", "Edge": "—",
                             "EV": "—", "Verdict": "no price history"})
                continue
            risk = assess(position, params, use_monte_carlo=False)
            if risk is None:
                rows.append({"Pair": idea.pair, "Side": idea.direction,
                             "P(target 1st)": "—", "P(stop 1st)": "—",
                             "R:R": "—", "Needs": "—", "Edge": "—",
                             "EV": "—", "Verdict": "unsized"})
                continue
            call = verdict(risk)
            rows.append({
                "Pair": risk.pair, "Side": risk.side,
                "P(target 1st)": "{0:.1f}%".format(risk.p_target * 100),
                "P(stop 1st)": "{0:.1f}%".format(risk.p_stop * 100),
                "R:R": "{0:.2f}".format(risk.reward_risk),
                "Needs": "{0:.1f}%".format(risk.breakeven * 100),
                "Edge": "{0:+.1f}pp".format(risk.edge_pp),
                "EV": "{0:+.2f}".format(risk.ev_usd),
                "Verdict": "✅ TAKE" if call["verdict"] == "TAKE" else "⛔ SKIP",
            })
            reasons.append((risk.pair, call["verdict"], call["reason"]))

        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption("**Edge**, not probability, is the column that decides. A 2:1 "
                   "target sits twice as far as its stop, so being stopped more "
                   "often than not is the design — what matters is whether the "
                   "hit rate beats the one the payoff requires.")
        for pair, call, reason in reasons:
            (st.success if call == "TAKE" else st.warning)(
                "**{0}** — {1}".format(pair, reason))

    @staticmethod
    def _exposure(book: List[Dict[str, Any]], balance: float) -> None:
        """What the book actually bets on, currency by currency.

        The blocked list says *this* trade is refused; this says why the book is
        shaped the way it is. Without it the guard is an oracle — it refuses a
        metal trade for a dollar reason the user cannot see anywhere on screen.
        """
        if not book or not balance:
            return
        exposure = net_exposure(book)
        if not exposure:
            return
        st.markdown("### 🌍 Net currency exposure")
        ranked = sorted(exposure.items(), key=lambda kv: -abs(kv[1]))
        frame = pd.DataFrame([{
            "Currency": ccy,
            "Side": "Long" if amount > 0 else "Short",
            "Notional (USD)": round(amount, 0),
            "x Balance": round(abs(amount) / balance, 2),
            "Over limit": "⚠️" if abs(amount) >= balance else "",
        } for ccy, amount in ranked if abs(amount) >= 1])
        st.dataframe(frame, width="stretch", hide_index=True)
        st.caption("Every position split into its two currency legs. Tickets on "
                   "different symbols can be one bet — four separate trades summed "
                   "to 2.1x the account short USD on 2026-08-10, each of which "
                   "passed the correlation check on its own.")

        for clash in offsetting_legs(book):
            st.warning("**{0}** is held both ways — long via {1}, short via {2}. "
                       "Usually accidental, and you pay spread and swap on both "
                       "legs to hold the difference.".format(
                           clash["currency"], ", ".join(clash["long"]),
                           ", ".join(clash["short"])))

    @staticmethod
    def _px(value: Optional[float]) -> str:
        """Format a price the way the desk reads it (see CLAUDE.md domain rules).

        5 decimals under 100 (FX), 3 above (JPY crosses, metals, oil). Printing
        a raw float here gives `1.87100748`, which is not a price anyone can
        type into a ticket.
        """
        if value is None:
            return "—"
        return "{0:.5f}".format(value) if abs(value) < 100 else "{0:.3f}".format(value)

    @staticmethod
    def _header(rows, ideas, takeable, blocked, book, balance) -> None:
        c = st.columns(5)
        c[0].metric("Signals today", len(rows))
        c[1].metric("Instruments", len({r["instrument"] for r in rows}))
        c[2].metric("✅ Takeable", len(takeable))
        c[3].metric("⛔ Blocked", len(blocked))
        c[4].metric("Balance", "${0:,.0f}".format(balance) if balance else "—")
        if book:
            st.caption("Open book: " + " · ".join(
                "{0} {1} {2}".format(p["pair"], str(p["direction"]).lower(),
                                     p.get("lot_size") or "")
                for p in book))
        else:
            st.caption("Open book: **flat** — no exposure conflicts, whole board available.")

    @staticmethod
    def _table(ideas: List[Idea], blocked_col: bool = False) -> pd.DataFrame:
        recs = []
        for i in ideas:
            rec = {
                "Pair": i.pair,
                "Side": "▲ LONG" if i.direction == LONG else "▼ SHORT",
                "Agree": len(i.agree),
                "Against": len(i.against),
                "Entry": TodaysTradesPage._px(i.entry),
                "Stop": TodaysTradesPage._px(i.stop),
                "Target": TodaysTradesPage._px(i.target),
                "Stop (pips)": i.sl_pips,
                "Lots": i.lots,
                "Risk $": i.risk_usd,
                "Risk %": i.risk_pct,
            }
            if blocked_col:
                rec["Blocked because"] = i.conflict
            recs.append(rec)
        return pd.DataFrame(recs)

    @staticmethod
    def _card(i: Idea) -> None:
        colour = T.AMBER if i.direction == LONG else "#ff3344"
        arrow = "▲" if i.direction == LONG else "▼"
        st.markdown(
            "<div style='border-left:3px solid {c};padding:8px 12px;margin:8px 0;'>"
            "<b style='color:{c};font-size:15px'>{a} {p} {d}</b>"
            "<span style='color:#9a9a9a'> — {n} sources agree"
            "{ag}</span><br>"
            "<span style='color:#9a9a9a;font-size:12px'>entry <b>{e}</b> · "
            "stop <b>{s}</b> ({sp}p) · target <b>{t}</b> · "
            "<b>{l} lots</b> = ${r} ({rp}%)</span><br>"
            "<span style='color:#6a6a6a;font-size:11px'>{srcs}</span></div>".format(
                c=colour, a=arrow, p=i.pair, d=i.direction.upper(),
                n=len(i.agree),
                ag="" if not i.against else ", {0} against".format(len(i.against)),
                e=TodaysTradesPage._px(i.entry), s=TodaysTradesPage._px(i.stop),
                sp=i.sl_pips, t=TodaysTradesPage._px(i.target),
                l=i.lots, r=i.risk_usd, rp=i.risk_pct,
                srcs=", ".join(i.agree)),
            unsafe_allow_html=True)
