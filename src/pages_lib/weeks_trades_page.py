"""This Week's Trades — what the system found across the week.

Rendering only. Every number here comes from ``src/core/weeks_trades``, which is
pure and tested; this module decides layout and nothing else.

**Not a seven-day copy of Today's Trades.** That page answers *"what is live and
takeable right now"* and reads a 45-day window to do it, because `consensus()`
holds each signal for its own horizon. This one answers *"what happened"*: which
pairs kept reappearing, on how many distinct days, driven by which sources, and
which ones the system changed its mind about.

The signal read is imported from ``TodaysTradesPage`` rather than rewritten —
one query, one source of truth for what "a persisted signal" means.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from src.core.weeks_trades import by_day, pair_activity, week_start
from src.pages_lib.base import BloombergPage, PageContext
from src.pages_lib.todays_trades_page import TodaysTradesPage


class WeeksTradesPage(BloombergPage):
    """The week's signal flow: persistence, conviction, and changes of mind."""

    def configure(self) -> PageContext:
        return PageContext(code="WEEK", title="This Week's Trades", icon="📆")

    def sidebar(self, ctx: PageContext) -> None:
        st.markdown("### 📆 THIS WEEK'S TRADES")
        st.caption(
            "Every signal persisted since Monday 00:00 UTC, grouped by pair and "
            "by day. This is the review view — Today's Trades is the one that "
            "tells you what to take now.")

    # ── data ────────────────────────────────────────────────────────────────
    @staticmethod
    def _rows() -> List[Dict[str, Any]]:
        """The persisted signal rows, via the existing reader.

        Deliberately not a second query: `TodaysTradesPage._signals_today`
        already encodes the sargable range predicate and the 45-day horizon, and
        a parallel implementation here would drift from it silently.
        """
        return TodaysTradesPage._signals_today()  # noqa: SLF001

    # ── render ──────────────────────────────────────────────────────────────
    def body(self, ctx: PageContext) -> None:
        now = datetime.now(timezone.utc)
        start = week_start(now)

        st.markdown(f"#### Week of {start:%a %d %b %Y} → {now:%a %d %b %Y}")

        rows = self._rows()
        if not rows:
            # Say which failure this is. A blank page cannot distinguish "no
            # signals" from "the database is unreachable", and those need
            # opposite responses.
            st.warning(
                "No persisted signals available. Either nothing has been logged "
                "yet, or the database is unreachable — check the PostgreSQL "
                "connection in the sidebar, then run the sweep.")
            return

        activity = pair_activity(rows, now)
        if not activity:
            st.info(
                f"Signals exist, but none since {start:%a %d %b}. "
                "The week is quiet so far — this is a real reading, not an error.")
            return

        self._headline(activity, rows, now)
        st.divider()
        self._activity_table(activity)
        st.divider()
        self._daily_breakdown(rows, now)

    @staticmethod
    def _headline(activity, rows, now) -> None:
        flipped = [a for a in activity if a.flipped]
        days = len(by_day(rows, now))
        cols = st.columns(4)
        cols[0].metric("Pairs seen", len(activity))
        cols[1].metric("Active days", days)
        cols[2].metric("Most persistent",
                       activity[0].pair if activity else "—",
                       f"{activity[0].days_seen}d" if activity else None)
        # "Self-reversed", not "changed side". A pair where two sources simply
        # disagree is normal and was previously counted here, which reported 22
        # of 27 pairs as reversing when 5 had. Only a source contradicting
        # *itself* is worth an alarm.
        cols[3].metric("Self-reversed", len(flipped),
                       delta=None if not flipped else ", ".join(
                           a.pair for a in flipped[:3]),
                       delta_color="off")

    @staticmethod
    def _activity_table(activity) -> None:
        st.markdown("##### By pair")
        frame = pd.DataFrame([{
            "Pair": a.pair,
            "Days": a.days_seen,
            "Long": a.longs,
            "Short": a.shorts,
            # The split is the reading: 10/3 is conviction with one dissenter,
            # 3/3 is a coin toss. A boolean erased that distinction.
            "Split L/S": f"{a.longs}/{a.shorts}",
            "Reversed": ", ".join(sorted(a.reversing_sources)) or "",
            "Sources": ", ".join(sorted(a.sources)) or "—",
            "First": a.first_seen.strftime("%a %H:%M") if a.first_seen else "—",
            "Last": a.last_seen.strftime("%a %H:%M") if a.last_seen else "—",
        } for a in activity])
        st.dataframe(frame, hide_index=True, width="stretch")
        st.caption(
            "Days counts *distinct days*, not rows — three sources firing on one "
            "morning is one day of conviction, not three. **Split** is how the "
            "sources divided; they disagree routinely and that is what the "
            "consensus board resolves. **Reversed** names only sources that "
            "contradicted themselves during the week — that is the one worth "
            "looking at.")

    @staticmethod
    def _daily_breakdown(rows, now) -> None:
        st.markdown("##### By day")
        grouped = by_day(rows, now)
        for day, day_rows in reversed(list(grouped.items())):
            pairs = sorted({r.get("instrument") for r in day_rows if r.get("instrument")})
            with st.expander(f"{day:%A %d %b} — {len(day_rows)} signal(s), "
                             f"{len(pairs)} pair(s)"):
                st.dataframe(pd.DataFrame([{
                    "Pair": r.get("instrument"),
                    "Direction": r.get("direction"),
                    "Source": r.get("source"),
                    "Logged": (r.get("logged_at").strftime("%H:%M")
                               if isinstance(r.get("logged_at"), datetime) else "—"),
                } for r in day_rows]), hide_index=True, width="stretch")
