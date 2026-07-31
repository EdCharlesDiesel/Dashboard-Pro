"""Triple-confluence alert — the only thing that earns an email.

The desk rule this encodes: **don't tell me about a setup unless three
independent reads agree on the same side.**

    1. Setup Ranker   — the multi-timeframe checklist grades the direction A
                        (``pct >= MIN_PCT``), i.e. the higher-timeframe
                        evidence is aligned.
    2. House view     — the canonical Weekly/Daily/4H composite
                        (``src/core/bias.py``) points the same way, so the
                        alert can never contradict the board every page shows.
    3. 15M Fib Entry  — a live golden-zone trigger (``ENTRY_FIRED`` or
                        ``IN_ZONE``) on the same bias, i.e. there is an actual
                        entry to take *right now* rather than a standing
                        opinion.

Ranker + house view answer "should I be short this pair at all?"; the 15M leg
answers "is this the moment?". Neither alone is worth an interruption — the
first is a week-long opinion, the second fires constantly. Together they are
rare and actionable, which is the entire point.

Pure and I/O-free: callers pass the three reads in, so the agreement logic is
unit-tested without network, DB, or Streamlit. ``src/services/
background_scanner.py`` gathers the inputs and sends the mail.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

# Grade A on the ranker's direction-only percentage.
MIN_PCT = 80.0

# Fib statuses that represent a live, takeable trigger. RETRACING/EXTENDED mean
# price hasn't reached the zone; INVALIDATED means it blew through it.
LIVE_FIB_STATUSES = ("ENTRY_FIRED", "IN_ZONE")

_HOUSE_TO_DIRECTION = {"BULLISH": "LONG", "BEARISH": "SHORT"}


@dataclass(frozen=True)
class Confluence:
    """One pair where all three reads agree. Carries the 15M trade levels —
    the fib leg is the one with an actual entry, so its levels are the plan."""
    pair: str
    direction: str          # LONG | SHORT
    ranker_pct: float
    ranker_grade: str
    house_direction: str    # BULLISH | BEARISH
    house_score: float
    fib_status: str         # ENTRY_FIRED | IN_ZONE
    entry: float
    sl: float
    tp1: float
    tp2: float
    rr1: float

    @property
    def is_fired(self) -> bool:
        """True when the confirming candle closed this bar (vs merely in zone)."""
        return self.fib_status == "ENTRY_FIRED"

    def dedupe_key(self) -> str:
        """Stable key for the alert ledger: one email per pair+direction+status.
        Status is included so an IN_ZONE alert doesn't suppress the later, more
        actionable ENTRY_FIRED on the same pair."""
        return f"{self.pair}|{self.direction}|{self.fib_status}"


def house_direction_to_trade(house_direction: Optional[str]) -> Optional[str]:
    """BULLISH -> LONG, BEARISH -> SHORT, anything else (NEUTRAL/None) -> None.
    A neutral board is not agreement — it's an absence of conviction."""
    if not house_direction:
        return None
    return _HOUSE_TO_DIRECTION.get(str(house_direction).strip().upper())


def evaluate(
    pair: str,
    ranker: Optional[Dict],
    house_direction: Optional[str],
    house_score: Optional[float],
    fib: Optional[Dict],
    fib_bias: Optional[str],
    min_pct: float = MIN_PCT,
) -> Optional[Confluence]:
    """A :class:`Confluence` when all three reads agree, else ``None``.

    ``ranker`` is a ``score_setup`` result (needs ``direction``/``pct``/``grade``),
    ``fib`` a ``fib_entry.fib_analysis`` result computed for ``fib_bias``.
    Anything missing, neutral, below grade, or pointing elsewhere returns None —
    the alert is deliberately hard to trigger.
    """
    if not ranker or not fib:
        return None

    direction = str(ranker.get("direction") or "").upper()
    if direction not in ("LONG", "SHORT"):
        return None

    # 1. Ranker must grade this direction at or above the threshold.
    try:
        pct = float(ranker.get("pct") or 0.0)
    except (TypeError, ValueError):
        return None
    if pct < min_pct:
        return None

    # 2. House view must point the same way (NEUTRAL is not agreement).
    if house_direction_to_trade(house_direction) != direction:
        return None

    # 3. The 15M leg must be live AND computed for this same side — a LONG fib
    #    read says nothing about a SHORT ranker call.
    if str(fib_bias or "").upper() != direction:
        return None
    if fib.get("status") not in LIVE_FIB_STATUSES:
        return None

    return Confluence(
        pair=pair,
        direction=direction,
        ranker_pct=pct,
        ranker_grade=str(ranker.get("grade") or "?"),
        house_direction=str(house_direction).upper(),
        house_score=float(house_score) if house_score is not None else 0.0,
        fib_status=str(fib["status"]),
        entry=float(fib.get("entry") or 0.0),
        sl=float(fib.get("sl") or 0.0),
        tp1=float(fib.get("tp1") or 0.0),
        tp2=float(fib.get("tp2") or 0.0),
        rr1=float(fib.get("rr1") or 0.0),
    )


def _fmt(price: float) -> str:
    """Project price convention: 5dp under 100 (FX), else 2dp."""
    if price is None:
        return "—"
    return f"{price:.5f}" if abs(price) < 100 else f"{price:.2f}"


def subject_for(items: Iterable[Confluence]) -> str:
    rows = list(items)
    if not rows:
        return "No confluence"
    fired = [r for r in rows if r.is_fired]
    lead = "⚡ ENTRY FIRED" if fired else "🎯 IN ZONE"
    names = ", ".join(f"{r.pair} {r.direction}" for r in rows[:3])
    more = "…" if len(rows) > 3 else ""
    return f"{lead} — {names}{more} (all 3 agree)"


def build_email(items: Iterable[Confluence]) -> tuple:
    """``(html, plain)`` for the confluence alert. Pure — no Streamlit, no I/O,
    so the body is unit-testable and identical from page or worker."""
    rows = list(items)
    if not rows:
        return "", ""

    cards = []
    for r in rows:
        colour = "#00a32a" if r.direction == "LONG" else "#c0392b"
        badge = "ENTRY FIRED" if r.is_fired else "IN ZONE"
        cards.append(f"""
        <div style="border:1px solid #d8dee4;border-left:4px solid {colour};
                    padding:14px 18px;margin:0 0 14px 0;background:#ffffff;">
          <div style="font:700 17px/1.3 Helvetica,Arial,sans-serif;color:#111;">
            {r.pair} &nbsp;<span style="color:{colour};">{r.direction}</span>
            &nbsp;<span style="font:600 11px/1 Helvetica,Arial;background:{colour};
                  color:#fff;padding:3px 8px;vertical-align:middle;">{badge}</span>
          </div>
          <table style="border-collapse:collapse;margin-top:10px;
                        font:13px/1.5 Helvetica,Arial,sans-serif;color:#333;">
            <tr><td style="padding:2px 18px 2px 0;color:#666;">Entry</td>
                <td style="font-weight:700;">{_fmt(r.entry)}</td></tr>
            <tr><td style="padding:2px 18px 2px 0;color:#666;">Stop</td>
                <td style="font-weight:700;color:#c0392b;">{_fmt(r.sl)}</td></tr>
            <tr><td style="padding:2px 18px 2px 0;color:#666;">Target 1</td>
                <td style="font-weight:700;color:#00a32a;">{_fmt(r.tp1)}
                    <span style="color:#666;font-weight:400;">({r.rr1:.2f}R)</span></td></tr>
            <tr><td style="padding:2px 18px 2px 0;color:#666;">Target 2</td>
                <td style="font-weight:700;color:#00a32a;">{_fmt(r.tp2)}</td></tr>
          </table>
          <div style="margin-top:12px;padding-top:10px;border-top:1px solid #eee;
                      font:12px/1.6 Helvetica,Arial,sans-serif;color:#555;">
            <b>Why this trade — all three agree:</b><br>
            • <b>Setup Ranker</b> grades {r.direction} <b>{r.ranker_grade}</b>
              ({r.ranker_pct:.0f}% of directional criteria)<br>
            • <b>House view</b> is <b>{r.house_direction}</b>
              (composite {r.house_score:+.2f}) — the same side as the board<br>
            • <b>15M Fib</b> status <b>{r.fib_status}</b> — price is at the
              0.382–0.618 golden zone, so there is an entry now
          </div>
        </div>""")

    html = f"""
    <div style="background:#f4f6f8;padding:24px;font-family:Helvetica,Arial,sans-serif;">
      <div style="max-width:640px;margin:0 auto;">
        <div style="font:700 20px/1.3 Helvetica,Arial;color:#111;margin-bottom:4px;">
          Triple-Confluence Alert
        </div>
        <div style="font:13px/1.5 Helvetica,Arial;color:#666;margin-bottom:18px;">
          {len(rows)} setup{'s' if len(rows) != 1 else ''} where the Setup Ranker,
          the house view and a live 15M Fib trigger all point the same way.
        </div>
        {''.join(cards)}
        <div style="font:11px/1.6 Helvetica,Arial;color:#8a8a8a;margin-top:18px;
                    padding-top:12px;border-top:1px solid #d8dee4;">
          Levels come from the 15M fib leg — the only leg with an actual entry.
          Confirm the spread and the news calendar before executing; this alert
          does not check either. Generated by Dashboard-Pro.
        </div>
      </div>
    </div>"""

    plain_rows = []
    for r in rows:
        badge = "ENTRY FIRED" if r.is_fired else "IN ZONE"
        plain_rows.append(
            f"{r.pair} {r.direction} [{badge}]\n"
            f"  entry {_fmt(r.entry)}  stop {_fmt(r.sl)}  "
            f"tp1 {_fmt(r.tp1)} ({r.rr1:.2f}R)  tp2 {_fmt(r.tp2)}\n"
            f"  ranker {r.ranker_grade} {r.ranker_pct:.0f}% | "
            f"house {r.house_direction} {r.house_score:+.2f} | fib {r.fib_status}")
    plain = ("Triple-Confluence Alert\n"
             f"{len(plain_rows)} setup(s) where Setup Ranker, house view and a live "
             "15M Fib trigger agree.\n\n" + "\n\n".join(plain_rows) +
             "\n\nLevels are from the 15M fib leg. Check spread and news before "
             "executing.\n")
    return html, plain
