"""Report builders + rule-based intelligence for the Reports page.

Each ``*_report()`` returns a plain dict: structured numbers/tables for display,
plus a ``verdict`` (rule-based, deterministic) and ``highlights`` list. The same
dict is what gets handed to :mod:`src.core.ai_reports` for an optional
Claude-written narrative — so the AI and the rules read from one source.

Data sources (all already in the app, nothing new fetched cheaply):
* AMD scans      — ``amd_scan`` events recorded by the AMD Scanner page.
* Trade journal  — the Postgres ``trade_setups`` table via ``TradeRepository``.
* System activity— observability ``events.jsonl``.
* Market/macro   — FRED macro data + the latest AMD phase distribution.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.core import observability


# ─────────────────────────────────────────────────────────────────────────────
# AMD scan report
# ─────────────────────────────────────────────────────────────────────────────

def amd_scan_report(limit: int = 4000) -> Dict[str, Any]:
    """Latest AMD phase per instrument + a run history, from recorded scans."""
    events = [e for e in observability.read_events(limit)
              if e.get("event_type") == "amd_scan"]

    latest: Dict[str, Dict[str, Any]] = {}
    history: List[Dict[str, Any]] = []
    for e in events:  # events are oldest→newest, so last write wins
        d = e.get("data", {}) or {}
        inst = d.get("instrument")
        row = {
            "ts": e.get("ts"),
            "instrument": inst,
            "phase": d.get("phase"),
            "bias": d.get("bias"),
            "last_close": d.get("last_close"),
            "detail": d.get("detail"),
        }
        history.append(row)
        if inst:
            latest[inst] = row

    phase_counts = Counter(
        (r["phase"] or "unknown") for r in latest.values()
    )
    instruments_in = lambda p: sorted(  # noqa: E731
        i for i, r in latest.items() if r["phase"] == p
    )

    distribution = instruments_in("distribution")
    manipulation = instruments_in("manipulation")
    highlights: List[str] = []
    if manipulation:
        highlights.append(
            f"{len(manipulation)} instrument(s) in MANIPULATION (sweep risk): "
            + ", ".join(manipulation)
        )
    if distribution:
        highlights.append(
            f"{len(distribution)} in DISTRIBUTION (topping/reversal watch): "
            + ", ".join(distribution)
        )
    if not latest:
        verdict = "No AMD scans recorded yet — open the AMD Scanner to populate."
    else:
        verdict = (
            f"{len(latest)} instrument(s) scanned; "
            f"{phase_counts.get('manipulation', 0)} in manipulation, "
            f"{phase_counts.get('distribution', 0)} in distribution, "
            f"{phase_counts.get('accumulation', 0)} in accumulation."
        )

    return {
        "scanned_instruments": len(latest),
        "total_scans_recorded": len(history),
        "phase_counts": dict(phase_counts),
        "latest_by_instrument": latest,
        "recent_history": list(reversed(history))[:100],
        "highlights": highlights,
        "verdict": verdict,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Trade journal / performance report
# ─────────────────────────────────────────────────────────────────────────────

def trade_performance_report(db_cfg: Optional[Dict[str, Any]] = None,
                             window: int = 100) -> Dict[str, Any]:
    """Win rate / R / profit factor + equity curve, from ``trade_setups``.

    ``db_cfg`` is an optional override; otherwise DB settings come from
    :func:`src.core.secrets.db_config`. Returns ``available: False`` when the
    database can't be reached or has no closed trades.
    """
    from src.core import secrets
    from src.db import DBConfig, TradeRepository

    cfg = DBConfig.from_mapping(db_cfg) if db_cfg else DBConfig.from_mapping(
        secrets.db_config()
    )
    repo = TradeRepository(cfg)

    try:
        stats = repo.performance_stats(n=window)
        setups = repo.load_setups(limit=window)
        losses = repo.daily_losses()
    except Exception as exc:
        return {"available": False, "reason": str(exc)[:200],
                "verdict": "Trade journal unavailable — check DB credentials."}

    if not stats:
        return {"available": False,
                "reason": "No closed trades recorded.",
                "open_setups": len(setups),
                "verdict": "No closed trades yet — log and close setups to build "
                           "performance history."}

    # Equity curve in R, oldest→newest. All closed trades count — imported MT4
    # fills without a stop loss carry a null R, which we treat as 0R so they
    # still appear on the curve and in the tallies (consistent with the Journal).
    closed = [s for s in reversed(setups)
              if not s.get("is_open", True) and s.get("outcome") is not None]
    equity, cum = [], 0.0
    for s in closed:
        cum += float(s.get("r_multiple") or 0.0)
        equity.append({"logged_at": s.get("logged_at"),
                       "instrument": s.get("instrument"),
                       "r_multiple": float(s.get("r_multiple") or 0.0),
                       "cum_r": round(cum, 2)})

    # Per-instrument R tally.
    by_instrument: Dict[str, float] = defaultdict(float)
    for s in closed:
        by_instrument[s.get("instrument", "?")] += float(s.get("r_multiple") or 0.0)

    # How the closed trades reached the journal (checklist vs imported fills).
    by_source = Counter(s.get("source") or "checklist" for s in closed)

    wr = stats["win_rate"]
    pf = stats["profit_factor"]
    highlights = [
        f"Win rate {wr:.0f}% over {stats['total']} closed trades "
        f"(target 66%).",
        f"Profit factor {pf:.2f}; expectancy {stats['expectancy']:+.2f}R/trade.",
        f"Net {cum:+.1f}R across the window.",
    ]
    if by_source:
        highlights.append(
            "Closed trades by source: "
            + ", ".join(f"{k}×{v}" for k, v in by_source.most_common())
        )
    if losses.get("blocked"):
        highlights.append(
            f"⛔ Daily loss limit hit ({losses['losses_today']}/{losses['limit']}) "
            "— new entries blocked today."
        )

    if wr >= 66 and stats["expectancy"] > 0:
        verdict = "On target — win rate and expectancy both healthy."
    elif stats["expectancy"] > 0:
        verdict = ("Profitable but below the 66% win-rate target — edge is in "
                   "R-multiple, protect it.")
    else:
        verdict = ("Negative expectancy over this window — review setup quality "
                   "before sizing up.")

    return {
        "available": True,
        "stats": stats,
        "net_r": round(cum, 2),
        "equity_curve": equity,
        "by_instrument": {k: round(v, 2) for k, v in
                          sorted(by_instrument.items(), key=lambda x: x[1])},
        "by_source": dict(by_source.most_common()),
        "daily_losses": losses,
        "open_setups": len([s for s in setups if s.get("is_open", True)]),
        "highlights": highlights,
        "verdict": verdict,
    }


# ─────────────────────────────────────────────────────────────────────────────
# System activity report
# ─────────────────────────────────────────────────────────────────────────────

def system_activity_report(limit: int = 4000) -> Dict[str, Any]:
    """Counts and recent errors from the observability event log."""
    events = observability.read_events(limit)
    by_type = Counter(e.get("event_type", "?") for e in events)
    by_page = Counter(e.get("page", "?") for e in events
                      if e.get("event_type") == "page_view")
    sessions = {e.get("session_id") for e in events if e.get("session_id")}

    fetches = [e for e in events if e.get("event_type") == "data_fetch"]
    fetch_ok = sum(1 for e in fetches if (e.get("data") or {}).get("ok"))
    durations = [float((e.get("data") or {}).get("duration_ms") or 0)
                 for e in fetches if (e.get("data") or {}).get("duration_ms")]
    avg_fetch = round(sum(durations) / len(durations), 1) if durations else 0.0

    errors = [e for e in events if e.get("event_type") == "exception"
              or e.get("level") == "ERROR"]
    recent_errors = [
        {"ts": e.get("ts"), "page": e.get("page"),
         "message": e.get("message"), "type": e.get("event_type")}
        for e in list(reversed(errors))[:25]
    ]

    highlights = [
        f"{by_type.get('page_view', 0)} page views across {len(sessions)} session(s).",
        f"{len(fetches)} data fetches "
        f"({fetch_ok} ok, {len(fetches) - fetch_ok} failed), "
        f"avg {avg_fetch} ms.",
        f"{len(errors)} error/exception event(s) recorded.",
    ]
    if errors:
        highlights.append("⚠️ Errors present — see the recent-errors table.")

    verdict = ("Clean — no errors recorded." if not errors
               else f"{len(errors)} error event(s) need attention.")

    return {
        "total_events": len(events),
        "sessions": len(sessions),
        "by_type": dict(by_type),
        "page_views_by_page": dict(by_page.most_common()),
        "data_fetches": {"total": len(fetches), "ok": fetch_ok,
                         "failed": len(fetches) - fetch_ok,
                         "avg_ms": avg_fetch},
        "errors": len(errors),
        "recent_errors": recent_errors,
        "highlights": highlights,
        "verdict": verdict,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Market / macro snapshot report
# ─────────────────────────────────────────────────────────────────────────────

def market_snapshot_report() -> Dict[str, Any]:
    """Point-in-time digest: FRED macro bias + latest AMD phase distribution.

    Deliberately built from FRED (cached 1h) and already-recorded AMD scans so
    opening the report doesn't trigger a 21-instrument live price sweep.
    """
    from src.core import secrets
    from src.core.data_provider import get_macro_data

    macro: Dict[str, Dict[str, float]] = {}
    macro_live = False
    try:
        macro, macro_live = get_macro_data(secrets.fred_api_key())
    except Exception:
        macro = {}

    # Simple macro bias per currency: positive growth + contained inflation = ↑.
    macro_bias: Dict[str, str] = {}
    for ccy, m in macro.items():
        gdp = m.get("GDP", 0.0)
        infl = m.get("Inflation", 0.0)
        rates = m.get("Rates", 0.0)
        score = (1 if gdp > 1.0 else -1 if gdp < 0 else 0) \
            + (1 if rates > 2.0 else 0) + (-1 if infl > 4.0 else 0)
        macro_bias[ccy] = "↑ hawkish/strong" if score >= 1 else (
            "↓ dovish/weak" if score <= -1 else "→ neutral")

    amd = amd_scan_report()
    phase_counts = amd.get("phase_counts", {})

    highlights = []
    strong = [c for c, b in macro_bias.items() if b.startswith("↑")]
    weak = [c for c, b in macro_bias.items() if b.startswith("↓")]
    if strong:
        highlights.append("Macro-strong currencies: " + ", ".join(strong))
    if weak:
        highlights.append("Macro-weak currencies: " + ", ".join(weak))
    if not macro_live:
        highlights.append("⚠️ FRED live data unavailable — macro uses fallback "
                          "values; treat bias as indicative only.")
    if phase_counts:
        highlights.append(
            "AMD phases now: "
            + ", ".join(f"{k}×{v}" for k, v in phase_counts.items())
        )

    verdict = ("Live macro + AMD snapshot ready." if macro_live
               else "Snapshot uses fallback macro data — refresh FRED for live bias.")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "macro_live": macro_live,
        "macro_bias": macro_bias,
        "macro_raw": macro,
        "amd_phase_counts": phase_counts,
        "amd_latest": amd.get("latest_by_instrument", {}),
        "highlights": highlights,
        "verdict": verdict,
    }
