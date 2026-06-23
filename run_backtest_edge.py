"""Edge-validation harness for the 18-point setup score.

The dashboard ranks setups 0–18 and treats high scores (Grade A) as tradeable
edge. This script *tests that claim*: it runs the production backtest engine
(src/services/backtest_service.py) across the full instrument universe at a LOW
entry threshold so trades across the whole score spectrum are captured, then
buckets the realised trades by their entry score and reports the per-bucket
performance gradient.

If the score carries genuine edge, average R / win-rate / profit-factor should
climb monotonically with the score bucket. A flat or inverted gradient means the
score is not selecting better trades and the ranking is cosmetic.

Run:  python run_backtest_edge.py
"""
import warnings
warnings.filterwarnings("ignore")

from collections import defaultdict
from datetime import date, timedelta

import numpy as np

from src.instruments.registry import INSTRUMENTS
from src.services.backtest_service import load_data, run_backtest

# ── Test configuration ───────────────────────────────────────────────
# Capture everything from a permissive floor (10/18) upward so we observe how
# performance scales with score. Critical checks must still pass at every level.
MIN_SCORE_FLOOR = 10
SL_MULT = 1.5
RR = 2.0
LOOKBACK_DAYS = 360                      # yfinance daily history is ~1y
BUCKETS = [(10, 11), (12, 13), (14, 15), (16, 18)]   # score ranges

end_dt = date.today()
start_dt = end_dt - timedelta(days=LOOKBACK_DAYS)


def bucket_label(score: int) -> str:
    for lo, hi in BUCKETS:
        if lo <= score <= hi:
            return f"{lo}-{hi}"
    return "other"


def stats(rs: list) -> dict:
    a = np.array(rs, dtype=float)
    wins = a[a > 0]
    return {
        "n": len(a),
        "win_rate": len(wins) / len(a) * 100 if len(a) else 0.0,
        "avg_r": float(a.mean()) if len(a) else 0.0,
        "net_r": float(a.sum()),
        "pf": (wins.sum() / abs(a[a <= 0].sum())) if a[a <= 0].sum() != 0 else float("inf"),
    }


def main() -> None:
    by_bucket: dict[str, list] = defaultdict(list)
    by_grade_A: list = []        # 16-18 = Grade A
    by_grade_below: list = []    # 10-15
    per_instrument = []
    total_trades = 0

    print(f"Edge validation · {start_dt} → {end_dt} · floor {MIN_SCORE_FLOOR}/18 "
          f"· SL {SL_MULT}×ATR · R:R {RR}")
    print("Scanning universe (live yfinance — this takes a minute)…\n")

    for name, inst in INSTRUMENTS.items():
        try:
            daily, weekly = load_data(inst.ticker)
        except Exception as e:
            print(f"  {name:8} — load failed: {e}")
            continue
        if daily.empty or len(daily) < 60:
            print(f"  {name:8} — insufficient history ({len(daily)} bars)")
            continue

        inst_trades = []
        for direction in ("Long", "Short"):
            trades = run_backtest(daily, weekly, direction, MIN_SCORE_FLOOR,
                                  SL_MULT, RR, start_dt, end_dt)
            inst_trades.extend(trades)

        for t in inst_trades:
            by_bucket[bucket_label(t["score"])].append(t["r"])
            (by_grade_A if t["score"] >= 16 else by_grade_below).append(t["r"])

        total_trades += len(inst_trades)
        if inst_trades:
            s = stats([t["r"] for t in inst_trades])
            per_instrument.append((name, s))
        print(f"  {name:8} — {len(inst_trades):3} trades")

    print("\n" + "=" * 70)
    print(f"  EDGE GRADIENT — does a higher score buy better trades?  (N={total_trades})")
    print("=" * 70)
    print(f"  {'SCORE':>7} {'TRADES':>7} {'WIN%':>7} {'AVG R':>8} {'NET R':>8} {'PF':>6}")
    print("  " + "-" * 66)
    for lo, hi in BUCKETS:
        key = f"{lo}-{hi}"
        rs = by_bucket.get(key, [])
        if not rs:
            print(f"  {key:>7} {'0':>7}      —        —        —      —")
            continue
        s = stats(rs)
        print(f"  {key:>7} {s['n']:>7} {s['win_rate']:>6.1f}% {s['avg_r']:>+8.2f} "
              f"{s['net_r']:>+8.1f} {s['pf']:>6.2f}")

    print("\n" + "=" * 70)
    print("  GRADE A (16-18)  vs  BELOW (10-15)")
    print("=" * 70)
    for tag, rs in (("Grade A (16-18)", by_grade_A), ("Below   (10-15)", by_grade_below)):
        if not rs:
            print(f"  {tag}: no trades")
            continue
        s = stats(rs)
        print(f"  {tag}: N={s['n']:>4}  win={s['win_rate']:>5.1f}%  "
              f"avgR={s['avg_r']:>+5.2f}  netR={s['net_r']:>+6.1f}  PF={s['pf']:.2f}")

    # Verdict
    a = stats(by_grade_A) if by_grade_A else None
    b = stats(by_grade_below) if by_grade_below else None
    print("\n  VERDICT:")
    if a and b and a["n"] >= 10 and b["n"] >= 10:
        if a["avg_r"] > b["avg_r"] and a["pf"] > b["pf"]:
            print("  ✅ Grade A outperforms — the score is selecting better trades (edge present).")
        elif a["avg_r"] > b["avg_r"]:
            print("  ◐ Grade A has higher avg R but not a cleanly higher profit factor — weak edge.")
        else:
            print("  ❌ Grade A does NOT outperform — the ranking is not predictive on this data.")
    else:
        print("  ⚠ Too few trades in one or both grades to conclude. Widen the window / universe.")

    print("\n  PER-INSTRUMENT (all eligible trades):")
    per_instrument.sort(key=lambda x: x[1]["net_r"], reverse=True)
    for name, s in per_instrument:
        print(f"    {name:8} N={s['n']:>3}  win={s['win_rate']:>5.1f}%  "
              f"avgR={s['avg_r']:>+5.2f}  netR={s['net_r']:>+6.1f}  PF={s['pf']:.2f}")


if __name__ == "__main__":
    main()
