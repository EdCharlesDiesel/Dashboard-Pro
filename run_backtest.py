"""CLI smoke-test for the 18-point workflow backtest.

The strategy engine now lives in src/services/backtest_service.py — this script
is a thin runner so the page and the CLI can never drift apart.
"""
import warnings
warnings.filterwarnings("ignore")

from datetime import date, timedelta

import pandas as pd

from src.services.backtest_service import load_data, run_backtest, compute_stats

# ── Instruments to test ──────────────────────────────────────────
RUNS = [
    ("GBP/JPY", "GBPJPY=X"),
    ("Gold", "GC=F"),
]

end_dt = date.today()
start_dt = end_dt - timedelta(days=180)


def print_results(name, trades, direction):
    if not trades:
        print(f"  {name} {direction:<6} : NO TRADES in 180-day window")
        return
    stats = compute_stats(trades, account=10000, risk_pct=1.0)
    print("=" * 60)
    print(f"  {name} {direction}  |  180-day backtest results")
    print("=" * 60)
    print(f"  Total trades   : {stats['total']}")
    print(f"  Win rate       : {stats['win_rate']:.0f}%  ({stats['wins']} wins / {stats['losses']} losses)")
    print(f"  Net R          : {stats['net_r']:+.1f}R")
    print(f"  Avg R / trade  : {stats['avg_r']:+.2f}R")
    print(f"  Profit factor  : {stats['profit_factor']:.2f}")
    print(f"  Max drawdown   : {stats['max_dd_r']:.1f}R  ({stats['max_dd_pct']:.1f}% compounded)")
    print(f"  Sharpe / SQN   : {stats['sharpe']:.2f} / {stats['sqn']:.2f}")
    print(f"  P&L (1% risk, $10k) : ${stats['final_equity']-10000:+,.0f}")
    print("=" * 60)
    print(f"  {'DATE':<12} {'SCORE':>5} {'ENTRY':>10} {'EXIT':>10} {'R':>7}  OUTCOME")
    print("  " + "-" * 60)
    for t in trades:
        print(f"  {str(t['date']):<12} {t['score']:>5} {float(t['entry']):>10.3f} "
              f"{float(t['exit']):>10.3f} {t['r']:>+7.2f}R  {t['outcome']}")
    print()


for name, ticker in RUNS:
    print(f"\nDownloading {name} ({ticker})...")
    daily, weekly = load_data(ticker)
    print(f"  Daily bars: {len(daily)}  |  Weekly bars: {len(weekly)}")
    print(f"  Window: {start_dt} to {end_dt}  |  Min score 14/18")

    for direction in ("Long", "Short"):
        trades = run_backtest(daily, weekly, direction, 14, 1.5, 2.0, start_dt, end_dt)
        print_results(name, trades, direction)
