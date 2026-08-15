---
layer: services
lines: 699
generated: true
---

# `src/services/backtest_service.py`

Production backtest engine — single source of truth for the 18-point

Source: [[Src/services/backtest_service.py]] · [open on disk](../src/services/backtest_service.py)

## Imports (1)

- [[src.db.market_cache]] — Streamlit read-through cache for market data, backed by Postgres

## Imported by (0)

_Nothing imports this. Either an entry point or dead code._

## Public surface

`calc_ema`, `calc_rsi`, `calc_macd`, `calc_atr`, `calc_stochastic`, `swing_highs`, `swing_lows`, `load_data`, `run_checks`, `simulate_trade`, `run_backtest`, `build_equity`, `compute_stats`, `MonteCarloResult`, `monte_carlo_resample`, `walk_forward_segments`, `parameter_sweep`, `portfolio_backtest`

Back to [[Architecture]].
