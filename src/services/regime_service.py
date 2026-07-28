"""Streamlit-facing wrapper over the statistical jump model
(:mod:`src.core.jump_model`).

Fetches daily bars through the data spine, builds the downside-risk feature
set, fits the jump model, and caches one canonical regime read per instrument
— the same shape of wiring `bias_service` does for the house view.

Page integration is one line::

    from src.services.regime_service import get_regime
    read = get_regime("XAU/USD")

Three deliberate design points:

**History, not the canonical 300-day window.** A regime model needs to have
*seen* several regimes; 300 daily bars minus the feature warm-up leaves barely
one. This module widens ``period`` to :data:`REGIME_PERIOD`, which the spine
contract explicitly permits ("a page may widen period/days") — interval,
resample rule and read-through caching stay owned by
:mod:`src.services.market_data`.

**The cached read is a confirmation, never a forecast.** `get_regime` reports
where the *offline* DP places the current label. That is the right thing to
show a human ("we are in a risk-off regime") and the wrong thing to feed a
backtest, because the DP positions every breakpoint using the whole sample.
Anything performance-shaped must go through `tune_jump_penalty`, which is
built on `online_state_sequence` (expanding-window refit, no lookahead).

**It is a filter, not a signal.** A regime read is not a directional pair+bias
call, so `log_regime` writes to ``tool_usage_log`` and this module never
touches ``trade_setups`` — same policy as the other context/research tools.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import streamlit as st

from src.core.jump_model import (
    DEFAULT_HALFLIVES,
    JumpModelFit,
    downside_features,
    fit_jump_model,
    log_returns,
    online_state_sequence,
)
from src.instruments.registry import INSTRUMENTS
from src.services.market_data import daily_ohlc

# ~10 years of daily bars: enough to contain several regimes, which is the
# minimum for the fitted centroids to mean anything.
REGIME_PERIOD = "10y"

# Daily bars and a signal that turns about once a year — re-fitting every 5
# minutes would burn CPU to produce the same label. One hour is generous.
REGIME_TTL = 3600

# Below this many feature rows a fit is not worth showing (roughly one year of
# usable history after the 63-day feature warm-up).
MIN_FEATURE_ROWS = 260

TRADING_DAYS = 252

# Penalty grid for `tune_jump_penalty`. Spans "plain k-means" to "almost never
# switches"; the default fit penalty sits inside it.
DEFAULT_PENALTY_GRID: tuple[float, ...] = (0.0, 25.0, 50.0, 100.0, 150.0, 250.0, 400.0)


# ---------------------------------------------------------------------------
# Pure: features off an OHLC frame
# ---------------------------------------------------------------------------
def regime_inputs(
    daily: pd.DataFrame,
    halflives: Sequence[int] = DEFAULT_HALFLIVES,
) -> tuple[pd.DataFrame, pd.Series]:
    """Turn a daily OHLC frame into ``(features, returns)``.

    Returns two empty objects rather than raising when the frame is missing a
    usable ``Close`` — every caller here degrades to "no read" the way the rest
    of the spine consumers do.
    """
    if daily is None or daily.empty or "Close" not in daily.columns:
        return pd.DataFrame(), pd.Series(dtype=float)
    close = pd.Series(daily["Close"]).astype(float).dropna()
    close = close[close > 0]
    if len(close) < 2:
        return pd.DataFrame(), pd.Series(dtype=float)

    returns = log_returns(close)
    features = downside_features(returns, halflives=halflives)
    return features, returns


# ---------------------------------------------------------------------------
# Pure: the read
# ---------------------------------------------------------------------------
@dataclass
class RegimeRead:
    """One instrument's current regime, as the offline fit sees it."""

    instrument: str
    label: str
    state: int
    n_states: int
    jump_penalty: float
    n_obs: int
    n_switches: int
    switches_per_year: float
    current_spell: int          # bars the current state has persisted
    state_shares: dict = field(default_factory=dict)
    asof: Optional[pd.Timestamp] = None

    @property
    def is_risk_off(self) -> bool:
        """True in the lowest (worst) state. For the canonical two-state model
        that is ``RISK_OFF``; ordering is guaranteed by `fit_jump_model`."""
        return self.state == 0

    def to_payload(self) -> dict:
        """JSON-safe row for ``tool_usage_log``."""
        return {
            "instrument": self.instrument,
            "label": self.label,
            "state": int(self.state),
            "n_states": int(self.n_states),
            "jump_penalty": float(self.jump_penalty),
            "n_obs": int(self.n_obs),
            "n_switches": int(self.n_switches),
            "switches_per_year": round(float(self.switches_per_year), 3),
            "current_spell": int(self.current_spell),
            "state_shares": {k: round(float(v), 4)
                             for k, v in self.state_shares.items()},
            "asof": str(self.asof) if self.asof is not None else None,
        }


def _current_spell(states: np.ndarray) -> int:
    """How many trailing bars the final state has held."""
    if len(states) == 0:
        return 0
    last = states[-1]
    changed = np.flatnonzero(states != last)
    return len(states) if len(changed) == 0 else int(len(states) - changed[-1] - 1)


def read_from_fit(instrument: str, fit: JumpModelFit) -> RegimeRead:
    """Summarise a fitted model as a `RegimeRead` (pure)."""
    n_obs = len(fit.states)
    years = n_obs / TRADING_DAYS if n_obs else 0.0
    return RegimeRead(
        instrument=instrument,
        label=fit.current_label,
        state=fit.current_state,
        n_states=fit.n_states,
        jump_penalty=fit.jump_penalty,
        n_obs=n_obs,
        n_switches=fit.n_switches,
        switches_per_year=(fit.n_switches / years) if years > 0 else 0.0,
        current_spell=_current_spell(fit.states),
        state_shares=fit.state_shares(),
        asof=fit.index[-1] if n_obs else None,
    )


# ---------------------------------------------------------------------------
# Pure: strategy evaluation (the only honest way to tune the penalty)
# ---------------------------------------------------------------------------
def annualised_sharpe(returns: pd.Series, ann_factor: int = TRADING_DAYS) -> float:
    """Annualised Sharpe of a return stream. 0.0 for an empty or flat stream
    rather than a NaN or a divide-by-zero."""
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return 0.0
    sd = float(r.std(ddof=1))
    # A constant stream has infinite Sharpe in theory and a float-noise std in
    # practice (~1e-18), which would score as astronomically good. Degenerate
    # streams deliberately score 0 so they can never win a penalty grid.
    if not np.isfinite(sd) or sd <= 1e-12:
        return 0.0
    return float(r.mean() / sd * np.sqrt(ann_factor))


def regime_strategy_returns(
    returns: pd.Series,
    states: pd.Series,
    risk_on_state: Optional[int] = None,
    n_states: int = 2,
    risk_off_exposure: float = 0.0,
    cost_bps: float = 10.0,
) -> pd.Series:
    """
    Returns of "hold the asset in the good regime, de-risk in the bad one".

    Two details that decide whether the result is real:

    * **One-day trading delay.** A label computed from today's close can only
      be traded from tomorrow, so the position is ``states.shift(1)``. Without
      it the strategy sells on the very day of the crash and the backtest is
      fiction.
    * **Transaction costs** are charged on position *changes* at ``cost_bps``
      (the source paper uses 10 bps). A signal that flips constantly should be
      punished for it — that is half the point of measuring this at all.

    ``risk_on_state`` defaults to ``n_states - 1``, which the canonical
    ordering makes the highest-return regime. It is derived from the model's
    state space and **not** from the states actually observed in this slice:
    a window that happens to be entirely risk-off would otherwise have its
    single observed state read as "the best one" and be scored fully
    invested — silently inverting the strategy exactly when de-risking
    matters most.
    """
    s = pd.Series(states).dropna().astype(int)
    r = pd.Series(returns).astype(float).reindex(s.index).dropna()
    s = s.reindex(r.index)
    if r.empty:
        return pd.Series(dtype=float)

    on_state = (int(n_states) - 1) if risk_on_state is None else int(risk_on_state)
    target = np.where(s.to_numpy() == on_state, 1.0, float(risk_off_exposure))
    position = pd.Series(target, index=r.index)

    # Act on yesterday's label; start flat-equivalent before the first signal.
    position = position.shift(1).fillna(float(risk_off_exposure))
    turnover = position.diff().abs().fillna(abs(position.iloc[0]))
    cost = turnover * (cost_bps / 10_000.0)
    return position * r - cost


@dataclass
class PenaltyTuning:
    """Result of a walk-forward sweep over candidate jump penalties."""

    best_penalty: float
    table: pd.DataFrame
    n_folds: int
    min_train: int
    refit_every: int
    cost_bps: float

    def to_payload(self) -> dict:
        return {
            "best_penalty": float(self.best_penalty),
            "n_folds": int(self.n_folds),
            "min_train": int(self.min_train),
            "refit_every": int(self.refit_every),
            "cost_bps": float(self.cost_bps),
            "grid": [
                {
                    "penalty": float(row.penalty),
                    "mean_sharpe": round(float(row.mean_sharpe), 4),
                    "min_fold_sharpe": round(float(row.min_fold_sharpe), 4),
                    "switches_per_year": round(float(row.switches_per_year), 3),
                }
                for row in self.table.itertuples()
            ],
        }


def tune_jump_penalty(
    features: pd.DataFrame,
    returns: pd.Series,
    candidates: Sequence[float] = DEFAULT_PENALTY_GRID,
    min_train: int = 500,
    refit_every: int = 5,
    n_folds: int = 3,
    cost_bps: float = 10.0,
    risk_off_exposure: float = 0.0,
    ann_factor: int = TRADING_DAYS,
    **fit_kwargs,
) -> PenaltyTuning:
    """
    Walk-forward selection of ``jump_penalty`` — the paper's procedure, and the
    only defensible way to pick this number.

    For each candidate the label path is built by `online_state_sequence`
    (expanding-window refit, so no observation is ever labelled using its own
    future), converted to a delayed, cost-charged strategy, split into
    ``n_folds`` contiguous out-of-sample blocks, and scored by Sharpe. The
    winner is the highest **mean** fold Sharpe.

    ``min_fold_sharpe`` is reported alongside deliberately: a penalty that wins
    on average while losing badly in one fold is overfit to the other two, and
    that is invisible if you only look at the mean.

    This is expensive — ``len(candidates)`` × O(T / refit_every) model fits —
    and is meant to be run deliberately (a research tab, a notebook), never on
    a page render.

    Raises:
        ValueError: on an empty candidate grid, ``n_folds < 1``, or history too
        short to produce at least one observation per fold.
    """
    if not len(candidates):
        raise ValueError("candidates must be a non-empty sequence of penalties")
    if n_folds < 1:
        raise ValueError("n_folds must be at least 1")

    frame = features.dropna()
    oos_len = len(frame) - min_train + 1
    if oos_len < n_folds:
        raise ValueError(
            "history too short: {0} out-of-sample rows for {1} folds "
            "(need min_train + n_folds <= {2})".format(
                max(oos_len, 0), n_folds, len(frame))
        )

    rows = []
    for penalty in candidates:
        states = online_state_sequence(
            frame, min_train=min_train, refit_every=refit_every,
            jump_penalty=float(penalty), **fit_kwargs,
        )
        strat = regime_strategy_returns(
            returns, states, n_states=int(fit_kwargs.get("n_states", 2)),
            risk_off_exposure=risk_off_exposure, cost_bps=cost_bps,
        )
        folds = np.array_split(strat, n_folds)
        sharpes = [annualised_sharpe(f, ann_factor=ann_factor) for f in folds]

        n_switch = int((states.diff().fillna(0) != 0).sum())
        years = len(states) / ann_factor if len(states) else 0.0
        rows.append({
            "penalty": float(penalty),
            "mean_sharpe": float(np.mean(sharpes)),
            "min_fold_sharpe": float(np.min(sharpes)),
            "std_fold_sharpe": float(np.std(sharpes)),
            "switches_per_year": float(n_switch / years) if years > 0 else 0.0,
            "n_switches": n_switch,
        })

    table = pd.DataFrame(rows)
    best = float(table.loc[table["mean_sharpe"].idxmax(), "penalty"])
    return PenaltyTuning(
        best_penalty=best, table=table, n_folds=n_folds, min_train=min_train,
        refit_every=refit_every, cost_bps=cost_bps,
    )


# ---------------------------------------------------------------------------
# Streamlit-facing: cached reads off the spine
# ---------------------------------------------------------------------------
@st.cache_data(ttl=REGIME_TTL, show_spinner=False)
def _fit_cached(instrument: str, ticker: str, jump_penalty: float,
                n_states: int) -> Optional[JumpModelFit]:
    daily = daily_ohlc(ticker, period=REGIME_PERIOD)
    features, _ = regime_inputs(daily)
    if len(features) < MIN_FEATURE_ROWS:
        return None
    return fit_jump_model(features, n_states=n_states, jump_penalty=jump_penalty)


def get_regime(instrument: str, jump_penalty: float = 150.0,
               n_states: int = 2) -> Optional[RegimeRead]:
    """The current regime for a registry instrument, or ``None`` when the
    symbol is unknown, history is too short, or the fetch failed.

    Cached for :data:`REGIME_TTL`. Never raises — a regime read is context, and
    context that can crash a page is worse than no context.
    """
    inst = INSTRUMENTS.get(instrument)
    if inst is None:
        return None
    try:
        fit = _fit_cached(instrument, inst.ticker, float(jump_penalty),
                          int(n_states))
    except Exception:
        return None
    if fit is None:
        return None
    return read_from_fit(instrument, fit)


@st.cache_data(ttl=REGIME_TTL, show_spinner=False)
def get_regime_history(instrument: str, jump_penalty: float = 150.0,
                       n_states: int = 2) -> pd.Series:
    """The full fitted label path, for **charting only**.

    ⚠ These are the offline DP's labels: every breakpoint was placed with the
    benefit of the whole sample, so this series is beautifully punctual and
    completely untradeable. Shade a chart with it; never score a strategy on
    it — use `tune_jump_penalty` (or `online_state_sequence` directly) for
    anything that keeps a P&L.

    Empty Series when there is no usable read.
    """
    inst = INSTRUMENTS.get(instrument)
    if inst is None:
        return pd.Series(dtype=object)
    try:
        fit = _fit_cached(instrument, inst.ticker, float(jump_penalty),
                          int(n_states))
    except Exception:
        return pd.Series(dtype=object)
    if fit is None:
        return pd.Series(dtype=object)
    return fit.label_series()


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------
def log_regime(read: Optional[RegimeRead]) -> bool:
    """Append the read to ``tool_usage_log`` under ``jump_regime``.

    Deduped per instrument + day + label, so a Streamlit rerun doesn't append
    an identical row on every widget touch, but a genuine regime change on the
    same day still gets recorded. Best-effort: returns whether a row was
    written and never raises.

    Deliberately **not** ``trade_setups`` — a regime is a filter on other
    signals, not a pair+bias call of its own.
    """
    if read is None:
        return False
    try:
        from src.services.alert_service import NotifyCache
        from src.services.tool_log import log_tool_usage

        day = "" if read.asof is None else str(pd.Timestamp(read.asof).date())
        key = "{0}|{1}|{2}".format(read.instrument, day, read.label)
        if not NotifyCache("jump_regime_log").filter_new([key]):
            return False
        return bool(log_tool_usage("jump_regime", read.to_payload()))
    except Exception:
        return False
