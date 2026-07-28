"""
jump_model.py
=============
Statistical jump model for regime detection — the Bemporad/Breschi/Piga/Boyd
(2018, *Automatica*) "fitting jump models" formulation, with the downside-risk
feature set and two-state framing of Shu, Yu & Mulvey (2024, *Journal of Asset
Management*).

The idea in one line: **k-means clustering with memory**. Plain clustering of
daily feature vectors flips state whenever a noisy day lands nearer the other
centroid; a hidden Markov model damps that only softly (the transition prior is
one term in a likelihood, easily swamped). A jump model instead charges a
*fixed penalty* ``jump_penalty`` for every state transition and then minimises

    sum_t ||z_t - c_{s_t}||^2  +  jump_penalty * #{t : s_t != s_{t-1}}

**exactly**, over the whole state sequence at once, by dynamic programming.
Persistence stops being a prior and becomes a hard cost, which is why the fitted
signal switches on the order of once a year rather than once a month.

Fitting is coordinate descent, alternating:
    1. states    <- DP over the current centroids   (`_dp_assign`, exact)
    2. centroids <- mean of the points assigned to each state
until the state sequence stops changing. Each pass is monotonically
non-increasing in the objective, so it converges; restarts (``n_init``) guard
against a poor k-means++ seeding.

**This module confirms regimes, it does not predict them.** The DP is inherently
offline — it uses the whole sequence to place every breakpoint, so
``fit_jump_model(...).states`` must never be read as a historical signal you
could have traded. For anything backtest-shaped use `online_state_sequence`
(expanding-window refit, last label only) or `assign_online` (greedy penalised
update from a frozen fit). Detection latency in the source paper's COVID-19
example was around half a month.

Pure numpy/pandas — no sklearn, no Streamlit, no I/O. Everything here is
deterministic given ``random_state``.

References:
    Bemporad, A., Breschi, V., Piga, D., Boyd, S.P. (2018). "Fitting Jump
        Models." Automatica, 96, 11-21. arXiv:1711.09220
    Shu, Y., Yu, C., Mulvey, J.M. (2024). "Downside Risk Reduction Using
        Regime-Switching Signals: A Statistical Jump Model Approach."
        Journal of Asset Management, 25(5), 493-507. arXiv:2402.05272
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

# Guard against divide-by-zero in Sortino / standardisation of a constant column.
_EPS = 1e-12

# Default feature horizons (trading days): ~1 week, ~1 month, ~1 quarter.
DEFAULT_HALFLIVES: tuple[int, ...] = (5, 21, 63)

# Canonical two-state naming for this app: state 0 is the low-return /
# high-downside state (see `fit_jump_model`'s `order_by` contract).
RISK_LABELS_2: tuple[str, ...] = ("RISK_OFF", "RISK_ON")


# ----------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------
def log_returns(close: pd.Series) -> pd.Series:
    """Daily log returns from a close series. NaNs (first bar) dropped."""
    close = pd.Series(close).astype(float)
    if (close <= 0).any():
        raise ValueError("log_returns needs strictly positive prices")
    return np.log(close).diff().dropna()


def downside_features(
    returns: pd.Series,
    halflives: Sequence[int] = DEFAULT_HALFLIVES,
) -> pd.DataFrame:
    """
    The Shu/Yu/Mulvey feature set: exponentially weighted mean return,
    exponentially weighted **downside** deviation, and their ratio (a Sortino
    ratio), at several halflives.

    Downside deviation uses only the negative part of each return,
    ``sqrt(EW[min(r, 0)^2])`` — that asymmetry is the whole point. Squaring
    raw returns would make a violent rally look like a crash; a bear regime is
    defined by the *left* tail.

    Every column is causal (``ewm`` looks backwards only), so a row carries no
    information from its own future. Rows before the longest halflife has
    enough observations are dropped.

    Returns a frame with ``3 * len(halflives)`` columns, grouped by statistic
    with halflives ascending, so column 0 is the shortest-halflife mean return
    — the default ordering anchor for `fit_jump_model` (see its ``order_by``).
    """
    r = pd.Series(returns).astype(float).dropna()
    hls = [int(h) for h in halflives]
    if not hls or any(h <= 0 for h in hls):
        raise ValueError("halflives must be a non-empty sequence of positive ints")

    negative = r.clip(upper=0.0)
    means, devs, sortinos = {}, {}, {}
    for hl in hls:
        ew_mean = r.ewm(halflife=hl, min_periods=hl).mean()
        ew_dd = np.sqrt((negative ** 2).ewm(halflife=hl, min_periods=hl).mean())
        means["ret_ew_h{0}".format(hl)] = ew_mean
        devs["dd_ew_h{0}".format(hl)] = ew_dd
        sortinos["sortino_h{0}".format(hl)] = ew_mean / (ew_dd + _EPS)

    frame = pd.DataFrame({**means, **devs, **sortinos}, index=r.index)
    return frame.dropna()


# ----------------------------------------------------------------------------
# Fit result
# ----------------------------------------------------------------------------
@dataclass
class JumpModelFit:
    """Result of a jump-model fit.

    ``states`` are canonically ordered (see `fit_jump_model`), so state 0 is
    always the low-return / high-downside regime and comparisons across refits
    are stable. ``center_`` / ``scale_`` are the fit-time standardisation
    statistics — `assign_online` reuses them so a live row is scaled by
    in-sample numbers only.
    """

    states: np.ndarray             # int array, one state per row of features
    centroids: np.ndarray          # (n_states, n_features) in standardised space
    loss: float                    # DP objective: distortion + penalty * switches
    n_switches: int
    n_states: int
    jump_penalty: float
    feature_names: list[str]
    center_: np.ndarray            # per-feature mean used to standardise
    scale_: np.ndarray             # per-feature std used to standardise
    index: pd.Index
    labels: tuple[str, ...]
    n_iter: int                    # coordinate-descent passes of the winning init

    # -- convenience views ----------------------------------------------------
    def state_series(self) -> pd.Series:
        """States as an int Series indexed like the input features."""
        return pd.Series(self.states, index=self.index, name="state")

    def label_series(self) -> pd.Series:
        """States as human-readable labels."""
        return pd.Series([self.labels[s] for s in self.states],
                         index=self.index, name="regime")

    @property
    def current_state(self) -> int:
        return int(self.states[-1])

    @property
    def current_label(self) -> str:
        return self.labels[self.current_state]

    def switch_points(self) -> pd.Index:
        """Index entries at which the state changed (the regime breakpoints)."""
        changed = np.flatnonzero(np.diff(self.states) != 0) + 1
        return self.index[changed]

    def state_shares(self) -> dict[str, float]:
        """Fraction of observations spent in each state, keyed by label."""
        total = len(self.states)
        return {
            self.labels[k]: float(np.sum(self.states == k) / total)
            for k in range(self.n_states)
        }

    def to_dict(self) -> dict:
        """JSON-safe summary — the shape the `tool_usage_log` payload wants.

        Deliberately excludes the full state path and centroid matrix: this is
        an audit record of *what the model said*, not a serialised model.
        """
        return {
            "n_states": self.n_states,
            "jump_penalty": float(self.jump_penalty),
            "loss": float(self.loss),
            "n_switches": int(self.n_switches),
            "n_obs": int(len(self.states)),
            "current_state": self.current_state,
            "current_label": self.current_label,
            "state_shares": self.state_shares(),
            "features": list(self.feature_names),
            "asof": str(self.index[-1]),
        }


# ----------------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------------
def _standardize(Z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score columns. k-means is scale-sensitive and downside deviation lives
    on a very different scale from a Sortino ratio, so this is not optional."""
    center = Z.mean(axis=0)
    scale = Z.std(axis=0, ddof=0)
    scale = np.where(scale < _EPS, 1.0, scale)     # constant column -> leave as 0
    return (Z - center) / scale, center, scale


def _sq_dists(Z: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """(T, K) matrix of squared euclidean distances from each row to each centroid."""
    diff = Z[:, None, :] - centroids[None, :, :]
    return np.einsum("tkf,tkf->tk", diff, diff)


def _kmeanspp_init(Z: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding: spread the initial centroids out, weighted by squared
    distance to the nearest one already chosen."""
    n = Z.shape[0]
    centroids = np.empty((k, Z.shape[1]), dtype=float)
    centroids[0] = Z[rng.integers(n)]
    for i in range(1, k):
        d2 = _sq_dists(Z, centroids[:i]).min(axis=1)
        total = float(d2.sum())
        if total <= _EPS:                       # all points identical
            centroids[i] = Z[rng.integers(n)]
        else:
            centroids[i] = Z[rng.choice(n, p=d2 / total)]
    return centroids


def _dp_assign(dist: np.ndarray, jump_penalty: float) -> tuple[np.ndarray, float]:
    """
    Exact minimiser of ``sum_t dist[t, s_t] + jump_penalty * #switches`` by
    dynamic programming (a Viterbi pass over a fully-connected state graph
    whose transition cost is 0 to stay and ``jump_penalty`` to move).

        V[0, k] = dist[0, k]
        V[t, k] = dist[t, k] + min_j ( V[t-1, j] + penalty * 1[j != k] )

    Returns the argmin state path and the optimal objective value. Ties break
    to the lowest state index, which keeps the whole fit deterministic.
    """
    T, K = dist.shape
    V = np.empty((T, K), dtype=float)
    back = np.zeros((T, K), dtype=np.int64)
    V[0] = dist[0]

    switch_cost = jump_penalty * (1.0 - np.eye(K))      # (j, k): 0 on diagonal
    for t in range(1, T):
        # candidate[j, k] = V[t-1, j] + penalty * 1[j != k]
        candidate = V[t - 1][:, None] + switch_cost
        best_prev = np.argmin(candidate, axis=0)
        back[t] = best_prev
        V[t] = dist[t] + candidate[best_prev, np.arange(K)]

    states = np.empty(T, dtype=np.int64)
    states[T - 1] = int(np.argmin(V[T - 1]))
    for t in range(T - 1, 0, -1):
        states[t - 1] = back[t, states[t]]
    return states, float(V[T - 1].min())


def _update_centroids(Z: np.ndarray, states: np.ndarray, k: int,
                      previous: np.ndarray) -> np.ndarray:
    """Mean of the points assigned to each state. An empty state keeps its old
    centroid so it stays available to be re-acquired on a later pass rather
    than collapsing the model to fewer states."""
    centroids = previous.copy()
    for j in range(k):
        members = Z[states == j]
        if len(members):
            centroids[j] = members.mean(axis=0)
    return centroids


def _canonical_order(centroids: np.ndarray, states: np.ndarray,
                     order_col: Optional[int]) -> tuple[np.ndarray, np.ndarray]:
    """Relabel states so they are sorted ascending by one centroid coordinate.

    k-means labels are arbitrary; without this, "state 1" means something
    different on every refit and the expanding-window online path would emit
    nonsense. Ordering by the mean-return feature ascending makes state 0 the
    low-return regime, every time.
    """
    if order_col is None:
        return centroids, states
    order = np.argsort(centroids[:, order_col], kind="stable")
    remap = np.empty(len(order), dtype=np.int64)
    remap[order] = np.arange(len(order))        # old label -> new label
    return centroids[order], remap[states]


def _resolve_order_col(order_by: Union[int, str, None],
                       feature_names: list[str]) -> Optional[int]:
    if order_by is None:
        return None
    if isinstance(order_by, str):
        if order_by not in feature_names:
            raise KeyError("order_by column {0!r} not in features".format(order_by))
        return feature_names.index(order_by)
    col = int(order_by)
    if not 0 <= col < len(feature_names):
        raise IndexError("order_by index {0} out of range".format(col))
    return col


def _resolve_labels(labels: Optional[Sequence[str]], n_states: int) -> tuple[str, ...]:
    if labels is not None:
        if len(labels) != n_states:
            raise ValueError("labels must have exactly n_states entries")
        return tuple(str(x) for x in labels)
    if n_states == 2:
        return RISK_LABELS_2
    return tuple("S{0}".format(i) for i in range(n_states))


# ----------------------------------------------------------------------------
# Fit
# ----------------------------------------------------------------------------
def fit_jump_model(
    features: pd.DataFrame,
    n_states: int = 2,
    jump_penalty: float = 150.0,
    max_iter: int = 50,
    n_init: int = 10,
    random_state: int = 0,
    order_by: Union[int, str, None] = 0,
    labels: Optional[Sequence[str]] = None,
) -> JumpModelFit:
    """
    Fit a statistical jump model by coordinate descent (DP assignment /
    centroid update), restarted ``n_init`` times from k-means++ seeds.

    Args:
        features: causal feature frame, e.g. from `downside_features`. Rows
            with NaN are dropped. Standardised internally.
        n_states: number of regimes. 2 is what the literature validates.
        jump_penalty: cost charged per state transition, in units of squared
            standardised distance. 0 reduces the model to plain k-means (no
            memory); larger values buy persistence. The paper tunes this by
            walk-forward CV on strategy Sharpe — that belongs in a service
            layer, not here. The 150.0 default is deliberately conservative
            (few switches) rather than optimised.

            **This number is not scale-free.** The objective sums squared
            distances over every feature column, so the penalty that produces
            a given switch rate grows with the number of features. 150 is
            calibrated for the 9-column `downside_features` default
            (``DEFAULT_HALFLIVES``); change the feature set and it must be
            re-calibrated, not carried over.
        order_by: feature (index or name) whose centroid coordinate sorts the
            states ascending, so state 0 is reproducibly the low-return
            regime. ``None`` leaves raw k-means labelling — only do that if
            you never compare two fits.

    Raises:
        ValueError: on an empty/too-short frame, or a bad n_states.
    """
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a DataFrame")
    if n_states < 2:
        raise ValueError("n_states must be at least 2")
    if jump_penalty < 0:
        raise ValueError("jump_penalty must be non-negative")

    frame = features.dropna()
    if len(frame) < n_states * 2:
        raise ValueError(
            "need at least {0} rows to fit {1} states, got {2}".format(
                n_states * 2, n_states, len(frame))
        )

    feature_names = [str(c) for c in frame.columns]
    order_col = _resolve_order_col(order_by, feature_names)
    state_labels = _resolve_labels(labels, n_states)

    Z, center, scale = _standardize(frame.to_numpy(dtype=float))
    rng = np.random.default_rng(random_state)

    best: Optional[tuple[float, np.ndarray, np.ndarray, int]] = None
    for _ in range(max(1, n_init)):
        centroids = _kmeanspp_init(Z, n_states, rng)
        states: Optional[np.ndarray] = None
        used = 0
        for used in range(1, max_iter + 1):
            new_states, _ = _dp_assign(_sq_dists(Z, centroids), jump_penalty)
            if states is not None and np.array_equal(new_states, states):
                break
            states = new_states
            centroids = _update_centroids(Z, states, n_states, centroids)

        # Final assignment against the centroids we are about to store, so
        # `states` and `centroids` always describe the same partition (a
        # no-op once converged; it matters only if max_iter ran out).
        states, loss = _dp_assign(_sq_dists(Z, centroids), jump_penalty)
        if best is None or loss < best[0]:
            best = (loss, states, centroids, used)

    loss, states, centroids, n_iter = best      # type: ignore[misc]
    centroids, states = _canonical_order(centroids, states, order_col)

    return JumpModelFit(
        states=states,
        centroids=centroids,
        loss=float(loss),
        n_switches=int(np.sum(np.diff(states) != 0)),
        n_states=n_states,
        jump_penalty=float(jump_penalty),
        feature_names=feature_names,
        center_=center,
        scale_=scale,
        index=frame.index,
        labels=state_labels,
        n_iter=int(n_iter),
    )


# ----------------------------------------------------------------------------
# Online paths (no lookahead)
# ----------------------------------------------------------------------------
def assign_online(fit: JumpModelFit, features: pd.DataFrame,
                  previous_state: Optional[int] = None,
                  jump_penalty: Optional[float] = None) -> pd.Series:
    """
    Label new rows from a **frozen** fit, greedily and causally: each row goes
    to the state minimising ``||z - c_k||^2 + penalty * 1[k != previous]``.

    Causal by construction (one bar at a time, no knowledge of the future), so
    it lags the offline path around a turn. That lag is the honest cost of not
    cheating; do not "fix" it by re-running `fit_jump_model` over history and
    reading the last label as if it were live.

    ⚠ **The fit's `jump_penalty` does not transfer to this rule.** In the DP it
    is a *sequence-level* cost, paid once per switch and weighed against the
    distortion of every bar in the run that follows. Here it is charged against
    the distance advantage of a **single bar**, which is bounded and typically
    far smaller — with the calibrated default the state can never change, and
    this function silently returns a constant. Pass an explicit
    ``jump_penalty`` (on the order of the typical per-bar distance gap, so
    single digits to a few tens for the default feature set) if you want a
    responsive greedy label; pass 0 for pure nearest-centroid.

    For anything that needs to be both causal *and* correctly calibrated,
    prefer `online_state_sequence` — an expanding-window refit, which is what
    the source paper actually evaluates.

    Standardisation reuses the fit's ``center_``/``scale_``, never the new
    rows' own statistics.
    """
    frame = features.dropna()
    missing = [c for c in fit.feature_names if c not in frame.columns]
    if missing:
        raise KeyError("features missing columns: {0}".format(missing))
    frame = frame[fit.feature_names]
    if frame.empty:
        return pd.Series([], dtype=np.int64, name="state")

    Z = (frame.to_numpy(dtype=float) - fit.center_) / fit.scale_
    dist = _sq_dists(Z, fit.centroids)

    penalty = fit.jump_penalty if jump_penalty is None else float(jump_penalty)
    if penalty < 0:
        raise ValueError("jump_penalty must be non-negative")

    prev = fit.current_state if previous_state is None else int(previous_state)
    out = np.empty(len(Z), dtype=np.int64)
    for t in range(len(Z)):
        cost = dist[t] + penalty * (np.arange(fit.n_states) != prev)
        prev = int(np.argmin(cost))
        out[t] = prev
    return pd.Series(out, index=frame.index, name="state")


def online_state_sequence(
    features: pd.DataFrame,
    min_train: int = 250,
    refit_every: int = 1,
    **fit_kwargs,
) -> pd.Series:
    """
    Walk-forward regime labels with **no lookahead**: refit on an expanding
    window ``features[:t]`` and keep only the last label of each fit.

    This is the only sequence that may be fed to a backtest. ``fit_jump_model``
    places breakpoints using the whole sample — reading its ``states`` as a
    historical signal would be a lookahead bug, and a flattering one, because
    the DP puts the switch exactly on the turn.

    ``refit_every > 1`` refits every N bars and carries the intervening bars
    with `assign_online` from the most recent fit. That is still causal (a
    frozen model applied forward), just cheaper — the full refit is O(T) fits.
    Note it is also *stickier*: those bars are labelled by the greedy rule at
    the fit's own penalty, which at the calibrated default effectively holds
    the previous state until the next refit (see `assign_online`). Causal and
    slightly stale, never anticipatory — but a turn can be reported up to
    ``refit_every`` bars late on top of the model's own latency.

    Returns an int Series over ``features.index[min_train - 1:]``.
    """
    frame = features.dropna()
    if min_train < 4:
        raise ValueError("min_train must be at least 4")
    if refit_every < 1:
        raise ValueError("refit_every must be at least 1")
    if len(frame) < min_train:
        raise ValueError(
            "need at least min_train={0} rows, got {1}".format(min_train, len(frame))
        )

    out_index, out_states = [], []
    fit: Optional[JumpModelFit] = None
    bars_since_fit = 0

    for t in range(min_train, len(frame) + 1):
        window = frame.iloc[:t]
        if fit is None or bars_since_fit >= refit_every:
            fit = fit_jump_model(window, **fit_kwargs)
            state = fit.current_state
            bars_since_fit = 0
        else:
            # Frozen model, one bar forward — causal by construction.
            previous = out_states[-1]
            state = int(assign_online(fit, window.iloc[[-1]],
                                      previous_state=previous).iloc[0])
            bars_since_fit += 1
        out_index.append(frame.index[t - 1])
        out_states.append(state)

    return pd.Series(out_states, index=pd.Index(out_index), name="state")


# ----------------------------------------------------------------------------
# Description
# ----------------------------------------------------------------------------
def state_summary(fit: JumpModelFit, returns: pd.Series,
                  ann_factor: int = 252) -> pd.DataFrame:
    """
    Per-state descriptive stats over the returns the model was fitted on —
    what each regime actually *was*, for the UI and for sanity-checking that
    state 0 really is the ugly one.

    Returns a frame indexed by label with: n_obs, share, mean annualised
    return, annualised vol, annualised downside deviation, and mean spell
    length in bars.
    """
    r = pd.Series(returns).astype(float).reindex(fit.index).dropna()
    states = fit.state_series().reindex(r.index)

    # Mean spell length per state: group consecutive runs.
    spells: dict[int, list[int]] = {k: [] for k in range(fit.n_states)}
    run_state, run_len = int(fit.states[0]), 0
    for s in fit.states:
        if s == run_state:
            run_len += 1
        else:
            spells[run_state].append(run_len)
            run_state, run_len = int(s), 1
    spells[run_state].append(run_len)

    rows = []
    for k in range(fit.n_states):
        block = r[states == k]
        negative = block.clip(upper=0.0)
        rows.append({
            "state": k,
            "n_obs": int(len(block)),
            "share": float(len(block) / len(r)) if len(r) else 0.0,
            "ann_return": float(block.mean() * ann_factor) if len(block) else np.nan,
            "ann_vol": (float(block.std(ddof=0) * np.sqrt(ann_factor))
                        if len(block) else np.nan),
            "ann_downside_dev": (
                float(np.sqrt((negative ** 2).mean()) * np.sqrt(ann_factor))
                if len(block) else np.nan),
            "mean_spell": float(np.mean(spells[k])) if spells[k] else np.nan,
        })

    return pd.DataFrame(rows, index=pd.Index(list(fit.labels), name="regime"))
