"""
Martingale & concentration diagnostics for trading P&L sequences.

Pure numpy/scipy. No Streamlit, no DB — importable and unit-testable.

The mapping from the graph-theory construction to a trade log:

    Doob's construction   ->  mark-to-market: X_i = E[final P&L | info at i]
    Filtration F_i        ->  the trade log up to and including trade i
    Martingale property   ->  no predictable drift = no edge
    Doob decomposition    ->  P&L = compensator (edge) + martingale (luck)
    Azuma-Hoeffding       ->  worst-case envelope from bounded per-trade risk
    Freedman              ->  variance-adaptive envelope (much tighter in practice)
    McDiarmid             ->  how far one trade can move a reported metric
    Optional stopping     ->  why "I'll exit when I'm up" is not an edge

References
----------
Alon & Spencer, *The Probabilistic Method*, ch. 7 (edge/vertex exposure).
Freedman (1975), Ann. Probab. 3(1), 100-118.
Lo & MacKinlay (1988), Rev. Financ. Stud. 1(1), 41-66.
McDiarmid (1989), *Surveys in Combinatorics*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
from scipy import stats

__all__ = [
    "DoobResult",
    "BandResult",
    "MartingaleTests",
    "doob_decomposition",
    "azuma_band",
    "freedman_band",
    "variance_ratio",
    "martingale_tests",
    "bounded_differences_bound",
    "azuma_risk_budget",
    "optional_stopping_mc",
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _as_1d(x: Sequence[float] | np.ndarray, name: str = "x") -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    if a.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains NaN or inf")
    return a


def _check_delta(delta: float) -> float:
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    return float(delta)


# --------------------------------------------------------------------------
# Doob decomposition:  S_n = S_0 + A_n (predictable) + M_n (martingale)
# --------------------------------------------------------------------------

@dataclass
class DoobResult:
    """Split of a cumulative P&L path into edge and luck."""

    cumulative: np.ndarray      # S_i, the realised equity curve (S_0 = 0)
    compensator: np.ndarray     # A_i, predictable part  -> "edge"
    martingale: np.ndarray      # M_i, martingale part   -> "luck"
    drift: np.ndarray           # per-step conditional mean estimate
    method: str

    @property
    def edge_share(self) -> float:
        """Fraction of final P&L attributable to predictable drift.

        >1 means luck ran against a real edge; <0 means the drift estimate
        points the other way from the realised outcome.
        """
        final = self.cumulative[-1]
        if abs(final) < 1e-12:
            return float("nan")
        return float(self.compensator[-1] / final)


def doob_decomposition(
    increments: Sequence[float] | np.ndarray,
    method: Literal["constant", "expanding", "rolling"] = "expanding",
    window: int | None = 50,
    min_periods: int = 10,
) -> DoobResult:
    """Decompose a P&L increment series into compensator + martingale.

    The compensator A_i = sum_{j<=i} E[dS_j | F_{j-1}] is the part you could
    in principle have predicted; the remainder M_i is a martingale by
    construction. Any honest estimate of the conditional mean must use only
    information strictly before step j, hence the shift.

    Parameters
    ----------
    increments : per-trade (or per-bar) P&L.
    method :
        "constant"  - single full-sample mean. Uses future data; diagnostic
                      only, never for live attribution.
        "expanding" - mean of everything strictly before step j. Honest.
        "rolling"   - trailing `window` mean, strictly before step j. Honest,
                      and adapts if the edge decays.
    window : lookback for method="rolling".
    min_periods : steps before the drift estimate is trusted; earlier steps
        get drift 0 (i.e. treated as pure martingale).
    """
    x = _as_1d(increments, "increments")
    n = x.size

    if method == "constant":
        drift = np.full(n, x.mean())
    elif method in ("expanding", "rolling"):
        drift = np.zeros(n)
        csum = np.concatenate(([0.0], np.cumsum(x)))  # csum[k] = sum of first k
        for j in range(n):
            avail = j  # observations strictly before j
            if avail < min_periods:
                continue
            if method == "expanding":
                lo = 0
            else:
                lo = max(0, j - int(window or 50))
            drift[j] = (csum[j] - csum[lo]) / (j - lo)
    else:
        raise ValueError(f"unknown method {method!r}")

    compensator = np.cumsum(drift)
    cumulative = np.cumsum(x)
    martingale = cumulative - compensator

    return DoobResult(
        cumulative=cumulative,
        compensator=compensator,
        martingale=martingale,
        drift=drift,
        method=method,
    )


# --------------------------------------------------------------------------
# Concentration envelopes
# --------------------------------------------------------------------------

@dataclass
class BandResult:
    """Running one-sided radius of a concentration envelope."""

    radius: np.ndarray          # r_i such that |M_i| <= r_i w.p. >= 1 - delta
    delta: float
    name: str
    meta: dict = field(default_factory=dict)

    def breaches(self, path: np.ndarray) -> np.ndarray:
        """Indices where the path escaped the band (should be rare)."""
        return np.flatnonzero(np.abs(np.asarray(path)) > self.radius)


def azuma_band(
    c: float | Sequence[float] | np.ndarray,
    n: int | None = None,
    delta: float = 0.05,
    two_sided: bool = True,
) -> BandResult:
    """Azuma-Hoeffding envelope for a martingale with |dM_i| <= c_i.

        P(|M_n| >= t) <= 2 exp(-t^2 / (2 sum c_i^2))

    inverted for t at confidence 1 - delta. The same constant holds for the
    running maximum (Doob's maximal inequality applied to exp(lambda M)), so
    plotting this as a path envelope is legitimate, not just pointwise.

    Parameters
    ----------
    c : scalar bound on every increment, or a per-step array of bounds.
    n : number of steps (required if `c` is scalar).
    delta : tail probability.
    two_sided : if True the bound carries the factor 2.
    """
    delta = _check_delta(delta)
    c_arr = np.asarray(c, dtype=float).ravel()
    if c_arr.size == 1:
        if n is None:
            raise ValueError("n is required when c is a scalar")
        c_arr = np.full(int(n), float(c_arr[0]))
    if (c_arr <= 0).any():
        raise ValueError("all c_i must be strictly positive")

    tail = delta / 2.0 if two_sided else delta
    v = np.cumsum(c_arr**2)
    radius = np.sqrt(2.0 * v * np.log(1.0 / tail))

    return BandResult(
        radius=radius,
        delta=delta,
        name="Azuma-Hoeffding",
        meta={"sum_c2_final": float(v[-1]), "two_sided": two_sided},
    )


def freedman_band(
    increments: Sequence[float] | np.ndarray,
    delta: float = 0.05,
    R: float | None = None,
    two_sided: bool = True,
    use_realised_qv: bool = True,
) -> BandResult:
    """Freedman envelope — Azuma but adaptive to realised variance.

        P(M_n >= t, V_n <= v) <= exp(-t^2 / (2(v + R t / 3)))

    solved for t:  t = LR/3 + sqrt((LR/3)^2 + 2 L v),  L = ln(1/tail).

    In the strict statement V_n is the *predictable* quadratic variation
    sum E[dM_i^2 | F_{i-1}]. Here it is replaced by the realised sum of
    squared increments, which is the usual practical substitution but makes
    the band an estimate rather than a theorem. Set use_realised_qv=False to
    use the worst-case n R^2 instead, which recovers something Azuma-like.

    This is the band that matters for a real trade log: Azuma has to assume
    every trade is a full-size loser, whereas most are not.
    """
    delta = _check_delta(delta)
    x = _as_1d(increments, "increments")
    n = x.size

    if R is None:
        R = float(np.max(np.abs(x)))
    R = float(R)
    if R <= 0:
        raise ValueError("R must be positive")

    v = np.cumsum(x**2) if use_realised_qv else np.arange(1, n + 1) * R**2

    tail = delta / 2.0 if two_sided else delta
    L = np.log(1.0 / tail)
    a = L * R / 3.0
    radius = a + np.sqrt(a**2 + 2.0 * L * v)

    return BandResult(
        radius=radius,
        delta=delta,
        name="Freedman",
        meta={"R": R, "V_final": float(v[-1]), "realised_qv": use_realised_qv},
    )


# --------------------------------------------------------------------------
# Tests of the martingale property itself
# --------------------------------------------------------------------------

def variance_ratio(
    increments: Sequence[float] | np.ndarray, q: int = 4
) -> tuple[float, float, float]:
    """Lo-MacKinlay variance ratio with heteroskedasticity-robust z.

    Under a martingale, variance scales linearly in horizon and VR(q) -> 1.
    VR > 1 indicates positive autocorrelation (trending / momentum);
    VR < 1 indicates mean reversion.

    Returns (VR, z, p_two_sided). The robust statistic is used because P&L
    increments are emphatically not homoskedastic.
    """
    x = _as_1d(increments, "increments")
    T = x.size
    if q < 2:
        raise ValueError("q must be >= 2")
    if T < q + 2:
        raise ValueError(f"need at least {q + 2} observations for q={q}")

    mu = x.mean()
    d = x - mu
    ss = np.sum(d**2)
    sigma_a = ss / (T - 1)

    # overlapping q-period sums
    csum = np.concatenate(([0.0], np.cumsum(x)))
    y = csum[q:] - csum[:-q]                      # length T - q + 1
    m = q * (T - q + 1) * (1.0 - q / T)
    sigma_c = np.sum((y - q * mu) ** 2) / m

    vr = sigma_c / sigma_a

    # heteroskedasticity-consistent variance of VR
    theta = 0.0
    for j in range(1, q):
        num = np.sum(d[j:] ** 2 * d[:-j] ** 2)
        delta_j = num / (ss**2)
        theta += (2.0 * (q - j) / q) ** 2 * delta_j

    z = (vr - 1.0) / np.sqrt(theta) if theta > 0 else np.nan
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    return float(vr), float(z), float(p)


@dataclass
class MartingaleTests:
    n: int
    mean_increment: float
    drift_t: float
    drift_p: float
    ljung_box_stat: float
    ljung_box_p: float
    vr_q: int
    vr: float
    vr_z: float
    vr_p: float

    @property
    def verdict(self) -> str:
        signals = []
        if self.drift_p < 0.05:
            signals.append("drift" if self.mean_increment > 0 else "negative drift")
        if self.ljung_box_p < 0.05:
            signals.append("serial correlation")
        if self.vr_p < 0.05:
            signals.append("momentum" if self.vr > 1 else "mean reversion")
        if not signals:
            return "consistent with a martingale (no detectable edge)"
        return "martingale rejected: " + ", ".join(signals)


def martingale_tests(
    increments: Sequence[float] | np.ndarray, lags: int = 10, q: int = 4
) -> MartingaleTests:
    """Battery testing H0: E[dS_i | F_{i-1}] = 0.

    Three angles, because each misses something the others catch:
      * t-test on the mean increment  - detects constant drift
      * Ljung-Box on increments       - martingale differences are uncorrelated
      * variance ratio                - detects longer-horizon dependence that
                                        low-order autocorrelation can hide

    Failing to reject is not proof of no edge; it is failure to find one at
    this sample size.
    """
    x = _as_1d(increments, "increments")
    n = x.size

    t_stat, t_p = stats.ttest_1samp(x, 0.0)

    lags = int(min(lags, max(1, n // 5)))
    d = x - x.mean()
    denom = np.sum(d**2)
    lb = 0.0
    for k in range(1, lags + 1):
        rk = np.sum(d[k:] * d[:-k]) / denom
        lb += rk**2 / (n - k)
    lb *= n * (n + 2)
    lb_p = float(1.0 - stats.chi2.cdf(lb, df=lags))

    try:
        vr, vr_z, vr_p = variance_ratio(x, q=q)
    except ValueError:
        vr, vr_z, vr_p = float("nan"), float("nan"), float("nan")

    return MartingaleTests(
        n=n,
        mean_increment=float(x.mean()),
        drift_t=float(t_stat),
        drift_p=float(t_p),
        ljung_box_stat=float(lb),
        ljung_box_p=lb_p,
        vr_q=q,
        vr=vr,
        vr_z=vr_z,
        vr_p=vr_p,
    )


# --------------------------------------------------------------------------
# McDiarmid: how fragile is a reported metric?
# --------------------------------------------------------------------------

def bounded_differences_bound(
    values: Sequence[float] | np.ndarray,
    metric,
    delta: float = 0.05,
) -> dict:
    """McDiarmid bound on a metric via jackknife bounded differences.

    Estimates c_i as the change in `metric` when observation i is dropped and
    replaced by the sample median — a jackknife proxy for the true worst-case
    bounded difference. Then

        P(|f - E f| >= t) <= 2 exp(-2 t^2 / sum c_i^2)

    Use this to answer "is my Sharpe of 1.4 a real number or one lucky trade?"
    The bound is conservative; a small radius is strong evidence, a large one
    is inconclusive rather than damning.
    """
    delta = _check_delta(delta)
    x = _as_1d(values, "values")
    n = x.size
    base = float(metric(x))
    med = float(np.median(x))

    c = np.empty(n)
    for i in range(n):
        y = x.copy()
        y[i] = med
        c[i] = abs(float(metric(y)) - base)

    sum_c2 = float(np.sum(c**2))
    radius = float(np.sqrt(sum_c2 * np.log(2.0 / delta) / 2.0)) if sum_c2 > 0 else 0.0

    return {
        "metric": base,
        "radius": radius,
        "lower": base - radius,
        "upper": base + radius,
        "max_single_influence": float(c.max()),
        "most_influential_index": int(np.argmax(c)),
        "sum_c2": sum_c2,
        "delta": delta,
    }


# --------------------------------------------------------------------------
# Position sizing straight out of Azuma
# --------------------------------------------------------------------------

def azuma_risk_budget(
    max_drawdown: float, n_trades: int, delta: float = 0.05
) -> dict:
    """Largest per-trade risk c such that cumulative P&L stays inside
    +/- max_drawdown over n trades with probability >= 1 - delta,
    assuming no edge (pure martingale).

        c <= D / sqrt(2 n ln(2/delta))

    Note the sqrt(n): doubling the trade count only raises the required
    envelope by ~41%, it does not double it. And note what this is *not* —
    it is a distribution-free worst-case bound assuming every trade could be
    a full loser. Freedman on your actual log will give a far smaller number.
    """
    if max_drawdown <= 0:
        raise ValueError("max_drawdown must be positive")
    if n_trades < 1:
        raise ValueError("n_trades must be >= 1")
    delta = _check_delta(delta)

    L = np.log(2.0 / delta)
    c = max_drawdown / np.sqrt(2.0 * n_trades * L)
    return {
        "risk_per_trade": float(c),
        "max_drawdown": float(max_drawdown),
        "n_trades": int(n_trades),
        "delta": delta,
        "envelope_at_n": float(c * np.sqrt(2.0 * n_trades * L)),
    }


# --------------------------------------------------------------------------
# Optional stopping: the "I'll exit when I'm up" test
# --------------------------------------------------------------------------

def optional_stopping_mc(
    increments: Sequence[float] | np.ndarray,
    take_profit: float,
    stop_loss: float,
    max_steps: int = 500,
    n_sims: int = 20_000,
    demean: bool = True,
    seed: int | None = 42,
) -> dict:
    """Bootstrap a target/stop exit rule against the optional stopping theorem.

    Resamples the empirical increment distribution, applies the exit rule, and
    reports the mean stopped value. If `demean=True` the increments are
    centred first, so the underlying process is a genuine martingale and the
    theorem predicts a mean of exactly zero regardless of the rule. Any
    apparent profitability in that case is the high win-rate illusion: many
    small wins exactly offset by rare large losses.

    Set demean=False to measure what the rule does to your *actual* edge — a
    tight stop can destroy a real edge, which is a separate question.
    """
    x = _as_1d(increments, "increments")
    if take_profit <= 0 or stop_loss <= 0:
        raise ValueError("take_profit and stop_loss must be positive magnitudes")

    pool = x - x.mean() if demean else x
    rng = np.random.default_rng(seed)

    draws = rng.choice(pool, size=(n_sims, max_steps), replace=True)
    paths = np.cumsum(draws, axis=1)

    hit_tp = paths >= take_profit
    hit_sl = paths <= -stop_loss
    hit_any = hit_tp | hit_sl

    has_exit = hit_any.any(axis=1)
    first = np.where(has_exit, hit_any.argmax(axis=1), max_steps - 1)
    rows = np.arange(n_sims)
    stopped = paths[rows, first]

    won = has_exit & hit_tp[rows, first]
    se = float(stopped.std(ddof=1) / np.sqrt(n_sims))
    mean = float(stopped.mean())

    return {
        "mean_stopped_value": mean,
        "std_error": se,
        "z_vs_zero": mean / se if se > 0 else float("nan"),
        "win_rate": float(won.mean()),
        "mean_holding_steps": float((first + 1).mean()),
        "timeout_rate": float((~has_exit).mean()),
        "demeaned": demean,
        "take_profit": float(take_profit),
        "stop_loss": float(stop_loss),
        "n_sims": int(n_sims),
    }
