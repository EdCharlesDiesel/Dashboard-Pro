"""Unit tests for src/core/jump_model.py — the statistical jump model
(Bemporad et al. 2018 / Shu, Yu & Mulvey 2024).

All synthetic data with known regime boundaries; no network, no Streamlit, no
sklearn. The properties under test are the ones the model is *bought* for:

  * jump_penalty=0 degenerates to plain k-means (no memory),
  * switches are monotonically non-increasing in jump_penalty,
  * a known breakpoint is recovered,
  * state labels are canonically ordered, so two fits are comparable,
  * the online path is genuinely causal — appending future data must not
    rewrite a single past label.

The last one is the important one. `fit_jump_model` is offline by construction
(the DP places every breakpoint using the whole sample), so anything
backtest-shaped has to come from `online_state_sequence`, and that claim needs
a test with teeth rather than a comment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.jump_model import (
    DEFAULT_HALFLIVES,
    RISK_LABELS_2,
    JumpModelFit,
    _sq_dists,
    assign_online,
    downside_features,
    fit_jump_model,
    log_returns,
    online_state_sequence,
    state_summary,
)

BLOCK = 220          # length of each synthetic regime
N_BLOCKS = 6
# Longest feature halflife -> how many leading rows downside_features drops.
WARMUP = max(DEFAULT_HALFLIVES)


def _regime_returns(seed: int = 11, block: int = BLOCK,
                    n_blocks: int = N_BLOCKS) -> pd.Series:
    """Alternating bear/bull blocks: bear = negative drift + high vol,
    bull = mild positive drift + low vol. Block 0 is bear."""
    rng = np.random.default_rng(seed)
    parts = []
    for i in range(n_blocks):
        bear = (i % 2 == 0)
        mu, sd = (-0.0015, 0.020) if bear else (0.0010, 0.006)
        parts.append(rng.normal(mu, sd, block))
    values = np.concatenate(parts)
    index = pd.bdate_range("2010-01-01", periods=len(values))
    return pd.Series(values, index=index, name="ret")


# Module-scoped: the synthetic series, its features and the default fit are
# deterministic and read-only, so building them once keeps this file from
# dominating a suite that otherwise runs in a few seconds. Tests must not
# mutate them.
@pytest.fixture(scope="module")
def regime_returns() -> pd.Series:
    return _regime_returns()


@pytest.fixture(scope="module")
def regime_features(regime_returns) -> pd.DataFrame:
    return downside_features(regime_returns)


@pytest.fixture(scope="module")
def default_fit(regime_features) -> JumpModelFit:
    """The canonical fit at library defaults — shared by every test that only
    reads it."""
    return fit_jump_model(regime_features, random_state=3)


def _true_boundary_positions(features: pd.DataFrame, returns: pd.Series) -> list[int]:
    """True regime boundaries expressed as positions in the *feature* index."""
    out = []
    for i in range(1, N_BLOCKS):
        ts = returns.index[i * BLOCK]
        pos = features.index.searchsorted(ts)
        if 0 < pos < len(features):
            out.append(int(pos))
    return out


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
class TestDownsideFeatures:
    def test_shape_and_column_names(self, regime_returns):
        f = downside_features(regime_returns, halflives=(5, 21))
        assert list(f.columns) == [
            "ret_ew_h5", "ret_ew_h21", "dd_ew_h5", "dd_ew_h21",
            "sortino_h5", "sortino_h21",
        ]
        # min_periods == longest halflife, so that many leading rows drop out.
        assert len(f) == len(regime_returns) - 21 + 1

    def test_first_column_is_shortest_halflife_mean_return(self, regime_features):
        assert regime_features.columns[0] == "ret_ew_h{0}".format(min(DEFAULT_HALFLIVES))

    def test_downside_deviation_ignores_upside(self):
        """A series with no negative returns has zero downside deviation, however
        violent the rallies — that asymmetry is the point of the feature."""
        up_only = pd.Series([0.05, 0.01, 0.20, 0.03, 0.11, 0.04, 0.09, 0.02])
        f = downside_features(up_only, halflives=(2,))
        assert (f["dd_ew_h2"] == 0.0).all()

    def test_downside_deviation_positive_when_losses_present(self, regime_returns):
        f = downside_features(regime_returns, halflives=(21,))
        assert (f["dd_ew_h21"] > 0).all()

    def test_is_causal(self, regime_returns):
        """A row must not move when later data is appended — ewm looks backwards
        only, and the whole no-lookahead story rests on it."""
        full = downside_features(regime_returns, halflives=(5, 21))
        truncated = downside_features(regime_returns.iloc[:600], halflives=(5, 21))
        shared = truncated.index
        pd.testing.assert_frame_equal(full.loc[shared], truncated)

    def test_no_nans_survive(self, regime_features):
        assert not regime_features.isna().any().any()

    def test_rejects_empty_halflives(self, regime_returns):
        with pytest.raises(ValueError):
            downside_features(regime_returns, halflives=())

    def test_rejects_non_positive_halflife(self, regime_returns):
        with pytest.raises(ValueError):
            downside_features(regime_returns, halflives=(5, 0))


class TestLogReturns:
    def test_matches_manual_diff(self):
        close = pd.Series([100.0, 101.0, 99.5, 103.0])
        got = log_returns(close)
        expected = np.log(close).diff().dropna()
        pd.testing.assert_series_equal(got, expected)
        assert len(got) == len(close) - 1

    def test_rejects_non_positive_prices(self):
        with pytest.raises(ValueError):
            log_returns(pd.Series([100.0, 0.0, 99.0]))


# ---------------------------------------------------------------------------
# Core fit properties
# ---------------------------------------------------------------------------
class TestFitJumpModel:
    def test_zero_penalty_is_plain_kmeans(self, regime_features):
        """With no transition cost the DP must reduce to nearest-centroid
        assignment — that equivalence is the sanity anchor for the whole DP."""
        fit = fit_jump_model(regime_features, jump_penalty=0.0, n_init=3, random_state=3)
        Z = (regime_features.to_numpy(dtype=float) - fit.center_) / fit.scale_
        nearest = _sq_dists(Z, fit.centroids).argmin(axis=1)
        assert np.array_equal(fit.states, nearest)

    def test_switches_are_non_increasing_in_penalty(self, regime_features):
        penalties = [0.0, 25.0, 150.0, 500.0]
        counts = [
            fit_jump_model(regime_features, jump_penalty=p, n_init=2,
                           random_state=3).n_switches
            for p in penalties
        ]
        assert counts == sorted(counts, reverse=True)
        # And the penalty must actually bite: unpenalised clustering of noisy
        # daily features flips far more than the penalised fit.
        assert counts[0] > counts[-1]

    def test_huge_penalty_collapses_to_one_state(self, regime_features):
        fit = fit_jump_model(regime_features, jump_penalty=1e9, n_init=2, random_state=3)
        assert fit.n_switches == 0
        assert len(np.unique(fit.states)) == 1

    def test_zero_penalty_flips_far_more_than_default(self, regime_features,
                                                      default_fit):
        """The headline claim: memory suppresses the chatter that plain
        clustering produces on the same features."""
        noisy = fit_jump_model(regime_features, jump_penalty=0.0, n_init=3,
                               random_state=3)
        assert noisy.n_switches > 5 * max(default_fit.n_switches, 1)

    def test_recovers_known_breakpoints(self, regime_features, regime_returns,
                                        default_fit):
        """Every true regime boundary gets a detected switch nearby, and no
        spurious extras. Tolerance is generous because the features are
        exponentially weighted — the model cannot see a turn before its own
        smoothing window has absorbed it."""
        fit = default_fit
        detected = [int(regime_features.index.searchsorted(ts))
                    for ts in fit.switch_points()]
        truth = _true_boundary_positions(regime_features, regime_returns)

        assert len(detected) == len(truth), (
            "expected {0} switches, got {1}".format(len(truth), len(detected))
        )
        for true_pos, got_pos in zip(truth, detected):
            assert abs(got_pos - true_pos) <= 50

    def test_detection_lags_never_leads(self, regime_features, regime_returns,
                                        default_fit):
        """Every breakpoint must be found *after* the true turn. A detected
        switch landing early would mean the features are peeking forward."""
        fit = default_fit
        detected = [int(regime_features.index.searchsorted(ts))
                    for ts in fit.switch_points()]
        truth = _true_boundary_positions(regime_features, regime_returns)
        for true_pos, got_pos in zip(truth, detected):
            assert got_pos >= true_pos

    def test_states_are_canonically_ordered(self, regime_returns, default_fit):
        """State 0 must be the low-return regime, always — otherwise labels are
        meaningless across refits and the online sequence is gibberish."""
        fit = default_fit
        aligned = regime_returns.reindex(fit.index)
        mean_by_state = [aligned[fit.states == k].mean() for k in range(fit.n_states)]
        assert mean_by_state[0] < mean_by_state[1]
        assert fit.centroids[0, 0] < fit.centroids[1, 0]

    def test_ordering_is_stable_across_seeds(self, regime_features):
        """k-means++ seeding is random; the canonical order must wash that out."""
        a = fit_jump_model(regime_features, random_state=0)
        b = fit_jump_model(regime_features, random_state=99)
        assert np.array_equal(a.states, b.states)

    def test_order_by_accepts_a_column_name(self, regime_features):
        fit = fit_jump_model(regime_features, order_by="ret_ew_h63", n_init=2,
                             random_state=3)
        col = regime_features.columns.get_loc("ret_ew_h63")
        assert fit.centroids[0, col] < fit.centroids[1, col]

    def test_custom_labels_are_used(self, regime_features):
        fit = fit_jump_model(regime_features, labels=("BEAR", "BULL"), n_init=2,
                             random_state=3)
        assert fit.labels == ("BEAR", "BULL")
        assert fit.current_label in ("BEAR", "BULL")
        assert set(fit.state_shares()) == {"BEAR", "BULL"}

    def test_identical_rows_do_not_break_seeding(self):
        """Degenerate input: every row the same, so k-means++ has no distance
        to sample from. Must fall back to a random pick, not divide by zero."""
        frame = pd.DataFrame(np.ones((20, 3)),
                             columns=["a", "b", "c"],
                             index=pd.bdate_range("2020-01-01", periods=20))
        fit = fit_jump_model(frame, n_init=2, random_state=3)
        assert np.isfinite(fit.loss)
        assert fit.n_switches == 0

    def test_order_by_none_leaves_raw_labels(self, regime_features):
        fit = fit_jump_model(regime_features, order_by=None, n_init=2, random_state=3)
        assert set(np.unique(fit.states)) <= {0, 1}

    def test_deterministic_for_same_seed(self, regime_features):
        a = fit_jump_model(regime_features, n_init=3, random_state=7)
        b = fit_jump_model(regime_features, n_init=3, random_state=7)
        assert np.array_equal(a.states, b.states)
        assert a.loss == pytest.approx(b.loss)

    def test_loss_equals_distortion_plus_penalty(self, regime_features):
        """The reported loss must be the actual objective, not just distortion."""
        fit = fit_jump_model(regime_features, jump_penalty=30.0, n_init=2, random_state=3)
        Z = (regime_features.to_numpy(dtype=float) - fit.center_) / fit.scale_
        dist = _sq_dists(Z, fit.centroids)
        distortion = dist[np.arange(len(Z)), fit.states].sum()
        expected = distortion + 30.0 * fit.n_switches
        assert fit.loss == pytest.approx(expected, rel=1e-9)

    def test_three_states_supported(self, regime_features):
        fit = fit_jump_model(regime_features, n_states=3, n_init=3, random_state=3)
        assert fit.n_states == 3
        assert fit.centroids.shape == (3, regime_features.shape[1])
        assert fit.labels == ("S0", "S1", "S2")
        # canonical order still holds on the anchor column
        assert fit.centroids[0, 0] < fit.centroids[1, 0] < fit.centroids[2, 0]

    def test_standardisation_stats_recorded(self, regime_features, default_fit):
        fit = default_fit
        np.testing.assert_allclose(fit.center_,
                                   regime_features.to_numpy(dtype=float).mean(axis=0))
        assert (fit.scale_ > 0).all()

    def test_constant_feature_column_does_not_blow_up(self, regime_features):
        frame = regime_features.copy()
        frame["flat"] = 1.0
        fit = fit_jump_model(frame, n_init=2, random_state=3)
        assert np.isfinite(fit.loss)
        assert fit.scale_[-1] == 1.0        # guarded, not divided by ~0


class TestFitValidation:
    def test_rejects_non_dataframe(self, regime_returns):
        with pytest.raises(TypeError):
            fit_jump_model(regime_returns)

    def test_rejects_single_state(self, regime_features):
        with pytest.raises(ValueError):
            fit_jump_model(regime_features, n_states=1)

    def test_rejects_negative_penalty(self, regime_features):
        with pytest.raises(ValueError):
            fit_jump_model(regime_features, jump_penalty=-1.0)

    def test_rejects_too_few_rows(self, regime_features):
        with pytest.raises(ValueError):
            fit_jump_model(regime_features.iloc[:3], n_states=2)

    def test_rejects_unknown_order_by_name(self, regime_features):
        with pytest.raises(KeyError):
            fit_jump_model(regime_features, order_by="not_a_column")

    def test_rejects_out_of_range_order_by_index(self, regime_features):
        with pytest.raises(IndexError):
            fit_jump_model(regime_features, order_by=99)

    def test_rejects_mismatched_label_count(self, regime_features):
        with pytest.raises(ValueError):
            fit_jump_model(regime_features, n_states=2, labels=("only_one",))


# ---------------------------------------------------------------------------
# Fit result accessors
# ---------------------------------------------------------------------------
class TestJumpModelFitAccessors:
    @pytest.fixture
    def fit(self, default_fit) -> JumpModelFit:
        return default_fit

    def test_state_and_label_series_align_to_features(self, fit, regime_features):
        s = fit.state_series()
        assert s.index.equals(regime_features.index)
        labels = fit.label_series()
        assert set(labels.unique()) <= set(RISK_LABELS_2)
        assert labels.iloc[0] == fit.labels[fit.states[0]]

    def test_current_state_is_last_row(self, fit):
        assert fit.current_state == int(fit.states[-1])
        assert fit.current_label == fit.labels[fit.states[-1]]

    def test_switch_points_match_state_changes(self, fit):
        assert len(fit.switch_points()) == fit.n_switches

    def test_state_shares_sum_to_one(self, fit):
        shares = fit.state_shares()
        assert set(shares) == set(RISK_LABELS_2)
        assert sum(shares.values()) == pytest.approx(1.0)

    def test_to_dict_is_json_safe(self, fit):
        import json
        payload = fit.to_dict()
        json.dumps(payload)          # must not raise
        assert payload["n_states"] == 2
        assert payload["current_label"] in RISK_LABELS_2
        assert payload["n_obs"] == len(fit.states)
        # It is an audit record, not a serialised model.
        assert "centroids" not in payload
        assert "states" not in payload


# ---------------------------------------------------------------------------
# Online paths — the no-lookahead contract
# ---------------------------------------------------------------------------
class TestAssignOnline:
    @pytest.fixture(scope="class")
    def fit(self, regime_features):
        return fit_jump_model(regime_features.iloc[:600], random_state=3)

    def test_labels_new_rows(self, fit, regime_features):
        out = assign_online(fit, regime_features.iloc[600:700])
        assert len(out) == 100
        assert out.index.equals(regime_features.index[600:700])
        assert set(out.unique()) <= {0, 1}

    # A window that genuinely straddles a regime boundary, so labels can vary.
    MIXED = slice(300, 500)

    def test_penalty_makes_assignment_sticky(self, fit, regime_features):
        """Same rows, same frozen model: a larger transition cost must produce
        no more switches than a smaller one."""
        rows = regime_features.iloc[self.MIXED]
        counts = [
            int((assign_online(fit, rows, jump_penalty=p).diff().fillna(0) != 0).sum())
            for p in (0.0, 5.0, 20.0, 150.0)
        ]
        assert counts == sorted(counts, reverse=True)

    def test_fit_penalty_freezes_the_greedy_rule(self, fit, regime_features):
        """Regression guard on a real trap: the DP penalty is a sequence-level
        cost, so charging it against one bar's distance advantage (bounded,
        ~50 here) can never be overcome. Inherited unchanged, `assign_online`
        returns a constant — which is why it takes an override and why
        `online_state_sequence` is the recommended online path."""
        rows = regime_features.iloc[self.MIXED]
        frozen = assign_online(fit, rows)
        assert frozen.nunique() == 1
        assert frozen.iloc[0] == fit.current_state
        # The same rows are genuinely mixed once the penalty is sane.
        assert assign_online(fit, rows, jump_penalty=0.0).nunique() == 2

    def test_zero_penalty_is_nearest_centroid(self, fit, regime_features):
        rows = regime_features.iloc[self.MIXED]
        got = assign_online(fit, rows, jump_penalty=0.0)

        Z = (rows.to_numpy(dtype=float) - fit.center_) / fit.scale_
        nearest = _sq_dists(Z, fit.centroids).argmin(axis=1)
        assert np.array_equal(got.to_numpy(), nearest)

    def test_respects_explicit_previous_state(self, fit, regime_features):
        rows = regime_features.iloc[self.MIXED]
        from_zero = assign_online(fit, rows, previous_state=0, jump_penalty=20.0)
        from_one = assign_online(fit, rows, previous_state=1, jump_penalty=20.0)
        # Stickiness can only ever pull toward the state you came from.
        assert (from_zero <= from_one).all()

    def test_uses_fit_time_scaling_only(self, fit, regime_features):
        """Standardising new rows by their *own* mean/std would leak the
        window's future into every row in it. Output must come from the
        frozen fit-time statistics."""
        rows = regime_features.iloc[self.MIXED]
        got = assign_online(fit, rows, jump_penalty=0.0).to_numpy()

        raw = rows.to_numpy(dtype=float)
        self_scaled = (raw - raw.mean(axis=0)) / raw.std(axis=0)
        leaky = _sq_dists(self_scaled, fit.centroids).argmin(axis=1)
        assert not np.array_equal(got, leaky)

    def test_rejects_negative_penalty_override(self, fit, regime_features):
        with pytest.raises(ValueError):
            assign_online(fit, regime_features.iloc[600:610], jump_penalty=-1.0)

    def test_missing_columns_raise(self, fit, regime_features):
        with pytest.raises(KeyError):
            assign_online(fit, regime_features.iloc[600:700].drop(columns=["dd_ew_h5"]))

    def test_empty_frame_returns_empty_series(self, fit, regime_features):
        out = assign_online(fit, regime_features.iloc[0:0])
        assert out.empty


class TestOnlineStateSequence:
    """The causality contract. These are the tests that stop a lookahead bug
    from turning into a flattering backtest."""

    FIT_KW = dict(n_init=2, max_iter=8, random_state=3)

    def test_appending_future_data_never_rewrites_the_past(self, regime_features):
        """The real no-lookahead test: labels computed on a short history must
        be byte-identical to the same dates computed on a longer history."""
        short = online_state_sequence(regime_features.iloc[:230],
                                      min_train=200, **self.FIT_KW)
        long = online_state_sequence(regime_features.iloc[:290],
                                     min_train=200, **self.FIT_KW)
        pd.testing.assert_series_equal(long.loc[short.index], short)

    def test_holds_under_refit_every(self, regime_features):
        """Carrying a frozen fit forward between refits must stay causal too."""
        kw = dict(self.FIT_KW)
        short = online_state_sequence(regime_features.iloc[:230], min_train=200,
                                      refit_every=5, **kw)
        long = online_state_sequence(regime_features.iloc[:290], min_train=200,
                                     refit_every=5, **kw)
        pd.testing.assert_series_equal(long.loc[short.index], short)

    def test_index_starts_at_min_train(self, regime_features):
        seq = online_state_sequence(regime_features.iloc[:260], min_train=200,
                                    **self.FIT_KW)
        assert seq.index[0] == regime_features.index[199]
        assert seq.index[-1] == regime_features.index[259]
        assert len(seq) == 61

    def test_last_label_matches_a_full_fit_on_the_same_window(self, regime_features):
        window = regime_features.iloc[:260]
        seq = online_state_sequence(window, min_train=200, **self.FIT_KW)
        offline = fit_jump_model(window, **self.FIT_KW)
        assert seq.iloc[-1] == offline.current_state

    def test_rejects_short_history(self, regime_features):
        with pytest.raises(ValueError):
            online_state_sequence(regime_features.iloc[:50], min_train=200)

    def test_rejects_bad_arguments(self, regime_features):
        with pytest.raises(ValueError):
            online_state_sequence(regime_features, min_train=2)
        with pytest.raises(ValueError):
            online_state_sequence(regime_features, min_train=200, refit_every=0)


# ---------------------------------------------------------------------------
# Description
# ---------------------------------------------------------------------------
class TestStateSummary:
    @pytest.fixture(scope="class")
    def summary(self, default_fit, regime_returns):
        return state_summary(default_fit, regime_returns)

    def test_shape_and_index(self, summary):
        assert list(summary.index) == list(RISK_LABELS_2)
        assert set(summary.columns) == {
            "state", "n_obs", "share", "ann_return", "ann_vol",
            "ann_downside_dev", "mean_spell",
        }

    def test_risk_off_state_is_the_ugly_one(self, summary):
        """Sanity check that the canonical ordering means what it says."""
        off, on = summary.loc["RISK_OFF"], summary.loc["RISK_ON"]
        assert off["ann_return"] < on["ann_return"]
        assert off["ann_vol"] > on["ann_vol"]
        assert off["ann_downside_dev"] > on["ann_downside_dev"]

    def test_shares_sum_to_one(self, summary):
        assert summary["share"].sum() == pytest.approx(1.0)

    def test_mean_spell_reflects_persistence(self, summary):
        """Blocks are BLOCK bars long, so spells should be on that order — the
        single most legible proof that the penalty is doing its job."""
        assert (summary["mean_spell"] > BLOCK * 0.5).all()

    def test_n_obs_sums_to_sample(self, summary, regime_features):
        assert summary["n_obs"].sum() == len(regime_features)
