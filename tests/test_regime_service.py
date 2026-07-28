"""Unit tests for src/services/regime_service.py — spine→jump-model wiring.

The spine fetcher is monkeypatched in the service's namespace, so no network,
DB or Streamlit runtime is touched. Distinct registry instruments are used per
test because ``st.cache_data`` memoizes across calls even in bare mode.

The tests with teeth are in `TestRegimeStrategyReturns`: the one-day trading
delay and the transaction-cost charge are the two details that separate an
honest regime backtest from a flattering one, and both are easy to lose in a
refactor without any test noticing.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.core.jump_model import downside_features, fit_jump_model, log_returns
from src.services import regime_service as rs
from src.services.regime_service import (
    PenaltyTuning,
    RegimeRead,
    _current_spell,
    annualised_sharpe,
    get_regime,
    get_regime_history,
    log_regime,
    read_from_fit,
    regime_inputs,
    regime_strategy_returns,
    tune_jump_penalty,
)

BLOCK = 150
N_BLOCKS = 6


def _regime_close(seed: int = 5, block: int = BLOCK,
                  n_blocks: int = N_BLOCKS) -> pd.Series:
    """Price series with alternating bear (negative drift, high vol) and bull
    (positive drift, low vol) blocks."""
    rng = np.random.default_rng(seed)
    parts = []
    for i in range(n_blocks):
        mu, sd = (-0.0015, 0.020) if i % 2 == 0 else (0.0010, 0.006)
        parts.append(rng.normal(mu, sd, block))
    rets = np.concatenate(parts)
    close = 100.0 * np.exp(np.cumsum(rets))
    idx = pd.bdate_range("2015-01-01", periods=len(close))
    return pd.Series(close, index=idx)


def _ohlc(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": close, "High": close * 1.002, "Low": close * 0.998,
            "Close": close, "Volume": np.full(len(close), 1000.0),
        },
        index=close.index,
    )


@pytest.fixture(scope="module")
def daily_frame() -> pd.DataFrame:
    return _ohlc(_regime_close())


@pytest.fixture(scope="module")
def inputs(daily_frame):
    return regime_inputs(daily_frame)


@pytest.fixture(scope="module")
def fit(inputs):
    features, _ = inputs
    return fit_jump_model(features, random_state=3)


# ---------------------------------------------------------------------------
# regime_inputs
# ---------------------------------------------------------------------------
class TestRegimeInputs:
    def test_builds_features_and_returns(self, daily_frame):
        features, returns = regime_inputs(daily_frame)
        assert len(returns) == len(daily_frame) - 1
        assert list(features.columns) == list(
            downside_features(returns).columns
        )
        assert len(features) > 0
        assert features.index[-1] == returns.index[-1]

    def test_features_are_a_subset_of_returns_index(self, inputs):
        features, returns = inputs
        assert features.index.isin(returns.index).all()

    def test_empty_frame_gives_empty_outputs(self):
        features, returns = regime_inputs(pd.DataFrame())
        assert features.empty and returns.empty

    def test_none_frame_gives_empty_outputs(self):
        features, returns = regime_inputs(None)
        assert features.empty and returns.empty

    def test_missing_close_column_gives_empty_outputs(self):
        frame = pd.DataFrame({"Open": [1.0, 2.0], "High": [1.0, 2.0]})
        features, returns = regime_inputs(frame)
        assert features.empty and returns.empty

    def test_single_row_gives_empty_outputs(self):
        frame = pd.DataFrame({"Close": [100.0]},
                             index=pd.bdate_range("2020-01-01", periods=1))
        features, returns = regime_inputs(frame)
        assert features.empty and returns.empty

    def test_non_positive_prices_are_dropped_not_fatal(self):
        """A zero/negative print would make log returns explode; it must be
        filtered rather than raise."""
        close = pd.Series(_regime_close(block=60, n_blocks=2).to_numpy())
        close.iloc[10] = 0.0
        frame = pd.DataFrame(
            {"Close": close.to_numpy()},
            index=pd.bdate_range("2020-01-01", periods=len(close)),
        )
        features, returns = regime_inputs(frame, halflives=(5, 21))
        assert np.isfinite(returns.to_numpy()).all()
        assert not features.isna().any().any()


# ---------------------------------------------------------------------------
# RegimeRead
# ---------------------------------------------------------------------------
class TestReadFromFit:
    def test_summarises_the_fit(self, fit):
        read = read_from_fit("XAU/USD", fit)
        assert read.instrument == "XAU/USD"
        assert read.label == fit.current_label
        assert read.state == fit.current_state
        assert read.n_obs == len(fit.states)
        assert read.n_switches == fit.n_switches
        assert read.asof == fit.index[-1]
        assert read.state_shares == fit.state_shares()

    def test_switches_per_year_uses_trading_days(self, fit):
        read = read_from_fit("XAU/USD", fit)
        expected = fit.n_switches / (len(fit.states) / rs.TRADING_DAYS)
        assert read.switches_per_year == pytest.approx(expected)

    def test_penalised_fit_switches_about_once_a_year(self, inputs):
        """The property the model is bought for, on the service's own path."""
        features, _ = inputs
        sticky = read_from_fit("XAU/USD", fit_jump_model(features, random_state=3))
        chatty = read_from_fit(
            "XAU/USD", fit_jump_model(features, jump_penalty=0.0, n_init=3,
                                      random_state=3),
        )
        assert sticky.switches_per_year < chatty.switches_per_year

    def test_is_risk_off_tracks_state_zero(self, fit):
        read = read_from_fit("XAU/USD", fit)
        assert read.is_risk_off == (read.state == 0)

    def test_to_payload_is_json_safe(self, fit):
        payload = read_from_fit("XAU/USD", fit).to_payload()
        json.dumps(payload)
        assert payload["instrument"] == "XAU/USD"
        assert payload["label"] in ("RISK_OFF", "RISK_ON")
        assert isinstance(payload["asof"], str)

    def test_to_payload_handles_missing_asof(self):
        read = RegimeRead(instrument="EUR/USD", label="RISK_ON", state=1,
                          n_states=2, jump_penalty=150.0, n_obs=0, n_switches=0,
                          switches_per_year=0.0, current_spell=0)
        payload = read.to_payload()
        json.dumps(payload)
        assert payload["asof"] is None


class TestCurrentSpell:
    def test_counts_trailing_run(self):
        assert _current_spell(np.array([0, 0, 1, 1, 1])) == 3

    def test_single_state_is_whole_sample(self):
        assert _current_spell(np.array([1, 1, 1, 1])) == 4

    def test_last_bar_alone(self):
        assert _current_spell(np.array([0, 0, 0, 1])) == 1

    def test_empty(self):
        assert _current_spell(np.array([], dtype=int)) == 0


# ---------------------------------------------------------------------------
# Strategy evaluation
# ---------------------------------------------------------------------------
class TestAnnualisedSharpe:
    def test_matches_manual_formula(self):
        r = pd.Series([0.01, -0.005, 0.02, 0.003, -0.001])
        expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
        assert annualised_sharpe(r) == pytest.approx(expected)

    def test_flat_stream_is_zero_not_nan(self):
        assert annualised_sharpe(pd.Series([0.01] * 10)) == 0.0

    def test_empty_is_zero(self):
        assert annualised_sharpe(pd.Series(dtype=float)) == 0.0

    def test_single_observation_is_zero(self):
        assert annualised_sharpe(pd.Series([0.01])) == 0.0


class TestRegimeStrategyReturns:
    IDX = pd.bdate_range("2020-01-01", periods=6)

    def _states(self, values):
        return pd.Series(values, index=self.IDX)

    def _returns(self, values):
        return pd.Series(values, index=self.IDX)

    def test_applies_one_day_trading_delay(self):
        """The signal turns risk-on at position 3; the position may only change
        at position 4. Without the delay the strategy would capture the same
        day's return, which is the classic way a regime backtest lies."""
        states = self._states([0, 0, 0, 1, 1, 1])
        returns = self._returns([0.01] * 6)
        strat = regime_strategy_returns(returns, states, cost_bps=0.0)

        assert strat.iloc[3] == pytest.approx(0.0)     # still flat on signal day
        assert strat.iloc[4] == pytest.approx(0.01)    # invested the day after

    def test_charges_cost_on_position_changes_only(self):
        states = self._states([0, 0, 0, 1, 1, 1])
        returns = self._returns([0.01] * 6)
        strat = regime_strategy_returns(returns, states, cost_bps=10.0)

        # One transition, at position 4: 10 bps off that day only.
        assert strat.iloc[4] == pytest.approx(0.01 - 0.001)
        assert strat.iloc[5] == pytest.approx(0.01)
        assert strat.iloc[3] == pytest.approx(0.0)

    def test_zero_cost_is_pure_position_times_return(self):
        states = self._states([1, 1, 0, 0, 1, 1])
        returns = self._returns([0.01, 0.02, -0.03, 0.04, -0.05, 0.06])
        strat = regime_strategy_returns(returns, states, cost_bps=0.0)
        # shift(1) of [1,1,0,0,1,1] -> [_,1,1,0,0,1], first bar flat.
        expected = (pd.Series([0.0, 1.0, 1.0, 0.0, 0.0, 1.0], index=self.IDX)
                    * returns)
        pd.testing.assert_series_equal(strat, expected)

    def test_starts_flat_before_the_first_signal(self):
        """There is no label to act on for the first bar, so the position must
        be the de-risked one — not silently invested."""
        states = self._states([1] * 6)
        returns = self._returns([0.01] * 6)
        strat = regime_strategy_returns(returns, states, cost_bps=0.0)
        assert strat.iloc[0] == pytest.approx(0.0)
        assert strat.iloc[1] == pytest.approx(0.01)

    def test_all_risk_off_window_is_not_treated_as_invested(self):
        """Regression: the risk-on state must come from the model's state
        space, not from ``states.max()`` within the slice. A window that is
        entirely risk-off would otherwise have state 0 read as "the best
        observed state" and be scored fully invested — inverting the strategy
        precisely during a sustained bear market."""
        states = self._states([0] * 6)
        returns = self._returns([-0.02] * 6)
        strat = regime_strategy_returns(returns, states, cost_bps=0.0)
        assert np.allclose(strat.to_numpy(), 0.0)

    def test_n_states_shifts_the_risk_on_state(self):
        states = self._states([0, 1, 2, 2, 1, 0])
        returns = self._returns([0.01] * 6)
        strat = regime_strategy_returns(returns, states, n_states=3,
                                        cost_bps=0.0)
        # Invested only on the bars following a state-2 label (positions 3, 4).
        assert strat.iloc[3] == pytest.approx(0.01)
        assert strat.iloc[4] == pytest.approx(0.01)
        assert strat.iloc[5] == pytest.approx(0.0)

    def test_risk_off_exposure_is_honoured(self):
        states = self._states([0, 0, 0, 0, 0, 0])
        returns = self._returns([0.01] * 6)
        strat = regime_strategy_returns(returns, states, risk_off_exposure=0.5,
                                        cost_bps=0.0)
        assert np.allclose(strat.to_numpy(), 0.005)

    def test_explicit_risk_on_state_overrides_default(self):
        states = self._states([0, 0, 0, 1, 1, 1])
        returns = self._returns([0.01] * 6)
        inverted = regime_strategy_returns(returns, states, risk_on_state=0,
                                           cost_bps=0.0)
        assert inverted.iloc[1] == pytest.approx(0.01)   # invested in state 0
        assert inverted.iloc[5] == pytest.approx(0.0)

    def test_aligns_mismatched_indices(self):
        """States are defined on the feature index, which starts later than the
        return index — the overlap is what gets traded."""
        returns = pd.Series([0.01] * 10, index=pd.bdate_range("2020-01-01",
                                                              periods=10))
        states = pd.Series([1] * 4, index=returns.index[-4:])
        strat = regime_strategy_returns(returns, states, cost_bps=0.0)
        assert len(strat) == 4
        assert strat.index.equals(returns.index[-4:])

    def test_empty_states_give_empty_result(self):
        returns = self._returns([0.01] * 6)
        strat = regime_strategy_returns(returns, pd.Series(dtype=float))
        assert strat.empty

    def test_de_risking_beats_buy_and_hold_on_a_clean_two_regime_series(self,
                                                                       inputs):
        """Sanity check that the wiring points the right way round: sitting out
        the fitted bear state should beat holding through it."""
        features, returns = inputs
        model = fit_jump_model(features, random_state=3)
        strat = regime_strategy_returns(returns, model.state_series(),
                                        cost_bps=10.0)
        buy_hold = returns.reindex(strat.index)
        assert annualised_sharpe(strat) > annualised_sharpe(buy_hold)


# ---------------------------------------------------------------------------
# Walk-forward penalty tuning
# ---------------------------------------------------------------------------
class TestTuneJumpPenalty:
    KW = dict(min_train=200, refit_every=25, n_folds=2,
              n_init=2, max_iter=6, random_state=3)

    @pytest.fixture(scope="class")
    def tuning(self, inputs) -> PenaltyTuning:
        features, returns = inputs
        return tune_jump_penalty(features, returns, candidates=(0.0, 150.0),
                                 **self.KW)

    def test_returns_one_row_per_candidate(self, tuning):
        assert list(tuning.table["penalty"]) == [0.0, 150.0]
        assert set(tuning.table.columns) == {
            "penalty", "mean_sharpe", "min_fold_sharpe", "std_fold_sharpe",
            "switches_per_year", "n_switches",
        }

    def test_best_penalty_is_the_grid_argmax(self, tuning):
        best_row = tuning.table.loc[tuning.table["mean_sharpe"].idxmax()]
        assert tuning.best_penalty == pytest.approx(best_row["penalty"])

    def test_penalty_reduces_switch_rate(self, tuning):
        """Measured on the walk-forward path, not the offline fit — the whole
        point of tuning here rather than eyeballing a chart."""
        by_penalty = tuning.table.set_index("penalty")["switches_per_year"]
        assert by_penalty.loc[0.0] > by_penalty.loc[150.0]

    def test_scores_are_finite(self, tuning):
        assert np.isfinite(tuning.table["mean_sharpe"]).all()
        assert np.isfinite(tuning.table["min_fold_sharpe"]).all()

    def test_min_fold_never_exceeds_mean(self, tuning):
        assert (tuning.table["min_fold_sharpe"]
                <= tuning.table["mean_sharpe"] + 1e-9).all()

    def test_records_its_own_settings(self, tuning):
        assert tuning.n_folds == 2
        assert tuning.min_train == 200
        assert tuning.refit_every == 25
        assert tuning.cost_bps == 10.0

    def test_to_payload_is_json_safe(self, tuning):
        payload = tuning.to_payload()
        json.dumps(payload)
        assert payload["best_penalty"] == tuning.best_penalty
        assert len(payload["grid"]) == 2

    def test_rejects_empty_candidate_grid(self, inputs):
        features, returns = inputs
        with pytest.raises(ValueError):
            tune_jump_penalty(features, returns, candidates=())

    def test_rejects_bad_fold_count(self, inputs):
        features, returns = inputs
        with pytest.raises(ValueError):
            tune_jump_penalty(features, returns, candidates=(0.0,), n_folds=0)

    def test_rejects_history_too_short_for_the_folds(self, inputs):
        features, returns = inputs
        with pytest.raises(ValueError):
            tune_jump_penalty(features, returns, candidates=(0.0,),
                              min_train=len(features), n_folds=5)


# ---------------------------------------------------------------------------
# Streamlit-facing wiring
# ---------------------------------------------------------------------------
class TestGetRegime:
    def test_unknown_instrument_is_none(self):
        assert get_regime("NOT/APAIR") is None

    def test_reads_through_the_spine(self, monkeypatch, daily_frame):
        monkeypatch.setattr(rs, "daily_ohlc", lambda t, period=None: daily_frame)
        read = get_regime("XAU/USD")
        assert read is not None
        assert read.instrument == "XAU/USD"
        assert read.label in ("RISK_OFF", "RISK_ON")
        assert read.n_obs > rs.MIN_FEATURE_ROWS

    def test_widens_the_period_beyond_the_canonical_window(self, monkeypatch,
                                                           daily_frame):
        """A regime model needs years of history, so the service must ask the
        spine for REGIME_PERIOD rather than the canonical 300d default."""
        seen = {}

        def spy(ticker, period=None):
            seen["period"] = period
            return daily_frame

        monkeypatch.setattr(rs, "daily_ohlc", spy)
        get_regime("XAG/USD")
        assert seen["period"] == rs.REGIME_PERIOD

    def test_short_history_is_none(self, monkeypatch, daily_frame):
        monkeypatch.setattr(rs, "daily_ohlc",
                            lambda t, period=None: daily_frame.iloc[:100])
        assert get_regime("XPT/USD") is None

    def test_empty_frame_is_none(self, monkeypatch):
        monkeypatch.setattr(rs, "daily_ohlc",
                            lambda t, period=None: pd.DataFrame())
        assert get_regime("WTI/USD") is None

    def test_fetch_error_is_none(self, monkeypatch):
        def boom(ticker, period=None):
            raise RuntimeError("network down")

        monkeypatch.setattr(rs, "daily_ohlc", boom)
        assert get_regime("USD/CAD") is None


class TestGetRegimeHistory:
    def test_returns_label_series(self, monkeypatch, daily_frame):
        monkeypatch.setattr(rs, "daily_ohlc", lambda t, period=None: daily_frame)
        hist = get_regime_history("EUR/JPY")
        assert not hist.empty
        assert set(hist.unique()) <= {"RISK_OFF", "RISK_ON"}
        assert isinstance(hist.index, pd.DatetimeIndex)

    def test_unknown_instrument_is_empty(self):
        assert get_regime_history("NOT/APAIR").empty

    def test_short_history_is_empty(self, monkeypatch, daily_frame):
        monkeypatch.setattr(rs, "daily_ohlc",
                            lambda t, period=None: daily_frame.iloc[:100])
        assert get_regime_history("GBP/CAD").empty

    def test_fetch_error_is_empty(self, monkeypatch):
        def boom(ticker, period=None):
            raise RuntimeError("network down")

        monkeypatch.setattr(rs, "daily_ohlc", boom)
        assert get_regime_history("EUR/CAD").empty


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------
class _FakeCache:
    """Stand-in for NotifyCache: in-memory, no JSON file, no Postgres."""

    seen: set = set()

    def __init__(self, namespace):
        self.namespace = namespace

    def filter_new(self, keys):
        new = [k for k in keys if k not in _FakeCache.seen]
        _FakeCache.seen.update(new)
        return new


@pytest.fixture
def fake_logging(monkeypatch):
    from src.services import alert_service, tool_log

    _FakeCache.seen = set()
    calls = []
    monkeypatch.setattr(alert_service, "NotifyCache", _FakeCache)
    monkeypatch.setattr(tool_log, "log_tool_usage",
                        lambda tool, payload: calls.append((tool, payload)) or True)
    return calls


def _read(label="RISK_OFF", day="2026-07-28") -> RegimeRead:
    return RegimeRead(instrument="XAU/USD", label=label, state=0, n_states=2,
                      jump_penalty=150.0, n_obs=1000, n_switches=5,
                      switches_per_year=1.26, current_spell=40,
                      state_shares={"RISK_OFF": 0.4, "RISK_ON": 0.6},
                      asof=pd.Timestamp(day))


class TestLogRegime:
    def test_writes_one_row(self, fake_logging):
        assert log_regime(_read()) is True
        assert len(fake_logging) == 1
        tool, payload = fake_logging[0]
        assert tool == "jump_regime"
        assert payload["instrument"] == "XAU/USD"

    def test_deduped_per_instrument_day_and_label(self, fake_logging):
        assert log_regime(_read()) is True
        assert log_regime(_read()) is False       # same rerun, same read
        assert len(fake_logging) == 1

    def test_regime_change_same_day_is_logged(self, fake_logging):
        assert log_regime(_read(label="RISK_OFF")) is True
        assert log_regime(_read(label="RISK_ON")) is True
        assert len(fake_logging) == 2

    def test_next_day_is_logged(self, fake_logging):
        assert log_regime(_read(day="2026-07-28")) is True
        assert log_regime(_read(day="2026-07-29")) is True
        assert len(fake_logging) == 2

    def test_none_read_is_false(self, fake_logging):
        assert log_regime(None) is False
        assert fake_logging == []

    def test_missing_asof_still_logs(self, fake_logging):
        read = _read()
        read.asof = None
        assert log_regime(read) is True

    def test_never_raises_when_logging_breaks(self, monkeypatch):
        from src.services import alert_service

        def boom(namespace):
            raise RuntimeError("db down")

        monkeypatch.setattr(alert_service, "NotifyCache", boom)
        assert log_regime(_read()) is False
