# pct_change Fill Method Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement
> this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop `currency_index` turning a holiday gap into a fake 0% return on
the pandas version production actually runs, by pinning `fill_method` instead of
inheriting a version-dependent default.

**Architecture:** One explicit argument. The behaviour it selects is the one the
function's docstring already claims.

**Tech Stack:** Python 3.14, pandas, pytest.

**Spec:** CI failure, 2026-09-06 —
`test_daily_currency_returns_cross_sectional_average`, asserting `NaN` and
receiving `0.0`.

## Global Constraints

- **Never commit.** The owner reviews and commits.
- This plan takes **1.10.61**.
- **Verify on both pandas versions.** The venv runs 3.0.2 and the containers run
  2.3.3; a green local suite proved nothing here.
- **Change one call site.** The other 21 are recorded, not silently altered.

---

## Context

CI failed one test while the local suite passed 2067. The divergence is pandas:

| | pandas | `Series.pct_change()` default | a holiday gap becomes |
|---|---|---|---|
| local venv | **3.0.2** | `fill_method=None` | `NaN` — skipped |
| containers / CI | **2.3.3** | `fill_method='pad'` | **`0.0` — counted as flat** |

`src/services/currency_index.py:34` calls `closes[ticker].pct_change()` bare, and
its own docstring promises:

> *"A pair missing on a given day (holiday gap) is excluded from that day's
> average via `skipna` rather than nulling out the whole currency."*

On pandas 2.3.3 that does not happen. The gap is padded to `0.0`, so `skipna`
excludes nothing and the missing pair contributes a **fake 0% move** to the
cross-sectional average. A currency whose only quiet pair is on holiday reads as
"unchanged" rather than "unknown" — and this is the version production runs, so
the bug is live and the passing local test was the misleading part.

The deprecation warning naming the exact line was in the same CI output.

**It is not one call site.** 22 bare `pct_change()` calls exist across `src/` and
`pages/` and **none** pins `fill_method`. Several `.dropna()` immediately after,
which hides the difference for *leading* NaNs but not for interior gaps — padding
converts those to `0.0`, and `dropna` then keeps them. Each needs its own
judgement about whether a gap means "flat" or "unknown", so they are recorded
here rather than swept.

---

## Task 1: Pin the failing call site

**Files:**
- Modify: `src/services/currency_index.py`
- Test: `tests/test_currency_index.py` (already covers it — it is the failing test)

- [ ] **Step 1:** Confirm the test fails under pandas 2.3.3 (in the container)
      and passes under 3.0.2 (the venv) — the divergence itself.
- [ ] **Step 2:** Pass `fill_method=None` explicitly, with the reason recorded.
- [ ] **Step 3:** Green **in both places**.

---

## Verification

1. **The test passes in the container (pandas 2.3.3)** — the environment that was
   failing, and the one production runs.
2. **It still passes in the venv (pandas 3.0.2).**
3. **Full suite** locally and in the container.
4. The remaining 21 call sites are listed for the owner, unchanged.
5. Show the owner the diff. **Never commit.**

## The 21 left alone

Each needs a decision about whether a missing bar means "flat" or "unknown":
`correlations.py:78`, `currency_strength_index.py:265`, `dxy_gold.py:83`,
`instrument_predictor.py:186`, `market_overview_lib.py:472,531`,
`index_analysis.py:26,33,34`, `swing_playbook_service.py:191`,
`bonds_gold_dxy_app.py:269,270,329,330`,
`cot_trade_signal_walk_forward_backtest_harness.py:194`,
`quant_models_tab.py:229`, `seasonality.py:80,95`, `surprise_tab.py:278`,
`vwap-ema-gold.py:261`, `week_ahead_tab.py:107`.
