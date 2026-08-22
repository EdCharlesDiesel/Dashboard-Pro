# Profit Factor & Sortino for the Source Scorecard

**Version on creation of this plan: 1.10.31** (VERSION currently reads 1.10.30).

## Goal

Add **profit factor** and **Sortino ratio** as two more per-source columns on
the Trade Journal's Source Scorecard, so a page's edge is judged on the
*shape* of its R distribution and not only on its mean.

## Why

Expectancy is a mean, and a mean hides everything about how it was earned. Two
sources at +0.30R are not the same source if one grinds small wins against
small losses and the other is carried by a single +8R outlier. Profit factor
separates gross winnings from gross losings; Sortino divides the mean by
**downside** deviation only, so a source is not penalised for large *wins* the
way a Sharpe ratio would penalise it.

Both were identified by reading `nautechsystems/nautilus_trader`, whose
`crates/analysis/src/statistics/` ships 34 portfolio statistics against the
three this repo computes. These two are the ones that apply directly to an
R-multiple series with no extra data.

## Architecture

`build_scorecard` already accumulates `agg["r_values"]` — the exact per-signal
R series both statistics need. Add two pure functions beside it and two more
keys in the output frame. No new data, no new query, no schema change.

## Spec

**`profit_factor(r_values) -> Optional[float]`**
Gross profit / gross loss, both absolute. `None` when there are no losses
(the ratio is undefined, not infinite) or no values at all.

**`sortino_ratio(r_values, mar=0.0) -> Optional[float]`**
`(mean(r) - mar) / downside_deviation`, where
`downside_deviation = sqrt(mean(min(r - mar, 0)**2))` computed over **all**
observations, not only the losing ones — the Sortino & Price convention.
`None` when fewer than 2 values, or when downside deviation is 0 (no
observation below the MAR, so the ratio is undefined).

### The one thing that must not be got wrong

**Neither statistic is annualised, and the Sortino must not be.** The
conventional Sortino multiplies by `sqrt(252)` because it assumes a series of
*periodic* returns at a fixed frequency. This series is nothing of the kind:
it is per-signal R-multiples, irregularly spaced, and `row_horizon` means a
15-minute fib trigger and a weekly-swing read are already marked on different
clocks. Annualising would invent a time basis the data does not have. This is
a **per-signal** Sortino and the column header and caption must say so.

### Deliberately not changed

- **The ranking stays `expectancy_r`.** These are new columns, not a new sort.
  Re-ranking the board is a different decision and was not asked for.
- **No minimum-observations floor inside the functions.** They compute
  whenever it is mathematically possible. The tab's existing "judge a source
  only once its resolved count is meaningful (≥20 is a reasonable floor)"
  caption already carries the epistemic warning, and burying a second, silent
  floor inside the maths would hide it.

## Global Constraints

- Never commit. Show the owner the diff.
- Every completed task bumps the patch: `python deploy/sync_version.py <next>`.
- Test first, run it, read the failure, then implement.
- `src/services/source_scorecard.py` stays pure — no DB, no Streamlit, no
  network.
- Existing scorecard behaviour is unchanged; every current test must still pass
  untouched.
- Use `.venv/Scripts/python.exe`, `PYTHONIOENCODING=utf-8` when piping.

## Starting state (measured)

- `VERSION` → `1.10.30`
- `src/services/source_scorecard.py` → 276 lines; `build_scorecard` emits 13
  columns and already collects `agg["r_values"]`.
- `tests/test_source_scorecard.py` → existing suite passes.
- Consumer: `pages/trade-journal.py:1345-1420` (tab 5), which sets
  `column_config` per column.

---

## Task 1 — The two statistics

**Files:** modify `src/services/source_scorecard.py`, `tests/test_source_scorecard.py`

- [ ] **Step 1: Write the failing tests**

```python
class TestProfitFactor:
    def test_gross_win_over_gross_loss(self):
        assert profit_factor([2.0, -1.0, 1.0, -1.0]) == pytest.approx(1.5)

    def test_all_winners_is_undefined_not_infinite(self):
        """No losses means the ratio has no denominator. Printing infinity
        would claim a certainty the sample does not support; the wins/losses
        columns already show why it is blank."""
        assert profit_factor([1.0, 2.0]) is None

    def test_all_losers_is_zero(self):
        assert profit_factor([-1.0, -2.0]) == pytest.approx(0.0)

    def test_empty_is_none(self):
        assert profit_factor([]) is None

    def test_break_even_is_one(self):
        assert profit_factor([1.0, -1.0]) == pytest.approx(1.0)


class TestSortinoRatio:
    def test_mean_over_downside_deviation(self):
        # r = [1, -1, 1, -1]: mean 0.0 -> ratio 0.0
        assert sortino_ratio([1.0, -1.0, 1.0, -1.0]) == pytest.approx(0.0)

    def test_downside_deviation_uses_all_observations(self):
        """Sortino & Price divide the squared downside by the FULL count, not
        the count of losers. r = [2, -1, 2, -1]: mean 0.5,
        dd = sqrt((0+1+0+1)/4) = 0.7071 -> 0.7071."""
        assert sortino_ratio([2.0, -1.0, 2.0, -1.0]) == pytest.approx(
            0.5 / (0.5 ** 0.5), rel=1e-6)

    def test_no_downside_is_undefined(self):
        assert sortino_ratio([1.0, 2.0, 3.0]) is None

    def test_fewer_than_two_observations_is_none(self):
        assert sortino_ratio([1.0]) is None
        assert sortino_ratio([]) is None

    def test_a_nonzero_mar_shifts_the_threshold(self):
        # Against a 1R hurdle, [1, 1] has no excess and no downside.
        assert sortino_ratio([1.0, 1.0], mar=1.0) is None

    def test_upside_outliers_do_not_penalise(self):
        """The whole reason to prefer Sortino over Sharpe here: replacing a
        win with a much bigger win must never lower the ratio."""
        base = sortino_ratio([1.0, -1.0, 1.0, -1.0, 1.0])
        spiky = sortino_ratio([1.0, -1.0, 1.0, -1.0, 9.0])
        assert base is not None and spiky is not None and spiky > base


class TestScorecardCarriesTheNewColumns:
    def test_columns_are_present_and_populated(self):
        rows = [
            _row(source="mixed", is_open=False, r_multiple=2.0),
            _row(source="mixed", is_open=False, r_multiple=-1.0),
            _row(source="mixed", is_open=False, r_multiple=1.0),
            _row(source="mixed", is_open=False, r_multiple=-1.0),
        ]
        df = build_scorecard(rows, {})
        assert df.loc["mixed", "profit_factor"] == pytest.approx(1.5)
        assert df.loc["mixed", "sortino"] is not None

    def test_a_source_with_no_losses_reports_no_profit_factor(self):
        rows = [_row(source="clean", is_open=False, r_multiple=1.0),
                _row(source="clean", is_open=False, r_multiple=2.0)]
        df = build_scorecard(rows, {})
        assert pd.isna(df.loc["clean", "profit_factor"])
        assert pd.isna(df.loc["clean", "sortino"])

    def test_the_ranking_metric_is_unchanged(self):
        """New columns, not a new sort."""
        rows = [_row(source="a", is_open=False, r_multiple=3.0),
                _row(source="b", is_open=False, r_multiple=0.5)]
        df = build_scorecard(rows, {})
        assert list(df.index) == ["a", "b"]
```

- [ ] **Step 2: Run and read the failure**

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/test_source_scorecard.py --no-cov -q
```

Expected: `ImportError: cannot import name 'profit_factor'`.

- [ ] **Step 3: Implement both functions**

Place them above `build_scorecard`, taking `Sequence[float]`:

```python
def profit_factor(r_values: Sequence[float]) -> Optional[float]:
    """Gross profit / gross loss over a series of R-multiples.

    Expectancy is a mean, and a mean cannot tell a source that grinds from one
    carried by a single outlier. This can: 1.0 is break-even, and the distance
    from it is how much work the winners are doing.

    ``None`` when there are no losses at all — the ratio has no denominator,
    and printing infinity would assert a certainty two winning signals do not
    support. The wins/losses columns already show why the cell is blank.
    """


def sortino_ratio(r_values: Sequence[float],
                  mar: float = 0.0) -> Optional[float]:
    """Mean excess R over downside deviation. **Per signal, not annualised.**

    Downside deviation is ``sqrt(mean(min(r - mar, 0) ** 2))`` over *all*
    observations rather than only the losing ones — the Sortino & Price
    convention, which is what keeps a source with few but deep losses from
    looking safer than one with many shallow ones.

    Deliberately not annualised. The conventional sqrt(252) assumes periodic
    returns at a fixed frequency; this series is per-signal R-multiples,
    irregularly spaced, and `row_horizon` already marks a 15-minute trigger and
    a weekly swing on different clocks. Multiplying by a time factor the data
    does not have would be inventing precision.

    ``None`` below two observations, or when no observation falls under the
    MAR (zero downside deviation is an undefined ratio, not an infinite one).
    """
```

- [ ] **Step 4: Wire them into `build_scorecard`**

Two keys in the per-source dict built from `r_vals`, and both names added to
the docstring's column list:

```python
"profit_factor": _round_opt(profit_factor(r_vals), 2),
"sortino": _round_opt(sortino_ratio(r_vals), 2),
```

`_round_opt` is a two-line local helper so `round(None, 2)` cannot raise —
every other column in this function guards with a conditional, and a third
inline conditional pair would be less readable than naming the idea once.

- [ ] **Step 5: Run the whole file**

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/test_source_scorecard.py --no-cov -q
```

Expected: PASS, including every pre-existing test untouched.

- [ ] **Step 6: Bump**

```bash
.venv/Scripts/python.exe deploy/sync_version.py 1.10.32
```

---

## Task 2 — Show them

**Files:** modify `pages/trade-journal.py` (tab 5, ~lines 1330-1420)

- [ ] **Step 1: Add both to `column_config`**

```python
"profit_factor": st.column_config.NumberColumn(
    "Profit factor", format="%.2f",
    help="Gross winning R / gross losing R. 1.00 is break-even. "
         "Blank when a source has no losses yet."),
"sortino":       st.column_config.NumberColumn(
    "Sortino (per signal)", format="%.2f",
    help="Mean R over downside deviation — big wins are not penalised the "
         "way a Sharpe ratio would. Per signal, NOT annualised: these "
         "R-multiples are irregularly spaced and different sources use "
         "different horizons."),
```

- [ ] **Step 2: Extend the tab caption**

Append one sentence to the existing caption naming what the two new columns
add over expectancy, and stating the not-annualised fact where a reader will
see it without hovering.

- [ ] **Step 3: Smoke the page**

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -c "from streamlit.testing.v1 import AppTest; at=AppTest.from_file('pages/trade-journal.py', default_timeout=240); at.run(); print('exceptions:', [str(e.value) for e in at.exception])"
```

Expected: `exceptions: []`.

- [ ] **Step 4: Full suite + compile**

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest
```

```bash
.venv/Scripts/python.exe -m py_compile pages/trade-journal.py src/services/source_scorecard.py
```

- [ ] **Step 5: Update the map and docs**

`.foglamp/introspect.py` then `.foglamp/render.py --check`. Update the
`scorecard` node's `sub` in `scan.json` (currently `win rate, avg R,
expectancy`) and the Source Scorecard line in `docs/System_Guide.md`, then
`python docs/generate_guide_pdf.py`.

- [ ] **Step 6: Bump and verify**

```bash
.venv/Scripts/python.exe deploy/sync_version.py 1.10.33 && .venv/Scripts/python.exe deploy/verify_deploy.py
```

---

## Known limits

- **Both statistics inherit the replay's assumptions.** Open-signal R comes
  from a bar-by-bar replay with no spread, slippage or sizing. A profit factor
  of 1.6 here is relative evidence between sources, not a P&L claim.
- **Small samples make both unstable**, profit factor especially: one more loss
  can halve it. The tab's ≥20-resolved floor applies to these columns more
  than to win rate.
- **Sortino says nothing about path.** It has no notion of consecutive losses;
  max drawdown would, and is the obvious next one to take from the same
  Nautilus list if these prove useful.
