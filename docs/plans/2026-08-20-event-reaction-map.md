# Event Reaction Map Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax for tracking.
> **This repository never commits.** The skill template's "Commit" step is
> replaced throughout by *bump the version and show the owner the diff*
> (`.claude/CLAUDE.md` working agreement takes precedence).

**Goal:** Generalize the NFP-only reaction map into a four-event map covering
NFP, CPI, PPI and FOMC, where the surprise components, the transmission chain,
the exposure betas and the session phases are all properties of the *event*.

**Architecture:** One `EventSpec` per event drives everything. A generic
`compute_surprise(spec, inputs)` z-scores each declared `Component` and
weights it into the same composite the current NFP code produces; a per-event
`Exposure` table and chain replace today's single module-level constants. The
page becomes a thin renderer over the selected spec. The release calendar is
lifted out of `pages/event_week_vol_tab.py` into `src/core/` so both pages
read one calendar.

**Tech Stack:** Python 3.14 (avoid 3.12+-only syntax), pandas, Plotly,
Streamlit, pytest.

**Spec:** This plan is the spec. It supersedes nothing in
`docs/plans/2026-08-20-nfp-reaction-wiring.md`, which stays as the record of
how the page was first wired in.

**Version on creation of this plan: 1.10.26** (VERSION currently reads 1.10.25).

## Global Constraints

- **Never commit.** Code changes only; the owner reviews and commits.
- **Every completed task bumps the patch.** Read `VERSION`, add one, write it
  back with `python deploy/sync_version.py <next>`. Global sequence, no
  reservations, no minor bumps.
- Use the venv: `.venv/Scripts/python.exe`. Set `PYTHONIOENCODING=utf-8` when
  piping output on Windows.
- **NFP behaviour is preserved exactly — but its tests are not.** Every
  function gains a `spec` first argument and `Surprise` changes shape, so
  every existing call site in `tests/test_nfp_reaction.py` is rewritten. That
  makes the existing tests useless as a regression gate, so Task 2 Step 1 adds
  `TestNFPRegression` carrying the exact numbers today's code produces
  (captured below). **That** is the gate. Rewriting an assertion to match new
  output is the failure mode this guards against.
- `trade_setups.source` is `VARCHAR(20)`. All four tags fit:
  `nfp_reaction` (12), `cpi_reaction` (12), `ppi_reaction` (12),
  `fomc_reaction` (13).
- `tests/test_signal_sweep.py`'s regex only matches **string literals** in
  `persist_signals("tag", ...)`. A computed tag is invisible to it, so the
  four tags must appear as four literal calls.
- No hard-coded colours: `BloombergTheme` tokens only.
- Do not touch `archive/`.

## Decisions taken (owner-confirmed)

1. **FOMC gets three inputs** — decision vs priced (bp), dot-plot median shift
   (bp), and a statement/presser tone dial — z-scored onto the same composite.
2. **Four source tags**, one per event, via four literal `persist_signals`
   calls, so the Source Scorecard grades each event's betas separately.
3. **The calendar is extracted to `src/core/`** and shared with
   `pages/event_week_vol_tab.py`.

## The three structural facts this design turns on

**1. "Hawkish" does not mean the same thing for growth in every event.** A hot
NFP is hawkish *and* good for growth. A hot CPI is hawkish and *bad* for growth
— it is a real-income squeeze plus tighter policy. Bake the sign into each
event's own growth betas rather than adding a separate sign flag; one
mechanism, and the stagflation asymmetry is then visible in the table itself.

**2. The chain is the point of the page, so the chain is per-event.** NFP runs
jobs → spending → inflation → rates. CPI does not pass through jobs at all,
and its third node moves *against* the surprise. Chain nodes therefore carry
their own sign.

**3. FOMC is not an 08:30 event, and the presser is not the statement.** The
decision lands 14:00 NY (≈20:00 SAST — the SA desk is live for it, unlike the
08:30 releases), and the 14:30 presser routinely reverses the statement move.
That reversal deserves its own phase, not a footnote.

---

## File Structure

| File | Responsibility |
|---|---|
| **Create** `src/core/event_calendar.py` | The one release calendar: FOMC/BoJ/SARB/CPI/PPI date lists, rule-computed NFP dates, `build_event_calendar`, `EVENT_RELEVANCE`, and `next_release()`. |
| **Create** `tests/test_event_calendar.py` | Calendar unit tests. |
| **Modify** `src/core/nfp_reaction.py` | `Component`, `Exposure`, `EventSpec`, `EVENTS`; generic `compute_surprise`/`score_instruments`/`timing_frame`/`chain_leaves`/`board_to_signals`. |
| **Modify** `tests/test_nfp_reaction.py` | Keep every NFP assertion; add CPI/PPI/FOMC and cross-event tests. |
| **Modify** `pages/nfp_reaction.py` | Event selector, per-event input form, four-branch persist dispatch. |
| **Modify** `pages/event_week_vol_tab.py` | Delete its local calendar block; import from `src.core.event_calendar`. |
| **Modify** `src/pages_lib/navigation.py` | `NFPR` → `EVNT`, label "Event Reaction Map". |
| **Modify** `src/services/signal_sweep.py` | Four tags → same page path; four `PREPARE` hooks that select the event. |
| **Modify** `.foglamp/scan.json`, `docs/README.md`, `docs/System_Guide.md` | Docs + map. |

**Filename note:** `pages/nfp_reaction.py` keeps its name deliberately — the
owner refers to the page by it, and renaming churns the URL, the sweep paths
and the nav. The nav *label* becomes "Event Reaction Map". If the mismatch
grates, renaming is a separate one-line change with its own bump.

---

## Task 1: Extract the shared release calendar

**Files:**
- Create: `src/core/event_calendar.py`
- Create: `tests/test_event_calendar.py`
- Modify: `pages/event_week_vol_tab.py` (delete lines 47–191's calendar block, add an import)

**Interfaces:**
- Produces: `FOMC_DATES`, `BOJ_DATES`, `SARB_DATES`, `US_CPI_DATES`,
  `US_PPI_DATES` (all `List[str]`, ISO); `EVENT_RELEVANCE: Dict[str, List[str]]`;
  `nfp_dates(start: date, end: date) -> List[pd.Timestamp]`;
  `build_event_calendar(start: date, end: date, extra: Optional[pd.DataFrame]) -> pd.DataFrame`;
  `next_release(event: str, today: Optional[date] = None) -> Optional[date]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_event_calendar.py
from datetime import date

import pandas as pd
import pytest

from src.core.event_calendar import (
    EVENT_RELEVANCE, FOMC_DATES, US_CPI_DATES, US_PPI_DATES,
    build_event_calendar, next_release, nfp_dates,
)


class TestNFPDates:
    def test_every_generated_date_is_a_first_friday(self):
        for ts in nfp_dates(date(2026, 1, 1), date(2026, 12, 31)):
            assert ts.weekday() == 4 and ts.day <= 7

    def test_it_yields_one_per_month(self):
        assert len(nfp_dates(date(2026, 1, 1), date(2026, 12, 31))) == 12


class TestSeededCalendars:
    @pytest.mark.parametrize("dates", [FOMC_DATES, US_CPI_DATES, US_PPI_DATES])
    def test_dates_are_iso_sorted_and_unique(self, dates):
        assert dates == sorted(dates), "keep seed lists sorted"
        assert len(dates) == len(set(dates))
        for d in dates:
            date.fromisoformat(d)          # raises if malformed

    def test_cpi_and_ppi_reach_the_current_year(self):
        # The whole reason the calendar was extracted: the old copy stopped
        # at 2025-12-10, so a 2026 default was impossible.
        assert any(d.startswith("2026") for d in US_CPI_DATES)
        assert any(d.startswith("2026") for d in US_PPI_DATES)


class TestNextRelease:
    def test_nfp_is_rule_computed_not_seeded(self):
        assert next_release("NFP", date(2026, 8, 20)) == date(2026, 9, 4)

    def test_a_seeded_event_returns_the_next_listed_date(self):
        nxt = next_release("FOMC", date(2026, 8, 20))
        assert nxt is not None and nxt >= date(2026, 8, 20)
        assert nxt.isoformat() in FOMC_DATES

    def test_the_release_day_itself_does_not_roll(self):
        first = date.fromisoformat(FOMC_DATES[0])
        assert next_release("FOMC", first) == first

    def test_an_exhausted_calendar_returns_none_rather_than_guessing(self):
        assert next_release("FOMC", date(2099, 1, 1)) is None

    def test_an_unknown_event_returns_none(self):
        assert next_release("NOT_AN_EVENT", date(2026, 8, 20)) is None


class TestBuildEventCalendar:
    def test_it_returns_long_format_within_the_window(self):
        cal = build_event_calendar(date(2026, 1, 1), date(2026, 6, 30), None)
        assert list(cal.columns) == ["date", "event"]
        assert cal["date"].is_monotonic_increasing
        assert (cal["date"].dt.date >= date(2026, 1, 1)).all()
        assert (cal["date"].dt.date <= date(2026, 6, 30)).all()

    def test_extra_rows_are_merged_and_deduped(self):
        extra = pd.DataFrame({"date": ["2026-02-02", "2026-02-02"],
                              "event": ["CUSTOM", "CUSTOM"]})
        cal = build_event_calendar(date(2026, 1, 1), date(2026, 3, 1), extra)
        assert (cal["event"] == "CUSTOM").sum() == 1


class TestEventRelevance:
    def test_jpy_pulls_in_boj_and_zar_pulls_in_sarb(self):
        assert "BOJ" in EVENT_RELEVANCE["USD/JPY"]
        assert "SARB" in EVENT_RELEVANCE["USD/ZAR"]

    def test_every_instrument_gets_the_us_macro_trio(self):
        for events in EVENT_RELEVANCE.values():
            assert {"NFP", "US_CPI", "FOMC"} <= set(events)
```

- [ ] **Step 2: Run it and read the failure**

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/test_event_calendar.py --no-cov -q
```

Expected: `ModuleNotFoundError: No module named 'src.core.event_calendar'`.

- [ ] **Step 3: Create the module by moving, not rewriting**

Copy `pages/event_week_vol_tab.py` lines 47–191 verbatim into
`src/core/event_calendar.py` (the four date lists, `EVENT_RELEVANCE`,
`nfp_dates`, `build_event_calendar`). Keep `days_to_nearest_event` in the page
— it is a study over a price index, not calendar data. Then add, in the same
file:

```python
# US PPI release days. BLS publishes PPI within a day or two of CPI. These are
# best-effort seeds on exactly the same footing as the CPI list above — VERIFY
# against bls.gov/schedule/news_release/ and extend. A wrong seed costs a wrong
# *default* in the date picker, never a wrong result: the page accepts any date
# you type, and the stored signal carries the date you entered.
US_PPI_DATES = [
    "2025-01-14", "2025-02-13", "2025-03-13", "2025-04-11",
    "2025-05-15", "2025-06-12", "2025-07-16", "2025-08-14",
    "2025-09-10", "2025-10-16", "2025-11-14", "2025-12-11",
    "2026-01-14", "2026-02-12", "2026-03-12", "2026-04-14",
    "2026-05-13", "2026-06-11", "2026-07-15", "2026-08-13",
    "2026-09-15", "2026-10-14", "2026-11-12", "2026-12-11",
]

def next_release(event: str, today: Optional[date] = None) -> Optional[date]:
    """The next occurrence of ``event`` on or after ``today``.

    NFP is rule-computed (first Friday), so it never runs out. Every other
    event is a seeded list, and a seed list *does* run out — when it has, this
    returns ``None`` rather than extrapolating. A made-up FOMC date presented
    as a default is worse than no default, because the page would be quietly
    stamping signals with a release that never happened.
    """
```

Extend `US_CPI_DATES` through 2026 — the existing list stops at 2025-12-10,
which is why a 2026 default was impossible and half the reason this extraction
is happening. Same "best-effort seed, verify" caveat:

```python
    "2026-01-13", "2026-02-11", "2026-03-11", "2026-04-10",
    "2026-05-12", "2026-06-10", "2026-07-14", "2026-08-12",
    "2026-09-11", "2026-10-13", "2026-11-10", "2026-12-10",
```

Add `("US_PPI", US_PPI_DATES)` to `build_event_calendar`'s loop and `"US_PPI"`
to every `EVENT_RELEVANCE` list.

**Verify both seed lists against bls.gov before relying on a default.** They
are the one part of this plan written from recollection rather than from the
codebase, and a release date is exactly the kind of fact that is quietly wrong.

- [ ] **Step 4: Rewire the page that owned the calendar**

In `pages/event_week_vol_tab.py`, delete the moved block and add:

```python
from src.core.event_calendar import (
    EVENT_RELEVANCE, build_event_calendar,
)
```

`import calendar` becomes unused there — remove it only if nothing else in the
file uses it (grep first; it is imported at line 24).

- [ ] **Step 5: Run both test files**

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/test_event_calendar.py --no-cov -q
```

Expected: PASS.

- [ ] **Step 6: Prove the page that lost the calendar still runs**

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -c "from streamlit.testing.v1 import AppTest; at=AppTest.from_file('pages/event_week_vol_tab.py', default_timeout=240); at.run(); print('exceptions:', [str(e.value) for e in at.exception])"
```

Expected: `exceptions: []`. This hits live yfinance and takes ~1–2 min.

- [ ] **Step 7: Bump and show the diff**

```bash
.venv/Scripts/python.exe deploy/sync_version.py 1.10.27
```

---

## Task 2: The event model in the core

**Files:**
- Modify: `src/core/nfp_reaction.py`
- Modify: `tests/test_nfp_reaction.py`

**Interfaces:**
- Consumes: `src.core.event_calendar.next_release` (Task 1).
- Produces:
  - `Component(key, label, sd, weight, invert=False, delta_only=False, help="")`
  - `Exposure(symbol, beta_rate, beta_growth, unit, unit_label, decimals)`
  - `ChainNode = Tuple[str, int]` — node label and its sign vs the surprise
  - `Phase(name, note, start_min=None, end_min=None, start_sast=None, end_sast=None)`
    — a phase is either relative to t0 (`start_min`/`end_min`) **or** an
    absolute SAST window (`start_sast`/`end_sast`), never both
  - `EventSpec(key, label, source_tag, calendar_key, release_time_ny, components, chain, exposures, phases, note)`
    where `calendar_key` is the name `src.core.event_calendar` uses
    (`"NFP" | "US_CPI" | "US_PPI" | "FOMC"`) — deliberately not the same as
    `key`, because the calendar's namespace already exists and is shared
  - `EVENTS: Dict[str, EventSpec]` keyed `"NFP" | "CPI" | "PPI" | "FOMC"`
  - `compute_surprise(spec: EventSpec, values: Dict[str, float]) -> Surprise`
  - `Surprise(z: Dict[str, float], composite: float)` — `.label`, `.direction`
  - `score_instruments(spec, composite, regime, overrides=None) -> pd.DataFrame`
  - `release_datetime_sast(spec: EventSpec, d: date) -> datetime`
  - `next_nfp_date` is **deleted** — `event_calendar.next_release` replaces it
  - `chain_leaves(spec, surprise, board)`, `timing_frame(spec, d)`,
    `board_to_signals(spec, board, release_date, regime, composite, ...)`
  - `REGIMES`, `MIN_ABS_COMPOSITE`, `MIN_CONVICTION` are unchanged and stay
    event-independent — the regime is a property of the market's reaction
    function, not of which statistic was released

### Why `Surprise` changes shape

Today it is four named z fields. Four events with two-to-four differently
named components cannot share that. It becomes `z: Dict[str, float]` keyed by
component key, with `composite` unchanged. `label`/`direction` and their
0.35/1.0/2.0 thresholds are untouched.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_nfp_reaction.py`)

```python
from src.core.nfp_reaction import EVENTS

NFP, CPI, PPI, FOMC = (EVENTS["NFP"], EVENTS["CPI"],
                       EVENTS["PPI"], EVENTS["FOMC"])


class TestEventSpecs:
    def test_all_four_events_are_registered(self):
        assert set(EVENTS) == {"NFP", "CPI", "PPI", "FOMC"}

    def test_every_source_tag_fits_the_column_and_is_unique(self):
        tags = [e.source_tag for e in EVENTS.values()]
        assert all(len(t) <= 20 for t in tags)
        assert len(set(tags)) == 4

    def test_component_weights_are_positive(self):
        for spec in EVENTS.values():
            assert spec.components
            assert all(c.weight > 0 for c in spec.components)

    def test_only_nfp_runs_through_jobs_and_spending(self):
        assert [n for n, _ in NFP.chain] == ["jobs", "spending",
                                             "inflation", "rates"]
        for spec in (CPI, PPI, FOMC):
            assert "jobs" not in [n for n, _ in spec.chain]

    def test_the_price_events_carry_a_node_that_moves_against_the_surprise(self):
        """A hot CPI is hawkish AND bad for real incomes. A chain whose every
        node points the same way is the naive chain this page exists to fix."""
        assert any(sign < 0 for _, sign in CPI.chain)
        assert any(sign < 0 for _, sign in PPI.chain)

    def test_fomc_is_an_afternoon_event(self):
        assert FOMC.release_time_ny.hour == 14
        for spec in (NFP, CPI, PPI):
            assert (spec.release_time_ny.hour,
                    spec.release_time_ny.minute) == (8, 30)


class TestNFPRegression:
    """The gate for the whole refactor.

    Every number below was captured from the pre-refactor implementation on
    2026-08-20 for the input (275 vs 150, +15k revision, U3 4.0 vs 4.2, AHE
    0.5 vs 0.3) under the Balanced regime. If one of these moves, the
    generalisation changed NFP behaviour and the implementation is wrong.
    Never edit an expected value here to make a run pass.
    """

    INPUT = {"nfp": 275.0, "nfp_c": 150.0, "rev": 15.0,
             "ur": 4.0, "ur_c": 4.2, "ahe": 0.5, "ahe_c": 0.3}

    def test_the_composite_and_every_component_z_are_unmoved(self):
        s = compute_surprise(NFP, self.INPUT)
        assert s.composite == pytest.approx(1.6197052947, abs=1e-9)
        assert s.z["nfp"] == pytest.approx(1.9230769231, abs=1e-9)
        assert s.z["rev"] == pytest.approx(0.2500000000, abs=1e-9)
        assert s.z["ur"] == pytest.approx(1.4285714286, abs=1e-9)
        assert s.z["ahe"] == pytest.approx(1.8181818182, abs=1e-9)
        assert s.label == "Significant" and s.direction == "hawkish"

    @pytest.mark.parametrize("symbol,score,conviction", [
        ("USDJPY", +1.6035082418, 1.0000000000),
        ("XAUUSD", -1.3565031843, 1.0000000000),
        ("US10Y",  +1.3119612887, 1.0000000000),
        ("XAGUSD", -0.9799217033, 0.6470588235),
        ("EURUSD", -0.8746408591, 0.8307692308),
        ("DXY",    +0.8665423327, 0.6184971098),
        ("GBPUSD", -0.7734092782, 0.7431906615),
        ("USDZAR", +0.7248181194, 0.3849462366),
        ("AUDUSD", -0.6195372752, 0.4358974359),
        ("US500",  +0.2672513736, 0.1764705882),
        ("BTCUSD", -0.2470050574, 0.2013201320),
        ("NAS100", +0.1295764236, 0.0707964602),
    ])
    def test_the_nfp_board_is_unmoved(self, symbol, score, conviction):
        board = score_instruments(NFP, compute_surprise(NFP, self.INPUT).composite,
                                  BALANCED)
        row = board.loc[board["symbol"] == symbol].iloc[0]
        assert row["score"] == pytest.approx(score, abs=1e-9)
        assert row["conviction"] == pytest.approx(conviction, abs=1e-9)


class TestGenericSurprise:
    def test_an_omitted_component_renormalises_rather_than_scoring_zero(self):
        full = compute_surprise(CPI, {"core_mm": 0.4, "core_mm_c": 0.3,
                                      "head_mm": 0.3, "head_mm_c": 0.3,
                                      "core_yy": 3.0, "core_yy_c": 3.0,
                                      "head_yy": 2.9, "head_yy_c": 2.9})
        partial = compute_surprise(CPI, {"core_mm": 0.4, "core_mm_c": 0.3})
        assert partial.composite > full.composite > 0

    def test_an_inverted_component_flips_sign(self):
        lower = compute_surprise(NFP, {"nfp": 150.0, "nfp_c": 150.0,
                                       "ur": 4.0, "ur_c": 4.2})
        assert lower.z["ur"] > 0 and lower.composite > 0

    def test_a_delta_only_component_needs_no_consensus(self):
        sd = {c.key: c.sd for c in FOMC.components}["decision_bp"]
        s = compute_surprise(FOMC, {"decision_bp": 25.0})
        assert s.z["decision_bp"] == pytest.approx(25.0 / sd)
        assert s.composite > 0

    def test_a_component_with_no_consensus_supplied_is_dropped_not_zeroed(self):
        """A paired component whose consensus box is left empty must not be
        scored as 'came in exactly on forecast' — that is a fabricated
        observation, and it would drag every composite toward zero."""
        s = compute_surprise(CPI, {"core_mm": 0.5, "core_mm_c": 0.3,
                                   "head_mm": 0.4})          # no head_mm_c
        assert "head_mm" not in s.z

    def test_hot_cpi_and_strong_nfp_are_both_hawkish(self):
        cpi = compute_surprise(CPI, {"core_mm": 0.5, "core_mm_c": 0.3})
        nfp = compute_surprise(NFP, {"nfp": 275.0, "nfp_c": 150.0})
        assert cpi.direction == nfp.direction == "hawkish"


class TestPerEventExposures:
    def test_a_hawkish_print_sells_gold_under_every_event(self):
        for spec in EVENTS.values():
            board = score_instruments(spec, 2.0, BALANCED)
            gold = board.loc[board["symbol"] == "XAUUSD", "score"].iloc[0]
            assert gold < 0, spec.key

    def test_hot_cpi_sells_equities_even_in_a_growth_scare(self):
        """The stagflation asymmetry: a hawkish NFP is a growth *positive*, a
        hawkish CPI is a growth negative, so they diverge exactly where the
        growth channel carries the most weight."""
        nfp = score_instruments(NFP, 2.0, GROWTH_SCARE)
        cpi = score_instruments(CPI, 2.0, GROWTH_SCARE)
        assert nfp.loc[nfp["symbol"] == "US500", "score"].iloc[0] > 0
        assert cpi.loc[cpi["symbol"] == "US500", "score"].iloc[0] < 0

    def test_every_event_covers_the_same_symbol_universe(self):
        universe = {e.symbol for e in NFP.exposures}
        for spec in EVENTS.values():
            assert {e.symbol for e in spec.exposures} == universe


class TestPerEventTiming:
    def test_the_fomc_frame_names_the_presser(self):
        df = timing_frame(FOMC, date(2026, 9, 16))
        assert any("presser" in p.lower() for p in df["Phase"])

    def test_an_0830_frame_does_not(self):
        df = timing_frame(NFP, date(2026, 9, 4))
        assert not any("presser" in p.lower() for p in df["Phase"])

    def test_fomc_lands_in_the_sast_evening(self):
        t0 = release_datetime_sast(FOMC, date(2026, 9, 16))
        assert t0.hour >= 19

    def test_phases_are_ordered_and_labelled(self):
        for spec in EVENTS.values():
            df = timing_frame(spec, date(2026, 9, 16))
            assert list(df.columns) == ["Phase", "SAST", "What is happening"]
            assert len(df) >= 4


class TestPerEventSignals:
    def test_each_event_stamps_its_own_source_tag_context(self):
        for spec in EVENTS.values():
            sigs = board_to_signals(spec, score_instruments(spec, 2.0, BALANCED),
                                    date(2026, 9, 16), BALANCED, 2.0)
            assert sigs, spec.key
            assert all(spec.label in s["thesis"] for s in sigs)

    def test_the_registry_filter_still_holds_for_every_event(self):
        for spec in EVENTS.values():
            sigs = board_to_signals(spec, score_instruments(spec, 2.0, BALANCED),
                                    date(2026, 9, 16), BALANCED, 2.0)
            assert not {s["pair"] for s in sigs} & {"DXY", "US500", "BTCUSD"}
```

- [ ] **Step 2: Run and read the failures**

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/test_nfp_reaction.py --no-cov -q
```

Expected: import errors for `EVENTS`, then signature failures.

- [ ] **Step 3: Add the dataclasses and the NFP spec first**

Build `Component`, `Exposure`, `EventSpec`, then define `EVENTS["NFP"]` from
today's existing constants so NFP is a pure re-expression of what is already
there. Make `compute_surprise` generic:

```python
def compute_surprise(spec: EventSpec, values: Dict[str, float]) -> Surprise:
    """Standardised composite. Positive = hawkish.

    A component is scored when its inputs are present and dropped otherwise,
    with the weights renormalised over what survived — so a headline-only
    entry still produces a usable score instead of one dragged toward zero.
    `delta_only` components (revisions, the FOMC dot shift, the tone dial)
    carry their own surprise and take no consensus.
    """
    z: Dict[str, float] = {}
    for c in spec.components:
        raw = values.get(c.key)
        if raw is None:
            continue
        if c.delta_only:
            diff = float(raw)
        else:
            cons = values.get(c.key + "_c")
            if cons is None:
                continue
            diff = float(raw) - float(cons)
        z[c.key] = (-diff if c.invert else diff) / c.sd

    live = {c.key: c.weight for c in spec.components if c.key in z}
    wsum = sum(live.values())
    composite = (sum(live[k] * z[k] for k in live) / wsum) if wsum else 0.0
    return Surprise(z=z, composite=composite)
```

Then rewrite `score_instruments`, `chain_leaves`, `timing_frame` and
`board_to_signals` to take `spec` as their first argument, reading
`spec.exposures` / `spec.chain` / `spec.phases` instead of the module globals.

- [ ] **Step 4: Prove NFP did not move**

Before adding CPI/PPI/FOMC, run only the pre-existing NFP tests:

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/test_nfp_reaction.py -k "Surprise or ScoreInstruments or Timing or BoardToSignals or ChainLeaves" --no-cov -q
```

Expected: PASS. If any NFP number moved, the generalisation is wrong — fix it
here, not by editing the assertion.

- [ ] **Step 5: Add the CPI, PPI and FOMC specs**

Component tables (`sd` in the component's own units; weights renormalise, so
they need not sum to 1):

| Event | Component | sd | weight | notes |
|---|---|---|---|---|
| CPI | `core_mm` | 0.10 | 0.45 | core m/m is the number that reprices the curve |
| | `head_mm` | 0.12 | 0.20 | |
| | `core_yy` | 0.12 | 0.25 | |
| | `head_yy` | 0.15 | 0.10 | |
| PPI | `core_mm` | 0.20 | 0.55 | PCE read-through lives here |
| | `head_mm` | 0.25 | 0.45 | |
| FOMC | `decision_bp` | 12.5 | 0.35 | delta-only; a full 25bp surprise is 2σ |
| | `dots_bp` | 15.0 | 0.35 | delta-only; current-year median dot shift |
| | `tone` | 1.0 | 0.30 | delta-only; −2..+2 dial, +2 = clearly hawkish |

Chains (node, sign):

```python
NFP:  (("jobs", +1), ("spending", +1), ("inflation", +1), ("rates", +1))
CPI:  (("prices", +1), ("inflation", +1), ("real income", -1), ("rates", +1))
PPI:  (("input costs", +1), ("margins", -1), ("consumer prices", +1), ("rates", +1))
FOMC: (("policy", +1), ("front end", +1), ("real yields", +1), ("liquidity", -1))
```

**Exposures.** Same twelve symbols per event. NFP's table is copied verbatim
from today's `EXPOSURES`. The other three have **the growth sign baked in**:
for CPI/PPI/FOMC a risk asset's growth beta is *negative* where NFP's is
positive, because a hawkish price shock is a growth negative, not a growth
positive. That single sign flip is the stagflation asymmetry, and it is why
one shared beta table could never have served four events. Magnitudes scale
with how directly the event reprices the curve: FOMC > CPI > NFP > PPI.

`(symbol, beta_rate, beta_growth, unit, unit_label, decimals)` —

```python
# CPI — the curve reprices harder than on payrolls, and every risk asset's
# growth beta is negative: hot prices squeeze real income and tighten policy.
("XAUUSD", -1.15, -0.30,   14.0, "USD",       2),
("XAGUSD", -1.20, -0.35,    0.38, "USD",      3),
("DXY",     1.10, -0.20,    0.42, "index pts", 2),
("EURUSD", -0.95, -0.10,   52.0, "pips",      0),
("GBPUSD", -0.90, -0.10,   48.0, "pips",      0),
("USDJPY",  1.20, -0.15,   70.0, "pips",      0),
("AUDUSD", -1.00, -0.45,   46.0, "pips",      0),
("USDZAR",  1.20, -0.55,    0.17, "ZAR",      3),
("US500",  -0.85, -0.55,   34.0, "pts",       1),
("NAS100", -1.05, -0.65,  175.0, "pts",       0),
("US10Y",   1.15,  0.10,    0.08, "%",        3),
("BTCUSD", -0.90, -0.40, 1100.0, "USD",       0),

# PPI — same shape as CPI, smaller magnitude. It matters for the PCE
# read-through, not in its own right, so it moves about 60% of a CPI.
("XAUUSD", -0.90, -0.25,    8.0, "USD",       2),
("XAGUSD", -0.95, -0.30,    0.24, "USD",      3),
("DXY",     0.85, -0.15,    0.26, "index pts", 2),
("EURUSD", -0.75, -0.10,   32.0, "pips",      0),
("GBPUSD", -0.70, -0.10,   30.0, "pips",      0),
("USDJPY",  0.95, -0.10,   42.0, "pips",      0),
("AUDUSD", -0.80, -0.35,   28.0, "pips",      0),
("USDZAR",  0.95, -0.45,    0.10, "ZAR",      3),
("US500",  -0.65, -0.40,   20.0, "pts",       1),
("NAS100", -0.80, -0.50,  100.0, "pts",       0),
("US10Y",   0.90,  0.10,    0.04, "%",        3),
("BTCUSD", -0.70, -0.30,  650.0, "USD",       0),

# FOMC — the purest rate event on the calendar and the largest mover.
("XAUUSD", -1.30, -0.20,   18.0, "USD",       2),
("XAGUSD", -1.35, -0.30,    0.48, "USD",      3),
("DXY",     1.20, -0.15,    0.55, "index pts", 2),
("EURUSD", -1.05, -0.10,   65.0, "pips",      0),
("GBPUSD", -1.00, -0.10,   60.0, "pips",      0),
("USDJPY",  1.30, -0.15,   90.0, "pips",      0),
("AUDUSD", -1.10, -0.45,   58.0, "pips",      0),
("USDZAR",  1.30, -0.60,    0.22, "ZAR",      3),
("US500",  -1.00, -0.60,   45.0, "pts",       1),
("NAS100", -1.20, -0.70,  230.0, "pts",       0),
("US10Y",   1.30,  0.10,    0.11, "%",        3),
("BTCUSD", -1.00, -0.45, 1500.0, "USD",       0),
```

**Phases.** NFP/CPI/PPI share one list (they are all 08:30 NY); FOMC has its
own. Offsets are minutes from t0 except where an absolute SAST window is
given — "Your window" is the desk's own local session and must not drift with
US daylight saving, which is exactly what a relative offset would do.

```python
# 08:30 NY events — NFP, CPI, PPI (t0 = 08:30 New York)
Phase("Pre-release drain",  start_min=-20, end_min=0,
      note="Books pulled, spreads widen, depth collapses. Stops sitting in "
           "the book are cheapest to take here."),
Phase("Algo impulse",       start_min=0,   end_min=2,
      note="First move is headline-only and often reverses once the detail "
           "is read. Two-sided."),
Phase("Repricing",          start_min=2,   end_min=15,
      note="The composite starts to matter. This is where the board below "
           "is most likely to be right."),
Phase("Fade window",        start_min=15,  end_min=60,
      note="Partial retracement of the impulse is the base case unless the "
           "composite is an outlier."),
Phase("US cash open",       start_min=60,  end_min=120,
      note="09:30 New York. Equity flow can overwrite the FX read, "
           "especially for gold."),
Phase("Your window",        start_sast=time(17, 0), end_sast=time(20, 0),
      note="Post-release continuation or drift. You are trading the "
           "aftermath, not the event."),

# FOMC (t0 = 14:00 New York = ~20:00 SAST — you are live for this one)
Phase("Pre-decision freeze", start_min=-30, end_min=0,
      note="Liquidity vanishes. Nobody wants inventory into a dot plot."),
Phase("Statement + dots",    start_min=0,   end_min=2,
      note="Both land at once, and the algo read is the dots, not the "
           "prose."),
Phase("First repricing",     start_min=2,   end_min=30,
      note="The curve moves before the equity market decides what it "
           "thinks. The board is most likely to be right here."),
Phase("Presser",             start_min=30,  end_min=90,
      note="The chair speaks at +30. The most common FOMC pattern on the "
           "tape is the presser reversing the statement move outright — do "
           "not marry the first leg."),
Phase("Cash close",          start_min=90,  end_min=120,
      note="16:00 New York. Positioning into the close overwrites the "
           "macro read."),
Phase("Your window",         start_sast=time(20, 0), end_sast=time(23, 0),
      note="Unlike the 08:30 releases you are awake for this one. That is "
           "a reason for smaller size, not larger."),
```

Note what FOMC's list does **not** contain: a US-cash-open phase. 09:30 New
York is four and a half hours *before* an FOMC decision, so shifting the 08:30
frame would have printed a phase that already happened — which is why the
phases belong to the spec rather than being derived by offsetting one list.

- [ ] **Step 6: Run the whole file**

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/test_nfp_reaction.py --no-cov -q
```

Expected: PASS.

- [ ] **Step 7: Bump and show the diff**

```bash
.venv/Scripts/python.exe deploy/sync_version.py 1.10.28
```

---

## Task 3: The page

**Files:**
- Modify: `pages/nfp_reaction.py`

**Interfaces:**
- Consumes: `EVENTS`, `compute_surprise`, `score_instruments`, `chain_leaves`,
  `timing_frame`, `board_to_signals` (Task 2); `next_release` (Task 1).

- [ ] **Step 1: Add the event selector to the sidebar**

First control in the sidebar, above the release date (sidebar contract: page
controls first, then `st.divider()`, then `render_sidebar_nav()`):

```python
ev_key = st.selectbox("Event", list(EVENTS), index=0, key="evt_event",
                      format_func=lambda k: EVENTS[k].label)
spec = EVENTS[ev_key]
default_date = next_release(spec.calendar_key) or date.today()
rel_date = st.date_input("Release date", value=default_date, key="evt_date")
```

When `next_release` returns `None` the seed list has run out; render
`st.caption("No scheduled date on file for this event — enter it manually.")`
rather than silently defaulting to today.

- [ ] **Step 2: Drive the input form off `spec.components`**

Replace the seven hard-coded `st.number_input` calls with a loop that renders
each component — `actual` plus `consensus` for a paired component, a single
input for a `delta_only` one — laid out four columns wide. FOMC's `tone`
renders as `st.slider("Statement / presser tone", -2.0, 2.0, 0.0, 0.5)`.
Keys must be namespaced per event (`f"{spec.key}_{c.key}"`) so switching
events does not carry a stale widget value across.

- [ ] **Step 3: Replace the fixed metric row**

Show the composite first, then one metric per scored component, from
`s.z`. Do not assume four.

- [ ] **Step 4: Four literal persist calls**

`tests/test_signal_sweep.py` finds tags only as string literals, so the
dispatch is explicit:

```python
def _persist_for(spec, signals) -> int:
    """Four literal calls, not one computed tag.

    `tests/test_signal_sweep.py` scans for `persist_signals("literal", ...)`;
    a computed tag would be invisible to it and the page would silently drop
    out of the sweep registry. Four branches is the price of that guard, and
    the guard is worth more than the branches.
    """
    if spec.key == "NFP":
        return persist_signals("nfp_reaction", signals)
    if spec.key == "CPI":
        return persist_signals("cpi_reaction", signals)
    if spec.key == "PPI":
        return persist_signals("ppi_reaction", signals)
    return persist_signals("fomc_reaction", signals)
```

Add `"event": spec.key` to the `log_tool_usage` payload and to its dedupe key,
and keep the single `"nfp_reaction"` literal for `log_tool_usage` — the audit
trail stays one stream with an `event` field, since it is queried by tool.

- [ ] **Step 5: Retitle**

`st.title("Event Reaction Map")`, `page_title="EVNT · Event Reaction Map"`,
sidebar header `### 🧾 EVENT REACTION MAP`, caption
`"Release surprise → regime-aware transmission chain → instrument exposure"`.
The "betas are priors" caption names the event's own source tag.

- [ ] **Step 6: Smoke each event**

```bash
PYTHONIOENCODING=utf-8 PYTEST_CURRENT_TEST=smoke .venv/Scripts/python.exe -c "
from streamlit.testing.v1 import AppTest
for i in range(4):
    at = AppTest.from_file('pages/nfp_reaction.py', default_timeout=120)
    at.run()
    at.selectbox[0].select_index(i); at.run()
    print(i, at.selectbox[0].value, 'exceptions:', [str(e.value) for e in at.exception])
"
```

Expected: four lines, every one `exceptions: []`.

- [ ] **Step 7: Prove persistence per event with the store stubbed**

Stub `signal_store.persist_signals` and `tool_log.log_tool_usage` before
importing the page (as in the NFP wiring plan's verification), drive a hawkish
input per event, and confirm the tag matches the event and the pairs are
registry-resolvable. **Do not run this unstubbed** — it would write signals
for a release that has not happened.

- [ ] **Step 8: Bump and show the diff**

```bash
.venv/Scripts/python.exe deploy/sync_version.py 1.10.29
```

---

## Task 4: Register, document, verify

**Files:**
- Modify: `src/pages_lib/navigation.py`, `src/services/signal_sweep.py`,
  `.foglamp/scan.json`, `docs/README.md`, `docs/System_Guide.md`

- [ ] **Step 1: Nav**

```python
NavEntry("EVNT", "Event Reaction Map",    "🧾", "pages/nfp_reaction.py"),
```

replacing the `NFPR` line at `src/pages_lib/navigation.py:120`.

- [ ] **Step 2: Sweep registry**

Replace the single `nfp_reaction` entry with four, all pointing at the same
page, and add four `PREPARE` hooks that select the event so each pass actually
exercises that event's code path:

```python
("nfp_reaction",  "pages/nfp_reaction.py"),
("cpi_reaction",  "pages/nfp_reaction.py"),
("ppi_reaction",  "pages/nfp_reaction.py"),
("fomc_reaction", "pages/nfp_reaction.py"),
```

```python
def _select_event(key: str):
    def prepare(at) -> None:
        for box in at.selectbox:
            if (box.label or "") == "Event":
                box.select(key)
                at.run()
                return
    return prepare

PREPARE = {
    "risk_reversal": _rr_use_free_proxy,
    "nfp_reaction": _select_event("NFP"),
    "cpi_reaction": _select_event("CPI"),
    "ppi_reaction": _select_event("PPI"),
    "fomc_reaction": _select_event("FOMC"),
}
```

Note in a comment that all four persist nothing at default inputs — that is
the composite gate working, and the runs are smoke coverage, not a scan.

- [ ] **Step 3: Update the map**

`sweeper` node `sub` and the `sweeper → pages` edge label go from
`28 pages under AppTest` to `31 pages under AppTest` (edge labels are capped
at 24 chars; this is 22). Then:

```bash
.venv/Scripts/python.exe .foglamp/introspect.py && .venv/Scripts/python.exe .foglamp/render.py --check
```

- [ ] **Step 4: Docs**

Rewrite the NFP row in `docs/README.md` and `docs/System_Guide.md` as an
Event Reaction Map row covering all four events, naming the four source tags
and the stagflation asymmetry. Then:

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe docs/generate_guide_pdf.py
```

- [ ] **Step 5: Full verification**

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest
```

```bash
.venv/Scripts/python.exe -m py_compile pages/nfp_reaction.py pages/event_week_vol_tab.py src/core/nfp_reaction.py src/core/event_calendar.py src/pages_lib/navigation.py src/services/signal_sweep.py
```

```bash
SIGNAL_SWEEP_NO_EMAIL=1 PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m src.services.signal_sweep --only nfp_reaction,cpi_reaction,ppi_reaction,fomc_reaction --no-sync
```

Expected: 4/4 pages ok, **0 signals saved**.

Then in the browser (`preview_start` name `dashboard` → EVNT): switch through
all four events and confirm the chain relabels (jobs/spending for NFP,
prices/real income for CPI, input costs/margins for PPI, policy/liquidity for
FOMC), the input form changes shape, FOMC shows a presser phase and an evening
SAST time, and the theme stays black-on-terminal-green.

- [ ] **Step 6: Bump, run deploy verify, show the owner the diff**

```bash
.venv/Scripts/python.exe deploy/sync_version.py 1.10.30 && .venv/Scripts/python.exe deploy/verify_deploy.py
```

---

## Known limits, carried honestly

- **The CPI/PPI/FOMC betas are priors with less standing than NFP's.** They are
  reasoned from the transmission channels, not fitted. The four separate source
  tags exist precisely so the Source Scorecard settles each one on its own
  evidence — expect the first useful read after roughly a dozen releases per
  event, which is a year for FOMC.
- **The FOMC tone dial is a judgement, not a measurement.** It is the one input
  on the page another person could not reproduce from the release. It carries
  the smallest weight of the three FOMC components for that reason.
- **Seeded release dates go stale.** CPI/PPI/FOMC dates are hand-maintained
  lists; `next_release` returns `None` rather than extrapolating when one runs
  out. Re-seed from bls.gov and federalreserve.gov annually.
- **The sweep cannot score these events.** Every page in the sweep except this
  one reads a live market; this one reads a human typing a release. Four sweep
  entries buy smoke coverage of four code paths, nothing more.
