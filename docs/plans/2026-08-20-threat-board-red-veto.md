# Threat Board — a red component vetoes a green headline

> **Reconstructed 2026-08-22** from the session transcript. Approved and
> executed on 2026-08-20 but never written to `docs/plans/` at the time — see
> `2026-08-20-plans-in-docs.md`. The text is the approved plan; checkboxes are
> left unticked because the record of which steps ran lives in the transcript.

**Goal:** Stop the board printing GREEN while one of its own components is pinned at 100/100 red. The headline state can never be greener than the worst single component, and the page says which component forced it.

**Architecture:** One new pure function in `threat_core`, `overall_state(total, comps)`, used by `build_report` in place of `band(total)`. The composite *score* is unchanged — it remains a legitimate magnitude — only the *state* changes, plus a `state_driver` entry in `detail` so the page can explain a headline that no longer matches the number beside it.

**Tech Stack:** Python 3.14, Streamlit, pytest.

**Spec:** The owner's decision, 2026-08-20, choosing option 1 of three: "let a red component veto the headline".

## Global Constraints

- **Never commit.** The owner reviews and commits.
- `VERSION` reads **1.10.25**, so this plan takes **1.10.26**.
- Branch `DEV-04/Market-Overview`.
- **TDD.** `threat_core` is inside `--cov=src` and now has 53 tests; they are the safety net.
- **The composite score is not touched.** `rep.score` keeps meaning "weighted average", so the journal's history stays comparable.

---

## Context

The board reports a correlated stop-out of **$6,080 — 173% of equity** and prints **GREEN — composite threat 30.0/100**. Its own component chart, read off the live page:

```
Concentration  100     <- red
Intervention     0
Squeeze          0
Calendar         0
Regime           0
```

`build_report` ends with `band(total)`, and `total` is the weighted mean:

```
(100 x 30 + 0 + 0 + 0 + 0) / 100 = 30.0   ->  band(30) = green   (amber >= 40)
```

**Concentration carries 30 of 100 points, so it can contribute at most 30 — and amber starts at 40.** No matter how extreme the correlated risk becomes, that component alone can never move the headline off green. The one condition that can empty the account is the one condition that cannot change the colour. The other four are yen-specific by design (`score_intervention`'s docstring: *"Only threatens accounts net SHORT yen"*), and this book holds no JPY, so half the scale is unreachable.

**Why "worst component" is the whole answer.** A weighted mean can never exceed its largest term, so `band(total) <= band(worst component)` always. The composite can only ever agree with the worst component or be greener than it — which is precisely the flaw. Taking the more severe of the two is therefore both the fix and, arithmetically, just "use the worst component". The implementation keeps the explicit `max()` of both anyway: it is one term, it states the intent in the code, and it stays correct if the weights are ever changed to something that is not a proper mean.

**Two things found while reading, neither changed here:**

1. `src/core/threat_sentry_hook.py:78` fires a **Telegram alert** whenever `prev != rep.state`. Nothing imports it, so it is dormant and this change will not send anything. Worth knowing before it is ever wired up, because after this change it would fire immediately.
2. That same hook still calls `tc.load_positions(conn)` — the hand-typed table the page stopped using — and defaults `EQUITY` to `935`. If switched on it will disagree with the page about both positions and equity. Flagged, not fixed; it became plan 1.10.27.

---

## Task 1: `overall_state()`

**Files:** Modify `src/core/threat_core.py` · Test `tests/test_threat_core.py`

**Interfaces:** `overall_state(total: float, comps: dict[str, float]) -> str`.

- [ ] **Step 1: Failing tests** — a red component forces red despite a green average; an amber component forces at least amber; all-green leaves the state alone; the worst component wins rather than the first; no components falls back to the average; and an invariant test asserting the state is never greener than any component across the whole range.
- [ ] **Step 2: Run, watch them fail** (`overall_state` undefined).
- [ ] **Step 3: Implement** beside `band()`, using a `_SEVERITY` ordering and `max(band(total), worst)`.
- [ ] **Step 4: Green.**

---

## Task 2: `build_report` uses it, and says why

- [ ] **Step 1:** Swap `band(total)` for `overall_state(total, comps)` in the return, and add `detail["state_driver"]` — the components sitting at the headline's severity — so the page can explain itself.
- [ ] **Step 2: Expect `test_build_report` to need one mechanical edit.** It asserts `rep.state == tc.band(total)`, the behaviour being replaced. Change it to `overall_state(total, expected)`; that preserves the test's intent — *state is derived from the scores* — rather than deleting a check. **Note the change explicitly in the summary; a test edited to go green is the thing that must never pass unremarked.**
- [ ] **Step 3: Whole file green.**

---

## Task 3: The page explains the veto

- [ ] **Step 1:** Append the driver to the headline (`RED — composite threat 30.0/100 · driven by Concentration`) and add a caption stating the rule: the headline follows the worst single component, not the average, so a maxed component cannot be averaged away. Without that sentence the pairing of "RED" with "30.0/100" reads as a broken widget.
- [ ] **Step 2:** `python -m py_compile pages/threat_board_tab.py`.

---

## Verification

1. **Unit tests:** 53 existing (one mechanically updated) plus the new ones.
2. **The live case flips:** `score 30.0 | old state green | new state red`.
3. **The board renders red with a reason** — composite still 30.0/100, driver named, component chart unchanged at Concentration 100.
4. **Nothing was alerted.** `threat_sentry_hook` is dormant; confirm by grep that nothing imports it.
5. **Full suite:** coverage ≥ 80%.
6. **Deploy:** 1.10.26, four containers in sync.
7. Show the owner the diff. **Never commit.**

## What actually happened

Executed as planned; 59 tests in the file passed. The rule was verified across the range, and every composite in that table bands green on its own:

```
concentration    0 -> composite  0.0 (green) -> headline green
concentration   55 -> composite 16.5 (green) -> headline amber
concentration   70 -> composite 21.0 (green) -> headline red
concentration  100 -> composite 30.0 (green) -> headline red
```

The pre-existing `test_aggregates_weighted_components` was updated as anticipated, with the date and reason recorded in the test itself, plus a new assertion on `state_driver`. The deployed board reads **RED — composite threat 30.0/100 · driven by Concentration**, with the explanatory caption beneath and the component scores untouched.
