# Threat Board — drive it from the MT5 book

> **Reconstructed 2026-08-22** from the session transcript. Approved and
> executed on 2026-08-20 but never written to `docs/plans/` at the time — see
> `2026-08-20-plans-in-docs.md`. The text is the approved plan; checkboxes are
> left unticked because the record of which steps ran lives in the transcript.

**Goal:** Make the Threat Board read your real open positions from the stored MT5 book instead of a hand-typed table, and fix the two unit errors that would otherwise make its risk numbers nonsense.

**Architecture:** Three layers, smallest first. `threat_core` stops hardcoding pip maths and asks `src/instruments/registry.py` — the documented single source of truth it currently duplicates. A new `positions_from_book()` converts stored rows into `Position` objects, absorbing three format mismatches. The page then reads the book and drops its editor.

**Tech Stack:** Python 3.14, Streamlit, pytest.

**Spec:** The owner's request, 2026-08-20: "wire threat board to the MT5 book", with metals handled via the registry and the manual table replaced entirely.

## Global Constraints

- **Never commit.** The owner reviews and commits.
- `VERSION` reads **1.10.24**, so this plan takes **1.10.25**.
- Branch `DEV-04/Market-Overview`.
- **TDD.** `src/core/threat_core.py` is inside `--cov=src` and already has **40 tests** — they are the safety net for changing it, and must stay green.
- **The registry lookups fall back to today's behaviour.** Any pair the registry does not know keeps the existing JPY/else logic, so no currently-passing test changes meaning.
- **Do not drop the `threat_positions` table.** The editor goes; the table and its rows stay on disk. Dropping it is irreversible and was not asked for.

---

## Context

The board reads `threat_positions`, a table filled in by hand (`threat_core.py:345`). `grep` for `open_positions|mt5` in that module returns **0** — it has never seen the real book. It holds **0 rows**, so the board renders nothing while nine real positions are open.

Wiring the book in naively would be worse than leaving it. Four mismatches, all silent:

| Mismatch | Stored book | `Position` expects | Consequence |
|---|---|---|---|
| Pair format | `"USD/ZAR"` | `"USDZAR"` — `quote` is `pair[3:6]` | `quote` becomes `"/ZA"`; every quote-currency branch misfires |
| Direction | `"SHORT"` | `"short"` — compared as `p.direction == "long"` | **Every long silently counts as short** in the exposure netting |
| Stop | `None` on one live leg | `stop: float` | `abs(entry - stop)` raises `TypeError` |
| Units | metals, ZAR crosses | 0.0001 pip, 100k contract | see below |

**The unit errors are the reason this needs the registry.** `pip_size()` returns 0.0001 for anything not JPY, and `pip_value_usd()` falls back to `$10/lot` for exotic quotes. Measured against the live book:

```
XAU/USD 0.2 lots, stop 136.9 away
  threat_core : 1,369,050 pips x $2.00  =  $2,738,100
  registry    :     1,369 pips x $2.00  =      $2,738.00   (true: $2,738.10)

USD/ZAR 0.2 lots
  threat_core : $2.00/pip   (the $10/lot exotic fallback)
  registry    : $0.124/pip  (pip 0.62 per lot)             -> 16x overstated
```

Seven of the nine open positions — two metals and five ZAR crosses — are materially wrong under the current maths. A threat board reporting $2.7m of risk on a $3.6k account is not a rounding problem.

`src/core/config.py`, `signals.py` and `todays_trades.py` already import the registry, so there is no layering objection to `threat_core` doing the same.

*Noticed in passing, not touched:* `src/core/config.py.tmp.4292.b3bb12556262`, a stray temp file.

---

## Task 1: `threat_core` asks the registry

**Files:** Modify `src/core/threat_core.py` (`pip_size`, `pip_value_usd`) · Test `tests/test_threat_core.py`

**Interfaces:** a private `_instrument(pair)` returning the registry entry for an unslashed pair, or `None`.

- [ ] **Step 1: Failing tests** — gold's pip comes from the registry (0.1); gold stop risk is dollars not millions; a ZAR cross uses its real pip value; FX majors are unchanged; an unknown pair keeps the old fallback.
- [ ] **Step 2: Run, watch them fail** — 0.0001 != 0.1 and $2.7m != $2,738.
- [ ] **Step 3: Implement.** `_instrument(pair)` does `INSTRUMENTS.get(f"{p[:3]}/{p[3:6]}")`; `pip_size()` returns `inst.pip_size` when found and the existing JPY/else rule otherwise; `pip_value_usd()` uses the registry where the code currently guesses.
- [ ] **Step 4: Green — and the whole file green.** All 40 pre-existing tests must still pass; any that break mean the fallback is not preserving behaviour.

---

## Task 2: `positions_from_book()`

**Files:** Modify `src/core/threat_core.py` · Test `tests/test_threat_core.py`

**Interfaces:** `positions_from_book(rows) -> tuple[list[Position], list[dict]]` — `(positions, unstopped_rows)`.

- [ ] **Step 1: Failing tests**, one per mismatch: the slash is stripped so `quote` parses; the direction is lowercased (**the silent one** — exposure netting does `1 if direction == "long"`, so an uppercased `"LONG"` counts as a short and the whole board inverts without raising); a position with no stop is separated, not dropped; a zero stop counts as no stop (both platforms spell "no stop" as `0.0`); a mixed book splits correctly; an empty book is two empty lists.
- [ ] **Step 2: Run, watch them fail.**
- [ ] **Step 3: Implement** — strip `/`, lowercase the direction, coerce floats, route rows without a usable stop into the second list.
- [ ] **Step 4: Green.**

---

## Task 3: The page reads the book

**Files:** Modify `pages/threat_board_tab.py`

- [ ] **Step 1: Replace the position source.** Drop `_positions_editor(conn)` and `tc.load_positions(conn)`; read `open_positions.load()` through `tc.positions_from_book()`. `conn` stays — `ensure_tables` and `journal` still use it.
- [ ] **Step 2: Show what the book says** above the gauges: position count and book age (red past 15 minutes), an explicit **unstopped** callout naming those pairs (unbounded risk, excluded from cluster maths because it cannot be quantified — never omitted in silence), and an empty-book message pointing at `logs/mt5_sync.log`.
- [ ] **Step 3: Remove the editor function** and any helpers it orphans. **Leave `ensure_tables` and the `threat_positions` table alone.**
- [ ] **Step 4:** `python -m py_compile pages/threat_board_tab.py`.

---

## Verification

1. **Unit tests:** the 40 pre-existing plus the new ones. A pre-existing failure means the registry fallback changed behaviour it should not have.
2. **The board's numbers against the real book:** 9 in book → 8 usable, 1 unstopped, every risk figure in the hundreds or low thousands, none in the millions.
3. **The page renders from the book** — gauges populated, the unstopped leg named, no "Add at least one position" prompt, no editor.
4. **A sanity cross-check against the terminal** (`mcp__mt5__get_positions`): pairs and directions must match one for one. A direction inverted here is the failure mode Task 2 exists to prevent, and it would not raise anything.
5. **Full suite:** coverage ≥ 80%.
6. **Deploy:** 1.10.25, four containers in sync.
7. Show the owner the diff. **Never commit.**

## What actually happened

The first implementation routed **all** pip values through the registry and broke three existing tests. Reading them showed why: the JPY branch computes pip value from the **live** USD/JPY rate passed in, while the registry's static 9.09 implies USD/JPY ≈ 110 — stale by roughly 35%. The change was narrowed so the registry governs only pip *size* (a contract constant) and the exotic branch (where the code admitted it was guessing $10/lot); the JPY and USD branches were left untouched.

**One pre-existing test was changed deliberately.** `test_exotic_quote_falls_back_to_flat_rate` asserted the crude $10/lot for EUR/GBP, which now returns its real 12.5. It was rewritten to assert the registry value with the reason recorded in the test, and a second test added proving the flat-rate fallback still governs symbols the registry does not know.

Result against the live book: **9 → 8 usable, 1 unstopped**, gold at **$2,738.10** (matching the true value to the cent), total stop risk **$6,649.10** — which agrees to 23 cents with the Trade Journal's independently computed "if all SL hit" of −$6,648.87. Terminal cross-check matched all nine pairs and directions, including the EUR/ZAR leg carrying `sl: 0`.
