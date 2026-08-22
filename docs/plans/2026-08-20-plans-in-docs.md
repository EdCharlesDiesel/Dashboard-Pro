# Plans live in docs/plans — the rule, the guard, and nine missing records

> **For agentic workers:** Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make "the plan is written into `docs/plans/`" a rule that fails loudly instead of one that quietly stops being followed, and restore the nine plan documents behind versions 1.10.20–1.10.28.

**Architecture:** Three parts, smallest first. A short addition to the existing `### Plan first` block in `.claude/CLAUDE.md` naming the failure mode. A guard test asserting the current `VERSION` is claimed by some plan — the check that would have failed nine bumps ago. Then the nine reconstructions, each labelled as such.

**Tech Stack:** Markdown, pytest.

**Spec:** The owner's request, 2026-08-20: *"add as a rule to always include the plan in docs folder"*, plus full-fidelity backfill and a test.

## Global Constraints

- **Never commit.** The owner reviews and commits.
- `VERSION` reads **1.10.28**, so this plan takes **1.10.31** — and this plan is itself saved to `docs/plans/` as its own first step, not last.
- Branch `DEV-04/Market-Overview`.
- **The CLAUDE.md edit is purely additive** — zero deletions, proven by `git diff --numstat`, as with the working-principles section.
- **Reconstructions are labelled.** Every backfilled plan carries a header saying it was reconstructed on 2026-08-20 from the session transcript. None is presented as contemporaneous.
- Reconstructions record what was *planned*, including anything that later changed during execution. Where the plan and the shipped code diverged, the plan text stands and a note records the divergence.

---

## Context

`VERSION` has moved **1.10.19 → 1.10.28** while the newest file in `docs/plans/` is `2026-08-18-signal-kit.md`. Nine bumps, no plan documents — and `938d3e6 … V1.10.22` is already committed, so the repo now contains shipped code whose plan does not exist.

**How it happened, precisely.** Plan mode assigns one working file, `~/.claude/plans/shimmering-petting-puzzle.md`, and reuses it for every plan in the session. CLAUDE.md step 1 says to write the plan into `docs/plans/YYYY-MM-DD-<slug>.md`; that was done for the first two plans on the 18th and then stopped. Every later plan lived only in the reused file, so **each new plan overwrote its predecessor**. That directory holds exactly one file today, containing only the most recent plan.

So this is not a missing rule. The rule was written, in the file loaded at the start of every session, and it stopped being followed after two uses with nothing to notice. That is what the guard is for: a rule with no failure mode is a preference.

**Why the guard can work.** Plans state their version in a consistent form — `takes **1.10.16**`, `takes **1.10.19**` — present in 7 of the 10 existing plans and in all recent ones. A test that asserts the current `VERSION` string appears somewhere under `docs/plans/` needs no new convention, and would have gone red the moment `VERSION` became 1.10.20 with no plan naming it.

### The nine missing records

| Version | Plan | Date |
|---|---|---|
| 1.10.20 | CLAUDE.md working principles (Karpathy adaptation) | 2026-08-18 |
| 1.10.21 | `.foglamp` scan — session subagents and audit hook | 2026-08-20 |
| 1.10.22 | Trade Journal — realised growth in money **(already committed)** | 2026-08-20 |
| 1.10.23 | Risk Suite — pull the live balance | 2026-08-20 |
| 1.10.24 | Swing Playbook & Threat Board — read the live account | 2026-08-20 |
| 1.10.25 | Threat Board — drive it from the MT5 book | 2026-08-20 |
| 1.10.26 | Threat Board — a red component vetoes a green headline | 2026-08-20 |
| 1.10.27 | Threat sentry hook — same book, same equity, never silent | 2026-08-20 |
| 1.10.28 | Entry alerts to Telegram | 2026-08-20 |

Dates come from the session: 1.10.20 was approved before the date rolled over, everything from 1.10.21 after.

---

## Task 1: Save this plan first, then write the rule

**Files:** Create `docs/plans/2026-08-20-plans-in-docs.md` · Modify `.claude/CLAUDE.md`

- [ ] **Step 1: Copy this plan to `docs/plans/2026-08-20-plans-in-docs.md` before anything else.** The plan about not losing plans is not allowed to be the tenth lost plan.

- [ ] **Step 2: Record the pre-change state**

```bash
wc -l .claude/CLAUDE.md            # expect 352
git diff --numstat .claude/CLAUDE.md
```

- [ ] **Step 3: Insert into the `### Plan first` block, immediately after item 1**, keeping the numbering intact:

```markdown
   **Write it to that path first, before implementing — not afterwards.** The
   plan-mode scratch file is not a record: the tooling assigns one file per
   session and reuses it, so each new plan silently overwrites the last. On
   2026-08-20 that ate eight plans, covering versions 1.10.20 to 1.10.28, one
   of which was already committed. A plan that exists only in a chat window
   did not happen. `tests/test_plans_are_recorded.py` enforces this by
   asserting the current `VERSION` is named by some plan in `docs/plans/`.
```

- [ ] **Step 4: Prove nothing was removed** — `git diff --numstat .claude/CLAUDE.md`, deletions column must be `0`.

---

## Task 2: The guard

**Files:** Create `tests/test_plans_are_recorded.py`

- [ ] **Step 1: Write the failing test**

```python
"""Every version bump leaves a plan behind.

CLAUDE.md has required plans in docs/plans/ since long before 2026-08-20, and
they still stopped being written after two uses — because nothing failed when
they were not. VERSION reached 1.10.28 with the newest plan naming 1.10.19,
and one of those unexplained versions was already committed.
"""
from __future__ import annotations

import os
import re

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PLANS = os.path.join(_REPO, "docs", "plans")
_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}-[a-z0-9-]+\.md$")


def _plan_files() -> list:
    return [f for f in os.listdir(_PLANS) if f.endswith(".md")]


def test_the_current_version_is_named_by_a_plan():
    version = open(os.path.join(_REPO, "VERSION"), encoding="utf-8").read().strip()
    hits = [f for f in _plan_files()
            if version in open(os.path.join(_PLANS, f), encoding="utf-8").read()]
    assert hits, (
        f"VERSION is {version} but no plan in docs/plans/ mentions it - the "
        f"change that bumped it has no written plan")


def test_plan_filenames_follow_the_convention():
    # YYYY-MM-DD-<slug>.md, so the directory sorts chronologically.
    bad = [f for f in _plan_files() if not _NAME.match(f)]
    assert not bad, f"misnamed plans: {bad}"


def test_the_plans_directory_is_not_empty():
    assert _plan_files()
```

- [ ] **Step 2: Run it and watch `test_the_current_version_is_named_by_a_plan` fail** — `VERSION` is 1.10.28 and no plan says so. That failure is the whole point; read it before fixing it.

- [ ] **Step 3:** It goes green in Task 3 when 1.10.28's plan is restored, and for this plan's own bump when Step 1's file records 1.10.29. No production code changes to make it pass.

---

## Task 3: Reconstruct the nine

**Files:** Create nine files under `docs/plans/`, per the table above.

- [ ] **Step 1: Write each at full fidelity** from the approved text echoed back at approval time — Goal, Architecture, Tech Stack, Spec, Global Constraints, Context, numbered Tasks with checkbox Steps, and Verification.

- [ ] **Step 2: Head each one with this banner**, so nobody mistakes a reconstruction for a contemporaneous record:

```markdown
> **Reconstructed 2026-08-20** from the session transcript. The plan was
> approved and executed on the date below, but was never written to
> `docs/plans/` at the time — see `2026-08-20-plans-in-docs.md`. The text is
> the approved plan; the checkboxes are left unticked because the record of
> which steps ran lives in the transcript, not here.
```

- [ ] **Step 3: Note divergences.** Where execution departed from the plan, add a short `## What actually happened` section rather than editing the plan to match. Known cases: 1.10.24's expected "8 usable / 1 unstopped" became 9/0 when a stop was added mid-run; 1.10.25 and 1.10.26 each required one pre-existing test to be updated; 1.10.28's scanner change was implemented before its tests, with red proven afterwards by reverting to HEAD.

- [ ] **Step 4: Run the guard** — `pytest tests/test_plans_are_recorded.py -q --no-cov`, all green.

---

## Verification

Evidence before claims.

1. **The guard goes red then green:** it must fail in Task 2 Step 2 with `VERSION is 1.10.28 but no plan ... mentions it`, and pass after Task 3.

2. **Nothing was removed from CLAUDE.md:** `git diff --numstat .claude/CLAUDE.md` — deletions column `0`.

3. **Every bumped version now has a plan:**
   ```bash
   for v in 1.10.20 1.10.21 1.10.22 1.10.23 1.10.24 1.10.25 1.10.26 1.10.27 1.10.28 1.10.30; do
     printf "%s: %s\n" "$v" "$(grep -rl "$v" docs/plans/ | head -1)"
   done
   ```
   Expected: a file named for each, including this plan's own 1.10.30.

4. **The directory reads chronologically:** `ls docs/plans/` — 20 files, sorting cleanly from 2026-08-14 to 2026-08-20.

5. **Full suite:** `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest -q`
   Expected: coverage ≥ 80%, the 4 known pre-existing failures, no fifth.

6. **Deploy:** `.venv/Scripts/python.exe deploy/sync_version.py 1.10.30`, rebuild, `deploy/verify_deploy.py` → 1.10.30, four containers in sync. Docs-only, but the rule is once per plan and this plan does not get an exemption from itself.

7. Show the owner the diff. **Never commit.**
