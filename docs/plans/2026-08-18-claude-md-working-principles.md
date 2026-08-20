# CLAUDE.md Working Principles — Implementation Plan

> **Reconstructed 2026-08-20** from the session transcript. This plan was
> approved and executed on 2026-08-18, but was never written to `docs/plans/`
> at the time — see `2026-08-20-plans-in-docs.md` for why. The text below is
> the approved plan; checkboxes are left unticked because the record of which
> steps ran lives in the transcript, not here.

**Goal:** Add one `### Working principles` section to `.claude/CLAUDE.md`, adapting the four Karpathy principles to this codebase — with carve-outs where importing them verbatim would break documented practice here — while removing nothing that is already there.

**Architecture:** A single insertion at one anchor point, before `## Domain`. Purely additive: **zero deletions**, which is the machine-checkable proof that the existing project rules were preserved. No other file changes.

**Tech Stack:** Markdown. `git diff --numstat` as the acceptance test.

**Spec:** `github.com/multica-ai/andrej-karpathy-skills` — `CLAUDE.md` and `skills/karpathy-guidelines/SKILL.md`, both read 2026-08-18.

## Global Constraints

- **Never commit.** The owner reviews and commits.
- **A plan gets its own bump.** `VERSION` reads **1.10.19**, so this plan takes **1.10.20**. Bump + rebuild + `deploy/verify_deploy.py` once per plan. Verified: `.claude/` is not in `.dockerignore` and `ls -a /app` in `dashboard-pro:1.10.19` finds `.claude`, so the rebuild genuinely changes the image.
- Branch `DEV-04/Market-Overview`.
- **Preserve every existing rule.** No line of the current file is edited, reordered or removed. The diff must show `0` deletions.
- **The section must obey its own principles**: ~60 lines, no restructuring of the file, every claim in it factually verified against this repo.

---

## Context

The owner asked to bring the Karpathy guidelines into their project instructions — adapted, not copied, preserving existing rules.

**What the source encodes** (four principles): Think Before Coding (surface assumptions, present interpretations, push back, stop when unclear) · Simplicity First (minimum code, nothing speculative, no unrequested abstraction or configurability, no error handling for impossible scenarios) · Surgical Changes (touch only what you must, don't refactor what isn't broken, match existing style, mention don't delete dead code, remove only your own orphans) · Goal-Driven Execution (convert tasks into verifiable goals, checkpoints per step).

**What the audit of the existing 283-line file found** — counts are `grep -ic` against `.claude/CLAUDE.md`:

| Concept | Mentions today | Verdict |
|---|---|---|
| `assum*` | **0** | genuine gap |
| `simplic*` / `YAGNI` / `speculat*` / `minimal` | **0** | genuine gap |
| `surgical` | **0** | genuine gap (partially implied by "preserve behavior byte-for-byte") |
| `success criteria` | **0** | gap on the *front* end; the back end exists as "Evidence before completion claims" |

**Two rules must not be imported verbatim** — this is the whole of "adapt, don't copy":

1. **"No error handling for impossible scenarios"** would license stripping the deliberate best-effort `try/except` boundaries in `account_state`, `open_positions`, `precomputed`, `score_history` and `alert_service` (6+ services carry an explicit "best-effort" comment). Those exist so an unreachable Postgres degrades a page instead of breaking a trading session. The carve-out is mandatory or this rule causes an outage.
2. **"Don't refactor things that aren't broken"** would contradict a practice the file already sanctions: *"legacy hand-rolled charts migrate to ChartKit opportunistically (leading-indicators and forecast_tab are the reference migrations)."* The carve-out names that exception and confines it to its own change.

One further reconciliation: **"Simplicity First" reads as contradicting the existing "Return complete implementations — no TODO comments, no placeholder code."** They are orthogonal — *simple* means build less, *complete* means finish what you build — but unless that is said, the two rules look like they are fighting.

The two concrete anchors used in the text are real incidents from 2026-08-18, both verifiable in the session history: the cold-email misread (25 files, wrong domain) and the `grep -c` that printed `0` because Git Bash had rewritten `/app` and `ls` never ran.

---

## Task 1: Insert the section

**Files:**
- Modify: `.claude/CLAUDE.md` — one insertion immediately before `## Domain` (then line 67; line 65 ended `was 1.4.0.`, line 66 blank)
- Modify: `VERSION` → `1.10.20` via `deploy/sync_version.py`

- [ ] **Step 1: Record the pre-change state (the RED equivalent)**

Markdown has no test to fail, so the acceptance criterion is a measurement taken *before* the edit:

```bash
cd /c/x/Dashboard-Pro
wc -l .claude/CLAUDE.md          # expect 283
git diff --numstat .claude/CLAUDE.md   # expect no output
```

- [ ] **Step 2: Insert the `### Working principles` section verbatim before the `## Domain` line.**

Four principles, each with this repo's carve-outs: (1) Surface assumptions, never guess silently — anchored on the cold-email misread. (2) The smallest complete thing — with "not a licence to stub" and "not a licence to strip error handling". (3) Surgical by default — with the ChartKit migration as the standing exception. (4) Define the finish line first — with "a command that errored is not a check that passed", anchored on the `MSYS_NO_PATHCONV` incident.

Insert with a script anchored on the `## Domain` heading rather than a line number, so a stale line number cannot corrupt the file.

- [ ] **Step 3: Prove nothing was removed** — `git diff --numstat .claude/CLAUDE.md`; the deletions column must be `0`.

- [ ] **Step 4: Bump** — `.venv/Scripts/python.exe deploy/sync_version.py 1.10.20`

---

## Verification

1. **Nothing was removed** — `git diff --numstat .claude/CLAUDE.md`, deletions `0`.
2. **Heading order intact** — `grep -n "^#\{1,4\} " .claude/CLAUDE.md`; original headings plus `### Working principles` between the plan-first block and `## Domain`.
3. **Every factual claim is true** — each of the 5 named services carries "best-effort"; the ChartKit quote matches verbatim; the cited plan exists; "Return complete implementations" appears.
4. **No test regression** — full suite; coverage ≥ 80%, 4 known pre-existing failures.
5. **Deploy** — 1.10.20, four containers in sync.
6. **Show the owner the diff.** Never commit.

## What actually happened

Executed as planned: **69 added, 0 deleted**. One self-correction during execution — the text originally cited `/simplify` and `/code-review` by name as examples of invited cleanups; `ListSkills` returned empty for both, showing they are harness commands rather than repo or claude.ai skills, so a future session in this repo may not have them. The sentence was rewritten generically before the change was finished.
