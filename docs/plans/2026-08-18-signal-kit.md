# Dashboard-Pro Signal Kit — Implementation Plan

> **For agentic workers:** Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refit the supplied kit *structure* onto Dashboard-Pro's trading-signal domain — the `.claude/` machinery it is missing (5 skills, 4 subagents, 1 slash command, 1 hook, shared settings) plus 4 agent-facing reference docs — and delete the cold-email subproject built from a misreading of that diagram.

**Architecture:** Everything lands in the **existing** `.claude/`, not a subproject. The diagram is a shape, not content: each node maps to the Dashboard-Pro thing that already plays its role, and only genuinely-absent nodes get created. The five skills are the ten steps of `docs/Live_Session_Runbook.md`, so the skills and the runbook cannot drift apart. No Python is added — `src/services/` already holds every client the pipeline needs.

**Tech Stack:** Claude Code skills / subagents / hooks / settings, Markdown, JSON, Git Bash, pytest.

**Spec:** The structure diagram supplied 2026-08-18, refitted against `docs/Live_Session_Runbook.md` (the desk's real 10-step flow) and `.claude/skills/experienced-institutional-fx-trade/SKILL.md` (the output contract for any trade thesis).

## Global Constraints

- **Never commit.** The owner reviews and commits.
- **Skills in force:** `writing-plans` (this document), `test-driven-development` (every task opens with a failing test — the structural test is the RED for Markdown and JSON), `verification-before-completion` (no completion claim without fresh command output in the same message).
- **A plan gets its own bump.** `VERSION` reads **1.10.18**, so this plan takes **1.10.19**. Bump + rebuild + `deploy/verify_deploy.py` run **once per plan, not per task** (owner's decision, 2026-08-18).
- Branch `DEV-04/Market-Overview`. Never `Production`.
- **Never duplicate a single source of truth** — `.claude/CLAUDE.md` forbids it by name for `src/instruments/registry.py`, `src/pages_lib/navigation.py`, `src/ui/charts.py`. The same rule is applied here to `src/core/signals.py` and `docs/MT5_MCP.md`: the reference docs **point**, they never restate.
- **Nothing may weaken the four MT5 trade gates** (`MT5_ALLOW_TRADING`, `confirm=true`, `MT5_MAX_VOLUME`, terminal Algo Trading). No skill, agent or setting may set, suggest or work around them.
- No production Python is added. If a task seems to need a new client, the client already exists in `src/services/`.

---

## Context

The diagram supplied on 2026-08-18 was read as content rather than structure, and a literal `ai-cold-email-kit/` was built inside the repo — 25 files of Apollo/Maildoso cold-outreach tooling. That is the wrong domain entirely. Task 1 removes it.

The correct reading: the diagram is the *shape* of a Claude-Code-driven pipeline, and Dashboard-Pro already has most of it. What it genuinely lacks is the `.claude/` machinery — it has 12 process skills but **no `agents/`, no `commands/`, no `hooks/`, and no shared `settings.json`.**

Two findings from exploration drive the refit:

1. **`docs/Live_Session_Runbook.md` already documents the pipeline** — 10 steps from macro backdrop to post-trade review. Inventing a different set of skills would create a second, competing description of how the desk trades. The five skills are those ten steps, grouped.
2. **`.mcp.json` must not be created.** `docs/MT5_MCP.md` shows the server is registered at **user scope** (`claude mcp add mt5 -s user`), and `"mt5"` is present in `~/.claude.json`. A project-scoped duplicate would start a second `mt5_mcp.py` holding the same terminal handle — the exact two-processes-one-terminal failure that wedged this desk on 2026-08-07.

## The refit — every diagram node, mapped

| Diagram node | Dashboard-Pro fit | Action |
|---|---|---|
| `CLAUDE.md` (campaign brain) | `.claude/CLAUDE.md` | exists — unchanged |
| `.env.example` | `.streamlit/secrets.toml.example` | exists — unchanged |
| `.mcp.json` | `mt5` registered at user scope | **not created** — see above |
| `.gitignore`, `requirements.txt` | both present | unchanged |
| `1-scoring-criteria.md` (ICP rubric) | Setup score: 9 direction + 3 quality criteria, grades A/B/C/D | **create** as pointer doc |
| `2-maildoso-api-docs.md` | `docs/MT5_MCP.md` | **create** thin pointer + the four gates as a pre-flight checklist |
| `3-copy-frameworks.md` (tiered copy) | Trade-plan output contract, from the institutional skill | **create** |
| `4-sequencer-connect.md` (export to sender) | Scored setup → sized MT5 order | **create** |
| `lib/apollo_client.py` (read API) | `src/services/mt5_link.py` — read-only by design | exists |
| `lib/maildoso_client.py` (write API) | `src/services/mt5_trade.py` — behind four gates | exists |
| `outputs/*.csv` | `open_positions.json`, `setup_ranker_score_history.json`, `precomputed` | exists |
| `skills/` ×5 | the runbook's 10 steps, grouped into 5 | **create** |
| `agents/` ×4 | setup scoring, risk audit, plan writing, trade review | **create** |
| `commands/run-campaign` | `commands/run-session.md` | **create** |
| `hooks/PostToolUse.sh` | audit log of every MT5 tool call | **create** |
| `settings.json` | shared settings; `settings.local.json` stays personal | **create** |

---

## Task 1: Remove the cold-email kit, and RED the signal kit

**Files:**
- Delete: `ai-cold-email-kit/` (25 tracked files + `__pycache__`), `tests/test_cold_email_kit_structure.py`, `tests/test_cold_email_clients.py`, `docs/plans/2026-08-18-ai-cold-email-kit.md`
- Modify: `.dockerignore` — remove the `ai-cold-email-kit/` block added at line 27
- Create: `tests/test_signal_kit_structure.py`
- Modify: `VERSION` → `1.10.19` via `deploy/sync_version.py`

**Interfaces:**
- Produces: `CLAUDE_DIR`, and the lists `SKILLS`, `AGENTS`, `REFERENCE_DOCS` that Tasks 2–5 are validated against.

- [ ] **Step 1: Confirm nothing was committed, then delete**

```bash
cd /c/x/Dashboard-Pro
git log --oneline -1                 # expect the pre-existing HEAD, no kit commit
git rm -r --cached ai-cold-email-kit 2>/dev/null || true
rm -rf ai-cold-email-kit
rm -f tests/test_cold_email_kit_structure.py tests/test_cold_email_clients.py
rm -f docs/plans/2026-08-18-ai-cold-email-kit.md
```

- [ ] **Step 2: Revert the `.dockerignore` block**

Delete the 6 lines added after `archive/` (the comment block plus `ai-cold-email-kit/`). Leave the rest of the file untouched — `deploy/` and its `!deploy/railway/` negations are load-bearing.

- [ ] **Step 3: Confirm the repo is back to its prior shape**

```bash
git status --short
```
Expected: no `ai-cold-email-kit/`, no cold-email tests, `.dockerignore` unmodified. The pre-existing modifications from earlier today (`deploy/mt5_watchdog.py`, `src/pages_lib/setup_ranker.py`, `.foglamp/*`, `tests/test_mt5_watchdog.py`, `tests/test_setup_ranker_staleness.py`) **stay** — they are a separate, verified piece of work.

- [ ] **Step 4: Write the failing structural test (RED)**

```python
"""Guards the .claude/ signal-kit layout.

Markdown and JSON have no import to break, so nothing else would notice a skill
whose frontmatter stopped parsing or an agent renamed by half a word. For files
that carry no executable code this test *is* the red-green cycle: it fails
before the file exists and passes after.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

CLAUDE_DIR = Path(__file__).parent.parent / ".claude"

SKILLS = ["scan-shortlist", "confirm-bias", "size-risk",
          "run-checklist", "execute-and-log"]
AGENTS = ["setup-scorer", "risk-auditor", "trade-plan-writer", "trade-reviewer"]
REFERENCE_DOCS = ["1-scoring-criteria.md", "2-mt5-tooling.md",
                  "3-trade-plan-framework.md", "4-execution-handoff.md"]


def _frontmatter(path: Path) -> dict:
    """The frontmatter block as a dict. Hand-parsed: the repo has no YAML
    dependency and the frontmatter is flat `key: value` only."""
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n"), f"{path.name} has no frontmatter"
    out = {}
    for line in text.split("---\n")[1].splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            out[key.strip()] = value.strip()
    return out


@pytest.mark.parametrize("name", SKILLS)
def test_every_pipeline_skill_is_loadable(name):
    path = CLAUDE_DIR / "skills" / name / "SKILL.md"
    assert path.is_file(), f"missing {path}"
    fm = _frontmatter(path)
    assert fm.get("name") == name
    assert fm.get("description", "").startswith("Use when"), \
        "a description the model cannot match on is a skill that never fires"


@pytest.mark.parametrize("name", AGENTS)
def test_every_agent_is_loadable(name):
    path = CLAUDE_DIR / "agents" / f"{name}.md"
    assert path.is_file(), f"missing {path}"
    fm = _frontmatter(path)
    assert fm.get("name") == name
    assert fm.get("description")


@pytest.mark.parametrize("doc", REFERENCE_DOCS)
def test_reference_docs_exist(doc):
    assert (CLAUDE_DIR / "reference" / doc).is_file()


def test_the_existing_process_skills_survive():
    # The 12 process skills predate this plan; adding pipeline skills beside
    # them must not disturb them.
    for name in ("writing-plans", "test-driven-development",
                 "verification-before-completion",
                 "experienced-institutional-fx-trade"):
        assert (CLAUDE_DIR / "skills" / name / "SKILL.md").is_file()


def test_settings_json_is_valid_and_does_not_pre_authorise_trading():
    """The four gates exist so a trade is never one careless allow-rule away.

    An `allow` entry for a trade tool would silently remove the confirmation
    step that `mt5_trade` is built around, so this asserts on the whole allow
    list rather than trusting review to catch it.
    """
    settings = json.loads((CLAUDE_DIR / "settings.json").read_text(encoding="utf-8"))
    allow = settings.get("permissions", {}).get("allow", [])
    for tool in ("open_position", "close_position", "modify_position",
                 "place_pending_order", "cancel_pending_order"):
        assert not any(tool in rule for rule in allow), \
            f"{tool} must never be pre-authorised"


def test_audit_hook_is_shell_and_wired():
    hook = CLAUDE_DIR / "hooks" / "PostToolUse.sh"
    assert hook.is_file()
    assert hook.read_text(encoding="utf-8").startswith("#!/")
    settings = json.loads((CLAUDE_DIR / "settings.json").read_text(encoding="utf-8"))
    wired = json.dumps(settings.get("hooks", {}))
    assert "PostToolUse.sh" in wired, "a hook on disk that nothing calls is not a hook"


def test_run_session_command_exists():
    assert (CLAUDE_DIR / "commands" / "run-session.md").is_file()
```

- [ ] **Step 5: Run it and watch it fail for the right reason**

Run: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/test_signal_kit_structure.py -q --no-cov`
Expected: **fails on missing files**, and `test_the_existing_process_skills_survive` **passes** — if that one fails too, the paths are wrong and the whole test is measuring nothing. Read the output before continuing.

- [ ] **Step 6: Bump the version for this plan**

```bash
cd /c/x/Dashboard-Pro && .venv/Scripts/python.exe deploy/sync_version.py 1.10.19
```

- [ ] **Step 7: Save this plan to `docs/plans/2026-08-18-signal-kit.md`**

---

## Task 2: `.claude/reference/` — the four docs

**Files:** Create `1-scoring-criteria.md`, `2-mt5-tooling.md`, `3-trade-plan-framework.md`, `4-execution-handoff.md` under `.claude/reference/`

**Interfaces:**
- Produces the vocabulary Tasks 3–4 depend on: the 12 criteria names, the A/B/C/D grade bands, the five trade-plan headers, and the pre-flight gate checklist.

- [ ] **Step 1: `1-scoring-criteria.md`**

Points at `src/core/signals.py` as the source; restates only the **names** so an agent can talk about them, never the thresholds (those live in code and would rot here).

- **9 direction criteria:** Weekly EMA, Weekly RSI, Weekly Structure, Daily Trend, Daily Structure, Daily MACD, Daily 200MA, 4H Structure, Currency Strength
- **3 quality criteria** (`_QUALITY_CRITERIA`): ATR Volatile, 4H Zone, Spread/ATR
- **Grades:** A >= 80%, B 60–79%, C 40–59%, D < 40%
- The rule that matters: **direction score and quality score are separate.** A Grade A that fails the quality gate is a good read on an untradeable instrument — the spread will eat the edge. Say so rather than ranking it first.

- [ ] **Step 2: `2-mt5-tooling.md`**

One paragraph plus a link to `docs/MT5_MCP.md`. **Do not restate the tool table** — that file is the source. What this adds is the pre-flight checklist an agent runs before proposing an order, phrased as the four gates:

```
1. MT5_ALLOW_TRADING=1 set on the server        (else read-only; say so, do not retry)
2. confirm=true passed explicitly                (never defaulted, never inferred from "yes")
3. volume <= MT5_MAX_VOLUME                      (currently 0.5)
4. Algo Trading enabled in the terminal          (else MT5 blocks it silently)
Then order_check dry-runs it; a broker-rejected request raises instead of sending.
```

Plus the read/write split: `mt5_link.py` exposes no order call by design; `mt5_trade.py` is the only writer. An agent that wants a read uses the read tools, always.

- [ ] **Step 3: `3-trade-plan-framework.md`**

The output contract for any trade thesis, taken from `.claude/skills/experienced-institutional-fx-trade/SKILL.md` — five headers, in order:

1. Macro View & Big Picture
2. Asset Analysis & Trade Thesis (state conviction: High/Medium/Low)
3. Trade Setup — entry, stop (ATR-based), target (min 1:2 R:R), position size, **counter-indicators that would invalidate it**
4. Recommended Execution Strategy — spread in pips, order type per the spread rule (>0.5 pips → limit inside the spread; <1bp → marketable limit)
5. Execution Cadence & Contingency — child orders, and the conditions that pause or abort

Hard rules carried over: risk-first, max 0.5–1% per trade, minimum 1:2 R:R, benchmark against arrival price, and **state confidence and counter-indicators every time**. A plan without an invalidation level is not a plan.

- [ ] **Step 4: `4-execution-handoff.md`**

How a scored setup becomes an order: read the grade and quality gate → size from `src/services/risk_service.py` `RiskService.compute()` (do not re-derive the maths — that function is the source) → check correlated exposure via `src/services/exposure.py` → run the four gates → place through the MT5 trade tools with `confirm=true` supplied by the owner, never by an agent.

Names the stores the outcome lands in: `open_positions.json` (the book), `setup_ranker_score_history.json` (the score at the time of the decision, for later scoring of the scorer).

---

## Task 3: `.claude/skills/` — the runbook as five skills

**Files:** Create `SKILL.md` under `scan-shortlist/`, `confirm-bias/`, `size-risk/`, `run-checklist/`, `execute-and-log/`

**Interfaces:** Each skill's frontmatter `name` matches its directory and its `description` starts `Use when`. Consumes the vocabulary from Task 2.

- [ ] **Step 1: Write all five to this shape**

Frontmatter, then: which runbook steps it covers, which pages/services it reads, what it produces, and its **stop condition**. One representative example, `size-risk/SKILL.md`:

```markdown
---
name: size-risk
description: Use when a setup has passed bias confirmation and needs a position size - computes lot size from the account balance, the ATR stop and the risk budget.
---

# Size Risk

Runbook step 7. Sizing is arithmetic, not judgement, and the arithmetic already
exists: `RiskService.compute(account_balance, risk_pct, pip_value, sl_pips,
tp1_pips, tp2_pips)` in `src/services/risk_service.py`. Call it. Never
re-derive lot size in prose - a second implementation of this maths is how a
position ends up 2.6x too large.

## Inputs
- Live balance from `account_state.get_balance()` - **check its age first**.
  A stale balance sizes every trade wrong, silently.
- Stop distance in pips from the setup's ATR stop
- Risk % per trade (default 1.00%, never above 2%)

## Steps
1. Read the balance and its `updated_at`. If older than 15 minutes, stop and
   say so - see `logs/mt5_sync.log`.
2. Compute pip value for the instrument from `src/instruments/registry.py`.
   Never hardcode one; gold and JPY crosses are not FX majors.
3. Call `RiskService.compute()`.
4. Report `actual_risk`, not just the target - lot rounding moves real money
   on wide-stop metals trades.
5. Check correlated exposure with `src/services/exposure.py`. A second position
   on the same currency leg is not a second trade, it is one bigger one.

## Stop
Report lot size, actual risk in account currency, and the correlated exposure
this would add. Do not place anything - that is `execute-and-log`, and it needs
the owner.
```

The other four, same shape:
- **`scan-shortlist`** — runbook steps 1–2. Macro backdrop and risk regime, then the ranked shortlist. Reads the Setup Ranker's stored scores rather than rescanning. Stops with a ranked list and the tier split.
- **`confirm-bias`** — steps 3–5. Second opinion, filter the day, then weekly → daily → 4H alignment. Refuses to pass a setup whose timeframes disagree, and says which one dissents.
- **`run-checklist`** — step 8, the GO gate. Every item must pass; a partial pass is a no. This is the skill that says "nothing reaches GO tonight" and means it.
- **`execute-and-log`** — steps 9–10. Requires the owner's explicit `confirm=true`; records the decision and the score at decision time, then the review.

- [ ] **Step 2: Verify GREEN** — `pytest tests/test_signal_kit_structure.py -q --no-cov -k skill`, expect 5 pass plus the process-skills guard.

---

## Task 4: `.claude/agents/` — four subagents

**Files:** Create `setup-scorer.md`, `risk-auditor.md`, `trade-plan-writer.md`, `trade-reviewer.md`

- [ ] **Step 1: Write all four**

Frontmatter `name`, `description`, and `tools` at least privilege. **No agent gets a trade tool** — the four gates are not delegable.

- **`setup-scorer`** — input: one instrument. Scores the 9 direction + 3 quality criteria, returns `{pair, direction, direction_score, quality_score, grade, dissent}`. One instrument per dispatch: with twenty in context the fifth pair inherits the fourth's optimism.
- **`risk-auditor`** — input: a proposed trade plus the open book. Checks correlated exposure (`exposure.py`), margin level, news blackout (`econ_calendar.py`), stop presence, and the 0.5–1% per-trade ceiling. Returns PASS/FAIL per rule with the offending number. Tools: read + MT5 read tools only.
- **`trade-plan-writer`** — input: one confirmed setup. Emits the five headers from `3-trade-plan-framework.md`, with conviction and counter-indicators. Returns `INSUFFICIENT_DATA` rather than inventing a macro catalyst it cannot cite.
- **`trade-reviewer`** — input: one closed trade. Returns exactly one of `thesis-correct` / `thesis-wrong` / `execution-error` / `invalidated-by-news`, plus the one line to carry into the next session. The label distinguishes a bad trade from bad luck — the whole point of reviewing.

- [ ] **Step 2: Verify GREEN** — `-k agent`, expect 4 pass.

---

## Task 5: Command, hook, and shared settings

**Files:** Create `.claude/commands/run-session.md`, `.claude/hooks/PostToolUse.sh`, `.claude/settings.json`

- [ ] **Step 1: `run-session.md`** — the runbook end to end: scan → confirm → size → checklist → **stop and present** → execute only on an explicit yes. Mirrors `docs/Live_Session_Runbook.md`, and says so, so the two are read together.

- [ ] **Step 2: `PostToolUse.sh`** — appends every tool call to `logs/mt5_tool_audit.jsonl`:

```bash
#!/usr/bin/env bash
# Append-only audit of every tool call in a trading session.
#
# Deterministic on purpose: an agent can summarise six calls as "checked the
# book", and the one call that mattered is the one it did not mention. When the
# question later is "what actually reached the broker, at what size", this file
# is the only answer not written by the thing being audited.
#
# logs/ is gitignored and .dockerignored, so this never leaves the machine.
set -euo pipefail

LOG="$(dirname "$0")/../../logs/mt5_tool_audit.jsonl"
mkdir -p "$(dirname "$LOG")"

printf '{"ts":"%s","tool":"%s","input":%s}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "${CLAUDE_TOOL_NAME:-unknown}" \
  "${CLAUDE_TOOL_INPUT:-null}" >> "$LOG"
```

- [ ] **Step 3: `.claude/settings.json`** — shared and committed; `settings.local.json` stays personal and gitignored. Contents: `permissions.allow` for MT5 **read** tools only (`account_info`, `get_positions`, `get_quote`, `get_candles`, `get_history`, `symbol_info`, `mt5_status`) plus `Bash(.venv/Scripts/python.exe -m pytest*)`; `permissions.ask` for every trade tool; `permissions.deny` for `Read(./.streamlit/secrets.toml)`; the `PostToolUse` hook wired as `bash .claude/hooks/PostToolUse.sh` (Git Bash — a `.sh` hook does not run under cmd.exe).

**No `enabledMcpjsonServers` key** — that is for project-scoped `.mcp.json` servers, and this repo deliberately has none.

- [ ] **Step 4: Verify GREEN** — full `tests/test_signal_kit_structure.py`, expect all pass.

---

## Verification

Per `verification-before-completion`: run each, read the output, and quote it. No claim without fresh evidence in the same message.

1. **Signal kit structure:**
   `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/test_signal_kit_structure.py -q --no-cov`
   Expected: all pass.

2. **The cold-email kit is gone, and nothing else went with it:**
   `git status --short && ls ai-cold-email-kit 2>&1`
   Expected: no `ai-cold-email-kit`, no cold-email tests, `.dockerignore` not modified; the earlier watchdog/setup-ranker work still listed.

3. **Full suite, no regression:**
   `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest -q`
   Expected: coverage >= 80%. Known pre-existing failures, unchanged: 2 GARCH, 2 `test_data_backbone_config` (DB resolves to `localhost:5432`). Any fifth failure is ours.

4. **The trade gates still hold** — the one thing this plan could break:
   `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/test_mt5_mcp.py -q --no-cov`
   Expected: pass, including `test_unconfirmed_call_is_refused` (`tests/test_mt5_mcp.py:93`), which is the `confirm=true` gate itself. (There is no `tests/test_mt5_trade.py`; `test_mt5_mcp.py` is where the gates are asserted.)
   Then read `.claude/settings.json` yourself and confirm no trade tool sits in `allow`. The structural test asserts it, but this is the rule whose failure costs real money.

5. **Deploy (once, per the owner's decision):**
   `docker compose build app && docker compose up -d && .venv/Scripts/python.exe deploy/verify_deploy.py`
   Expected: reports 1.10.19, all four containers in sync.

6. **The skills and agents actually load** — a definition that does not load is indistinguishable from one that does not exist. Start a session in `C:\x\Dashboard-Pro` and confirm the 5 pipeline skills appear alongside the 12 process skills, and that the 4 agents are listed.

7. Show the owner the diff. **Never commit.**
