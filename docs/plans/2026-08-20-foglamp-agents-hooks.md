# .foglamp scan — add the session subagents and the audit hook

> **Reconstructed 2026-08-22** from the session transcript. Approved and
> executed on 2026-08-20 but never written to `docs/plans/` at the time — see
> `2026-08-20-plans-in-docs.md`. The text is the approved plan; checkboxes are
> left unticked because the record of which steps ran lives in the transcript.

**Goal:** Put the signal kit's four subagents and its PostToolUse audit hook on the architecture map, so `.foglamp/scan.json` describes the repo as it now is.

**Architecture:** Two nodes and three edges added to `graph`, plus `stats.agents` corrected 4 → 5. Both new nodes hang off the existing `claude-code` node, because that is what actually drives them. Then regenerate `code.json` and `scan.html` with the documented two-step. No new node carries a `sourceRef`.

**Tech Stack:** JSON, `.foglamp/introspect.py`, `.foglamp/render.py`.

**Spec:** `.claude/CLAUDE.md` — "The architecture scan (`.foglamp/`) — keep it current".

## Global Constraints

- **Never commit.** The owner reviews and commits.
- `VERSION` reads **1.10.20**, so this plan takes **1.10.21**.
- **`scan.json` is a published contract. Make the edit fit the caps; never relax them.** Measured: **52/60 nodes, 83/120 edges**. This adds 2 and 3 → **54/60, 86/120**.
- `KINDS` is fixed at 8 (`entry, cron, agent, model, tool, service, store, external` — **there is no `hook`**), `EDGE_KINDS` is `calls, reads, writes, triggers`, `label ≤28`, `sub ≤40`, `group ≤24`, `edge.label ≤24`.
- **Never hand-number the rebuild order.** Phases are derived from the graph at render time. `stats` is different — an authored, rendered header field — and must be corrected by hand to stay truthful.

---

## Context

Four findings from reading `render.py` and `introspect.py` shape the design:

1. **There is no `hook` kind and the vocabulary cannot be extended.** The hook is modelled as `service` — the 20 existing `service` nodes are internal modules, which is what a local script that fires automatically is.
2. **No new node may carry a `sourceRef`.** `introspect.py:88` runs `ast.parse` on any file ref and returns `None` on `SyntaxError`; the caller appends it to `missing` and **exits 1**. A `.md` or `.sh` ref would break the toolchain. Existing practice matches: `claude-code`, `mt5-read-tools` and `mt5-trade-tools` all carry a `domain` and no `sourceRef`.
3. **`stats` is rendered, so it must be corrected.** `template.html:297` reads `DATA.stats`; `render.py` does not validate it. `stats.agents` is `4` and matches the 4 `agent` nodes, so it becomes `5`.
4. **`code.json` and `scan.html` are generated but tracked.** `.gitignore:43-44` lists them, yet `git ls-files` shows both — committed before the ignore rule. Regenerating churns them in the diff. Pre-existing; mention, do not fix.

Scope note: the 5 skills, the slash command and the 4 reference docs are deliberately **not** added — the graph reads correctly without them, since the subagents' caller `claude-code` is already a node.

---

## Task 1: Add the nodes, edges and the stats correction

**Files:** Modify `.foglamp/scan.json` (`graph.nodes` +2, `graph.edges` +3, `stats.agents` 4→5); regenerate `.foglamp/code.json`, `.foglamp/scan.html`; `VERSION` → `1.10.21`

- [ ] **Step 1: Record the pre-change measurement** — `52 nodes 83 edges`, and `render.py --check` must already exit 0. If it fails before the edit, stop: the breach is not ours.

- [ ] **Step 2: Append two nodes**

```json
{ "id": "session-subagents", "label": "Session subagents", "kind": "agent",
  "sub": "score, risk, plan, review", "domain": "claude.ai", "group": "AI surfaces",
  "detail": "Four read-only reviewers in .claude/agents, one per session step: setup scoring, risk audit, trade-plan writing, closed-trade review. None gets a trade tool - the gates are not delegable." },
{ "id": "tool-audit-hook", "label": "Tool-call audit hook", "kind": "service",
  "sub": "PostToolUse -> jsonl, append-only", "group": "AI surfaces",
  "detail": "Fires on every tool call in a session and appends it to logs/mt5_tool_audit.jsonl. The only record of what actually reached the broker that the agent did not write itself." }
```

- [ ] **Step 3: Append three edges** — `claude-code --calls--> session-subagents` ("one per session step"), `session-subagents --calls--> mt5-read-tools` ("read-only, no trade"), `claude-code --triggers--> tool-audit-hook` ("every tool call"). Both new nodes connect to the existing graph rather than floating.

- [ ] **Step 4: Correct `stats.agents` 4 → 5.** Leave `models`, `tools`, `integrations` untouched.

- [ ] **Step 5: Regenerate in the documented order** — `introspect.py` first (it exits 1 on a bad `sourceRef`, the cheapest proof the new nodes did not break the toolchain), then `render.py`.

- [ ] **Step 6: Bump** to 1.10.21.

---

## Verification

1. **The contract still validates:** `render.py --check` → exit 0.
2. **The toolchain accepts the new nodes:** `introspect.py` → exit 0.
3. **Counts inside the caps and strings fit:** `54/60 nodes, 86/120 edges`; `stats.agents == agent node count`.
4. **Both new nodes are reachable** — a node with no edge is invisible on the map.
5. **No regression:** full suite, coverage ≥ 80%.
6. **Deploy:** 1.10.21, four containers in sync.
7. Show the owner the diff. **Never commit.**

## What actually happened

`render.py --check` rejected the first attempt: **`node.detail` is capped at 200 chars and the subagents detail was 207**. The full `STR_CAPS` list had not been read — only a truncated view of it. Trimmed to 187 and re-rendered clean: `scan.json OK - 54 nodes, 86 edges`. A `json.dump` round-trip was also tested and rejected before use: it would have rewritten **1160 lines**, because the file uses compact one-line objects that `indent=2` explodes. Text-level insertion was used instead, giving an 11-line diff.
