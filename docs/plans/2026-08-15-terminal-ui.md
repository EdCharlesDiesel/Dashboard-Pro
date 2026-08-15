# Terminal UI: Stop Streamlit Showing Through — Implementation Plan

> **For agentic workers:** Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every surface on the signal pages read as one terminal. The identity is already right — black ground, phosphor green, JetBrains Mono, dense readouts. What breaks it is Streamlit's default chrome punching through: light-grey table headers on a black page, stock success/warning panels, and emoji headings that dilute a house style the rest of the app earns honestly.

**Architecture:** No redesign. `src/ui/theme.py` already owns the palette and injects global CSS; `src/ui/components.py` owns the shared widgets. The work is to (a) give the theme a real table treatment so `st.dataframe` stops being the odd one out, (b) promote the one pattern that already works — the trade card — into a reusable readout component, and (c) delete decoration that does not encode anything. Pages change only by calling the component instead of hand-rolling markup.

**Tech Stack:** Python 3.14 venv (deploy target 3.12), Streamlit, Postgres 18, pytest + pure `ast` for the static guards, Docker Compose.

**Spec:** `.claude/CLAUDE.md` for the repo rules; `src/ui/theme.py` is the single source of truth for colour and type — this plan extends that rule to tables and status panels, which currently bypass it.

## Global Constraints

- Never commit. Make changes only; the repo owner reviews and commits.
- **A plan gets its own bump too.** This plan took **1.10.1** on creation, so
  Task 1 lands on 1.10.2, Task 2 on 1.10.3, and so on.
- **Bump the version on every completed task** (`python deploy/sync_version.py <next>`), and rebuild + `python deploy/verify_deploy.py` before calling a task done — a fix in git is not a fix in production.
- Run tests as `PYTHONIOENCODING=utf-8 python -m pytest`.
- Return complete implementations — no TODO comments, no placeholder code.
- Use type hints on new code.
- Coverage gate is `--cov-fail-under=80`, scoped to `src/` pure logic + DB layer. `src/ui/*` and `src/pages_lib/*` are omitted from coverage, so **these tasks are guarded by static tests and screenshots, not unit coverage.**
- Never remove an `@st.cache_data` decorator.
- No raw hex in a page or `pages_lib` module — every colour comes from `BloombergTheme`.

---

## Measured starting state (2026-08-15, v1.10.1)

Observed on `/todays-trades` at 1280×800, dark theme:

| Surface | Renders as | Problem |
|---|---|---|
| Takeable / Target-before-stop / Exposure tables | `st.dataframe` | Light-grey header row and gridlines. Three white bands across a black page — the loudest break on the screen. |
| Verdict reasons | `st.success` / `st.warning` | Streamlit's stock green and amber panels; neither colour is in the palette. |
| Verdict column | `✅ TAKE` / `⛔ SKIP` emoji | Says what the card above already says, in a different visual language. |
| Section headings | `### ✅ Takeable now`, `### 🎲 Target before stop`, `### 🌍 Net currency exposure` | Emoji as decoration. The app's own `FN`/`PG` codes are the real structural device. |
| Trade card | Hand-rolled markup in `_card` | **This one works.** Direction glyph, coloured left border, monospace levels line. It reads like an order ticket, which is what it is. |

Palette facts worth knowing before touching the theme:

- `BloombergTheme.AMBER = "#00ff41"` — a token named *amber* holding phosphor green. It is the primary accent and used everywhere. Renaming is in Task 4, deliberately last and on its own, because it touches every page.
- `FONT_MONO` is JetBrains Mono; `FONT_UI` (IBM Plex Sans) is declared but barely used. The terminal reads mono-first and should stay that way.

**Design decision, stated once:** the signature is the readout — a left-ruled block with a direction glyph, a dense monospace data line, and a muted provenance line. It is already on the page and already good. Every task below either extends that language to a surface that lacks it, or removes something competing with it. Nothing new is invented.

---

### Task 1: Give the theme a table treatment

The tables carry the most information on the page and are the least designed thing on it.

- [ ] **Step 1: Write the failing guard** — `tests/test_theme_tables.py`, modelled on the static-guard style of `tests/test_ohlc_spine.py`. Assert that `BloombergTheme.css()` (the injected global stylesheet) contains rules targeting `[data-testid="stDataFrame"]` and that those rules reference `cls.BG_PANEL`, `cls.BORDER` and `cls.FONT_MONO`. It will fail: no such rules exist.
- [ ] **Step 2: Run it to verify it fails**, and capture the message.
- [ ] **Step 3: Add the rules to `src/ui/theme.py`.** Style the header row to `BG_HEADER` with `GREY` uppercase labels at the existing caption size, body rows to `BG_PANEL`, gridlines to `BORDER`, and force `FONT_MONO` throughout. Numbers are the content — set them right-aligned and tabular. Keep the rules in the same f-string block as the rest so there is still one stylesheet.
  - Streamlit's grid is **canvas-based** (glide-data-grid): confirm which parts actually respond to CSS before assuming. If the canvas ignores the header styling, say so in the plan notes and fall back to `st.dataframe(column_config=…)` plus a themed container border rather than fighting it.
- [ ] **Step 4: Run the guard; then look.** `docker compose build app && docker compose up -d`, open `/todays-trades`, screenshot at 1280×800 in **both** light and dark. The white bands must be gone in both.
- [ ] **Step 5: Check the other heavy tables** — `/setup-ranker`, `/trend-signals`, `/market-overview` — for anything the new rules break.
- [ ] **Step 6: Bump to 1.10.2, rebuild, `verify_deploy.py`, show the owner the diff. Do not commit.**

---

### Task 2: Promote the trade card to a shared readout

- [ ] **Step 1: Write the failing test** — `tests/test_components_readout.py`. `readout(...)` must return a string containing the direction glyph, the label, and the provenance line, and must contain **no raw hex** (assert `re.search(r"#[0-9a-fA-F]{6}", html)` finds only values present in `BloombergTheme`).
- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement `readout()` in `src/ui/components.py`**, lifted from `todays_trades_page._card`: signature `readout(*, title: str, tone: str, lines: list[str], provenance: str | None = None) -> str`. `tone` maps to `GREEN` / `RED` / `GREY` for long / short / neutral. Returns HTML; the caller does `st.markdown(..., unsafe_allow_html=True)`.
- [ ] **Step 4: Replace `_card` with a call to it**, then replace the `st.success`/`st.warning` verdict panels in `_probabilities` with `readout(tone="green"|"red")`. The verdict reason text is already written — reuse it verbatim, it is unit-tested in `test_position_risk.py::TestVerdict`.
- [ ] **Step 5: Drop the `Verdict` column** from the Target-before-stop table. The readout beneath it says the same thing in the page's own language; two visual grammars for one fact is the thing being fixed.
- [ ] **Step 6: Screenshot, compare against the Task-1 shot**, confirm the page now has exactly one status idiom.
- [ ] **Step 7: Bump to 1.10.3, rebuild, `verify_deploy.py`, show the diff.**

---

### Task 3: Headings that encode something

- [ ] **Step 1: Write the guard** — extend `tests/test_theme_tables.py` (or a sibling) to assert no `st.markdown("### <emoji>` in `src/pages_lib/todays_trades_page.py`. Keep it scoped to this page; a repo-wide emoji ban is a separate argument and not this plan's.
- [ ] **Step 2: Replace the four headings** with the app's existing structural device — the short code already used in the status bar (`FN TDAY`), rendered as an eyebrow: a `GREY` uppercase label, letter-spaced, above a `WHITE` title, with a `BORDER` hairline under it. `TAKEABLE`, `PROBABILITY`, `BLOCKED`, `EXPOSURE`.
  - Sidebar emoji **stay**. There they are wayfinding across 57 entries and are doing a job; in a section heading they were decoration.
- [ ] **Step 3: Screenshot. Then remove one more thing** — whatever now reads as the weakest element. Record what was removed in the notes, so the next pass knows what has been tried.
- [ ] **Step 4: Bump to 1.10.4, rebuild, `verify_deploy.py`, show the diff.**

---

### Task 4: Rename the misleading token — last, and alone

`AMBER` holds `#00ff41`, which is green. Anyone theming this app reads that token and is wrong about the colour.

- [ ] **Step 1: Count the blast radius** — `grep -rn "AMBER" src/ pages/ | wc -l`. If it is large, that is the point: do this in its own commit-sized change with nothing else moving.
- [ ] **Step 2: Rename to `PHOSPHOR` / `PHOSPHOR_DIM`**, keeping `AMBER = PHOSPHOR` as a deprecated alias with a comment giving the removal version, so a missed reference fails loudly at review rather than silently at runtime.
- [ ] **Step 3: Update every call site**, then `grep` for `AMBER` again and confirm only the alias line remains.
- [ ] **Step 4: Full suite, screenshot every page group once** (morning brief, pre-session, session, weekend, research). A palette rename is exactly the change that silently blanks one page.
- [ ] **Step 5: Bump to 1.10.5, rebuild, `verify_deploy.py`, show the diff.**

---

## Out of scope, deliberately

- **Any change to the palette itself.** The phosphor-on-black identity is correct for the subject and already distinctive. This plan removes what breaks it; it does not replace it.
- **The 57-entry sidebar.** It is long, but it is grouped by session and the codes make it scannable. Reworking navigation is its own plan with its own risk.
- **Charts.** Plotly figures already take theme colours; they are not the thing showing through.

## Verification for the whole plan

- [ ] Screenshots of `/todays-trades` at 1280×800 and 375×812, light and dark, before and after.
- [ ] `PYTHONIOENCODING=utf-8 python -m pytest` — full suite green apart from the two known GARCH failures (no `arch` wheel for Python 3.14).
- [ ] `python deploy/verify_deploy.py` after the final bump.
- [ ] No raw hex outside `src/ui/theme.py` in any file this plan touched.

---

Module map: [[Architecture]] · Docs index: [[README]]
