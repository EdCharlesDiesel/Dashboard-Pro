# Macro Event Playbook — give the service a page

**Goal:** Make the event playbook visible. Move the module to where its own
docstring says it belongs, write the page that was assumed to exist, and link it
in the sidebar.

**Architecture:** No new logic. `pages/Event_Playbook.py` is 776 lines of pure
functions with zero Streamlit imports — it is a service that landed in `pages/`.
It moves to `src/services/`, and a thin `pages/event_playbook_tab.py` renders it
following the repo's existing page idiom.

**Tech Stack:** Python 3.14, Streamlit, pytest.

**Spec:** The owner's request, 2026-08-23: "added page Fix i need to se it event
play book".

## Global Constraints

- **Never commit.** The owner reviews and commits.
- `VERSION` reads **1.10.43**, so this plan takes **1.10.44**.
- **No logic changes.** The service is moved verbatim; if the page needs
  something it does not expose, that is a separate change.
- **The page owns no maths.** Rendering only — the same split the rest of
  `pages/` follows.

---

## Context

`pages/Event_Playbook.py` renders **nothing**: 0 widgets, no exceptions, and not
a single `st.` call in 776 lines. Streamlit lists it as a page and draws a blank
screen, which is exactly what the owner saw.

Its docstring names the problem itself:

> `event_playbook_service.py` — pure logic for the Macro Event Playbook page…
> **No Streamlit imports here. Page code lives in pages/Event_Playbook.py.**

So the module believes it is the service and expects a page beside it. The page
was never written. Linking it in the nav would only route to a blank screen —
this is not the missing-`NavEntry` case that hid the platinum tab.

**The service works.** Exercised directly it returns 7 events for the week, with
`session_fit` classifying each against the 17:00–20:00 SAST window,
`score_surprise` producing a bias, and `scenario_ladder` producing five rungs.
There is real output to render.

**Where it belongs.** `src/services/` holds this kind of module, and
`pyproject.toml` measures it — while `pages/` is omitted from coverage entirely.
Left in `pages/`, 776 lines of trade-planning maths are invisible to the
coverage gate; moved, they are counted.

---

## Task 1: Move the service

**Files:** `pages/Event_Playbook.py` → `src/services/event_playbook_service.py`

- [ ] **Step 1:** `git mv`, verbatim.
- [ ] **Step 2:** Correct the docstring's self-reference — it currently points at
      the path it is moving away from.
- [ ] **Step 3:** Confirm it imports as `src.services.event_playbook_service`.

---

## Task 2: The page

**Files:** Create `pages/event_playbook_tab.py`

Three things the service can already answer, in the order a trader needs them:

- [ ] **Step 1: This week's events** — name, currency, time, tier, and the
      session-fit verdict, since that is what decides RETRACE vs BREAKOUT.
- [ ] **Step 2: Scenario ladder** for a selected event — the five rungs with
      their bias and conviction, so the reaction is decided *before* the print.
- [ ] **Step 3: Both plans** — `retrace_plan` and `breakout_plan` side by side
      with entry band, stop, targets and R multiples, driven by inputs for the
      impulse/range levels and M15 ATR.
- [ ] **Step 4:** Follow the page idiom — `set_page_config`, `BloombergTheme`,
      `render_sidebar_nav`, and no maths in the page.

---

## Task 3: Link it

**Files:** Modify `src/pages_lib/navigation.py`

- [ ] **Step 1:** Add a `NavEntry` under **📅 WEEKEND** — it is weekly
      preparation, not a mid-session tool.
- [ ] **Step 2:** `tests/test_navigation_covers_every_page.py` must go green;
      it is currently failing on exactly this page, which is how it was found.

---

## Verification

1. **The nav guard passes** — it named this page unprompted and must stop.
2. **The page renders** under AppTest with **no exception and a non-zero widget
   count** — the widget count is the check that matters, since the blank page it
   replaces also raised nothing.
3. **The service imports** from its new home.
4. **Full suite**, coverage ≥ 80% — it should *rise*, since 776 measured lines
   move into scope.
5. **Deployed and visible** in the running container, not just on disk.
6. Show the owner the diff. **Never commit.**
