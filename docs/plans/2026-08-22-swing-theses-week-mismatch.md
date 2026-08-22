# Fix Week Ahead's thesis save — wrong `week_start`, not a missing submit button

Version on creation of this plan: **1.10.36** (VERSION currently reads 1.10.35).

## Context — what the bug actually is (diagnosed live, not assumed)

The user reported the "💾 Save thesis" button on `pages/week_ahead_tab.py` as
silently doing nothing, and a Streamlit "Missing Submit Button" warning seen
in-session pointed at the form itself. That diagnosis was wrong. Reading
`pages/week_ahead_tab.py:277-297`, the form is built correctly — `st.form`
wraps a real `st.form_submit_button`, and the submit branch calls
`sp.save_thesis(...)` with proper validation. Live-testing it (a genuine
DOM-dispatched click, not a stale element reference) **does** save a row.

The real bug, confirmed by two live test rows sitting in `swing_theses` right
now:

| id | instrument | week_start | note |
|---|---|---|---|
| 1 | EUR/USD | 2026-08-24 | hand-inserted earlier this session, targeting the week the page displayed |
| 2 | EUR/USD | 2026-08-17 | saved *by the actual app* just now, while the page was showing "WEEK OF 24 Aug – 28 Aug 2026" |

`src/services/swing_playbook_service.py::week_start()` computes **today's**
Monday unconditionally:

```python
def week_start(now: Optional[datetime] = None):
    now = now or datetime.now(timezone.utc)
    return (now - pd.Timedelta(days=now.weekday())).date()
```

`save_thesis`/`load_thesis` call this with no override. But Week Ahead has a
"Which week" toggle (`nextweek` / `thisweek`, defaulting to **`nextweek`** —
the page's whole framing is "Sunday pre-flight" for the week ahead) and
displays that chosen week's dates in the header. The thesis form never
threads that choice through — it always saves against the *current* calendar
week's Monday, regardless of which week is on screen. On the page's own
default view (`nextweek`), every save silently misfiles under last week's
Monday instead of the week actually being planned for.

`pages/swing_playbook_tab.py` calls the same two functions with no week
argument and has no "which week" concept of its own — its usage is correct
today and must stay unchanged.

## Decision

Add an optional `week_start` override to `save_thesis`/`load_thesis`,
defaulting to today's Monday (current behaviour — Swing Playbook keeps
working exactly as-is). `week_ahead_tab.py` computes the real date for its
selected `week_key` and passes it explicitly. The date math already exists,
duplicated, inside `_week_label()` — extract it once rather than duplicate a
third time.

No schema change: `trade_repository`'s `save_swing_thesis`/`load_swing_thesis`
already take `week_start` as an explicit argument (`swing_playbook_service.py`
line 255/263) — only the service-level wrapper functions hardcode "today."

## Global constraints

- Never commit. Show the diff.
- Every completed task bumps the patch: read `VERSION`, add one, write it back
  via `python deploy/sync_version.py <next>`.
- Tests first (`test-driven-development`).
- `swing_playbook_tab.py`'s existing calls must keep behaving identically —
  no signature change that forces it to pass anything.

## Starting state (measured)

- `VERSION` → `1.10.35`
- `swing_theses` has 2 rows: `(EUR/USD, 2026-08-24, Bullish, ...)` [manual] and
  `(EUR/USD, 2026-08-17, Neutral, "Test invalidation text for bug diagnosis")`
  [live app, proving the bug].
- No existing test file for `swing_playbook_service`.
- Today (container clock): 2026-08-22 (Saturday) → current-week Monday =
  2026-08-17; `nextweek` Monday = 2026-08-24.

---

## Task 1 — thread the displayed week into the thesis save/load

This task takes **1.10.37**.

**Steps**
- [x] Write `tests/test_swing_playbook_service.py` first, watch it fail —
  confirmed: `TypeError: save_thesis() got an unexpected keyword argument`.
- [x] In `src/services/swing_playbook_service.py`: added
  `week_start_override: Optional[date] = None` to `save_thesis` and
  `load_thesis` (named `_override`, not `week_start`, because that name
  would shadow the module-level `week_start()` function inside the function
  body and break the `week_start_override or week_start()` fallback call —
  discovered during implementation, plan text above said `week_start`).
  No behaviour change when omitted — Swing Playbook's call sites untouched.
- [x] In `pages/week_ahead_tab.py`: extracted `_week_monday(week_key) -> date`
  out of `_week_label`; proved byte-identical label output for both
  `nextweek`/`thisweek`. Both `sp.save_thesis(...)` and `sp.load_thesis(...)`
  now pass `week_start_override=t_week`.
- [x] Verified — not via live browser clicking (Streamlit's WebSocket-based
  form submission proved flaky to drive from outside, consistent with why
  this repo's own suite uses `AppTest` rather than browser E2E everywhere
  else). Used `AppTest` instead: selected Bias=Long, set an invalidation
  string, clicked the real submit button, reran. Result: `st.success`,
  and the DB row confirms `week_start=2026-08-24` (the displayed "next
  week"), not `2026-08-17` (today's Monday) — the exact mismatch this task
  fixes. Unit tests: 4/4 passing.
- [x] `python deploy/sync_version.py 1.10.37`

Out of scope, noted but not touched: the two rows already sitting in
`swing_theses` from this session's diagnosis are test/manual data, not
production content — leave them; the owner can clear them if they don't want
them.
