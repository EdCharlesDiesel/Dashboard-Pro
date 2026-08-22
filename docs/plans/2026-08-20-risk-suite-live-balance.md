# Risk Suite — pull the live balance

> **Reconstructed 2026-08-22** from the session transcript. Approved and
> executed on 2026-08-20 but never written to `docs/plans/` at the time — see
> `2026-08-20-plans-in-docs.md`. The text is the approved plan; checkboxes are
> left unticked because the record of which steps ran lives in the transcript.

**Goal:** Stop the Risk Suite sizing positions against a hardcoded $10,000. Read the balance the MT5 sync already stores, exactly the way the Setup Ranker does, and say so on screen when that balance is stale.

**Architecture:** Mirror the existing Setup Ranker pattern rather than invent a second one — same `account_bal` session key, same live/manual checkbox, same disabled state when no live balance exists. One new shared helper, `open_positions.age_minutes()`, so the staleness check has a single implementation instead of a second copy.

**Tech Stack:** Python 3.14, Streamlit, pytest.

**Spec:** The owner's request, 2026-08-20: "fix risk-suite to pull the live balance."

## Global Constraints

- **Never commit.** The owner reviews and commits.
- `VERSION` reads **1.10.22**, so this plan takes **1.10.23**.
- Branch `DEV-04/Market-Overview`.
- **Surgical.** `pages/risk-suite.py` is 1600+ lines of working sizing logic. Only the sidebar Account block changes; no sizing maths is touched.
- **Match the existing style** — the Setup Ranker block is the template, down to the session keys.
- TDD applies to `open_positions.age_minutes()` (a pure function in a covered module). The Streamlit wiring has no unit test; it is verified by running the page and reading the number.

---

## Context

`pages/risk-suite.py:646` takes the account balance as a typed-in number:

```python
account_bal = st.number_input("Account Balance ($)",
    value=float(st.session_state.get("account_bal", 10000.0)), ...)
```

It never calls `account_state.get_balance()`. Opened directly in a fresh session it starts at **$10,000** against a live balance of **$3,844.15** — every lot size it suggests is **2.60x too large**. That is the same failure the repo already records: *"a stale $5,182 reading against a real $1,989 produced position sizes 2.6x too large."*

It is masked in normal use: `account_bal` is a **shared** session key, so visiting the Setup Ranker first — which does read the live balance at `setup_ranker.py:287` — leaves the correct number behind and the Risk Suite looks fine. Land on the Risk Suite first and it does not.

**The timestamp trap this must avoid.** `account_state.set_balance` writes `updated_at` as `datetime.now().strftime("%Y-%m-%d %H:%M:%S")` — **naive local time**. The host writes it at UTC+2; the container runs UTC. Computing an age against `datetime.now()` in the container yields roughly **minus two hours** — a balance from the future, reported as fresh forever.

So freshness is taken from `open_positions.saved_at()` instead, which is unambiguous ISO UTC and is written by the *same* `mt5_link.sync()` call in the same run. When there is no book (an MT4 statement import updates the balance alone), no age is claimed. The balance's own format is left alone: `setup_ranker.py:378` slices `updated_at[-8:]` for display, and switching to ISO would render `6+00:00`.

---

## Task 1: `age_minutes()` — one implementation of "how old is the book"

**Files:** Modify `src/services/open_positions.py` · Test `tests/test_open_positions.py`

**Interfaces:** `age_minutes(now: Optional[datetime] = None) -> Optional[float]` — minutes since `saved_at()`, or `None` when nothing is stored or the stamp is unparseable.

- [ ] **Step 1: Failing tests** — none when nothing stored; minutes since the stamp; **a naive stamp is read as UTC, not local** (read as local in a UTC container it comes out negative, i.e. permanently fresh); an unparseable stamp is `None` rather than an exception; a future stamp clamps to zero so clock skew never prints a negative age.
- [ ] **Step 2: Run, watch them fail** (`age_minutes` undefined).
- [ ] **Step 3: Implement** beside `saved_at()`: parse with `datetime.fromisoformat`, attach `timezone.utc` when naive, return `max(0.0, (now - stamp).total_seconds() / 60)`, `None` on `TypeError`/`ValueError`.
- [ ] **Step 4: Green.**

---

## Task 2: The Risk Suite sidebar

**Files:** Modify `pages/risk-suite.py` (the `**💰 Account**` block at ~line 645)

- [ ] **Step 1:** Add the missing import — `from src.services import account_state, open_positions`.

- [ ] **Step 2:** Replace the bare `number_input` with the Setup Ranker pattern. Session keys stay `account_bal` and `risk_pct` so the value keeps carrying across pages:

```python
acct = account_state.get()
live_bal = account_state.get_balance(0.0)
has_live = bool(acct) and live_bal > 0

use_live = st.checkbox("🔗 Use live balance from MT5",
    value=bool(st.session_state.get("rs_use_live_bal", has_live)) and has_live,
    disabled=not has_live, help="...")
st.session_state["rs_use_live_bal"] = use_live

if use_live and has_live:
    account_bal = live_bal
    st.session_state["account_bal"] = live_bal
    age = open_positions.age_minutes()
    if age is not None and age > 15:
        st.error(f"⚠ Balance is **{age:.0f} min old** — the MT5 sync is not running...")
    ...
else:
    account_bal = st.number_input("Account Balance ($)",
        value=float(st.session_state.get("account_bal", 10000.0)), ...)
    st.session_state["account_bal"] = account_bal
```

`rs_use_live_bal` is a page-local key (the Setup Ranker uses `sr_use_live_bal`), so the two pages' checkboxes stay independent while the balance itself stays shared.

- [ ] **Step 3:** Confirm `account_bal` is defined on every path — it feeds `size_position()` at three call sites (`risk-suite.py:786, 893, 1051`). A missing assignment would be a `NameError` on page load.

- [ ] **Step 4:** `python -m py_compile pages/risk-suite.py`.

---

## Verification

1. **Unit tests:** `pytest tests/test_open_positions.py -q --no-cov` — all pass, including the naive-stamp and future-stamp cases.
2. **The number the page will actually use**, before touching the browser: live balance and a small **positive** age — **not** a negative number, which would mean the timezone trap is still live.
3. **The page renders and shows the live figure** — pages are outside coverage, so this is the proof. Expected: checkbox ticked, `BAL $3,844.15 live`, and no `$10,000` anywhere.
4. **The 2.6x bug is actually gone** — open the Risk Suite *directly in a fresh browser session*, without visiting the Setup Ranker first. That ordering is what hid the bug, so it is the only ordering that proves the fix.
5. **Full suite:** coverage ≥ 80%, the known pre-existing failures, no new one.
6. **Deploy:** 1.10.23, four containers in sync.
7. Show the owner the diff. **Never commit.**

## What actually happened

Executed as planned; 35 tests in `test_open_positions.py` passed and the full suite reached 1720.

The timezone trap was visible in the live data: the balance stamp read `19:45:56` (local) while UTC was `17:46`, so using it for an age would have given ≈ −2 hours. The book's ISO UTC stamp gave `+2.6 min`.

Verified in a fresh tab navigated straight to `/risk-suite`, never touching the Setup Ranker: the checkbox ticked, `BAL $3,844.15 live · MT5 terminal`, `mentions_10000: false` across the whole sidebar, and the Account Risk tab reading `EUR/USD · risk 1.00% of $3,844` → **$38.44 TOTAL $ AT RISK, 0.04 lots**. Under the old default the same row would have read $100.00 and 0.10 lots.
