# Entry alerts to Telegram

> **Reconstructed 2026-08-22** from the session transcript. Approved and
> executed on 2026-08-20 but never written to `docs/plans/` at the time — see
> `2026-08-20-plans-in-docs.md`. The text is the approved plan; checkboxes are
> left unticked because the record of which steps ran lives in the transcript.

**Goal:** Send the TRIPLE-CONFLUENCE entry signal — the alert that means *enter buy/sell now* — to Telegram as well as email, and stop it being suppressed entirely when email is unconfigured.

**Architecture:** No new alerting machinery. `src/core/secrets.send_telegram_message()` is already the canonical sender and the credentials are already configured; the work is routing. One new formatter beside the existing email builder, one restructure of the scanner's dispatch so the two channels are independent, and one fix to `write_env` so the container can be given the token at all.

**Tech Stack:** Python 3.14, Docker Compose, pytest.

**Spec:** The owner's request, 2026-08-20, with two decisions: the confluence entry signal only, and credentials via a `.env` that `sync_version.py` stops truncating.

## Global Constraints

- **Never commit.** The owner reviews and commits.
- `VERSION` reads **1.10.27**, so this plan takes **1.10.28**.
- Branch `DEV-04/Market-Overview`.
- **I do not handle the token.** The plumbing reads `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` from the environment; the owner puts the values in the gitignored `.env` themselves. No credential is read out of `secrets.toml`, printed, or written into a tracked file.
- **No test may send a Telegram message.**
- **TDD** for `write_env` and the new formatter — both pure and inside `--cov=src`.

---

## Context

Everything needed already exists, in the wrong arrangement:

- **The sender exists.** `src/core/secrets.py:159` `send_telegram_message(text) -> (ok, detail)`, reading `[telegram] bot_token/chat_id` with env fallback. Both values are already present on the host — verified by presence, not by reading them.
- **The message exists.** `confluence_alert.build_email()` already returns `(html, plain)`, and the `plain` half carries pair, direction, ENTRY FIRED/IN ZONE, entry/stop/tp1/tp2 and the three-leg rationale.
- **The signal exists.** `background_scanner` already computes confluences every cycle.

Three things block it:

1. **The alert is gated on email.** `background_scanner.py:255` reads `if confluences and alert_service.email_configured():` — with no email configured there is no alert at all, whatever Telegram is doing.
2. **The container cannot see the credentials.** `.streamlit/secrets.toml` is excluded by `.dockerignore:3` **by design**, and compose passes `GMAIL_*` but no `TELEGRAM_*`.
3. **`.env` is destroyed on every version bump.** `deploy/sync_version.py:70` `write_env()` opens the file with `"w"` and writes only the header plus `APP_VERSION` — so any `TELEGRAM_*` line vanishes the next time a plan bumps the version, which happened six times in one day. That is a silent data-loss bug in its own right.

**Two details that would have caused bugs:**

- `send_telegram_message` posts **without `parse_mode`**, so the message is plain text. That matters: `fib_status` is `ENTRY_FIRED`, and under Markdown a lone `_` starts italics and an unmatched one makes Telegram reject the whole message with a 400. Because there is no parse mode, the existing `plain` body can be reused verbatim with no escaping.
- `NotifyCache.filter_new()` "returns only keys not seen before, **and records them as seen**" — so it must be called exactly once, after a send succeeds. Calling it to *test* freshness would mark the alert delivered before it was.

*Noted, not changed:* there are three separate Telegram senders — `secrets.py` (canonical), `threat_sentry_hook.py` and `evening_sentry.py`. Consolidating them is a worthwhile separate change.

---

## Task 1: Stop `write_env` truncating `.env`

**Files:** Modify `deploy/sync_version.py` · Create `tests/test_sync_version.py`

- [ ] **Step 1: Failing tests** — it keeps unrelated keys; it does not duplicate `APP_VERSION`; it creates the file when absent; a missing `APP_VERSION` line is added; repeated bumps are stable; `env_version` still reads it back.
- [ ] **Step 2: Run, watch them fail** (the current version wipes the file).
- [ ] **Step 3: Implement** — read existing lines, replace an `APP_VERSION=` line in place or append one, keep every other line.
- [ ] **Step 4: Green**, then confirm the real `.env` lost nothing.

---

## Task 2: `build_telegram()`

**Files:** Modify `src/services/confluence_alert.py` · Test `tests/test_confluence_alert.py`

- [ ] **Step 1: Failing tests** — it leads with the subject so the push preview is readable; it carries the levels a trader needs; it names all three agreeing legs; it stays inside Telegram's 4096-char limit and marks truncation; a short alert is not marked truncated; no items is an empty string.
- [ ] **Step 2: Run, watch them fail.**
- [ ] **Step 3: Implement** — `subject_for(items)` plus the `plain` half of `build_email(items)`, truncated at 4096. No new formatting.
- [ ] **Step 4: Green.**

---

## Task 3: Two independent channels in the scanner

**Files:** Modify `src/services/background_scanner.py` · Test `tests/test_background_scanner.py`

- [ ] **Step 1: Failing tests** — Telegram is sent even when email is unconfigured; delivery by Telegram alone still dedupes; nothing is marked seen when every channel fails.
- [ ] **Step 2: Run, watch them fail.**
- [ ] **Step 3: Implement.** Compute `fresh` once; attempt email only when configured; attempt Telegram unconditionally; call `cache.filter_new(...)` once, only if **either** channel returned ok. Log each channel's failure separately. Keep the block inside its existing `try/except`.
- [ ] **Step 4: Green.**

---

## Task 4: Give the container the credentials

**Files:** Modify `docker-compose.yml` (the `scanner` service)

- [ ] **Step 1:** Add `TELEGRAM_BOT_TOKEN: ${TELEGRAM_BOT_TOKEN:-}` and `TELEGRAM_CHAT_ID: ${TELEGRAM_CHAT_ID:-}` — interpolated from the gitignored `.env`, never written into the tracked file. Empty is safe.
- [ ] **Step 2:** Do **not** add the values. The owner appends them to `.env` themselves; Task 1 is what makes that stick across the next bump.
- [ ] **Step 3:** `docker compose config --quiet`.

---

## Verification

1. Unit tests across the three affected files.
2. **`.env` survives a bump** — append a probe line, run a real `sync_version.py`, confirm it is still there.
3. **The message the owner would receive** — printed, not sent.
4. **The container can see the variables** — booleans only, never the values.
5. **One real end-to-end send, on the owner's say-so.** A live Telegram message is outward-facing, so it is not sent without asking.
6. **Full suite:** coverage ≥ 80%.
7. **Deploy:** 1.10.28, four containers in sync.
8. Show the owner the diff. **Never commit.**

## What actually happened

All three bugs confirmed and fixed; 17 new tests. `.env` survived a real bump with a probe line intact.

**The scanner change was implemented before its tests, inverting TDD.** Rather than claim a cycle that had not been run, the file was backed up, reverted to HEAD, and the tests run against the old code: **4 of 5 failed**. The fifth passed vacuously, because under the old code nothing was ever sent to mark.

The live send was attempted only after the owner asked twice. It failed `400 chat not found` — the stored `chat_id` was a second copy of the bot token — and, worse, **the token leaked into the transcript**, because `requests` embeds the request URL in its error string and Telegram puts the token in the URL. That defect went straight into `logger.warning(...)` on any failed alert, and became plan 1.10.29.
