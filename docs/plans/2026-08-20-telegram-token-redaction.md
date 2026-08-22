# Telegram: stop leaking the token, survive a network blip

> **For agentic workers:** Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a failed Telegram send report *why* without ever printing the bot token, and stop a momentary network fault from silently destroying an entry alert.

**Architecture:** One canonical sender does both jobs. `src/core/secrets.send_telegram_message` gains token redaction, Telegram's own error description, and a bounded retry; the two other senders stop having their own failure paths. No new module and no new dependency.

**Tech Stack:** Python 3.14, `requests`, pytest.

**Spec:** The owner's request, 2026-08-20 — items 1 and 2 of three, after the token leaked twice in one session and a DNS blip ate a send.

## Global Constraints

- **Never commit.** The owner reviews and commits.
- `VERSION` reads **1.10.28**, so this plan takes **1.10.29**. **The backfill plan already saved at `docs/plans/2026-08-20-plans-in-docs.md` also claims 1.10.29 and must be amended to 1.10.30** — two plans cannot hold the same number, and `tests/test_plans_are_recorded.py` now checks the claim.
- Branch `DEV-04/Market-Overview`.
- **No test may send a real Telegram message.** `requests.post` is stubbed everywhere.
- **No test may contain a real token.** Fixtures use an obvious fake such as `123:FAKE`.
- **TDD.** `src/core/secrets.py` is in the coverage omit list, so these tests exist for correctness rather than coverage — a leak is not the kind of bug to leave unpinned.

---

## Context

The token leaked **twice in this session**, and a third path was found by reading. All three senders fail differently and all three expose the same thing, because Telegram puts the token in the URL and `requests` puts the URL in its error string.

| Sender | Failure path | What escapes |
|---|---|---|
| `src/core/secrets.py:178` | `return False, str(exc)` | Returned to callers. `background_scanner` logs it: `logger.warning("[bg-scanner] confluence telegram failed: %s", msg_tg)` |
| `src/core/threat_sentry_hook.py::_send_telegram` | **no `try`/`except` at all** | The exception propagates with the URL in it. It also never calls `raise_for_status`, so a 400 is swallowed and the alert is reported as sent |
| `src/services/evening_sentry.py:247` | `log.error("telegram send failed: %s", e)` | Straight into the log file |

Confirmed rather than assumed — running the sanctioned API inside the scanner:

```
detail : 400 Client Error: Bad Request for url: https://api.telegram.org/bot<REDACTED>/sendMessage
does the raw detail contain the token? True
```

**The second problem is that a blip loses the alert.** The container briefly failed DNS (`Failed to resolve 'api.telegram.org'`) and the send was simply lost — `send_telegram_message` makes one attempt and reports failure. DNS was healthy in all four containers minutes later, so nothing is misconfigured; the fault was transient and the alert was destroyed by it. A confluence signal is rare by design, so losing one to a two-second network fault is losing the thing the whole pipeline exists to produce.

**Retry only what retrying can fix.** A `400 chat not found` will never succeed on attempt two — retrying it just delays the scanner and hammers the API. Connection and timeout errors are the retryable class; HTTP status errors are not.

*Scope note:* `evening_sentry` keeps its own credential source and its own `requests` call. Only its log line is scrubbed — it is a live scheduled module this session has not otherwise touched, and rerouting its credentials is a bigger change than the leak warrants.

---

## Task 1: A sender that cannot leak

**Files:** Modify `src/core/secrets.py` · Create `tests/test_secrets_telegram.py`

**Interfaces:**
- `redact(text: str) -> str` — replaces the configured bot token with `<REDACTED>`. Exported so other modules can scrub their own logs.
- `send_telegram_message(text: str, parse_mode: str | None = None) -> tuple[bool, str]` — `parse_mode` is new, defaulting to today's behaviour (none).

- [ ] **Step 1: Failing tests**

```python
FAKE = "123456:FAKE-TOKEN-abcdef"


class TestNoTokenEverEscapes:
    """The token is in the URL, and requests puts the URL in its error string.

    It leaked twice on 2026-08-20 - once to the screen, once through the
    detail string that background_scanner writes to its log.
    """

    def test_a_network_error_detail_has_no_token(self, monkeypatch):
        monkeypatch.setattr(secrets, "telegram_config",
                            lambda: {"bot_token": FAKE, "chat_id": "1"})
        def boom(*a, **k):
            raise requests.ConnectionError(
                f"Max retries exceeded with url: /bot{FAKE}/sendMessage")
        monkeypatch.setattr(requests, "post", boom)
        ok, detail = secrets.send_telegram_message("hi")
        assert ok is False
        assert FAKE not in detail
        assert "REDACTED" in detail

    def test_an_http_error_detail_has_no_token(self, monkeypatch):
        ...
        assert FAKE not in detail

    def test_it_reports_telegrams_own_description(self, monkeypatch):
        # "chat not found" is what the operator needs; the URL is noise.
        ...
        assert "chat not found" in detail

    def test_redact_is_a_no_op_without_a_token(self, monkeypatch):
        monkeypatch.setattr(secrets, "telegram_config",
                            lambda: {"bot_token": "", "chat_id": ""})
        assert secrets.redact("nothing to hide") == "nothing to hide"

    def test_success_still_returns_true_and_sent(self, monkeypatch):
        ...
        assert (ok, detail) == (True, "sent")

    def test_parse_mode_is_passed_through_only_when_given(self, monkeypatch):
        # Default stays absent: the plain body contains ENTRY_FIRED, whose lone
        # underscore makes Telegram reject the whole message under Markdown.
        ...
        assert "parse_mode" not in sent_payload
        ...
        assert sent_payload["parse_mode"] == "Markdown"
```

- [ ] **Step 2: Run, watch them fail.**
- [ ] **Step 3: Implement.** Add `redact()`. In `send_telegram_message`, catch the response separately: on a non-2xx read `resp.json().get("description")` and return that; on any exception return `redact(str(exc))`. Pass `parse_mode` only when supplied.
- [ ] **Step 4: Green.**

---

## Task 2: Survive a transient fault

**Files:** Modify `src/core/secrets.py` · Test `tests/test_secrets_telegram.py`

**Interfaces:** module constants `TELEGRAM_ATTEMPTS = 3`, `TELEGRAM_BACKOFF_S = 1.0`.

- [ ] **Step 1: Failing tests**

```python
class TestTransientRetry:
    def test_a_connection_error_is_retried_and_can_succeed(self, monkeypatch):
        # The real case: DNS failed for a moment and the alert was destroyed.
        calls = []
        def flaky(*a, **k):
            calls.append(1)
            if len(calls) < 3:
                raise requests.ConnectionError("temporary failure in name resolution")
            return _ok_response()
        ...
        assert ok is True and len(calls) == 3

    def test_it_gives_up_after_the_configured_attempts(self, monkeypatch):
        ...
        assert len(calls) == secrets.TELEGRAM_ATTEMPTS
        assert ok is False and FAKE not in detail

    def test_an_http_400_is_not_retried(self, monkeypatch):
        # "chat not found" will never succeed on attempt two; retrying it only
        # delays the scanner and hammers the API.
        ...
        assert len(calls) == 1

    def test_retries_do_not_take_longer_than_the_budget(self, monkeypatch):
        # The scanner runs on a cycle; alerting must not stall it.
        monkeypatch.setattr(secrets.time, "sleep", lambda s: slept.append(s))
        ...
        assert sum(slept) <= 3.0
```

- [ ] **Step 2: Run, watch them fail.**
- [ ] **Step 3: Implement** — loop `TELEGRAM_ATTEMPTS` times, retrying only `requests.ConnectionError` and `requests.Timeout`, sleeping `TELEGRAM_BACKOFF_S` between attempts. Return on the first success or on any HTTP status error.
- [ ] **Step 4: Green.**

---

## Task 3: Close the other two leak sites

**Files:** Modify `src/core/threat_sentry_hook.py`, `src/services/evening_sentry.py` · Test `tests/test_threat_sentry_hook.py`

- [ ] **Step 1: Failing test** — the hook must delegate rather than post for itself:

```python
def test_the_hook_sends_through_the_canonical_sender(monkeypatch):
    # Its own _send_telegram had no try/except, so a network error propagated
    # with the token in it, and no raise_for_status, so a 400 was silently
    # treated as delivered.
    seen = {}
    monkeypatch.setattr(hook.secrets, "send_telegram_message",
                        lambda text, parse_mode=None: seen.update(
                            text=text, parse_mode=parse_mode) or (True, "sent"))
    hook._send_telegram("*bold*")
    assert seen["parse_mode"] == "Markdown"     # its text is Markdown by design
```

- [ ] **Step 2: Implement** — `_send_telegram` becomes a call to `secrets.send_telegram_message(msg, parse_mode="Markdown")`, logging a redacted failure instead of ignoring it. Keeps the Markdown behaviour its alert text depends on.

- [ ] **Step 3: Scrub `evening_sentry`'s log line only** — `log.error("telegram send failed: %s", secrets.redact(str(e)))`. Its credential source and its own `requests` call are left alone.

- [ ] **Step 4:** Confirm no sender can still leak:
  `grep -rn "str(exc)\|str(e)\|%s\", e" --include=*.py src/ | grep -i telegram` — every hit must go through `redact`.

---

## Verification

Evidence before claims.

1. **Unit tests:** `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/test_secrets_telegram.py tests/test_threat_sentry_hook.py -q --no-cov` — all pass.

2. **The leak is closed, reproduced the same way it was found** — in the scanner container, with the chat ID still wrong so the 400 path runs:
   ```bash
   docker exec dashboard-pro-scanner-1 python -c "
   from src.core.secrets import send_telegram_message, telegram_config
   ok, detail = send_telegram_message('probe')
   print('ok:', ok); print('detail:', detail)
   print('CONTAINS TOKEN:', telegram_config()['bot_token'] in str(detail))"
   ```
   Expected: `CONTAINS TOKEN: False`, and `detail` reading `chat not found` rather than a URL. Before this change the same command printed `True`.

3. **Retry is bounded** — time a forced-failure send; it must return in under ~3 s, not hang the scan cycle.

4. **Full suite:** `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest -q` — coverage ≥ 80%, the 4 known pre-existing failures, no fifth.

5. **Deploy:** bump to 1.10.29, rebuild, `deploy/verify_deploy.py` → 1.10.29, four containers in sync. Then amend `docs/plans/2026-08-20-plans-in-docs.md` to claim **1.10.30**, and save this plan as `docs/plans/2026-08-20-telegram-token-redaction.md` — before implementing, per the rule added earlier today.

6. Show the owner the diff. **Never commit.**
