# Two failing tests, a staged secrets file, and production credentialsGoal: Fix the two data_backbone tests that have failed all session, stop any secrets file from being committable, and give the Production branch a deploy path that injects credentials from GitHub Environment secrets instead of from a file in the repo. at deploy

> **For agentic workers:** Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the two `data_backbone` tests that have failed all session, stop any secrets file from being committable, and give the `Production` branch a deploy path that injects credentials from GitHub Environment secrets instead of from a file in the repo.

**Architecture:** Three independent pieces. The test fix is in the harness, not the app — the resolver is already correct. The secrets guard is a `.gitignore` pattern plus a test that fails while any secrets file is tracked. The deploy path extends the existing `build.yml`, which today only syntax-checks.

**Tech Stack:** pytest, GitHub Actions, Railway CLI.

**Spec:** The owner's request, 2026-08-22: fix the two failing tests, and make merges to `Production` use the credentials in `.streamlit/secrets.production.toml` — delivered via GitHub Actions secrets, per their choice.

## Global Constraints

- **Never commit.** The owner reviews and commits.
- `VERSION` reads **1.10.29**, so this plan takes **1.10.30**. The pending backfill plan currently claims 1.10.30 and must be re-pointed to **1.10.31**, which it re-reads from `VERSION` when it finally runs.
- Branch `DEV-04/Market-Overview`.
- **I do not touch the git index.** The owner chose to handle the staged `secrets.production.toml` themselves. I add the ignore pattern and the guard; unstaging is theirs.
- **I never read a value out of any secrets file.** Key names and character counts only.
- **This plan knowingly adds a 5th test failure** — the guard fails while `secrets.production.toml` is staged. That is the guard working, not a regression, and it clears the moment it is unstaged.

---

## Context

### The two failing tests are defeated by a safety fixture

Diagnosed, not guessed. The app code is correct: with `_section` stubbed empty and `DB_HOST=db` in the environment, `db_config()` returns `db 5432` and `data_backbone.config` reloads to `db 5432`. Run outside pytest, everything resolves properly.

Inside pytest it does not, because `tests/conftest.py:48` has an autouse fixture `_no_live_db` that replaces **`db_config` itself**:

```python
monkeypatch.setattr(_secrets, "db_config", lambda: {
    "host": "localhost", "port": 5432, "dbname": "trading", ...})
```

Its purpose is sound — the fast suite must never reach a real Postgres, and a developer machine carries live credentials in `secrets.toml`. But it replaces the whole resolver, so:

- `test_env_vars_are_used_when_there_is_no_secrets_toml` patches `_section` (one level below `db_config`) and therefore patches something that is never called. It asserts `db`, gets `localhost`.
- `test_the_host_really_resolves_to_the_container_database` reads the real `secrets.toml` deliberately and gets the stub instead. It asserts `5433`, gets `5432`.

**The fix is to patch one level lower.** If the fixture stubs `_section` rather than `db_config`, the protection is identical — an empty `[database]` section means `secrets.toml` is ignored, resolution falls to env vars, and on a machine with no `DB_*` set that yields `localhost:5432/trading` with an empty password, which `_resolve_cfg()` turns into `None`, i.e. unconfigured. Same guarantee, and it models the container exactly.

That fixes the first test outright. The second is different in kind: it asserts that *this developer's* `secrets.toml` resolves to `127.0.0.1:5433`, which is machine-specific and **would fail in CI**, where no `secrets.toml` exists. It needs an explicit opt-out plus a skip when the file is absent — otherwise adding the suite to CI (Task 3) breaks the build on day one.

### The staged secrets file

`.streamlit/secrets.production.toml` is **staged** (`AM`) and **not ignored** — `.gitignore:21` names the exact path `.streamlit/secrets.toml`, which does not match. It has never been committed (`git log --all` is empty for it), but the next commit would put `DATABASE_URL`, `PGPASSWORD`, `POSTGRES_PASSWORD`, three API keys, a Gmail app password and a bot token into history on a branch that is already pushed to GitHub.

### Nothing reads that file, and nothing should ship it

`_section()` reads `secrets.toml` only, and `.dockerignore:3` keeps secrets out of images by design. So the file is currently a reference copy. The owner chose GitHub Actions injection, which has one important consequence worth stating plainly: **Railway's own GitHub integration builds from the repo, so an Actions job cannot inject anything into a build Railway performs itself.** For injection to mean anything, the Action must be what deploys — using the Railway CLI with a `RAILWAY_TOKEN` — or must push the values into Railway's variables. Otherwise the secrets sit in GitHub unused while Railway keeps reading its own dashboard variables.

`build.yml` today triggers on `Production` and only runs `py_compile`. It does not run the test suite.

---

## Task 1: Fix the two tests

**Files:** Modify `tests/conftest.py`, `tests/test_data_backbone_config.py`, `pyproject.toml` (marker registration)

- [ ] **Step 1:** Confirm the current failure signature so the fix is provably the cause:
      `pytest tests/test_data_backbone_config.py -q --no-cov` → 2 failed, 7 passed.

- [ ] **Step 2: Patch one level lower in `conftest._no_live_db`.** Replace the `db_config` stub with a `_section` stub:

```python
    # Stub `_section`, not `db_config`. Stubbing the resolver itself also
    # defeats the tests that exist to verify the resolver - they end up
    # asserting against this fixture. Emptying the [database] section gives the
    # same protection (no secrets.toml -> env vars -> localhost/trading with no
    # password -> _resolve_cfg() returns None) and models the container exactly.
    monkeypatch.setattr(_secrets, "_section",
                        lambda name: {} if name == "database" else _real_section(name))
```

Capture `_real_section` before patching so other sections (`[api]`, `[telegram]`, `[gmail]`) keep resolving — several tests read those.

- [ ] **Step 3: Add a `live_secrets` exemption** to the same fixture, beside the existing `"slow" in request.keywords` check, and register the marker in `pyproject.toml` so `--strict-markers` does not reject it.

- [ ] **Step 4: Mark and guard the machine-specific test:**

```python
    @pytest.mark.live_secrets
    def test_the_host_really_resolves_to_the_container_database(self):
        """This developer's own secrets.toml, not a fixture.

        Skipped where there is no [database] section - CI has no secrets.toml,
        and a test that cannot pass there would break the build the day the
        suite is added to it.
        """
        from src.core.secrets import _section
        if not (_section("database") or {}).get("port"):
            pytest.skip("no local [database] in secrets.toml")
        ...
```

- [ ] **Step 5: Green** — `pytest tests/test_data_backbone_config.py -q --no-cov` → 9 passed (or 8 passed 1 skipped on a machine without local secrets).

- [ ] **Step 6: Whole suite** — the 2 known `data_backbone` failures must be gone, leaving only the 2 GARCH ones plus Task 2's deliberate guard failure.

---

## Task 2: Make a secrets file uncommittable

**Files:** Modify `.gitignore` · Create `tests/test_no_secrets_tracked.py`

- [ ] **Step 1: Widen the ignore** — replace the exact-name line with a pattern that catches every variant, keeping the example templates tracked:

```gitignore
.streamlit/secrets*.toml
!.streamlit/secrets.toml.example
```

- [ ] **Step 2: Write the guard**

```python
"""No secrets file may be tracked by git.

.gitignore named one exact path, `.streamlit/secrets.toml`, so
`secrets.production.toml` was not covered and reached the index on 2026-08-22
carrying DATABASE_URL, two Postgres passwords, three API keys, a Gmail app
password and a bot token. It was caught before any commit; git history has no
delete.
"""

def _tracked() -> list:
    out = subprocess.run(["git", "ls-files"], cwd=_REPO,
                         capture_output=True, text=True, timeout=60)
    return out.stdout.splitlines()


def test_no_secrets_file_is_tracked():
    bad = [f for f in _tracked()
           if re.search(r"secrets.*\.toml$", f) and not f.endswith(".example")]
    assert not bad, f"secrets files tracked by git: {bad}"


def test_the_example_templates_are_still_tracked():
    # The ignore pattern must not hide the templates a new machine needs.
    assert any(f.endswith("secrets.toml.example") for f in _tracked())
```

`git ls-files` lists the **index**, so this fails while the file is staged — deliberately. It goes green when the owner runs `git rm --cached .streamlit/secrets.production.toml`.

- [ ] **Step 3: Run it and read the failure** — it must name `.streamlit/secrets.production.toml`.

---

## Task 3: Deploy with injected credentials

**Files:** Modify `.github/workflows/build.yml`

- [ ] **Step 1: Run the test suite in CI.** Today the workflow only runs `py_compile`, so nothing would have caught the two broken tests. Add a step running the fast suite with `--no-cov` (coverage needs the full source tree and is a local gate).

- [ ] **Step 2: Add a `deploy` job**, gated on `github.ref == 'refs/heads/Production'` and `needs: build`, using the `production` GitHub Environment that already exists — that is what makes the environment's secrets available and lets the owner add a required reviewer.

- [ ] **Step 3: Inject at deploy, never into the repo.** The job writes `.streamlit/secrets.toml` on the runner from `${{ secrets.* }}` and deploys with the Railway CLI:

```yaml
      - name: Write runtime secrets
        run: |
          mkdir -p .streamlit
          cat > .streamlit/secrets.toml <<'TOML'
          [database]
          url = "${{ secrets.DATABASE_URL }}"
          [telegram]
          bot_token = "${{ secrets.TELEGRAM_BOT_TOKEN }}"
          chat_id = "${{ secrets.TELEGRAM_CHAT_ID }}"
          TOML
      - name: Deploy
        env:
          RAILWAY_TOKEN: ${{ secrets.RAILWAY_TOKEN }}
        run: railway up --service dashboard-pro --detach
```

The file exists only for the life of the runner and is `.gitignore`d by Task 1, so it cannot be committed by accident.

- [ ] **Step 4: Document the secrets the owner must add** to the `production` environment — the key names from `secrets.production.toml`, plus `RAILWAY_TOKEN`. Names only; the owner enters the values.

- [ ] **Step 5:** `python -c "import yaml; yaml.safe_load(open('.github/workflows/build.yml'))"` to prove the workflow parses.

---

## Verification

Evidence before claims.

1. **The two tests pass:** `pytest tests/test_data_backbone_config.py -q --no-cov` → no failures.
2. **The safety net still holds** — `pytest tests/test_precomputed.py tests/test_signal_store.py -q --no-cov` passes, proving the `_section` stub protects the DB-optional paths as the `db_config` stub did.
3. **The guard names the staged file:** `pytest tests/test_no_secrets_tracked.py -q --no-cov` fails with `.streamlit/secrets.production.toml`, and passes after `git rm --cached`.
4. **The ignore pattern covers every variant:** `git check-ignore -v .streamlit/secrets.production.toml` and `.../secrets.toml` both match; `git check-ignore .streamlit/secrets.toml.example` does **not**.
5. **Full suite:** expect the 2 GARCH failures **and** the guard failure — 3 total, with the two `data_backbone` ones gone. Any fourth is ours.
6. **Workflow parses**, and `deploy` is gated on the `Production` ref.
7. **Deploy:** bump to 1.10.30, rebuild, `verify_deploy.py`, four containers in sync. Re-point the backfill plan to 1.10.31.
8. Show the owner the diff. **Never commit.**
