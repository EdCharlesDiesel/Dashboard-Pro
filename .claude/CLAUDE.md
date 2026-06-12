# Project AI Instructions

Authoritative project guidance lives in the root **CLAUDE.md** (architecture,
commands, forex domain conventions). Read that first.

The previous content of this file described a different stack (FastAPI,
SQLAlchemy, Redis, Docker, Pytest) and did not match this repository — this is
a Streamlit + yfinance + psycopg2 application with no test suite or Docker.

Still-applicable preferences:

- Use type hints on new code.
- Return complete implementations — no TODO comments, no placeholder code.
- Prefer composition over inheritance (see `BloombergPage` template-method
  pattern in `src/pages_lib/base.py`).
- Never hardcode secrets; they belong in `.streamlit/secrets.toml`
  (gitignored), not `.streamlit/config.toml` (tracked).
