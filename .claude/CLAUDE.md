# Project AI Instructions

Authoritative project guidance lives in the root **CLAUDE.md** (architecture,
commands, forex domain conventions). Read that first.

The previous content of this file described a different stack (FastAPI,
SQLAlchemy, Redis, Docker, Pytest) and did not match this repository — this is
a Streamlit + yfinance + psycopg2 application with no test suite or Docker.

Still-applicable preferences:

- Forex domain conventions.
- Signals engine
- Everything Forex and predicting market data.
- Traders, market makers, and other financial professionals.
- Mathematical models and algorithms.
- Predictive modeling.
- Data science.
- Price forecasting.
- Optimization.
- Oil and gas.
- Gold
- Streamlit.
- Python.
- Quantitative trading.
- Technical analysis.
- Fundamental analysis.
- Sentiment analysis.
- Risk management.
- Object-Oriented Programming.
- Data structures.
- Machine learning.
- Financial Engineering.
- Use type hints on new code.
- Return complete implementations — no TODO comments, no placeholder code.
- Prefer composition over inheritance (see `BloombergPage` template-method
  pattern in `src/pages_lib/base.py`).
- Never hardcode secrets; they belong in `.streamlit/secrets.toml`
  (gitignored), not `.streamlit/config.toml` (tracked).
