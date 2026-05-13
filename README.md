# Dashboard-Pro

A modular and professional Forex Macro Dashboard with technical analysis, fundamentals, and entry signals.

## Project Structure

- `main_dashboard.py`: Consolidated production-grade dashboard.
- `AnalysisDashboardBackTest.py`: Tool for backtesting strategies on historical EUR/USD data.
- `src/core/`: Modular core logic.
  - `analyzer.py`: Technical indicator calculations using the `ta` library.
  - `data_provider.py`: Market data fetching (Yahoo Finance, QuantConnect, FRED).
  - `signals.py`: Entry signals and trading idea generation logic.
  - `config.py`: Centralized application configuration.
- `archive/`: Legacy and experimental versions of the dashboard.

## Setup

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Set your FRED API key (optional but recommended for live macro data):
    - Add to `.streamlit/secrets.toml`:
      ```toml
      FRED_API_KEY = "your_api_key_here"
      ```
    - Or set as an environment variable:
      ```bash
      export FRED_API_KEY=your_api_key_here
      ```
3.  Run the dashboard:
    ```bash
    streamlit run main_dashboard.py
    ```
