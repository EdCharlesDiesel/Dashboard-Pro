"""
data_backbone — the durable data pipeline that sits behind the dashboard.

    Postgres (durable)  ->  yfinance / FRED (source of truth)

This package is self-contained and Streamlit-free (except the optional
``app_demo`` entry point). Run the background refresher with::

    python -m src.data_backbone.worker

and the one-shot historical backfill with::

    python -m src.data_backbone.seed_history
"""
