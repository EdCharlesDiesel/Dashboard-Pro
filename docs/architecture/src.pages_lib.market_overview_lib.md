---
layer: pages_lib
lines: 1055
generated: true
---

# `src/pages_lib/market_overview_lib.py`

Shared logic for the Market Overview workbench and its broken-out pages

Source: [[Src/pages_lib/market_overview_lib.py]] · [open on disk](../src/pages_lib/market_overview_lib.py)

## Imports (13)

- [[src.core.analyzer]] — —
- [[src.core.bias]] — Canonical directional bias — the single definition of BULLISH / BEARIS
- [[src.core.config]] — —
- [[src.core.data_provider]] — —
- [[src.core.signals]] — —
- [[src.instruments.registry]] — Instrument registry — single source of truth
- [[src.pages_lib.navigation]] — Single source of truth for the page navigation
- [[src.services.alert_service]] — Shared email-alert plumbing for the scanner pages
- [[src.services.bias_service]] — Streamlit-facing wrapper over the canonical bias engine
- [[src.services.parallel_fetch]] — Thread-pool fan-out for I/O-bound per-item fetches (yfinance/Postgres)
- [[src.services.precomputed]] — Precomputed score board — the worker's output the UI reads instead of
- [[src.services.signal_store]] — Shared trade-signal persistence — save any page's signals to ``trade_s
- [[src.services.tool_log]] — Shared usage logging for the interactive tool pages (R:R Calculator,

## Imported by (0)

_Nothing imports this. Either an entry point or dead code._

## Public surface

`inject_css`, `play_alert_sound`, `send_email_alert`, `init_notification_state`, `check_and_notify`, `load_all_market_data`, `clear_data_cache`, `ensure_loaded`, `render_full_sidebar`, `subpage_sidebar`, `render_kpis`, `render_overview_tab`, `render_mtf_matrix_tab`, `render_technical_chart_tab`, `render_trading_view_tab`, `render_macro_pro_tab`, `render_trading_ideas_tab`, `render_volume_profile_tab`

Back to [[Architecture]].
