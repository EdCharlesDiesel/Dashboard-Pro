---
layer: db
lines: 89
generated: true
---

# `src/db/connection.py`

App-level database auto-connection

Source: [[Src/db/connection.py]] · [open on disk](../src/db/connection.py)

## Imports (3)

- [[src.db.cache]] — Connection pooling for the Postgres trade repository
- [[src.db.market_data_repository]] — Market-data persistence via PostgreSQL
- [[src.db.trade_repository]] — Trade persistence via PostgreSQL

## Imported by (3)

- [[src.pages_lib.daily_trading.sidebar]]
- [[src.pages_lib.daily_trading.state]]
- [[src.ui.theme]]

## Public surface

`current_db_config`, `auto_connect`

Back to [[Architecture]].
