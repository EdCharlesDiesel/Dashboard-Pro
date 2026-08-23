-- ===========================================================================
-- Dashboard-Pro execution layer schema
--
-- Postgres is the queue between the sentry (Railway) and the MT5 executor
-- (your Windows box). No inbound ports, no tunnel, and the queue survives a
-- restart on either side.
--
-- Apply with: psql "$DATABASE_URL" -f schema_execution.sql
-- ===========================================================================

CREATE TABLE IF NOT EXISTS pending_signals (
    id              BIGSERIAL PRIMARY KEY,

    -- Idempotency. Derived from (symbol, direction, entry, stop, leg timestamp).
    -- The UNIQUE constraint is what kills the 09:19 / 09:22 duplicate: the
    -- second insert conflicts and is discarded before it ever reaches MT5.
    signal_id       TEXT NOT NULL UNIQUE,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    source          TEXT NOT NULL DEFAULT 'evening_sentry',

    symbol          TEXT NOT NULL,
    direction       TEXT NOT NULL CHECK (direction IN ('BUY', 'SELL')),
    entry           NUMERIC(18, 6) NOT NULL,
    stop            NUMERIC(18, 6) NOT NULL,
    tp1             NUMERIC(18, 6),
    tp2             NUMERIC(18, 6),
    risk_pct        NUMERIC(6, 4),          -- NULL -> executor default
    meta            JSONB NOT NULL DEFAULT '{}'::jsonb,

    status          TEXT NOT NULL DEFAULT 'PENDING'
                    CHECK (status IN ('PENDING', 'CLAIMED', 'PLACED', 'FILLED',
                                      'REJECTED', 'EXPIRED', 'CANCELLED', 'ERROR')),

    -- Signals go stale fast. A fib entry from 40 minutes ago is not the same
    -- trade; expire rather than chase.
    expires_at      TIMESTAMPTZ NOT NULL DEFAULT now() + INTERVAL '15 minutes',

    claimed_at      TIMESTAMPTZ,
    claimed_by      TEXT,
    attempts        SMALLINT NOT NULL DEFAULT 0,

    reject_reason   TEXT,
    ticket          BIGINT,
    order_type      TEXT,
    lots            NUMERIC(12, 4),
    fill_price      NUMERIC(18, 6),
    resolved_at     TIMESTAMPTZ
);

-- Partial index: the claim loop only ever scans actionable rows.
CREATE INDEX IF NOT EXISTS ix_pending_signals_claimable
    ON pending_signals (created_at)
    WHERE status = 'PENDING';

CREATE INDEX IF NOT EXISTS ix_pending_signals_ticket
    ON pending_signals (ticket) WHERE ticket IS NOT NULL;

CREATE INDEX IF NOT EXISTS ix_pending_signals_status_created
    ON pending_signals (status, created_at DESC);


-- ---------------------------------------------------------------------------
-- Append-only audit. Never updated, never deleted. This is the record you
-- reconcile against when shadow mode and the backtest disagree.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS execution_log (
    id              BIGSERIAL PRIMARY KEY,
    ts              TIMESTAMPTZ NOT NULL DEFAULT now(),
    signal_id       TEXT,
    worker          TEXT,
    event           TEXT NOT NULL,      -- CLAIMED / GATE_PASS / GATE_BLOCK /
                                        -- SIZED / DRY_RUN / SENT / FILLED /
                                        -- REJECTED / ERROR / RECONCILED
    dry_run         BOOLEAN NOT NULL DEFAULT TRUE,
    detail          JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS ix_execution_log_ts ON execution_log (ts DESC);
CREATE INDEX IF NOT EXISTS ix_execution_log_signal ON execution_log (signal_id);
CREATE INDEX IF NOT EXISTS ix_execution_log_event ON execution_log (event, ts DESC);


-- ---------------------------------------------------------------------------
-- Single-row control table. The kill switch you can flip from the dashboard.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS executor_state (
    id              SMALLINT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    enabled         BOOLEAN NOT NULL DEFAULT FALSE,   -- OFF until you turn it on
    dry_run         BOOLEAN NOT NULL DEFAULT TRUE,    -- shadow mode by default
    halt_reason     TEXT,
    daily_loss_r    NUMERIC(10, 4) NOT NULL DEFAULT 0,
    daily_loss_date DATE NOT NULL DEFAULT CURRENT_DATE,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO executor_state (id, enabled, dry_run)
VALUES (1, FALSE, TRUE)
ON CONFLICT (id) DO NOTHING;


-- ---------------------------------------------------------------------------
-- What the executor actually did, mirrored for the journal / R-multiple tabs.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS executed_trades (
    id              BIGSERIAL PRIMARY KEY,
    signal_id       TEXT REFERENCES pending_signals (signal_id),
    ticket          BIGINT UNIQUE,
    symbol          TEXT NOT NULL,
    direction       TEXT NOT NULL,
    lots            NUMERIC(12, 4) NOT NULL,
    entry_price     NUMERIC(18, 6),
    stop_price      NUMERIC(18, 6),
    tp_price        NUMERIC(18, 6),
    opened_at       TIMESTAMPTZ,
    closed_at       TIMESTAMPTZ,
    close_price     NUMERIC(18, 6),
    pnl             NUMERIC(18, 4),
    r_multiple      NUMERIC(10, 4),
    dry_run         BOOLEAN NOT NULL DEFAULT TRUE,
    meta            JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS ix_executed_trades_closed
    ON executed_trades (closed_at DESC) WHERE closed_at IS NOT NULL;


-- ---------------------------------------------------------------------------
-- Convenience view for the dashboard: today's activity at a glance.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_execution_today AS
SELECT  p.signal_id,
        p.created_at,
        p.symbol,
        p.direction,
        p.status,
        p.reject_reason,
        p.lots,
        p.fill_price,
        p.ticket,
        t.pnl,
        t.r_multiple,
        t.dry_run
FROM    pending_signals p
LEFT    JOIN executed_trades t ON t.signal_id = p.signal_id
WHERE   p.created_at >= CURRENT_DATE
ORDER BY p.created_at DESC;
