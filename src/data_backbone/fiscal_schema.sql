-- Treasury Fiscal Data series.
--
-- Deliberately NOT stored in `fred_series`, whose `value` is `double precision`.
-- The headline debt figure needs 16 significant digits (40102964278586.10) and
-- float64 gives ~15.95, so a double holds it as 40102964278586.1015625. That
-- still *formats* to the right cents today, which is what makes it dangerous:
-- it reads correctly in every dashboard while being wrong in the store, and it
-- stops rounding back as the figure grows.
--
-- NUMERIC is exact. The cost is that reads come back as Decimal rather than
-- float, which is the correct trade for money.
--
-- Idempotent throughout, matching src/execution/schema.sql -- safe to re-run.

CREATE TABLE IF NOT EXISTS fiscal_series (
    series_id   TEXT           NOT NULL,   -- e.g. tot_pub_debt_out_amt
    record_date DATE           NOT NULL,   -- Treasury's record_date, not fetch time
    value       NUMERIC(24, 2) NOT NULL,   -- exact to the cent
    source      TEXT           NOT NULL DEFAULT 'treasury_fiscal_data',
    fetched_at  TIMESTAMPTZ    NOT NULL DEFAULT now(),
    PRIMARY KEY (series_id, record_date)
);

-- The dashboard reads "latest value per series" and "one series over time";
-- both are served by walking this index backwards.
CREATE INDEX IF NOT EXISTS ix_fiscal_series_id_date
    ON fiscal_series (series_id, record_date DESC);
