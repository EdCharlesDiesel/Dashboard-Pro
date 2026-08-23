"""Signal execution: queue, pre-trade gate, and the MT5 executor worker.

A separate package because these three form one pipeline that the existing
layout has no home for — a signal is enqueued (Postgres), gated and sized (pure
logic), then sent to the terminal (Windows/MT5). `core/` is pure analysis and
`services/` is per-concern helpers; neither fits a stateful pipeline.

Nothing here is imported by a page. `mt5_executor` is a standalone worker and
must be launched deliberately -- see its module docstring.
"""
