---
name: the-appication-needs-to-do
description: Pointer to the project conventions in .claude/CLAUDE.md - forex price formatting, the ATR risk model and 18-point checklist, session windows, metals naming, correlation stacking, and repo code rules.
---

# Project conventions live in `.claude/CLAUDE.md`

This content was moved there on 2026-08-14 and **is not duplicated here**.

The reason is the failure mode this repo keeps hitting: two copies of the same
truth drift apart, and nothing tells you which one is stale. The version file
and `.env` drifted; a container ran 28-hour-old code while every check passed.
A second copy of the conventions would do the same, and the copy that loses is
always the one you happen to read.

`.claude/CLAUDE.md` is loaded automatically at the start of every session, which
is what this guidance needs — price formatting, the risk model and the code
rules have to apply by default, not only when a skill is invoked. Read it there.
