# AGENTS.md

Guidance for AI coding agents working in this repository. The design contract
is `../Creative_MCP_MUSE/spark_v3_master_plan.md` — read the relevant section
before changing anything.

## What this is

`spark-mcp`: the Spark creativity/institutional-learning MCP, built per the
v3 master plan, Phases 0-3 in this goal (ledger + gates MVP). Standalone
package; QuantFlow and alpha_engine are optional peers, never dependencies.

## Hard rules

1. **Independence contract.** Never import, read, or subprocess the reference
   repos (hermes-agent, docling, langmem, graphiti, dspy, mlflow, aflow,
   OpenHands, QDax). `tests/independence/` scans for violations — it is
   blocking. New runtime deps must be PyPI-installable on this machine; note
   that this machine's index lacks `pyribs` (hence the `[qd]` extra).
2. **Spark never computes returns.** All backtest/causal/risk math lives
   behind provider protocols (`SimulatedProvider`, `RecordedProvider`,
   `AlphaEngineProvider`). The only native statistics are PSR/DSR, trial
   counting, reward composition, bandit posteriors.
3. **PIT correctness.** Every data-dependent row carries `as_of`/
   `data_through`; retrieval filters before ranking. Leakage tests are
   blocking.
4. **Trial accounting.** A pending trial row is inserted before dispatch;
   idempotent retries never increment M; out-of-band outcomes create
   `reconciled` trials. The gate is bypassable only by explicitly breaking
   rule 1-3 in review.
5. **Milestones close only with a review.** Each phase: tests green → run the
   `milestone-review` skill (muse headless reviewer) → verdict recorded in
   STATUS.md → local git commit. Never mark a phase done on self-assessment.
6. **G1 citation.** Never cite G1 as bare GREEN. The only correct form is:
   G1 — judge RED [5,1,1,1,1] single-draw; GREEN only under Tier-3 union-of-5
   amendment. The single-draw RED is the original §18.0 criterion result; the
   GREEN applies only to the amended distribution criterion. See
   `docs/gates.md` and STATUS.md.

## Commands

```bash
uv pip install -e '.[dev]'   # sync deps
python -m pytest -q          # full suite (independence gate included)
.venv/bin/ruff check src tests
spark doctor                 # capability report
```

## Conventions

- Python 3.12, stdlib-first; numpy/pandas only where they earn their place.
- Time fields: ISO-8601 UTC text (`YYYY-MM-DDTHH:MM:SSZ`).
- IDs: ULIDs with type prefixes (`hyp_`, `ep_`, `trial_`, `belief_`).
- Storage: SQLite WAL, `busy_timeout=5000`, migrations via
  `PRAGMA user_version` in `src/spark_mcp/storage.py`.
- Config defaults for any experiment-affecting constant belong in
  `ResearchConfig` (`config.py`), not literals in algorithm code.
