# spark-mcp

Spark — the creativity and institutional-learning MCP for quant research
organizations. Implements the Spark v3 master plan:
`/home/kanav/Documents/Projects/Creative_MCP_MUSE/spark_v3_master_plan.md`.

Spark remembers what a research organization has tried (including failures),
counts every trial so statistical verdicts are deflated rather than lucky,
points at unexplored regions of research space, and turns validated
trajectories into replay-gated skills. It never backtests and never owns
market data — alpha_engine computes, Spark judges.

## Install

```bash
uv venv .venv --python 3.12
uv pip install -e '.[dev]'
```

Python 3.12 recommended (3.13 works for the core, but `pyribs` is not
published for it on this machine's package index — it is an optional `[qd]`
extra in any case).

## Quick start

```bash
spark init-db          # create ~/.spark/spark.db (schema migrations)
spark doctor           # capability report (degraded mode, FTS5, sqlite-vec)
spark serve            # MCP server over stdio
python -m pytest -q    # test suite incl. the independence gate
```

## Layout

- `src/spark_mcp/` — package (config, storage, llm protocols+fakes, server, cli)
- `tests/` — unit tests + `tests/independence/` (the blocking independence gate)
- `reviews/` — independent milestone reviews (muse-spark-1.3-contributor, max effort)
- `STATUS.md` — per-phase verdicts and program-gate evidence

## The independence contract

This package must work when installed alone from PyPI. The reference repos
studied during design (hermes-agent, docling, langmem, graphiti, dspy,
mlflow, aflow, OpenHands, QDax) are never imported, read, or required at
runtime. `tests/independence/` enforces this on every test run.
