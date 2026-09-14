# spark-mcp

Spark — the creativity and institutional-learning MCP for quant research
organizations. Spark remembers what a research organization has tried
(including failures), counts every trial so statistical verdicts are
deflated rather than lucky, points at unexplored regions of research
space, and turns validated trajectories into replay-gated skills. It
never backtests and never owns market data — a peer engine computes,
Spark judges.

Built per the Spark v3 master plan through phases 0–7 (hypothesis
ledger, method graph + novelty, gates, QuantFlow hook seam, MAP-Elites
QD archive, skill evolution, analogy/consolidation/curriculum). Current
state and per-phase evidence: `STATUS.md`; independent milestone
reviews: `reviews/`.

## Install

```bash
uv venv .venv --python 3.12
uv pip install -e '.[dev]'
```

Requires Python ≥3.11. The `[qd]` extra needs `pyribs` (optional —
the QD archive is pure-Python by design).

## Quick start

```bash
spark init-db          # create ~/.spark/spark.db (schema migrations)
spark doctor           # capability report (degraded mode, FTS5, sqlite-vec)
spark serve            # MCP server over stdio (equivalent: python -m spark_mcp)
python -m pytest -q    # test suite incl. the independence gate
```

## Configuration

All settings live in `config.example.yaml` (copy to your `SPARK_CONFIG`
path or rely on defaults: DB at `~/.spark/spark.db`). Secrets are
environment-variable references only — never values in files:

| Setting | Env var | Purpose |
|---|---|---|
| LLM key | `OPENAI_API_KEY` (configurable) | scoring / judging calls |
| Gateway token | `QUANTFLOW_GATEWAY_TOKEN` | skill-promotion manifest POST |

Without an LLM key Spark runs in degraded mode (lexical novelty proxy,
deterministic exploration) — fully usable, no network needed.

## Use as an MCP server

Any MCP-capable agent or harness can spawn it over stdio:

```json
{ "command": "<venv>/bin/python",
  "args": ["-m", "spark_mcp"],
  "env": { "SPARK_CONFIG": "/path/to/spark.yaml" } }
```

28 tools, highlights:

- Ledger & memory: `register_hypothesis`, `hypothesis_propose`,
  `log_outcome`, `memory_recall`, `memory_promote`, `research_history`
- Exploration: `explore_methods`, `suggest_next`, `archive_status`,
  `find_analogies`, `combine_methods`
- Gates & evidence: `significance_gate`, `check_novelty`,
  `backtest_run`, `causal_check`, `risk_decompose`
- Skill evolution: `distill_skill`, `propose_skill_challenger`,
  `run_replay`, `decide_skill_promotion`, `approve`, `retire_skill`
- Peer seam: `run_summary_ingest` (QuantFlow Tier-2 hook),
  `report_write`, `consolidate`, `generate_exercise`,
  `spark_health`, `spark_doctor`

## Layout

- `src/spark_mcp/` — package (config, storage, llm protocols+fakes,
  server, cli, gates, QD archive, skill evolution)
- `tests/` — unit tests + `tests/independence/` (the blocking
  independence gate)
- `reviews/` — independent milestone reviews
- `STATUS.md` — per-phase verdicts and program-gate evidence

## The independence contract

This package works installed alone. The reference repos studied during
design are never imported, read, or required at runtime.
`tests/independence/` enforces this on every test run.

## Conventions

- Time fields are ISO-8601 UTC text; IDs are ULIDs with type prefixes
  (`hyp_`, `ep_`, `trial_`, `belief_`).
- Storage is SQLite WAL with `PRAGMA user_version` migrations.
- Every data-dependent row carries `as_of`/`data_through`; retrieval
  filters before ranking (no leakage).
- Trial rows are inserted before dispatch; retries never double-count.

## License

MIT — see `pyproject.toml`.
