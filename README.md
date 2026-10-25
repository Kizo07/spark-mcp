# spark-mcp

Creativity and institutional-learning MCP for quant research
organizations. Spark remembers what a research organization has tried
(including failures), counts every trial so statistical verdicts are
deflated rather than lucky, points at unexplored regions of research
space, and turns validated trajectories into replay-gated skills. It
never backtests and never owns market data — a peer engine computes,
Spark judges.

## Install

```bash
uv venv .venv --python 3.12
uv pip install -e '.[dev]'
```

Requires Python ≥3.11. The `[qd]` extra needs `pyribs` (optional).

## Quick start

```bash
spark init-db          # create ~/.spark/spark.db (schema migrations)
spark doctor           # capability report (degraded mode, FTS5, sqlite-vec)
spark serve            # MCP server over stdio (equivalent: python -m spark_mcp)
python -m pytest -q    # test suite incl. the independence gate
```

## Program gates

G1 — judge RED [5,1,1,1,1] single-draw; GREEN only under Tier-3 union-of-5 amendment. The single-draw figure is the original §18.0 criterion result; the GREEN applies only to the amended distribution criterion. See [STATUS.md](STATUS.md) for the phase verdicts and [docs/gates.md](docs/gates.md) for the G1–G5 original vs amended/current criteria and evidence refs. Transport decision: [docs/adr/transport.md](docs/adr/transport.md).

## Configuration

Copy `config.example.yaml` to your `SPARK_CONFIG` path or rely on
defaults (DB at `~/.spark/spark.db`). Secrets are environment-variable
references only — never values in files:

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

30 tools, highlights:

- Ledger & memory: `register_hypothesis`, `hypothesis_propose`,
  `log_outcome`, `memory_recall`, `memory_promote`, `research_history`
- Exploration: `explore_methods`, `suggest_next`, `archive_status`,
  `find_analogies`, `combine_methods`
- Gates & evidence: `significance_gate`, `check_novelty`,
  `backtest_run`, `causal_check`, `risk_decompose`
- Skill evolution: `distill_skill`, `propose_skill_challenger`,
  `run_replay`, `decide_skill_promotion`, `approve`, `retire_skill`
- Artifact evolution: `evolution_run`, `evolution_status` (bounded
  hypothesis → backtest → DSR-gate → computed-promotion loop; the LLM
  narrates but never decides; veto post-run via `approve`)
- Peer seam: `run_summary_ingest` (QuantFlow Tier-2 hook),
  `report_write`, `consolidate`, `generate_exercise`,
  `spark_health`, `spark_doctor`

## Artifact evolution

`evolution_run` evolves research artifacts (hypotheses + backtest specs),
scored by executed provider results through the existing gate stack — trial
governor, DSR `significance_gate`, strict DSR/SR promotion margins,
challenger/champion/rollback line heads, and human veto. GEPA stays scoped
to skill/procedure text only. Knobs live under `research.evolution_*` in
`config.example.yaml` (rounds, margins, first-promotion floor, SOTA-jump
policy, render budget, level-2 spec template + grids). Runs are
bit-reproducible given (seed, caps, provider): reruns agree on decisions
and heads by hypothesis content + episode metrics. Held-out/OOS evidence
never enters evolution feedback — it is promotion-time evidence only
(`consolidate` also alarms while the replay fixture suite counts below
`replay_min_fixtures`).

## Layout

- `src/spark_mcp/` — package (config, storage, llm protocols+fakes,
  server, cli, gates, QD archive, skill evolution)
- `tests/` — unit tests + `tests/independence/` (blocking gate:
  no third-party research repos imported, read, or required at runtime)

## Conventions

- Time fields are ISO-8601 UTC text; IDs are ULIDs with type prefixes
  (`hyp_`, `ep_`, `trial_`, `belief_`).
- Storage is SQLite WAL with `PRAGMA user_version` migrations.
- Every data-dependent row carries `as_of`/`data_through`; retrieval
  filters before ranking (no leakage).
- Trial rows are inserted before dispatch; retries never double-count.

## License

MIT — see `pyproject.toml`.
