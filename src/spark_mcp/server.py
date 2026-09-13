"""MCP server: Phase-0 health tools, Phase-1 ledger & memory tools, Phase-2
method graph & novelty tools, Phase-3 gates & backtest-seam tools, Phase-4
QuantFlow Tier-2 hook seam, Phase-5 QD archive & exploration tools.

`create_server(config)` builds a `Services` (config + database handle +
embedder/llm clients + backtest provider) that the tool functions close over —
no module-level mutable state, so tests can inject a temp-db Config and fake
clients. When no Config is passed, `SPARK_CONFIG` (env) is used if set, else
defaults (db_path ~/.spark/spark.db, plan §16).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from fastmcp import FastMCP

from . import __version__, archive, explore, graph, hooks, ledger, memory, novelty
from .budgets import spend_summary
from .config import Config
from .gates import service as gates_service
from .llm import build_embedder, build_llm
from .storage import Database, open_db
from .timeutil import now_iso, validate_iso


@dataclass
class Services:
    """One config + one database handle + model clients + provider, shared by
    every tool handler. `embedder`/`llm` are None in degraded mode (§4.3)."""

    cfg: Config
    db: Database
    embedder: object | None = None
    llm: object | None = None
    provider: object | None = None


def _resolve_config(config: Config | None) -> Config:
    if config is not None:
        return config
    env_path = os.environ.get("SPARK_CONFIG")
    return Config.from_yaml(env_path) if env_path else Config.from_yaml()


def _build_provider(cfg: Config):
    """Backtest provider per §15.2: AlphaEngineProvider when the peer is
    enabled (stdio/http via peers.alpha_engine), SimulatedProvider otherwise
    (default; CI/demo)."""
    peer = cfg.peers.alpha_engine
    if peer.enabled and (peer.command or peer.url):
        from .providers.alpha_engine import AlphaEngineProvider

        return AlphaEngineProvider(peer)
    from .providers.simulated import SimulatedProvider

    return SimulatedProvider(cfg.artifacts_dir())


def _as_of_or_now(as_of: str | None) -> str:
    return validate_iso(as_of) if as_of else now_iso()


def create_server(
    config: Config | None = None,
    *,
    embedder: object | None = None,
    llm: object | None = None,
    seeds: bool = True,
    provider: object | None = None,
) -> FastMCP:
    cfg = _resolve_config(config)
    db_path = cfg.db_path()
    services = Services(
        cfg=cfg,
        db=open_db(db_path),
        embedder=build_embedder(cfg) if embedder is None else embedder,
        llm=build_llm(cfg) if llm is None else llm,
        provider=_build_provider(cfg) if provider is None else provider,
    )
    cfg.artifacts_dir().mkdir(parents=True, exist_ok=True)
    if seeds:
        graph.import_seeds(services.db, services.embedder)

    mcp = FastMCP("spark", version=__version__)

    @mcp.tool()
    def spark_health() -> dict:
        """Liveness + storage readiness."""
        return {
            "service": "spark",
            "version": __version__,
            "db": str(db_path),
            "db_exists": Path(db_path).exists(),
            "schema_version": services.db.conn.execute("PRAGMA user_version").fetchone()[0],
        }

    @mcp.tool()
    def spark_doctor() -> dict:
        """Config/capability report: degraded mode, FTS5, sqlite-vec, peers."""
        peers = {}
        for name in ("quantflow", "alpha_engine", "documents"):
            peer = getattr(cfg.peers, name)
            peers[name] = {
                "enabled": peer.enabled,
                "transport": peer.transport,
                "configured": bool(peer.command or peer.url),
            }
        return {
            "version": __version__,
            "mode": cfg.active_mode(),
            "llm_model": cfg.llm.model,
            "embedder_model": cfg.llm.embedder_model,
            "fts5": services.db.fts5_available(),
            "sqlite_vec": services.db.sqlite_vec_available(),
            "peers": peers,
            "artifacts_dir": str(cfg.artifacts_dir()),
        }

    # -- Phase 1: ledger & memory tools (plan §6) -----------------------------

    @mcp.tool()
    def hypothesis_propose(
        idea: str,
        family: str,
        universe: str,
        horizon: str,
        scope: str,
        mechanism: str,
        expected_direction: str,
        falsifier: str,
        predicted_ic: float,
        predicted_sigma: float,
        hypothesis_type: str,
        dag: dict | None = None,
        adjustment_set: list[str] | None = None,
        parent_hypothesis_ids: list[str] | None = None,
        as_of: str | None = None,
        client_id: str | None = None,
    ) -> dict:
        """Propose a hypothesis. C6 refusal rule: mechanism, expected_direction,
        horizon, falsifier, scope, predicted_ic, predicted_sigma required for
        ALL types; dag + adjustment_set required iff hypothesis_type='causal'.
        Returns {hypothesis_id, N, S, novelty{...}} — N bounded per §7 (lexical
        proxy in degraded mode), S null until a realized_ic exists. Idempotent
        on client_id.
        """
        as_of = _as_of_or_now(as_of)
        result = ledger.propose_hypothesis(
            services.db,
            cfg,
            research_family=family,
            hypothesis_type=hypothesis_type,
            mechanism=mechanism,
            expected_direction=expected_direction,
            horizon=horizon,
            scope=scope,
            falsifier=falsifier,
            predicted_ic=predicted_ic,
            predicted_sigma=predicted_sigma,
            universe=universe,
            idea=idea,
            dag=dag,
            adjustment_set=adjustment_set,
            parent_hypothesis_ids=parent_hypothesis_ids,
            as_of=as_of,
            client_id=client_id,
            embedder=services.embedder,
        )
        # trajectory event for this call is recorded inside ledger.propose_hypothesis
        return (result)

    @mcp.tool()
    def register_hypothesis(
        hypothesis_id: str,
        spec: dict,
        seed: int,
        client_id: str | None = None,
    ) -> dict:
        """Create the trial episode (status 'registered'), compute protocol_hash
        (SHA-256 over canonical JSON of {hypothesis_id, spec, seed, code_version}),
        flip the hypothesis to 'registered'. Idempotent on client_id.
        """
        return (
            ledger.register_hypothesis(services.db, cfg, hypothesis_id, spec, seed, client_id)
        )

    @mcp.tool()
    def log_outcome(
        episode_id: str,
        outcome: str,
        metrics: dict | None = None,
        failure_reason: str | None = None,
        costs: dict | None = None,
        reward: float | None = None,
        client_id: str | None = None,
    ) -> dict:
        """Close the loop on an episode: flip status per outcome; costs are
        required-or-estimated (missing -> {usd: 0, tokens: 0, estimated: true});
        reconciliation rule (C4): metrics for an episode with no counted trial
        insert a 'reconciled' trial and flag the episode. Failure outcomes write
        a dedupe-keyed tombstone; every outcome appends to spend_ledger and
        ticks the per-family kill-rule counters. Idempotent on client_id.
        """
        return (
            ledger.log_outcome(
                services.db,
                cfg,
                episode_id,
                outcome,
                metrics=metrics,
                failure_reason=failure_reason,
                costs=costs,
                reward=reward,
                client_id=client_id,
                embedder=services.embedder,
            )
        )

    @mcp.tool()
    def research_history(
        query: str,
        as_of: str | None = None,
        k: int = 5,
        kinds: list[str] | None = None,
    ) -> dict:
        """PIT-filtered history search: beliefs (FTS5 on claim), tombstones
        (FTS5 on summary), episodes (LIKE on mechanism/failure_reason),
        merged by kind with k results per kind. as_of defaults to now.
        """
        as_of = _as_of_or_now(as_of)
        return (memory.research_history(services.db, query, as_of, k, kinds))

    @mcp.tool()
    def memory_recall(query: str, as_of: str | None = None, k: int = 10) -> dict:
        """PIT-filtered belief recall ordered by effective confidence
        (confidence x decay) descending; claims carry supporting/contradicting
        episode pointers. as_of defaults to now.
        """
        as_of = _as_of_or_now(as_of)
        return ({"beliefs": memory.recall(services.db, query, as_of, k), "as_of": as_of})

    @mcp.tool()
    def check_novelty(
        idea_or_code: str,
        family: str | None = None,
        realized_ic: float | None = None,
        predicted_ic: float | None = None,
        predicted_sigma: float | None = None,
        as_of: str | None = None,
    ) -> dict:
        """Bounded novelty/surprise per §7: N = w·(1−max_cos, 1−structural_overlap,
        cell_emptiness) vs the PIT archive (graph nodes + prior hypotheses in
        family); S needs a realized_ic, else null. Degraded mode uses a
        deterministic lexical max_cos proxy (reported as max_cos_basis).
        cell_emptiness uses the real QD archive occupancy when the family is
        in the bin vocabulary (cell_emptiness_basis='qd_archive'), else the
        family-count approximation.
        """
        as_of = _as_of_or_now(as_of)
        return novelty.check_novelty(
            services.db,
            cfg,
            idea_or_code,
            family=family,
            as_of=as_of,
            embedder=services.embedder,
            realized_ic=realized_ic,
            predicted_ic=predicted_ic,
            predicted_sigma=predicted_sigma,
            archive=archive.Archive(services.db, cfg),
        )

    @mcp.tool()
    def explore_methods(
        problem: str,
        k: int = 5,
        mode: str = "balanced",
        as_of: str | None = None,
        seed: int | None = None,
    ) -> dict:
        """Score, sample and diversify method candidates (§6.1 tool 2, §8.1/§8.2).
        Every suggestion carries its score decomposition, family,
        selection_probability and prior_trials; `scoring_mode` is
        'deterministic' without an LLM key. Pass `seed` for a reproducible draw.
        """
        as_of = _as_of_or_now(as_of)
        return explore.explore_methods(
            services.db,
            cfg,
            problem,
            k=k,
            mode=mode,
            as_of=as_of,
            embedder=services.embedder,
            llm=services.llm,
            seed=seed,
        )

    @mcp.tool()
    def memory_promote(
        episode_ids: list[str],
        claim: str,
        confidence: float,
        scope: str | None = None,
        rationale: str | None = None,
        as_of: str | None = None,
        client_id: str | None = None,
    ) -> dict:
        """Promote a belief: contradiction check first (lexical token overlap
        in no-key mode). Auto-promotes iff confidence >= belief_auto_promote_threshold
        and no contradictions; else promotion_status='pending_human'. Idempotent
        on client_id.
        """
        as_of = _as_of_or_now(as_of)
        return (
            memory.promote_belief(
                services.db,
                cfg,
                episode_ids,
                claim,
                scope,
                rationale,
                confidence,
                as_of=as_of,
                client_id=client_id,
                embedder=services.embedder,
            )
        )

    # -- Phase 3: gates & backtest seam (plan §6 tools 11-14, §9, §12.2, §15.2) --

    @mcp.tool()
    def backtest_run(
        episode_id: str,
        spec: dict,
        acknowledge_tombstone: dict | None = None,
        client_id: str | None = None,
    ) -> dict:
        """Dispatch a backtest under the trial governor. Mandatory §12.2
        tombstone pre-check (logged on episodes.tombstone_check; blocked
        dispatches may re-attempt with acknowledge_tombstone={tombstone_id,
        rationale}). A PENDING trial row lands before dispatch; retries on the
        same (episode_id, spec, seed) return the winner's trial without a new
        dispatch (M never double-counts). Returns trial + provider metrics
        (sr/T/skew/kurt net of costs, simulated flag). Idempotent on client_id.
        """
        return gates_service.backtest_run(
            services.db,
            cfg,
            services.provider,
            episode_id,
            spec,
            client_id=client_id,
            acknowledge_tombstone=acknowledge_tombstone,
            embedder=services.embedder,
            llm=services.llm,
        )

    @mcp.tool()
    def significance_gate(episode_id: str) -> dict:
        """Type-A gate (§9.1): DSR = PSR(SR_hat; E[max SR|H0, M_family, skew,
        kurt, T]) — no subtraction — on the episode's net-of-costs metrics,
        with the MinBTL pre-check. Flips the episode to validated iff DSR
        p < dsr_alpha (SR_hat > 0, non-degenerate) and the outcome was not
        failed, else inconclusive. SR is always reported paired with DSR;
        C6 wording: no causal promotion without DAG, no promotion of any kind
        without falsifier.
        """
        return gates_service.significance_gate(services.db, cfg, episode_id)

    @mcp.tool()
    def causal_check(
        episode_id: str,
        spec: dict,
        client_id: str | None = None,
    ) -> dict:
        """Type-B causal probe (§9.3, C6 scoped): validate CausalSpec, delegate
        to the provider (granger/pcmci-*, targets/conditioning/lags), store
        under episodes.metrics.causal, return the provider result verbatim.
        Providers without causal support raise NotImplementedError -> ToolError.
        """
        return gates_service.causal_check(
            services.db, cfg, services.provider, episode_id, spec, client_id=client_id
        )

    @mcp.tool()
    def risk_decompose(
        episode_id: str,
        spec: dict,
        client_id: str | None = None,
    ) -> dict:
        """Book risk decomposition (§15.2): validate BookSpec (exactly one of
        book_id / holdings_uri / returns_uri), delegate to the provider,
        store on episodes.risk_exposures, return verbatim. Providers without
        risk support raise NotImplementedError -> ToolError.
        """
        return gates_service.risk_decompose(
            services.db, cfg, services.provider, episode_id, spec, client_id=client_id
        )

    @mcp.tool()
    def run_summary_ingest(payload: dict) -> dict:
        """Ingest a QuantFlow Tier-2 run summary (plan §15.1 C9/C10): counts,
        failure sketches, skill activations, difficulty signals — not full
        transcripts. The payload must carry schema_version=1 (newer versions
        are rejected with HookSchemaError naming the received version) and is
        validated before any write. A valid summary records a Tier-2
        trajectory event (actor 'quantflow-hook', kind 'hook_summary') and,
        when episode_id resolves, enriches the episode with the
        agent-assembly fingerprint (fingerprint_spec_version defaults to
        'unversioned'), skill versions (default {}), and retrospective
        difficulty signals (ResearchConfig.hook_*_thresholds). QuantFlow-side
        emission is optional — manual log_outcome close-out remains the
        supported path (§15.1 fallback row 1).
        """
        return hooks.ingest_run_summary(services.db, cfg, payload)

    # -- Phase 5: QD archive & exploration (plan §6.1 tool 6, §6.3 tool 19) ----

    @mcp.tool()
    def suggest_next(
        n: int = 3,
        as_of: str | None = None,
        seed: int | None = None,
        gap_report_id: str | None = None,
    ) -> dict:
        """Underexplored-region suggestions (§6.1 tool 6): Thompson-sampled
        family utilities (QD cell emptiness x family posterior mean), softmax
        -> selection probabilities; each suggestion is the family's emptiest
        cell with resolved descriptor names + up to n candidate methods and
        hypotheses from the graph (family match). Pass `seed` for a
        reproducible draw. gap_report_id lands in Phase 7 (consolidation gap
        reports) — passing one raises a named ToolError, it is never silently
        ignored. A trajectory event records the call.
        """
        as_of = _as_of_or_now(as_of)
        return explore.suggest_next(
            services.db, cfg, n=n, as_of=as_of, seed=seed, gap_report_id=gap_report_id
        )

    @mcp.tool()
    def archive_status(as_of: str | None = None) -> dict:
        """Archive + exploration status (§6.3 tool 19): coverage
        (occupied/total cells, per-family), top-N elites by cell score,
        family_M (§9.2 trial counts), spend (§14 spend_summary), family
        Beta posteriors (§8.3), and the predictor error map (per-family mean
        |predicted_ic - realized_ic| over episodes carrying both, §21).
        as_of defaults to now.
        """
        as_of = _as_of_or_now(as_of)
        return archive.status(services.db, cfg, as_of)

    @mcp.tool()
    def report_write(
        hypothesis_id: str | None = None,
        episode_id: str | None = None,
        format: str = "md",
    ) -> dict:
        """Minimal Phase-1 ledger dump (M13): exactly one of hypothesis_id /
        episode_id (N10); md-only (N5). Pure ledger read — no LLM. Includes
        episode/hypothesis fields, trial rows with counts, and costs.
        """
        if (hypothesis_id is None) == (episode_id is None):
            raise ValueError("exactly one of hypothesis_id / episode_id is required (N10)")
        if format != "md":
            raise ValueError("format must be 'md' (md-only until Phase 7 names an HTML renderer)")
        return (
            {
                "format": "md",
                "markdown": _report_markdown(services, hypothesis_id, episode_id),
            }
        )

    return mcp


def _fmt_costs(costs: dict | None) -> str:
    if not costs:
        return "n/a"
    est = " (estimated)" if costs.get("estimated") else ""
    return f"${costs.get('usd', 0.0):.4f} usd, {costs.get('tokens', 0)} tokens{est}"


def _trials_markdown(db: Database, episode_ids: list[str]) -> tuple[list[str], float, int]:
    """Trial rows for the given episodes, plus total usd/tokens across their costs."""
    lines = [
        "| trial_id | episode_id | source | status | family | created | completed |",
        "|---|---|---|---|---|---|---|",
    ]
    total_usd = 0.0
    total_tokens = 0
    for ep_id in episode_ids:
        rows = db.conn.execute(
            "SELECT * FROM trials WHERE episode_id = ? ORDER BY created_at", (ep_id,)
        ).fetchall()
        for t in rows:
            lines.append(
                f"| {t['trial_id']} | {t['episode_id']} | {t['source']} | {t['status']}"
                f" | {t['family']} | {t['created_at']} | {t['completed_at'] or '-'} |"
            )
        ep = db.conn.execute(
            "SELECT costs FROM episodes WHERE episode_id = ?", (ep_id,)
        ).fetchone()
        if ep and ep["costs"]:
            costs = json.loads(ep["costs"])
            total_usd += float(costs.get("usd", 0.0))
            total_tokens += int(costs.get("tokens", 0))
    return lines, total_usd, total_tokens


def _report_markdown(services: Services, hypothesis_id: str | None, episode_id: str | None) -> str:
    db = services.db
    lines: list[str] = ["# Spark ledger report", ""]
    if hypothesis_id:
        hyp = ledger.get_hypothesis(db, hypothesis_id)
        if hyp is None:
            raise ValueError(f"unknown hypothesis_id {hypothesis_id!r}")
        lines += [
            f"## Hypothesis {hypothesis_id}",
            "",
            f"- family: {hyp['research_family']}",
            f"- type: {hyp['hypothesis_type']}",
            f"- status: {hyp['status']}",
            f"- mechanism: {hyp['mechanism']}",
            f"- falsifier: {hyp['falsifier']}",
            f"- predicted_ic: {hyp['predicted_ic']} (sigma {hyp['predicted_sigma']})",
            f"- as_of: {hyp['as_of']}",
            "",
        ]
        ep_rows = db.conn.execute(
            "SELECT episode_id, status, outcome, costs FROM episodes WHERE hypothesis_id = ?"
            " ORDER BY created_at",
            (hypothesis_id,),
        ).fetchall()
        episode_ids = [r["episode_id"] for r in ep_rows]
        lines += ["## Episodes", ""]
        for r in ep_rows:
            costs = json.loads(r["costs"]) if r["costs"] else None
            lines.append(
                f"- `{r['episode_id']}` status={r['status']} outcome={r['outcome'] or '-'}"
                f" costs={_fmt_costs(costs)}"
            )
        lines.append("")
    else:
        assert episode_id is not None  # guaranteed by the exactly-one check in report_write
        ep = ledger.get_episode(db, episode_id)
        if ep is None:
            raise ValueError(f"unknown episode_id {episode_id!r}")
        hyp = ledger.get_hypothesis(db, ep["hypothesis_id"]) or {}
        episode_ids = [episode_id]
        lines += [
            f"## Episode {episode_id}",
            "",
            f"- hypothesis: {ep['hypothesis_id']}",
            f"- family: {ep['research_family']}",
            f"- run_kind: {ep['run_kind']}",
            f"- status: {ep['status']} (outcome {ep['outcome'] or '-'})",
            f"- protocol_hash: {ep['protocol_hash'] or '-'}",
            f"- reconciled: {bool(ep['reconciled'])}",
            f"- failure_reason: {ep['failure_reason'] or '-'}",
            f"- mechanism: {hyp.get('mechanism', '-')}",
            f"- as_of: {ep['as_of']}",
            "",
        ]

    lines.append("## Trials")
    lines.append("")
    trial_lines, total_usd, total_tokens = _trials_markdown(db, episode_ids)
    lines += trial_lines
    counts = db.conn.execute(
        "SELECT source, status, COUNT(*) AS n FROM trials WHERE episode_id IN ("
        + ",".join("?" for _ in episode_ids)
        + ") GROUP BY source, status",
        episode_ids,
    ).fetchall()
    lines += ["", "### Trial counts", ""]
    if counts:
        for c in counts:
            lines.append(f"- {c['source']}/{c['status']}: {c['n']}")
    else:
        lines.append("- none")
    lines += [
        "",
        "## Costs",
        "",
        f"- total: ${total_usd:.4f} usd, {total_tokens} tokens (summed over reported episodes)",
    ]
    spend = spend_summary(db)
    lines.append(
        f"- spend ledger: ${spend['today_usd']:.4f} usd today, ${spend['total_usd']:.4f} usd total"
        " (§14; archive_status exposes the same summary)"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:  # `python -m spark_mcp` entrypoint (stdio MCP)
    create_server().run()


if __name__ == "__main__":
    main()
