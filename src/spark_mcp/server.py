"""MCP server: Phase-0 health tools, Phase-1 ledger & memory tools, Phase-2
method graph & novelty tools, Phase-3 gates & backtest-seam tools, Phase-4
QuantFlow Tier-2 hook seam, Phase-5 QD archive & exploration tools, Phase-6
skill evolution (distill/challenger/replay/approve), Phase-7 analogy,
consolidation & curriculum (find_analogies/combine_methods/consolidate/
generate_exercise), plus artifact evolution (evolution_run/evolution_status).

`create_server(config)` builds a `Services` (config + database handle +
embedder/llm clients + backtest provider) that the tool functions close over —
no module-level mutable state, so tests can inject a temp-db Config and fake
clients. When no Config is passed, `SPARK_CONFIG` (env) is used if set, else
defaults (db_path ~/.spark/spark.db, plan §16).
"""

from __future__ import annotations

import html
import json
import os
from dataclasses import dataclass
from pathlib import Path

from fastmcp import FastMCP

from . import __version__, archive, explore, graph, hooks, ledger, memory, novelty, trajectory
from . import analogy as analogy_service
from . import consolidate as consolidate_service
from . import curriculum as curriculum_service
from . import skills as skills_service
from . import trace as trace_service
from .budgets import spend_summary
from .config import Config
from .errors import ToolError
from .evolve import run as evolution_service
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
        hypothesis_type MUST be one of: causal, descriptive, execution,
        portfolio, predictive, risk.
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
        reproducible draw. gap_report_id consumes a consolidation gap report
        by id (plan §13/G5): families with open gaps get the configured
        utility boost, and the report id is recorded on the trajectory +
        result. A trajectory event records the call.
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
        """Ledger report (M13): exactly one of hypothesis_id / episode_id
        (N10); format 'md' (default) or 'html' (P2-6 built-in single-file
        renderer). Pure ledger read — no LLM. Includes episode/hypothesis
        fields, trial rows with counts, and costs. HTML escapes all
        interpolated text and carries a provenance stamp (generated-at +
        spark-mcp version).
        """
        if (hypothesis_id is None) == (episode_id is None):
            raise ValueError("exactly one of hypothesis_id / episode_id is required (N10)")
        if format == "md":
            return {
                "format": "md",
                "markdown": _report_markdown(services, hypothesis_id, episode_id),
            }
        if format == "html":
            if cfg.research.report_html_renderer != "builtin":
                raise ValueError(
                    "format='html' requires research.report_html_renderer='builtin'"
                    f" (got {cfg.research.report_html_renderer!r});"
                    " set report_html_renderer to 'builtin' to enable the"
                    " built-in single-file HTML renderer (P2-6;"
                    " plan §6.3 tool 21 N5 + §18.1)"
                )
            return {
                "format": "html",
                "html": _report_html(services, hypothesis_id, episode_id),
            }
        raise ValueError(
            f"format must be 'md' or 'html' (got {format!r});"
            f" research.report_html_renderer={cfg.research.report_html_renderer!r}"
            " (plan §6.3 tool 21 N5 + §18.1)"
        )

    # -- Phase 6: skill evolution (plan §6.3 tools 16-18/22, §11, §19 M11) -----

    @mcp.tool()
    def distill_skill(
        episode_ids: list[str],
        name: str,
        role: str | None = None,
        procedure_steps: list[str] | None = None,
        references: list[str] | None = None,
        client_id: str | None = None,
    ) -> dict:
        """Draft candidate from validated trajectories (plan §6.3 tool 16,
        C10): every episode must exist and carry trajectory evidence (Tier-1
        events and/or Tier-2 hook summaries). Writes a skills row status=
        'draft' at the next free version; body_uri is the registry-internal
        spark://skills/{name}/{version} pointer (filesystem deployment is
        forbidden, §3.3); the procedure body lives in the lineage JSON.
        Credit is assigned at promotion. Idempotent on client_id.
        """
        if procedure_steps is None:
            raise ValueError("procedure_steps is required")
        return skills_service.distill_skill(
            services.db,
            cfg,
            episode_ids,
            name,
            role,
            procedure_steps,
            references=references,
            client_id=client_id,
        )

    @mcp.tool()
    def propose_skill_challenger(
        skill_name: str,
        op: str,
        source_episode_ids: list[str] | None = None,
        client_id: str | None = None,
    ) -> dict:
        """Challenger via mutation operators / mini-GEPA (plan §6.3 tool 17).
        op in {repair, split, merge, paraphrase, gepa} (deterministic
        definitions in spark_mcp.skills). Writes the challenger skills row
        (status='challenger', version = next free for the name) plus a pending
        skill_candidates row awaiting run_replay evidence. Idempotent on
        client_id.
        """
        return skills_service.propose_skill_challenger(
            services.db,
            cfg,
            skill_name,
            op,
            source_episode_ids=source_episode_ids,
            client_id=client_id,
            llm=services.llm,
        )

    @mcp.tool()
    def run_replay(
        skill_name: str | None = None,
        suite: str | None = None,
        task_ids: list[str] | None = None,
        seed: int | None = None,
    ) -> dict:
        """Replay-suite verdicts (plan §6.3 tool 18, §11): scores skill
        versions against the seeded quant-research fixtures and reports the
        held-in/held-out split with heldout_gap in every verdict (anti-
        gaming alarm, §21). Without an LLM the score is deterministic
        keyword/rubric coverage; with one, an LLM judge 0-1 per rubric item,
        cached per (skill_version, task_id, model_id). Scores land on the
        newest pending candidate for each skill; with no pending candidate
        the latest version is scored as a regression run.
        """
        return skills_service.run_replay(
            services.db,
            cfg,
            skill_name=skill_name,
            suite=suite,
            task_ids=task_ids,
            seed=seed,
            llm=services.llm,
        )

    @mcp.tool()
    def approve(
        target_type: str,
        target_id: str,
        decision: str,
        rationale: str,
        decided_by: str = "human:operator",
    ) -> dict:
        """Human gate (plan §6.3 tool 22, M11): target_type in {belief, skill,
        adoption, evolution}, decision in {approved, rejected}. belief
        resolves a pending_human promotion (§12.3); skill ratifies or vetoes
        a staged champion (veto rolls back to the rollback_target, §11);
        adoption records the human authorization for taking a champion into
        a live QuantFlow book (the book edit itself is out of band);
        evolution ratifies or vetoes an evolution line head (veto restores
        the rollback head). Idempotent on (target_type, target_id, decision).
        """
        return skills_service.approve(
            services.db,
            cfg,
            target_type,
            target_id,
            decision,
            rationale,
            decided_by=decided_by,
        )

    @mcp.tool()
    def decide_skill_promotion(
        skill_name: str,
        candidate_id: str,
        decision: str,
        rationale: str,
        decided_by: str = "gate",
    ) -> dict:
        """The promotion gate (plan §11), reachable from the product surface
        (Phase-6 re-review blocker B1). decision in {promoted, rejected,
        auto}; 'auto' applies the documented criterion — beat the incumbent
        champion's held-in mean by more than replay_promotion_margin (first
        promotion: exceed replay_promotion_floor by more than the margin;
        ties never promote) with heldout_gap under the alarm. Requires FULL
        replay-suite evidence on the candidate. Gate decisions write an
        approvals row (decided_by='gate') as the human-notify/veto surface,
        then attempt the Gateway write (C7); a block stages the champion in
        Spark.
        """
        return skills_service.decide_promotion(
            services.db,
            cfg,
            skill_name,
            candidate_id,
            decision,
            rationale,
            decided_by,
        )

    @mcp.tool()
    def retire_skill(
        name: str,
        version: int,
        rationale: str,
    ) -> dict:
        """Retire a skill version (plan §11 lifecycle). A retired champion
        names the newest lower fallback version as rollback_target so the
        veto/restore path always exists.
        """
        return skills_service.retire_skill(services.db, name, version, rationale)

    # -- Artifact evolution (evolution upgrade plan §2.E, §3.3) ------------------

    @mcp.tool()
    def evolution_run(
        family: str,
        seed_hypothesis_id: str | None = None,
        max_rounds: int | None = None,
        seed: int = 0,
        client_id: str | None = None,
    ) -> dict:
        """Bounded RD-shaped evolution run over research artifacts (hypothesis
        → spec variant → executed backtest → DSR significance gate →
        execution-grounded fitness → computed promotion → narrated trace).
        Selection is always computed from numbers (trial governor, DSR gate,
        strict DSR/SR margins, C6 blockers); the LLM may narrate but never
        decides. Promotions move the line head (prior head kept as rollback)
        and write a gate approvals row as the human-veto surface — veto
        post-run via approve(target_type='evolution'); the run itself is a
        closed loop and never blocks on input. Without an LLM the run needs a
        seed hypothesis or an already-headed line (root synthesis from
        nothing needs a model). Deterministic given (seed, caps, provider):
        reruns agree on decisions and heads by hypothesis content + episode
        metrics (ULID keys and timestamps excluded). Stops on max_rounds,
        consecutive rejections, kill-rule death, or budget exhaustion
        (fail_closed budgets close the run with status stopped_budget rather
        than raising). Refuses unknown seeds and cross-family seeds.
        Idempotent on client_id.
        """
        return evolution_service.run_evolution(
            services.db,
            cfg,
            services.provider,
            family=family,
            seed_hypothesis_id=seed_hypothesis_id,
            max_rounds=max_rounds,
            seed=seed,
            llm=services.llm,
            client_id=client_id,
        )

    @mcp.tool()
    def evolution_status(run_id: str) -> dict:
        """Read-only evolution run report: the run record, per-round feedback
        (decisions with DSR/SR, model, prompt sha), the line's current head +
        rollback, and the computed SOTA ancestor. Unknown runs are refused.
        """
        return evolution_service.evolution_status(services.db, run_id)

    # -- Phase 7: analogy, consolidation, curriculum (plan §6.1 tools 3-4,
    # §6.3 tools 20-21, §12.3, §13, §18.1) ------------------------------------

    @mcp.tool()
    def find_analogies(
        target_problem: str,
        mech_tags: list[str],
        k: int = 5,
        as_of: str | None = None,
        client_id: str | None = None,
    ) -> dict:
        """SME-lite structure-mapping analogy search (§6.1 tool 4): bases from
        the method graph whose mech-tag predicate structure maps onto the
        target's, each with a Gentner-style systematicity score (formula +
        decomposition in spark_mcp.analogy), the shared predicate structure,
        and a falsifiable financial translation (LLM-generated, cached per
        (base, target_class, model_id); documented degraded template without
        an LLM). Mappings below the configured systematicity threshold are
        refused (counted in `refusals`; named refusal when none pass). Each
        call consumes one analogy trial against the §8.3 budget cap. A
        trajectory event records the call. Idempotent on client_id.
        """
        as_of = _as_of_or_now(as_of)
        return analogy_service.find_analogies(
            services.db,
            cfg,
            target_problem,
            mech_tags,
            k=k,
            as_of=as_of,
            llm=services.llm,
            client_id=client_id,
        )

    @mcp.tool()
    def combine_methods(
        method_ids: list[str],
        target_problem: str,
        as_of: str | None = None,
        client_id: str | None = None,
    ) -> dict:
        """Compose 2-3 candidate compositions from method graph nodes (§6.1
        tool 3). Grammar: pipeline (A's mech-compound output feeds B's input),
        ensemble (same-family parallel), residualizer (B applied to A's
        residuals via a visible DIAGNOSES/FAILS_WHEN edge). Every candidate
        carries structural-fit notes (shared predicates, family bridge,
        expected interactions), the composed mech_tags set, and a one-line
        rationale (LLM-cached or degraded template). No returns computed.
        Idempotent on client_id.
        """
        as_of = _as_of_or_now(as_of)
        return analogy_service.combine_methods(
            services.db,
            cfg,
            method_ids,
            target_problem,
            as_of=as_of,
            client_id=client_id,
            llm=services.llm,
        )

    @mcp.tool()
    def consolidate(
        scope: str | None = None,
        as_of: str | None = None,
        since: str | None = None,
    ) -> dict:
        """On-demand consolidation pass (plan §6.3 tool 20, §13): applies
        belief decay (§12.3 formula, writes beliefs.decay), lists
        validation-SLO breaches, and emits gap entries from the archive
        (empty families, killed families, predictor-error outliers), family
        posterior review, and the skill pipeline (stale pending candidates).
        LLM gap annotations are capped per-run by
        budgets.per_consolidation_usd — the pass stops mid-way with
        budget_exhausted=True (warn) or refuses before any write
        (fail_closed). Persists a consolidation_runs row + an immutable gap
        report artifact (spark://consolidation/{report_id}) consumable by
        suggest_next (gap_report_id) and generate_exercise. Same service
        function as `spark consolidate` (N13). A trajectory event with actor
        'consolidator' records the run. Also alarms when the replay fixture
        suite counts below replay_min_fixtures (alarm only).
        """
        as_of = _as_of_or_now(as_of)
        return consolidate_service.consolidate(
            services.db, cfg, scope=scope, as_of=as_of, since=since, llm=services.llm
        )

    @mcp.tool()
    def generate_exercise(
        gap_report_id: str | None = None,
        seed: int | None = None,
    ) -> dict:
        """Minimal curriculum (plan §18.1 Phase 7): synthesize one exercise
        from a real consolidation gap (a fresh consolidation pass runs when
        no gap_report_id is given) and replay-score the current champion
        skill's procedure_steps against the replay fixture mapped from the
        gap family (the deterministic skills.run_replay scorer). Deferral is
        explicit ({deferred: true, reason}) when there is no usable gap or
        fixture mapping. A trajectory event records the call.
        """
        return curriculum_service.generate_exercise(
            services.db, cfg, gap_report_id=gap_report_id, seed=seed
        )

    # -- P1-5 trace views + P2-3 debate (read-only trace tools; the debate
    # block is persisted audit text only — significance_gate/decide never
    # read it) -------------------------------------------------------------

    @mcp.tool()
    def trace_hypothesis(hypothesis_id: str) -> dict:
        """Ordered trace for one hypothesis (P1-5): the hypothesis root plus
        its trial episodes and narrated evolution feedback, merged append-only
        by created_at. Pure ledger read — no LLM. Unknown hypotheses are
        refused. Returns {hypothesis_id, entries: [...]} with one concise
        view dict per entry (kind in {hypothesis, experiment, feedback}).
        """
        views = trace_service.trace_for_hypothesis(services.db, hypothesis_id)
        return {
            "hypothesis_id": hypothesis_id,
            "entries": [v.model_dump() for v in views],
        }

    @mcp.tool()
    def sota_status() -> dict:
        """Pointer to the best gated trial in the ledger (P1-5), or
        {sota: None} on an empty/ungated ledger. Pure ledger read — no LLM.
        Ranks stored gate verdicts (pass is True) by stored DSR descending,
        earliest episode_id breaking ties.
        """
        return {"sota": trace_service.sota_pointer(services.db)}

    @mcp.tool()
    def debate_evidence(
        hypothesis_id: str,
        client_id: str | None = None,
    ) -> dict:
        """Adversarial audit debate over one hypothesis' ledger record (P2-3):
        exactly two bounded LLM passes — a prosecution pass arguing the record
        fails, then a defence pass answering — over the concise trace
        (truncated to 2000 chars). The block is persisted as a trajectory
        event (run_id=hypothesis_id, kind='debate') and returned; it is audit
        text only and is never read by significance_gate/decide. Refuses
        unknown hypotheses, and refuses (never fabricates) in degraded mode
        with no LLM configured. client_id is recorded on the event.
        """
        llm = services.llm
        if llm is None:
            raise ToolError(
                "debate_evidence refused: degraded mode (no LLM configured);"
                " refusing rather than fabricating a debate"
            )
        views = trace_service.trace_for_hypothesis(services.db, hypothesis_id)
        record = json.dumps(
            [v.model_dump() for v in views], sort_keys=True, default=str
        )[:2000]
        prosecution = llm.complete(
            "[PROSECUTION PASS] Argue, from the ledger record below only, that"
            f" hypothesis {hypothesis_id} should NOT be trusted. Ledger record:"
            f" {record}"
        ).strip()
        defence = llm.complete(
            "[DEFENCE PASS] Answer the prosecution and argue, from the ledger"
            f" record below only, that hypothesis {hypothesis_id} SHOULD be"
            f" trusted. Ledger record: {record}"
        ).strip()
        block = {
            "hypothesis_id": hypothesis_id,
            "prosecution": prosecution,
            "defence": defence,
            "model_id": llm.model_id(),
            "record_chars": len(record),
            "client_id": client_id,
        }
        trajectory.record(
            db=services.db,
            run_id=hypothesis_id,
            actor="spark",
            kind="debate",
            payload={"tool": "debate_evidence", **block},
        )
        return block

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


def _esc(value: object) -> str:
    """Escape one interpolated value for HTML text/attribute context."""
    if value is None:
        return "-"
    return html.escape(str(value), quote=True)


def _trials_html(db: Database, episode_ids: list[str]) -> tuple[str, float, int]:
    """Trial rows as an HTML table, plus total usd/tokens (mirrors _trials_markdown)."""
    rows_html: list[str] = []
    total_usd = 0.0
    total_tokens = 0
    for ep_id in episode_ids:
        rows = db.conn.execute(
            "SELECT * FROM trials WHERE episode_id = ? ORDER BY created_at", (ep_id,)
        ).fetchall()
        for t in rows:
            rows_html.append(
                "<tr>"
                f"<td>{_esc(t['trial_id'])}</td>"
                f"<td>{_esc(t['episode_id'])}</td>"
                f"<td>{_esc(t['source'])}</td>"
                f"<td>{_esc(t['status'])}</td>"
                f"<td>{_esc(t['family'])}</td>"
                f"<td>{_esc(t['created_at'])}</td>"
                f"<td>{_esc(t['completed_at'])}</td>"
                "</tr>"
            )
        ep = db.conn.execute(
            "SELECT costs FROM episodes WHERE episode_id = ?", (ep_id,)
        ).fetchone()
        if ep and ep["costs"]:
            costs = json.loads(ep["costs"])
            total_usd += float(costs.get("usd", 0.0))
            total_tokens += int(costs.get("tokens", 0))
    if rows_html:
        body = "".join(rows_html)
    else:
        body = '<tr><td colspan="7">none</td></tr>'
    table = (
        "<table><thead><tr>"
        "<th>trial_id</th><th>episode_id</th><th>source</th><th>status</th>"
        "<th>family</th><th>created</th><th>completed</th>"
        f"</tr></thead><tbody>{body}</tbody></table>"
    )
    return table, total_usd, total_tokens


def _report_html(services: Services, hypothesis_id: str | None, episode_id: str | None) -> str:
    """Minimal built-in single-file HTML renderer for report_write (P2-6).

    Mirrors _report_markdown's ledger reads; every interpolated value is
    escaped via _esc/html.escape. The document is self-contained (inline
    CSS, no external resources) and carries a provenance stamp with the
    generation timestamp and the spark-mcp version.
    """
    db = services.db
    generated_at = now_iso()
    parts: list[str] = []
    episode_ids: list[str] = []
    if hypothesis_id:
        hyp = ledger.get_hypothesis(db, hypothesis_id)
        if hyp is None:
            raise ValueError(f"unknown hypothesis_id {hypothesis_id!r}")
        title = f"Spark ledger report — Hypothesis {hypothesis_id}"
        parts.append(f"<h2>Hypothesis {_esc(hypothesis_id)}</h2>")
        parts.append(
            "<ul>"
            f"<li>family: {_esc(hyp['research_family'])}</li>"
            f"<li>type: {_esc(hyp['hypothesis_type'])}</li>"
            f"<li>status: {_esc(hyp['status'])}</li>"
            f"<li>mechanism: {_esc(hyp['mechanism'])}</li>"
            f"<li>falsifier: {_esc(hyp['falsifier'])}</li>"
            f"<li>predicted_ic: {_esc(hyp['predicted_ic'])}"
            f" (sigma {_esc(hyp['predicted_sigma'])})</li>"
            f"<li>as_of: {_esc(hyp['as_of'])}</li>"
            "</ul>"
        )
        ep_rows = db.conn.execute(
            "SELECT episode_id, status, outcome, costs FROM episodes WHERE hypothesis_id = ?"
            " ORDER BY created_at",
            (hypothesis_id,),
        ).fetchall()
        episode_ids = [r["episode_id"] for r in ep_rows]
        parts.append("<h2>Episodes</h2>")
        if ep_rows:
            items = []
            for r in ep_rows:
                costs = json.loads(r["costs"]) if r["costs"] else None
                items.append(
                    f"<li><code>{_esc(r['episode_id'])}</code>"
                    f" status={_esc(r['status'])}"
                    f" outcome={_esc(r['outcome'])}"
                    f" costs={_esc(_fmt_costs(costs))}</li>"
                )
            parts.append("<ul>" + "".join(items) + "</ul>")
        else:
            parts.append("<ul><li>none</li></ul>")
    else:
        assert episode_id is not None  # guaranteed by the exactly-one check in report_write
        ep = ledger.get_episode(db, episode_id)
        if ep is None:
            raise ValueError(f"unknown episode_id {episode_id!r}")
        hyp = ledger.get_hypothesis(db, ep["hypothesis_id"]) or {}
        episode_ids = [episode_id]
        title = f"Spark ledger report — Episode {episode_id}"
        parts.append(f"<h2>Episode {_esc(episode_id)}</h2>")
        parts.append(
            "<ul>"
            f"<li>hypothesis: {_esc(ep['hypothesis_id'])}</li>"
            f"<li>family: {_esc(ep['research_family'])}</li>"
            f"<li>run_kind: {_esc(ep['run_kind'])}</li>"
            f"<li>status: {_esc(ep['status'])}"
            f" (outcome {_esc(ep['outcome'])})</li>"
            f"<li>protocol_hash: {_esc(ep['protocol_hash'])}</li>"
            f"<li>reconciled: {_esc(bool(ep['reconciled']))}</li>"
            f"<li>failure_reason: {_esc(ep['failure_reason'])}</li>"
            f"<li>mechanism: {_esc(hyp.get('mechanism'))}</li>"
            f"<li>as_of: {_esc(ep['as_of'])}</li>"
            "</ul>"
        )

    parts.append("<h2>Trials</h2>")
    trials_table, total_usd, total_tokens = _trials_html(db, episode_ids)
    parts.append(trials_table)
    if episode_ids:
        counts = db.conn.execute(
            "SELECT source, status, COUNT(*) AS n FROM trials WHERE episode_id IN ("
            + ",".join("?" for _ in episode_ids)
            + ") GROUP BY source, status",
            episode_ids,
        ).fetchall()
    else:
        counts = []
    parts.append("<h3>Trial counts</h3>")
    if counts:
        items = "".join(
            f"<li>{_esc(c['source'])}/{_esc(c['status'])}: {_esc(c['n'])}</li>" for c in counts
        )
        parts.append(f"<ul>{items}</ul>")
    else:
        parts.append("<ul><li>none</li></ul>")
    parts.append("<h2>Costs</h2>")
    spend = spend_summary(db)
    total_str = f"{total_usd:.4f}"
    today_str = f"{spend['today_usd']:.4f}"
    total_spend_str = f"{spend['total_usd']:.4f}"
    parts.append(
        "<ul>"
        f"<li>total: ${_esc(total_str)} usd, {_esc(total_tokens)} tokens"
        " (summed over reported episodes)</li>"
        f"<li>spend ledger: ${_esc(today_str)} usd today,"
        f" ${_esc(total_spend_str)} usd total"
        " (§14; archive_status exposes the same summary)</li>"
        "</ul>"
    )
    provenance = (
        f"Generated at {_esc(generated_at)} · spark-mcp {_esc(__version__)}"
    )
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        f"<title>{_esc(title)}</title>\n"
        "<style>"
        "body{font-family:system-ui,sans-serif;margin:2rem;line-height:1.5;color:#111}"
        "table{border-collapse:collapse;margin:1rem 0}"
        "th,td{border:1px solid #ccc;padding:.3rem .6rem;text-align:left;font-size:.9rem}"
        "th{background:#f3f3f3}code{font-family:ui-monospace,monospace}"
        ".provenance{color:#555;font-size:.85rem;border-top:1px solid #ddd;"
        "margin-top:2rem;padding-top:.5rem}"
        "</style>\n</head>\n<body>\n"
        "<h1>Spark ledger report</h1>\n"
        + "\n".join(parts)
        + f'\n<footer class="provenance">{provenance}</footer>\n'
        "</body>\n</html>\n"
    )


def main() -> None:  # `python -m spark_mcp` entrypoint (stdio MCP)
    create_server().run()


if __name__ == "__main__":
    main()
