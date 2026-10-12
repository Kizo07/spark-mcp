"""RD-shaped outer loop driver + status (plan §2.E1, §3.3).

One bounded run: select parent → vary hypothesis (level 1) → vary spec
(level 2) → `backtest_run` (trial governor + budgets) → `significance_gate`
(DSR, computed) → execution-grounded fitness → computed promotion → narrated
feedback row → line-head update. The LLM may narrate (observations,
evaluation, next-hypothesis seed) but never decides: narration is recorded
after the decision and is never an input to `promote.decide` (constraint 2).

Closed loop (constraint 3): the driver never calls `approve` and never
blocks on input — veto applies post-run at the line head via
`approve(target_type='evolution')`. On promotion the driver performs the E3
line-head update itself (prior head → `rollback_*` via `advance_head`, plus
a gate `approvals` INSERT as the human-notify surface, mirroring
`decide_promotion` M11).

Determinism (constraint 4): one `random.Random(seed)` threads through all
rounds; spec seeds derive from `(run seed, round, attempt)` by hash (no rng
state consumed) — never from ULID hypothesis ids, so reruns are
byte-identical in decisions/heads by content+metrics (ULID keys and
timestamps excluded by column). Providers consume only the spec seed.

Stopping: `max_rounds`, consecutive computed rejections ≥
`evolution_max_consecutive_rejections`, kill-rule family death, and budget
exhaustion — a fail_closed budget raise inside `backtest_run` closes the run
gracefully with status `stopped_budget` instead of propagating.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from pydantic import ValidationError

from .. import ledger, trajectory
from ..config import Config
from ..errors import ToolError
from ..gates import service as gates
from ..ids import new_approval_id
from ..llm import LLMClient
from ..storage import Database
from ..timeutil import now_iso
from . import lineage
from .fitness import FitnessReport, fitness_from_episode
from .promote import PromotionDecision, decide
from .render import cache_render, empty_render, render_ancestors
from .selectors import select_parent
from .variation import vary_hypothesis, vary_spec

#: Level-2 attempts per round before escalating to a fresh level-1 variation
#: next round (plan §2.D: bounded retries, then escalate).
_LEVEL2_ATTEMPTS = 2

#: Substrings identifying fail_closed budget raises (budgets.enforce_*).
_BUDGET_MARKER = "budget exceeded"

#: Substrings identifying tombstone pre-check blocks (not retried: new params
#: would not change the mechanism/scope match that blocked dispatch).
_TOMBSTONE_MARKER = "tombstone pre-check blocked dispatch"


def _spec_seed(run_seed: int, round_no: int, attempt: int) -> int:
    """Hash-derived spec seed from (run seed, round, attempt).

    Consumes no rng state (mirrors the `_op_gepa` content-seed precedent)
    and — critically — never derives from ULID hypothesis/episode ids, so
    two runs from the same seed dispatch identical spec seeds. Masked to 63
    bits: the seed lands in a SQLite INTEGER column (signed 64-bit).
    """
    material = f"evolution|{run_seed}|{round_no}|{attempt}"
    digest = hashlib.sha1(material.encode()).digest()  # noqa: S324 - seed derivation
    return int.from_bytes(digest[:8], "little") & 0x7FFF_FFFF_FFFF_FFFF


def _family_killed(db: Database, family: str) -> bool:
    row = db.conn.execute(
        "SELECT killed FROM exploration_state WHERE family = ?", (family,)
    ).fetchone()
    return row is not None and bool(row["killed"])


def _latest_feedback_for(db: Database, hypothesis_id: str) -> dict | None:
    row = db.conn.execute(
        "SELECT * FROM evolution_feedback WHERE hypothesis_id = ?"
        " ORDER BY rowid DESC LIMIT 1",
        (hypothesis_id,),
    ).fetchone()
    return dict(row) if row else None


def _narrate(
    llm: LLMClient | None,
    round_no: int,
    fitness: FitnessReport,
    decision: PromotionDecision,
    gate_pass: bool,
) -> tuple[dict, str | None]:
    """Feedback narration: one LLM call when available, else a computed template.

    Both paths use IS numbers only (fitness fields + the computed decision).
    The LLM must reply with a single JSON object
    `{observations, hypothesis_evaluation, new_hypothesis}`; any failure —
    no LLM, raise, unparsable reply, wrong shape — degrades to the
    deterministic template (model None). Returns (narration, model_id).
    """
    template = {
        "observations": (
            f"Round {round_no}: SR {fitness.sr:.4f}, DSR prob"
            f" {fitness.dsr_prob:.4f} (p={fitness.p_value:.4f}, M={fitness.M_family},"
            f" T={fitness.T}); gate pass={gate_pass}."
        ),
        "hypothesis_evaluation": decision.reason,
        "new_hypothesis": (
            f"Exploit the new head {fitness.episode_id}."
            if decision.promote
            else "Vary away from this candidate next round."
        ),
    }
    if llm is None:
        return template, None
    prompt = (
        "You are narrating one evolution round. Reply with a single JSON object"
        " {\"observations\": ..., \"hypothesis_evaluation\": ...,"
        " \"new_hypothesis\": ...} (short strings). Do not decide anything:"
        " the promotion decision below is already computed and final.\n"
        f"Round: {round_no}\n"
        f"SR: {fitness.sr:.4f}\nDSR prob: {fitness.dsr_prob:.4f}\n"
        f"p_value: {fitness.p_value:.4f}\nM_family: {fitness.M_family}\n"
        f"T: {fitness.T}\nGate pass: {gate_pass}\nDecision: {decision.reason}"
    )
    try:
        parsed = json.loads(llm.complete(prompt))
        if (
            isinstance(parsed, dict)
            and all(isinstance(parsed.get(k), str) for k in template)
        ):
            return {k: parsed[k] for k in template}, llm.model_id()
    except Exception:
        pass
    return template, None


def _effect_lookup(db: Database, client_id: str) -> dict | None:
    row = db.conn.execute(
        "SELECT result FROM tool_effects WHERE client_id = ? AND tool = 'evolution_run'",
        (client_id,),
    ).fetchone()
    return json.loads(row["result"]) if row else None


def _record_effect(db: Database, client_id: str, run_id: str, result: dict) -> None:
    db.execute(
        "INSERT OR IGNORE INTO tool_effects (client_id, tool, subject_id, result, created_at)"
        " VALUES (?, 'evolution_run', ?, ?, ?)",
        (client_id, run_id, json.dumps(result, sort_keys=True, default=str), now_iso()),
    )


def _resolve_line(db: Database, family: str, seed_hypothesis_id: str | None) -> dict:
    if seed_hypothesis_id is not None:
        seed = ledger.get_hypothesis(db, seed_hypothesis_id)
        if seed is None:
            raise ToolError(f"unknown seed_hypothesis_id {seed_hypothesis_id!r}")
        if seed.get("research_family") != family:
            raise ToolError(
                f"seed hypothesis {seed_hypothesis_id} belongs to family"
                f" {seed.get('research_family')!r}, not {family!r}"
            )
        # Branch point: a fresh line headed at the seed (no incumbent
        # episode yet, so the first promotion uses the floor rule).
        return lineage.create_line(db, family, head_hypothesis_id=seed_hypothesis_id)
    active = lineage.active_line(db, family)
    if active is not None:
        return active
    return lineage.create_line(db, family)


def run_evolution(
    db: Database,
    cfg: Config,
    provider: Any,
    *,
    family: str,
    seed_hypothesis_id: str | None = None,
    max_rounds: int | None = None,
    seed: int = 0,
    llm: LLMClient | None = None,
    client_id: str | None = None,
) -> dict:
    """Run one bounded evolution loop for a family (plan §2.E1).

    Returns `{run_id, line_id, family, seed, status, rounds, decisions,
    head, rollback, consecutive_rejections}`. `status` is `finished` or one
    of `stopped_consecutive_rejections | stopped_kill_rule | stopped_budget`.
    Idempotent on `client_id` via `tool_effects` (repo convention): a replay
    returns the recorded summary with no side effects.
    """
    if not isinstance(family, str) or not family.strip():
        raise ToolError("family is required")
    if client_id:
        recorded = _effect_lookup(db, client_id)
        if recorded is not None:
            return recorded
    if max_rounds is None:
        max_rounds = int(cfg.research.evolution_max_rounds)
    if isinstance(max_rounds, bool) or not isinstance(max_rounds, int) or max_rounds < 1:
        raise ToolError(f"max_rounds must be an int >= 1; got {max_rounds!r}")

    line = _resolve_line(db, family, seed_hypothesis_id)
    line_id = line["line_id"]
    rng = random.Random(seed)
    run = lineage.create_run(
        db, line_id, seed, caps={"max_rounds": max_rounds, "family": family}
    )
    run_id = run["run_id"]
    max_consec = int(cfg.research.evolution_max_consecutive_rejections)

    decisions: list[dict] = []
    consecutive = 0
    status = "finished"

    def _fail_round(round_no: int, stage: str, error: Exception, extra: dict) -> None:
        decisions.append(
            {"round": round_no, "outcome": "failed", "stage": stage,
             "error": str(error), **extra}
        )
        trajectory.record(
            db, run_id, "spark", "tool_call",
            {"tool": "evolution_run", "round": round_no, "outcome": "failed",
             "stage": stage, "error": str(error)[:300]},
        )

    for round_no in range(max_rounds):
        if _family_killed(db, family):
            status = "stopped_kill_rule"
            break

        selection = select_parent(db, cfg, line_id, rng)
        if selection.policy == "sota_jump_new_root":
            # Selectors never write: the driver creates the new line here and
            # moves the run onto it, so status queries follow the live line.
            line = lineage.create_line(db, family)
            line_id = line["line_id"]
            db.execute(
                "UPDATE evolution_runs SET line_id = ? WHERE run_id = ?",
                (line_id, run_id),
            )
            parent: dict | None = None
            parent_ids: list[str] = []
            parent_episode: dict | None = None
        else:
            parent = (
                ledger.get_hypothesis(db, selection.hypothesis_id)
                if selection.hypothesis_id is not None
                else None
            )
            parent_ids = [selection.hypothesis_id] if selection.hypothesis_id else []
            parent_episode = (
                ledger.get_episode(db, selection.episode_id)
                if selection.episode_id is not None
                else None
            )

        if parent is not None:
            render = render_ancestors(db, cfg, parent["hypothesis_id"])
        else:
            render = empty_render()
        cache_render(db, render, llm.model_id() if llm is not None else None)
        feedback_ctx = (
            _latest_feedback_for(db, parent["hypothesis_id"]) if parent else None
        )

        # -- Level 1: successor hypothesis ------------------------------------
        try:
            body = vary_hypothesis(
                db, cfg, parent, render.text, feedback_ctx, rng, llm
            )
        except (ToolError, ValueError) as exc:
            # Parentless root synthesis without an LLM lands here (recorded
            # failure, no silent retry).
            _fail_round(round_no, "vary_hypothesis", exc,
                        {"policy": selection.policy, "reason": selection.reason})
            consecutive += 1
            if consecutive >= max_consec:
                status = "stopped_consecutive_rejections"
                break
            continue
        try:
            hyp_id = ledger.propose_hypothesis(
                db, cfg, parent_hypothesis_ids=parent_ids, **body
            )["hypothesis_id"]
        except (ValueError, ToolError) as exc:
            # C6 refusal is the schema gate: recorded as a failed variation.
            _fail_round(round_no, "propose_refused", exc,
                        {"policy": selection.policy, "reason": selection.reason})
            consecutive += 1
            if consecutive >= max_consec:
                status = "stopped_consecutive_rejections"
                break
            continue
        hyp_row = ledger.get_hypothesis(db, hyp_id)
        assert hyp_row is not None

        # -- Level 2: spec variant + dispatch (bounded retries) ---------------
        episode_id: str | None = None
        attempt_errors: list[str] = []
        for attempt in range(_LEVEL2_ATTEMPTS):
            try:
                spec = vary_spec(db, cfg, hyp_row, parent_episode, round_no, rng)
            except (ValueError, ValidationError) as exc:
                attempt_errors.append(f"vary_spec: {exc}")
                continue
            spec = spec.model_copy(update={"seed": _spec_seed(seed, round_no, attempt)})
            spec_dict = spec.model_dump(mode="json")
            ep_id = ledger.register_hypothesis(db, cfg, hyp_id, spec_dict, spec.seed)[
                "episode_id"
            ]
            try:
                gates.backtest_run(db, cfg, provider, ep_id, spec_dict)
            except ToolError as exc:
                attempt_errors.append(f"backtest: {exc}")
                if _BUDGET_MARKER in str(exc):
                    status = "stopped_budget"
                    break
                if _TOMBSTONE_MARKER in str(exc):
                    break  # no retry: escalate to a fresh level-1 next round
                continue  # provider/infra error: one more level-2 attempt
            episode_id = ep_id
            break
        if status == "stopped_budget":
            _fail_round(round_no, "budget", ToolError(attempt_errors[-1]),
                        {"hypothesis_id": hyp_id, "policy": selection.policy})
            break
        if episode_id is None:
            _fail_round(round_no, "backtest_failed",
                        ToolError("; ".join(attempt_errors) or "level-2 retries exhausted"),
                        {"hypothesis_id": hyp_id, "policy": selection.policy})
            consecutive += 1
            if consecutive >= max_consec:
                status = "stopped_consecutive_rejections"
                break
            continue

        # -- Gate → fitness → computed promotion -------------------------------
        try:
            gates.significance_gate(db, cfg, episode_id)
        except ToolError as exc:
            _fail_round(round_no, "gate_failed", exc,
                        {"hypothesis_id": hyp_id, "episode_id": episode_id,
                         "policy": selection.policy})
            consecutive += 1
            if consecutive >= max_consec:
                status = "stopped_consecutive_rejections"
                break
            continue
        try:
            fitness = fitness_from_episode(db, cfg, episode_id)
        except ToolError as exc:
            _fail_round(round_no, "fitness_refused", exc,
                        {"hypothesis_id": hyp_id, "episode_id": episode_id,
                         "policy": selection.policy})
            consecutive += 1
            if consecutive >= max_consec:
                status = "stopped_consecutive_rejections"
                break
            continue
        incumbent_ep = (lineage.get_line(db, line_id) or {}).get("head_episode_id")
        decision = decide(db, cfg, episode_id, incumbent_ep)
        episode = ledger.get_episode(db, episode_id) or {}
        gate_pass = bool((episode.get("corrected_metrics") or {}).get("pass"))
        narration, model_id = _narrate(llm, round_no, fitness, decision, gate_pass)
        lineage.record_feedback(
            db, run_id, hyp_id, episode_id, parent_ids, narration, fitness,
            decision.reason, model_id, render.sha,
        )

        if decision.promote:
            # E3 line-head update: prior head → rollback_* (same transaction
            # inside advance_head), then the gate-notify approvals row — the
            # human-veto surface, mirroring decide_promotion M11.
            line = lineage.advance_head(db, line_id, hyp_id, episode_id)
            db.execute(
                "INSERT INTO approvals ("
                " approval_id, target_type, target_id, decision, rationale,"
                " decided_by, created_at"
                ") VALUES (?, 'evolution', ?, 'approved', ?, 'gate', ?)",
                (new_approval_id(), line_id, decision.reason, now_iso()),
            )
            consecutive = 0
        else:
            consecutive += 1
        decisions.append(
            {"round": round_no, "outcome": "decided", "policy": selection.policy,
             "reason": selection.reason, "hypothesis_id": hyp_id,
             "episode_id": episode_id, "promote": decision.promote,
             "decision": decision.reason, "dsr_prob": fitness.dsr_prob,
             "sr": fitness.sr, "dsr_margin": decision.dsr_margin,
             "sr_margin": decision.sr_margin}
        )
        trajectory.record(
            db, run_id, "spark", "gate",
            {"tool": "evolution_run", "round": round_no, "hypothesis_id": hyp_id,
             "episode_id": episode_id, "promote": decision.promote,
             "dsr_prob": fitness.dsr_prob, "sr": fitness.sr,
             "notify": decision.promote},
        )
        if consecutive >= max_consec:
            status = "stopped_consecutive_rejections"
            break

    head = lineage.get_line(db, line_id) or {}
    lineage.finish_run(db, run_id, status, head.get("head_hypothesis_id"))
    result = {
        "run_id": run_id,
        "line_id": line_id,
        "family": family,
        "seed": seed,
        "status": status,
        "rounds": len(decisions),
        "decisions": decisions,
        "head": {"hypothesis_id": head.get("head_hypothesis_id"),
                 "episode_id": head.get("head_episode_id")},
        "rollback": {"hypothesis_id": head.get("rollback_hypothesis_id"),
                     "episode_id": head.get("rollback_episode_id")},
        "consecutive_rejections": consecutive,
    }
    if client_id:
        _record_effect(db, client_id, run_id, result)
    return result


def evolution_status(db: Database, run_id: str) -> dict:
    """Run record + per-round decisions + current head/SOTA (read-only).

    Raises ToolError on an unknown run.
    """
    run = db.conn.execute(
        "SELECT * FROM evolution_runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    if run is None:
        raise ToolError(f"unknown run_id {run_id!r}")
    run_row = dict(run)
    line = lineage.get_line(db, run_row["line_id"]) or {}
    rows = db.conn.execute(
        "SELECT * FROM evolution_feedback WHERE run_id = ? ORDER BY rowid", (run_id,)
    ).fetchall()
    feedback = []
    for row in rows:
        row_d = dict(row)
        try:
            fit = json.loads(row_d.get("fitness_json") or "{}")
            dsr_prob = float(fit["dsr_prob"])
            sr = float(fit["sr"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            dsr_prob, sr = None, None
        feedback.append(
            {"feedback_id": row_d["feedback_id"], "hypothesis_id": row_d["hypothesis_id"],
             "episode_id": row_d["episode_id"], "decision": row_d["decision"],
             "dsr_prob": dsr_prob, "sr": sr, "model_id": row_d["model_id"],
             "prompt_sha": row_d["prompt_sha"]}
        )
    head_hyp = line.get("head_hypothesis_id")
    sota = lineage.sota_ancestor(db, head_hyp) if head_hyp else None
    return {
        "run": {k: run_row[k] for k in (
            "run_id", "line_id", "seed", "caps_json", "status",
            "head_before", "head_after", "started_at", "finished_at")},
        "line": {k: line.get(k) for k in (
            "line_id", "family", "head_hypothesis_id", "head_episode_id",
            "rollback_hypothesis_id", "rollback_episode_id", "status")},
        "feedback": feedback,
        "sota": sota,
    }
