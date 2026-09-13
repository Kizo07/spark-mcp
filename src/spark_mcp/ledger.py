"""The research ledger: hypotheses, trial episodes, trials, outcomes (plan §5,
§6.0, §6.3, §9.2). Typed service functions over `Database` — the MCP tools in
server.py and the CLI are thin wrappers over these (N13: no logic fork).

Trial accounting (§9.2, AGENTS.md rule 4): the pending row lands before
dispatch in Phase 3; `log_outcome` closes the loop and reconciles out-of-band
runs via a `reconciled` trial so M never silently restarts.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any

from . import __version__, archive, budgets, graph, memory, novelty, trajectory
from .config import Config
from .errors import ToolError
from .ids import new_episode_id, new_hypothesis_id, new_trial_id
from .storage import Database
from .timeutil import now_iso, validate_iso

# outcome vocabulary accepted by log_outcome -> episode status (§5.1)
OUTCOME_STATUS = {
    "success": "validated",
    "failure": "failed",
    "inconclusive": "inconclusive",
    "validated": "validated",
    "failed": "failed",
}

HYPOTHESIS_TYPES = {"predictive", "causal", "descriptive", "portfolio", "risk", "execution"}

_EPISODE_JSON_FIELDS = (
    "skill_versions",
    "datasets",
    "params",
    "metrics",
    "corrected_metrics",
    "risk_exposures",
    "costs",
    "tombstone_check",
)


def _canonical_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _sha256(obj: Any) -> str:
    return hashlib.sha256(_canonical_bytes(obj)).hexdigest()


def _idem_key(episode_id: str, spec_hash: str | None, seed: int | None) -> str:
    raw = f"{episode_id}|{spec_hash or ''}|{seed if seed is not None else ''}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()  # noqa: S324 (plan-mandated sha1)


def _parse(text: str | None, default: Any = None) -> Any:
    if text is None:
        return default
    return json.loads(text)


def _hypothesis_dict(row: Any) -> dict:
    d = dict(row)
    d["parent_hypothesis_ids"] = _parse(d["parent_hypothesis_ids"], [])
    d["dag"] = _parse(d["dag"])
    d["adjustment_set"] = _parse(d["adjustment_set"])
    return d


def _episode_dict(row: Any) -> dict:
    d = dict(row)
    for field in _EPISODE_JSON_FIELDS:
        d[field] = _parse(d[field])
    return d


def get_hypothesis(db: Database, hypothesis_id: str) -> dict | None:
    row = db.conn.execute(
        "SELECT * FROM hypotheses WHERE hypothesis_id = ?", (hypothesis_id,)
    ).fetchone()
    return _hypothesis_dict(row) if row else None


def get_episode(db: Database, episode_id: str) -> dict | None:
    row = db.conn.execute(
        "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
    ).fetchone()
    return _episode_dict(row) if row else None


def _record_effect(db: Database, tool: str, client_id: str, subject_id: str, result: dict) -> None:
    db.execute(
        "INSERT OR IGNORE INTO tool_effects (client_id, tool, subject_id, result, created_at)"
        " VALUES (?, ?, ?, ?, ?)",
        (client_id, tool, subject_id, json.dumps(result, sort_keys=True), now_iso()),
    )


def _replay_effect(db: Database, tool: str, client_id: str) -> dict | None:
    row = db.conn.execute(
        "SELECT result FROM tool_effects WHERE client_id = ? AND tool = ?", (client_id, tool)
    ).fetchone()
    return json.loads(row["result"]) if row else None


def propose_hypothesis(
    db: Database,
    cfg: Config,
    *,
    research_family: str,
    hypothesis_type: str,
    mechanism: str,
    expected_direction: str,
    horizon: str,
    scope: str,
    falsifier: str,
    predicted_ic: float,
    predicted_sigma: float,
    universe: str,
    idea: str = "",
    dag: dict | None = None,
    adjustment_set: list[str] | None = None,
    parent_hypothesis_ids: list[str] | None = None,
    as_of: str | None = None,
    client_id: str | None = None,
    embedder: Any | None = None,
) -> dict:
    """Insert a hypotheses row (status 'proposed'). Enforces the C6 refusal rule.

    Returns {hypothesis_id, N, S} — N scored against the Phase 2 archive
    (§7; lexical proxy in degraded mode), S always None at propose time (no
    realized_ic exists yet). Idempotent on client_id: reuse replays the
    original recorded result envelope, no side effects.
    """
    if client_id:
        row = db.conn.execute(
            "SELECT hypothesis_id FROM hypotheses WHERE client_id = ?", (client_id,)
        ).fetchone()
        if row:
            recorded = _replay_effect(db, "hypothesis_propose", client_id)
            if recorded is not None:
                return recorded
            return {"hypothesis_id": row["hypothesis_id"], "N": None, "S": None}

    as_of = validate_iso(as_of) if as_of else now_iso()

    # C6 refusal rule: core fields required for ALL types; dag + adjustment_set
    # required iff causal (§6.1 tool 1, §9.3).
    required = {
        "mechanism": mechanism,
        "expected_direction": expected_direction,
        "horizon": horizon,
        "falsifier": falsifier,
        "scope": scope,
        "predicted_ic": predicted_ic,
        "predicted_sigma": predicted_sigma,
    }
    missing = [
        k
        for k, v in required.items()
        if v is None or (isinstance(v, str) and not v.strip())
    ]
    if hypothesis_type == "causal":
        if not dag:
            missing.append("dag")
        if not adjustment_set:
            missing.append("adjustment_set")
    if hypothesis_type not in HYPOTHESIS_TYPES:
        raise ValueError(
            f"unknown hypothesis_type {hypothesis_type!r};"
            f" expected one of {sorted(HYPOTHESIS_TYPES)}"
        )
    if missing:
        raise ValueError(
            "hypothesis refused (C6): missing required fields: " + ", ".join(sorted(missing))
        )

    hypothesis_id = new_hypothesis_id()
    try:
        with db.transaction():
            db.execute(
                "INSERT INTO hypotheses ("
                " hypothesis_id, parent_hypothesis_ids, research_family, hypothesis_type,"
                " mechanism, expected_direction, horizon, scope, dag, adjustment_set,"
                " falsifier, predicted_ic, predicted_sigma, universe,"
                " as_of, data_through, status, client_id, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'proposed', ?, ?)",
                (
                    hypothesis_id,
                    json.dumps(parent_hypothesis_ids or []),
                    research_family,
                    hypothesis_type,
                    mechanism,
                    expected_direction,
                    horizon,
                    scope,
                    json.dumps(dag) if dag is not None else None,
                    json.dumps(adjustment_set) if adjustment_set is not None else None,
                    falsifier,
                    predicted_ic,
                    predicted_sigma,
                    universe,
                    as_of,
                    as_of,
                    client_id,
                    now_iso(),
                ),
            )
    except sqlite3.IntegrityError:
        # §9.2: racing writers resolve to one winner; the loser reads the winner's row.
        winner = db.conn.execute(
            "SELECT hypothesis_id FROM hypotheses WHERE client_id = ?", (client_id,)
        ).fetchone()
        if winner is None:
            raise
        recorded = _replay_effect(db, "hypothesis_propose", client_id)
        if recorded is not None:
            return recorded
        return {"hypothesis_id": winner["hypothesis_id"], "N": None, "S": None}
    trajectory.record(
        db,
        hypothesis_id,
        "spark",
        "tool_call",
        {"tool": "hypothesis_propose", "client_id": client_id, "as_of": as_of, "idea": idea},
    )
    # Phase 2 (§6.1 tool 1, §7): embed-on-write, then bounded novelty vs the
    # archive (excluding the idea itself). S stays None until realized_ic.
    idea_text = " ".join(p for p in (idea, mechanism) if p)
    graph.embed_text(db, embedder, "hypothesis", hypothesis_id, idea_text)
    scored = novelty.check_novelty(
        db,
        cfg,
        idea_text,
        family=research_family,
        as_of=as_of,
        embedder=embedder,
        exclude_subject_ids={hypothesis_id},
        archive=archive.Archive(db, cfg),  # real §7 cell_emptiness when in vocabulary
    )
    result = {
        "hypothesis_id": hypothesis_id,
        "N": scored["N"],
        "S": None,
        "novelty": {
            "max_cos": scored["max_cos"],
            "structural_overlap": scored["structural_overlap"],
            "cell_emptiness": scored["cell_emptiness"],
        },
    }
    if client_id:
        _record_effect(db, "hypothesis_propose", client_id, hypothesis_id, result)
    return result


def register_hypothesis(
    db: Database,
    cfg: Config,
    hypothesis_id: str,
    spec: dict,
    seed: int,
    client_id: str | None = None,
) -> dict:
    """Create the trial episode (status 'registered'), compute protocol_hash
    (§5.0: SHA-256 over canonical JSON of {hypothesis_id, spec, seed,
    code_version}), flip the hypothesis to 'registered'. Idempotent on client_id.
    """
    if client_id:
        row = db.conn.execute(
            "SELECT * FROM episodes WHERE client_id = ?", (client_id,)
        ).fetchone()
        if row:
            return {
                "episode_id": row["episode_id"],
                "hypothesis_id": row["hypothesis_id"],
                "protocol_hash": row["protocol_hash"],
                "spec_hash": row["spec_hash"],
                "seed": row["seed"],
                "status": row["status"],
                "reused": True,
            }

    hypothesis = get_hypothesis(db, hypothesis_id)
    if hypothesis is None:
        raise ValueError(f"unknown hypothesis_id {hypothesis_id!r}")

    payload = {
        "hypothesis_id": hypothesis_id,
        "spec": spec,
        "seed": seed,
        "code_version": __version__,
    }
    spec_hash = _sha256(spec)
    canonical = _canonical_bytes(payload)
    protocol_hash = hashlib.sha256(canonical).hexdigest()
    # §5.0/N8: store the canonical pre-registration bytes for audit.
    protocols_dir = cfg.artifacts_dir() / "protocols"
    protocols_dir.mkdir(parents=True, exist_ok=True)
    (protocols_dir / f"{protocol_hash}.json").write_bytes(canonical)
    episode_id = new_episode_id()
    as_of = now_iso()
    try:
        with db.transaction():
            db.execute(
                "INSERT INTO episodes ("
                " episode_id, hypothesis_id, run_kind, research_family,"
                " protocol_hash, spec_hash, seed, status,"
                " as_of, data_through, client_id, created_at"
                ") VALUES (?, ?, 'primary', ?, ?, ?, ?, 'registered', ?, ?, ?, ?)",
                (
                    episode_id,
                    hypothesis_id,
                    hypothesis["research_family"],
                    protocol_hash,
                    spec_hash,
                    seed,
                    as_of,
                    as_of,
                    client_id,
                    now_iso(),
                ),
            )
            db.execute(
                "UPDATE hypotheses SET status = 'registered' WHERE hypothesis_id = ?",
                (hypothesis_id,),
            )
    except sqlite3.IntegrityError:
        # racing retry: the winner's episode already holds this client_id.
        winner = db.conn.execute(
            "SELECT * FROM episodes WHERE client_id = ?", (client_id,)
        ).fetchone()
        if winner is None:
            raise
        return {
            "episode_id": winner["episode_id"],
            "hypothesis_id": winner["hypothesis_id"],
            "protocol_hash": winner["protocol_hash"],
            "spec_hash": winner["spec_hash"],
            "seed": winner["seed"],
            "status": winner["status"],
            "reused": True,
        }
    trajectory.record(
        db,
        episode_id,
        "spark",
        "tool_call",
        {
            "tool": "register_hypothesis",
            "hypothesis_id": hypothesis_id,
            "client_id": client_id,
        },
    )
    return {
        "episode_id": episode_id,
        "hypothesis_id": hypothesis_id,
        "protocol_hash": protocol_hash,
        "spec_hash": spec_hash,
        "seed": seed,
        "status": "registered",
        "reused": False,
    }


def _log_outcome_impl(
    db: Database,
    cfg: Config,
    episode_id: str,
    outcome: str,
    metrics: dict | None = None,
    failure_reason: str | None = None,
    costs: dict | None = None,
    reward: float | None = None,
    client_id: str | None = None,
    embedder: Any | None = None,
) -> dict:
    """Close the loop on an episode: flip status per outcome, record
    costs (required-or-estimated, §14), keep reward non-null (M13), and apply
    the C4 reconciliation rule — metrics arriving for an episode with no
    counted trial insert a `reconciled` trial and flag the episode. Failure
    outcomes write a tombstone (§12.2, dedupe-keyed + embedded). Every outcome
    appends to spend_ledger and ticks the per-family kill-rule counters
    (§8.3). Idempotent on client_id.
    """
    if client_id:
        replayed = _replay_effect(db, "log_outcome", client_id)
        if replayed is not None:
            return replayed

    if outcome not in OUTCOME_STATUS:
        raise ValueError(
            f"unknown outcome {outcome!r}; expected one of {sorted(OUTCOME_STATUS)}"
        )
    episode = get_episode(db, episode_id)
    if episode is None:
        raise ValueError(f"unknown episode_id {episode_id!r}")

    if costs is None:
        costs = {"usd": 0.0, "tokens": 0, "estimated": True}
    else:
        costs = {**costs, "estimated": bool(costs.get("estimated", False))}
    if reward is None:
        reward = 0.0

    status = OUTCOME_STATUS[outcome]
    now = now_iso()
    trial_id: str | None = None
    trial_source: str | None = None
    reconciled = episode["reconciled"]
    usd, tokens, spend_estimated = budgets.spend_from_costs(cfg, costs)

    with db.transaction():
        if metrics is not None:
            trial = db.conn.execute(
                "SELECT * FROM trials WHERE episode_id = ? ORDER BY created_at DESC LIMIT 1",
                (episode_id,),
            ).fetchone()
            if trial is not None:
                if trial["status"] != "completed":
                    db.execute(
                        "UPDATE trials SET status = 'completed', completed_at = ?"
                        " WHERE trial_id = ?",
                        (now, trial["trial_id"]),
                    )
                trial_id = trial["trial_id"]
                trial_source = trial["source"]
            else:
                # C4: out-of-band outcome — backfill a reconciled trial so M counts it.
                idem = _idem_key(episode_id, episode["spec_hash"], episode["seed"])
                trial_id = new_trial_id()
                try:
                    db.execute(
                        "INSERT INTO trials ("
                        " trial_id, episode_id, idem_key, family, source, status,"
                        " provider, created_at, completed_at"
                        ") VALUES (?, ?, ?, ?, 'reconciled', 'completed', 'unknown', ?, ?)",
                        (trial_id, episode_id, idem, episode["research_family"], now, now),
                    )
                except sqlite3.IntegrityError:
                    # racing log_outcome calls: exactly one reconciled winner;
                    # the losers read the winner's row (§9.2).
                    winner = db.conn.execute(
                        "SELECT * FROM trials WHERE episode_id = ? AND idem_key = ?",
                        (episode_id, idem),
                    ).fetchone()
                    trial_id = winner["trial_id"]
                    trial_source = winner["source"]
                else:
                    trial_source = "reconciled"
                    db.execute(
                        "UPDATE episodes SET reconciled = 1 WHERE episode_id = ?", (episode_id,)
                    )
                    reconciled = 1

        db.execute(
            "UPDATE episodes SET status = ?, outcome = ?, metrics = ?, failure_reason = ?,"
            " costs = ?, reward = ? WHERE episode_id = ?",
            (
                status,
                outcome,
                json.dumps(metrics) if metrics is not None else None,
                failure_reason,
                json.dumps(costs),
                reward,
                episode_id,
            ),
        )

        tombstone_id = None
        if failure_reason:
            hypothesis = get_hypothesis(db, episode["hypothesis_id"]) or {}
            tombstone = memory.add_tombstone(
                db,
                kind="failed_hypothesis",
                summary=(
                    f"[{episode['research_family']}/{hypothesis.get('hypothesis_type', '?')}]"
                    f" {failure_reason}"
                ),
                match_keys={
                    "hypothesis_id": episode["hypothesis_id"],
                    "mech_tags": [
                        episode["research_family"],
                        hypothesis.get("hypothesis_type", ""),
                    ],
                    "keywords": (hypothesis.get("mechanism") or "").lower().split(),
                },
                evidence_episode_ids=[episode_id],
                as_of=now,
                # Phase 1 review nit: one tombstone per failed episode even
                # under concurrent racing log_outcomes.
                dedupe_key=hashlib.sha1(
                    f"episode-failure|{episode_id}".encode()
                ).hexdigest(),
                embedder=embedder,
            )
            tombstone_id = tombstone["tombstone_id"]

        # §14: every spend event appends to spend_ledger; over-limit days
        # (policy warn) or episodes are warned about, or refused whole under
        # fail_closed — the raise rolls this outcome's episode update, trial
        # accounting, and spend row back together (see budgets module docstring).
        budgets.record_spend(
            db,
            cfg,
            family=episode["research_family"],
            source="log_outcome",
            usd=usd,
            tokens=tokens,
            estimated=spend_estimated,
            run_id=episode_id,
        )
        budgets.enforce_daily_budget(db, cfg, run_id=episode_id, source="log_outcome")
        budgets.enforce_episode_budget(db, cfg, episode_id=episode_id, source="log_outcome")

        # §8.3 kill rule: success resets consecutive_failures, failure ticks it.
        from .gates.service import record_family_outcome

        record_family_outcome(
            db,
            cfg,
            episode["research_family"],
            success=outcome in ("success", "validated"),
            episode_id=episode_id,
            failure_reason=failure_reason,
        )

        # Phase 5 (§6.3 tool 15, §10): a success/validated outcome inserts the
        # hypothesis into the QD archive (SQL durable truth, M7; the write
        # joins this transaction). Named refusals — unknown descriptor
        # category, or the descriptive/execution exclusion (§10.2) — are
        # logged on the trajectory, never an error to the caller.
        if outcome in ("success", "validated"):
            hypothesis = get_hypothesis(db, episode["hypothesis_id"]) or {}
            params = episode.get("params") or {}
            merged_metrics = {**(episode.get("metrics") or {}), **(metrics or {})}
            if episode.get("corrected_metrics"):
                merged_metrics["corrected_metrics"] = episode["corrected_metrics"]
            try:
                inserted = archive.Archive(db, cfg).insert(
                    episode["hypothesis_id"],
                    {
                        "family": episode["research_family"],
                        "horizon": hypothesis.get("horizon"),
                        "turnover": params.get("turnover"),
                    },
                    merged_metrics,
                    hypothesis.get("hypothesis_type") or "",
                    episode_id=episode_id,
                )
                trajectory.record(
                    db,
                    episode_id,
                    "spark",
                    "gate",
                    {
                        "kind": "qd_insert",
                        "cell_key": inserted["cell_key"],
                        "elite_score": inserted["elite_score"],
                        "elite_replaced": inserted["elite_replaced"],
                        "reward_type": inserted["reward_type"],
                    },
                )
            except ToolError as e:
                trajectory.record(
                    db,
                    episode_id,
                    "spark",
                    "observation",
                    {"kind": "qd_insert_refused", "reason": str(e)},
                )

        trajectory.record(
            db,
            episode_id,
            "spark",
            "tool_call",
            {
                "tool": "log_outcome",
                "outcome": outcome,
                "status": status,
                "reconciled": reconciled,
                "client_id": client_id,
            },
        )

    result = {
        "episode_id": episode_id,
        "status": status,
        "outcome": outcome,
        "reconciled": reconciled,
        "trial_id": trial_id,
        "trial_source": trial_source,
        "costs": costs,
        "reward": reward,
        "tombstone_id": tombstone_id,
    }
    if client_id:
        _record_effect(db, "log_outcome", client_id, episode_id, result)
    return result


def log_outcome(
    db: Database,
    cfg: Config,
    episode_id: str,
    outcome: str,
    metrics: dict | None = None,
    failure_reason: str | None = None,
    costs: dict | None = None,
    reward: float | None = None,
    client_id: str | None = None,
    embedder: Any | None = None,
) -> dict:
    """Thread-safe wrapper: see _log_outcome_impl (storage.locked serializes
    the shared connection across MCP worker threads)."""
    with db.locked():
        return _log_outcome_impl(
            db,
            cfg,
            episode_id,
            outcome,
            metrics=metrics,
            failure_reason=failure_reason,
            costs=costs,
            reward=reward,
            client_id=client_id,
            embedder=embedder,
        )
