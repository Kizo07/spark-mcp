"""Semantic + negative memory: beliefs, tombstones, PIT retrieval (plan §12).

Retrieval is FTS5-only in degraded/no-embedder mode (§4.3); PIT filters apply
BEFORE ranking (§12.1) — a row invisible at `as_of` never enters the candidate
set. All timestamps compare lexicographically (timeutil guarantees one format).
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any

from . import graph, trajectory
from .config import Config
from .ids import new_belief_id, new_tombstone_id
from .storage import Database
from .timeutil import now_iso, validate_iso

TOKEN_RE = re.compile(r"[a-z0-9]+")
HISTORY_KINDS = ("beliefs", "tombstones", "episodes")

# Lexical contradiction threshold for promote (§12.3 no-key mode: lexical-only,
# conservative). Token overlap of the new claim against a candidate claim.
CONTRADICTION_OVERLAP = 0.6


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _fts_query(query: str) -> str | None:
    """Build a safe FTS5 MATCH expression: quoted tokens OR-ed; None if no tokens."""
    toks = _tokens(query)
    if not toks:
        return None
    return " OR ".join(f'"{tok}"' for tok in toks)


def _pit_clause(alias: str = "b") -> str:
    return (
        f"{alias}.valid_from <= :as_of"
        f" AND ({alias}.valid_to IS NULL OR :as_of < {alias}.valid_to)"
        f" AND {alias}.as_of <= :as_of"
    )


def _belief_dict(row: Any) -> dict:
    d = dict(row)
    d["supporting_runs"] = json.loads(d["supporting_runs"])
    d["contradicting_runs"] = json.loads(d["contradicting_runs"])
    d["effective_confidence"] = d["confidence"] * d["decay"]
    return d


def recall(db: Database, query: str, as_of: str, k: int = 10) -> list[dict]:
    """PIT-filtered belief recall, ordered by effective confidence
    (confidence × decay, §12.3) descending, limit k. FTS5 lexical filtering in
    degraded mode; an empty/unmatchable query degrades to all visible beliefs.
    """
    validate_iso(as_of)
    fts = _fts_query(query)
    if fts:
        sql = (
            "SELECT b.* FROM beliefs b JOIN beliefs_fts f ON f.rowid = b.rowid"
            " WHERE beliefs_fts MATCH :fts AND " + _pit_clause()
        )
    else:
        sql = "SELECT b.* FROM beliefs b WHERE " + _pit_clause()
    sql += " ORDER BY b.confidence * b.decay DESC, b.belief_id LIMIT :k"
    rows = db.conn.execute(sql, {"as_of": as_of, "k": k, "fts": fts}).fetchall()
    return [_belief_dict(r) for r in rows]


def research_history(
    db: Database,
    query: str,
    as_of: str,
    k: int = 5,
    kinds: list[str] | None = None,
) -> dict[str, list[dict]]:
    """Merged PIT-filtered history search: beliefs (FTS5 on claim), tombstones
    (FTS5 on summary), episodes (LIKE on hypothesis mechanism / failure_reason).
    k results per requested kind; all kinds PIT-filtered (as_of <= T).
    """
    validate_iso(as_of)
    selected = tuple(kinds) if kinds else HISTORY_KINDS
    unknown = set(selected) - set(HISTORY_KINDS)
    if unknown:
        raise ValueError(
            f"unknown history kinds {sorted(unknown)};"
            f" expected subsets of {HISTORY_KINDS}"
        )

    out: dict[str, list[dict]] = {}
    fts = _fts_query(query)

    if "beliefs" in selected:
        if fts:
            sql = (
                "SELECT b.* FROM beliefs b JOIN beliefs_fts f ON f.rowid = b.rowid"
                " WHERE beliefs_fts MATCH :fts AND "
                + _pit_clause()
                + " ORDER BY b.confidence * b.decay DESC, b.belief_id LIMIT :k"
            )
            rows = db.conn.execute(sql, {"as_of": as_of, "k": k, "fts": fts}).fetchall()
        else:
            rows = []
        out["beliefs"] = [_belief_dict(r) for r in rows]

    if "tombstones" in selected:
        if fts:
            sql = (
                "SELECT t.* FROM tombstones t JOIN tombstones_fts f ON f.rowid = t.rowid"
                " WHERE tombstones_fts MATCH :fts AND t.as_of <= :as_of"
                " ORDER BY t.as_of DESC, t.tombstone_id LIMIT :k"
            )
            rows = db.conn.execute(sql, {"as_of": as_of, "k": k, "fts": fts}).fetchall()
        else:
            rows = []
        out["tombstones"] = [
            {**dict(r), "evidence_episode_ids": json.loads(r["evidence_episode_ids"])}
            for r in rows
        ]

    if "episodes" in selected:
        toks = _tokens(query)
        params: dict = {"as_of": as_of, "k": k}
        clauses = []
        for i, tok in enumerate(toks):
            clauses.append(f"(h.mechanism LIKE :t{i} OR e.failure_reason LIKE :t{i})")
            params[f"t{i}"] = f"%{tok}%"
        where = " AND ".join(clauses) if clauses else "0"
        rows = db.conn.execute(
            "SELECT e.* FROM episodes e JOIN hypotheses h ON h.hypothesis_id = e.hypothesis_id"
            f" WHERE e.as_of <= :as_of AND {where}"
            " ORDER BY e.as_of DESC, e.episode_id LIMIT :k",
            params,
        ).fetchall()
        out["episodes"] = [
            {
                **dict(r),
                "metrics": json.loads(r["metrics"]) if r["metrics"] else None,
                "costs": json.loads(r["costs"]) if r["costs"] else None,
            }
            for r in rows
        ]

    return out


def _find_contradictions(db: Database, claim: str, as_of: str) -> list[dict]:
    """FTS5 candidates on claim tokens among PIT-visible beliefs; lexical
    token-overlap >= CONTRADICTION_OVERLAP (measured against the new claim's
    tokens) confirms a contradiction in no-key mode (§12.3, conservative)."""
    fts = _fts_query(claim)
    if not fts:
        return []
    new_tokens = set(_tokens(claim))
    if not new_tokens:
        return []
    rows = db.conn.execute(
        "SELECT b.* FROM beliefs b JOIN beliefs_fts f ON f.rowid = b.rowid"
        " WHERE beliefs_fts MATCH :fts AND " + _pit_clause(),
        {"as_of": as_of, "fts": fts},
    ).fetchall()
    contradictions = []
    for row in rows:
        cand_tokens = set(_tokens(row["claim"]))
        if not cand_tokens:
            continue
        overlap = len(new_tokens & cand_tokens) / len(new_tokens)
        if overlap >= CONTRADICTION_OVERLAP:
            contradictions.append(
                {"belief_id": row["belief_id"], "claim": row["claim"], "overlap": overlap}
            )
    return contradictions


def promote_belief(
    db: Database,
    cfg: Config,
    episode_ids: list[str],
    claim: str,
    scope: str | None,
    rationale: str | None,
    confidence: float,
    as_of: str | None = None,
    client_id: str | None = None,
    embedder: Any | None = None,
) -> dict:
    """Promote a belief: contradiction check first (§12.3). Contradictions
    found -> evidence attaches to contradicting_runs and status is
    'pending_human'. Otherwise auto-promote iff confidence >=
    cfg.research.belief_auto_promote_threshold. Idempotent on client_id.
    """
    if client_id:
        row = db.conn.execute(
            "SELECT result FROM tool_effects WHERE client_id = ? AND tool = 'memory_promote'",
            (client_id,),
        ).fetchone()
        if row:
            return json.loads(row["result"])

    as_of = validate_iso(as_of) if as_of else now_iso()
    contradictions = _find_contradictions(db, claim, as_of)
    threshold = cfg.research.belief_auto_promote_threshold
    promotion_status = (
        "auto" if confidence >= threshold and not contradictions else "pending_human"
    )

    belief_id = new_belief_id()
    evidence = list(episode_ids)
    with db.conn:
        db.execute(
            "INSERT INTO beliefs ("
            " belief_id, claim, confidence, decay, scope,"
            " supporting_runs, contradicting_runs,"
            " valid_from, valid_to, as_of, data_through, last_validated,"
            " promotion_status, created_at"
            ") VALUES (?, ?, ?, 1.0, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?)",
            (
                belief_id,
                claim,
                confidence,
                scope,
                json.dumps(evidence if not contradictions else []),
                json.dumps(evidence if contradictions else []),
                as_of,
                as_of,
                as_of,
                as_of,
                promotion_status,
                now_iso(),
            ),
        )
        run_id = evidence[0] if evidence else belief_id
        trajectory.record(
            db,
            run_id,
            "spark",
            "tool_call",
            {
                "tool": "memory_promote",
                "belief_id": belief_id,
                "rationale": rationale,  # rationale is audit-payload, not a beliefs column (§5.1)
                "promotion_status": promotion_status,
                "contradictions": [c["belief_id"] for c in contradictions],
                "client_id": client_id,
            },
        )
        # Phase 2 embed-on-write (§4.3): beliefs enter the embedding archive.
        graph.embed_text(db, embedder, "belief", belief_id, claim)

    result = {
        "belief_id": belief_id,
        "promotion_status": promotion_status,
        "contradictions": [c["belief_id"] for c in contradictions],
        "confidence": confidence,
        "effective_confidence": confidence,  # decay 1.0 until consolidate (Phase 7)
    }
    if client_id:
        try:
            db.execute(
                "INSERT INTO tool_effects (client_id, tool, subject_id, result, created_at)"
                " VALUES (?, 'memory_promote', ?, ?, ?)",
                (client_id, belief_id, json.dumps(result, sort_keys=True), now_iso()),
            )
        except sqlite3.IntegrityError:
            # racing retries: return the winner's recorded result, no extra effects.
            winner = db.conn.execute(
                "SELECT result FROM tool_effects WHERE client_id = ? AND tool = 'memory_promote'",
                (client_id,),
            ).fetchone()
            if winner is None:
                raise
            return json.loads(winner["result"])
    return result


def add_tombstone(
    db: Database,
    kind: str,
    summary: str,
    match_keys: dict,
    evidence_episode_ids: list[str],
    as_of: str | None = None,
) -> dict:
    """Record negative knowledge (§12.2). Written by log_outcome on failure
    outcomes, memory_promote contradiction findings, and consolidate.
    """
    as_of = validate_iso(as_of) if as_of else now_iso()
    tombstone_id = new_tombstone_id()
    db.execute(
        "INSERT INTO tombstones ("
        " tombstone_id, kind, summary, match_keys, evidence_episode_ids, as_of, created_at"
        ") VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            tombstone_id,
            kind,
            summary,
            json.dumps(match_keys),
            json.dumps(list(evidence_episode_ids)),
            as_of,
            now_iso(),
        ),
    )
    return {
        "tombstone_id": tombstone_id,
        "kind": kind,
        "summary": summary,
        "match_keys": match_keys,
        "evidence_episode_ids": list(evidence_episode_ids),
        "as_of": as_of,
    }
