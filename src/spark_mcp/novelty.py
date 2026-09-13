"""Bounded novelty N and surprise S (plan §7, C8).

    N = w1·(1 − max_cos) + w2·(1 − structural_overlap) + w3·cell_emptiness
    S = min(|realized_ic − predicted_ic| / max(predicted_sigma, sigma_floor), s_cap)

Defined behavior at the edges: an empty archive ⇒ max_cos = 0 (the idea is
maximally novel vs nothing); no graph grounding ⇒ structural_overlap = 0;
both keep N in [0, 1] and deterministic. cell_emptiness has two modes
(Phase 5): with an archive.Archive injected AND a known candidate family the
term is REAL QD occupancy (§7) — 1.0 when the candidate family's slice has
zero trials, else 1 − slice_trials/max_slice_trials over the grid, bounded
[0,1] (a check_novelty candidate carries a family but no horizon/turnover,
so the family slice is the finest archive granularity it can address; the
basis is reported as cell_emptiness_basis='qd_archive'). Degraded mode — no
archive injected (default None) or a family outside the bin vocabulary —
keeps the v1 approximation 1 − distinct_families_tried / family_count_norm
(basis 'family_count_approximation'), families PIT from episodes. In
degraded mode (no embedder) max_cos falls back to a deterministic lexical
proxy (max Jaccard of idea tokens vs archive text) so N stays meaningful
without external calls; the basis is reported.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import retrieval
from .config import Config
from .errors import ToolError
from .llm import split_model_id
from .storage import Database
from .timeutil import now_iso, validate_iso


def _archive_hits(
    db: Database,
    idea_tokens: set[str],
    embedding: np.ndarray | None,
    model_id: str | None,
    family: str | None,
    as_of: str,
    k: int,
    exclude_subject_ids: set[str] | None,
) -> tuple[list[dict], float, str]:
    """Archive cosine search over graph nodes + prior hypotheses in family.
    Returns (hits, max_cos, basis). Empty archive ⇒ max_cos 0.0."""
    members: list[dict] = []  # {subject_type, subject_id, label, vec|tokens}
    node_rows = db.conn.execute(
        "SELECT node_id, label, mech_tags, summary FROM graph_nodes WHERE as_of <= ?",
        (as_of,),
    ).fetchall()
    for r in node_rows:
        members.append(
            {
                "subject_type": "graph_node",
                "subject_id": r["node_id"],
                "label": r["label"],
                "tokens": retrieval.tokens(
                    " ".join([r["label"], r["summary"] or "", r["mech_tags"]])
                ),
            }
        )
    hyp_sql = (
        "SELECT hypothesis_id, mechanism FROM hypotheses WHERE as_of <= ?"
        + (" AND research_family = ?" if family else "")
    )
    params: tuple = (as_of, family) if family else (as_of,)
    for r in db.conn.execute(hyp_sql, params).fetchall():
        members.append(
            {
                "subject_type": "hypothesis",
                "subject_id": r["hypothesis_id"],
                "label": r["mechanism"],
                "tokens": retrieval.tokens(r["mechanism"]),
            }
        )

    vec_rows = []
    if embedding is not None and model_id is not None and db.vec_table_exists():
        model_key = split_model_id(model_id)
        rows = db.conn.execute(
            "SELECT subject_id, embedding, model, model_version FROM vec_embeddings"
            " WHERE subject_type IN ('graph_node', 'hypothesis')",
        ).fetchall()
        hyp_families = {
            r["hypothesis_id"]: r["research_family"]
            for r in db.conn.execute(
                "SELECT hypothesis_id, research_family FROM hypotheses WHERE as_of <= ?",
                (as_of,),
            ).fetchall()
        }
        visible_nodes = {
            m["subject_id"] for m in members if m["subject_type"] == "graph_node"
        }
        for r in rows:
            sid = r["subject_id"]
            if (r["model"], r["model_version"]) != model_key:
                continue
            if sid in visible_nodes:
                pass  # visible graph node, keep
            elif sid in hyp_families and (family is None or hyp_families[sid] == family):
                pass  # hypothesis in family, keep
            else:
                continue
            if exclude_subject_ids and sid in exclude_subject_ids:
                continue
            vec = np.frombuffer(bytes(r["embedding"]), dtype=np.float32)
            if vec.shape != embedding.shape:
                continue
            vec_rows.append((sid, vec))

    hits: list[dict] = []
    max_cos = 0.0
    if embedding is not None and vec_rows:
        scored = [
            (sid, retrieval._cosine(embedding, vec)) for sid, vec in vec_rows
        ]
        scored.sort(key=lambda t: (-t[1], t[0]))
        label_by_id = {m["subject_id"]: m["label"] for m in members}
        type_by_id = {m["subject_id"]: m["subject_type"] for m in members}
        hits = [
            {
                "subject_type": type_by_id[sid],
                "subject_id": sid,
                "label": label_by_id.get(sid, sid),
                "cosine": cos,
            }
            for sid, cos in scored[:k]
        ]
        max_cos = scored[0][1]
        return hits, max_cos, "embedding"

    # degraded / no embeddings: deterministic lexical proxy
    pool = [
        m
        for m in members
        if not (exclude_subject_ids and m["subject_id"] in exclude_subject_ids)
    ]
    if pool:
        max_cos = max(retrieval.jaccard(idea_tokens, m["tokens"]) for m in pool)
        best = max(pool, key=lambda m: retrieval.jaccard(idea_tokens, m["tokens"]))
        hits = [
            {
                "subject_type": best["subject_type"],
                "subject_id": best["subject_id"],
                "label": best["label"],
                "cosine": max_cos,
            }
        ]
        return hits, max_cos, "lexical"
    return [], 0.0, "lexical"


def _structural_overlap(
    db: Database,
    edge_set: set[tuple[str, str]] | None,
    as_of: str,
    exclude_node_ids: set[str] | None = None,
) -> float:
    """max Jaccard of the idea's (predicate, dst_type) edge set vs archive
    members' sets; 0 when the idea has no graph grounding (§7)."""
    if not edge_set:
        return 0.0
    rows = db.conn.execute(
        "SELECT e.src AS src, e.predicate AS predicate, n.type AS dst_type"
        " FROM graph_edges e JOIN graph_nodes n ON n.node_id = e.dst"
        " WHERE e.valid_from <= ? AND (e.valid_to IS NULL OR ? < e.valid_to)"
        " AND e.as_of <= ?",
        (as_of, as_of, as_of),
    ).fetchall()
    by_src: dict[str, set[tuple[str, str]]] = {}
    for r in rows:
        if exclude_node_ids and r["src"] in exclude_node_ids:
            continue
        by_src.setdefault(r["src"], set()).add((r["predicate"], r["dst_type"]))
    best = 0.0
    for profile in by_src.values():
        if not profile:
            continue
        best = max(best, retrieval.jaccard(set(edge_set), profile))
    return best


def _cell_emptiness(db: Database, cfg: Config, as_of: str) -> tuple[float, int]:
    """Degraded approximation (the REAL QD occupancy lives in
    archive.Archive.cell_emptiness, §7): 1 − distinct_families_tried /
    family_count_norm, families PIT from episodes."""
    row = db.conn.execute(
        "SELECT COUNT(DISTINCT research_family) AS n FROM episodes WHERE as_of <= ?",
        (as_of,),
    ).fetchone()
    tried = int(row["n"]) if row else 0
    norm = max(int(cfg.research.family_count_norm), 1)
    return 1.0 - min(tried / norm, 1.0), tried


def _cell_emptiness_term(
    db: Database,
    cfg: Config,
    as_of: str,
    family: str | None,
    archive: Any | None,
) -> tuple[float, int, str]:
    """§7 cell_emptiness with basis: the archive's REAL family-slice occupancy
    when an archive is injected and the family is in the QD vocabulary; the
    degraded family-count approximation otherwise."""
    approx, tried = _cell_emptiness(db, cfg, as_of)
    if archive is not None and family is not None:
        try:
            # families_tried stays the episode-derived count (field meaning
            # unchanged across modes); the basis flags which term fed N.
            return archive.cell_emptiness(family), tried, "qd_archive"
        except ToolError:
            pass  # family outside the bin vocabulary: degraded approximation
    return approx, tried, "family_count_approximation"


def surprise(
    cfg: Config,
    realized_ic: float | None,
    predicted_ic: float | None,
    predicted_sigma: float | None,
) -> float | None:
    """S per §7; None until a realized_ic exists (caller supplies it)."""
    if realized_ic is None or predicted_ic is None:
        return None
    sigma = max(predicted_sigma or 0.0, cfg.research.sigma_floor)
    return min(abs(realized_ic - predicted_ic) / sigma, cfg.research.s_cap)


def check_novelty(
    db: Database,
    cfg: Config,
    idea_or_code: str,
    family: str | None = None,
    as_of: str | None = None,
    embedder: Any | None = None,
    realized_ic: float | None = None,
    predicted_ic: float | None = None,
    predicted_sigma: float | None = None,
    edge_set: set[tuple[str, str]] | None = None,
    exclude_subject_ids: set[str] | None = None,
    nearest_k: int = 5,
    archive: Any | None = None,
) -> dict:
    """Bounded novelty + surprise for an idea/hypothesis (§6.1 tool 5, §7).

    `archive`: an archive.Archive instance for the REAL §7 cell_emptiness
    term; None (default) keeps the degraded family-count approximation.
    Unknown families (outside the QD bin vocabulary) fall back to the
    approximation even when an archive is present.
    """
    as_of = validate_iso(as_of) if as_of else now_iso()
    idea_tokens = retrieval.tokens(idea_or_code)

    embedding = None
    model_id = None
    if embedder is not None:
        try:
            embedding = np.asarray(embedder.embed([idea_or_code]), dtype=np.float32)[0]
            model_id = embedder.model_id()
        except Exception:
            embedding = None  # embedding failure degrades, never refuses

    nearest_hits, max_cos, basis = _archive_hits(
        db,
        idea_tokens,
        embedding,
        model_id,
        family,
        as_of,
        nearest_k,
        exclude_subject_ids,
    )
    structural = _structural_overlap(db, edge_set, as_of)
    emptiness, families_tried, emptiness_basis = _cell_emptiness_term(
        db, cfg, as_of, family, archive
    )

    w1, w2, w3 = (list(cfg.research.novelty_weights) + [0.0, 0.0, 0.0])[:3]
    n = w1 * (1.0 - max_cos) + w2 * (1.0 - structural) + w3 * emptiness
    n = max(0.0, min(1.0, n))  # each term ∈ [0,1] ⇒ N ∈ [0,1]

    return {
        "N": n,
        "S": surprise(cfg, realized_ic, predicted_ic, predicted_sigma),
        "max_cos": max_cos,
        "max_cos_basis": basis,  # embedding | lexical (degraded)
        "structural_overlap": structural,
        "cell_emptiness": emptiness,
        "cell_emptiness_basis": emptiness_basis,  # qd_archive | family_count_approximation
        "families_tried": families_tried,
        "nearest_hits": nearest_hits,
        "as_of": as_of,
    }
