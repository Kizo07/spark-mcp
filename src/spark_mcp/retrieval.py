"""Hybrid recall: vector top-K ∪ FTS5 top-K fused by RRF (plan §12.1).

PIT filters apply BEFORE ranking — a node whose as_of is after the query
time never enters the candidate set (§5.0, §12.1 restated). Degraded mode
(§4.3): FTS5-only when no embedder is configured or no embeddings exist.
RRF: score(d) = Σ_sources 1 / (rrf_k + rank_source(d)).
"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

from .llm import split_model_id
from .storage import Database
from .timeutil import validate_iso

TOKEN_RE = re.compile(r"[a-z0-9]+")

# Generic English stopwords (stdlib-only IR hygiene): they carry no
# family/semantic signal and, being rare in node texts, would otherwise hold
# inflated IDF and dominate matches.
STOPWORDS = frozenset(
    """
    a an and are as at be been being but by can could did do does doing for
    from had has have having he her hers him his how i if in into is it its
    just me more most my no nor not of on once only or other our ours over
    own same she should so some such than that the their theirs them
    themselves then there these they this those through to too under until
    very was we were what when where which while who whom why will with
    would you your yours
    """.split()
)


def tokens(text: str) -> set[str]:
    return {t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _fts_match(query: str) -> str | None:
    toks = TOKEN_RE.findall(query.lower())
    if not toks:
        return None
    return " OR ".join(f'"{t}"' for t in toks)


def _stored_model_id(model: str, model_version: str) -> tuple[str, str]:
    return (model, model_version)


def _node_embedding_rows(db: Database, as_of: str) -> list[Any]:
    """Visible graph-node embedding rows (PIT filter applied here, before
    any ranking happens downstream)."""
    if not db.vec_table_exists():
        return []
    return db.conn.execute(
        "SELECT v.subject_id AS subject_id, v.embedding AS embedding,"
        " v.model AS model, v.model_version AS model_version"
        " FROM vec_embeddings v JOIN graph_nodes n ON n.node_id = v.subject_id"
        " WHERE v.subject_type = 'graph_node' AND n.as_of <= ?",
        (as_of,),
    ).fetchall()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _vector_ranking(
    db: Database,
    query: str,
    embedder: Any,
    as_of: str,
    k: int,
    exclude_ids: set[str] | None = None,
) -> list[str]:
    """Top-k visible node ids by cosine of the query embedding vs node
    embeddings stored for this exact model_id (never mix models)."""
    q = np.asarray(embedder.embed([query]), dtype=np.float32)[0]
    model_key = split_model_id(embedder.model_id())
    scored = []
    for r in _node_embedding_rows(db, as_of):
        if exclude_ids and r["subject_id"] in exclude_ids:
            continue
        vec = np.frombuffer(bytes(r["embedding"]), dtype=np.float32)
        if vec.shape != q.shape:  # dimension mismatch = different model
            continue
        if _stored_model_id(r["model"], r["model_version"]) != model_key:
            continue
        scored.append((r["subject_id"], _cosine(q, vec)))
    scored.sort(key=lambda t: (-t[1], t[0]))
    return [nid for nid, _ in scored[:k]]


def _fts_ranking(db: Database, query: str, as_of: str, k: int) -> list[str]:
    match = _fts_match(query)
    if match is None:
        return []
    rows = db.conn.execute(
        "SELECT n.node_id AS node_id FROM graph_nodes_fts f"
        " JOIN graph_nodes n ON n.rowid = f.rowid"
        " WHERE graph_nodes_fts MATCH ? AND n.as_of <= ?"
        " ORDER BY bm25(graph_nodes_fts), n.node_id LIMIT ?",
        (match, as_of, k),
    ).fetchall()
    return [r["node_id"] for r in rows]


def rrf_fuse(rankings: list[list[str]], rrf_k: int) -> list[tuple[str, float]]:
    """Fuse per-source rankings into (id, rrf_score) descending."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, node_id in enumerate(ranking, start=1):
            scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda t: (-t[1], t[0]))


def hybrid_recall(
    db: Database,
    query: str,
    as_of: str,
    embedder: Any | None = None,
    k_vec: int = 50,
    k_fts: int = 50,
    rrf_k: int = 60,
    node_types: tuple[str, ...] = ("technique", "concept"),
) -> list[dict]:
    """Hybrid recall over graph nodes. PIT filter (as_of <= T) applies BEFORE
    ranking in both sources. Returns node dicts with rrf_score and which
    sources contributed, ordered by fused score. Degraded: FTS5-only."""
    validate_iso(as_of)
    type_clause = ",".join("?" for _ in node_types)

    vec_ids: list[str] = []
    if embedder is not None:
        try:
            vec_ids = _vector_ranking(db, query, embedder, as_of, k_vec)
        except Exception:
            vec_ids = []  # embedding must never take retrieval down
    fts_ids = _fts_ranking(db, query, as_of, k_fts)

    fused = rrf_fuse([r for r in (vec_ids, fts_ids) if r], rrf_k)
    if not fused:
        return []

    ids = [nid for nid, _ in fused]
    marks = ",".join("?" for _ in ids)
    rows = db.conn.execute(
        f"SELECT * FROM graph_nodes WHERE node_id IN ({marks}) AND type IN ({type_clause})",
        [*ids, *node_types],
    ).fetchall()
    by_id = {r["node_id"]: dict(r) for r in rows}
    out = []
    vec_set, fts_set = set(vec_ids), set(fts_ids)
    for nid, score in fused:
        if nid not in by_id:
            continue
        node = by_id[nid]
        node["mech_tags"] = json.loads(node["mech_tags"]) if node.get("mech_tags") else []
        node["rrf_score"] = score
        node["sources"] = [
            src for src, s in (("vector", vec_set), ("fts5", fts_set)) if nid in s
        ]
        out.append(node)
    return out


def nearest_nodes(
    db: Database,
    embedding: np.ndarray,
    model_id: str,
    as_of: str,
    k: int = 5,
    exclude_ids: set[str] | None = None,
) -> list[dict]:
    """Nearest visible graph nodes to an arbitrary embedding (novelty archive
    math). Model-scoped: only rows written by this exact model_id count."""
    q = np.asarray(embedding, dtype=np.float32)
    model_key = split_model_id(model_id)
    hits = []
    for r in _node_embedding_rows(db, as_of):
        if exclude_ids and r["subject_id"] in exclude_ids:
            continue
        if _stored_model_id(r["model"], r["model_version"]) != model_key:
            continue
        vec = np.frombuffer(bytes(r["embedding"]), dtype=np.float32)
        if vec.shape != q.shape:
            continue
        hits.append({"node_id": r["subject_id"], "cosine": _cosine(q, vec)})
    hits.sort(key=lambda h: (-h["cosine"], h["node_id"]))
    return hits[:k]
