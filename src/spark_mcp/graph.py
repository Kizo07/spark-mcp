"""Method graph: nodes, edges, seed import, embed-on-write (plan §5.1, §16.1).

Predicates are restricted to the 6 active at launch (§2.7); anything else is
rejected with a message. Embeddings follow N3: no BLOB columns on base
tables — every embedding lands in vec_embeddings keyed by (subject_type,
subject_id, model, model_version) so a model change can never silently
re-scale novelty or cached scores (§4.3).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .ids import new_edge_id, new_node_id
from .llm import split_model_id
from .storage import Database
from .timeutil import now_iso, validate_iso

# §2.7: 12 predicate types defined, 6 active at launch; the rest are
# rejected-with-message until enabled.
PREDICATES = frozenset(
    {
        "SOLVES",
        "FAILS_WHEN",
        "COMPOSES_WITH",
        "DIAGNOSES",
        "ASSUMES",
        "DERIVED_FROM",
    }
)

SEEDS_DIR = Path(__file__).resolve().parent / "seeds"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_name(text: str) -> str:
    """Case/space-insensitive key for matching card references to node labels."""
    return "_".join(_TOKEN_RE.findall(text.lower()))


def node_text(node: dict) -> str:
    """The text that represents a node for embedding and lexical scoring."""
    parts = [node.get("label") or "", node.get("summary") or ""]
    parts.extend(node.get("mech_tags") or [])
    return " ".join(p for p in parts if p)


# -- embeddings (N3: vec_embeddings virtual table only) -----------------------


def embed_text(
    db: Database, embedder: Any, subject_type: str, subject_id: str, text: str
) -> np.ndarray | None:
    """Embed-on-write: store the embedding with model + model_version.
    Returns the vector, or None when no embedder is configured (degraded)."""
    if embedder is None or not text.strip():
        return None
    vec = np.asarray(embedder.embed([text]), dtype=np.float32)[0]
    db.ensure_vec_embeddings(int(vec.shape[0]))
    model, model_version = split_model_id(embedder.model_id())
    with db.transaction():
        db.conn.execute(
            "INSERT INTO vec_embeddings (subject_type, subject_id, model, model_version,"
            " embedding) VALUES (?, ?, ?, ?, ?)",
            (subject_type, subject_id, model, model_version, vec.tobytes()),
        )
    return vec


def fetch_embeddings(
    db: Database, model_id: str, subject_type: str | None = None
) -> list[tuple[str, np.ndarray]]:
    """All embeddings for a model (optionally one subject type), as numpy rows.
    Reads work identically against the vec0 virtual table and the degraded
    BLOB fallback; similarity itself is computed in numpy (archive scale)."""
    if not db.vec_table_exists():
        return []
    model, model_version = split_model_id(model_id)
    sql = (
        "SELECT subject_id, embedding FROM vec_embeddings"
        " WHERE model = ? AND model_version = ?"
    )
    params: list[Any] = [model, model_version]
    if subject_type is not None:
        sql += " AND subject_type = ?"
        params.append(subject_type)
    rows = db.conn.execute(sql, params).fetchall()
    out = []
    for r in rows:
        buf = bytes(r["embedding"])
        out.append((r["subject_id"], np.frombuffer(buf, dtype=np.float32).copy()))
    return out


# -- nodes & edges ------------------------------------------------------------


def add_node(
    db: Database,
    type: str,
    label: str,
    mech_tags: list[str] | None = None,
    summary: str | None = None,
    family: str | None = None,
    as_of: str | None = None,
    embedder: Any | None = None,
    node_id: str | None = None,
) -> dict:
    """Insert a graph node (FTS5 trigger indexes the label). With an embedder
    configured the node text is embedded on write (§4.3 embed-on-write)."""
    as_of = validate_iso(as_of) if as_of else now_iso()
    node_id = node_id or new_node_id()
    with db.transaction():
        db.conn.execute(
            "INSERT INTO graph_nodes (node_id, type, label, family, mech_tags, summary,"
            " as_of, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                node_id,
                type,
                label,
                family,
                json.dumps(mech_tags or []),
                summary,
                as_of,
                now_iso(),
            ),
        )
    node = get_node(db, node_id)
    embed_text(db, embedder, "graph_node", node_id, node_text(node))
    return node


def get_node(db: Database, node_id: str) -> dict:
    row = db.conn.execute(
        "SELECT * FROM graph_nodes WHERE node_id = ?", (node_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown node_id {node_id!r}")
    d = dict(row)
    d["mech_tags"] = json.loads(d["mech_tags"])
    return d


def find_node_by_label(db: Database, label: str) -> dict | None:
    key = normalize_name(label)
    rows = db.conn.execute("SELECT node_id, label FROM graph_nodes").fetchall()
    for r in rows:
        if normalize_name(r["label"]) == key:
            return get_node(db, r["node_id"])
    return None


def add_edge(
    db: Database,
    src: str,
    dst: str,
    predicate: str,
    weight: float | None = None,
    evidence_episode_ids: list[str] | None = None,
    as_of: str | None = None,
) -> dict:
    """Insert a graph edge. Predicate must be one of the 6 active (§2.7);
    evidence episode links ride along for provenance (M5)."""
    if predicate not in PREDICATES:
        raise ValueError(
            f"unknown predicate {predicate!r}; active predicates are {sorted(PREDICATES)}"
        )
    for endpoint in (src, dst):
        if db.conn.execute(
            "SELECT 1 FROM graph_nodes WHERE node_id = ?", (endpoint,)
        ).fetchone() is None:
            raise ValueError(f"unknown endpoint node_id {endpoint!r}")
    as_of = validate_iso(as_of) if as_of else now_iso()
    edge_id = new_edge_id()
    db.execute(
        "INSERT INTO graph_edges (edge_id, src, dst, predicate, weight,"
        " evidence_episode_ids, valid_from, valid_to, as_of, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)",
        (
            edge_id,
            src,
            dst,
            predicate,
            weight,
            json.dumps(evidence_episode_ids or []),
            as_of,
            as_of,
            now_iso(),
        ),
    )
    return get_edge(db, edge_id)


def get_edge(db: Database, edge_id: str) -> dict:
    row = db.conn.execute(
        "SELECT * FROM graph_edges WHERE edge_id = ?", (edge_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown edge_id {edge_id!r}")
    d = dict(row)
    d["evidence_episode_ids"] = json.loads(d["evidence_episode_ids"])
    return d


def edge_profile(db: Database, node_id: str, as_of: str) -> set[tuple[str, str]]:
    """The structural fingerprint of a node: its visible (predicate, dst_type)
    edge set, per the §7 structural-overlap definition. Empty when ungrounded."""
    rows = db.conn.execute(
        "SELECT e.predicate AS predicate, n.type AS dst_type"
        " FROM graph_edges e JOIN graph_nodes n ON n.node_id = e.dst"
        " WHERE e.src = ? AND e.valid_from <= ?"
        " AND (e.valid_to IS NULL OR ? < e.valid_to) AND e.as_of <= ?",
        (node_id, as_of, as_of, as_of),
    ).fetchall()
    return {(r["predicate"], r["dst_type"]) for r in rows}


# -- seed import (§16.1) ------------------------------------------------------


def _load_seed_cards(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if not isinstance(data, list):
        raise ValueError(f"seed file {path} must contain a YAML list of cards")
    return data


def import_seeds(
    db: Database,
    embedder: Any | None = None,
    seeds_dir: str | Path | None = None,
    as_of: str | None = None,
) -> dict:
    """Load seeds/techniques.yaml + seeds/concepts.yaml into the graph.

    Technique cards become type='technique' nodes; concept cards become
    type='concept' nodes. Edges are derived from card fields:
    composes_with -> COMPOSES_WITH (both directions; the relation is
    symmetric), assumptions -> ASSUMES, failure_modes -> FAILS_WHEN, each
    pointing at the matching card/concept node. References that do not
    resolve are counted and reported, not fatal. Every node is embedded on
    write when an embedder is configured.
    """
    directory = Path(seeds_dir) if seeds_dir else SEEDS_DIR
    # Seeds are timeless: an early valid timestamp (year <1000 breaks the
    # zero-padded strftime round-trip in validate_iso, so not MIN_ISO).
    as_of = validate_iso(as_of) if as_of else "1000-01-01T00:00:00Z"
    techniques = _load_seed_cards(directory / "techniques.yaml")
    concepts = _load_seed_cards(directory / "concepts.yaml")

    stats = {
        "techniques": len(techniques),
        "concepts": len(concepts),
        "nodes": 0,
        "edges": 0,
        "embeddings": 0,
        "unresolved_refs": 0,
        "skipped_existing": 0,
    }
    id_by_key: dict[str, str] = {}

    with db.transaction():
        for card in concepts:
            key = normalize_name(card["name"])
            existing = find_node_by_label(db, card["name"])
            if existing is not None:
                id_by_key[key] = existing["node_id"]
                stats["skipped_existing"] += 1
                continue
            node = add_node(
                db,
                type="concept",
                label=card["name"],
                mech_tags=card.get("mechanism_tags") or [],
                summary=card.get("summary"),
                family=None,
                as_of=as_of,
                embedder=embedder,
            )
            id_by_key[key] = node["node_id"]
            stats["nodes"] += 1
            if embedder is not None:
                stats["embeddings"] += 1

        for card in techniques:
            key = normalize_name(card["name"])
            existing = find_node_by_label(db, card["name"])
            if existing is not None:
                id_by_key[key] = existing["node_id"]
                stats["skipped_existing"] += 1
                continue
            mech_tags = list(card.get("mechanism_tags") or [])
            for field in ("assumptions", "failure_modes", "analog_domains"):
                mech_tags.extend(card.get(field) or [])
            summary = "; ".join(
                f"{k}: {', '.join(card.get(k) or [])}"
                for k in ("assumptions", "failure_modes")
                if card.get(k)
            ) or None
            node = add_node(
                db,
                type="technique",
                label=card["name"],
                mech_tags=mech_tags,
                summary=summary,
                family=card.get("family"),
                as_of=as_of,
                embedder=embedder,
            )
            id_by_key[key] = node["node_id"]
            stats["nodes"] += 1
            if embedder is not None:
                stats["embeddings"] += 1

        def _resolve(ref: str) -> str | None:
            return id_by_key.get(normalize_name(ref))

        for card in techniques:
            src = _resolve(card["name"])
            assert src is not None
            for ref in card.get("composes_with") or []:
                dst = _resolve(ref)
                if dst is None:
                    stats["unresolved_refs"] += 1
                    continue
                for a, b in ((src, dst), (dst, src)):
                    add_edge(db, a, b, "COMPOSES_WITH", as_of=as_of)
                    stats["edges"] += 1
            for predicate, field in (("ASSUMES", "assumptions"), ("FAILS_WHEN", "failure_modes")):
                for ref in card.get(field) or []:
                    dst = _resolve(ref)
                    if dst is None:
                        stats["unresolved_refs"] += 1
                        continue
                    add_edge(db, src, dst, predicate, as_of=as_of)
                    stats["edges"] += 1

    return stats
