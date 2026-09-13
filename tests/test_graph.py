"""Method graph: predicate validation, seed import, FTS5, embeddings, PIT."""

import pytest

from spark_mcp.graph import PREDICATES, add_edge, add_node, get_node, import_seeds
from spark_mcp.timeutil import now_iso

ALL_SIX = [
    "SOLVES",
    "FAILS_WHEN",
    "COMPOSES_WITH",
    "DIAGNOSES",
    "ASSUMES",
    "DERIVED_FROM",
]


def test_exactly_the_six_active_predicates():
    assert sorted(PREDICATES) == sorted(ALL_SIX)


def test_add_edge_rejects_inactive_predicates(db, embedder):
    a = add_node(db, "technique", "Method A", embedder=embedder)
    b = add_node(db, "concept", "Concept B", embedder=embedder)
    with pytest.raises(ValueError, match="unknown predicate"):
        add_edge(db, a["node_id"], b["node_id"], "CAUSES")  # §2.7 rejected-with-message
    with pytest.raises(ValueError, match="unknown predicate"):
        add_edge(db, a["node_id"], b["node_id"], "correlates_with")


def test_add_edge_accepts_all_active_predicates(db, embedder):
    a = add_node(db, "technique", "Method A", embedder=embedder)
    b = add_node(db, "concept", "Concept B", embedder=embedder)
    for pred in ALL_SIX:
        edge = add_edge(db, a["node_id"], b["node_id"], pred, weight=0.5)
        assert edge["predicate"] == pred
        assert edge["weight"] == 0.5


def test_add_edge_rejects_unknown_endpoints(db):
    with pytest.raises(ValueError, match="unknown endpoint"):
        add_edge(db, "node_missing", "node_missing2", "SOLVES")


def test_add_edge_carries_evidence_episode_links(db, embedder):
    a = add_node(db, "technique", "Method A", embedder=embedder)
    b = add_node(db, "concept", "Concept B", embedder=embedder)
    edge = add_edge(db, a["node_id"], b["node_id"], "SOLVES", evidence_episode_ids=["ep_1"])
    assert edge["evidence_episode_ids"] == ["ep_1"]


def test_seed_import_counts_nodes_edges_embeddings(seeded, embedder):
    n_nodes = seeded.conn.execute("SELECT COUNT(*) FROM graph_nodes").fetchone()[0]
    n_edges = seeded.conn.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0]
    n_emb = seeded.conn.execute("SELECT COUNT(*) FROM vec_embeddings").fetchone()[0]
    assert 135 <= n_nodes <= 175  # ~101 technique cards + ~40-52 concept nodes
    assert n_edges > 0
    assert n_emb == n_nodes  # embed-on-write: every seeded node embedded
    families = {
        r[0]
        for r in seeded.conn.execute(
            "SELECT DISTINCT family FROM graph_nodes WHERE family IS NOT NULL"
        ).fetchall()
    }
    assert {
        "quant",
        "econometrics",
        "causal",
        "signal_processing",
        "network_optimization",
    } <= families


def test_seed_embeddings_record_model_version(seeded):
    rows = seeded.conn.execute(
        "SELECT DISTINCT model, model_version FROM vec_embeddings"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["model"] == "fake-embedder"
    assert rows[0]["model_version"] == "0.1"


def test_seed_import_is_idempotent(seeded, embedder):
    before = seeded.conn.execute("SELECT COUNT(*) FROM graph_nodes").fetchone()[0]
    stats = import_seeds(seeded, embedder)
    after = seeded.conn.execute("SELECT COUNT(*) FROM graph_nodes").fetchone()[0]
    assert stats["nodes"] == 0
    assert stats["skipped_existing"] == before
    assert after == before


def test_fts5_label_search_on_seeded_graph(seeded):
    rows = seeded.conn.execute(
        "SELECT label FROM graph_nodes_fts WHERE graph_nodes_fts MATCH 'momentum'"
        " ORDER BY label LIMIT 5"
    ).fetchall()
    labels = [r["label"] for r in rows]
    assert "Cross-sectional momentum" in labels


def test_fts5_matches_mechanism_vocabulary(seeded):
    # labels alone never carry the mechanism vocabulary; the extended FTS
    # columns keep degraded recall able to find methods by mechanism.
    rows = seeded.conn.execute(
        "SELECT label FROM graph_nodes_fts WHERE graph_nodes_fts MATCH 'causal'"
    ).fetchall()
    assert {r["label"] for r in rows} >= {"Confounding", "Identification"}


def test_node_as_of_controls_pit_visibility(db, embedder):
    past = "2020-01-01T00:00:00Z"
    add_node(db, "technique", "Old method", as_of=past, embedder=embedder)
    add_node(
        db,
        "technique",
        "Future method",
        as_of="2999-01-01T00:00:00Z",
        embedder=embedder,
    )
    labels_now = {
        r[0]
        for r in db.conn.execute(
            "SELECT label FROM graph_nodes WHERE as_of <= ?", (now_iso(),)
        ).fetchall()
    }
    assert "Old method" in labels_now and "Future method" not in labels_now


def test_add_node_without_embedder_skips_embeddings(db):
    node = add_node(db, "technique", "Bare method")
    assert get_node(db, node["node_id"])["label"] == "Bare method"
    assert db.vec_table_exists() is False  # lazy table only created on embed write
