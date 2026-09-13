"""Hybrid retrieval: RRF fusion, PIT-before-ranking, degraded FTS5-only path."""

import numpy as np
import pytest

from spark_mcp import graph, retrieval
from spark_mcp.storage import Database, open_db

AS_OF = "2024-01-01T00:00:00Z"


class DictEmbedder:
    """Test embedder with exact, hand-controlled vectors (no randomness)."""

    def __init__(self, vectors: dict[str, list[float]], dim: int = 3):
        self.vectors = vectors
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim))
        for i, t in enumerate(texts):
            out[i] = self.vectors.get(t, np.zeros(self.dim))
        return out

    def model_id(self) -> str:
        return "dict-embedder@1.0"


def _craft_db(tmp_path) -> tuple[Database, DictEmbedder, dict[str, str]]:
    """Three nodes engineered so the vector source ranks N_a first and the
    FTS source ranks N_b first, while N_star is second in BOTH — the classic
    case where fusion should beat either single source."""
    db = open_db(tmp_path / "retrieval.db")
    emb = DictEmbedder(
        {
            "alpha": [1.0, 0.0, 0.0],
            "N_a vec_a": [0.9, 0.1, 0.0],
            "alpha frontier vec_star": [0.7, 0.3, 0.0],
            "alpha far_vec": [0.0, 0.0, 1.0],
            "zztop future_vec": [1.0, 0.0, 0.0],
        }
    )
    ids = {}
    ids["N_a"] = graph.add_node(
        db, "technique", "N_a", summary="vec_a", as_of=AS_OF, embedder=emb
    )["node_id"]
    ids["N_star"] = graph.add_node(
        db, "technique", "alpha frontier", summary="vec_star", as_of=AS_OF, embedder=emb
    )["node_id"]
    ids["N_b"] = graph.add_node(
        db, "technique", "alpha", summary="far_vec", as_of=AS_OF, embedder=emb
    )["node_id"]
    return db, emb, ids


def test_rrf_fuse_math():
    fused = retrieval.rrf_fuse([["a", "b"], ["b", "c"]], rrf_k=60)
    assert fused[0][0] == "b"  # 1/62 (rank 2) + 1/61 (rank 1) beats one mention
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_fusion_beats_either_single_source(tmp_path):
    db, emb, ids = _craft_db(tmp_path)
    vec_ranking = retrieval._vector_ranking(db, "alpha", emb, AS_OF, k=2)
    fts_ranking = retrieval._fts_ranking(db, "alpha", AS_OF, k=2)
    assert vec_ranking[0] == ids["N_a"] and ids["N_star"] in vec_ranking
    assert fts_ranking[0] == ids["N_b"] and ids["N_star"] in fts_ranking

    fused = retrieval.hybrid_recall(db, "alpha", AS_OF, embedder=emb, k_vec=2, k_fts=2)
    assert fused[0]["node_id"] == ids["N_star"]  # fusion lifts the doubly-supported node
    # single-source winners never topped both lists; fusion outranks them
    assert fused[1]["node_id"] in (ids["N_a"], ids["N_b"])
    assert fused[0]["sources"] == ["vector", "fts5"]
    db.close()


def test_pit_filter_applies_before_ranking(tmp_path):
    db, emb, ids = _craft_db(tmp_path)
    # a FUTURE node whose embedding is identical to the query vector — it
    # would rank #1 in the vector source if leakage happened after ranking
    future = graph.add_node(
        db, "technique", "zztop", summary="future_vec", as_of="2999-01-01T00:00:00Z",
        embedder=emb,
    )
    vec_ranking = retrieval._vector_ranking(db, "alpha", emb, AS_OF, k=3)
    assert future["node_id"] not in vec_ranking
    fused = retrieval.hybrid_recall(db, "alpha", AS_OF, embedder=emb, k_vec=3, k_fts=3)
    assert future["node_id"] not in [n["node_id"] for n in fused]
    # visible again once as_of reaches it
    fused_late = retrieval.hybrid_recall(
        db, "alpha", "3000-01-01T00:00:00Z", embedder=emb, k_vec=3, k_fts=3
    )
    assert future["node_id"] in [n["node_id"] for n in fused_late]
    db.close()


def test_degraded_mode_is_fts5_only(tmp_path):
    db, emb, ids = _craft_db(tmp_path)
    fused = retrieval.hybrid_recall(db, "alpha", AS_OF, embedder=None)
    assert fused, "FTS5-only path must still return matches"
    assert all(n["sources"] == ["fts5"] for n in fused)
    assert ids["N_a"] not in [n["node_id"] for n in fused]  # vector-only node out
    db.close()


def test_degraded_mode_with_unmatchable_query_returns_empty(tmp_path):
    db, _, _ = _craft_db(tmp_path)
    assert retrieval.hybrid_recall(db, "qqqq", AS_OF, embedder=None) == []


def test_hybrid_recall_empty_db_no_crash(tmp_path):
    db = open_db(tmp_path / "empty.db")
    assert retrieval.hybrid_recall(db, "anything", AS_OF, embedder=None) == []
    db.close()


def test_seeded_graph_recall_is_provenance_carrying(seeded, embedder):
    out = retrieval.hybrid_recall(
        seeded, "volatility clustering regime detection", AS_OF, embedder=embedder
    )
    assert out
    for node in out:
        assert node["node_id"].startswith("node_")
        assert "sources" in node and node["sources"]
