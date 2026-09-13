"""Bounded N/S (plan §7): range, identical-vs-novel, determinism, edge cases."""

import pytest

from spark_mcp import ledger, novelty
from spark_mcp.timeutil import now_iso

AS_OF = "2024-06-01T00:00:00Z"
IDEA = "cross-sectional momentum filtered by volatility regimes"
PROPOSE_FIELDS = dict(
    research_family="quant",
    hypothesis_type="predictive",
    mechanism=IDEA,
    expected_direction="positive",
    horizon="20d",
    scope="cross_section",
    falsifier="IC negative net of costs",
    predicted_ic=0.03,
    predicted_sigma=0.01,
    universe="us_equity",
    idea=IDEA,
)


def test_empty_archive_is_maximally_novel(db, cfg, embedder):
    out = novelty.check_novelty(db, cfg, IDEA, family="quant", as_of=AS_OF, embedder=embedder)
    # max_cos 0 (nothing to be similar to), structural 0 (no grounding),
    # cell empty (no episodes): N = w1·1 + w2·1 + w3·1 = 1.0
    assert out["N"] == pytest.approx(1.0)
    assert out["max_cos"] == 0.0
    assert out["structural_overlap"] == 0.0
    assert out["cell_emptiness"] == 1.0
    assert out["S"] is None
    assert out["nearest_hits"] == []


def test_n_bounded_in_unit_interval(seeded, cfg, embedder):
    ideas = [
        "momentum reversal cross-section",
        "wavelet denoising of intraday prices",
        "kalman filter for latent factor tracking",
        "graph laplacian regularized factor exposure",
        "completely orthogonal zebras radiating quasars",
    ]
    for idea in ideas:
        out = novelty.check_novelty(
            seeded, cfg, idea, family="quant", as_of=AS_OF, embedder=embedder
        )
        assert 0.0 <= out["N"] <= 1.0
        assert out["max_cos_basis"] == "embedding"


def test_identical_idea_scores_low_novelty(seeded, cfg, embedder):
    # idea="" so the archived hypothesis text is exactly IDEA (mechanism only);
    # the direct check then embeds the identical string → max_cos = 1.
    prop = ledger.propose_hypothesis(
        seeded, cfg, **{**PROPOSE_FIELDS, "idea": ""}, as_of=AS_OF, embedder=embedder
    )
    assert 0.0 <= prop["N"] <= 1.0
    out = novelty.check_novelty(seeded, cfg, IDEA, family="quant", as_of=AS_OF, embedder=embedder)
    # max_cos = 1 kills the w1 term; N is carried by the structural (0) and
    # cell terms: 0.3·1 + 0.2·1 = 0.5
    assert out["N"] == pytest.approx(0.5)
    hit = out["nearest_hits"][0]
    assert hit["subject_type"] == "hypothesis"
    assert hit["cosine"] == pytest.approx(1.0)


def test_novel_idea_scores_high_novelty(seeded, cfg, embedder):
    wild = "bismuth crystal hopper growth staircase"
    out = novelty.check_novelty(seeded, cfg, wild, family="quant", as_of=AS_OF, embedder=embedder)
    assert out["N"] > 0.7
    # Contrast against text identical to a seeded node (max_cos = 1.0, the
    # maximally un-novel case). Free-string vs free-string comparisons are
    # collision-noise with the sha256 hashing embedder — the exact-node anchor
    # is what pins "orthogonal beats corpus-known" robustly as the corpus grows.
    import json

    from spark_mcp import graph

    row = seeded.conn.execute(
        "SELECT * FROM graph_nodes WHERE label = 'Cross-sectional momentum'"
    ).fetchone()
    node = dict(row)
    node["mech_tags"] = json.loads(node["mech_tags"])
    dup = novelty.check_novelty(
        seeded, cfg, graph.node_text(node), family="quant", as_of=AS_OF, embedder=embedder
    )
    assert dup["max_cos"] == pytest.approx(1.0)
    assert dup["N"] == pytest.approx(0.5)  # N floor: w2·1 + w3·1 with max_cos = 1
    assert out["N"] > dup["N"] + 0.2


def test_check_novelty_is_deterministic(seeded, cfg, embedder):
    a = novelty.check_novelty(seeded, cfg, IDEA, family="quant", as_of=AS_OF, embedder=embedder)
    b = novelty.check_novelty(seeded, cfg, IDEA, family="quant", as_of=AS_OF, embedder=embedder)
    assert a["N"] == b["N"]
    assert [h["subject_id"] for h in a["nearest_hits"]] == [
        h["subject_id"] for h in b["nearest_hits"]
    ]


def test_degraded_mode_uses_lexical_proxy(seeded, cfg):
    out = novelty.check_novelty(seeded, cfg, IDEA, family="quant", as_of=AS_OF, embedder=None)
    assert out["max_cos_basis"] == "lexical"
    assert 0.0 <= out["N"] <= 1.0
    # lexical self-similarity of a momentum-flavored idea vs the seeded graph
    assert out["max_cos"] > 0.0


def test_surprise_formula_and_cap(cfg):
    assert novelty.surprise(cfg, realized_ic=None, predicted_ic=0.03, predicted_sigma=0.01) is None
    assert novelty.surprise(cfg, realized_ic=0.05, predicted_ic=None, predicted_sigma=0.01) is None
    assert (
        novelty.surprise(
            cfg, realized_ic=0.05, predicted_ic=0.03, predicted_sigma=0.01
        )
        == pytest.approx(2.0)
    )
    # sigma floor: tiny declared sigma still floors at cfg.research.sigma_floor
    tiny = novelty.surprise(cfg, realized_ic=0.05, predicted_ic=0.03, predicted_sigma=1e-9)
    assert tiny == pytest.approx(0.02 / cfg.research.sigma_floor)
    # cap
    huge = novelty.surprise(cfg, realized_ic=1.0, predicted_ic=0.0, predicted_sigma=0.01)
    assert huge == cfg.research.s_cap


def test_s_returned_when_realized_ic_supplied(seeded, cfg, embedder):
    out = novelty.check_novelty(
        seeded,
        cfg,
        IDEA,
        family="quant",
        as_of=AS_OF,
        embedder=embedder,
        realized_ic=0.07,
        predicted_ic=0.03,
        predicted_sigma=0.01,
    )
    assert out["S"] == pytest.approx(4.0)


def test_structural_overlap_with_grounding(seeded, cfg, embedder):
    node_id = seeded.conn.execute(
        "SELECT node_id FROM graph_nodes WHERE label = 'GARCH volatility model'"
    ).fetchone()["node_id"]
    from spark_mcp.graph import edge_profile

    edge_set = edge_profile(seeded, node_id, AS_OF)
    assert edge_set  # seeded nodes are grounded
    out = novelty.check_novelty(
        seeded, cfg, "volatility clustering model", edge_set=edge_set, as_of=AS_OF
    )
    assert out["structural_overlap"] > 0.0
    # a method structurally identical to itself has overlap 1 → kills w2 term
    assert out["structural_overlap"] == pytest.approx(1.0)


def test_families_tried_drive_cell_emptiness(db, cfg, embedder):
    out = novelty.check_novelty(db, cfg, "anything", as_of=AS_OF, embedder=embedder)
    assert out["families_tried"] == 0
    assert out["cell_emptiness"] == 1.0
    # two families TRIED — trial episodes exist for them (PIT-visible)
    for fam in ("quant", "causal"):
        hyp = ledger.propose_hypothesis(
            db,
            cfg,
            **{**PROPOSE_FIELDS, "research_family": fam, "mechanism": f"{fam} idea"},
            as_of=AS_OF,
            embedder=embedder,
        )
        ledger.register_hypothesis(db, cfg, hyp["hypothesis_id"], {"signal": fam}, seed=7)
    out2 = novelty.check_novelty(db, cfg, "anything", as_of=now_iso(), embedder=embedder)
    assert out2["families_tried"] == 2
    assert out2["cell_emptiness"] == pytest.approx(1 - 2 / cfg.research.family_count_norm)
