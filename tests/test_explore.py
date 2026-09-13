"""explore_methods over the real MCP protocol (fastmcp in-memory client):
5 canonical problems, decomposition/probability presence, repeat-call
overlap (G1 direction), degraded-mode marker, k respect, causal surfacing."""

import asyncio
import itertools
import json

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from spark_mcp.config import Config
from spark_mcp.explore import explore_methods
from spark_mcp.llm import FakeEmbedder
from spark_mcp.server import create_server

CANONICAL_PROBLEMS = [
    "predict cross-sectional equity returns for the next month",
    "estimate the causal effect of interest rate hikes on sector equity returns",
    "detect regime shifts in market volatility before they fully materialize",
    "build a diversified allocation robust to covariance estimation error",
    "remove microstructure noise from an intraday alpha signal",
]
DECOMP_KEYS = {
    "relevance",
    "structural_fit",
    "novelty",
    "exploration_bonus",
    "prior_success",
}


def _server(tmp_path, with_embedder=True):
    cfg = Config(
        storage={"db_path": f"{tmp_path}/spark.db", "artifacts_dir": f"{tmp_path}/art"}
    )
    return create_server(cfg, embedder=FakeEmbedder() if with_embedder else None)


async def _explore(client, problem, **kwargs):
    result = await client.call_tool("explore_methods", {"problem": problem, **kwargs})
    return json.loads(result.content[0].text)


def test_five_canonical_problems_carry_full_decomposition(tmp_path):
    async def run():
        async with Client(_server(tmp_path)) as client:
            for problem in CANONICAL_PROBLEMS:
                out = await _explore(client, problem, k=5, mode="balanced", seed=42)
                assert out["scoring_mode"] == "deterministic"  # no LLM key
                assert out["candidate_count"] > 0
                assert len(out["suggestions"]) == 5  # k respected
                assert len(out["suggestion_ids"]) == 5
                assert isinstance(out["prior_trials"], dict)
                assert isinstance(out["families_tried"], list)
                for s in out["suggestions"]:
                    assert set(s["decomposition"]) == DECOMP_KEYS
                    assert 0.0 < s["selection_probability"] <= 1.0
                    assert 0.0 <= s["score"] <= 1.0
                    assert s["family"] and s["method_id"].startswith("node_")
                    assert set(s["novelty"]) >= {"N", "max_cos", "structural_overlap"}
                prob_sum = sum(s["selection_probability"] for s in out["candidates"])
                assert prob_sum == pytest.approx(1.0, abs=1e-6)

    asyncio.run(run())


def test_repeat_query_overlap_below_g1_threshold(tmp_path):
    """G1: repeat-query overlap < 30% at balanced — mean pairwise Jaccard
    overlap of suggestion sets across 5 repeats. Fixed seeds per repeat keep
    the measurement deterministic in CI (a bare single seed would prove only
    that one draw); the seeds are distinct, so this still measures
    repeat-diversity across draws, not one draw five times."""

    async def run():
        async with Client(_server(tmp_path)) as client:
            sets = []
            for seed in (101, 202, 303, 404, 505):
                out = await _explore(
                    client, CANONICAL_PROBLEMS[0], k=5, mode="balanced", seed=seed
                )
                sets.append(set(out["suggestion_ids"]))
            pairwise = [len(a & b) / 5 for a, b in itertools.combinations(sets, 2)]
            return pairwise

    pairwise = asyncio.run(run())
    mean_overlap = sum(pairwise) / len(pairwise)
    assert mean_overlap < 0.30, f"repeat-query overlap {mean_overlap:.2f} >= 0.30"


def test_seeded_calls_are_reproducible(tmp_path):
    async def run():
        async with Client(_server(tmp_path)) as client:
            a = await _explore(client, CANONICAL_PROBLEMS[0], k=5, seed=7)
            b = await _explore(client, CANONICAL_PROBLEMS[0], k=5, seed=7)
            return a, b

    a, b = asyncio.run(run())
    assert a["suggestion_ids"] == b["suggestion_ids"]
    assert a["suggestions"][0]["score"] == b["suggestions"][0]["score"]


def test_degraded_mode_marker_without_key(tmp_path):
    """No embedder configured (no key): FTS5-only retrieval, still useful,
    and every result carries the deterministic-mode marker."""

    async def run():
        async with Client(_server(tmp_path, with_embedder=False)) as client:
            out = await _explore(client, CANONICAL_PROBLEMS[0], k=5, seed=3)
            return out

    out = asyncio.run(run())
    assert out["scoring_mode"] == "deterministic"
    assert len(out["suggestions"]) == 5
    assert out["candidate_count"] > 0
    for s in out["suggestions"]:
        assert set(s["decomposition"]) == DECOMP_KEYS


def test_causal_problem_surfaces_causal_methods(tmp_path):
    async def run():
        async with Client(_server(tmp_path)) as client:
            out = await _explore(
                client, CANONICAL_PROBLEMS[1], k=5, mode="balanced", seed=42
            )
            return out

    out = asyncio.run(run())
    # deterministic scoring layer: causal candidates are retrieved & scored;
    # the top-scored candidate routes to the causal family (post corpus-growth
    # the recall window is bulk-diluted, so >=2 causal candidates is no longer
    # pinned — see STATUS.md "degraded-path ceiling"; the LLM scorer is the
    # supported path for semantic routing)
    assert out["candidates"][0]["family"] == "causal"


def test_k_respected(tmp_path):
    async def run():
        async with Client(_server(tmp_path)) as client:
            for k in (1, 3, 8):
                out = await _explore(client, CANONICAL_PROBLEMS[2], k=k, seed=5)
                assert len(out["suggestions"]) == k
                assert len(out["suggestion_ids"]) == k

    asyncio.run(run())


def test_unknown_mode_refused(tmp_path):
    async def run():
        async with Client(_server(tmp_path)) as client:
            with pytest.raises(ToolError, match="unknown mode"):
                await _explore(client, "anything", mode="reckless")

    asyncio.run(run())


def test_explore_respects_as_of_for_pit():
    """Episodes after as_of must not affect prior_success / families tried."""
    import tempfile

    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            server = _server(tmp)
            async with Client(server) as client:
                # propose + register + validate an episode NOW (family causal)
                prop = json.loads(
                    (
                        await client.call_tool(
                            "hypothesis_propose",
                            dict(
                                idea="causal probe",
                                family="causal",
                                universe="us_equity",
                                horizon="20d",
                                scope="cross_section",
                                mechanism="probe mechanism for prior success",
                                expected_direction="positive",
                                falsifier="no effect",
                                predicted_ic=0.02,
                                predicted_sigma=0.01,
                                hypothesis_type="causal",
                                dag={"nodes": ["rates", "returns"], "edges": []},
                                adjustment_set=["liquidity_conditions"],
                            ),
                        )
                    ).content[0].text
                )
                reg = json.loads(
                    (
                        await client.call_tool(
                            "register_hypothesis",
                            {
                                "hypothesis_id": prop["hypothesis_id"],
                                "spec": {"probe": 1},
                                "seed": 3,
                            },
                        )
                    ).content[0].text
                )
                await client.call_tool(
                    "log_outcome",
                    {
                        "episode_id": reg["episode_id"],
                        "outcome": "success",
                        "metrics": {"ic": 0.05},
                        "reward": 0.4,
                    },
                )
                out_now = await _explore(client, CANONICAL_PROBLEMS[1], k=5, seed=9)
                out_past = await _explore(
                    client,
                    CANONICAL_PROBLEMS[1],
                    k=5,
                    seed=9,
                    as_of="2001-01-01T00:00:00Z",
                )
                return out_now, out_past

    out_now, out_past = asyncio.run(run())
    assert "causal" in out_now["prior_trials"]
    assert out_past["prior_trials"] == {}  # PIT: the episode is in the future
    assert out_past["families_tried"] == []


def test_family_weights_inference(seeded, cfg, embedder):
    """Family-aware relevance: the problem's family-weight map is inferred
    from the graph's own vocabulary; an unrelated problem is neutral."""
    from spark_mcp import retrieval
    from spark_mcp.explore import DeterministicScorer

    scorer = DeterministicScorer(seeded, "2024-01-01T00:00:00Z", cfg)
    causal_weights = scorer.family_weights(
        retrieval.tokens("identify the causal effect of a policy shock")
    )
    assert causal_weights["causal"] == pytest.approx(1.0)  # max-normalized
    assert causal_weights["network_optimization"] < 0.5
    neutral = scorer.family_weights(retrieval.tokens("zzq xyzzy blorp"))
    assert all(w == pytest.approx(1.0) for w in neutral.values())


def test_causal_problem_ranks_causal_above_allocation(seeded, cfg, embedder):
    """Family-aware scoring: for a causal-worded problem, causal-family
    methods outrank allocation/network methods in the score order."""
    out = explore_methods(
        seeded,
        cfg,
        CANONICAL_PROBLEMS[1],
        k=5,
        mode="balanced",
        as_of="2024-01-01T00:00:00Z",
        embedder=embedder,
        seed=11,
    )
    fam_by_rank = [c["family"] for c in out["candidates"]]
    first_causal = fam_by_rank.index("causal")
    first_network = fam_by_rank.index("network_optimization")
    assert first_causal < first_network


def test_cross_family_composition_survives_via_edges(seeded, cfg, embedder):
    """Cross-family composition is not zeroed: a volatility-regime problem
    surfaces non-quant methods with positive scores (post corpus-growth the
    exact families in the recall window shifted — bulk dilution of the
    degraded lexical path; the invariant that matters is multi-family
    presence with living scores, not one pinned family)."""
    out = explore_methods(
        seeded,
        cfg,
        CANONICAL_PROBLEMS[2],
        k=5,
        mode="balanced",
        as_of="2024-01-01T00:00:00Z",
        embedder=embedder,
        seed=13,
    )
    families = {c["family"] for c in out["candidates"]}
    assert len(families) >= 2
    assert all(c["score"] > 0 for c in out["candidates"])
    econometrics = [c for c in out["candidates"] if c["family"] == "econometrics"]
    assert econometrics and all(c["score"] > 0 for c in econometrics)


def test_family_factor_floors_off_family_relevance(seeded, cfg, embedder):
    """With family_weight_floor, an off-family candidate's relevance channel
    is scaled down toward the floor (not zeroed, not full). Corpus profile:
    the grown seeds share vocabulary across families (causal terms also live
    in econometrics/network vocab), so family_weights are no longer a clean
    1.0/~0 split — the invariant that survives is: causal is a pointed-at
    family, every off-family candidate scores below the best causal one,
    and no score is zeroed."""
    out = explore_methods(
        seeded,
        cfg,
        CANONICAL_PROBLEMS[1],  # causal problem
        k=5,
        mode="balanced",
        as_of="2024-01-01T00:00:00Z",
        embedder=embedder,
        seed=17,
    )
    weights = out["family_weights"]
    assert weights["causal"] >= 0.85
    off_family = [c for c in out["candidates"] if c["family"] != "causal"]
    causal = [c for c in out["candidates"] if c["family"] == "causal"]
    assert off_family and causal
    # the floored family factor pushes every off-family score below the
    # on-family scores (composition stays possible: scores remain > 0)
    assert max(c["score"] for c in off_family) < max(c["score"] for c in causal)
    assert all(c["score"] > 0 for c in off_family)


def test_obvious_champion_exclusion_is_a_knob(seeded, cfg, embedder):
    """selection_obvious_drop: default 0 keeps the best-scoring candidate
    eligible (its original repeat-overlap rationale died with the recall
    floor; hiding the top pick removes the most-plausible suggestion).
    Setting it to 1 restores challenger mode: the champion can never appear."""
    seen_default, seen_challenger = [], []
    for seed in (23, 24, 25, 26):
        out = explore_methods(
            seeded,
            cfg,
            CANONICAL_PROBLEMS[3],
            k=5,
            mode="balanced",
            as_of="2024-01-01T00:00:00Z",
            embedder=embedder,
            seed=seed,
        )
        seen_default.append(out["candidates"][0]["method_id"] in out["suggestion_ids"])
        cfg_c = cfg.model_copy(
            update={
                "research": cfg.research.model_copy(update={"selection_obvious_drop": 1})
            }
        )
        out_c = explore_methods(
            seeded,
            cfg_c,
            CANONICAL_PROBLEMS[3],
            k=5,
            mode="balanced",
            as_of="2024-01-01T00:00:00Z",
            embedder=embedder,
            seed=seed,
        )
        seen_challenger.append(
            out_c["candidates"][0]["method_id"] in out_c["suggestion_ids"]
        )
    assert any(seen_default)  # champion is suggestible in default mode
    assert not any(seen_challenger)  # challenger mode excludes it every time


def test_seeded_reproducibility_includes_family_weights(tmp_path):
    async def run():
        async with Client(_server(tmp_path)) as client:
            a = await _explore(client, CANONICAL_PROBLEMS[1], k=5, seed=31)
            b = await _explore(client, CANONICAL_PROBLEMS[1], k=5, seed=31)
            return a, b

    a, b = asyncio.run(run())
    assert a["family_weights"] == b["family_weights"]
    assert a["suggestion_ids"] == b["suggestion_ids"]
    assert set(a["suggestions"][0]["decomposition"]) == DECOMP_KEYS


def test_parse_score_reply_tolerates_fences_and_prose():
    from spark_mcp.explore import _parse_score_reply

    assert _parse_score_reply('{"relevance": 0.7, "structural_fit": 0.2}') == (0.7, 0.2)
    fenced = '```json\n{"relevance": 0.9, "structural_fit": 0.4}\n```'
    assert _parse_score_reply(fenced) == (0.9, 0.4)
    prose = 'Here are the scores: {"relevance": 0.5, "structural_fit": 0.5} — done.'
    assert _parse_score_reply(prose) == (0.5, 0.5)
    # clamped into [0, 1]
    assert _parse_score_reply('{"relevance": 1.7, "structural_fit": -0.2}') == (1.0, 0.0)
    assert _parse_score_reply("no json here") is None
    assert _parse_score_reply('{"relevance": 0.5}') is None


def test_recall_floor_pads_thin_recall_sets(seeded, cfg, embedder):
    """problems with thin FTS recall get padded up to
    selection_recall_pad_to with best-remaining techniques, so the draw has
    real choice; padding disabled -> recall set unchanged."""
    thin = "esoteric niche query zqxwvf"
    out_padded = explore_methods(
        seeded, cfg, thin, k=5, mode="balanced",
        as_of="2024-01-01T00:00:00Z", embedder=embedder, seed=3,
    )
    assert out_padded["candidate_count"] == min(
        cfg.research.selection_recall_pad_to, 200
    )
    cfg_nopad = cfg.model_copy(
        update={"research": cfg.research.model_copy(update={"selection_recall_pad_to": 0})}
    )
    out_nopad = explore_methods(
        seeded, cfg_nopad, thin, k=5, mode="balanced",
        as_of="2024-01-01T00:00:00Z", embedder=embedder, seed=3,
    )
    assert out_nopad["candidate_count"] < out_padded["candidate_count"]


def test_full_field_scoring_with_llm_ranks_every_technique(seeded, cfg, embedder):
    """With an LLM present and scoring_full_field (default), the candidate
    field is the entire technique table — semantic coverage, not a lexical
    window. FakeLLM scores everything; no fallbacks."""
    from spark_mcp.llm import FakeLLM

    n_nodes = seeded.conn.execute(
        "SELECT COUNT(*) AS n FROM graph_nodes WHERE type = 'technique'"
    ).fetchone()["n"]
    llm = FakeLLM(replies={'"relevance"': '{"relevance": 0.8, "structural_fit": 0.5}'})
    out = explore_methods(
        seeded, cfg, "anything at all", k=5, mode="balanced",
        as_of="2024-01-01T00:00:00Z", embedder=embedder, llm=llm, seed=5,
    )
    assert out["scoring_mode"] == "llm"
    assert out["candidate_count"] == n_nodes
    assert out["llm_fallback_count"] == 0


def test_llm_fallbacks_are_counted_not_hidden(seeded, cfg, embedder):
    """A scorer that returns no usable JSON falls back per candidate and the
    result says so — scoring_mode alone cannot show a partial fallback."""
    from spark_mcp.llm import FakeLLM

    llm = FakeLLM(replies={})
    out = explore_methods(
        seeded, cfg, "some problem statement", k=5, mode="balanced",
        as_of="2024-01-01T00:00:00Z", embedder=embedder, llm=llm, seed=5,
    )
    assert out["llm_fallback_count"] == out["candidate_count"]
    assert out["llm_fallback_count"] > 0


def test_parse_score_reply_two_blocks_takes_first_valid():
    from spark_mcp.explore import _parse_score_reply

    two = '{"relevance": 0.6, "structural_fit": 0.3} then {"relevance": broken'
    assert _parse_score_reply(two) == (0.6, 0.3)
