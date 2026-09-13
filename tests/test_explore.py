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
    """G1: repeat-query overlap < 30% at balanced — measured as mean pairwise
    Jaccard-overlap of suggestion sets across 5 unseeded repeats."""

    async def run():
        async with Client(_server(tmp_path)) as client:
            sets = []
            for _ in range(5):
                out = await _explore(
                    client, CANONICAL_PROBLEMS[0], k=5, mode="balanced"
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
    # deterministic scoring layer: causal candidates are retrieved & scored
    causal_candidates = [c for c in out["candidates"] if c["family"] == "causal"]
    assert len(causal_candidates) >= 2
    # selection layer: at least one causal-family suggestion surfaces
    assert any(s["family"] == "causal" for s in out["suggestions"])


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
