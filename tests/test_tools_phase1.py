"""Phase-1 tools over the real MCP protocol (fastmcp in-memory client, temp db):
the full propose -> register -> log -> promote -> recall/history -> report flow,
plus tool-level refusals and error paths."""

import asyncio
import json

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from spark_mcp.config import Config
from spark_mcp.server import create_server

PROPOSE_ARGS = dict(
    idea="study reversal after forced rebalancing",
    family="momentum",
    universe="us_equity",
    horizon="20d",
    scope="cross_section",
    mechanism="forced rebalancing drives short term reversal",
    expected_direction="negative",
    falsifier="reversal disappears net of transaction costs",
    predicted_ic=0.03,
    predicted_sigma=0.01,
    hypothesis_type="predictive",
)


def _server(tmp_path):
    cfg = Config(
        storage={"db_path": f"{tmp_path}/spark.db", "artifacts_dir": f"{tmp_path}/art"}
    )
    return create_server(cfg)


def test_phase1_tools_full_flow(tmp_path):
    async def run():
        async with Client(_server(tmp_path)) as client:
            health = json.loads((await client.call_tool("spark_health", {})).content[0].text)
            assert health["schema_version"] == 7  # Phase 5: migration v7 adds qd_cells

            prop = json.loads(
                (await client.call_tool("hypothesis_propose", PROPOSE_ARGS)).content[0].text
            )
            assert prop["hypothesis_id"].startswith("hyp_")

            reg = json.loads(
                (
                    await client.call_tool(
                        "register_hypothesis",
                        {
                            "hypothesis_id": prop["hypothesis_id"],
                            "spec": {"signal": "reversal"},
                            "seed": 11,
                        },
                    )
                ).content[0].text
            )
            assert reg["episode_id"].startswith("ep_")

            out = json.loads(
                (
                    await client.call_tool(
                        "log_outcome",
                        {
                            "episode_id": reg["episode_id"],
                            "outcome": "failure",
                            "failure_reason": "sharpe negative after costs",
                            "metrics": {"net_sharpe": -0.4},
                            "costs": {"usd": 0.5, "tokens": 300},
                            "reward": -0.2,
                        },
                    )
                ).content[0].text
            )
            assert out["trial_source"] == "reconciled"  # C4 backfill
            assert out["tombstone_id"]

            promote = json.loads(
                (
                    await client.call_tool(
                        "memory_promote",
                        {
                            "episode_ids": [reg["episode_id"]],
                            "claim": "forced rebalancing reversal is not tradeable",
                            "confidence": 0.9,
                            "scope": "cross_section",
                            "rationale": "out-of-sample failure after costs",
                        },
                    )
                ).content[0].text
            )
            assert promote["promotion_status"] == "auto"

            recall = json.loads(
                (await client.call_tool("memory_recall", {"query": "rebalancing"})).content[0].text
            )
            assert any(
                b["belief_id"] == promote["belief_id"] for b in recall["beliefs"]
            )
            assert recall["beliefs"][0]["supporting_runs"] == [reg["episode_id"]]

            history = json.loads(
                (
                    await client.call_tool(
                        "research_history", {"query": "sharpe", "kinds": ["tombstones"]}
                    )
                ).content[0].text
            )
            assert history["tombstones"][0]["tombstone_id"] == out["tombstone_id"]

            report = json.loads(
                (
                    await client.call_tool(
                        "report_write", {"episode_id": reg["episode_id"]}
                    )
                ).content[0].text
            )
            assert report["format"] == "md"
            assert reg["episode_id"] in report["markdown"]
            assert "## Trials" in report["markdown"]
            assert "reconciled/completed: 1" in report["markdown"]
            assert "$0.5000 usd" in report["markdown"]

            report_h = json.loads(
                (
                    await client.call_tool(
                        "report_write", {"hypothesis_id": prop["hypothesis_id"]}
                    )
                ).content[0].text
            )
            assert "## Hypothesis" in report_h["markdown"]
            assert reg["episode_id"] in report_h["markdown"]

    asyncio.run(run())


def test_tool_refusals_and_error_paths(tmp_path):
    async def run():
        async with Client(_server(tmp_path)) as client:
            with pytest.raises(ToolError, match="C6"):
                await client.call_tool(
                    "hypothesis_propose", {**PROPOSE_ARGS, "hypothesis_type": "causal"}
                )
            with pytest.raises(ToolError, match="exactly one"):
                await client.call_tool("report_write", {})
            with pytest.raises(ToolError, match="exactly one"):
                await client.call_tool(
                    "report_write",
                    {"hypothesis_id": "hyp_x", "episode_id": "ep_x"},
                )
            with pytest.raises(ToolError, match="md"):
                await client.call_tool(
                    "report_write", {"hypothesis_id": "hyp_x", "format": "html"}
                )
            with pytest.raises(ToolError, match="unknown hypothesis_id"):
                await client.call_tool(
                    "report_write", {"hypothesis_id": "hyp_missing"}
                )
            with pytest.raises(ToolError, match="invalid ISO"):
                await client.call_tool(
                    "memory_recall", {"query": "x", "as_of": "not-a-date"}
                )

    asyncio.run(run())


def test_tools_accept_explicit_as_of(tmp_path):
    async def run():
        async with Client(_server(tmp_path)) as client:
            await client.call_tool(
                "hypothesis_propose",
                {**PROPOSE_ARGS, "as_of": "2020-06-01T00:00:00Z"},
            )
            hist = json.loads(
                (
                    await client.call_tool(
                        "research_history",
                        {"query": "rebalancing", "as_of": "2019-01-01T00:00:00Z"},
                    )
                ).content[0].text
            )
            assert hist["episodes"] == []  # PIT: proposal not yet visible

    asyncio.run(run())
