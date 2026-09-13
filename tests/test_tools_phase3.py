"""Phase-3 tools over the real MCP protocol (fastmcp in-memory client, temp db):
backtest_run -> significance_gate -> causal_check -> risk_decompose, tombstone
blocking + acknowledge, and the degraded no-key path end to end."""

import asyncio
import json

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from spark_mcp.config import Config
from spark_mcp.llm import FakeEmbedder
from spark_mcp.server import create_server

PROPOSE_ARGS = dict(
    idea="toy momentum evaluated by the simulated provider",
    family="momentum",
    universe="toy",
    horizon="21d",
    scope="cross_section",
    mechanism="12-1 momentum in toy universe",
    expected_direction="positive",
    falsifier="IC <= 0 net of costs",
    predicted_ic=0.03,
    predicted_sigma=0.01,
    hypothesis_type="predictive",
)


def _server(tmp_path):
    cfg = Config(
        storage={"db_path": f"{tmp_path}/spark.db", "artifacts_dir": f"{tmp_path}/art"}
    )
    return create_server(cfg, embedder=FakeEmbedder())


def _spec(hypothesis_id: str, seed: int) -> dict:
    return {
        "hypothesis_id": hypothesis_id,
        "signal_ref": "artifacts://signals/mom.json",
        "params": {"type": "momentum", "lookback": 20},
        "universe_pit": "TOYA,TOYB,TOYC,TOYD,TOYE,TOYF,TOYG,TOYH@2019-01-01",
        "dates": {
            "is_start": "2019-01-01",
            "is_end": "2019-12-31",
            "oos_start": "2020-01-01",
            "oos_end": "2020-06-30",
        },
        "costs": {"bps": 5, "slippage": 2},
        "splits": {"purge": 5, "embargo": 5},
        "seed": seed,
    }


def test_phase3_tools_full_flow(tmp_path):
    async def run():
        async with Client(_server(tmp_path)) as client:
            prop = json.loads(
                (await client.call_tool("hypothesis_propose", PROPOSE_ARGS)).content[0].text
            )
            reg = json.loads(
                (
                    await client.call_tool(
                        "register_hypothesis",
                        {
                            "hypothesis_id": prop["hypothesis_id"],
                            "spec": {"signal": "mom"},
                            "seed": 11,
                        },
                    )
                ).content[0].text
            )
            bt = json.loads(
                (
                    await client.call_tool(
                        "backtest_run",
                        {"episode_id": reg["episode_id"], "spec": _spec(prop["hypothesis_id"], 11)},
                    )
                ).content[0].text
            )
            assert bt["status"] == "testing"
            assert bt["provider"] == "simulated"
            assert bt["metrics"]["simulated"] is True
            assert {"sr", "T", "skew", "kurt"} <= set(bt["metrics"])
            assert bt["tombstone_check"]["status"] == "miss"

            # idempotent retry over MCP: same episode+spec+seed, new client_id
            bt2 = json.loads(
                (
                    await client.call_tool(
                        "backtest_run",
                        {
                            "episode_id": reg["episode_id"],
                            "spec": _spec(prop["hypothesis_id"], 11),
                            "client_id": "retry-1",
                        },
                    )
                ).content[0].text
            )
            assert bt2["reused"] is True and bt2["trial_id"] == bt["trial_id"]

            gate = json.loads(
                (
                    await client.call_tool("significance_gate", {"episode_id": reg["episode_id"]})
                ).content[0].text
            )
            assert gate["sr_reported_with_dsr"] is True
            assert gate["M_family"] == 1
            assert gate["status"] in ("validated", "inconclusive")
            assert 0.0 <= gate["p"] <= 1.0 and 0.0 <= gate["dsr"] <= 1.0

            causal = json.loads(
                (
                    await client.call_tool(
                        "causal_check",
                        {
                            "episode_id": reg["episode_id"],
                            "spec": {
                                "episode_id": reg["episode_id"],
                                "targets": ["TOYA", "TOYB"],
                                "conditioning": ["mkt"],
                                "method": "granger",
                                "lags": {"min_lag": 1, "max_lag": 5},
                            },
                        },
                    )
                ).content[0].text
            )
            assert causal["simulated"] is True

            risk = json.loads(
                (
                    await client.call_tool(
                        "risk_decompose",
                        {
                            "episode_id": reg["episode_id"],
                            "spec": {"book_id": "book1", "factor_model": "ff5+momentum"},
                        },
                    )
                ).content[0].text
            )
            assert risk["simulated"] is True

            report = json.loads(
                (
                    await client.call_tool(
                        "report_write", {"episode_id": reg["episode_id"]}
                    )
                ).content[0].text
            )
            assert "spend ledger:" in report["markdown"]  # §14 cost section

    asyncio.run(run())


def test_phase3_tombstone_block_and_acknowledge_over_mcp(tmp_path):
    # Seed a tombstone whose embedding is IDENTICAL to the pre-check query text
    # (FakeEmbedder: same text -> same vector -> cosine 1.0 >= block threshold).
    query_text = " ".join(
        [
            PROPOSE_ARGS["mechanism"],
            PROPOSE_ARGS["scope"],
            "artifacts://signals/mom.json",
            json.dumps({"type": "momentum", "lookback": 20}, sort_keys=True),
        ]
    )

    from spark_mcp.memory import add_tombstone
    from spark_mcp.storage import open_db

    db = open_db(f"{tmp_path}/spark.db")
    tomb_id = add_tombstone(
        db,
        kind="failed_hypothesis",
        summary=query_text,
        match_keys={"keywords": []},
        evidence_episode_ids=[],
        embedder=FakeEmbedder(),
    )["tombstone_id"]
    db.close()

    async def run():
        async with Client(_server(tmp_path)) as client:
            prop = json.loads(
                (await client.call_tool("hypothesis_propose", PROPOSE_ARGS)).content[0].text
            )
            reg = json.loads(
                (
                    await client.call_tool(
                        "register_hypothesis",
                        {
                            "hypothesis_id": prop["hypothesis_id"],
                            "spec": {"signal": "mom"},
                            "seed": 11,
                        },
                    )
                ).content[0].text
            )
            # the same idea is blocked before dispatch, with the matched id
            with pytest.raises(ToolError, match="tombstone pre-check blocked") as exc_info:
                await client.call_tool(
                    "backtest_run",
                    {"episode_id": reg["episode_id"], "spec": _spec(prop["hypothesis_id"], 11)},
                )
            assert tomb_id in str(exc_info.value)
            # ...unless the caller acknowledges the tombstone (logged on episode)
            bt = json.loads(
                (
                    await client.call_tool(
                        "backtest_run",
                        {
                            "episode_id": reg["episode_id"],
                            "spec": _spec(prop["hypothesis_id"], 11),
                            "acknowledge_tombstone": {
                                "tombstone_id": tomb_id,
                                "rationale": "new cost model justifies the retry",
                            },
                        },
                    )
                ).content[0].text
            )
            assert bt["status"] == "testing"
            assert bt["tombstone_check"]["status"] == "ack"
            assert bt["tombstone_check"]["rationale"] == "new cost model justifies the retry"

    asyncio.run(run())


def test_phase3_alpha_engine_provider_resolution(tmp_path, monkeypatch):
    """peers.alpha_engine.enabled + command -> AlphaEngineProvider resolution
    (constructed, not connected; no live peer needed for wiring)."""
    from spark_mcp.providers.alpha_engine import AlphaEngineProvider

    cfg = Config(
        storage={"db_path": f"{tmp_path}/spark.db", "artifacts_dir": f"{tmp_path}/art"},
        peers={"alpha_engine": {"enabled": True, "command": "/bin/false"}},
    )
    server = create_server(cfg)

    async def tool_names():
        return {tool.name for tool in await server.list_tools()}

    names = asyncio.run(tool_names())
    assert {
        # core + phase1/2 tools still registered alongside phase 3
        "spark_health",
        "spark_doctor",
        "hypothesis_propose",
        "register_hypothesis",
        "log_outcome",
        "research_history",
        "memory_recall",
        "check_novelty",
        "explore_methods",
        "memory_promote",
        "report_write",
        "backtest_run",
        "significance_gate",
        "causal_check",
        "risk_decompose",
    } <= names
    provider = AlphaEngineProvider(cfg.peers.alpha_engine)
    assert provider.name == "alpha_engine"
