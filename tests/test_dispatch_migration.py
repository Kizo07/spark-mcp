"""Dispatch migration: every MCP tool routes through its declared profile.

Proves: TOOL_PROFILES covers all 33 registered tools with profiles that
exist in dispatch.PROFILES; every tool def in server.py carries the
routing decorator (structural AST check); unregistered tools are refused
at registration; representative tools from each profile genuinely
traverse Pipeline.run; routed errors are identical to service errors.
"""

from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError as ClientToolError

import spark_mcp.server as server_module
from spark_mcp import dispatch as d
from spark_mcp.config import Config
from spark_mcp.dispatch import Pipeline
from spark_mcp.errors import ToolError
from spark_mcp.evolve import run as evolution_service
from spark_mcp.llm import FakeEmbedder
from spark_mcp.server import TOOL_PROFILES, create_server

EXPECTED_COUNTS = {"tool-gated": 4, "tool-evolution": 8, "tool-standard": 21}


def _server(tmp_path):
    cfg = Config(
        storage={"db_path": f"{tmp_path}/spark.db", "artifacts_dir": f"{tmp_path}/art"}
    )
    return create_server(cfg, embedder=FakeEmbedder())


def _registered_names(server) -> set[str]:
    async def _list():
        return {t.name for t in await server.list_tools()}

    return asyncio.run(_list())


def _spy_pipeline_runs(monkeypatch) -> list:
    """Record every Pipeline.run's step names, then run for real."""
    runs: list[list[str]] = []
    real_run = Pipeline.run

    def spy_run(self, ctx):
        runs.append(list(self.names))
        return real_run(self, ctx)

    monkeypatch.setattr(Pipeline, "run", spy_run)
    return runs


# ---------------------------------------------------------------------------
# registry coverage + profile validity


def test_registry_covers_all_registered_tools(tmp_path):
    names = _registered_names(_server(tmp_path))
    assert names == set(TOOL_PROFILES), (
        f"registry mismatch: unregistered={sorted(names - set(TOOL_PROFILES))} "
        f"stale={sorted(set(TOOL_PROFILES) - names)}"
    )
    for tool, profile in TOOL_PROFILES.items():
        assert profile in d.PROFILES, f"{tool}: unknown profile {profile!r}"
    counts: dict[str, int] = {}
    for profile in TOOL_PROFILES.values():
        counts[profile] = counts.get(profile, 0) + 1
    assert counts == EXPECTED_COUNTS


def test_tool_profiles_are_single_step_canonical():
    for profile in EXPECTED_COUNTS:
        assert d.PROFILES[profile] == (d.TOOL_CALL,)
        ordered = d.PROFILES[profile]
        assert tuple(sorted(ordered, key=d.EXTENDED_STEP_ORDER.index)) == ordered
        assert d.build_pipeline(profile, lambda n: lambda c: None).names == [d.TOOL_CALL]
    assert d.STEP_ORDER == (
        d.TOMBSTONE_PRECHECK,
        d.BUDGET_CHECK,
        d.TRIAL_INSERT,
        d.PROVIDER_DISPATCH,
        d.VERDICT_RECORD,
    )


def test_unregistered_tool_cannot_register(tmp_path):
    services = server_module.Services(cfg=Config(
        storage={"db_path": f"{tmp_path}/s.db", "artifacts_dir": f"{tmp_path}/a"}
    ), db=None)

    def bogus_tool() -> dict:
        return {}

    with pytest.raises(KeyError, match="no dispatch profile"):
        server_module._route_tool_handler(services, bogus_tool)
    with pytest.raises(KeyError, match="no dispatch profile"):
        server_module._dispatch_tool_call(services, "bogus_tool", lambda: {})


def test_every_tool_handler_is_routed():
    """Structural: each @mcp.tool def inside create_server has @_routed."""
    tree = ast.parse(Path(server_module.__file__).read_text())
    (create_fn,) = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "create_server"
    ]
    tool_defs = {}
    for node in create_fn.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        deco_kinds = set()
        for deco in node.decorator_list:
            if isinstance(deco, ast.Call) and getattr(deco.func, "attr", "") == "tool":
                deco_kinds.add("tool")
            if isinstance(deco, ast.Name) and deco.id == "_routed":
                deco_kinds.add("routed")
        if "tool" in deco_kinds:
            tool_defs[node.name] = deco_kinds
    assert set(tool_defs) == set(TOOL_PROFILES)
    unrouted = [n for n, kinds in tool_defs.items() if "routed" not in kinds]
    assert not unrouted, f"tools missing @_routed: {unrouted}"


# ---------------------------------------------------------------------------
# real traversal per profile (spy on Pipeline.run)


def test_standard_tool_traverses_pipeline(tmp_path, monkeypatch):
    runs = _spy_pipeline_runs(monkeypatch)

    async def _call():
        async with Client(_server(tmp_path)) as client:
            return await client.call_tool("spark_health", {})

    payload = json.loads(asyncio.run(_call()).content[0].text)
    assert payload["service"] == "spark" and payload["db_exists"] is True
    assert runs == [[d.TOOL_CALL]]


def test_evolution_tool_traverses_pipeline_and_matches_service_error(
    tmp_path, monkeypatch, db
):
    runs = _spy_pipeline_runs(monkeypatch)

    async def _call():
        async with Client(_server(tmp_path)) as client:
            return await client.call_tool("evolution_status", {"run_id": "run-nope"})

    with pytest.raises(ClientToolError) as exc_info:
        asyncio.run(_call())
    assert runs == [[d.TOOL_CALL]]
    with pytest.raises(ToolError) as direct:
        evolution_service.evolution_status(db, "run-nope")
    assert str(exc_info.value) == str(direct.value)


def test_gated_tool_traverses_wrapper_and_service_pipelines(tmp_path, monkeypatch):
    runs = _spy_pipeline_runs(monkeypatch)

    async def _flow():
        async with Client(_server(tmp_path)) as client:
            prop = json.loads(
                (await client.call_tool("hypothesis_propose", {
                    "idea": "toy momentum", "family": "momentum", "universe": "toy",
                    "horizon": "21d", "scope": "cross_section",
                    "mechanism": "12-1 momentum in toy universe",
                    "expected_direction": "positive",
                    "falsifier": "IC <= 0 net of costs",
                    "predicted_ic": 0.03, "predicted_sigma": 0.01,
                    "hypothesis_type": "predictive",
                })).content[0].text
            )
            reg = json.loads(
                (await client.call_tool("register_hypothesis", {
                    "hypothesis_id": prop["hypothesis_id"],
                    "spec": {"signal": "mom"}, "seed": 11,
                })).content[0].text
            )
            runs.clear()
            bt = json.loads(
                (await client.call_tool("backtest_run", {
                    "episode_id": reg["episode_id"],
                    "spec": {
                        "hypothesis_id": prop["hypothesis_id"],
                        "signal_ref": "artifacts://signals/mom.json",
                        "params": {"type": "momentum", "lookback": 20},
                        "universe_pit": "TOYA,TOYB,TOYC,TOYD,TOYE,TOYF,TOYG,TOYH@2019-01-01",
                        "dates": {
                            "is_start": "2019-01-01", "is_end": "2019-12-31",
                            "oos_start": "2020-01-01", "oos_end": "2020-06-30",
                        },
                        "costs": {"bps": 5, "slippage": 2},
                        "splits": {"purge": 5, "embargo": 5},
                        "seed": 11,
                    },
                })).content[0].text
            )
            return bt

    bt = asyncio.run(_flow())
    assert bt["status"] == "testing"
    assert [d.TOOL_CALL] in runs  # server wrapper layer
    assert list(d.STEP_ORDER) in runs  # gates-service gated pipeline
