"""P1-5 trace tools + P2-3 debate tool over the real MCP protocol.

trace_hypothesis/sota_status are pure ledger reads; debate_evidence runs
exactly two bounded LLM passes and persists the block as a trajectory
event (kind='debate', run_id=hypothesis_id). The independence test proves
the HARD invariant: significance_gate and evolve.promote.decide verdicts
are identical with and without a debate block.
"""

import asyncio
import json

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError as ClientToolError

from spark_mcp.config import Config
from spark_mcp.evolve import promote
from spark_mcp.llm import FakeEmbedder, FakeLLM
from spark_mcp.server import create_server
from spark_mcp.storage import open_db

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


@pytest.fixture(autouse=True)
def _no_ambient_llm_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def _cfg(tmp_path, tag="spark.db"):
    return Config(
        storage={"db_path": f"{tmp_path}/{tag}", "artifacts_dir": f"{tmp_path}/art"}
    )


def _replies():
    return {
        "PROSECUTION PASS": "prosecution: the record is thin, distrust it",
        "DEFENCE PASS": "defence: the record holds, trust it",
    }


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


async def _call(client, tool, args=None):
    return json.loads((await client.call_tool(tool, args or {})).content[0].text)


def test_trace_hypothesis_tool_returns_root_and_experiments(tmp_path):
    async def run():
        async with Client(
            create_server(_cfg(tmp_path), embedder=FakeEmbedder(), seeds=False)
        ) as client:
            prop = await _call(client, "hypothesis_propose", PROPOSE_ARGS)
            hid = prop["hypothesis_id"]
            trace = await _call(client, "trace_hypothesis", {"hypothesis_id": hid})
            assert trace["hypothesis_id"] == hid
            assert [e["kind"] for e in trace["entries"]] == ["hypothesis"]
            assert trace["entries"][0]["hypothesis_id"] == hid
            reg = await _call(
                client,
                "register_hypothesis",
                {"hypothesis_id": hid, "spec": {"signal": "mom"}, "seed": 11},
            )
            trace2 = await _call(client, "trace_hypothesis", {"hypothesis_id": hid})
            assert [e["kind"] for e in trace2["entries"]] == ["hypothesis", "experiment"]
            assert trace2["entries"][1]["episode_id"] == reg["episode_id"]

    asyncio.run(run())


def test_trace_hypothesis_unknown_refused(tmp_path):
    async def run():
        async with Client(
            create_server(_cfg(tmp_path), embedder=FakeEmbedder(), seeds=False)
        ) as client:
            with pytest.raises(ClientToolError, match="unknown hypothesis_id"):
                await client.call_tool("trace_hypothesis", {"hypothesis_id": "hx_nope"})

    asyncio.run(run())


def test_sota_status_empty_then_pointing(tmp_path):
    cfg = _cfg(tmp_path)

    async def run():
        async with Client(
            create_server(cfg, embedder=FakeEmbedder(), seeds=False)
        ) as client:
            assert await _call(client, "sota_status") == {"sota": None}

    asyncio.run(run())
    # A stored passing gate verdict becomes the SOTA pointer (direct rows,
    # same shape significance_gate writes to corrected_metrics).
    db = open_db(cfg.db_path())
    try:
        db.execute(
            "INSERT INTO hypotheses (hypothesis_id, parent_hypothesis_ids,"
            " research_family, hypothesis_type, mechanism, expected_direction,"
            " horizon, scope, falsifier, predicted_ic, predicted_sigma,"
            " universe, as_of, data_through, status, created_at)"
            " VALUES ('hx_sota', '[]', 'toy', 'predictive', 'm', 'positive',"
            " '21d', 'x', 'f', 0.03, 0.01, 'toy',"
            " '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z', 'proposed',"
            " '2026-01-01T00:00:00Z')"
        )
        db.execute(
            "INSERT INTO episodes (episode_id, hypothesis_id, run_kind,"
            " research_family, corrected_metrics, status, as_of, data_through,"
            " created_at) VALUES ('ep_sota', 'hx_sota', 'primary', 'toy',"
            " '{\"pass\": true, \"dsr\": 0.97, \"sr\": 1.5, \"p_value\": 0.03}',"
            " 'validated', '2026-01-02T00:00:00Z', '2026-01-02T00:00:00Z',"
            " '2026-01-02T00:00:00Z')"
        )
    finally:
        db.close()

    async def run2():
        async with Client(
            create_server(cfg, embedder=FakeEmbedder(), seeds=False)
        ) as client:
            got = await _call(client, "sota_status")
            assert got["sota"]["hypothesis_id"] == "hx_sota"
            assert got["sota"]["episode_id"] == "ep_sota"
            assert got["sota"]["dsr_prob"] == 0.97

    asyncio.run(run2())


def test_debate_evidence_degraded_mode_refuses_and_writes_nothing(tmp_path):
    cfg = _cfg(tmp_path)  # no llm arg + no API key -> services.llm is None

    async def run():
        async with Client(
            create_server(cfg, embedder=FakeEmbedder(), seeds=False)
        ) as client:
            prop = await _call(client, "hypothesis_propose", PROPOSE_ARGS)
            with pytest.raises(ClientToolError, match="degraded mode"):
                await client.call_tool(
                    "debate_evidence", {"hypothesis_id": prop["hypothesis_id"]}
                )
            return prop["hypothesis_id"]

    hid = asyncio.run(run())
    db = open_db(cfg.db_path())
    try:
        n = db.conn.execute(
            "SELECT COUNT(*) FROM trajectory_events WHERE run_id = ? AND kind = 'debate'",
            (hid,),
        ).fetchone()[0]
    finally:
        db.close()
    assert n == 0


def test_debate_evidence_two_passes_persisted_and_returned(tmp_path):
    cfg = _cfg(tmp_path)
    llm = FakeLLM(_replies())

    async def run():
        async with Client(
            create_server(cfg, embedder=FakeEmbedder(), llm=llm, seeds=False)
        ) as client:
            prop = await _call(client, "hypothesis_propose", PROPOSE_ARGS)
            block = await _call(
                client,
                "debate_evidence",
                {"hypothesis_id": prop["hypothesis_id"], "client_id": "deb-1"},
            )
            assert block["hypothesis_id"] == prop["hypothesis_id"]
            assert block["prosecution"] == _replies()["PROSECUTION PASS"]
            assert block["defence"] == _replies()["DEFENCE PASS"]
            assert block["model_id"] == llm.model_id()
            assert block["record_chars"] <= 2000
            return prop["hypothesis_id"]

    hid = asyncio.run(run())
    assert len(llm.calls) == 2  # exactly two bounded passes
    db = open_db(cfg.db_path())
    try:
        row = db.conn.execute(
            "SELECT kind, payload FROM trajectory_events WHERE run_id = ?"
            " AND kind = 'debate'",
            (hid,),
        ).fetchone()
    finally:
        db.close()
    assert row is not None
    payload = json.loads(row["payload"])
    assert payload["tool"] == "debate_evidence"
    assert payload["prosecution"] == _replies()["PROSECUTION PASS"]
    assert payload["defence"] == _replies()["DEFENCE PASS"]
    assert payload["client_id"] == "deb-1"


def test_debate_evidence_unknown_hypothesis_refused(tmp_path):
    llm = FakeLLM(_replies())

    async def run():
        async with Client(
            create_server(_cfg(tmp_path), embedder=FakeEmbedder(), llm=llm, seeds=False)
        ) as client:
            with pytest.raises(ClientToolError, match="unknown hypothesis_id"):
                await client.call_tool("debate_evidence", {"hypothesis_id": "hx_nope"})

    asyncio.run(run())
    assert llm.calls == []  # refused before any LLM pass


def _flow_gate_and_decide(tmp_path, tag, with_debate):
    """Identical propose/register/backtest/gate flow on an isolated db, with
    or without a debate block persisted before the gate. Returns the gate
    verdict (volatile ids stripped) plus the promote.decide outcome."""
    cfg = _cfg(tmp_path, tag)
    llm = FakeLLM(_replies())

    async def run():
        async with Client(
            create_server(cfg, embedder=FakeEmbedder(), llm=llm, seeds=False)
        ) as client:
            prop = await _call(client, "hypothesis_propose", PROPOSE_ARGS)
            reg = await _call(
                client,
                "register_hypothesis",
                {
                    "hypothesis_id": prop["hypothesis_id"],
                    "spec": {"signal": "mom"},
                    "seed": 11,
                },
            )
            bt = await _call(
                client,
                "backtest_run",
                {
                    "episode_id": reg["episode_id"],
                    "spec": _spec(prop["hypothesis_id"], 11),
                },
            )
            assert bt["provider"] == "simulated"
            if with_debate:
                await _call(
                    client,
                    "debate_evidence",
                    {"hypothesis_id": prop["hypothesis_id"]},
                )
            gate = await _call(
                client, "significance_gate", {"episode_id": reg["episode_id"]}
            )
            return reg["episode_id"], gate

    episode_id, gate = asyncio.run(run())
    db = open_db(cfg.db_path())
    try:
        decision = promote.decide(db, cfg, episode_id, None)
    finally:
        db.close()
    gate.pop("episode_id", None)
    return gate, (decision.promote, decision.reason)


def test_gate_and_decide_verdicts_identical_with_and_without_debate(tmp_path):
    plain_gate, plain_decide = _flow_gate_and_decide(tmp_path, "plain.db", False)
    debated_gate, debated_decide = _flow_gate_and_decide(tmp_path, "debated.db", True)
    assert debated_gate == plain_gate
    assert debated_decide == plain_decide
