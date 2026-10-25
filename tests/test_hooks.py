"""Phase 4: the QuantFlow Tier-2 run-summary hook adapter (plan §15.1 C9/C10).

Covers strict validation (incl. the forward-compat rule), the record/enrich
split, the retrospective difficulty heuristics (v1 §6), the unknown-episode
decision, the end-to-end desk close-out, and both entry points (MCP tool,
`spark closeout` CLI) over one shared service function (plan §13, no fork).
"""

import asyncio
import json

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from spark_mcp import hooks, ledger
from spark_mcp.config import Config
from spark_mcp.hooks import HookSchemaError, ingest_run_summary, validate_run_summary
from spark_mcp.llm import FakeEmbedder
from spark_mcp.server import create_server
from spark_mcp.storage import open_db


def _payload(**overrides) -> dict:
    payload = {
        "run_id": "qf-run-1",
        "tool_call_count": 12,
        "failure_sketches": [
            {"stage": "backtest", "error_class": "DataGap", "resolution": "switched provider"}
        ],
        "skills_activated": {"quant-desk": "1.2.0"},
        "tokens": 5000,
        "wall_sec": 42.5,
        "schema_version": 1,
    }
    payload.update(overrides)
    return payload


def _desk_episode(db, cfg, embedder=None) -> str:
    """propose -> register, the Spark-side half of a desk run."""
    prop = ledger.propose_hypothesis(
        db,
        cfg,
        research_family="momentum",
        hypothesis_type="predictive",
        mechanism="12-1 momentum in toy universe",
        expected_direction="positive",
        horizon="21d",
        scope="cross_section",
        falsifier="IC <= 0 net of costs",
        predicted_ic=0.03,
        predicted_sigma=0.01,
        universe="toy",
        embedder=embedder,
    )
    reg = ledger.register_hypothesis(db, cfg, prop["hypothesis_id"], {"signal": "mom"}, 7)
    return reg["episode_id"]


def _tier2_events(db, run_id: str) -> list[dict]:
    rows = db.conn.execute(
        "SELECT actor, kind, payload FROM trajectory_events WHERE run_id = ? ORDER BY seq",
        (run_id,),
    ).fetchall()
    return [
        {"actor": r["actor"], "kind": r["kind"], "payload": json.loads(r["payload"])}
        for r in rows
        if r["kind"] == "hook_summary"
    ]


# -- validation ---------------------------------------------------------------


def test_validate_good_payload_passes():
    payload = _payload(
        episode_id="ep_x",
        task_family="momentum",
        outcome_summary="validated after cost fix",
        fingerprint="asm:v1:abc",
        fingerprint_spec_version="fingerprint-spec/2",
    )
    assert validate_run_summary(payload) is payload


def test_validate_minimal_payload_passes():
    minimal = _payload(failure_sketches=[], skills_activated={})
    assert validate_run_summary(minimal)["run_id"] == "qf-run-1"


@pytest.mark.parametrize(
    "key",
    [
        "run_id",
        "tool_call_count",
        "failure_sketches",
        "skills_activated",
        "tokens",
        "wall_sec",
        "schema_version",
    ],
)
def test_validate_rejects_missing_required_field(key):
    payload = _payload()
    del payload[key]
    with pytest.raises(HookSchemaError, match=key):
        validate_run_summary(payload)


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"schema_version": "1"}, "'schema_version' must be an integer"),
        ({"run_id": ""}, "'run_id' must be a non-empty string"),
        ({"run_id": 42}, "'run_id' must be a non-empty string"),
        ({"episode_id": 123}, "'episode_id' must be a string"),
        ({"fingerprint": ["a"]}, "'fingerprint' must be a string"),
        ({"tool_call_count": -1}, "'tool_call_count' must be >= 0"),
        ({"tool_call_count": 1.5}, "'tool_call_count' must be an integer"),
        ({"tool_call_count": True}, "'tool_call_count' must be an integer"),
        ({"tokens": -5}, "'tokens' must be >= 0"),
        ({"wall_sec": -0.5}, "'wall_sec' must be >= 0"),
        ({"wall_sec": "fast"}, "'wall_sec' must be a number"),
        ({"failure_sketches": {}}, "'failure_sketches' must be an array"),
        ({"failure_sketches": ["nope"]}, "'failure_sketches[0]' must be an object"),
        ({"failure_sketches": [{"stage": "s", "error_class": "e"}]}, "resolution"),
        (
            {"failure_sketches": [{"stage": "s", "error_class": "e", "resolution": None}]},
            "resolution",
        ),
        ({"skills_activated": []}, "'skills_activated' must be an object"),
        ({"skills_activated": {"a": 1}}, "version strings"),
    ],
)
def test_validate_rejects_violations(patch, message):
    with pytest.raises(HookSchemaError, match=message.replace("[", r"\[").replace("]", r"\]")):
        validate_run_summary(_payload(**patch))


def test_validate_rejects_non_object_payload():
    with pytest.raises(HookSchemaError, match="JSON object"):
        validate_run_summary([1, 2, 3])  # type: ignore[arg-type]


def test_validate_rejects_newer_schema_version_naming_the_received_one():
    with pytest.raises(HookSchemaError) as exc_info:
        validate_run_summary(_payload(schema_version=2))
    message = str(exc_info.value)
    assert "schema_version" in message and "2" in message


def test_validate_ignores_extra_keys():
    payload = _payload(peer_extra={"anything": True}, another_field=[1, 2])
    assert validate_run_summary(payload)["run_id"] == "qf-run-1"


# -- ingestion: record & enrich -------------------------------------------------


def test_ingest_records_tier2_event_without_episode(db, cfg):
    result = ingest_run_summary(db, cfg, _payload(outcome_summary="ran clean"))
    assert result["enriched"] is False
    assert "warning" not in result  # no episode_id: nothing to warn about
    assert result["episode_id"] is None
    events = _tier2_events(db, "qf-run-1")
    assert len(events) == 1
    assert events[0]["actor"] == "quantflow-hook"
    assert events[0]["payload"]["tier"] == 2
    assert events[0]["payload"]["run_id"] == "qf-run-1"
    assert events[0]["payload"]["outcome_summary"] == "ran clean"


def test_ingest_enriches_episode_with_fingerprint_and_skills(db, cfg):
    episode_id = _desk_episode(db, cfg)
    result = ingest_run_summary(
        db,
        cfg,
        _payload(
            episode_id=episode_id,
            fingerprint="asm:v1:abc123",
            fingerprint_spec_version="fingerprint-spec/2",
            skills_activated={"quant-desk": "1.2.0", "risk-check": "0.9.1"},
        ),
    )
    assert result["enriched"] is True
    ep = ledger.get_episode(db, episode_id)
    assert ep["agent_assembly_fingerprint"] == "asm:v1:abc123"
    assert ep["fingerprint_spec_version"] == "fingerprint-spec/2"
    assert ep["skill_versions"] == {"quant-desk": "1.2.0", "risk-check": "0.9.1"}
    # Tier-2 event joins the episode's trajectory (keyed by episode_id).
    assert len(_tier2_events(db, episode_id)) == 1


def test_ingest_fingerprint_without_version_defaults_unversioned(db, cfg):
    episode_id = _desk_episode(db, cfg)
    result = ingest_run_summary(
        db, cfg, _payload(episode_id=episode_id, fingerprint="asm:v1:abc123")
    )
    assert result["fingerprint_spec_version"] == "unversioned"  # §15.1 fallback row 2
    assert ledger.get_episode(db, episode_id)["fingerprint_spec_version"] == "unversioned"


def test_ingest_skills_default_to_empty_object(db, cfg):
    episode_id = _desk_episode(db, cfg)
    ingest_run_summary(db, cfg, _payload(episode_id=episode_id, skills_activated={}))
    ep = ledger.get_episode(db, episode_id)
    assert ep["skill_versions"] == {}  # §15.1 fallback row 3


def test_ingest_unknown_episode_records_event_but_does_not_enrich(db, cfg):
    """Documented decision: a valid payload is evidence even when its
    episode_id does not resolve — the Tier-2 event lands, the episode is not
    touched, and the caller gets enriched=False plus a warning instead of an
    exception (peer emission stays crash-free, §15.1 failure semantics)."""
    result = ingest_run_summary(db, cfg, _payload(episode_id="ep_missing"))
    assert result["enriched"] is False
    assert "ep_missing" in result["warning"]
    assert ledger.get_episode(db, "ep_missing") is None
    events = _tier2_events(db, "ep_missing")
    assert len(events) == 1 and events[0]["payload"]["tier"] == 2


def test_ingest_invalid_payload_writes_nothing(db, cfg):
    """Failure semantics: a bad payload never corrupts ledger state — the
    schema error raises before any write (no trajectory event either)."""
    episode_id = _desk_episode(db, cfg)
    before = db.conn.execute("SELECT COUNT(*) AS n FROM trajectory_events").fetchone()["n"]
    with pytest.raises(HookSchemaError):
        ingest_run_summary(db, cfg, _payload(episode_id=episode_id, tool_call_count=-1))
    after = db.conn.execute("SELECT COUNT(*) AS n FROM trajectory_events").fetchone()["n"]
    assert after == before
    ep = ledger.get_episode(db, episode_id)
    assert ep["agent_assembly_fingerprint"] is None
    assert ep["skill_versions"] == {}


# -- retrospective difficulty heuristics (v1 §6) -------------------------------


def test_difficulty_tool_calls_over_threshold_triggers(db, cfg):
    result = ingest_run_summary(
        db, cfg, _payload(tool_call_count=51, tokens=100, failure_sketches=[])
    )
    assert result["difficulty"]["retrospective_triggered"] is True


def test_difficulty_failures_overcome_triggers(db, cfg):
    result = ingest_run_summary(
        db,
        cfg,
        _payload(
            tool_call_count=10,
            tokens=100,
            failure_sketches=[{"stage": "s", "error_class": "e", "resolution": "fixed"}],
        ),
    )
    assert result["difficulty"]["failures_overcome"] == 1
    assert result["difficulty"]["retrospective_triggered"] is True


def test_difficulty_failures_without_resolution_do_not_count(db, cfg):
    result = ingest_run_summary(
        db,
        cfg,
        _payload(
            tool_call_count=10,
            tokens=100,
            failure_sketches=[
                {"stage": "s1", "error_class": "e", "resolution": ""},
                {"stage": "s2", "error_class": "e", "resolution": ""},
            ],
        ),
    )
    assert result["difficulty"]["failures_overcome"] == 0
    assert result["difficulty"]["retrospective_triggered"] is False


def test_difficulty_tokens_over_threshold_triggers(db, cfg):
    result = ingest_run_summary(
        db, cfg, _payload(tool_call_count=10, tokens=200_001, failure_sketches=[])
    )
    assert result["difficulty"]["retrospective_triggered"] is True


def test_difficulty_all_quiet_does_not_trigger(db, cfg):
    result = ingest_run_summary(
        db, cfg, _payload(tool_call_count=50, tokens=200_000, failure_sketches=[])
    )
    # boundary values are NOT over threshold (strict >)
    assert result["difficulty"]["retrospective_triggered"] is False


def test_difficulty_thresholds_are_config_named(tmp_path):
    cfg = Config(
        storage={"db_path": f"{tmp_path}/spark.db", "artifacts_dir": f"{tmp_path}/art"},
        research={"hook_tool_calls_threshold": 5, "hook_tokens_threshold": 900},
    )
    db = open_db(cfg.db_path())
    try:
        result = ingest_run_summary(
            db, cfg, _payload(tool_call_count=6, tokens=100, failure_sketches=[])
        )
        assert result["difficulty"]["retrospective_triggered"] is True
        result = ingest_run_summary(
            db, cfg, _payload(tool_call_count=4, tokens=901, failure_sketches=[])
        )
        assert result["difficulty"]["retrospective_triggered"] is True
    finally:
        db.close()


# -- end-to-end desk run --------------------------------------------------------


def test_desk_run_closes_with_fingerprint_and_skill_versions(db, cfg, embedder):
    """A real desk run per §15.1: Spark-side propose/register, QuantFlow-side
    Tier-2 summary, manual log_outcome close-out. The episode must carry the
    fingerprint/spec/skill versions and the Tier-2 event must be in the
    episode's trajectory."""
    episode_id = _desk_episode(db, cfg, embedder)
    result = ingest_run_summary(
        db,
        cfg,
        _payload(
            episode_id=episode_id,
            run_id="qf-desk-77",
            fingerprint="asm:v1:desk77",
            fingerprint_spec_version="fingerprint-spec/3",
            skills_activated={"quant-desk": "2.0.0"},
            tool_call_count=64,  # over threshold -> retrospective triggered
            tokens=9000,
        ),
    )
    assert result["enriched"] is True
    assert result["difficulty"]["retrospective_triggered"] is True

    out = ledger.log_outcome(
        db,
        cfg,
        episode_id,
        "success",
        metrics={"realized_ic": 0.021},
        costs={"usd": 0.01, "tokens": 120},
    )
    assert out["status"] == "validated"

    ep = ledger.get_episode(db, episode_id)
    assert ep["status"] == "validated"
    assert ep["agent_assembly_fingerprint"] == "asm:v1:desk77"
    assert ep["fingerprint_spec_version"] == "fingerprint-spec/3"
    assert ep["skill_versions"] == {"quant-desk": "2.0.0"}

    events = _tier2_events(db, episode_id)
    assert len(events) == 1
    assert events[0]["payload"]["run_id"] == "qf-desk-77"
    assert events[0]["payload"]["skills_activated"] == {"quant-desk": "2.0.0"}


# -- entry points: MCP tool and CLI (plan §13: same service function) -----------


def test_run_summary_ingest_over_mcp(tmp_path):
    cfg = Config(
        storage={"db_path": f"{tmp_path}/spark.db", "artifacts_dir": f"{tmp_path}/art"}
    )
    db = open_db(cfg.db_path())
    episode_id = _desk_episode(db, cfg)
    db.close()
    server = create_server(cfg, embedder=FakeEmbedder())

    async def run():
        async with Client(server) as client:
            ok = await client.call_tool(
                "run_summary_ingest",
                {
                    "payload": _payload(
                        episode_id=episode_id,
                        fingerprint="asm:v1:mcp",
                        skills_activated={"quant-desk": "1.2.0"},
                    )
                },
            )
            with pytest.raises(ToolError, match="schema_version"):
                await client.call_tool(
                    "run_summary_ingest", {"payload": _payload(schema_version=2)}
                )
            return ok

    result = json.loads(asyncio.run(run()).content[0].text)
    assert result["enriched"] is True
    assert result["fingerprint_spec_version"] == "unversioned"


def test_closeout_cli(tmp_path, cfg):
    """`spark closeout --payload <path.json>`: the cron/manual-operator path
    over the same hooks.ingest_run_summary (no fork, plan §13)."""
    from typer.testing import CliRunner

    from spark_mcp.cli import app

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"storage:\n  db_path: {tmp_path}/spark.db\n  artifacts_dir: {tmp_path}/art\n",
        encoding="utf-8",
    )
    db = open_db(tmp_path / "spark.db")
    episode_id = _desk_episode(db, cfg)
    db.close()
    payload_path = tmp_path / "summary.json"
    payload_path.write_text(
        json.dumps(
            _payload(
                episode_id=episode_id,
                fingerprint="asm:v1:cli",
                fingerprint_spec_version="fingerprint-spec/1",
            )
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    invoked = runner.invoke(
        app, ["closeout", "--payload", str(payload_path), "--config", str(cfg_path)]
    )
    assert invoked.exit_code == 0, invoked.output
    result = json.loads(invoked.output)
    assert result["enriched"] is True
    assert result["fingerprint"] == "asm:v1:cli"

    db = open_db(tmp_path / "spark.db")
    assert ledger.get_episode(db, episode_id)["agent_assembly_fingerprint"] == "asm:v1:cli"
    db.close()

    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps(_payload(schema_version=2)), encoding="utf-8")
    rejected = runner.invoke(
        app, ["closeout", "--payload", str(bad_path), "--config", str(cfg_path)]
    )
    assert rejected.exit_code != 0
    assert isinstance(rejected.exception, HookSchemaError)


# -- §15.2 telemetry outbox (durable local queue; no peer push) -------------------


def test_telemetry_push_queues_and_persists(db, cfg):
    payload = {
        "tool": "backtest_run",
        "episode_id": "ep_tel_1",
        "trial_id": "trial_tel_1",
        "verdict": "pass",
        "sharpe": 1.25,
    }
    result = hooks.telemetry_push(db, cfg, payload)
    assert result["queued"] is True
    assert result["effect_id"]
    row = db.conn.execute(
        "SELECT tool, subject_id, result FROM tool_effects WHERE client_id = ?",
        (result["effect_id"],),
    ).fetchone()
    assert row is not None
    assert row["tool"] == "telemetry_push"
    assert row["subject_id"] == "ep_tel_1"
    stored = json.loads(row["result"])
    assert stored["tool"] == "backtest_run"
    assert stored["episode_id"] == "ep_tel_1"
    assert stored["trial_id"] == "trial_tel_1"
    assert stored["verdict"] == "pass"
    assert stored["sharpe"] == 1.25


def test_telemetry_push_never_raises_for_dict_payloads(db, cfg):
    for payload in ({}, {"tool": "risk_decompose"}, {"episode_id": 123, "verdict": None}):
        result = hooks.telemetry_push(db, cfg, payload)
        assert result["queued"] is True
        assert result["effect_id"]


def test_telemetry_push_strips_secrets(db, cfg):
    payload = {
        "tool": "causal_test",
        "episode_id": "ep_tel_secret",
        "verdict": "fail",
        "api_key": "sk-live-123",
        "auth_token": "tok-abc",
        "client_secret": "shh",
        "db_password": "hunter2",
        "nested": {"access_token": "nested-tok", "note": "keep-me"},
        "items": [{"private_key": "pem-body", "label": "keep-too"}],
    }
    result = hooks.telemetry_push(db, cfg, payload)
    assert result["queued"] is True
    row = db.conn.execute(
        "SELECT result FROM tool_effects WHERE client_id = ?", (result["effect_id"],)
    ).fetchone()
    raw = row["result"]
    for leaked in ("sk-live-123", "tok-abc", "shh", "hunter2", "nested-tok", "pem-body"):
        assert leaked not in raw
    stored = json.loads(raw)
    assert "api_key" not in stored and "auth_token" not in stored
    assert "client_secret" not in stored and "db_password" not in stored
    assert stored["nested"] == {"note": "keep-me"}
    assert stored["items"] == [{"label": "keep-too"}]
    assert stored["tool"] == "causal_test"
    assert stored["verdict"] == "fail"


def test_telemetry_outbox_count(db, cfg):
    assert hooks.telemetry_outbox_count(db) == 0
    hooks.telemetry_push(db, cfg, {"tool": "backtest_run", "verdict": "pass"})
    assert hooks.telemetry_outbox_count(db) == 1
    hooks.telemetry_push(db, cfg, {"tool": "risk_decompose", "verdict": "fail"})
    assert hooks.telemetry_outbox_count(db) == 2


def test_empty_skills_preserve_prior_enrichment(db, cfg):
    """{} means 'unknown' (row-3 fallback), never 'no skills': a later
    summary carrying less must not wipe earlier evidence (review N1)."""
    import json as _json

    ep = _desk_episode(db, cfg)
    base = _payload(episode_id=ep, skills_activated={"quant-desk": "2.0.0"})
    hooks.ingest_run_summary(db, cfg, base)
    hooks.ingest_run_summary(db, cfg, _payload(episode_id=ep, skills_activated={}))
    row = db.conn.execute(
        "SELECT skill_versions FROM episodes WHERE episode_id = ?", (ep,)
    ).fetchone()
    assert _json.loads(row["skill_versions"]) == {"quant-desk": "2.0.0"}


def test_whitespace_only_strings_treated_as_absent(db, cfg):
    """run_id/fingerprint/resolution of '   ' are absent, not values (N5)."""
    ep = _desk_episode(db, cfg)
    out = hooks.ingest_run_summary(
        db,
        cfg,
        _payload(
            episode_id=ep,
            fingerprint="   ",
            fingerprint_spec_version="  ",
            failure_sketches=[{"stage": "s", "error_class": "e", "resolution": "  "}],
        ),
    )
    assert out["fingerprint"] is None
    assert out["fingerprint_spec_version"] is None
    assert out["difficulty"]["failures_overcome"] == 0
    row = db.conn.execute(
        "SELECT agent_assembly_fingerprint, fingerprint_spec_version FROM episodes"
        " WHERE episode_id = ?",
        (ep,),
    ).fetchone()
    assert row["agent_assembly_fingerprint"] is None
    assert row["fingerprint_spec_version"] is None


def test_orphan_spec_version_without_fingerprint_ignored(db, cfg):
    ep = _desk_episode(db, cfg)
    hooks.ingest_run_summary(
        db,
        cfg,
        _payload(episode_id=ep, fingerprint_spec_version="spec/9"),
    )
    row = db.conn.execute(
        "SELECT fingerprint_spec_version FROM episodes WHERE episode_id = ?", (ep,)
    ).fetchone()
    assert row["fingerprint_spec_version"] is None


def _stored_telemetry(db, effect_id) -> dict:
    row = db.conn.execute(
        "SELECT result FROM tool_effects WHERE client_id = ?", (effect_id,)
    ).fetchone()
    return json.loads(row["result"])


def test_telemetry_strips_authorization_and_credential_variants(db, cfg):
    payload = {
        "tool": "backtest_run",
        "episode_id": "ep_auth",
        "authorization": "auth-v1",
        "Authorization": "auth-v2",
        "credentials": {"user": "u"},
        "credential": "cred-v",
        "api-key": "k1",
        "apiKey": "k2",
        "authToken": "k3",
        "session": "sess-1",
        "passwd": "pw",
        "bearer": "b1",
    }
    result = hooks.telemetry_push(db, cfg, payload)
    assert result["queued"] is True
    assert _stored_telemetry(db, result["effect_id"]) == {
        "tool": "backtest_run",
        "episode_id": "ep_auth",
    }


def test_telemetry_keeps_monkey_and_keyboard(db, cfg):
    payload = {
        "tool": "causal_test",
        "monkey": "banana",
        "keyboard": "qwerty",
        "nested": {"monkey": "m", "keyboard": "k", "note": "keep"},
    }
    result = hooks.telemetry_push(db, cfg, payload)
    stored = _stored_telemetry(db, result["effect_id"])
    assert stored["monkey"] == "banana"
    assert stored["keyboard"] == "qwerty"
    assert stored["nested"] == {"monkey": "m", "keyboard": "k", "note": "keep"}


def test_telemetry_strips_secret_values_by_pattern(db, cfg):
    payload = {
        "tool": "backtest_run",
        "note": "Bearer abcdef123456",
        "other": "key is sk-abc123xyz7890 for access",
        "safe": "just a normal note",
        "items": ["Bearer zzzzzz1111", "keep-me"],
    }
    result = hooks.telemetry_push(db, cfg, payload)
    stored = _stored_telemetry(db, result["effect_id"])
    assert "note" not in stored
    assert "other" not in stored
    assert stored["safe"] == "just a normal note"
    assert stored["items"] == ["keep-me"]
