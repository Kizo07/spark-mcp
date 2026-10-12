"""Loop driver + MCP surface (evolution upgrade plan §2.E1/E4, §5).

Happy path on SimulatedProvider, stopping rules (max_rounds, consecutive
rejections, kill-rule, fail_closed budget), IS-only feedback, trajectory
events, closed loop (never calls approve), promotion head/rollback/gate
bookkeeping, idempotent client_id, status reads, and the MCP tools over the
real protocol. Rerun determinism excludes ULID/timestamp columns by design.
"""

import ast
import asyncio
import json
from pathlib import Path

import pytest
from fastmcp import Client

from spark_mcp import budgets, ledger, skills
from spark_mcp.config import Config
from spark_mcp.errors import ToolError
from spark_mcp.evolve import lineage as lineage_mod
from spark_mcp.evolve import run as run_mod
from spark_mcp.llm import FakeLLM
from spark_mcp.providers.recorded import RecordedProvider
from spark_mcp.providers.simulated import SimulatedProvider
from spark_mcp.server import create_server
from spark_mcp.storage import open_db

FAMILY = "toy_momentum"

PROPOSE = dict(
    research_family=FAMILY,
    hypothesis_type="predictive",
    mechanism="12-1 momentum in toy universe",
    expected_direction="positive",
    horizon="21d",
    scope="cross-section",
    falsifier="IC <= 0 net of costs",
    predicted_ic=0.03,
    predicted_sigma=0.01,
    universe="toy",
)


def _seed(db, cfg, **overrides):
    return ledger.propose_hypothesis(db, cfg, **{**PROPOSE, **overrides})["hypothesis_id"]


def _exploit(cfg):
    """Keep the driver exploiting the head (a lone early rejection would
    otherwise trip the default SOTA-jump rule: 0/1 < 0.2)."""
    cfg.research.evolution_sota_jump_threshold = 0.0
    return cfg


class _StubProvider:
    name = "stub"

    def __init__(self, metrics, fail_times=0):
        self._metrics = dict(metrics)
        self.fail_times = fail_times
        self.calls = 0

    def backtest(self, spec):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("flaky endpoint")
        return dict(self._metrics)


STRONG = {
    "sr": 2.0, "T": 500, "skew": 0.0, "kurt": 3.0, "gross_sr": 2.1,
    "cost_drag": 0.1, "ic_mean": 0.05, "simulated": True, "provider": "stub",
}
WEAK = {
    "sr": -1.5, "T": 500, "skew": 0.0, "kurt": 3.0, "gross_sr": -1.4,
    "cost_drag": 0.1, "ic_mean": -0.02, "simulated": True, "provider": "stub",
}


def _root_llm(**extra_calls):
    root = {
        "mechanism": "m", "expected_direction": "positive", "horizon": "5d",
        "scope": "s", "falsifier": "f", "predicted_ic": 0.01,
        "predicted_sigma": 0.02, "research_family": FAMILY,
        "hypothesis_type": "predictive", "universe": "toy",
    }
    replies = {
        "successor": json.dumps(root),
        "narrating": json.dumps(
            {"observations": "o", "hypothesis_evaluation": "e", "new_hypothesis": "n"}
        ),
    }
    replies.update(extra_calls)
    return FakeLLM(replies)


# -- happy path + determinism -----------------------------------------------


def test_happy_path_simulated_provider(db, cfg):
    _exploit(cfg)
    seed = _seed(db, cfg)
    out = run_mod.run_evolution(
        db, cfg, SimulatedProvider(cfg.artifacts_dir()),
        family=FAMILY, seed_hypothesis_id=seed, max_rounds=2, seed=7,
    )
    assert out["family"] == FAMILY
    assert out["run_id"].startswith("erun_")
    assert out["line_id"].startswith("elif_")
    assert out["rounds"] == 2
    assert all(d["outcome"] == "decided" for d in out["decisions"])
    rows = db.conn.execute(
        "SELECT * FROM evolution_feedback WHERE run_id = ? ORDER BY rowid",
        (out["run_id"],),
    ).fetchall()
    assert len(rows) == 2
    for row in rows:
        assert row["prompt_sha"]  # ancestors render sha recorded
        assert row["model_id"] is None  # llm=None: deterministic narration
        assert row["decision"].startswith(("promoted", "rejected"))


def _run_signature(out):
    """Decisions minus ULID/timestamp identity (the constraint-4 scope)."""
    sig = []
    for d in out["decisions"]:
        sig.append(
            (d.get("round"), d.get("outcome"), d.get("promote"), d.get("stage"),
             d.get("decision"), d.get("dsr_prob"), d.get("sr"), d.get("policy"))
        )
    return (out["status"], out["rounds"], out["consecutive_rejections"], sig)


def _head_content(db, head):
    if head["hypothesis_id"] is None:
        return None
    hyp = ledger.get_hypothesis(db, head["hypothesis_id"])
    ep = ledger.get_episode(db, head["episode_id"]) if head["episode_id"] else None
    return {
        "mechanism": hyp["mechanism"],
        "horizon": hyp["horizon"],
        "metrics": (ep.get("metrics") or {}) if ep else None,
        "corrected": (ep.get("corrected_metrics") or {}) if ep else None,
    }


def _fresh_run(tmp_path, n, seed, **kwargs):
    d = tmp_path / f"db{n}"
    d.mkdir()
    cfg = Config(storage={"db_path": str(d / "spark.db"), "artifacts_dir": str(d / "art")})
    _exploit(cfg)
    db = open_db(cfg.db_path())
    hyp = _seed(db, cfg)
    out = run_mod.run_evolution(
        db, cfg, SimulatedProvider(cfg.artifacts_dir()),
        family=FAMILY, seed_hypothesis_id=hyp, seed=seed, **kwargs,
    )
    sig = (_run_signature(out), _head_content(db, out["head"]))
    db.close()
    return sig


def test_rerun_determinism_same_seed_identical(tmp_path):
    first = _fresh_run(tmp_path, 1, seed=7, max_rounds=3)
    second = _fresh_run(tmp_path, 2, seed=7, max_rounds=3)
    assert first == second


def test_different_seeds_may_diverge_but_remain_valid(tmp_path):
    # Seeds choose variation paths; both runs must complete cleanly.
    for n, seed in ((3, 7), (4, 8)):
        sig = _fresh_run(tmp_path, n, seed=seed, max_rounds=2)
        assert sig[0][0] in ("finished", "stopped_consecutive_rejections")


# -- stopping rules ----------------------------------------------------------


def test_stops_on_consecutive_rejections(db, cfg):
    _exploit(cfg)
    cfg.research.evolution_max_consecutive_rejections = 2
    seed = _seed(db, cfg)
    out = run_mod.run_evolution(
        db, cfg, _StubProvider(WEAK), family=FAMILY,
        seed_hypothesis_id=seed, max_rounds=8, seed=1,
    )
    assert out["status"] == "stopped_consecutive_rejections"
    assert out["rounds"] == 2
    assert out["consecutive_rejections"] == 2


def test_stops_on_max_rounds(db, cfg):
    _exploit(cfg)
    seed = _seed(db, cfg)
    out = run_mod.run_evolution(
        db, cfg, _StubProvider(STRONG), family=FAMILY,
        seed_hypothesis_id=seed, max_rounds=1, seed=1,
    )
    assert out["status"] == "finished"
    assert out["rounds"] == 1


def test_stops_on_kill_rule(db, cfg):
    _exploit(cfg)
    db.execute(
        "INSERT INTO exploration_state (family, killed, posterior, trials, successes,"
        " updated_at) VALUES (?, 1, '{\"alpha\": 1.0, \"beta\": 9.0}', 10, 1, ?)",
        (FAMILY, "2026-01-01T00:00:00Z"),
    )
    seed = _seed(db, cfg)
    out = run_mod.run_evolution(
        db, cfg, _StubProvider(STRONG), family=FAMILY,
        seed_hypothesis_id=seed, max_rounds=8, seed=1,
    )
    assert out["status"] == "stopped_kill_rule"
    assert out["rounds"] == 0


def test_budget_fail_closed_closes_gracefully(db, cfg):
    _exploit(cfg)
    cfg.budgets.policy = "fail_closed"
    budgets.record_spend(db, cfg, family=FAMILY, source="setup", usd=100.0,
                         run_id="setup")
    seed = _seed(db, cfg)
    out = run_mod.run_evolution(
        db, cfg, _StubProvider(STRONG), family=FAMILY,
        seed_hypothesis_id=seed, max_rounds=8, seed=1,
    )
    assert out["status"] == "stopped_budget"
    assert out["rounds"] == 1  # the blocked round is recorded, not raised
    assert out["decisions"][0]["stage"] == "budget"


# -- promotion bookkeeping + veto --------------------------------------------


def test_promotion_advances_head_and_writes_gate_notify(db, cfg):
    _exploit(cfg)
    seed = _seed(db, cfg)
    out = run_mod.run_evolution(
        db, cfg, _StubProvider(STRONG), family=FAMILY,
        seed_hypothesis_id=seed, max_rounds=1, seed=3,
    )
    assert out["decisions"][0]["promote"] is True
    line = lineage_mod.get_line(db, out["line_id"])
    assert line["head_hypothesis_id"] == out["head"]["hypothesis_id"]
    assert line["head_episode_id"] == out["head"]["episode_id"]
    # Prior head (the seed) moved to rollback before the new head was set.
    assert line["rollback_hypothesis_id"] == seed
    notify = db.conn.execute(
        "SELECT * FROM approvals WHERE target_type = 'evolution'"
        " AND target_id = ? AND decided_by = 'gate'",
        (out["line_id"],),
    ).fetchone()
    assert notify is not None and notify["decision"] == "approved"
    # Human veto post-run restores the rollback head (E3 path).
    veto = skills.approve(db, cfg, "evolution", out["line_id"], "rejected", "veto test")
    assert veto["applied"]["restored_head_hypothesis_id"] == seed


def test_second_promotion_needs_strict_margin(db, cfg):
    """Identical strong metrics twice: the tie must not promote (plan §2.E3)."""
    _exploit(cfg)
    seed = _seed(db, cfg)
    out = run_mod.run_evolution(
        db, cfg, _StubProvider(STRONG), family=FAMILY,
        seed_hypothesis_id=seed, max_rounds=2, seed=3,
    )
    assert out["decisions"][0]["promote"] is True
    assert out["decisions"][1]["promote"] is False
    assert "not above required" in out["decisions"][1]["decision"]


# -- closed loop, refusal paths, retries -------------------------------------


def test_driver_never_calls_approve(db, cfg, monkeypatch):
    _exploit(cfg)
    seed = _seed(db, cfg)

    def _boom(*args, **kwargs):
        raise AssertionError("evolution driver must never call approve")

    monkeypatch.setattr(skills, "approve", _boom)
    out = run_mod.run_evolution(
        db, cfg, _StubProvider(STRONG), family=FAMILY,
        seed_hypothesis_id=seed, max_rounds=2, seed=3,
    )
    assert out["rounds"] == 2
    # Structural check too: no `approve(` call in the driver source.
    source = Path(run_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and ((isinstance(n.func, ast.Name) and n.func.id == "approve")
             or (isinstance(n.func, ast.Attribute) and n.func.attr == "approve"))
    ]
    assert calls == []


def test_seed_validation(db, cfg):
    _exploit(cfg)
    with pytest.raises(ToolError, match="unknown seed_hypothesis_id"):
        run_mod.run_evolution(db, cfg, _StubProvider(STRONG), family=FAMILY,
                              seed_hypothesis_id="hyp_nope")
    other = _seed(db, cfg, research_family="other_family")
    with pytest.raises(ToolError, match="belongs to family"):
        run_mod.run_evolution(db, cfg, _StubProvider(STRONG), family=FAMILY,
                              seed_hypothesis_id=other)
    with pytest.raises(ToolError, match="family is required"):
        run_mod.run_evolution(db, cfg, _StubProvider(STRONG), family="  ")
    seed = _seed(db, cfg)
    with pytest.raises(ToolError, match="max_rounds must be an int >= 1"):
        run_mod.run_evolution(db, cfg, _StubProvider(STRONG), family=FAMILY,
                              seed_hypothesis_id=seed, max_rounds=0)


def test_level2_retry_then_escalate(db, cfg):
    """Provider errors retry level-2 once per round, then escalate (the next
    round runs a fresh level-1): an always-failing provider burns exactly two
    backtest calls per round before the run stops on rejections."""
    _exploit(cfg)
    cfg.research.evolution_max_consecutive_rejections = 2
    seed = _seed(db, cfg)
    provider = _StubProvider(STRONG, fail_times=10**9)
    out = run_mod.run_evolution(
        db, cfg, provider, family=FAMILY, seed_hypothesis_id=seed,
        max_rounds=8, seed=1,
    )
    assert out["status"] == "stopped_consecutive_rejections"
    assert out["rounds"] == 2
    assert all(d["stage"] == "backtest_failed" for d in out["decisions"])
    assert provider.calls == 4  # 2 rounds x 2 level-2 attempts
    # Every attempt is a counted trial (infrastructure errors fail the trial
    # without ticking the kill rule, per N6).
    n = db.conn.execute("SELECT COUNT(*) AS n FROM trials").fetchone()["n"]
    assert n == 4


def test_recorded_provider_missing_cassette_fails_rounds_closed(db, cfg, tmp_path):
    """RecordedProvider integration: an unknown fingerprint raises (fail-closed,
    never fabricated) and the driver records failed rounds instead of
    promoting unaudited numbers. (Cross-run cassette *replay* is out of reach
    by design: specs embed ULID hypothesis ids, so fingerprints differ run to
    run; replay stability itself is pinned in test_providers.py.)"""
    _exploit(cfg)
    cfg.research.evolution_max_consecutive_rejections = 2
    seed = _seed(db, cfg)
    out = run_mod.run_evolution(
        db, cfg, RecordedProvider(tmp_path / "empty.json"), family=FAMILY,
        seed_hypothesis_id=seed, max_rounds=8, seed=1,
    )
    assert out["status"] == "stopped_consecutive_rejections"
    assert all(d["stage"] == "backtest_failed" for d in out["decisions"])
    assert all("no cassette entry" in d["error"] for d in out["decisions"])
    n_feedback = db.conn.execute(
        "SELECT COUNT(*) AS n FROM evolution_feedback WHERE run_id = ?",
        (out["run_id"],),
    ).fetchone()["n"]
    assert n_feedback == 0  # nothing unaudited reaches the trace


def test_transient_provider_error_recovers_on_retry(db, cfg):
    _exploit(cfg)
    seed = _seed(db, cfg)
    provider = _StubProvider(STRONG, fail_times=1)
    out = run_mod.run_evolution(
        db, cfg, provider, family=FAMILY, seed_hypothesis_id=seed,
        max_rounds=1, seed=1,
    )
    assert out["decisions"][0]["outcome"] == "decided"
    assert provider.calls == 2


def test_root_without_llm_is_recorded_failure(db, cfg):
    """Fallback cannot invent content from nothing (component-2 contract #4):
    a headless line with no LLM records failed rounds, never hangs."""
    cfg.research.evolution_max_consecutive_rejections = 2
    out = run_mod.run_evolution(
        db, cfg, _StubProvider(STRONG), family=FAMILY, max_rounds=8, seed=1,
    )
    assert out["status"] == "stopped_consecutive_rejections"
    assert all(d["stage"] == "vary_hypothesis" for d in out["decisions"])


def test_sota_jump_creates_new_line_with_llm_root(db, cfg):
    """Default jump rule fires after one rejection (0/1 < 0.2); with an LLM
    the new root synthesizes and the run continues on the new line."""
    seed = _seed(db, cfg)
    out = run_mod.run_evolution(
        db, cfg, _StubProvider(WEAK), family=FAMILY,
        seed_hypothesis_id=seed, max_rounds=2, seed=1, llm=_root_llm(),
    )
    assert out["decisions"][0]["policy"] == "exploit_head"
    assert out["decisions"][1]["policy"] == "sota_jump_new_root"
    assert out["decisions"][1]["outcome"] == "decided"
    lines = db.conn.execute(
        "SELECT line_id FROM evolution_lines WHERE family = ?", (FAMILY,)
    ).fetchall()
    assert len(lines) == 2  # seed line + jump line
    run_row = db.conn.execute(
        "SELECT line_id FROM evolution_runs WHERE run_id = ?", (out["run_id"],)
    ).fetchone()
    assert run_row["line_id"] == out["line_id"]  # status follows the live line


# -- held-out exclusion + trace audit ----------------------------------------


def test_feedback_rows_never_carry_oos(db, cfg):
    _exploit(cfg)
    seed = _seed(db, cfg)
    oos_metrics = dict(STRONG, oos_ic=0.99, oos_sharpe=9.9)
    out = run_mod.run_evolution(
        db, cfg, _StubProvider(oos_metrics), family=FAMILY,
        seed_hypothesis_id=seed, max_rounds=1, seed=1,
    )
    assert out["decisions"][0]["outcome"] == "decided"
    row = db.conn.execute(
        "SELECT metrics_json, fitness_json FROM evolution_feedback WHERE run_id = ?",
        (out["run_id"],),
    ).fetchone()
    assert "oos" not in row["metrics_json"].lower()
    assert "oos" not in row["fitness_json"].lower()
    # ... even though the episode itself carries the OOS keys.
    ep = ledger.get_episode(db, out["decisions"][0]["episode_id"])
    assert ep["metrics"]["oos_ic"] == 0.99


def test_trajectory_events_emitted_per_round(db, cfg):
    _exploit(cfg)
    seed = _seed(db, cfg)
    out = run_mod.run_evolution(
        db, cfg, _StubProvider(STRONG), family=FAMILY,
        seed_hypothesis_id=seed, max_rounds=2, seed=3,
    )
    events = db.conn.execute(
        "SELECT kind, payload FROM trajectory_events WHERE run_id = ?",
        (out["run_id"],),
    ).fetchall()
    assert len(events) == 2
    payloads = [json.loads(e["payload"]) for e in events]
    assert payloads[0]["promote"] is True
    assert payloads[0]["notify"] is True
    assert payloads[1]["promote"] is False


def test_narration_llm_path_with_fallback_on_garbage(db, cfg):
    _exploit(cfg)
    seed = _seed(db, cfg)
    llm = FakeLLM({"narrating": "not json at all"})  # degrades to the template
    out = run_mod.run_evolution(
        db, cfg, _StubProvider(STRONG), family=FAMILY,
        seed_hypothesis_id=seed, max_rounds=1, seed=1, llm=llm,
    )
    row = db.conn.execute(
        "SELECT observations, hypothesis_evaluation, model_id"
        " FROM evolution_feedback WHERE run_id = ?",
        (out["run_id"],),
    ).fetchone()
    assert "Round 0" in row["observations"]  # deterministic template
    assert "promoted" in row["hypothesis_evaluation"]
    assert row["model_id"] is None  # fallback used: no model claimed


# -- idempotency + status -----------------------------------------------------


def test_client_id_replay_has_no_side_effects(db, cfg):
    _exploit(cfg)
    seed = _seed(db, cfg)
    provider = SimulatedProvider(cfg.artifacts_dir())
    first = run_mod.run_evolution(
        db, cfg, provider, family=FAMILY, seed_hypothesis_id=seed,
        max_rounds=1, seed=5, client_id="evo-1",
    )
    trials_before = db.conn.execute("SELECT COUNT(*) AS n FROM trials").fetchone()["n"]
    second = run_mod.run_evolution(
        db, cfg, provider, family=FAMILY, seed_hypothesis_id=seed,
        max_rounds=1, seed=5, client_id="evo-1",
    )
    assert second == first
    trials_after = db.conn.execute("SELECT COUNT(*) AS n FROM trials").fetchone()["n"]
    assert trials_after == trials_before


def test_evolution_status_reports_run_feedback_and_sota(db, cfg):
    _exploit(cfg)
    seed = _seed(db, cfg)
    out = run_mod.run_evolution(
        db, cfg, _StubProvider(STRONG), family=FAMILY,
        seed_hypothesis_id=seed, max_rounds=1, seed=3,
    )
    status = run_mod.evolution_status(db, out["run_id"])
    assert status["run"]["status"] == "finished"
    assert status["line"]["head_hypothesis_id"] == out["head"]["hypothesis_id"]
    assert len(status["feedback"]) == 1
    assert status["feedback"][0]["decision"].startswith("promoted")
    assert status["feedback"][0]["dsr_prob"] == pytest.approx(1.0)
    assert status["feedback"][0]["prompt_sha"]
    with pytest.raises(ToolError, match="unknown run_id"):
        run_mod.evolution_status(db, "erun_nope")


# -- MCP surface over the real protocol ---------------------------------------


def _client_call(server, tool, args):
    async def _run():
        async with Client(server) as client:
            return await client.call_tool(tool, args)

    result = asyncio.run(_run())
    return json.loads(result.content[0].text)


def test_mcp_evolution_run_and_status(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = Config(
        storage={"db_path": str(tmp_path / "spark.db"),
                 "artifacts_dir": str(tmp_path / "art")}
    )
    server = create_server(cfg)
    hyp = _client_call(server, "hypothesis_propose", {
        "idea": "momentum", "family": FAMILY, "universe": "toy", "horizon": "21d",
        "scope": "cross-section", "mechanism": "12-1 momentum in toy universe",
        "expected_direction": "positive", "falsifier": "IC <= 0 net of costs",
        "predicted_ic": 0.03, "predicted_sigma": 0.01,
        "hypothesis_type": "predictive",
    })["hypothesis_id"]
    out = _client_call(server, "evolution_run", {
        "family": FAMILY, "seed_hypothesis_id": hyp, "max_rounds": 1, "seed": 9,
    })
    assert out["rounds"] == 1
    assert out["decisions"][0]["outcome"] == "decided"
    status = _client_call(server, "evolution_status", {"run_id": out["run_id"]})
    assert status["run"]["run_id"] == out["run_id"]
    assert len(status["feedback"]) == 1
    # Unknown runs are refused through the tool too.
    with pytest.raises(Exception, match="unknown run_id"):
        _client_call(server, "evolution_status", {"run_id": "erun_nope"})
