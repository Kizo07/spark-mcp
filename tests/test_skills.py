"""Phase 6: skill evolution service (plan §11, tools 16-18/22, §19 M11, §21).

Covers distill evidence rules (C10: Tier-1 + Tier-2, Tier-2-only allowed),
the five mutation operators, replay held-in/held-out discipline with the
heldout_gap anti-gaming alarm, promotion with credit assignment and the
auto-with-human-notify approvals surface, human veto + rollback, the Gateway
write-path (C7) blocked + acknowledged contracts, retirement wiring, and
the four CLI subcommands / MCP tools over the same service functions (N13).
"""

import asyncio
import builtins
import json

import httpx
import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from spark_mcp import hooks, ledger, memory, skills
from spark_mcp.config import Config, GatewayConfig, PeersConfig
from spark_mcp.llm import FakeEmbedder, FakeLLM
from spark_mcp.server import create_server
from spark_mcp.storage import open_db

# Steps aligned with the seeded replay fixtures (kept deliberately partial so
# scores are neither 0 nor 1 for most fixtures).
AUDIT_STEPS = [
    "Rank the cross section on 12-1 momentum.",
    "Verify point-in-time universe membership before ranking.",
    "Apply explicit transaction cost and slippage assumptions.",
    "Check turnover and capacity before trusting the result.",
    "Report sharpe paired with a deflated significance check.",
    "Validate the signal out-of-sample.",
]
GUARD_STEPS = [
    "Estimate a panel model with fixed effects for firm and time.",
    "Cluster standard errors at the firm level.",
    "Test residuals for cross-sectional dependence.",
    "Check stability of the dominant frequencies across windows.",
]


def _episode(db, cfg) -> str:
    """propose + register: a real episode with Tier-1 trajectory evidence."""
    prop = ledger.propose_hypothesis(
        db,
        cfg,
        research_family="quant",
        hypothesis_type="predictive",
        mechanism="12-1 momentum in toy universe",
        expected_direction="positive",
        horizon="20d",
        scope="cross_section",
        falsifier="IC <= 0 net of costs",
        predicted_ic=0.03,
        predicted_sigma=0.01,
        universe="toy",
    )
    reg = ledger.register_hypothesis(db, cfg, prop["hypothesis_id"], {"signal": "mom"}, 7)
    return reg["episode_id"]


def _episode_without_evidence(db, cfg) -> str:
    episode_id = _episode(db, cfg)
    db.execute("DELETE FROM trajectory_events WHERE run_id = ?", (episode_id,))
    return episode_id


def _distill(db, cfg, name="cost-audit", steps=None, episodes=None, role="quant", **kw):
    episodes = episodes if episodes is not None else [_episode(db, cfg)]
    return skills.distill_skill(
        db, cfg, episodes, name, role, steps if steps is not None else AUDIT_STEPS, **kw
    )


def _ingest_sketch(db, cfg, episode_id, error_class="DataGap", run_id="qf-r1"):
    return hooks.ingest_run_summary(
        db,
        cfg,
        {
            "run_id": run_id,
            "episode_id": episode_id,
            "tool_call_count": 30,
            "failure_sketches": [
                {"stage": "backtest", "error_class": error_class, "resolution": "fixed"}
            ],
            "skills_activated": {},
            "tokens": 1000,
            "wall_sec": 12.0,
            "schema_version": 1,
        },
    )


def _skill_statuses(db, name):
    return {
        r["version"]: r["status"]
        for r in db.conn.execute(
            "SELECT version, status FROM skills WHERE name = ? ORDER BY version", (name,)
        )
    }


# -- migration -----------------------------------------------------------------


def test_phase6_tables_exist(db):
    for table in ("skill_candidates", "approvals", "replay_score_cache"):
        row = db.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
        ).fetchone()
        assert row is not None, table
    assert db.conn.execute("PRAGMA user_version").fetchone()[0] == 11


# -- tool 16: distill_skill ------------------------------------------------------


def test_distill_writes_draft_with_version_and_lineage(db, cfg):
    ep = _episode(db, cfg)
    result = _distill(db, cfg, episodes=[ep])
    assert result["version"] == 1
    assert result["status"] == "draft"
    assert result["body_uri"] == "spark://skills/cost-audit/1"
    row = db.conn.execute(
        "SELECT * FROM skills WHERE skill_id = ?", (result["skill_id"],)
    ).fetchone()
    lineage = json.loads(row["lineage"])
    assert lineage["source_episode_ids"] == [ep]
    assert lineage["procedure_steps"] == AUDIT_STEPS
    assert json.loads(row["credit"]) == {}
    assert row["promotion_status"] == "auto"


def test_distill_versions_increment_per_name(db, cfg):
    _distill(db, cfg)
    second = _distill(db, cfg)
    assert second["version"] == 2


def test_distill_refuses_unknown_episode(db, cfg):
    with pytest.raises(ToolError, match="unknown episode_id"):
        _distill(db, cfg, episodes=["ep_doesnotexist"])


def test_distill_refuses_episode_without_trajectory_evidence(db, cfg):
    ep = _episode_without_evidence(db, cfg)
    with pytest.raises(ToolError, match="no trajectory evidence"):
        _distill(db, cfg, episodes=[ep])


def test_distill_is_idempotent_on_client_id(db, cfg):
    ep = _episode(db, cfg)
    first = _distill(db, cfg, episodes=[ep], client_id="cli-1")
    second = _distill(db, cfg, episodes=[ep], client_id="cli-1")
    assert first["skill_id"] == second["skill_id"]
    n = db.conn.execute("SELECT COUNT(*) AS n FROM skills WHERE name = 'cost-audit'").fetchone()
    assert n["n"] == 1


def test_distill_allows_tier2_only_episode(db, cfg):
    """C10: a Tier-2 hook summary alone satisfies the evidence requirement."""
    ep = _episode_without_evidence(db, cfg)
    _ingest_sketch(db, cfg, ep)
    kinds = [
        r["kind"]
        for r in db.conn.execute(
            "SELECT kind FROM trajectory_events WHERE run_id = ?", (ep,)
        )
    ]
    assert kinds == ["hook_summary"]
    result = _distill(db, cfg, episodes=[ep])
    assert result["version"] == 1


def test_distill_validates_role_and_steps(db, cfg):
    ep = _episode(db, cfg)
    with pytest.raises(ToolError, match="role"):
        _distill(db, cfg, episodes=[ep], role="wizard")
    with pytest.raises(ToolError, match="procedure_steps"):
        _distill(db, cfg, episodes=[ep], steps=[])


# -- tool 17: propose_skill_challenger --------------------------------------------


def test_repair_appends_step_for_most_common_error_class(db, cfg):
    ep = _episode(db, cfg)
    _ingest_sketch(db, cfg, ep, error_class="DataGap")
    _ingest_sketch(db, cfg, ep, error_class="DataGap", run_id="qf-r2")
    _ingest_sketch(db, cfg, ep, error_class="LookaheadLeak", run_id="qf-r3")
    _distill(db, cfg, episodes=[ep])
    ch = skills.propose_skill_challenger(db, cfg, "cost-audit", "repair")
    assert ch["version"] == 2
    assert ch["repaired_error_class"] == "DataGap"
    assert ch["occurrences"] == 2
    assert "DataGap" in ch["procedure_steps"][-1]
    assert len(ch["procedure_steps"]) == len(AUDIT_STEPS) + 1
    cand = db.conn.execute(
        "SELECT * FROM skill_candidates WHERE candidate_id = ?", (ch["candidate_id"],)
    ).fetchone()
    assert cand["verdict"] == "pending"
    assert cand["parent_version"] == 1
    assert cand["mutation_op"] == "repair"
    assert _skill_statuses(db, "cost-audit") == {1: "draft", 2: "challenger"}


def test_repair_refuses_without_tier2_failures(db, cfg):
    _distill(db, cfg)
    with pytest.raises(ToolError, match="failure sketches"):
        skills.propose_skill_challenger(db, cfg, "cost-audit", "repair")


def test_split_takes_first_coherent_subprocedure(db, cfg):
    _distill(db, cfg)
    ch = skills.propose_skill_challenger(db, cfg, "cost-audit", "split")
    assert ch["procedure_steps"] == AUDIT_STEPS[:3]
    assert ch["dropped_steps"] == 3


def test_split_is_deterministic(db, cfg):
    ep = _episode(db, cfg)
    _distill(db, cfg, name="det-a", episodes=[ep])
    _distill(db, cfg, name="det-b", episodes=[ep])
    a = skills.propose_skill_challenger(db, cfg, "det-a", "split", client_id="s1")
    b = skills.propose_skill_challenger(db, cfg, "det-b", "split", client_id="s2")
    assert a["procedure_steps"] == b["procedure_steps"]


def test_split_refuses_single_step_procedure(db, cfg):
    _distill(db, cfg, steps=["Only one step."])
    with pytest.raises(ToolError, match="multi-step"):
        skills.propose_skill_challenger(db, cfg, "cost-audit", "split")


def test_merge_combines_same_role_steps_deduped(db, cfg):
    ep = _episode(db, cfg)
    _distill(db, cfg, name="panel-guard", steps=GUARD_STEPS, episodes=[ep])
    _distill(db, cfg, name="audit-extra", episodes=[ep], steps=[
        "Check turnover and capacity before trusting the result.",
        "Cluster standard errors at the firm level.",
    ])
    ch = skills.propose_skill_challenger(db, cfg, "panel-guard", "merge")
    assert ch["merged_with"] == "audit-extra:1"
    steps = ch["procedure_steps"]
    assert len(steps) == len(GUARD_STEPS) + 1  # one exact duplicate removed
    assert "Check turnover and capacity before trusting the result." in steps


def test_merge_refuses_without_second_skill(db, cfg):
    _distill(db, cfg)
    with pytest.raises(ToolError, match="second live skill"):
        skills.propose_skill_challenger(db, cfg, "cost-audit", "merge")


def test_paraphrase_degraded_reorders_and_clarifies(db, cfg):
    steps = [
        "Run the momentum ranking",
        "Apply explicit transaction cost and slippage assumptions",
        "Verify point-in-time universe membership before ranking",
    ]
    _distill(db, cfg, steps=steps)
    ch = skills.propose_skill_challenger(db, cfg, "cost-audit", "paraphrase", llm=None)
    assert ch["paraphrase_mode"] == "degraded_reorder_clarify"
    assert ch["procedure_steps"][0].startswith("Run the momentum ranking")
    assert ch["procedure_steps"][-1].startswith("Verify")
    assert all(s.endswith(".") for s in ch["procedure_steps"])


def test_paraphrase_uses_llm_when_configured(db, cfg):
    _distill(db, cfg)
    llm = FakeLLM(
        replies={
            "Return a JSON array": '["Rewritten step one.", "Rewritten step two."]'
        }
    )
    ch = skills.propose_skill_challenger(db, cfg, "cost-audit", "paraphrase", llm=llm)
    assert ch["paraphrase_mode"] == "llm"
    assert ch["procedure_steps"] == ["Rewritten step one.", "Rewritten step two."]


def test_paraphrase_falls_back_when_llm_reply_unparsable(db, cfg):
    _distill(db, cfg)
    llm = FakeLLM(replies={"Return a JSON array": "not json at all"})
    ch = skills.propose_skill_challenger(db, cfg, "cost-audit", "paraphrase", llm=llm)
    assert ch["paraphrase_mode"] == "degraded_reorder_clarify"


def test_gepa_operator_runs_within_caps_and_is_deterministic(db, cfg):
    ep = _episode(db, cfg)
    _distill(db, cfg, name="gepa-a", episodes=[ep])
    _distill(db, cfg, name="gepa-b", episodes=[ep])
    a = skills.propose_skill_challenger(db, cfg, "gepa-a", "gepa")
    b = skills.propose_skill_challenger(db, cfg, "gepa-b", "gepa")
    assert a["gepa"]["population_size"] == cfg.research.gepa_population_size
    assert a["gepa"]["generations"] == cfg.research.gepa_generations
    assert a["procedure_steps"] == b["procedure_steps"]  # seed derived from parent content
    # F1 behavior change (was: challenger always differed via appended rubric
    # text — the leak): with no Tier-2 failure evidence the guard op appends
    # only the generic audit sentence, which adds no coverage, so the parent
    # itself may legitimately win best. Pin well-formedness + no-leak instead.
    assert a["procedure_steps"]
    blob = " ".join(a["procedure_steps"]).lower()
    for fixture in skills.load_fixtures():
        for item in fixture["rubric"]:
            assert item.lower() not in blob, item


def test_gepa_operator_passes_failure_sketches(db, cfg, monkeypatch):
    """F1 wiring: _op_gepa feeds the source episodes' Tier-2 failure sketches
    to run_gepa (failure evidence, never rubric text)."""
    from spark_mcp.evolve import gepa as gepa_mod

    ep = _episode(db, cfg)
    _ingest_sketch(db, cfg, ep, error_class="DataGap")
    _distill(db, cfg, name="gepa-sketch", episodes=[ep])
    captured = {}
    real_run_gepa = gepa_mod.run_gepa

    def spy(seed_individuals, fixtures, **kwargs):
        captured.update(kwargs)
        return real_run_gepa(seed_individuals, fixtures, **kwargs)

    monkeypatch.setattr(gepa_mod, "run_gepa", spy)
    skills.propose_skill_challenger(db, cfg, "gepa-sketch", "gepa")
    assert [s.get("error_class") for s in captured["failure_sketches"]] == ["DataGap"]


def test_gepa_operator_threads_llm_to_run_gepa(db, cfg, monkeypatch):
    """F2 wiring: the challenger passes its llm through to run_gepa so the
    `reflect` op can rewrite (and None keeps the seeded fallback)."""
    import json

    from spark_mcp.evolve import gepa as gepa_mod
    from spark_mcp.llm import FakeLLM

    ep = _episode(db, cfg)
    _distill(db, cfg, name="gepa-llm", episodes=[ep])
    captured = {}
    real_run_gepa = gepa_mod.run_gepa

    def spy(seed_individuals, fixtures, **kwargs):
        captured.update(kwargs)
        return real_run_gepa(seed_individuals, fixtures, **kwargs)

    monkeypatch.setattr(gepa_mod, "run_gepa", spy)
    llm = FakeLLM({"procedure": json.dumps({"steps": ["Rank on quality."]})})
    ch = skills.propose_skill_challenger(db, cfg, "gepa-llm", "gepa", llm=llm)
    assert captured["llm"] is llm
    assert ch["procedure_steps"]
    # Without an LLM the op degrades deterministically (existing behavior).
    _distill(db, cfg, name="gepa-nollm", episodes=[ep])
    a = skills.propose_skill_challenger(db, cfg, "gepa-nollm", "gepa")
    assert captured["llm"] is None
    assert a["procedure_steps"]


def test_challenger_refuses_unknown_op_and_unknown_skill(db, cfg):
    _distill(db, cfg)
    with pytest.raises(ToolError, match="unknown op"):
        skills.propose_skill_challenger(db, cfg, "cost-audit", "transmute")
    with pytest.raises(ToolError, match="unknown skill"):
        skills.propose_skill_challenger(db, cfg, "nope", "split")


def test_challenger_is_idempotent_on_client_id(db, cfg):
    _distill(db, cfg)
    a = skills.propose_skill_challenger(db, cfg, "cost-audit", "split", client_id="c-1")
    b = skills.propose_skill_challenger(db, cfg, "cost-audit", "split", client_id="c-1")
    assert a["candidate_id"] == b["candidate_id"]
    n = db.conn.execute("SELECT COUNT(*) AS n FROM skill_candidates").fetchone()
    assert n["n"] == 1


# -- tool 18: run_replay ------------------------------------------------------------


def test_fixture_seed_file_loads_with_heldout_split():
    fixtures = skills.load_fixtures()
    assert len(fixtures) >= 5
    held_in = [f for f in fixtures if f.get("held_in", True)]
    held_out = [f for f in fixtures if not f.get("held_in", True)]
    assert len(held_in) >= 4
    assert len(held_out) >= 1
    for fixture in fixtures:
        assert {"task_id", "prompt", "family", "rubric", "reference_keywords"} <= set(fixture)


def _pending_challenger(db, cfg, name="cost-audit"):
    _distill(db, cfg, name=name)
    return skills.propose_skill_challenger(db, cfg, name, "split")


def test_run_replay_reports_split_and_gap_and_writes_candidate(db, cfg):
    ch = _pending_challenger(db, cfg)
    verdict = skills.run_replay(db, cfg, skill_name="cost-audit", seed=3)
    assert verdict["n_held_in"] == 4
    assert verdict["n_held_out"] == 2
    assert set(verdict["held_in"]) == {
        "replay_momentum_cost_audit",
        "replay_regime_switch",
        "replay_causal_dag",
        "replay_leakage_screen",
    }
    assert set(verdict["held_out"]) == {"replay_panel_fe", "replay_spectral_micro"}
    expected_gap = abs(
        sum(verdict["held_in"].values()) / 4 - sum(verdict["held_out"].values()) / 2
    )
    assert verdict["heldout_gap"] == pytest.approx(expected_gap)
    assert verdict["gap_alarm"] is False
    assert verdict["scored_as"] == "challenger"
    cand = db.conn.execute(
        "SELECT * FROM skill_candidates WHERE candidate_id = ?", (ch["candidate_id"],)
    ).fetchone()
    eval_scores = json.loads(cand["eval_scores"])
    assert eval_scores["held_in"] == verdict["held_in"]
    assert eval_scores["held_out"] == verdict["held_out"]
    assert cand["heldout_gap"] == pytest.approx(expected_gap)


def test_run_replay_deterministic_without_llm(db, cfg):
    _pending_challenger(db, cfg)
    v1 = skills.run_replay(db, cfg, skill_name="cost-audit", seed=1)
    v2 = skills.run_replay(db, cfg, skill_name="cost-audit", seed=2)
    assert v1["held_in"] == v2["held_in"]
    assert v1["held_out"] == v2["held_out"]


def test_run_replay_suite_and_task_filters(db, cfg):
    _pending_challenger(db, cfg)
    econ = skills.run_replay(db, cfg, skill_name="cost-audit", suite="econometrics")
    assert econ["n_held_in"] == 0
    assert econ["n_held_out"] == 1
    one = skills.run_replay(
        db, cfg, skill_name="cost-audit", task_ids=["replay_momentum_cost_audit"]
    )
    assert one["n_held_in"] == 1
    assert one["n_held_out"] == 0
    with pytest.raises(ToolError, match="unknown replay task_id"):
        skills.run_replay(db, cfg, skill_name="cost-audit", task_ids=["replay_nope"])
    with pytest.raises(ToolError, match="matches no replay fixtures"):
        skills.run_replay(db, cfg, skill_name="cost-audit", suite="alchemy")


def test_run_replay_scores_all_pending_skills_when_name_omitted(db, cfg):
    _pending_challenger(db, cfg, name="skill-a")
    _pending_challenger(db, cfg, name="skill-b")
    result = skills.run_replay(db, cfg)
    assert set(result["skills"]) == {"skill-a", "skill-b"}


def test_run_replay_regression_run_without_pending_candidate(db, cfg):
    _distill(db, cfg)
    verdict = skills.run_replay(db, cfg, skill_name="cost-audit")
    assert verdict["scored_as"] == "latest"
    assert verdict["version"] == 1
    n = db.conn.execute(
        "SELECT COUNT(*) AS n FROM skill_candidates WHERE eval_scores IS NOT NULL"
    ).fetchone()
    assert n["n"] == 0


def test_run_replay_llm_judge_caches_per_model(db, cfg):
    _pending_challenger(db, cfg)
    llm = FakeLLM(model="judge/1")
    v = skills.run_replay(db, cfg, skill_name="cost-audit", llm=llm)
    assert all(0.0 <= s <= 1.0 for s in v["held_in"].values())
    calls_after_first = len(llm.calls)
    assert calls_after_first > 0
    skills.run_replay(db, cfg, skill_name="cost-audit", llm=llm)  # fully cached
    assert len(llm.calls) == calls_after_first
    rows = db.conn.execute("SELECT DISTINCT model_id FROM replay_score_cache").fetchall()
    assert [r["model_id"] for r in rows] == ["judge/1"]
    # A different model id is a different cache key: re-judged, not reused.
    other = FakeLLM(model="judge/2")
    skills.run_replay(db, cfg, skill_name="cost-audit", llm=other)
    assert len(other.calls) > 0


# -- promotion gate + auto-with-human-notify -----------------------------------------


def _promoted(db, cfg, name="cost-audit", episodes=None):
    if episodes is None:
        _pending_challenger(db, cfg, name=name)
    else:
        _distill(db, cfg, name=name, episodes=episodes)
        skills.propose_skill_challenger(db, cfg, name, "split")
    skills.run_replay(db, cfg, skill_name=name)
    return skills.decide_promotion(
        db,
        cfg,
        name,
        skills._newest_pending_candidate(db, name)["candidate_id"],
        "promoted",
        "held-in green, gap small",
        "gate",
    )


def test_decide_requires_replay_evidence(db, cfg):
    _pending_challenger(db, cfg)
    cand = skills._newest_pending_candidate(db, "cost-audit")
    with pytest.raises(ToolError, match="replay evidence"):
        skills.decide_promotion(
            db, cfg, "cost-audit", cand["candidate_id"], "promoted", "x", "gate"
        )


def test_decide_refuses_already_decided_candidate(db, cfg):
    res = _promoted(db, cfg)
    with pytest.raises(ToolError, match="already decided"):
        skills.decide_promotion(
            db, cfg, "cost-audit", res["candidate_id"], "promoted", "x", "gate"
        )


def test_promotion_assigns_credit_and_records_gate_approval(db, cfg):
    ep = _episode(db, cfg)
    res = _promoted(db, cfg, episodes=[ep])
    assert res["champion_version"] == 2
    assert res["gateway"]["status"] == "blocked"
    assert res["gateway"]["payload"]["reason"] == "no_gateway_endpoint_configured"
    row = db.conn.execute(
        "SELECT * FROM skills WHERE name = 'cost-audit' AND version = 2"
    ).fetchone()
    assert row["status"] == "champion"
    # C7 blocked state: staged in Spark's registry, the system of record.
    assert row["promotion_status"] == "staged_in_spark"
    assert row["body_uri"] == "spark://skills/cost-audit/2"
    credit = json.loads(row["credit"])
    source = json.loads(
        db.conn.execute(
            "SELECT source_episode_ids FROM skill_candidates WHERE candidate_id = ?",
            (res["candidate_id"],),
        ).fetchone()["source_episode_ids"]
    )
    assert credit == {ep_id: pytest.approx(1.0 / len(source)) for ep_id in source}
    # champion version recorded on the source episodes
    ep_row = db.conn.execute(
        "SELECT skill_versions FROM episodes WHERE episode_id = ?", (ep,)
    ).fetchone()
    ep_versions = json.loads(ep_row["skill_versions"])
    assert ep_versions["cost-audit"] == "2"
    # M11 auto-with-human-notify: approvals row with decided_by='gate' exists,
    # and the notify trajectory event is recorded.
    approval = db.conn.execute(
        "SELECT * FROM approvals WHERE target_type = 'skill' AND target_id = ?",
        (res["candidate_id"],),
    ).fetchone()
    assert approval is not None
    assert approval["decision"] == "approved"
    assert approval["decided_by"] == "gate"
    notify = db.conn.execute(
        "SELECT payload FROM trajectory_events WHERE run_id = 'cost-audit'"
        " AND kind = 'tool_call' ORDER BY seq DESC LIMIT 1"
    ).fetchone()
    assert json.loads(notify["payload"])["tool"] == "promote_via_gateway"
    history = json.loads(row["benchmark_history"])
    assert history[-1]["candidate_id"] == res["candidate_id"]


def test_rejected_decision_marks_challenger_rejected(db, cfg):
    _pending_challenger(db, cfg)
    skills.run_replay(db, cfg, skill_name="cost-audit")
    cand = skills._newest_pending_candidate(db, "cost-audit")
    res = skills.decide_promotion(
        db, cfg, "cost-audit", cand["candidate_id"], "rejected", "weak", "gate"
    )
    assert res["gateway"]["status"] == "not_attempted"
    assert _skill_statuses(db, "cost-audit") == {1: "draft", 2: "rejected"}
    assert res["credit"] == {}


def test_rejected_challenger_does_not_block_the_next_one(db, cfg):
    """Versions are per-name and never reused: after rejecting v2, the next
    challenger from the same parent is v3 (parent+1 would collide with the
    rejected row)."""
    _pending_challenger(db, cfg)
    skills.run_replay(db, cfg, skill_name="cost-audit")
    first = skills._newest_pending_candidate(db, "cost-audit")
    skills.decide_promotion(
        db, cfg, "cost-audit", first["candidate_id"], "rejected", "weak", "gate"
    )
    retry = skills.propose_skill_challenger(db, cfg, "cost-audit", "split", client_id="retry")
    assert retry["version"] == 3
    assert retry["parent_version"] == 1
    verdict = skills.run_replay(db, cfg, skill_name="cost-audit")
    assert verdict["version"] == 3
    assert verdict["scored_as"] == "challenger"


# -- tool 22: approve (M11) ------------------------------------------------------------

def _two_champion_setup(db, cfg):
    """v2 promoted to champion over draft v1; v3 challenger then promoted so
    v2 becomes the rollback_target."""
    _promoted(db, cfg)  # v2 champion
    ch3 = skills.propose_skill_challenger(db, cfg, "cost-audit", "split", client_id="v3")
    skills.run_replay(db, cfg, skill_name="cost-audit")
    res3 = skills.decide_promotion(
        db, cfg, "cost-audit", ch3["candidate_id"], "promoted", "better", "gate"
    )
    assert _skill_statuses(db, "cost-audit") == {
        1: "draft",
        2: "rollback_target",
        3: "champion",
    }
    return ch3, res3


def test_human_veto_rolls_back_to_rollback_target(db, cfg):
    ch3, _ = _two_champion_setup(db, cfg)
    veto = skills.approve(
        db, cfg, "skill", ch3["candidate_id"], "rejected",
        "overfit held-in", decided_by="human:kanav",
    )
    assert veto["duplicated"] is False
    assert veto["applied"] == {"vetoed_version": 3, "restored_version": 2}
    assert _skill_statuses(db, "cost-audit") == {1: "draft", 2: "champion", 3: "rejected"}


def test_approve_is_idempotent_on_duplicates(db, cfg):
    ch3, _ = _two_champion_setup(db, cfg)
    first = skills.approve(
        db, cfg, "skill", ch3["candidate_id"], "rejected", "overfit", decided_by="human:a"
    )
    second = skills.approve(
        db, cfg, "skill", ch3["candidate_id"], "rejected", "overfit", decided_by="human:b"
    )
    assert second["duplicated"] is True
    assert second["approval_id"] == first["approval_id"]
    n = db.conn.execute(
        "SELECT COUNT(*) AS n FROM approvals WHERE target_type = 'skill'"
        " AND target_id = ? AND decision = 'rejected'",
        (ch3["candidate_id"],),
    ).fetchone()
    assert n["n"] == 1  # one human veto; the gate's notify rows are 'approved'


def test_approve_ratifies_staged_champion(db, cfg):
    ch3, _ = _two_champion_setup(db, cfg)
    ok = skills.approve(
        db, cfg, "skill", ch3["candidate_id"], "approved",
        "looks right", decided_by="human:kanav",
    )
    assert ok["applied"] == {"ratified_version": 3}
    assert _skill_statuses(db, "cost-audit")[3] == "champion"


def test_approve_skill_refuses_non_promoted_candidate(db, cfg):
    _pending_challenger(db, cfg)
    cand = skills._newest_pending_candidate(db, "cost-audit")
    with pytest.raises(ToolError, match="promoted"):
        skills.approve(db, cfg, "skill", cand["candidate_id"], "rejected", "x")


def test_approve_adoption_records_human_authorization(db, cfg):
    res = _promoted(db, cfg)
    row = db.conn.execute(
        "SELECT skill_id FROM skills WHERE name = 'cost-audit' AND version = 2"
    ).fetchone()
    ok = skills.approve(
        db, cfg, "adoption", row["skill_id"], "approved",
        "deploy to book X", decided_by="human:desk",
    )
    assert ok["applied"]["adoption"] == "recorded_out_of_band"
    assert ok["applied"]["skill"] == "cost-audit:2"
    approval = db.conn.execute(
        "SELECT * FROM approvals WHERE approval_id = ?", (ok["approval_id"],)
    ).fetchone()
    assert approval["target_type"] == "adoption"
    # Adoption is recorded separately from the gate's promotion-notify row.
    assert res["candidate_id"] != approval["target_id"]


def test_approve_adoption_requires_live_champion(db, cfg):
    _distill(db, cfg)  # v1 draft
    row = db.conn.execute(
        "SELECT skill_id FROM skills WHERE name = 'cost-audit' AND version = 1"
    ).fetchone()
    with pytest.raises(ToolError, match="champion"):
        skills.approve(db, cfg, "adoption", row["skill_id"], "approved", "x")


def test_approve_belief_resolves_pending_human(db, cfg):
    ep = _episode(db, cfg)
    promoted = memory.promote_belief(
        db, cfg, [ep], "low confidence claim", None, "r", 0.3
    )
    assert promoted["promotion_status"] == "pending_human"
    ok = skills.approve(
        db, cfg, "belief", promoted["belief_id"], "approved",
        "human checked", decided_by="human:kanav",
    )
    assert ok["applied"] == {"belief_id": promoted["belief_id"], "promotion_status": "approved"}
    row = db.conn.execute(
        "SELECT promotion_status FROM beliefs WHERE belief_id = ?", (promoted["belief_id"],)
    ).fetchone()
    assert row["promotion_status"] == "approved"


def test_approve_belief_refuses_auto_promoted(db, cfg):
    ep = _episode(db, cfg)
    promoted = memory.promote_belief(
        db, cfg, [ep], "an unrelated claim about copper", None, "r", 0.95
    )
    assert promoted["promotion_status"] == "auto"
    with pytest.raises(ToolError, match="pending_human"):
        skills.approve(db, cfg, "belief", promoted["belief_id"], "approved", "x")


def test_approve_persists_structured_evidence_in_trajectory(db, cfg):
    _promoted(db, cfg)
    row = db.conn.execute(
        "SELECT skill_id FROM skills WHERE name = 'cost-audit' AND version = 2"
    ).fetchone()
    evidence = {"reviewer": "desk", "checks": ["sharpe", "pit"], "score": 0.9}
    ok = skills.approve(
        db, cfg, "adoption", row["skill_id"], "approved",
        "deploy to book X", decided_by="human:desk", evidence=evidence,
    )
    assert ok["evidence"] == evidence
    traj = db.conn.execute(
        "SELECT payload FROM trajectory_events WHERE run_id = ? AND kind = 'tool_call'"
        " ORDER BY seq DESC LIMIT 1",
        (row["skill_id"],),
    ).fetchone()
    payload = json.loads(traj["payload"])
    assert payload["tool"] == "approve"
    assert payload["evidence"] == evidence


def test_approve_rejects_oversized_evidence(db, cfg):
    _promoted(db, cfg)
    row = db.conn.execute(
        "SELECT skill_id FROM skills WHERE name = 'cost-audit' AND version = 2"
    ).fetchone()
    oversized = {f"k{i}": i for i in range(17)}
    with pytest.raises(ToolError, match="at most 16 keys"):
        skills.approve(
            db, cfg, "adoption", row["skill_id"], "approved",
            "x", decided_by="human:desk", evidence=oversized,
        )
    # no approval side effect from the rejected call
    n = db.conn.execute(
        "SELECT COUNT(*) AS n FROM approvals WHERE target_type = 'adoption'"
        " AND target_id = ?",
        (row["skill_id"],),
    ).fetchone()
    assert n["n"] == 0


def test_approve_duplicate_unchanged_with_evidence(db, cfg):
    ch3, _ = _two_champion_setup(db, cfg)
    first = skills.approve(
        db, cfg, "skill", ch3["candidate_id"], "approved",
        "looks right", decided_by="human:a", evidence={"a": 1},
    )
    second = skills.approve(
        db, cfg, "skill", ch3["candidate_id"], "approved",
        "looks right", decided_by="human:b", evidence={"b": 2},
    )
    assert second["duplicated"] is True
    assert second["approval_id"] == first["approval_id"]
    assert "evidence" not in second
    n = db.conn.execute(
        "SELECT COUNT(*) AS n FROM approvals WHERE target_type = 'skill'"
        " AND target_id = ? AND decision = 'approved' AND decided_by != 'gate'",
        (ch3["candidate_id"],),
    ).fetchone()
    assert n["n"] == 1


# -- Gateway write-path (C7) -------------------------------------------------------------


def _with_gateway(cfg) -> Config:
    return cfg.model_copy(
        update={
            "peers": PeersConfig(
                gateway=GatewayConfig(enabled=True, url="http://gateway.test/skills")
            )
        }
    )


def test_gateway_blocked_without_endpoint_and_no_filesystem(db, cfg, monkeypatch):
    _promoted(db, cfg)  # gateway attempt inside decide_promotion is blocked

    def _forbid_open(*args, **kwargs):
        raise AssertionError("filesystem deployment is forbidden (§3.3)")

    monkeypatch.setattr(builtins, "open", _forbid_open)
    row = db.conn.execute(
        "SELECT * FROM skill_candidates WHERE verdict = 'promoted'"
    ).fetchone()
    with pytest.raises(skills.GatewayWriteBlocked) as excinfo:
        skills.promote_via_gateway(db, cfg, row)
    assert excinfo.value.payload["reason"] == "no_gateway_endpoint_configured"
    skill = db.conn.execute(
        "SELECT promotion_status FROM skills WHERE name = 'cost-audit' AND version = 2"
    ).fetchone()
    assert skill["promotion_status"] == "staged_in_spark"


def test_gateway_happy_path_posts_manifest_and_acks(db, cfg, monkeypatch):
    _promoted(db, cfg)
    cfg = _with_gateway(cfg)
    row = db.conn.execute("SELECT * FROM skill_candidates WHERE verdict = 'promoted'").fetchone()
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = json.loads(request.content)
        manifest = body["manifest"]
        return httpx.Response(
            200,
            json={
                "manifest_ack": True,
                "manifest": {"skill_name": manifest["skill_name"], "version": manifest["version"]},
            },
        )

    monkeypatch.setenv("QUANTFLOW_GATEWAY_TOKEN", "tok-123")
    ack = skills.promote_via_gateway(db, cfg, row, transport=httpx.MockTransport(handler))
    assert ack["manifest_ack"] is True
    request = requests[0]
    assert request.headers["Authorization"] == "Bearer tok-123"
    manifest = json.loads(request.content)["manifest"]
    assert manifest["schema_version"] == skills.GATEWAY_MANIFEST_SCHEMA_VERSION
    assert manifest["skill_name"] == "cost-audit"
    assert manifest["version"] == 2
    assert manifest["status"] == "champion"
    assert manifest["body_uri"] == "spark://skills/cost-audit/2"
    assert manifest["procedure_steps"]  # body carried in the manifest
    assert manifest["replay"]["heldout_gap"] is not None
    assert manifest["credit"]  # assigned before the gateway push
    skill = db.conn.execute(
        "SELECT promotion_status FROM skills WHERE name = 'cost-audit' AND version = 2"
    ).fetchone()
    assert skill["promotion_status"] == "deployed_to_gateway"


def test_gateway_blocks_on_wrong_contract(db, cfg, monkeypatch):
    _promoted(db, cfg)
    cfg = _with_gateway(cfg)
    monkeypatch.setenv("QUANTFLOW_GATEWAY_TOKEN", "tok-123")
    row = db.conn.execute("SELECT * FROM skill_candidates WHERE verdict = 'promoted'").fetchone()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})  # 200 but no manifest ack

    with pytest.raises(skills.GatewayWriteBlocked) as excinfo:
        skills.promote_via_gateway(db, cfg, row, transport=httpx.MockTransport(handler))
    assert excinfo.value.payload["reason"] == "manifest_not_acknowledged"
    skill = db.conn.execute(
        "SELECT promotion_status FROM skills WHERE name = 'cost-audit' AND version = 2"
    ).fetchone()
    assert skill["promotion_status"] == "staged_in_spark"


def test_gateway_blocks_on_transport_error_and_bad_status(db, cfg, monkeypatch):
    _promoted(db, cfg)
    cfg = _with_gateway(cfg)
    monkeypatch.setenv("QUANTFLOW_GATEWAY_TOKEN", "tok-123")
    row = db.conn.execute("SELECT * FROM skill_candidates WHERE verdict = 'promoted'").fetchone()

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    with pytest.raises(skills.GatewayWriteBlocked) as excinfo:
        skills.promote_via_gateway(db, cfg, row, transport=httpx.MockTransport(boom))
    assert excinfo.value.payload["reason"] == "transport_error"

    def not_found(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "nope"})

    with pytest.raises(skills.GatewayWriteBlocked) as excinfo:
        skills.promote_via_gateway(db, cfg, row, transport=httpx.MockTransport(not_found))
    assert excinfo.value.payload["reason"] == "unexpected_status"
    assert excinfo.value.payload["status_code"] == 404


def test_gateway_token_missing_blocks_and_stays_staged(db, cfg, monkeypatch):
    _promoted(db, cfg)
    cfg = _with_gateway(cfg)
    assert cfg.peers.gateway.require_token is True
    monkeypatch.delenv("QUANTFLOW_GATEWAY_TOKEN", raising=False)
    row = db.conn.execute("SELECT * FROM skill_candidates WHERE verdict = 'promoted'").fetchone()
    called = []

    def handler(request: httpx.Request) -> httpx.Response:
        called.append(request)
        return httpx.Response(200, json={"manifest_ack": True, "manifest": {}})

    with pytest.raises(skills.GatewayWriteBlocked) as excinfo:
        skills.promote_via_gateway(db, cfg, row, transport=httpx.MockTransport(handler))
    assert excinfo.value.payload["reason"] == "gateway_token_missing"
    assert called == []  # never send unauthenticated POST
    skill = db.conn.execute(
        "SELECT promotion_status FROM skills WHERE name = 'cost-audit' AND version = 2"
    ).fetchone()
    assert skill["promotion_status"] == "staged_in_spark"


def test_gateway_empty_token_blocks_and_stays_staged(db, cfg, monkeypatch):
    _promoted(db, cfg)
    cfg = _with_gateway(cfg)
    monkeypatch.setenv("QUANTFLOW_GATEWAY_TOKEN", "")
    row = db.conn.execute("SELECT * FROM skill_candidates WHERE verdict = 'promoted'").fetchone()

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("must not POST without token")

    with pytest.raises(skills.GatewayWriteBlocked) as excinfo:
        skills.promote_via_gateway(db, cfg, row, transport=httpx.MockTransport(handler))
    assert excinfo.value.payload["reason"] == "gateway_token_missing"
    skill = db.conn.execute(
        "SELECT promotion_status FROM skills WHERE name = 'cost-audit' AND version = 2"
    ).fetchone()
    assert skill["promotion_status"] == "staged_in_spark"


# -- retirement + rollback ---------------------------------------------------------------


def test_retire_champion_sets_rollback_target(db, cfg):
    _two_champion_setup(db, cfg)  # v3 champion, v2 rollback_target
    res = skills.retire_skill(db, "cost-audit", 3, "superseded by desk policy")
    assert res["was_champion"] is True
    assert res["rollback_target_version"] == 2
    assert _skill_statuses(db, "cost-audit") == {1: "draft", 2: "rollback_target", 3: "retired"}


def test_retire_non_champion_sets_no_rollback(db, cfg):
    _distill(db, cfg)
    res = skills.retire_skill(db, "cost-audit", 1, "never worked")
    assert res["was_champion"] is False
    assert res["rollback_target_version"] is None
    assert _skill_statuses(db, "cost-audit") == {1: "retired"}


def test_retire_unknown_or_twice_refused(db, cfg):
    _distill(db, cfg)
    with pytest.raises(ToolError, match="unknown skill"):
        skills.retire_skill(db, "cost-audit", 99, "x")
    skills.retire_skill(db, "cost-audit", 1, "x")
    with pytest.raises(ToolError, match="already retired"):
        skills.retire_skill(db, "cost-audit", 1, "x")


# -- entry points: MCP tools + CLI (plan §13/N13: same service functions) -------------------


def test_phase6_tools_over_mcp(tmp_path):
    cfg = Config(
        storage={"db_path": f"{tmp_path}/spark.db", "artifacts_dir": f"{tmp_path}/art"}
    )
    db = open_db(cfg.db_path())
    ep = _episode(db, cfg)
    db.close()
    server = create_server(cfg, embedder=FakeEmbedder(), seeds=False)

    async def run():
        async with Client(server) as client:
            distilled = await client.call_tool(
                "distill_skill",
                {
                    "episode_ids": [ep],
                    "name": "cost-audit",
                    "role": "quant",
                    "procedure_steps": AUDIT_STEPS,
                },
            )
            challenged = await client.call_tool(
                "propose_skill_challenger",
                {"skill_name": "cost-audit", "op": "split"},
            )
            replayed = await client.call_tool(
                "run_replay", {"skill_name": "cost-audit"}
            )
            promoted_belief = await client.call_tool(
                "memory_promote",
                {
                    "episode_ids": [ep],
                    "claim": "a tentative claim needing a human",
                    "confidence": 0.2,
                },
            )
            approved = await client.call_tool(
                "approve",
                {
                    "target_type": "belief",
                    "target_id": json.loads(promoted_belief.content[0].text)["belief_id"],
                    "decision": "approved",
                    "rationale": "checked by the desk",
                },
            )
            return distilled, challenged, replayed, approved

    distilled, challenged, replayed, approved = asyncio.run(run())
    d = json.loads(distilled.content[0].text)
    c = json.loads(challenged.content[0].text)
    v = json.loads(replayed.content[0].text)
    a = json.loads(approved.content[0].text)
    assert d["version"] == 1 and d["status"] == "draft"
    assert c["op"] == "split" and c["version"] == 2
    assert v["heldout_gap"] >= 0.0 and v["n_held_in"] == 4
    assert a["duplicated"] is False and a["applied"]["promotion_status"] == "approved"


def _write_config(tmp_path) -> str:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"storage:\n  db_path: {tmp_path}/spark.db\n  artifacts_dir: {tmp_path}/art\n",
        encoding="utf-8",
    )
    return str(cfg_path)


def test_skills_distill_cli(tmp_path, cfg):
    from typer.testing import CliRunner

    from spark_mcp.cli import app

    db = open_db(cfg.db_path())
    ep = _episode(db, cfg)
    db.close()
    steps_path = tmp_path / "steps.json"
    steps_path.write_text(json.dumps(AUDIT_STEPS), encoding="utf-8")

    runner = CliRunner()
    invoked = runner.invoke(
        app,
        [
            "skills", "distill",
            "--episode-ids", ep,
            "--name", "cost-audit",
            "--role", "quant",
            "--procedure-steps", f"@{steps_path}",
            "--config", _write_config(tmp_path),
        ],
    )
    assert invoked.exit_code == 0, invoked.output
    result = json.loads(invoked.output)
    assert result["version"] == 1
    assert result["status"] == "draft"


def test_skills_challenger_and_replay_cli(tmp_path, cfg):
    from typer.testing import CliRunner

    from spark_mcp.cli import app

    db = open_db(cfg.db_path())
    _distill(db, cfg)  # v1 draft; the CLI proposes the challenger itself
    db.close()
    runner = CliRunner()

    challenged = runner.invoke(
        app,
        ["skills", "challenger", "--skill-name", "cost-audit", "--op", "split",
         "--config", _write_config(tmp_path)],
    )
    assert challenged.exit_code == 0, challenged.output
    ch = json.loads(challenged.output)
    assert ch["op"] == "split"

    replayed = runner.invoke(
        app,
        ["skills", "replay", "--skill-name", "cost-audit", "--seed", "5",
         "--config", _write_config(tmp_path)],
    )
    assert replayed.exit_code == 0, replayed.output
    verdict = json.loads(replayed.output)
    assert verdict["n_held_in"] == 4
    assert verdict["heldout_gap"] >= 0.0


def test_skills_approve_cli(tmp_path, cfg):
    from typer.testing import CliRunner

    from spark_mcp.cli import app

    db = open_db(cfg.db_path())
    _promoted(db, cfg)  # v2 champion staged (gateway absent)
    skill_id = db.conn.execute(
        "SELECT skill_id FROM skills WHERE name = 'cost-audit' AND version = 2"
    ).fetchone()["skill_id"]
    db.close()
    runner = CliRunner()
    invoked = runner.invoke(
        app,
        [
            "skills", "approve",
            "--target-type", "adoption",
            "--target-id", skill_id,
            "--decision", "approved",
            "--rationale", "desk approved the rollout",
            "--config", _write_config(tmp_path),
        ],
    )
    assert invoked.exit_code == 0, invoked.output
    result = json.loads(invoked.output)
    assert result["applied"]["skill"] == "cost-audit:2"


# -- Phase-6 re-review blockers (reviews/2026-09-13_phase-6) -----------------


def test_veto_after_lineage_fork_hits_the_named_candidates_version(db, cfg):
    """Blocker B2 repro: v2 promoted from parent 1 -> v2 retired -> v3
    challenger from parent 1 -> v3 promoted. A veto naming the V2 candidate
    must not touch the live v3 champion."""
    name = "fork-skill"
    _distill(db, cfg, name=name)
    # v2 challenger from parent v1, promoted
    skills.propose_skill_challenger(db, cfg, name, "split")
    skills.run_replay(db, cfg, skill_name=name)
    cand_v2 = skills._newest_pending_candidate(db, name)["candidate_id"]
    skills.decide_promotion(db, cfg, name, cand_v2, "promoted", "v2", "gate")
    skills.retire_skill(db, name, 2, "retire v2")
    # v3 challenger from the SAME parent 1, promoted
    skills.propose_skill_challenger(db, cfg, name, "split")
    skills.run_replay(db, cfg, skill_name=name)
    cand_v3 = skills._newest_pending_candidate(db, name)["candidate_id"]
    skills.decide_promotion(db, cfg, name, cand_v3, "promoted", "v3", "gate")
    # veto naming the V2 candidate: pre-fix this resolved the LIVE v3 via
    # (name, parent_version) and killed it; with exact challenger linkage it
    # resolves v2 (already retired) and SAFELY refuses, leaving v3 untouched.
    import pytest

    from spark_mcp.errors import ToolError

    with pytest.raises(ToolError, match="not a live champion"):
        skills.approve(db, cfg, "skill", cand_v2, "rejected", "veto v2 late", "human:test")
    statuses = {
        r["version"]: r["status"]
        for r in db.conn.execute("SELECT version, status FROM skills WHERE name = ?", (name,))
    }
    assert statuses[3] == "champion"
    assert statuses[2] == "retired"  # untouched


def test_promotion_requires_full_suite_coverage(db, cfg):
    """Major M1: a cherry-picked subset (even held-out-only) cannot satisfy
    the gate."""
    name = "coverage-skill"
    _distill(db, cfg, name=name)
    skills.propose_skill_challenger(db, cfg, name, "split")
    skills.run_replay(db, cfg, skill_name=name, suite="econometrics")  # held-out only
    cand = skills._newest_pending_candidate(db, name)["candidate_id"]
    import pytest

    from spark_mcp.errors import ToolError

    with pytest.raises(ToolError, match="FULL replay-suite"):
        skills.decide_promotion(db, cfg, name, cand, "promoted", "cherry", "gate")
    # full replay afterwards satisfies coverage
    skills.run_replay(db, cfg, skill_name=name)
    out = skills.decide_promotion(db, cfg, name, cand, "promoted", "full", "gate")
    assert out["decision"] == "promoted"


def test_auto_gate_criterion_promotes_and_rejects(db, cfg):
    """decision='auto' applies the documented criterion: first promotion
    needs held-in mean >= replay_promotion_floor; a later challenger must
    beat the incumbent's held-in mean; heldout_gap over the alarm rejects."""
    name = "auto-skill"
    _distill(db, cfg, name=name)
    skills.propose_skill_challenger(db, cfg, name, "split")
    skills.run_replay(db, cfg, skill_name=name)
    cand = skills._newest_pending_candidate(db, name)["candidate_id"]
    out = skills.decide_promotion(db, cfg, name, cand, "auto", "auto", "gate")
    assert out["decision"] in ("promoted", "rejected")
    # determinism: same inputs -> same auto outcome
    if out["decision"] == "promoted":
        statuses = {
            r["version"]: r["status"]
            for r in db.conn.execute("SELECT version, status FROM skills WHERE name = ?", (name,))
        }
        assert statuses[2] == "champion"


def test_decide_and_retire_tools_over_mcp(tmp_path):
    """Blocker B1: the promotion gate and retirement are reachable from the
    product surface, not just as Python functions."""
    cfg = Config(
        storage={"db_path": f"{tmp_path}/spark.db", "artifacts_dir": f"{tmp_path}/art"}
    )
    db = open_db(cfg.db_path())
    ep = _episode(db, cfg)
    db.close()
    server = create_server(cfg, embedder=FakeEmbedder(), seeds=False)

    async def run():
        async with Client(server) as client:
            await client.call_tool(
                "distill_skill",
                {
                    "episode_ids": [ep],
                    "name": "mcp-gate",
                    "role": "quant",
                    "procedure_steps": AUDIT_STEPS,
                },
            )
            await client.call_tool(
                "propose_skill_challenger", {"skill_name": "mcp-gate", "op": "split"}
            )
            await client.call_tool("run_replay", {"skill_name": "mcp-gate"})
            cand_row = skills._newest_pending_candidate(
                open_db(cfg.db_path()), "mcp-gate"
            )
            out = await client.call_tool(
                "decide_skill_promotion",
                {
                    "skill_name": "mcp-gate",
                    "candidate_id": cand_row["candidate_id"],
                    "decision": "auto",
                    "rationale": "gate via MCP",
                },
            )
            decided = json.loads(out.content[0].text)
            assert decided["decision"] in ("promoted", "rejected")
            retired = await client.call_tool(
                "retire_skill",
                {"name": "mcp-gate", "version": 2, "rationale": "eol"},
            )
            assert json.loads(retired.content[0].text)["status"] == "retired"

    asyncio.run(run())


def _cand_with_scores(db, cfg, name, parent_version, held_in, held_out, gap):
    """Insert a pending candidate + its challenger row with hand-set replay
    evidence (auto-criterion unit pinning)."""
    skills._distill_check_episodes(db, []) if False else None  # no-op guard
    sid = skills.new_skill_id()
    cid = skills.new_candidate_id()
    db.execute(
        "INSERT INTO skills (skill_id, name, version, role, body_uri, lineage,"
        " status, credit, benchmark_history, promotion_status, created_at)"
        " VALUES (?, ?, ?, NULL, ?, ?, 'challenger', '{}', '[]', 'auto', ?)",
        (sid, name, _next_free(db, name), f"spark://skills/{name}/x",
         json.dumps({"parent_version": parent_version}), skills.now_iso()),
    )
    db.execute(
        "INSERT INTO skill_candidates (candidate_id, skill_name, parent_version,"
        " source_episode_ids, mutation_op, eval_scores, heldout_gap, verdict,"
        " challenger_skill_id) VALUES (?, ?, ?, '[]', 'split', ?, ?, 'pending', ?)",
        (cid, name, parent_version,
         json.dumps({"held_in": held_in, "held_out": held_out}), gap, sid),
    )
    return cid


def _next_free(db, name):
    row = db.conn.execute(
        "SELECT COALESCE(MAX(version),0) AS v FROM skills WHERE name = ?", (name,)
    ).fetchone()
    return int(row["v"]) + 1


def test_auto_gate_all_four_branches(db, cfg):
    """Pin every auto-criterion branch (re-review nit): first-promotion
    floor pass AND fail; incumbent beat AND regress; heldout_gap alarm."""
    floor = cfg.research.replay_promotion_floor
    alarm = cfg.research.replay_heldout_gap_alarm
    fixtures = skills.load_fixtures()
    in_ids = [f["task_id"] for f in fixtures if f.get("held_in", True)]
    out_ids = [f["task_id"] for f in fixtures if not f.get("held_in", True)]
    assert len(in_ids) >= 4 and len(out_ids) >= 2  # full-suite shape

    def mk_in(val):
        return {t: val for t in in_ids}

    def mk_out(val):
        return {t: val for t in out_ids}

    # branch 1a/1b: first promotion, mean vs floor
    for label, hin, expect in (
        ("above floor", mk_in(max(floor, 0.9)), "promoted"),
        ("below floor", mk_in(0.0), "rejected"),
    ):
        name = f"auto-{label.replace(' ', '-')}"
        cid = _cand_with_scores(db, cfg, name, 1, hin, mk_out(0.5), 0.0)
        out = skills.decide_promotion(db, cfg, name, cid, "auto", label, "gate")
        assert out["decision"] == expect, (label, out)

    # branch 2: incumbent beat vs regress (incumbent held_in mean = 0.6)
    incumbent_name = "auto-incumbent"
    inc_sid = skills.new_skill_id()
    db.execute(
        "INSERT INTO skills (skill_id, name, version, role, body_uri, lineage,"
        " status, credit, benchmark_history, promotion_status, created_at)"
        " VALUES (?, ?, 1, NULL, 'spark://skills/auto-incumbent/1', '{}',"
        " 'champion', '{}', ?, 'auto', ?)",
        (inc_sid, incumbent_name,
         json.dumps([{"eval_scores": {"held_in": {"t1": 0.6, "t2": 0.6}}}]),
         skills.now_iso()),
    )
    for label, hin, expect in (
        ("beat", mk_in(0.7), "promoted"),
        ("regress", mk_in(0.5), "rejected"),
    ):
        cid = _cand_with_scores(db, cfg, incumbent_name, 1, hin, mk_out(0.5), 0.0)
        out = skills.decide_promotion(db, cfg, incumbent_name, cid, "auto", label, "gate")
        assert out["decision"] == expect, (label, out)

    # branch 3: heldout_gap over the alarm rejects even with strong held-in
    name3 = "auto-gap-alarm"
    cid3 = _cand_with_scores(db, cfg, name3, 1, mk_in(0.9),
                             mk_out(0.0), alarm + 0.05)
    out3 = skills.decide_promotion(db, cfg, name3, cid3, "auto", "gap", "gate")
    assert out3["decision"] == "rejected"


def test_auto_gate_margin_tie_rejects(db, cfg):
    """Evolution upgrade plan §2.F3: the auto gate requires a STRICT margin —
    an exact tie with the incumbent (or exactly margin above) rejects.

    Values are exactly-representable binary fractions so ties and the
    margin boundary are hit exactly, not approached through float dust.
    """
    cfg.research.replay_promotion_margin = 0.25
    fixtures = skills.load_fixtures()
    in_ids = [f["task_id"] for f in fixtures if f.get("held_in", True)]
    out_ids = [f["task_id"] for f in fixtures if not f.get("held_in", True)]

    def mk_in(val):
        return {t: val for t in in_ids}

    def mk_out(val):
        return {t: val for t in out_ids}

    name = "auto-margin-incumbent"
    db.execute(
        "INSERT INTO skills (skill_id, name, version, role, body_uri, lineage,"
        " status, credit, benchmark_history, promotion_status, created_at)"
        " VALUES (?, ?, 1, NULL, 'spark://skills/auto-margin-incumbent/1', '{}',"
        " 'champion', '{}', ?, 'auto', ?)",
        (skills.new_skill_id(), name,
         json.dumps([{"eval_scores": {"held_in": {"t1": 0.625}}}]),
         skills.now_iso()),
    )
    for label, val, expect in (
        ("exact tie", 0.625, "rejected"),
        ("exactly margin above", 0.875, "rejected"),
        ("above margin", 1.0, "promoted"),
    ):
        cid = _cand_with_scores(db, cfg, name, 1, mk_in(val), mk_out(0.5), 0.0)
        out = skills.decide_promotion(db, cfg, name, cid, "auto", label, "gate")
        assert out["decision"] == expect, (label, out)


def test_auto_gate_first_promotion_margin(db, cfg):
    """F3 first-promotion branch: the floor needs the same strict margin."""
    floor = cfg.research.replay_promotion_floor
    assert floor == 0.5
    cfg.research.replay_promotion_margin = 0.25
    fixtures = skills.load_fixtures()
    in_ids = [f["task_id"] for f in fixtures if f.get("held_in", True)]
    out_ids = [f["task_id"] for f in fixtures if not f.get("held_in", True)]

    def mk_in(val):
        return {t: val for t in in_ids}

    def mk_out(val):
        return {t: val for t in out_ids}

    for label, val, expect in (
        ("exactly floor", 0.5, "rejected"),
        ("exactly margin above floor", 0.75, "rejected"),
        ("above margin", 0.875, "promoted"),
    ):
        name = f"auto-margin-floor-{label.replace(' ', '-')}"
        cid = _cand_with_scores(db, cfg, name, 1, mk_in(val), mk_out(0.5), 0.0)
        out = skills.decide_promotion(db, cfg, name, cid, "auto", label, "gate")
        assert out["decision"] == expect, (label, out)
