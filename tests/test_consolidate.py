"""Consolidation runtime (plan §13, §12.3, §14): pinned decay math, SLO
listing, gap kinds, partial report on exhausted budget (warn), fail_closed
refusal before any write, persistence (consolidation_runs + artifact)."""

from __future__ import annotations

import json
import math

import pytest

from spark_mcp import consolidate as consolidate_mod
from spark_mcp.consolidate import load_report, report_path
from spark_mcp.errors import ToolError
from spark_mcp.llm import FakeLLM
from spark_mcp.memory import promote_belief

PINNED = "2026-01-02T00:00:00Z"  # exactly 180 days after VALIDATED
VALIDATED = "2025-07-06T00:00:00Z"


def _belief(db, cfg, confidence=0.9, as_of=VALIDATED):
    return promote_belief(db, cfg, [], "earnings drift persists post-announcement",
                          scope=None, rationale="consolidate test", confidence=confidence,
                          as_of=as_of)


def test_decay_formula_pinned_value_written(db, cfg):
    _belief(db, cfg, confidence=0.9, as_of=VALIDATED)
    report = consolidate_mod.consolidate(db, cfg, as_of=PINNED)
    row = db.conn.execute(
        "SELECT confidence, decay FROM beliefs LIMIT 1"
    ).fetchone()
    expected = math.exp(-180.0 / 180.0) * 1.0  # e^-1, regime_factor 1.0 v1
    assert row["decay"] == pytest.approx(expected, rel=1e-12)
    assert row["decay"] == pytest.approx(math.exp(-1.0), rel=1e-12)
    assert row["confidence"] == 0.9  # raw confidence untouched
    assert report["decay"]["updated"] == 1
    assert report["decay"]["regime_factor"] == 1.0
    # effective confidence per §12.3
    assert row["confidence"] * row["decay"] == pytest.approx(0.9 * math.exp(-1.0))


def test_decay_skips_null_last_validated(db, cfg):
    _belief(db, cfg)
    db.execute("UPDATE beliefs SET last_validated = NULL")
    report = consolidate_mod.consolidate(db, cfg, as_of=PINNED)
    assert report["decay"]["skipped_no_last_validated"] == 1
    assert report["decay"]["updated"] == 0
    assert db.conn.execute("SELECT decay FROM beliefs").fetchone()["decay"] == 1.0


def test_slo_lists_stale_belief(db, cfg):
    _belief(db, cfg)  # validated 180 days before PINNED > 90-day SLO
    report = consolidate_mod.consolidate(db, cfg, as_of=PINNED)
    slo = [g for g in report["gaps"] if g["kind"] == "belief_stale"]
    assert len(slo) == 1
    assert slo[0]["evidence"]["days_unvalidated"] == pytest.approx(180.0)
    assert slo[0]["severity"] == "medium"


def test_fresh_belief_not_slo_listed(db, cfg):
    _belief(db, cfg, as_of="2026-01-01T00:00:00Z")  # 1 day old at PINNED
    report = consolidate_mod.consolidate(db, cfg, as_of=PINNED)
    assert not [g for g in report["gaps"] if g["kind"] == "belief_stale"]


def test_archive_gap_kinds_emitted(db, cfg):
    db.execute(
        "INSERT INTO exploration_state (family, killed, posterior, trials, successes,"
        " updated_at) VALUES ('quant', 1, '{\"alpha\": 1.0, \"beta\": 9.0}', 10, 1, ?)",
        (PINNED,),
    )
    db.execute(
        "INSERT INTO exploration_state (family, killed, posterior, trials, successes,"
        " updated_at) VALUES ('causal', 0, '{\"alpha\": 2.0, \"beta\": 23.0}', 25, 2, ?)",
        (PINNED,),
    )
    report = consolidate_mod.consolidate(db, cfg, as_of=PINNED)
    kinds = {g["kind"] for g in report["gaps"]}
    assert "empty_family" in kinds  # fresh db: every bin family is empty
    assert "killed_family" in kinds
    killed = next(g for g in report["gaps"] if g["kind"] == "killed_family")
    assert killed["family"] == "quant"
    assert killed["severity"] == "high"
    weak = next(g for g in report["gaps"] if g["kind"] == "family_posterior_weak")
    assert weak["family"] == "causal"
    assert weak["severity"] == "info"
    assert weak["evidence"]["mean"] == pytest.approx(2.0 / 25.0)
    # all 8 configured families listed as empty
    empties = {g["family"] for g in report["gaps"] if g["kind"] == "empty_family"}
    assert empties == set(cfg.research.qd_family_bins)


def test_scope_restricts_family_gaps(db, cfg):
    report = consolidate_mod.consolidate(db, cfg, scope="quant", as_of=PINNED)
    family_gaps = [g for g in report["gaps"] if g.get("family")]
    assert family_gaps
    assert all(g["family"] == "quant" for g in family_gaps)
    row = db.conn.execute(
        "SELECT scope FROM consolidation_runs WHERE run_id = ?", (report["report_id"],)
    ).fetchone()
    assert row["scope"] == "quant"


def test_predictor_error_outlier_gap(db, cfg):
    # hypothesis predicted_ic 0.05, episode realized oos_ic 0.60 -> error 0.55
    db.execute(
        "INSERT INTO hypotheses (hypothesis_id, research_family, hypothesis_type,"
        " mechanism, expected_direction, horizon, scope, falsifier, predicted_ic,"
        " predicted_sigma, universe, as_of, data_through, created_at)"
        " VALUES ('hyp_pred', 'quant', 'predictive', 'm', 'positive', '20d',"
        " 'cross_section', 'f', 0.05, 0.01, 'toy', ?, ?, ?)",
        (PINNED, PINNED, PINNED),
    )
    db.execute(
        "INSERT INTO episodes (episode_id, hypothesis_id, research_family, as_of,"
        " data_through, created_at, metrics)"
        " VALUES ('ep_pred', 'hyp_pred', 'quant', ?, ?, ?,"
        " '{\"oos_ic\": 0.60}')",
        (PINNED, PINNED, PINNED),
    )
    report = consolidate_mod.consolidate(db, cfg, as_of=PINNED)
    outlier = [g for g in report["gaps"] if g["kind"] == "predictor_error_outlier"]
    assert len(outlier) == 1
    assert outlier[0]["family"] == "quant"
    assert outlier[0]["evidence"]["mean_abs_predicted_minus_realized"] == pytest.approx(0.55)


def test_skill_pipeline_stale_candidate_gap(db, cfg):
    db.execute(
        "INSERT INTO skills (skill_id, name, version, body_uri, lineage, status,"
        " created_at) VALUES ('skill_1', 'audit', 1, 'spark://skills/audit/1',"
        " '{\"parent_version\": 1, \"procedure_steps\": [\"step\"]}',"
        " 'challenger', '2025-01-01T00:00:00Z')"
    )
    db.execute(
        "INSERT INTO skill_candidates (candidate_id, skill_name, parent_version,"
        " source_episode_ids, mutation_op, verdict)"
        " VALUES ('cand_1', 'audit', 1, '[]', 'split', 'pending')"
    )
    report = consolidate_mod.consolidate(db, cfg, as_of=PINNED)  # >14d stale
    stale = [g for g in report["gaps"] if g["kind"] == "skill_candidate_stale"]
    assert len(stale) == 1
    assert stale[0]["evidence"]["candidate_id"] == "cand_1"
    assert stale[0]["evidence"]["age_days"] > 14


def test_fresh_candidate_not_listed(db, cfg):
    db.execute(
        "INSERT INTO skills (skill_id, name, version, body_uri, lineage, status,"
        " created_at) VALUES ('skill_1', 'audit', 1, 'spark://skills/audit/1',"
        " '{\"parent_version\": 1, \"procedure_steps\": [\"step\"]}',"
        " 'challenger', '2026-01-01T00:00:00Z')"
    )
    db.execute(
        "INSERT INTO skill_candidates (candidate_id, skill_name, parent_version,"
        " source_episode_ids, mutation_op, verdict)"
        " VALUES ('cand_1', 'audit', 1, '[]', 'split', 'pending')"
    )
    report = consolidate_mod.consolidate(db, cfg, as_of=PINNED)
    assert not [g for g in report["gaps"] if g["kind"] == "skill_candidate_stale"]


def test_runs_row_and_artifact_persisted(db, cfg):
    report = consolidate_mod.consolidate(db, cfg, as_of=PINNED)
    row = db.conn.execute(
        "SELECT * FROM consolidation_runs WHERE run_id = ?", (report["report_id"],)
    ).fetchone()
    assert row is not None
    assert row["report_id"] == report["report_id"]
    assert row["report_uri"] == f"spark://consolidation/{report['report_id']}"
    assert row["llm_spend_usd"] == 0.0
    assert row["started_at"] <= row["finished_at"]
    path = report_path(cfg, report["report_id"])
    assert path.exists()
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["report_id"] == report["report_id"]
    assert on_disk["gap_count"] == report["gap_count"]
    assert on_disk["gaps"] == report["gaps"]
    assert report["artifact_path"] == str(path)


def test_trajectory_event_actor_consolidator(db, cfg):
    report = consolidate_mod.consolidate(db, cfg, as_of=PINNED)
    row = db.conn.execute(
        "SELECT actor, kind, payload FROM trajectory_events WHERE run_id = ?",
        (report["report_id"],),
    ).fetchone()
    assert row["actor"] == "consolidator"
    payload = json.loads(row["payload"])
    assert payload["tool"] == "consolidate"
    assert payload["gap_count"] == report["gap_count"]


def test_warn_policy_partial_report_on_exhausted_budget(db, cfg):
    cfg.budgets.per_consolidation_usd = 0.03  # one 0.02 call fits, then stop
    llm = FakeLLM(replies={"sharpen the suggested action": "Do the thing now."})
    report = consolidate_mod.consolidate(db, cfg, as_of=PINNED, llm=llm)
    assert report["gap_count"] >= 2  # 8 empty-family gaps at minimum
    assert report["budget"]["budget_exhausted"] is True
    assert report["budget"]["completed_llm_calls"] == 1
    assert report["budget"]["planned_llm_calls"] == report["gap_count"]
    assert report["budget"]["llm_spend_usd"] == pytest.approx(0.02)
    annotated = [g for g in report["gaps"] if "llm_note" in g]
    unannotated = [g for g in report["gaps"] if "llm_note" not in g]
    assert len(annotated) == 1  # partial: the pass stopped mid-way
    assert unannotated
    assert annotated[0]["llm_note"] == "Do the thing now."
    # spend row attributed to the run
    spend = db.conn.execute(
        "SELECT usd, source, run_id, estimated FROM spend_ledger WHERE source = 'consolidate'"
    ).fetchone()
    assert spend["run_id"] == report["report_id"]
    assert spend["usd"] == pytest.approx(0.02)
    assert spend["estimated"] == 1


def test_full_budget_no_llm_calls(db, cfg):
    cfg.budgets.per_consolidation_usd = 0.0
    llm = FakeLLM(replies={"sharpen": "x"})
    report = consolidate_mod.consolidate(db, cfg, as_of=PINNED, llm=llm)
    assert report["budget"]["budget_exhausted"] is True
    assert report["budget"]["completed_llm_calls"] == 0
    assert report["budget"]["llm_spend_usd"] == 0.0


def test_fail_closed_raises_before_any_write(db, cfg):
    cfg.budgets.policy = "fail_closed"
    cfg.budgets.per_consolidation_usd = 0.01  # less than one 0.02 call
    _belief(db, cfg)  # would get a decay write on a normal run
    llm = FakeLLM(replies={"sharpen": "x"})
    with pytest.raises(ToolError, match="fail_closed"):
        consolidate_mod.consolidate(db, cfg, as_of=PINNED, llm=llm)
    # NOTHING was written: no run row, no artifact, decay untouched, no spend
    assert db.conn.execute("SELECT COUNT(*) AS n FROM consolidation_runs").fetchone()["n"] == 0
    assert not report_path(cfg, "cons").parent.exists() or not list(
        report_path(cfg, "cons").parent.glob("*.json")
    )
    assert db.conn.execute("SELECT decay FROM beliefs").fetchone()["decay"] == 1.0
    assert db.conn.execute(
        "SELECT COUNT(*) AS n FROM spend_ledger WHERE source = 'consolidate'"
    ).fetchone()["n"] == 0
    assert db.conn.execute(
        "SELECT COUNT(*) AS n FROM trajectory_events WHERE actor = 'consolidator'"
    ).fetchone()["n"] == 0


def test_fail_closed_with_no_llm_runs_normally(db, cfg):
    cfg.budgets.policy = "fail_closed"
    report = consolidate_mod.consolidate(db, cfg, as_of=PINNED, llm=None)
    assert report["budget"]["llm_spend_usd"] == 0.0


def test_since_filter_makes_run_incremental(db, cfg):
    _belief(db, cfg, as_of=VALIDATED)
    since = "2025-12-01T00:00:00Z"  # belief's last activity predates this
    report = consolidate_mod.consolidate(db, cfg, as_of=PINNED, since=since)
    assert report["decay"]["updated"] == 0
    assert report["decay"]["skipped_since_filter"] == 1
    assert db.conn.execute("SELECT decay FROM beliefs").fetchone()["decay"] == 1.0


def test_load_report_unknown_id_refused(cfg):
    with pytest.raises(ToolError, match="unknown gap_report_id"):
        load_report(cfg, "cons_nope")


def test_reports_are_immutable_for_pit_consumption(db, cfg):
    report = consolidate_mod.consolidate(db, cfg, as_of=PINNED)
    reloaded = load_report(cfg, report["report_id"])
    assert reloaded == {k: report[k] for k in reloaded}
    # artifact content is stable across reloads (immutability = PIT safety)
    again = load_report(cfg, report["report_id"])
    assert again == reloaded


def test_gap_entry_shape(db, cfg):
    report = consolidate_mod.consolidate(db, cfg, as_of=PINNED)
    for gap in report["gaps"]:
        assert set(gap) >= {
            "gap_id", "kind", "family", "severity", "evidence", "suggested_action",
        }
        assert gap["kind"] in {
            "belief_stale", "empty_family", "killed_family",
            "predictor_error_outlier", "family_posterior_weak", "skill_candidate_stale",
            "fixture_suite_small",
        }
        assert gap["severity"] in {"high", "medium", "info"}
        assert gap["suggested_action"]


def test_fixture_growth_gap_quiet_at_default_floor(db, cfg):
    """F4: the seeded suite (6 fixtures) meets the default floor (6) — no gap."""
    from spark_mcp.skills import load_fixtures

    assert len(load_fixtures()) == 6
    assert cfg.research.replay_min_fixtures == 6
    report = consolidate_mod.consolidate(db, cfg, as_of=PINNED)
    assert not [g for g in report["gaps"] if g["kind"] == "fixture_suite_small"]


def test_fixture_growth_gap_fires_below_floor(db, cfg):
    """F4: raising the floor surfaces fixture growth as an open gap — alarm
    only (the report still completes; the gate never reads this)."""
    cfg.research.replay_min_fixtures = 7
    report = consolidate_mod.consolidate(db, cfg, as_of=PINNED)
    gaps = [g for g in report["gaps"] if g["kind"] == "fixture_suite_small"]
    assert len(gaps) == 1
    assert gaps[0]["severity"] == "medium"
    assert gaps[0]["evidence"] == {"fixture_count": 6, "min_fixtures": 7}
    assert gaps[0]["suggested_action"]
    assert report["report_id"]
