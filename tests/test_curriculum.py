"""Minimal curriculum (plan §18.1 Phase 7): exercise from a real gap, explicit
deferral paths, champion replay score via the deterministic run_replay scorer."""

from __future__ import annotations

import json

from spark_mcp.consolidate import report_path
from spark_mcp.curriculum import generate_exercise
from spark_mcp.evolve.gepa import fixture_score
from spark_mcp.skills import load_fixtures

CHAMPION_STEPS = [
    "State the universe and point-in-time membership rules before ranking.",
    "Apply explicit transaction cost and slippage assumptions.",
    "Report Sharpe paired with a deflated significance check.",
    "Check momentum turnover and capacity before trusting the result.",
]


def _write_report(cfg, gaps):
    report = {
        "report_id": "cons_test_report",
        "schema": "spark.consolidation_report/v1",
        "gap_count": len(gaps),
        "gaps": gaps,
    }
    path = report_path(cfg, report["report_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report), encoding="utf-8")
    return report["report_id"]


def _gap(kind="empty_family", family="quant", severity="high"):
    return {
        "gap_id": f"gap_{kind}_{family}",
        "kind": kind,
        "family": family,
        "severity": severity,
        "evidence": {"occupied_cells": 0},
        "suggested_action": f"occupy the {family!r} region",
    }


def _make_champion(db):
    db.execute(
        "INSERT INTO skills (skill_id, name, version, body_uri, lineage, status,"
        " created_at) VALUES ('skill_champ', 'audit', 1, 'spark://skills/audit/1',"
        " ?, 'champion', '2026-01-01T00:00:00Z')",
        (json.dumps({"procedure_steps": CHAMPION_STEPS}),),
    )


def test_exercise_from_real_gap_with_champion_score(db, cfg):
    _make_champion(db)
    report_id = _write_report(cfg, [_gap()])
    exercise = generate_exercise(db, cfg, gap_report_id=report_id, seed=7)
    assert exercise["exercise_id"].startswith("exercise_")
    assert exercise["gap_report_id"] == report_id
    assert exercise["source_gap_id"] == "gap_empty_family_quant"
    assert exercise["family"] == "quant"
    assert "occupy the" in exercise["task_prompt"]
    assert exercise["evaluation"]["fixture_task_id"] == "replay_momentum_cost_audit"
    assert exercise["evaluation"]["rubric"]  # rubric items carried from the fixture
    # champion scored with the SAME deterministic scorer run_replay uses
    fixture = next(
        f for f in load_fixtures() if f["task_id"] == "replay_momentum_cost_audit"
    )
    assert exercise["champion_score"] == fixture_score(CHAMPION_STEPS, fixture)
    assert 0.0 <= exercise["champion_score"] <= 1.0
    assert exercise["champion"]["name"] == "audit"
    # trajectory event records the exercise
    row = db.conn.execute(
        "SELECT payload FROM trajectory_events WHERE run_id = ?", (exercise["exercise_id"],)
    ).fetchone()
    payload = json.loads(row["payload"])
    assert payload["tool"] == "generate_exercise"
    assert payload["gap_report_id"] == report_id
    assert payload["champion_score"] == exercise["champion_score"]


def test_exercise_highest_severity_gap_preferred(db, cfg):
    _make_champion(db)
    report_id = _write_report(
        cfg,
        [
            _gap(kind="family_posterior_weak", family="quant", severity="info"),
            _gap(kind="empty_family", family="causal", severity="high"),
        ],
    )
    exercise = generate_exercise(db, cfg, gap_report_id=report_id)
    assert exercise["family"] == "causal"  # high outranks info
    assert exercise["evaluation"]["fixture_task_id"] == "replay_causal_dag"


def test_deferral_when_report_has_no_gaps(db, cfg):
    report_id = _write_report(cfg, [])
    out = generate_exercise(db, cfg, gap_report_id=report_id)
    assert out["deferred"] is True
    assert "no open gaps" in out["reason"]
    assert out["gap_report_id"] == report_id


def test_deferral_when_gap_family_has_no_fixture(db, cfg):
    report_id = _write_report(cfg, [_gap(family="portfolio")])
    out = generate_exercise(db, cfg, gap_report_id=report_id)
    assert out["deferred"] is True
    assert "no replay fixture maps family 'portfolio'" in out["reason"]


def test_exercise_without_champion_scores_none(db, cfg):
    report_id = _write_report(cfg, [_gap()])
    exercise = generate_exercise(db, cfg, gap_report_id=report_id)
    assert exercise["champion"] is None
    assert exercise["champion_score"] is None
    assert "no champion skill" in exercise["champion_note"]


def test_generate_exercise_runs_consolidation_when_no_report_id(db, cfg):
    _make_champion(db)
    # no gap_report_id -> a fresh consolidation pass supplies the gaps
    exercise = generate_exercise(db, cfg)
    assert exercise.get("deferred") is not True
    assert exercise["family"]  # a real gap from the fresh report
    row = db.conn.execute("SELECT COUNT(*) AS n FROM consolidation_runs").fetchone()
    assert row["n"] == 1  # the fresh pass was persisted


def test_seeded_choice_is_stable(db, cfg):
    _make_champion(db)
    report_id = _write_report(
        cfg,
        [
            _gap(kind="empty_family", family="causal", severity="high"),
            _gap(kind="empty_family", family="quant", severity="high"),
        ],
    )
    a = generate_exercise(db, cfg, gap_report_id=report_id, seed=3)
    b = generate_exercise(db, cfg, gap_report_id=report_id, seed=3)
    assert a["family"] == b["family"]
