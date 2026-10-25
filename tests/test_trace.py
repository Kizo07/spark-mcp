"""Typed read-only trace (P1-5): ordering, truncation, empty/unknown,
SOTA pointer over stored gate verdicts, and read-only purity.

Minimal rows are inserted directly (explicit `created_at` values make
ordering deterministic despite 1-second timestamp resolution), plus
one ledger-helper integration case. Fixtures: `db`/`cfg` from
tests/conftest.py.
"""

import json

import pytest

from spark_mcp import ledger
from spark_mcp.errors import ToolError
from spark_mcp.trace import (
    CONCISE_LIMIT,
    ExperimentView,
    FeedbackView,
    HypothesisView,
    sota_pointer,
    trace_for_hypothesis,
)

TS = [f"2026-01-0{i}T00:00:00Z" for i in range(1, 8)]  # TS[0] < TS[1] < ...


def _hyp(db, hid, created_at, mechanism="momentum mechanism", falsifier="IC <= 0"):
    db.execute(
        "INSERT INTO hypotheses (hypothesis_id, parent_hypothesis_ids, research_family,"
        " hypothesis_type, mechanism, expected_direction, horizon, scope, falsifier,"
        " predicted_ic, predicted_sigma, universe, as_of, data_through, status, created_at)"
        " VALUES (?, '[]', 'toy', 'predictive', ?, 'positive', '21d', 'xsec', ?,"
        " 0.03, 0.01, 'toy', ?, ?, 'proposed', ?)",
        (hid, mechanism, falsifier, created_at, created_at, created_at),
    )
    return hid


def _ep(db, eid, hid, created_at, corrected=None, failure_reason=None, status="registered"):
    db.execute(
        "INSERT INTO episodes (episode_id, hypothesis_id, run_kind, research_family,"
        " corrected_metrics, failure_reason, status, as_of, data_through, created_at)"
        " VALUES (?, ?, 'primary', 'toy', ?, ?, ?, ?, ?, ?)",
        (eid, hid, json.dumps(corrected) if corrected is not None else None,
         failure_reason, status, created_at, created_at, created_at),
    )
    return eid


def _run(db):
    """A real evolution_lines/evolution_runs pair (FK target for feedback)."""
    db.execute(
        "INSERT INTO evolution_lines (line_id, family, status, created_at, updated_at)"
        " VALUES ('elif_test', 'toy', 'active', ?, ?)", (TS[0], TS[0]),
    )
    db.execute(
        "INSERT INTO evolution_runs (run_id, line_id, seed, caps_json, status, started_at)"
        " VALUES ('erun_test', 'elif_test', 0, '{}', 'running', ?)", (TS[0],),
    )
    return "erun_test"


def _fb(db, fid, hid, eid, created_at, observations="obs", decision="promoted: ok"):
    db.execute(
        "INSERT INTO evolution_feedback (feedback_id, run_id, hypothesis_id, episode_id,"
        " parent_ids_json, observations, hypothesis_evaluation, new_hypothesis,"
        " metrics_json, fitness_json, decision, as_of, data_through, created_at)"
        " VALUES (?, 'erun_test', ?, ?, '[]', ?, 'eval', 'next', '{}', '{}', ?, ?, ?, ?)",
        (fid, hid, eid, observations, decision, created_at, created_at, created_at),
    )
    return fid


def _counts(db):
    return {
        t: db.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        for t in ("hypotheses", "episodes", "trials", "evolution_feedback")
    }


def test_trace_orders_by_created_at_append_only(db):
    _hyp(db, "hyp_1", TS[2])
    _run(db)
    # Inserted out of chronological order on purpose.
    _ep(db, "ep_late", "hyp_1", TS[4])
    _ep(db, "ep_early", "hyp_1", TS[1])
    _fb(db, "efb_mid", "hyp_1", "ep_early", TS[3])

    trace = trace_for_hypothesis(db, "hyp_1")
    assert [e.created_at for e in trace] == sorted(e.created_at for e in trace)
    assert [e.kind for e in trace] == ["experiment", "hypothesis", "feedback", "experiment"]
    assert isinstance(trace[0], ExperimentView) and trace[0].episode_id == "ep_early"
    assert isinstance(trace[1], HypothesisView)
    assert isinstance(trace[2], FeedbackView) and trace[2].episode_id == "ep_early"


def test_concise_fields_truncate_deterministically(db):
    long_mech = "m" * 500
    long_fail = "f" * 400
    _hyp(db, "hyp_long", TS[0], mechanism=long_mech, falsifier=long_fail)
    _run(db)
    _ep(db, "ep_long", "hyp_long", TS[1], failure_reason="r" * 500)
    _fb(db, "efb_long", "hyp_long", "ep_long", TS[2], observations="o" * 500)

    trace = trace_for_hypothesis(db, "hyp_long")
    hyp, exp, fb = trace
    assert hyp.concise_mechanism == long_mech[:CONCISE_LIMIT]
    assert hyp.concise_falsifier == long_fail[:CONCISE_LIMIT]
    assert exp.concise_failure_reason == "r" * CONCISE_LIMIT
    assert fb.concise_observations == "o" * CONCISE_LIMIT
    for view in trace:
        for name in type(view).model_fields:
            if name.startswith("concise_"):
                assert len(getattr(view, name)) <= CONCISE_LIMIT
    # Deterministic: a second read is byte-identical.
    assert trace_for_hypothesis(db, "hyp_long") == trace


def test_direct_construction_also_truncates(db):
    view = HypothesisView(hypothesis_id="hyp_x", concise_mechanism="x" * 999)
    assert view.concise_mechanism == "x" * CONCISE_LIMIT


def test_lone_hypothesis_empty_experiment_feedback_and_sota_none(db):
    _hyp(db, "hyp_solo", TS[0])
    trace = trace_for_hypothesis(db, "hyp_solo")
    assert len(trace) == 1 and isinstance(trace[0], HypothesisView)
    assert [e for e in trace if isinstance(e, ExperimentView)] == []
    assert [e for e in trace if isinstance(e, FeedbackView)] == []
    assert sota_pointer(db) is None


def test_unknown_hypothesis_raises_tool_error(db):
    with pytest.raises(ToolError, match="unknown hypothesis_id"):
        trace_for_hypothesis(db, "hyp_nope")


def test_sota_pointer_picks_best_gated_only(db):
    _hyp(db, "hyp_sota", TS[0])
    _ep(db, "ep_ungated", "hyp_sota", TS[1])  # no corrected_metrics
    _ep(db, "ep_failed_gate", "hyp_sota", TS[2],
        corrected={"pass": False, "dsr": 0.99, "sr": 2.0, "p_value": 0.01})
    _ep(db, "ep_mid", "hyp_sota", TS[3],
        corrected={"pass": True, "dsr": 0.96, "sr": 1.1, "p_value": 0.04})
    _ep(db, "ep_best", "hyp_sota", TS[4],
        corrected={"pass": True, "dsr": 0.99, "sr": 1.3, "p_value": 0.01})

    sota = sota_pointer(db)
    assert sota is not None
    assert sota["episode_id"] == "ep_best"
    assert sota["hypothesis_id"] == "hyp_sota"
    assert sota["dsr_prob"] == pytest.approx(0.99)


def test_sota_pointer_tie_breaks_earliest_episode_id(db):
    _hyp(db, "hyp_tie", TS[0])
    _ep(db, "ep_bbb", "hyp_tie", TS[1], corrected={"pass": True, "dsr": 0.97, "sr": 1.0})
    _ep(db, "ep_aaa", "hyp_tie", TS[2], corrected={"pass": True, "dsr": 0.97, "sr": 9.9})
    sota = sota_pointer(db)
    assert sota is not None and sota["episode_id"] == "ep_aaa"  # earliest id wins ties


def test_experiment_view_carries_trials_and_gate_fields(db):
    _hyp(db, "hyp_tr", TS[0])
    _ep(db, "ep_tr", "hyp_tr", TS[1],
        corrected={"pass": True, "dsr": 0.95, "sr": 1.2, "p_value": 0.02})
    db.execute(
        "INSERT INTO trials (trial_id, episode_id, idem_key, family, source, status,"
        " provider, created_at) VALUES ('trial_2', 'ep_tr', 'k2', 'toy', 'spark',"
        " 'completed', 'simulated', ?)", (TS[3],),
    )
    db.execute(
        "INSERT INTO trials (trial_id, episode_id, idem_key, family, source, status,"
        " provider, created_at) VALUES ('trial_1', 'ep_tr', 'k1', 'toy', 'spark',"
        " 'completed', 'simulated', ?)", (TS[2],),
    )
    (exp,) = [e for e in trace_for_hypothesis(db, "hyp_tr") if isinstance(e, ExperimentView)]
    assert exp.trial_ids == ["trial_1", "trial_2"]  # created_at order, not insert order
    assert exp.trial_count == 2 and exp.gate_pass is True
    assert exp.dsr_prob == pytest.approx(0.95) and exp.sr == pytest.approx(1.2)


def test_trace_and_sota_are_read_only(db):
    _hyp(db, "hyp_ro", TS[0])
    _run(db)
    _ep(db, "ep_ro", "hyp_ro", TS[1], corrected={"pass": True, "dsr": 0.9, "sr": 1.0})
    _fb(db, "efb_ro", "hyp_ro", "ep_ro", TS[2])
    before = _counts(db)
    trace_for_hypothesis(db, "hyp_ro")
    sota_pointer(db)
    assert _counts(db) == before


def test_ledger_helper_rows_trace_integration(db, cfg):
    hyp = ledger.propose_hypothesis(
        db, cfg, research_family="toy", hypothesis_type="predictive",
        mechanism="ledger-built mechanism", expected_direction="positive",
        horizon="21d", scope="xsec", falsifier="IC <= 0",
        predicted_ic=0.03, predicted_sigma=0.01, universe="toy",
    )["hypothesis_id"]
    ep = ledger.register_hypothesis(db, cfg, hyp, {"signal": "mom"}, seed=1)["episode_id"]
    trace = trace_for_hypothesis(db, hyp)
    assert isinstance(trace[0], HypothesisView)
    assert trace[0].concise_mechanism == "ledger-built mechanism"
    exps = [e for e in trace if isinstance(e, ExperimentView)]
    assert [e.episode_id for e in exps] == [ep] and exps[0].gate_pass is None
