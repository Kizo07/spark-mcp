"""DAG lineage + branch points (evolution upgrade plan §2.C1, §5).

Typed helpers over hypotheses.parent_hypothesis_ids, evolution line
(branch) pointers with promotion/rollback bookkeeping, run audit records,
and structured narrated feedback rows with IS-only metrics.
"""

import json

import pytest

from spark_mcp import ledger
from spark_mcp.evolve import lineage as lineage_mod
from spark_mcp.evolve.fitness import FitnessReport

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


def _propose(db, cfg, parents=None):
    return ledger.propose_hypothesis(db, cfg, parent_hypothesis_ids=parents or [], **PROPOSE)[
        "hypothesis_id"
    ]


def _episode(db, cfg, hyp_id, seed=7):
    return ledger.register_hypothesis(db, cfg, hyp_id, {"signal": "mom"}, seed)["episode_id"]


def _fitness(episode_id, dsr=0.96):
    return {
        "episode_id": episode_id,
        "sr": 1.2,
        "dsr_prob": dsr,
        "p_value": 1.0 - dsr,
        "M_family": 3,
        "T": 250,
        "skew": 0.0,
        "kurt": 3.0,
        "heavy_tail_caveat": False,
        "degenerate": False,
        "simulated": True,
        "costs_basis": {"bps": 5.0, "slippage": 2.0, "borrow": 0.0},
    }


def _line_with_run(db, family=FAMILY):
    line = lineage_mod.create_line(db, family)
    run = lineage_mod.create_run(db, line["line_id"], seed=0)
    return line, run


def test_ancestors_nearest_first_multi_parent(db, cfg):
    a = _propose(db, cfg)
    b = _propose(db, cfg, [a])
    c = _propose(db, cfg, [a])
    d = _propose(db, cfg, [b, c])
    assert lineage_mod.ancestors(db, d) == [b, c, a]
    assert lineage_mod.ancestors(db, b) == [a]
    assert lineage_mod.ancestors(db, a) == []


def test_ancestors_unknown_raises(db, cfg):
    with pytest.raises(ValueError, match="unknown hypothesis_id"):
        lineage_mod.ancestors(db, "hyp_nope")


def test_children_ordered_and_unknown(db, cfg):
    a = _propose(db, cfg)
    kids = sorted([_propose(db, cfg, [a]) for _ in range(3)])
    assert lineage_mod.children(db, a) == kids
    assert lineage_mod.children(db, kids[0]) == []
    with pytest.raises(ValueError, match="unknown hypothesis_id"):
        lineage_mod.children(db, "hyp_nope")


def test_walk_to_root_self_first_follows_first_parent(db, cfg):
    a = _propose(db, cfg)
    b = _propose(db, cfg, [a])
    c = _propose(db, cfg, [a])
    d = _propose(db, cfg, [b, c])
    assert lineage_mod.walk_to_root(db, d) == [d, b, a]
    assert lineage_mod.walk_to_root(db, a) == [a]
    with pytest.raises(ValueError, match="unknown hypothesis_id"):
        lineage_mod.walk_to_root(db, "hyp_nope")


def _promote(db, run_id, hyp_id, ep_id, dsr):
    return lineage_mod.record_feedback(
        db,
        run_id,
        hyp_id,
        ep_id,
        ledger.get_hypothesis(db, hyp_id)["parent_hypothesis_ids"],
        {"observations": "o", "hypothesis_evaluation": "e", "new_hypothesis": "n"},
        _fitness(ep_id, dsr),
        f"promoted: test dsr {dsr}",
        "fake-llm/0.1",
        "sha",
    )


def test_sota_ancestor_argmax_over_promoted(db, cfg):
    a = _propose(db, cfg)
    b = _propose(db, cfg, [a])
    c = _propose(db, cfg, [b])
    d = _propose(db, cfg, [c])
    _, run = _line_with_run(db)
    ep_b, ep_c = _episode(db, cfg, b), _episode(db, cfg, c)
    _promote(db, run["run_id"], b, ep_b, 0.96)
    _promote(db, run["run_id"], c, ep_c, 0.97)
    sota = lineage_mod.sota_ancestor(db, d)
    assert sota == {"hypothesis_id": c, "episode_id": ep_c, "dsr_prob": 0.97}


def test_sota_ancestor_tie_breaks_earliest_episode(db, cfg):
    a = _propose(db, cfg)
    b = _propose(db, cfg, [a])
    c = _propose(db, cfg, [b])
    d = _propose(db, cfg, [c])
    _, run = _line_with_run(db)
    # episodes created in order: ep_b sorts before ep_c lexicographically only
    # if ULIDs align — pin the tie-break by comparing against sorted ids.
    ep_b, ep_c = _episode(db, cfg, b), _episode(db, cfg, c)
    _promote(db, run["run_id"], b, ep_b, 0.97)
    _promote(db, run["run_id"], c, ep_c, 0.97)
    winner = b if ep_b < ep_c else c
    sota = lineage_mod.sota_ancestor(db, d)
    assert sota is not None and sota["hypothesis_id"] == winner
    assert sota["episode_id"] == min(ep_b, ep_c)


def test_sota_ancestor_ignores_rejected_and_empty(db, cfg):
    a = _propose(db, cfg)
    b = _propose(db, cfg, [a])
    _, run = _line_with_run(db)
    ep = _episode(db, cfg, b)
    lineage_mod.record_feedback(
        db, run["run_id"], b, ep, [a], {}, _fitness(ep), "rejected: weak", None, None
    )
    assert lineage_mod.sota_ancestor(db, b) is None
    assert lineage_mod.sota_ancestor(db, a) is None
    with pytest.raises(ValueError, match="unknown hypothesis_id"):
        lineage_mod.sota_ancestor(db, "hyp_nope")


def test_create_get_active_line(db, cfg):
    hyp = _propose(db, cfg)
    ep = _episode(db, cfg, hyp)
    line = lineage_mod.create_line(db, FAMILY, hyp, ep)
    assert line["line_id"].startswith("elif_")
    assert line["status"] == "active"
    assert lineage_mod.get_line(db, line["line_id"])["head_hypothesis_id"] == hyp
    assert lineage_mod.get_line(db, "elif_nope") is None
    root = lineage_mod.create_line(db, FAMILY)
    assert root["head_hypothesis_id"] is None
    # newest active wins
    assert lineage_mod.active_line(db, FAMILY)["line_id"] == root["line_id"]
    assert lineage_mod.active_line(db, "other_family") is None
    with pytest.raises(ValueError, match="unknown hypothesis_id"):
        lineage_mod.create_line(db, FAMILY, "hyp_nope")
    with pytest.raises(ValueError, match="unknown head_episode_id"):
        lineage_mod.create_line(db, FAMILY, hyp, "ep_nope")


def test_advance_head_moves_prior_to_rollback(db, cfg):
    h1, h2 = _propose(db, cfg), _propose(db, cfg)
    e1, e2 = _episode(db, cfg, h1), _episode(db, cfg, h2)
    line = lineage_mod.create_line(db, FAMILY, h1, e1)
    updated = lineage_mod.advance_head(db, line["line_id"], h2, e2)
    assert updated["head_hypothesis_id"] == h2
    assert updated["head_episode_id"] == e2
    assert updated["rollback_hypothesis_id"] == h1
    assert updated["rollback_episode_id"] == e1
    with pytest.raises(ValueError, match="unknown line_id"):
        lineage_mod.advance_head(db, "elif_nope", h2, e2)
    with pytest.raises(ValueError, match="unknown hypothesis_id"):
        lineage_mod.advance_head(db, line["line_id"], "hyp_nope", e2)


def test_create_finish_run(db, cfg):
    line = lineage_mod.create_line(db, FAMILY)
    run = lineage_mod.create_run(db, line["line_id"], seed=42, caps={"max_rounds": 8})
    assert run["run_id"].startswith("erun_")
    assert run["status"] == "running"
    assert json.loads(run["caps_json"]) == {"max_rounds": 8}
    done = lineage_mod.finish_run(db, run["run_id"], "finished", "hyp_x")
    assert done["status"] == "finished" and done["head_after"] == "hyp_x"
    assert done["finished_at"]
    with pytest.raises(ValueError, match="unknown line_id"):
        lineage_mod.create_run(db, "elif_nope", 0)
    with pytest.raises(ValueError, match="unknown run_id"):
        lineage_mod.finish_run(db, "erun_nope", "finished", None)


def test_record_feedback_metrics_are_is_only_despite_oos_keys(db, cfg):
    """Constraint 5 at the trace edge: even an episode carrying oos_* metric
    keys yields a feedback row with no OOS content anywhere."""
    hyp = _propose(db, cfg)
    ep = _episode(db, cfg, hyp)
    db.execute(
        "UPDATE episodes SET metrics = ?, corrected_metrics = ?, costs = ? WHERE episode_id = ?",
        (
            json.dumps({"simulated": True, "oos_ic": 0.09, "oos": {"x": 1}}),
            json.dumps({"sr": 1.1, "dsr": 0.96, "pass": True, "oos_leak": 1.0}),
            json.dumps({"bps": 5.0, "slippage": 2.0, "oos_borrow": 3.0}),
            ep,
        ),
    )
    _, run = _line_with_run(db)
    row = lineage_mod.record_feedback(
        db, run["run_id"], hyp, ep, [], {"observations": "o"}, _fitness(ep),
        "promoted: test", "m", "s",
    )
    blob = row["metrics_json"] + row["fitness_json"]
    assert "oos" not in blob
    metrics = json.loads(row["metrics_json"])
    assert metrics["simulated"] is True
    assert metrics["corrected"]["dsr"] == 0.96
    assert metrics["costs"]["bps"] == 5.0


def test_record_feedback_validation_and_provenance(db, cfg):
    hyp = _propose(db, cfg)
    ep = _episode(db, cfg, hyp)
    _, run = _line_with_run(db)
    episode = ledger.get_episode(db, ep)
    report = FitnessReport(**_fitness(ep))
    row = lineage_mod.record_feedback(
        db, run["run_id"], hyp, ep, [], {"observations": "o"}, report,
        "rejected: weak margins", None, None,
    )
    assert row["as_of"] == episode["as_of"]
    assert row["data_through"] == episode["data_through"]
    assert json.loads(row["fitness_json"])["dsr_prob"] == 0.96
    assert row["model_id"] is None and row["prompt_sha"] is None
    with pytest.raises(ValueError, match="must start with"):
        lineage_mod.record_feedback(
            db, run["run_id"], hyp, ep, [], {}, _fitness(ep), "maybe", None, None
        )
    with pytest.raises(ValueError, match="unknown run_id"):
        lineage_mod.record_feedback(
            db, "erun_nope", hyp, ep, [], {}, _fitness(ep), "rejected: x", None, None
        )
    with pytest.raises(ValueError, match="must start with"):
        lineage_mod.record_feedback(
            db, run["run_id"], hyp, ep, [], {}, _fitness(ep), None, None, None
        )
