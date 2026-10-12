"""Computed promotion decisions (evolution upgrade plan §2.E3, §5).

`decide` promotes on in-sample significance + strict numeric margins only.
Held-out evidence rides along as an advisory `oos_report` that never feeds
the boolean. Human veto/restore runs through approve(target_type='evolution').
"""

import json

import pytest
from test_evolution_fitness import FAMILY, PROPOSE, make_spec

from spark_mcp import ledger, skills
from spark_mcp.errors import ToolError
from spark_mcp.evolve import promote as promote_mod
from spark_mcp.gates import service as gates
from spark_mcp.ids import new_approval_id, new_evolution_line_id, new_trial_id
from spark_mcp.providers.simulated import SimulatedProvider
from spark_mcp.timeutil import now_iso


def _registered_episode(db, cfg):
    hyp = ledger.propose_hypothesis(db, cfg, **PROPOSE)
    reg = ledger.register_hypothesis(db, cfg, hyp["hypothesis_id"], {"signal": "mom"}, 7)
    return reg["episode_id"], hyp["hypothesis_id"]


def _set_corrected(
    db,
    episode_id,
    *,
    sr=1.2,
    dsr=0.97,
    M=1,
    T=252,
    skew=0.1,
    kurt=3.2,
    passed=True,
    metrics_extra=None,
):
    """Hand-set gate outputs for precise margin pinning (deterministic)."""
    corrected = {
        "sr": sr,
        "psr": dsr,
        "dsr": dsr,
        "p_value": 1.0 - dsr,
        "M_family": M,
        "min_T": 30,
        "alpha": 0.05,
        "T": T,
        "skew": skew,
        "kurt": kurt,
        "heavy_tail_caveat": False,
        "degenerate": False,
        "pass": passed,
    }
    metrics = {"sr": sr, "T": T, "simulated": True, "provider": "simulated"}
    metrics.update(metrics_extra or {})
    db.execute(
        "UPDATE episodes SET corrected_metrics = ?, metrics = ?, costs = ?"
        " WHERE episode_id = ?",
        (
            json.dumps(corrected, sort_keys=True),
            json.dumps(metrics, sort_keys=True),
            json.dumps({"bps": 5, "slippage": 2, "borrow": 0}),
            episode_id,
        ),
    )


def test_first_promotion_floor(db, cfg):
    floor = cfg.research.evolution_first_promotion_floor
    assert floor == 0.95
    ep_hi, _ = _registered_episode(db, cfg)
    _set_corrected(db, ep_hi, dsr=0.97, sr=1.2)
    decision = promote_mod.decide(db, cfg, ep_hi, None)
    assert decision.promote is True
    assert decision.incumbent is None
    assert "first promotion" in decision.reason

    ep_lo, _ = _registered_episode(db, cfg)
    _set_corrected(db, ep_lo, dsr=0.90, sr=1.2)
    decision = promote_mod.decide(db, cfg, ep_lo, None)
    assert decision.promote is False
    assert "floor" in decision.reason


def test_margin_pass_and_fail_default_margins(db, cfg):
    dsr_req = cfg.research.evolution_dsr_margin
    sr_req = cfg.research.evolution_sr_margin
    assert (dsr_req, sr_req) == (0.02, 0.0)

    ep_inc, _ = _registered_episode(db, cfg)
    _set_corrected(db, ep_inc, dsr=0.96, sr=1.0)

    # Clear pass on both margins.
    ep_win, _ = _registered_episode(db, cfg)
    _set_corrected(db, ep_win, dsr=0.99, sr=1.1)
    won = promote_mod.decide(db, cfg, ep_win, ep_inc)
    assert won.promote is True
    assert won.dsr_margin == pytest.approx(0.03)
    assert won.sr_margin == pytest.approx(0.1)

    # Identical scores: margin exactly 0.0, ties never promote.
    ep_same, _ = _registered_episode(db, cfg)
    _set_corrected(db, ep_same, dsr=0.96, sr=1.0)
    same = promote_mod.decide(db, cfg, ep_same, ep_inc)
    assert same.promote is False
    assert "DSR margin" in same.reason

    # DSR margin clears but SR ties exactly: still rejected (strict on both).
    ep_sr_tie, _ = _registered_episode(db, cfg)
    _set_corrected(db, ep_sr_tie, dsr=0.99, sr=1.0)
    sr_tied = promote_mod.decide(db, cfg, ep_sr_tie, ep_inc)
    assert sr_tied.promote is False
    assert "SR margin" in sr_tied.reason

    # Below the DSR margin: rejected.
    ep_weak, _ = _registered_episode(db, cfg)
    _set_corrected(db, ep_weak, dsr=0.965, sr=1.5)
    weak = promote_mod.decide(db, cfg, ep_weak, ep_inc)
    assert weak.promote is False


def test_margin_boundary_is_strict(db, cfg):
    """A margin exactly AT the requirement still rejects (strict `>`).

    Uses exactly-representable binary fractions so the boundary is hit
    exactly, not approached through decimal float dust.
    """
    cfg.research.evolution_dsr_margin = 0.25
    cfg.research.evolution_sr_margin = 0.5
    ep_inc, _ = _registered_episode(db, cfg)
    _set_corrected(db, ep_inc, dsr=0.5, sr=1.0)

    ep_edge, _ = _registered_episode(db, cfg)
    _set_corrected(db, ep_edge, dsr=0.75, sr=1.5)  # margins exactly 0.25 / 0.5
    edge = promote_mod.decide(db, cfg, ep_edge, ep_inc)
    assert edge.dsr_margin == 0.25 and edge.sr_margin == 0.5
    assert edge.promote is False
    assert "DSR margin" in edge.reason

    ep_over, _ = _registered_episode(db, cfg)
    _set_corrected(db, ep_over, dsr=0.875, sr=2.0)
    over = promote_mod.decide(db, cfg, ep_over, ep_inc)
    assert over.promote is True


def test_dsr_fail_rejects_despite_margins(db, cfg):
    ep_inc, _ = _registered_episode(db, cfg)
    _set_corrected(db, ep_inc, dsr=0.96, sr=1.0)
    ep_bad, _ = _registered_episode(db, cfg)
    _set_corrected(db, ep_bad, dsr=0.99, sr=2.0, passed=False)
    decision = promote_mod.decide(db, cfg, ep_bad, ep_inc)
    assert decision.promote is False
    assert "significance gate" in decision.reason


def test_c6_blockers_reject_despite_pass(db, cfg):
    # A passing verdict with non-empty promotion_blockers must not promote:
    # flip a hypothesis to causal-without-DAG after the fact.
    ep_inc, _ = _registered_episode(db, cfg)
    _set_corrected(db, ep_inc, dsr=0.96, sr=1.0)
    ep_ch, hyp_ch = _registered_episode(db, cfg)
    _set_corrected(db, ep_ch, dsr=0.99, sr=2.0)
    db.execute(
        "UPDATE hypotheses SET hypothesis_type = 'causal' WHERE hypothesis_id = ?",
        (hyp_ch,),
    )
    decision = promote_mod.decide(db, cfg, ep_ch, ep_inc)
    assert decision.promote is False
    assert "C6" in decision.reason

    # First promotion is blocked the same way.
    first = promote_mod.decide(db, cfg, ep_ch, None)
    assert first.promote is False
    assert "C6" in first.reason


def test_degenerate_inputs_reject(db, cfg):
    ep_inc, _ = _registered_episode(db, cfg)
    _set_corrected(db, ep_inc, dsr=0.96, sr=1.0)
    ep_deg, _ = _registered_episode(db, cfg)
    _set_corrected(db, ep_deg, dsr=0.99, sr=2.0, passed=False)
    db.execute(
        "UPDATE episodes SET corrected_metrics = json_set(corrected_metrics,"
        " '$.degenerate', json('true')) WHERE episode_id = ?",
        (ep_deg,),
    )
    report = promote_mod.decide(db, cfg, ep_deg, ep_inc)
    assert report.challenger.degenerate is True
    assert report.promote is False


def test_stored_gate_time_M_used(db, cfg):
    """Trials recorded between gate and decide must not shift the comparison:
    fitness carries each row's stored gate-time M_family."""
    ep_inc, _ = _registered_episode(db, cfg)
    _set_corrected(db, ep_inc, dsr=0.96, sr=1.0, M=1)
    ep_ch, _ = _registered_episode(db, cfg)
    _set_corrected(db, ep_ch, dsr=0.99, sr=1.1, M=2)
    # Five more counted trials land after gating.
    for i in range(5):
        db.execute(
            "INSERT INTO trials (trial_id, episode_id, idem_key, family, source,"
            " status, provider, created_at) VALUES (?, ?, ?, ?, 'spark',"
            " 'completed', 'simulated', ?)",
            (new_trial_id(), ep_ch, f"late-{i}", FAMILY, now_iso()),
        )
    assert gates.M_family(db, FAMILY) == 5
    decision = promote_mod.decide(db, cfg, ep_ch, ep_inc)
    assert decision.challenger.M_family == 2
    assert decision.incumbent is not None and decision.incumbent.M_family == 1
    assert decision.promote is True


def test_oos_report_advisory_never_feeds_boolean(db, cfg):
    ep_inc, _ = _registered_episode(db, cfg)
    _set_corrected(db, ep_inc, dsr=0.96, sr=1.0)
    # No OOS producer today: the advisory is {available: False}.
    ep_plain, _ = _registered_episode(db, cfg)
    _set_corrected(db, ep_plain, dsr=0.965, sr=1.5)
    plain = promote_mod.decide(db, cfg, ep_plain, ep_inc)
    assert plain.oos_report == {"available": False}
    assert plain.promote is False

    # Even stellar OOS values cannot flip the boolean either way.
    ep_oos, _ = _registered_episode(db, cfg)
    _set_corrected(
        db, ep_oos, dsr=0.965, sr=1.5, metrics_extra={"oos_ic": 0.99, "oos_sr": 9.9}
    )
    oosy = promote_mod.decide(db, cfg, ep_oos, ep_inc)
    assert oosy.oos_report["available"] is True
    assert oosy.oos_report["metrics"] == {"oos_ic": 0.99, "oos_sr": 9.9}
    assert oosy.promote is False
    assert (oosy.dsr_margin, oosy.sr_margin) == (plain.dsr_margin, plain.sr_margin)

    ep_win, _ = _registered_episode(db, cfg)
    _set_corrected(
        db, ep_win, dsr=0.99, sr=1.1, metrics_extra={"oos_ic": -0.5}
    )
    won = promote_mod.decide(db, cfg, ep_win, ep_inc)
    assert won.oos_report["available"] is True
    assert won.promote is True  # terrible OOS still cannot veto the IS decision


def test_decide_propagates_fitness_refusal(db, cfg):
    ep_inc, _ = _registered_episode(db, cfg)
    _set_corrected(db, ep_inc, dsr=0.96, sr=1.0)
    ep_raw, _ = _registered_episode(db, cfg)  # never gated
    with pytest.raises(ToolError, match="no gated corrected_metrics"):
        promote_mod.decide(db, cfg, ep_raw, ep_inc)
    with pytest.raises(ToolError, match="no gated corrected_metrics"):
        promote_mod.decide(db, cfg, ep_raw, None)


def test_decide_is_deterministic(db, cfg):
    ep_inc, _ = _registered_episode(db, cfg)
    _set_corrected(db, ep_inc, dsr=0.96, sr=1.0)
    ep_ch, _ = _registered_episode(db, cfg)
    _set_corrected(db, ep_ch, dsr=0.99, sr=1.1)
    assert promote_mod.decide(db, cfg, ep_ch, ep_inc) == promote_mod.decide(
        db, cfg, ep_ch, ep_inc
    )


def test_decide_end_to_end_over_simulated_provider(db, cfg):
    """Smoke path: real backtest_run + significance_gate episodes decide
    deterministically (outcome depends on simulated draw, not asserted)."""
    provider = SimulatedProvider(cfg.artifacts_dir())
    episodes = []
    for seed in (11, 12):
        hyp = ledger.propose_hypothesis(db, cfg, **PROPOSE)
        reg = ledger.register_hypothesis(db, cfg, hyp["hypothesis_id"], {"s": 1}, seed)
        gates.backtest_run(
            db, cfg, provider, reg["episode_id"], make_spec(hyp["hypothesis_id"], seed)
        )
        gates.significance_gate(db, cfg, reg["episode_id"])
        episodes.append(reg["episode_id"])
    first = promote_mod.decide(db, cfg, episodes[0], None)
    assert isinstance(first.promote, bool) and first.reason
    assert first.oos_report == {"available": False}
    second = promote_mod.decide(db, cfg, episodes[1], episodes[0])
    assert second.incumbent is not None
    assert promote_mod.decide(db, cfg, episodes[1], episodes[0]) == second


# -- human veto over evolution lines -------------------------------------------


def _insert_line(db, family=FAMILY, head=("hyp_h", "ep_h"), rollback=("hyp_r", "ep_r")):
    line_id = new_evolution_line_id()
    now = now_iso()
    db.execute(
        "INSERT INTO evolution_lines (line_id, family, head_hypothesis_id,"
        " head_episode_id, rollback_hypothesis_id, rollback_episode_id, status,"
        " created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?)",
        (line_id, family, head[0], head[1], rollback[0], rollback[1], now, now),
    )
    db.execute(
        "INSERT INTO approvals (approval_id, target_type, target_id, decision,"
        " rationale, decided_by, created_at) VALUES (?, 'evolution', ?, 'approved',"
        " ?, 'gate', ?)",
        (new_approval_id(), line_id, "gate notify", now),
    )
    return line_id


def _line(db, line_id):
    return db.conn.execute(
        "SELECT * FROM evolution_lines WHERE line_id = ?", (line_id,)
    ).fetchone()


def test_evolution_veto_restores_rollback_head(db, cfg):
    line_id = _insert_line(db)
    veto = skills.approve(
        db, cfg, "evolution", line_id, "rejected", "overfit head", "human:desk"
    )
    assert veto["duplicated"] is False
    assert veto["applied"]["vetoed_head"] == {"hypothesis_id": "hyp_h", "episode_id": "ep_h"}
    assert veto["applied"]["restored_head_hypothesis_id"] == "hyp_r"
    assert veto["applied"]["restored_head_episode_id"] == "ep_r"
    line = _line(db, line_id)
    assert line["head_hypothesis_id"] == "hyp_r"
    assert line["head_episode_id"] == "ep_r"
    assert line["rollback_hypothesis_id"] is None
    assert line["rollback_episode_id"] is None
    assert line["status"] == "active"


def test_evolution_approve_ratifies_head(db, cfg):
    line_id = _insert_line(db)
    ok = skills.approve(
        db, cfg, "evolution", line_id, "approved", "head looks right", "human:desk"
    )
    assert ok["applied"] == {
        "ratified_head_hypothesis_id": "hyp_h",
        "ratified_head_episode_id": "ep_h",
    }
    line = _line(db, line_id)
    assert line["head_hypothesis_id"] == "hyp_h"  # untouched


def test_evolution_veto_refusals(db, cfg):
    with pytest.raises(ToolError, match="unknown evolution line"):
        skills.approve(db, cfg, "evolution", "elif_missing", "rejected", "x", "human:a")
    line_id = _insert_line(db, rollback=(None, None))
    with pytest.raises(ToolError, match="no rollback head"):
        skills.approve(db, cfg, "evolution", line_id, "rejected", "x", "human:a")


def test_evolution_approve_idempotent_per_actor_class(db, cfg):
    line_id = _insert_line(db)
    first = skills.approve(db, cfg, "evolution", line_id, "approved", "ok", "human:a")
    second = skills.approve(db, cfg, "evolution", line_id, "approved", "ok", "human:b")
    assert second["duplicated"] is True
    assert second["approval_id"] == first["approval_id"]
