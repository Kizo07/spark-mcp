"""Explore/exploit parent selectors (evolution upgrade plan §2.C2, §5).

Exploit-head default, SOTA-jump threshold boundary behavior, jump-back
branch points, and determinism (policies consume no rng draws).
"""

import random

import pytest

from spark_mcp import ledger
from spark_mcp.errors import ToolError
from spark_mcp.evolve import lineage as lineage_mod
from spark_mcp.evolve import selectors as selectors_mod

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


def test_exploit_head_default(db, cfg):
    hyp = _propose(db, cfg)
    ep = _episode(db, cfg, hyp)
    line = lineage_mod.create_line(db, FAMILY, hyp, ep)
    sel = selectors_mod.select_parent(db, cfg, line["line_id"], random.Random(0))
    assert sel.policy == "exploit_head"
    assert (sel.hypothesis_id, sel.episode_id) == (hyp, ep)
    assert sel.reason


def test_empty_line_proposes_root_on_this_line(db, cfg):
    line = lineage_mod.create_line(db, FAMILY)
    sel = selectors_mod.select_parent(db, cfg, line["line_id"], random.Random(0))
    assert sel.policy == "exploit_head"
    assert sel.hypothesis_id is None and sel.episode_id is None


def test_unknown_line_raises(db, cfg):
    with pytest.raises(ValueError, match="unknown line_id"):
        selectors_mod.select_parent(db, cfg, "elif_nope", random.Random(0))


def _chain(db, cfg, n):
    hyps = []
    parents = []
    for _ in range(n):
        hyp = _propose(db, cfg, parents)
        hyps.append(hyp)
        parents = [hyp]
    return hyps


def _history(db, cfg, line_id, outcomes):
    """One run on the line with one verdict per outcome (oldest first)."""
    run = lineage_mod.create_run(db, line_id, seed=0)
    hyps = _chain(db, cfg, len(outcomes))
    for hyp, promoted in zip(hyps, outcomes, strict=True):
        ep = _episode(db, cfg, hyp)
        lineage_mod.record_feedback(
            db,
            run["run_id"],
            hyp,
            ep,
            [],
            {},
            {"dsr_prob": 0.96, "sr": 1.0},
            "promoted: test" if promoted else "rejected: test",
            None,
            None,
        )
    return hyps


def test_sota_jump_below_threshold(db, cfg):
    cfg.research.evolution_sota_jump_window = 4
    cfg.research.evolution_sota_jump_threshold = 0.25
    line = lineage_mod.create_line(db, FAMILY)
    _history(db, cfg, line["line_id"], [True, False, False, False])  # 1/4 = 0.25
    sel = selectors_mod.select_parent(db, cfg, line["line_id"], random.Random(0))
    # exactly at the threshold still exploits (strict `<` jumps)
    assert sel.policy == "exploit_head"
    _history(db, cfg, line["line_id"], [False])  # window now 0/4... see below
    sel = selectors_mod.select_parent(db, cfg, line["line_id"], random.Random(0))
    assert sel.policy == "sota_jump_new_root"
    assert sel.hypothesis_id is None
    # the selector never writes: no new line appeared
    assert lineage_mod.active_line(db, FAMILY)["line_id"] == line["line_id"]


def test_sota_jump_window_caps_history(db, cfg):
    cfg.research.evolution_sota_jump_window = 2
    cfg.research.evolution_sota_jump_threshold = 0.5
    line = lineage_mod.create_line(db, FAMILY)
    # old wins fall outside the window: last two are rejections → jump
    _history(db, cfg, line["line_id"], [True, True, False, False])
    sel = selectors_mod.select_parent(db, cfg, line["line_id"], random.Random(0))
    assert sel.policy == "sota_jump_new_root"
    assert selectors_mod.recent_hit_rate(db, line["line_id"], 2) == 0.0
    assert selectors_mod.recent_hit_rate(db, line["line_id"], 200) == 0.5


def test_sota_jump_disabled_window_zero_and_no_history(db, cfg):
    cfg.research.evolution_sota_jump_window = 0
    line = lineage_mod.create_line(db, FAMILY)
    _history(db, cfg, line["line_id"], [False, False])
    sel = selectors_mod.select_parent(db, cfg, line["line_id"], random.Random(0))
    assert sel.policy == "exploit_head"
    fresh = lineage_mod.create_line(db, "fresh_family")
    assert selectors_mod.recent_hit_rate(db, fresh["line_id"], 6) is None


def test_select_parent_consumes_no_rng_and_is_deterministic(db, cfg):
    hyp = _propose(db, cfg)
    ep = _episode(db, cfg, hyp)
    line = lineage_mod.create_line(db, FAMILY, hyp, ep)
    rng = random.Random(1234)
    before = rng.getstate()
    first = selectors_mod.select_parent(db, cfg, line["line_id"], rng)
    assert rng.getstate() == before
    second = selectors_mod.select_parent(db, cfg, line["line_id"], random.Random(999))
    assert first == second


def _promoted_chain(db, cfg, n):
    line = lineage_mod.create_line(db, FAMILY)
    run = lineage_mod.create_run(db, line["line_id"], seed=0)
    hyps = _chain(db, cfg, n)
    for hyp in hyps:
        ep = _episode(db, cfg, hyp)
        lineage_mod.record_feedback(
            db, run["run_id"], hyp, ep, [], {}, {"dsr_prob": 0.96, "sr": 1.0},
            "promoted: test", None, None,
        )
    head_ep = ledger.get_episode(db, _episode(db, cfg, hyps[-1]))
    lineage_mod.advance_head(db, line["line_id"], hyps[-1], head_ep["episode_id"])
    return lineage_mod.get_line(db, line["line_id"]), hyps


def test_jump_back_lands_on_nth_last_passing_ancestor(db, cfg):
    line, hyps = _promoted_chain(db, cfg, 4)  # A → B → C → H
    sel = selectors_mod.jump_back(db, line["line_id"], n_sota=2)
    assert sel.policy == "jump_back"
    assert sel.hypothesis_id == hyps[1]  # B: two SOTAs back from H
    assert sel.episode_id
    sel1 = selectors_mod.jump_back(db, line["line_id"], n_sota=1)
    assert sel1.hypothesis_id == hyps[2]  # C


def test_jump_back_clamps_and_refuses(db, cfg):
    line, hyps = _promoted_chain(db, cfg, 3)
    sel = selectors_mod.jump_back(db, line["line_id"], n_sota=9)
    assert sel.hypothesis_id == hyps[0]
    assert "clamped" in sel.reason
    with pytest.raises(ValueError, match="n_sota must be >= 1"):
        selectors_mod.jump_back(db, line["line_id"], n_sota=0)
    with pytest.raises(ValueError, match="unknown line_id"):
        selectors_mod.jump_back(db, "elif_nope")
    empty = lineage_mod.create_line(db, FAMILY)
    with pytest.raises(ToolError, match="no head"):
        selectors_mod.jump_back(db, empty["line_id"])
    # head with no passing ancestor
    lonely_hyp = _propose(db, cfg)
    lonely = lineage_mod.create_line(db, FAMILY, lonely_hyp, _episode(db, cfg, lonely_hyp))
    with pytest.raises(ToolError, match="no passing ancestor"):
        selectors_mod.jump_back(db, lonely["line_id"])


def test_jump_back_skips_rejected_ancestors(db, cfg):
    line = lineage_mod.create_line(db, FAMILY)
    run = lineage_mod.create_run(db, line["line_id"], seed=0)
    a, b, c = _chain(db, cfg, 3)
    for hyp, ok in ((a, True), (b, False), (c, True)):
        ep = _episode(db, cfg, hyp)
        lineage_mod.record_feedback(
            db, run["run_id"], hyp, ep, [], {}, {"dsr_prob": 0.96, "sr": 1.0},
            "promoted: test" if ok else "rejected: test", None, None,
        )
    lineage_mod.advance_head(db, line["line_id"], c, _episode(db, cfg, c))
    sel = selectors_mod.jump_back(db, line["line_id"], n_sota=1)
    assert sel.hypothesis_id == a  # B rejected → skipped
