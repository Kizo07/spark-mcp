"""Capped ancestor rendering (evolution upgrade plan §2.E2, §5).

Ancestors-only, newest-last "Trial 1..n" order, explicit truncation policy
(oldest non-SOTA dropped first), budget respected, audit cache roundtrip.
"""

import hashlib
import json

import pytest

from spark_mcp import ledger
from spark_mcp.evolve import lineage as lineage_mod
from spark_mcp.evolve import render as render_mod

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


def _propose(db, cfg, parents=None, **overrides):
    return ledger.propose_hypothesis(
        db, cfg, parent_hypothesis_ids=parents or [], **{**PROPOSE, **overrides}
    )["hypothesis_id"]


def _episode(db, cfg, hyp_id):
    return ledger.register_hypothesis(db, cfg, hyp_id, {"signal": "mom"}, 7)["episode_id"]


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


def _feedback(db, run_id, hyp_id, ep_id, verdict="promoted: test", dsr=0.96):
    return lineage_mod.record_feedback(
        db,
        run_id,
        hyp_id,
        ep_id,
        ledger.get_hypothesis(db, hyp_id)["parent_hypothesis_ids"],
        {"observations": "o", "hypothesis_evaluation": "e", "new_hypothesis": "n"},
        _fitness(ep_id, dsr),
        verdict,
        "fake-llm/0.1",
        "sha",
    )


def _chain(db, cfg, n=3):
    """Linear chain h1 -> h2 -> ... with a run; returns (ids, run_id)."""
    ids = [_propose(db, cfg)]
    for _ in range(n - 1):
        ids.append(_propose(db, cfg, [ids[-1]]))
    line = lineage_mod.create_line(db, FAMILY)
    run = lineage_mod.create_run(db, line["line_id"], seed=0)
    return ids, run["run_id"]


def test_renders_ancestors_only_newest_last(db, cfg):
    ids, run_id = _chain(db, cfg, 3)
    h1, h2, h3 = ids
    sibling = _propose(db, cfg, [h1])  # same forest, not an ancestor of h3
    for hyp in (h1, h2, h3, sibling):
        _feedback(db, run_id, hyp, _episode(db, cfg, hyp))
    render = render_mod.render_ancestors(db, cfg, h3)
    assert render.n_ancestors == 2
    assert render.truncated_from_n == 2
    assert "Trial 1" in render.text and "Trial 2" in render.text
    # Oldest first (Trial 1 = root h1), newest last — never RD's reversed bug.
    assert render.text.index("Trial 1") < render.text.index("Trial 2")
    assert render.text.index(h1) < render.text.index(h2)
    assert h3 not in render.text  # the node itself is not its own ancestor
    assert sibling not in render.text  # siblings excluded
    assert render.sha == hashlib.sha1(render.text.encode()).hexdigest()


def test_render_carries_concise_fields_and_metrics_not_logs(db, cfg):
    ids, run_id = _chain(db, cfg, 2)
    h1, h2 = ids
    _feedback(db, run_id, h1, _episode(db, cfg, h1), dsr=0.9711)
    render = render_mod.render_ancestors(db, cfg, h2)
    assert "12-1 momentum" in render.text
    assert "dsr_prob=0.9711" in render.text
    assert "sr=1.2000" in render.text
    assert "M=3 T=250" in render.text
    assert "stdout" not in render.text.lower()


def test_render_without_feedback_marks_no_trial(db, cfg):
    ids, _ = _chain(db, cfg, 2)
    render = render_mod.render_ancestors(db, cfg, ids[1])
    assert render.n_ancestors == 1
    assert "no trial yet" in render.text


def test_root_renders_empty_and_unknown_raises(db, cfg):
    (root,), _ = _chain(db, cfg, 1)
    render = render_mod.render_ancestors(db, cfg, root)
    assert render.n_ancestors == 0
    assert render.truncated_from_n == 0
    assert render_mod.empty_render().n_ancestors == 0
    with pytest.raises(ValueError, match="unknown hypothesis_id"):
        render_mod.render_ancestors(db, cfg, "hyp_nope")


def test_truncation_drops_oldest_non_sota_first(db, cfg):
    ids, run_id = _chain(db, cfg, 4)
    h1, h2, h3, h4 = ids
    # h1 promoted (SOTA), h2/h3 rejected, h4 is the query node.
    _feedback(db, run_id, h1, _episode(db, cfg, h1), "promoted: sota")
    _feedback(db, run_id, h2, _episode(db, cfg, h2), "rejected: weak")
    _feedback(db, run_id, h3, _episode(db, cfg, h3), "rejected: weak")
    full = render_mod.render_ancestors(db, cfg, h4)
    assert full.n_ancestors == 3
    # Just under the full cost: exactly one trial must drop (each trial is
    # tens of words, so dropping one always fits).
    budget = render_mod.estimate_tokens(full.text) - 5
    cut = render_mod.render_ancestors(db, cfg, h4, max_tokens=budget)
    assert cut.truncated_from_n == 3
    assert cut.n_ancestors == 2
    assert f"truncated from {3}" in cut.text
    # Oldest non-SOTA (h2) dropped; the SOTA root (h1) survives.
    assert h2 not in cut.text
    assert h1 in cut.text and h3 in cut.text
    assert render_mod.estimate_tokens(cut.text) <= budget


def test_truncation_keeps_one_trial_and_respects_budget(db, cfg):
    ids, run_id = _chain(db, cfg, 3)
    for hyp in ids:
        _feedback(db, run_id, hyp, _episode(db, cfg, hyp), "rejected: weak")
    cut = render_mod.render_ancestors(db, cfg, ids[-1], max_tokens=10)
    assert cut.n_ancestors == 1  # never truncated to zero trials
    assert cut.truncated_from_n == 2
    # A single trial can exceed a tiny budget; the policy pins count, not bytes.
    tiny = render_mod.render_ancestors(db, cfg, ids[-1], max_tokens=1_000_000)
    assert tiny.n_ancestors == 2
    assert tiny.truncated_from_n == 2
    assert "truncated from" not in tiny.text


def test_config_default_budget_applies(db, cfg):
    ids, run_id = _chain(db, cfg, 2)
    _feedback(db, run_id, ids[0], _episode(db, cfg, ids[0]))
    cfg.research.evolution_render_max_tokens = 10**9
    big = render_mod.render_ancestors(db, cfg, ids[1])
    assert big.n_ancestors == 1
    assert render_mod.estimate_tokens("a  b\nc") == 3
    assert render_mod.estimate_tokens("") == 0


def test_render_deterministic_and_cached_for_audit(db, cfg):
    ids, run_id = _chain(db, cfg, 2)
    _feedback(db, run_id, ids[0], _episode(db, cfg, ids[0]))
    first = render_mod.render_ancestors(db, cfg, ids[1])
    second = render_mod.render_ancestors(db, cfg, ids[1])
    assert first == second
    key = render_mod.cache_render(db, first, "fake-llm/0.1")
    assert render_mod.cached_render(db, first.sha, "fake-llm/0.1") == first.text
    assert render_mod.cached_render(db, first.sha, "other-model/9") is None
    assert render_mod.cached_render(db, "nope", "fake-llm/0.1") is None
    row = db.conn.execute(
        "SELECT purpose, subject_key FROM generated_text_cache WHERE cache_key = ?",
        (key,),
    ).fetchone()
    assert row["purpose"] == render_mod.CACHE_PURPOSE
    assert row["subject_key"] == first.sha


def test_oos_keys_never_reach_the_render(db, cfg):
    """Feedback rows are IS-only, so renders built on them cannot carry OOS."""
    ids, run_id = _chain(db, cfg, 2)
    ep = _episode(db, cfg, ids[0])
    metrics = {"sr": 1.0, "simulated": True, "oos_ic": 0.99, "oos_sharpe": 9.9}
    db.execute(
        "UPDATE episodes SET metrics = ? WHERE episode_id = ?",
        (json.dumps(metrics), ep),
    )
    _feedback(db, run_id, ids[0], ep)
    render = render_mod.render_ancestors(db, cfg, ids[1])
    assert "oos" not in render.text.lower()
