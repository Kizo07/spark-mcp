"""SME-lite analogy engine (plan §6.1 tool 4): systematicity formula cases,
threshold refusal, budget-cap accounting, falsifiable translation in both LLM
(FakeLLM) and degraded paths, PIT visibility, idempotency."""

from __future__ import annotations

import json

import pytest

from spark_mcp import graph
from spark_mcp.analogy import (
    ANALOGY_TRIALS_FAMILY,
    AnalogyBudgetExceeded,
    AnalogyMappingRefused,
    find_analogies,
    systematicity,
)
from spark_mcp.errors import ToolError
from spark_mcp.llm import FakeLLM

AS_OF = "2024-06-01T00:00:00Z"

# A: two compounds overlapping the target exactly -> pure relations.
# B: one shared compound, disjoint other compounds, no head collision.
# D: head-collision compound only (regime_switching vs regime_detection).
# E: shared compound + a head-collision side compound.
# F: bare tags only (attribute-only overlap).
NODES = {
    "A": (["regime_detection", "volatility_scaling"], "quant"),
    "B": (["regime_detection", "trend_following"], "quant"),
    "D": (["regime_switching", "trend_following"], "quant"),
    "E": (["regime_detection", "regime_switching"], "quant"),
    "F": (["momentum", "trend"], "quant"),
}
TARGET = ["regime_detection", "volatility_scaling"]


def _build(db):
    ids = {}
    for key, (tags, family) in NODES.items():
        node = graph.add_node(
            db, type="technique", label=f"method {key}", mech_tags=tags,
            family=family, as_of=AS_OF,
        )
        ids[key] = node["node_id"]
    return ids


# ---------------------------------------------------------------------------
# systematicity formula


def test_systematicity_pure_relations_scores_one():
    out = systematicity(TARGET, NODES["A"][0])
    assert out["score"] == 1.0
    assert len(out["mapped_relations"]) == 2
    rel = next(r for r in out["mapped_relations"] if r["relation"] == "regime_detection")
    assert rel["subject"] == "regime" and rel["object"] == "detection"
    assert out["attribute_matches"] == []
    assert out["inconsistent_relations"] == []


def test_systematicity_single_relation_without_penalty_scores_one():
    out = systematicity(TARGET, NODES["B"][0])
    assert out["score"] == 1.0  # one mapped relation, no I, no A
    assert [r["relation"] for r in out["mapped_relations"]] == ["regime_detection"]


def test_systematicity_attribute_only_scores_zero():
    out = systematicity(["momentum", "trend"], NODES["F"][0])
    assert out["score"] == 0.0
    assert out["mapped_relations"] == []
    assert set(out["attribute_matches"]) == {"momentum", "trend"}


def test_systematicity_mixed_relation_and_attribute():
    out = systematicity(["regime_detection", "momentum"], ["regime_detection", "momentum"])
    assert out["score"] == pytest.approx(1.0 / 1.5)  # M=1, A=1


def test_systematicity_inconsistency_penalty():
    # E shares regime_detection but adds regime_switching: same subject
    # ('regime'), disagreeing relation -> I=1, M=1 -> 1/2.
    out = systematicity(TARGET, NODES["E"][0])
    assert out["score"] == 0.5
    assert out["inconsistent_relations"] == ["regime_switching"]


def test_systematicity_head_collision_without_mapping_scores_zero():
    out = systematicity(TARGET, NODES["D"][0])
    assert out["score"] == 0.0
    # both unshared compounds claim subject 'regime' with disagreeing
    # relations -> the disagreement is symmetric (documented rule)
    assert out["inconsistent_relations"] == ["regime_detection", "regime_switching"]


def test_systematicity_no_overlap_is_zero():
    assert systematicity(["alpha_beta"], ["gamma_delta"])["score"] == 0.0
    assert systematicity([], [])["score"] == 0.0


# ---------------------------------------------------------------------------
# find_analogies


def test_find_analogies_returns_mappings_with_degraded_translation(db, cfg):
    _build(db)
    out = find_analogies(db, cfg, "regime-aware sizing of trend signals", TARGET, as_of=AS_OF)
    assert out["mappings"], "expected at least one passing mapping"
    by_label = {m["label"]: m for m in out["mappings"]}
    assert by_label["method A"]["systematicity"] == 1.0
    assert out["refusals"] >= 1  # D/F scored 0 -> refused and counted
    for m in out["mappings"]:
        assert 0.5 <= m["systematicity"] <= 1.0
        assert m["falsifiable_translation"].startswith("If this analogy holds, then")
        assert "should co-move" in m["falsifiable_translation"]  # degraded template
        assert m["translation_basis"] == "degraded_template"
        assert m["shared_predicates"]
    # budget accounting: one analogy trial consumed
    assert out["budget"]["analogy_trials"] == 1
    row = db.conn.execute(
        "SELECT analogy_trials FROM exploration_state WHERE family = ?",
        (ANALOGY_TRIALS_FAMILY,),
    ).fetchone()
    assert int(row["analogy_trials"]) == 1
    # trajectory event recorded
    event = db.conn.execute(
        "SELECT payload FROM trajectory_events WHERE kind = 'tool_call'"
        " AND payload LIKE '%find_analogies%'"
    ).fetchone()
    assert json.loads(event["payload"])["tool"] == "find_analogies"
    # the sentinel row must not leak into family posteriors
    from spark_mcp.archive import family_posteriors

    assert ANALOGY_TRIALS_FAMILY not in family_posteriors(db)


def test_find_analogies_llm_translation_generated_and_cached(db, cfg):
    ids = _build(db)
    llm = FakeLLM(
        replies={
            "falsifiable with market data": "If this analogy holds, then the volatility"
            " regime should exhibit detectable co-movement in sizing outcomes."
        }
    )
    out = find_analogies(
        db, cfg, "regime-aware sizing of trend signals", TARGET, as_of=AS_OF, llm=llm
    )
    top = out["mappings"][0]
    assert top["translation_basis"] == "llm"
    assert top["falsifiable_translation"].startswith("If this analogy holds, then")
    calls_after_first = len(llm.calls)
    # second call with the same problem class: cache hit, no new completion
    out2 = find_analogies(
        db, cfg, "regime-aware sizing of trend signals", TARGET, as_of=AS_OF, llm=llm
    )
    assert len(llm.calls) == calls_after_first
    assert out2["mappings"][0]["translation_basis"] == "llm_cached"
    assert out2["mappings"][0]["falsifiable_translation"] == top["falsifiable_translation"]
    cached = db.conn.execute("SELECT COUNT(*) AS n FROM generated_text_cache").fetchone()
    assert cached["n"] >= 1
    assert ids["A"]  # nodes built


def test_find_analogies_llm_failure_degrades_to_template(db, cfg):
    _build(db)

    class BrokenLLM:
        def complete(self, prompt, *, json_schema=None):
            raise RuntimeError("endpoint down")

        def model_id(self):
            return "broken/0.1"

    out = find_analogies(
        db, cfg, "regime-aware sizing of trend signals", TARGET,
        as_of=AS_OF, llm=BrokenLLM(),
    )
    assert out["mappings"][0]["translation_basis"] == "degraded_template"


def test_find_analogies_refuses_below_threshold(db, cfg):
    _build(db)
    with pytest.raises(AnalogyMappingRefused, match="systematicity"):
        find_analogies(db, cfg, "unrelated problem", ["lottery_demand"], as_of=AS_OF)
    # refusal happens before any budget consumption
    row = db.conn.execute(
        "SELECT COUNT(*) AS n FROM exploration_state WHERE family = ?",
        (ANALOGY_TRIALS_FAMILY,),
    ).fetchone()
    assert row["n"] == 0


def test_find_analogies_budget_cap_refuses_with_named_error(db, cfg):
    _build(db)
    db.execute(
        "INSERT INTO exploration_state (family, trials, successes, updated_at)"
        " VALUES ('quant', 10, 4, ?)",
        (AS_OF,),
    )
    db.execute(
        "INSERT INTO exploration_state (family, analogy_trials, updated_at)"
        " VALUES (?, 2, ?)",
        (ANALOGY_TRIALS_FAMILY, AS_OF),
    )
    # cap = 0.20 * 10 = 2, already at 2 -> refused
    with pytest.raises(AnalogyBudgetExceeded, match="beating the base family hit rate"):
        find_analogies(db, cfg, "regime-aware sizing of trend signals", TARGET, as_of=AS_OF)
    # no increment on the refused call
    row = db.conn.execute(
        "SELECT analogy_trials FROM exploration_state WHERE family = ?",
        (ANALOGY_TRIALS_FAMILY,),
    ).fetchone()
    assert int(row["analogy_trials"]) == 2


def test_find_analogies_budget_under_cap_increments(db, cfg):
    _build(db)
    db.execute(
        "INSERT INTO exploration_state (family, trials, successes, updated_at)"
        " VALUES ('quant', 20, 8, ?)",
        (AS_OF,),
    )  # cap = 4
    for expected in (1, 2):
        out = find_analogies(
            db, cfg, "regime-aware sizing of trend signals", TARGET, as_of=AS_OF
        )
        assert out["budget"]["analogy_trials"] == expected
        assert out["budget"]["cap"] == pytest.approx(4.0)
    row = db.conn.execute(
        "SELECT analogy_trials FROM exploration_state WHERE family = ?",
        (ANALOGY_TRIALS_FAMILY,),
    ).fetchone()
    assert int(row["analogy_trials"]) == 2


def test_find_analogies_zero_family_trials_cap_does_not_bind(db, cfg):
    _build(db)  # no exploration_state rows at all
    out = find_analogies(db, cfg, "regime-aware sizing of trend signals", TARGET, as_of=AS_OF)
    assert out["budget"]["family_trials"] == 0
    assert out["budget"]["cap"] == 0.0
    assert out["mappings"]


def test_find_analogies_idempotent_on_client_id(db, cfg):
    _build(db)
    first = find_analogies(
        db, cfg, "regime-aware sizing of trend signals", TARGET,
        as_of=AS_OF, client_id="analogy-idem-1",
    )
    trials = db.conn.execute(
        "SELECT analogy_trials FROM exploration_state WHERE family = ?",
        (ANALOGY_TRIALS_FAMILY,),
    ).fetchone()["analogy_trials"]
    second = find_analogies(
        db, cfg, "regime-aware sizing of trend signals", TARGET,
        as_of=AS_OF, client_id="analogy-idem-1",
    )
    assert second == first
    assert db.conn.execute(
        "SELECT analogy_trials FROM exploration_state WHERE family = ?",
        (ANALOGY_TRIALS_FAMILY,),
    ).fetchone()["analogy_trials"] == trials  # retry consumed no trial


def test_find_analogies_pit_hides_future_nodes(db, cfg):
    _build(db)
    graph.add_node(
        db, type="technique", label="future method",
        mech_tags=["regime_detection", "volatility_scaling"],
        family="quant", as_of="2025-01-01T00:00:00Z",
    )
    out = find_analogies(db, cfg, "regime-aware sizing", TARGET, as_of=AS_OF)
    assert all(m["label"] != "future method" for m in out["mappings"])
    out_future = find_analogies(
        db, cfg, "regime-aware sizing", TARGET, as_of="2025-06-01T00:00:00Z"
    )
    assert any(m["label"] == "future method" for m in out_future["mappings"])


def test_find_analogies_seeded_corpus_integration(seeded, cfg):
    out = find_analogies(
        seeded, cfg, "regime-aware sizing of trend signals",
        ["regime_switching", "regime_aware", "volatility"], k=3,
    )
    assert out["mappings"]
    assert len(out["mappings"]) <= 3
    for m in out["mappings"]:
        assert m["systematicity"] >= cfg.research.analogy_systematicity_threshold
        assert m["falsifiable_translation"]


def test_find_analogies_validates_k(db, cfg):
    _build(db)
    with pytest.raises(ValueError):
        find_analogies(db, cfg, "x", TARGET, k=0, as_of=AS_OF)


def test_analogy_errors_are_tool_errors():
    assert issubclass(AnalogyMappingRefused, ToolError)
    assert issubclass(AnalogyBudgetExceeded, ToolError)
