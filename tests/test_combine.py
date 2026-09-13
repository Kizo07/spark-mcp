"""combine_methods (plan §6.1 tool 3): the composition grammar (pipeline /
ensemble / residualizer), structural-fit notes, idempotency, validation."""

from __future__ import annotations

import pytest

from spark_mcp import graph
from spark_mcp.analogy import combine_methods
from spark_mcp.errors import ToolError
from spark_mcp.llm import FakeLLM

AS_OF = "2024-06-01T00:00:00Z"


def _build(db):
    """A/B/D share the quant family; A-B bridge on 'regime' compound atoms;
    D DIAGNOSES A (residualizer evidence); C is an unconnected stranger."""
    a = graph.add_node(
        db, type="technique", label="regime filter", family="quant",
        mech_tags=["regime_detection", "volatility_scaling"], as_of=AS_OF,
    )
    b = graph.add_node(
        db, type="technique", label="trend follower", family="quant",
        mech_tags=["regime_detection", "trend_following"], as_of=AS_OF,
    )
    c = graph.add_node(
        db, type="technique", label="earnings drift", family="econometrics",
        mech_tags=["earnings_drift", "event_window"], as_of=AS_OF,
    )
    d = graph.add_node(
        db, type="technique", label="regime diagnoser", family="quant",
        mech_tags=["regime_diagnosis", "error_correction"], as_of=AS_OF,
    )
    graph.add_edge(db, d["node_id"], a["node_id"], "DIAGNOSES", as_of=AS_OF)
    return a, b, c, d


def test_combine_produces_all_three_kinds(db, cfg):
    a, b, c, d = _build(db)
    out = combine_methods(
        db, cfg,
        [a["node_id"], b["node_id"], c["node_id"], d["node_id"]],
        "regime-aware trend sizing", as_of=AS_OF,
    )
    kinds = {comp["kind"] for comp in out["compositions"]}
    assert kinds == {"pipeline", "ensemble", "residualizer"}
    by_kind = {comp["kind"]: comp for comp in out["compositions"]}
    for comp in out["compositions"]:
        assert comp["composition_id"]
        notes = comp["structural_fit_notes"]
        assert set(notes) == {"shared_predicates", "family_bridge", "expected_interactions"}
        assert isinstance(comp["composed_mech_tags"], list)
        assert comp["rationale"]
    # pipeline: bridges the regime compound atoms
    pipe = by_kind["pipeline"]
    assert "regime" in pipe["structural_fit_notes"]["shared_predicates"]
    # ensemble: the quant pair (first repeated family in input order)
    ens = by_kind["ensemble"]
    assert ens["structural_fit_notes"]["family_bridge"] == "same family (quant)"
    # residualizer: D diagnoses A
    res = by_kind["residualizer"]
    labels = [m["label"] for m in res["methods"]]
    assert set(labels) == {"regime filter", "regime diagnoser"}
    assert "DIAGNOSES" in res["structural_fit_notes"]["expected_interactions"]


def test_combine_composed_tags_are_the_union(db, cfg):
    a, b, _, _ = _build(db)
    out = combine_methods(
        db, cfg, [a["node_id"], b["node_id"]], "p", as_of=AS_OF
    )
    for comp in out["compositions"]:
        members = {m["method_id"] for m in comp["methods"]}
        if members == {a["node_id"], b["node_id"]}:
            assert set(comp["composed_mech_tags"]) == {
                "regime_detection", "volatility_scaling", "trend_following",
            }


def test_combine_requires_two_ids(db, cfg):
    a, _, _, _ = _build(db)
    with pytest.raises(ToolError, match=">= 2"):
        combine_methods(db, cfg, [a["node_id"]], "p", as_of=AS_OF)
    with pytest.raises(ToolError, match=">= 2"):
        combine_methods(db, cfg, [], "p", as_of=AS_OF)


def test_combine_unknown_ids_refused(db, cfg):
    _build(db)
    with pytest.raises(ToolError, match="unknown or not-yet-visible"):
        combine_methods(db, cfg, ["node_nope", "node_nada"], "p", as_of=AS_OF)


def test_combine_no_structure_refused(db, cfg):
    x = graph.add_node(
        db, type="technique", label="lone wolf", family="nlp",
        mech_tags=["sentiment_parsing"], as_of=AS_OF,
    )
    y = graph.add_node(
        db, type="technique", label="other wolf", family="macro",
        mech_tags=["yield_curve"], as_of=AS_OF,
    )
    with pytest.raises(ToolError, match="no composable structure"):
        combine_methods(db, cfg, [x["node_id"], y["node_id"]], "p", as_of=AS_OF)


def test_combine_idempotent_on_client_id(db, cfg):
    a, b, _, _ = _build(db)
    first = combine_methods(
        db, cfg, [a["node_id"], b["node_id"]], "p",
        as_of=AS_OF, client_id="combine-idem-1",
    )
    events = db.conn.execute(
        "SELECT COUNT(*) AS n FROM trajectory_events WHERE payload LIKE '%combine_methods%'"
    ).fetchone()["n"]
    second = combine_methods(
        db, cfg, [a["node_id"], b["node_id"]], "p",
        as_of=AS_OF, client_id="combine-idem-1",
    )
    assert second == first
    assert db.conn.execute(
        "SELECT COUNT(*) AS n FROM trajectory_events WHERE payload LIKE '%combine_methods%'"
    ).fetchone()["n"] == events  # replay recorded no new event


def test_combine_llm_rationale_cached(db, cfg):
    a, b, _, _ = _build(db)
    llm = FakeLLM(
        replies={"why combining": "Shared regime vocabulary chains the two stages."}
    )
    out = combine_methods(
        db, cfg, [a["node_id"], b["node_id"]], "regime-aware trend sizing",
        as_of=AS_OF, llm=llm,
    )
    assert any(c["rationale_basis"] == "llm" for c in out["compositions"])
    calls = len(llm.calls)
    out2 = combine_methods(
        db, cfg, [a["node_id"], b["node_id"]], "regime-aware trend sizing",
        as_of=AS_OF, llm=llm,
    )
    assert len(llm.calls) == calls  # rationales served from the cache
    assert any(c["rationale_basis"] == "llm_cached" for c in out2["compositions"])


def test_combine_pit_hides_future_edges(db, cfg):
    # nodes visible from January; the DIAGNOSES edge only from June — a
    # reader at March sees the pair but not the residualizer evidence.
    a = graph.add_node(
        db, type="technique", label="host", family="quant",
        mech_tags=["regime_detection", "volatility_scaling"], as_of="2024-01-01T00:00:00Z",
    )
    d = graph.add_node(
        db, type="technique", label="diagnoser", family="quant",
        mech_tags=["regime_diagnosis", "error_correction"], as_of="2024-01-01T00:00:00Z",
    )
    graph.add_edge(db, d["node_id"], a["node_id"], "DIAGNOSES", as_of=AS_OF)
    out = combine_methods(
        db, cfg, [a["node_id"], d["node_id"]], "p", as_of="2024-03-01T00:00:00Z"
    )
    assert "residualizer" not in {c["kind"] for c in out["compositions"]}
    out_later = combine_methods(
        db, cfg, [a["node_id"], d["node_id"]], "p", as_of="2024-07-01T00:00:00Z"
    )
    assert "residualizer" in {c["kind"] for c in out_later["compositions"]}
    # nodes invisible at the earlier horizon entirely -> unknown ids
    with pytest.raises(ToolError, match="unknown or not-yet-visible"):
        combine_methods(db, cfg, [a["node_id"], d["node_id"]], "p", as_of="1024-01-01T00:00:00Z")
