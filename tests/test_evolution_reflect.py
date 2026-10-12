"""Generative LLM variation (evolution upgrade plan §2.B, §5).

One LLM call asking for a schema-conformant successor; every failure mode
(no LLM / raise / unparsable / schema-invalid) degrades to the seeded
deterministic fallback. Same (parent, feedback, seed) → same proposal.
"""

import copy
import hashlib
import json
import random

import pytest

from spark_mcp.errors import ToolError
from spark_mcp.evolve import reflect as reflect_mod
from spark_mcp.evolve.gepa import GUARD_STEP_FALLBACK
from spark_mcp.llm import FakeLLM

PARENT = {
    "research_family": "toy_momentum",
    "hypothesis_type": "predictive",
    "mechanism": "12-1 momentum in toy universe",
    "expected_direction": "positive",
    "horizon": "21d",
    "scope": "cross-section",
    "falsifier": "IC <= 0 net of costs",
    "predicted_ic": 0.03,
    "predicted_sigma": 0.01,
    "universe": "toy",
    "idea": "momentum persists",
    "dag": None,
    "adjustment_set": None,
}

LLM_HYPOTHESIS = {
    "mechanism": "6-1 momentum with volatility scaling",
    "expected_direction": "positive",
    "horizon": "126d",
    "scope": "cross-section ex microcaps",
    "falsifier": "scaled IC <= 0 net of costs",
    "predicted_ic": 0.04,
    "predicted_sigma": 0.012,
}


def _reflect(**kw):
    args = dict(
        parent=copy.deepcopy(PARENT),
        feedback={"verdict": "rejected: weak margins"},
        ancestors_render="Trial 1: ...",
        rng=random.Random(0),
        llm=None,
        schema="hypothesis",
    )
    args.update(kw)
    return reflect_mod.reflect_variation(**args)


class _RaisingLLM:
    def complete(self, prompt, *, json_schema=None):
        raise RuntimeError("boom")

    def model_id(self):
        return "raising/0"


def test_llm_success_returns_validated_body_and_audits():
    llm = FakeLLM({"successor": json.dumps(LLM_HYPOTHESIS)}, model="test-llm/1")
    proposal = _reflect(llm=llm)
    assert proposal.fallback_used is False
    assert proposal.model_id == "test-llm/1"
    assert len(llm.calls) == 1  # exactly one completion
    assert proposal.prompt_sha == hashlib.sha1(llm.calls[0].encode()).hexdigest()
    for key in reflect_mod.HYPOTHESIS_REQUIRED_FIELDS:
        assert proposal.body[key] == LLM_HYPOTHESIS[key]
    # inherited, not LLM-supplied
    assert proposal.body["research_family"] == "toy_momentum"
    assert proposal.body["universe"] == "toy"
    assert proposal.body["hypothesis_type"] == "predictive"


def test_llm_reply_drops_unknown_keys_and_tolerates_fences():
    reply = (
        "Here is the successor:\n```json\n"
        + json.dumps(
            {**LLM_HYPOTHESIS, "parent_hypothesis_ids": ["hyp_x"], "client_id": "c"}
        )
        + "\n```"
    )
    llm = FakeLLM({"successor": reply})
    proposal = _reflect(llm=llm)
    assert proposal.fallback_used is False
    assert "parent_hypothesis_ids" not in proposal.body
    assert "client_id" not in proposal.body
    assert set(proposal.body) == set(
        reflect_mod.HYPOTHESIS_REQUIRED_FIELDS + reflect_mod.HYPOTHESIS_INHERITED_FIELDS
    )


@pytest.mark.parametrize(
    "reply",
    [
        "not json at all",
        json.dumps({"mechanism": "x"}),  # missing required fields
        json.dumps({**LLM_HYPOTHESIS, "falsifier": "  "}),  # blank required
        json.dumps({**LLM_HYPOTHESIS, "predicted_ic": "high"}),  # non-numeric
        json.dumps({**LLM_HYPOTHESIS, "hypothesis_type": "vibes"}),  # unknown type
        json.dumps({**LLM_HYPOTHESIS, "hypothesis_type": "causal"}),  # causal w/o dag
    ],
)
def test_llm_failures_degrade_to_deterministic_fallback(reply):
    llm = FakeLLM({"successor": reply})
    first = _reflect(llm=llm, rng=random.Random(5))
    assert first.fallback_used is True
    assert first.model_id is None
    assert first.prompt_sha is not None
    second = _reflect(llm=FakeLLM({"successor": reply}), rng=random.Random(5))
    assert first.body == second.body
    # fallback bodies are C6-complete
    for key in reflect_mod.HYPOTHESIS_REQUIRED_FIELDS:
        assert first.body[key]


def test_llm_raise_and_none_degrade_to_fallback():
    for llm in (_RaisingLLM(), None):
        proposal = _reflect(llm=llm, rng=random.Random(5))
        assert proposal.fallback_used is True
        assert proposal.body["research_family"] == "toy_momentum"


def test_fallback_never_mutates_parent_and_covers_all_ops():
    seen = set()
    for seed in range(60):
        parent = copy.deepcopy(PARENT)
        snapshot = copy.deepcopy(parent)
        proposal = _reflect(parent=parent, rng=random.Random(seed))
        assert parent == snapshot
        tag = proposal.body["idea"].rsplit("[evolution variant: ", 1)[1].rstrip("]")
        seen.add(tag)
    assert seen == set(reflect_mod.HYPOTHESIS_FALLBACK_OPS)


def test_fallback_negate_direction_flips_sign_and_falsifier():
    for seed in range(200):
        proposal = _reflect(rng=random.Random(seed))
        if proposal.body["idea"].endswith("[evolution variant: negate_direction]"):
            break
    else:
        raise AssertionError("negate_direction never drawn in 200 seeds")
    assert proposal.body["expected_direction"] == "negative"
    assert proposal.body["predicted_ic"] == pytest.approx(-PARENT["predicted_ic"])
    assert proposal.body["falsifier"] != PARENT["falsifier"]
    assert proposal.body["falsifier"].strip()


def test_fallback_preserves_causal_dag():
    parent = dict(
        PARENT,
        hypothesis_type="causal",
        dag={"nodes": ["a", "b"]},
        adjustment_set=["z"],
    )
    for seed in range(60):
        proposal = _reflect(parent=copy.deepcopy(parent), rng=random.Random(seed))
        assert proposal.body["dag"] == {"nodes": ["a", "b"]}
        assert proposal.body["adjustment_set"] == ["z"]
        assert proposal.body["hypothesis_type"] == "causal"


def test_root_synthesis_needs_llm():
    full = dict(
        LLM_HYPOTHESIS,
        research_family="toy_momentum",
        hypothesis_type="predictive",
        universe="toy",
    )
    proposal = _reflect(parent=None, llm=FakeLLM({"successor": json.dumps(full)}))
    assert proposal.fallback_used is False
    assert proposal.body["research_family"] == "toy_momentum"
    with pytest.raises(ToolError, match="without an LLM"):
        _reflect(parent=None, llm=None)
    with pytest.raises(ToolError, match="without an LLM"):
        _reflect(parent=None, llm=FakeLLM({"successor": "garbage"}))


def test_skill_steps_llm_paths():
    steps = ["Load the data.", "Run the check."]
    obj = _reflect(
        parent=list(steps),
        llm=FakeLLM({"research procedure": json.dumps({"steps": ["A", "B"]})}),
        schema="skill_steps",
    )
    assert obj.body == ["A", "B"] and obj.fallback_used is False
    arr = _reflect(
        parent=list(steps), llm=FakeLLM({"research procedure": '["X"]'}),
        schema="skill_steps",
    )
    assert arr.body == ["X"]
    bad = _reflect(
        parent=list(steps),
        llm=FakeLLM({"research procedure": json.dumps({"steps": []})}),
        schema="skill_steps", rng=random.Random(1),
    )
    assert bad.fallback_used is True and bad.body  # fallback never empties


def test_skill_steps_fallback_ops_and_guard_evidence():
    steps = ["Load the data.", "Run the check.", "Write the report."]
    ops_seen = set()
    for seed in range(60):
        proposal = _reflect(
            parent=list(steps),
            feedback={"failure_sketches": [{"error_class": "leak"}]},
            rng=random.Random(seed),
            schema="skill_steps",
        )
        body = proposal.body
        assert body and all(isinstance(s, str) and s.strip() for s in body)
        if len(body) == 2 and set(body) < set(steps):
            ops_seen.add("drop_or_split")
        if any("leak" in s for s in body):
            ops_seen.add("guard_step")
        if body == ["Run the check.", "Write the report.", "Load the data."]:
            ops_seen.add("clarify")
    assert {"drop_or_split", "guard_step", "clarify"} <= ops_seen
    # guard falls back to the generic audit sentence without failure evidence
    # (only the guard op grows 3 steps to 4, so len == 4 pins it exactly)
    for seed in range(60):
        proposal = _reflect(parent=list(steps), feedback=None, rng=random.Random(seed),
                            schema="skill_steps")
        if len(proposal.body) == 4:
            assert proposal.body[-1] == GUARD_STEP_FALLBACK
            break
    else:
        raise AssertionError("guard_step never drawn in 60 seeds")


def test_contract_violations_raise_value_error():
    with pytest.raises(ValueError, match="unknown schema"):
        _reflect(schema="nope")
    with pytest.raises(ValueError, match="parent hypothesis dict"):
        _reflect(parent=["not", "a", "dict"])
    with pytest.raises(ValueError, match="parent step list"):
        _reflect(parent={"a": 1}, schema="skill_steps")
    with pytest.raises(ValueError, match="feedback must be"):
        _reflect(feedback=123)
    with pytest.raises(ValueError, match="ancestors_render must be"):
        _reflect(ancestors_render=None)


def test_llm_path_never_mutates_inputs():
    parent = copy.deepcopy(PARENT)
    feedback = {"verdict": "x", "n": [1, 2]}
    llm = FakeLLM({"successor": json.dumps(LLM_HYPOTHESIS)})
    _reflect(parent=parent, feedback=feedback, llm=llm)
    assert parent == PARENT and feedback == {"verdict": "x", "n": [1, 2]}
