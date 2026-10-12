"""Phase 6: native minimal GEPA (plan §11) — fixture scoring, pareto front,
and the seeded evolution loop. No dspy anywhere (independence contract);
deterministic given (seeds, fixtures, caps, seed).
"""

import pytest

from spark_mcp.evolve import gepa
from spark_mcp.skills import load_fixtures

FIXTURE = {
    "task_id": "t1",
    "prompt": "audit a momentum backtest",
    "family": "quant",
    "held_in": True,
    "rubric": [
        "applies explicit transaction cost and slippage assumptions",
        "checks turnover and capacity before trusting the result",
    ],
    "reference_keywords": ["momentum", "turnover", "out-of-sample"],
}

GOOD_STEPS = [
    "Rank the cross section on 12-1 momentum.",
    "Apply explicit transaction cost and slippage assumptions.",
    "Check turnover and capacity before trusting the result.",
    "Validate the signal out-of-sample.",
]


def test_fixture_score_perfect_steps():
    assert gepa.fixture_score(GOOD_STEPS, FIXTURE) == pytest.approx(1.0)


def test_fixture_score_empty_steps_is_zero():
    assert gepa.fixture_score([], FIXTURE) == 0.0


def test_fixture_score_partial_coverage():
    # one of two rubric items; none of the three reference keywords:
    # (0.5 + 0.0) / 2
    steps = ["Apply explicit transaction cost and slippage assumptions."]
    assert gepa.fixture_score(steps, FIXTURE) == pytest.approx(0.25)


def test_reference_keywords_are_substring_matched():
    # "out-of-sample" matches as a literal substring; short words like "fft"
    # are not subject to the >=4-letter rubric token filter. Rubric is empty
    # here, so the rubric half is vacuously 1.0.
    fx = {"rubric": [], "reference_keywords": ["fft", "out-of-sample"]}
    assert gepa.fixture_score(["run an fft check out-of-sample"], fx) == pytest.approx(1.0)
    assert gepa.fixture_score(["run an fft check"], fx) == pytest.approx(0.75)


def test_llm_judge_replaces_binary_rubric_half():
    class _Judge:
        def complete(self, prompt, *, json_schema=None):
            return "0.8"

        def model_id(self):
            return "fake/1"

    fx = {"rubric": ["anything at all here"], "reference_keywords": ["momentum"]}
    # LLM rubric half 0.8, reference half 1.0 -> 0.9
    assert gepa.fixture_score(["momentum"], fx, llm=_Judge()) == pytest.approx(0.9)


def test_llm_judge_without_number_scores_zero():
    class _Mute:
        def complete(self, prompt, *, json_schema=None):
            return "no idea"

        def model_id(self):
            return "fake/1"

    fx = {"rubric": ["applies explicit transaction cost assumptions"],
          "reference_keywords": ["panel"]}  # absent from GOOD_STEPS
    assert gepa.fixture_score(GOOD_STEPS, fx, llm=_Mute()) == pytest.approx(0.0)


def test_pareto_front_prefers_shorter_at_equal_score():
    a = {"name": "a", "procedure_steps": ["one", "two"]}
    b = {"name": "b", "procedure_steps": ["one", "two", "three"]}
    front = gepa.pareto_front([(a, 0.5), (b, 0.5)])
    assert [ind["name"] for ind, _ in front] == ["a"]


def test_pareto_front_keeps_tradeoffs():
    a = {"name": "a", "procedure_steps": ["one"]}  # short, low score
    b = {"name": "b", "procedure_steps": ["one", "two", "three"]}  # long, high score
    front = gepa.pareto_front([(a, 0.2), (b, 0.8)])
    assert {ind["name"] for ind, _ in front} == {"a", "b"}


def test_run_gepa_caps_and_determinism():
    fixtures = [f for f in load_fixtures() if f.get("held_in", True)]
    seeds = [{"name": "s", "procedure_steps": ["Rank on momentum."], "references": []}]
    caps = dict(population_size=6, generations=2, pareto_top_k=3)
    r1 = gepa.run_gepa(seeds, fixtures, seed=11, **caps)
    r2 = gepa.run_gepa(seeds, fixtures, seed=11, **caps)
    r3 = gepa.run_gepa(seeds, fixtures, seed=99, **caps)
    assert r1["population_size"] == 6
    assert r1["generations"] == 2
    assert r1["best"]["procedure_steps"] == r2["best"]["procedure_steps"]
    assert r1["best"]["held_in_score"] == r2["best"]["held_in_score"]
    assert r1["best"]["held_in_score"] >= 0.0
    # A different seed must be permitted to differ (not asserted to, but the
    # loop must stay valid either way).
    assert r3["best"]["held_in_score"] >= 0.0


def test_run_gepa_population_padded_with_mutations():
    fixtures = [f for f in load_fixtures() if f.get("held_in", True)]
    seeds = [{"name": "s", "procedure_steps": ["a", "b", "c", "d"], "references": []}]
    result = gepa.run_gepa(seeds, fixtures, population_size=6, generations=1, pareto_top_k=3)
    assert result["population_size"] == 6
    assert result["pareto_front"]  # front reported


def test_run_gepa_requires_seeds_and_valid_caps():
    with pytest.raises(ValueError, match="seed individual"):
        gepa.run_gepa([], [], population_size=6, generations=2, pareto_top_k=3)
    with pytest.raises(ValueError, match=">= 1"):
        gepa.run_gepa(
            [{"name": "s", "procedure_steps": ["a"]}],
            [],
            population_size=0,
            generations=2,
            pareto_top_k=3,
        )


def test_no_add_check_step_symbol_in_source():
    """Evolution upgrade plan §2.F1 regression: the rubric-leaking operator
    (including its name) must not survive anywhere under src/."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "spark_mcp"
    offenders = []
    for py in src.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        if "_add_check_step" in text or '"add_check_step"' in text:
            offenders.append(py.name)
    assert not offenders, f"leaking operator survives in: {offenders}"
    assert "add_guard_step" in gepa.MUTATION_OPS
    assert "add_check_step" not in gepa.MUTATION_OPS


def test_add_guard_step_conditions_on_failure_evidence():
    ind = {"name": "s", "procedure_steps": ["Rank on momentum."]}
    sketches = [
        {"error_class": "DataGap"},
        {"error_class": "DataGap"},
        {"error_class": "SchemaDrift"},
        {"error_class": "  "},  # blank classes are ignored
    ]
    import random

    steps = gepa._add_guard_step(ind, sketches, random.Random(0))
    assert len(steps) == 2
    assert steps[0] == "Rank on momentum."
    # Exact skills._op_repair template, most-common class first.
    assert steps[1] == (
        "Guard against DataGap: check data coverage, preconditions, and"
        " inputs before executing this procedure (observed in 2 source"
        " episode(s))."
    )
    # The input is never mutated in place.
    assert ind["procedure_steps"] == ["Rank on momentum."]


def test_add_guard_step_fallback_sentence_pinned():
    import random

    ind = {"name": "s", "procedure_steps": ["a"]}
    assert gepa.GUARD_STEP_FALLBACK == (
        "Record the inputs, assumptions, and outputs of this step for audit."
    )
    for sketches in (None, [], [{"error_class": ""}, {"stage": "x"}]):
        steps = gepa._add_guard_step(ind, sketches, random.Random(0))
        assert steps[-1] == gepa.GUARD_STEP_FALLBACK


def test_guard_step_never_emits_rubric_text():
    """F1 leak regression: guard steps (evidence + fallback paths) carry no
    verbatim rubric item and no reference keyword from any fixture."""
    import random

    fixtures = load_fixtures()
    assert len(fixtures) >= 6
    ind = {"name": "s", "procedure_steps": ["Rank on momentum."]}
    candidates = [
        gepa._add_guard_step(
            ind,
            [{"error_class": "DataGap"}, {"error_class": "DataGap"}],
            random.Random(0),
        )[-1],
        gepa._add_guard_step(ind, None, random.Random(0))[-1],
    ]
    for step in candidates:
        lowered = step.lower()
        for fixture in fixtures:
            for item in fixture["rubric"]:
                assert item.lower() not in lowered, (step, item)
            for kw in fixture["reference_keywords"]:
                assert kw.lower() not in lowered, (step, kw)


def test_mutate_never_receives_fixtures():
    """F1 routing rule: _mutate takes failure sketches, not fixtures."""
    import inspect
    import random

    params = inspect.signature(gepa._mutate).parameters
    assert "fixtures" not in params
    assert "failure_sketches" in params
    ind = {"name": "s", "procedure_steps": ["a", "b"]}
    rng = random.Random(3)
    for op in gepa.MUTATION_OPS:
        out = gepa._mutate(ind, [{"error_class": "DataGap"}], rng, op)
        assert out["name"] == "s"
        assert out["procedure_steps"]


def test_reflect_replaces_rotate_steps():
    """F2: the score-neutral rotate op is gone; `reflect` holds its slot."""
    assert gepa.MUTATION_OPS == ("add_guard_step", "drop_step", "reflect")
    assert not hasattr(gepa, "_rotate_steps")
    with pytest.raises(ValueError, match="unknown mutation op"):
        import random

        gepa._mutate(
            {"name": "s", "procedure_steps": ["a", "b"]}, None, random.Random(0),
            "rotate_steps",
        )


def test_reflect_fallback_deterministic_and_well_formed():
    """F2 degraded path (no LLM): seeded, well-formed, never in-place."""
    import random

    ind = {"name": "s", "procedure_steps": ["Rank on momentum.", "Check costs."]}
    first = gepa._reflect_step(ind, [{"error_class": "DataGap"}], random.Random(4), None)
    second = gepa._reflect_step(ind, [{"error_class": "DataGap"}], random.Random(4), None)
    assert first == second
    assert first and all(isinstance(s, str) and s.strip() for s in first)
    assert ind["procedure_steps"] == ["Rank on momentum.", "Check costs."]


def test_reflect_llm_path_rewrites_steps():
    """F2 reflective path: a schema-conformant LLM reply becomes the steps."""
    import json
    import random

    from spark_mcp.llm import FakeLLM

    ind = {"name": "s", "procedure_steps": ["Rank on momentum."]}
    llm = FakeLLM({"procedure": json.dumps({"steps": ["Rank on quality.", "Size it."]})})
    steps = gepa._reflect_step(ind, None, random.Random(0), llm)
    assert steps == ["Rank on quality.", "Size it."]
    assert llm.calls  # exactly one completion per reflect call
    assert len(llm.calls) == 1


def test_run_gepa_llm_kwarg_deterministic():
    """run_gepa threads llm to the reflect op; scripted replies replay."""
    import json

    from spark_mcp.llm import FakeLLM

    fixtures = [f for f in load_fixtures() if f.get("held_in", True)]
    seeds = [{"name": "s", "procedure_steps": ["Rank on momentum."], "references": []}]
    caps = dict(population_size=6, generations=2, pareto_top_k=3)
    # Force every reflect call onto the LLM path with a fixed rewrite.
    llm = FakeLLM({"procedure": json.dumps({"steps": ["Rank on momentum, then audit."]})})
    r1 = gepa.run_gepa(seeds, fixtures, seed=11, llm=llm, **caps)
    llm2 = FakeLLM({"procedure": json.dumps({"steps": ["Rank on momentum, then audit."]})})
    r2 = gepa.run_gepa(seeds, fixtures, seed=11, llm=llm2, **caps)
    assert r1["best"]["procedure_steps"] == r2["best"]["procedure_steps"]
    assert llm.calls and len(llm.calls) == len(llm2.calls)
    # The reflective rewrite actually entered the population (not a silent
    # fallback): every reflect LLM call returned the audit sentence.
    assert any(
        "audit" in s
        for ind in [r1["best"], *[m for m in r1["pareto_front"]]]
        for s in ind["procedure_steps"]
    )


def test_run_gepa_failure_sketches_deterministic():
    fixtures = [f for f in load_fixtures() if f.get("held_in", True)]
    seeds = [{"name": "s", "procedure_steps": ["Rank on momentum."], "references": []}]
    caps = dict(population_size=6, generations=2, pareto_top_k=3)
    sketches = [{"error_class": "DataGap"}]
    r1 = gepa.run_gepa(seeds, fixtures, seed=11, failure_sketches=sketches, **caps)
    r2 = gepa.run_gepa(seeds, fixtures, seed=11, failure_sketches=sketches, **caps)
    assert r1["best"]["procedure_steps"] == r2["best"]["procedure_steps"]
    # Default (no sketches) keeps old call-sites compiling and deterministic.
    r3 = gepa.run_gepa(seeds, fixtures, seed=11, **caps)
    r4 = gepa.run_gepa(seeds, fixtures, seed=11, **caps)
    assert r3["best"]["procedure_steps"] == r4["best"]["procedure_steps"]
