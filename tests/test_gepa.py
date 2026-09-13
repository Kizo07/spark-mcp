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
