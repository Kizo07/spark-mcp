"""Native minimal GEPA (plan §11): a loop + scoring + mutation, not a
framework. See the package docstring for the loop shape; the functions here
are shared with skills.run_replay, which scores stored skill versions
against the same fixtures (adding an LLM judge and a persistence cache).

Fixture scoring, deterministic mode (no LLM):
- rubric item: tokenize the item ([a-z0-9]+, lowercase); keywords = tokens
  with len >= 4, stemmed by stripping one trailing "s" so basic inflections
  (check/checks, apply/applies) match; covered iff every stemmed keyword
  appears among the stems of the steps text. An item with no >=4-letter
  tokens counts as covered (vacuous).
- reference_keywords coverage: each entry is one word or phrase; covered iff
  its exact lowercase string appears as a substring of the steps text
  (so "fft" and "out-of-sample" both match naturally).
- fixture score = mean(rubric coverage, reference coverage); an empty list
  contributes its half vacuously (fixtures in seeds/ carry both).

LLM mode: each rubric item is judged 0-1 by the LLM (mean of item scores
replaces binary rubric coverage; reference coverage is unchanged). Judging
is the caller's job to cache — this module is pure.
"""

from __future__ import annotations

import random
import re
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)")
_JUDGE_MAX_TOKENS = 400

# Deterministic mutation operators, applied round-robin when refreshing a
# population. Each returns new steps (never mutates in place).
MUTATION_OPS = ("add_check_step", "drop_step", "rotate_steps")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _stem(token: str) -> str:
    # Minimal inflection handling, deliberately not a real stemmer:
    # checks -> check, assumptions -> assumption, applies -> apply.
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token


def _keywords(text: str) -> list[str]:
    return [_stem(t) for t in _tokens(text) if len(t) >= 4]


def _rubric_item_covered(steps_blob: str, item: str) -> bool:
    keywords = _keywords(item)
    if not keywords:
        return True
    step_stems = {_stem(t) for t in _tokens(steps_blob)}
    return all(kw in step_stems for kw in keywords)


def _reference_coverage(steps_blob: str, reference_keywords: list[str]) -> float:
    if not reference_keywords:
        return 1.0
    covered = sum(1 for kw in reference_keywords if kw.lower() in steps_blob)
    return covered / len(reference_keywords)


def _judge_item(llm: Any, steps: list[str], fixture: dict, item: str) -> float:
    """LLM judge for one rubric item: asks for a single 0-1 number and parses
    the first numeric token of the reply (clamped to [0, 1])."""
    steps_text = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
    prompt = (
        "You are grading one item of a research-procedure rubric.\n"
        f"Task: {fixture.get('prompt', '')}\n"
        f"Rubric item: {item}\n"
        "Procedure:\n"
        f"{steps_text}\n"
        "Reply with a single number between 0 and 1: how completely does the"
        " procedure satisfy the rubric item?"
    )
    reply = llm.complete(prompt[:_JUDGE_MAX_TOKENS * 4])
    match = _NUM_RE.search(reply)
    if match is None:
        return 0.0
    return min(1.0, max(0.0, float(match.group(0))))


def fixture_score(
    procedure_steps: list[str],
    fixture: dict,
    *,
    llm: Any | None = None,
) -> float:
    """Score one procedure variant against one fixture in [0, 1].

    Without an LLM: deterministic binary rubric coverage + reference-keyword
    coverage (definitions in the module docstring). With an LLM: per-item
    judge scores replace the binary rubric half.
    """
    steps_blob = " ".join(procedure_steps).lower()
    rubric = fixture.get("rubric") or []
    if llm is None:
        rubric_score = (
            sum(1 for item in rubric if _rubric_item_covered(steps_blob, item)) / len(rubric)
            if rubric
            else 1.0
        )
    else:
        rubric_score = (
            sum(_judge_item(llm, procedure_steps, fixture, item) for item in rubric) / len(rubric)
            if rubric
            else 1.0
        )
    ref_score = _reference_coverage(steps_blob, fixture.get("reference_keywords") or [])
    return (rubric_score + ref_score) / 2


def score_population(
    population: list[dict], fixtures: list[dict], *, llm: Any | None = None
) -> list[float]:
    """Mean fixture_score per individual (held-in fixtures only — held-out
    fixtures are promotion-time evidence, never evolution feedback, §21)."""
    return [
        sum(fixture_score(ind["procedure_steps"], f, llm=llm) for f in fixtures) / len(fixtures)
        if fixtures
        else 0.0
        for ind in population
    ]


def pareto_front(scored: list[tuple[dict, float]]) -> list[tuple[dict, float]]:
    """Nondominated set over (score ↑, len(procedure_steps) ↓): individual A
    dominates B iff A scores at least as high with no more steps, strictly on
    one axis. Ties are all kept (front, not ranking)."""
    front: list[tuple[dict, float]] = []
    for ind, score in scored:
        dominated = False
        for other, other_score in scored:
            if other is ind:
                continue
            no_worse = other_score >= score and len(other["procedure_steps"]) <= len(
                ind["procedure_steps"]
            )
            strictly_better = other_score > score or len(other["procedure_steps"]) < len(
                ind["procedure_steps"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append((ind, score))
    return front


def _add_check_step(ind: dict, fixtures: list[dict], rng: random.Random) -> list[str]:
    """Append a verification step derived from an uncovered rubric item of a
    randomly chosen fixture; when everything is covered, append an audit step."""
    steps = list(ind["procedure_steps"])
    candidates = [f for f in fixtures if f.get("rubric")]
    if candidates:
        fixture = candidates[rng.randrange(len(candidates))]
        blob = " ".join(steps).lower()
        uncovered = [
            item
            for item in fixture["rubric"]
            if not _rubric_item_covered(blob, item)
        ]
        if uncovered:
            steps.append(f"Verify: {uncovered[0]}.")
            return steps
    steps.append("Record the inputs, assumptions, and outputs of this step for audit.")
    return steps


def _drop_step(ind: dict, rng: random.Random) -> list[str]:
    steps = list(ind["procedure_steps"])
    if len(steps) > 1:
        steps.pop(rng.randrange(len(steps)))
    return steps


def _rotate_steps(ind: dict, rng: random.Random) -> list[str]:
    steps = list(ind["procedure_steps"])
    if len(steps) > 1:
        shift = rng.randint(1, len(steps) - 1)
        steps = steps[shift:] + steps[:shift]
    return steps


def _mutate(ind: dict, fixtures: list[dict], rng: random.Random, op: str) -> dict:
    if op == "add_check_step":
        steps = _add_check_step(ind, fixtures, rng)
    elif op == "drop_step":
        steps = _drop_step(ind, rng)
    else:
        steps = _rotate_steps(ind, rng)
    return {
        "name": ind["name"],
        "procedure_steps": steps,
        "references": list(ind.get("references") or []),
    }


def run_gepa(
    seed_individuals: list[dict],
    fixtures: list[dict],
    *,
    population_size: int,
    generations: int,
    pareto_top_k: int,
    seed: int = 0,
) -> dict:
    """Run the minimal evolution loop. `seed_individuals` must be non-empty;
    the population is padded to `population_size` with mutated variants and
    refreshed from the pareto top-k for `generations` rounds. Deterministic
    given (seed_individuals, fixtures, caps, seed)."""
    if not seed_individuals:
        raise ValueError("run_gepa needs at least one seed individual")
    if population_size < 1 or generations < 1 or pareto_top_k < 1:
        raise ValueError("population_size, generations and pareto_top_k must be >= 1")
    rng = random.Random(seed)

    population: list[dict] = [
        dict(ind, procedure_steps=list(ind["procedure_steps"])) for ind in seed_individuals
    ]
    i = 0
    while len(population) < population_size:
        base = seed_individuals[i % len(seed_individuals)]
        population.append(_mutate(base, fixtures, rng, MUTATION_OPS[i % len(MUTATION_OPS)]))
        i += 1

    for _ in range(generations):
        scores = score_population(population, fixtures)
        front = pareto_front(list(zip(population, scores, strict=True)))
        front.sort(key=lambda p: (-p[1], len(p[0]["procedure_steps"]), p[0]["name"]))
        parents = [ind for ind, _ in front[:pareto_top_k]]
        if not parents:  # pragma: no cover - front is never empty for a non-empty population
            parents = population[:1]
        refreshed = [dict(p, procedure_steps=list(p["procedure_steps"])) for p in parents]
        j = 0
        while len(refreshed) < population_size:
            base = parents[j % len(parents)]
            refreshed.append(_mutate(base, fixtures, rng, MUTATION_OPS[j % len(MUTATION_OPS)]))
            j += 1
        population = refreshed

    final_scores = score_population(population, fixtures)
    best_idx = max(
        range(len(population)),
        key=lambda k: (final_scores[k], -len(population[k]["procedure_steps"])),
    )
    best = population[best_idx]
    return {
        "best": {
            "name": best["name"],
            "procedure_steps": list(best["procedure_steps"]),
            "references": list(best.get("references") or []),
            "held_in_score": final_scores[best_idx],
        },
        "pareto_front": [
            {
                "name": ind["name"],
                "procedure_steps": list(ind["procedure_steps"]),
                "held_in_score": score,
            }
            for ind, score in pareto_front(list(zip(population, final_scores, strict=True)))
        ],
        "population_size": population_size,
        "generations": generations,
    }
