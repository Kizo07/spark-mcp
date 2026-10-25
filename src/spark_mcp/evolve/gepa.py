"""Native minimal GEPA (plan §11): a loop + scoring + mutation, not a
framework. See the package docstring for the loop shape; the functions here
are shared with skills.run_replay, which scores stored skill versions
against the same fixtures (adding an LLM judge and a persistence cache).

Fixture scoring, deterministic mode (no LLM):
- substance pre-filter: only steps with instruction substance enter the
  scored blob — a step must open with an instruction verb (imperative or
  third-person, from a fixed verb set) and carry at least 3 tokens.
  Empty, whitespace-only, and keyword-only steps (no leading instruction
  verb) are dropped; with no substantive step left the score is 0.0.
  This is the scorer-level anti-soup defence: a keyword dump matches no
  rubric from a substantive step. The LLM-judge path is unfiltered — the
  judge itself assesses substance per rubric item.
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
from collections import Counter
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)")
_JUDGE_MAX_TOKENS = 400

#: Minimum tokens for a step to carry instruction substance (alongside the
#: leading-verb rule in _is_substantive_step). Steps are plain non-empty
#: strings per the skill schema (skills.distill validates non-empty str;
#: reflect._validate_steps_body the same), so substance is a content rule:
#: empty/whitespace/keyword-only steps never enter the scored blob.
_MIN_SUBSTANTIVE_TOKENS = 3

#: Base instruction verbs opening a substantive procedure step. Matched
#: against the step's first token plus its regular third-person forms
#: (verb + "s", with explicit -es/-ies forms below), all lowercase.
_INSTRUCTION_VERB_BASES = frozenset(
    """
    state apply report check name validate verify draw run document choose
    cluster test select estimate rank size guard record use compute compare
    measure assess ensure confirm review audit analyze analyse evaluate monitor
    track rebalance weight normalize standardize winsorize filter exclude
    include define specify set fix hold buy sell trade execute backtest
    simulate calibrate fit train split partition perform implement build
    create write add remove drop keep list show plot save load fetch download
    clean handle detect identify determine decide repeat average aggregate
    sort group regress enforce constrain cap floor lag smooth detrend
    orthogonalize residualize hedge do
    """.split()
)
_INSTRUCTION_VERBS = (
    _INSTRUCTION_VERB_BASES
    | {b + "s" for b in _INSTRUCTION_VERB_BASES}
    | {
        "applies",
        "specifies",
        "identifies",
        "verifies",
        "fixes",
        "fetches",
        "regresses",
        "does",
    }
)

# Fallback guard step appended when no failure evidence exists (pinned: the
# regression test asserts this exact sentence and that it carries no rubric
# text).
GUARD_STEP_FALLBACK = "Record the inputs, assumptions, and outputs of this step for audit."

# Deterministic mutation operators, applied round-robin when refreshing a
# population. Each returns new steps (never mutates in place). Fixtures are
# never passed to mutation ops — they route to the scorer only — so no op
# can read rubric text (evolution upgrade plan §2.F1). `reflect` (plan §2.F2)
# replaced the score-neutral `rotate_steps`: an LLM rewrite when a model is
# available, else reflect's seeded syntactic fallback.
MUTATION_OPS = ("add_guard_step", "drop_step", "reflect")


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


def _is_substantive_step(step: str) -> bool:
    """A step carries instruction substance iff it opens with an instruction
    verb and has at least _MIN_SUBSTANTIVE_TOKENS tokens. Keyword dumps —
    sorted word lists with no leading verb — fail the verb rule even when
    they contain verb tokens mid-string."""
    tokens = _tokens(step.strip())
    return len(tokens) >= _MIN_SUBSTANTIVE_TOKENS and tokens[0] in _INSTRUCTION_VERBS


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
    coverage (definitions in the module docstring), computed over substantive
    steps only — no substantive step means 0.0. With an LLM: per-item
    judge scores replace the binary rubric half (unfiltered; the judge
    assesses substance itself).
    """
    if llm is None:
        substantive = [s for s in procedure_steps if _is_substantive_step(s)]
        if not substantive:
            return 0.0
        steps_blob = " ".join(substantive).lower()
    else:
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


#: Allowlist for failure-sketch error classes interpolated into guard
#: steps: identifier characters only, no whitespace — so rubric-shaped
#: multi-word text cannot be copied verbatim into a procedure step.
_ERROR_CLASS_RE = re.compile(r"[^A-Za-z0-9_.-]")
_MAX_ERROR_CLASS_LEN = 64


def _sanitize_error_class(error_class: str) -> str:
    """Strip everything outside the identifier allowlist and cap length.
    There is no known-class registry (Tier-2 hook data is free-form), so
    charset + length is the defence; it cannot consult fixtures (the F1
    routing rule forbids mutation paths from reading rubric text)."""
    return _ERROR_CLASS_RE.sub("", error_class)[:_MAX_ERROR_CLASS_LEN]


def _guard_sentence(error_class: str, count: int) -> str:
    # Exact template of skills._op_repair (kept in sync by the GEPA repair
    # tests): the guard conditions on failure evidence, never rubric text.
    # The class is sanitized at this choke point (shared with reflect's
    # guard_step fallback), so a hostile failure sketch cannot inject
    # rubric-shaped sentences; a class with no identifier content at all
    # degrades to the generic fallback sentence.
    safe = _sanitize_error_class(error_class)
    if not safe:
        return GUARD_STEP_FALLBACK
    return (
        f"Guard against {safe}: check data coverage, preconditions, and"
        f" inputs before executing this procedure (observed in {count} source"
        " episode(s))."
    )


def _add_guard_step(
    ind: dict, failure_sketches: list[dict] | None, rng: random.Random
) -> list[str]:
    """Append a guard step conditioned on *failure evidence* — the most common
    Tier-2 `failure_sketches` error class — or the generic audit sentence
    when no failure evidence exists. Never reads fixtures or rubric text.
    Deterministic given (ind, failure_sketches): no randomness consumed."""
    del rng  # deterministic by construction; kept for _mutate uniformity
    steps = list(ind["procedure_steps"])
    classes = [
        str(s.get("error_class", "")).strip()
        for s in (failure_sketches or [])
        if str(s.get("error_class", "")).strip()
    ]
    if classes:
        error_class, count = Counter(classes).most_common(1)[0]
        steps.append(_guard_sentence(error_class, count))
    else:
        steps.append(GUARD_STEP_FALLBACK)
    return steps


def _drop_step(ind: dict, rng: random.Random) -> list[str]:
    steps = list(ind["procedure_steps"])
    if len(steps) > 1:
        steps.pop(rng.randrange(len(steps)))
    return steps


def _reflect_step(
    ind: dict, failure_sketches: list[dict] | None, rng: random.Random, llm: Any | None
) -> list[str]:
    """Reflective rewrite of the procedure (plan §2.F2): component B applied
    to skill text — one LLM rewrite when a model is available, else the
    seeded syntactic fallback (deterministic given parent + sketches + rng).
    The import is function-local: reflect imports this module's guard
    constants, so a top-level import would be circular.
    """
    from .reflect import reflect_variation

    feedback: dict | None = (
        {"failure_sketches": list(failure_sketches)} if failure_sketches else None
    )
    proposal = reflect_variation(
        parent=list(ind["procedure_steps"]),
        feedback=feedback,
        ancestors_render="",
        rng=rng,
        llm=llm,
        schema="skill_steps",
    )
    assert isinstance(proposal.body, list)
    return list(proposal.body)


def _mutate(
    ind: dict,
    failure_sketches: list[dict] | None,
    rng: random.Random,
    op: str,
    llm: Any | None = None,
) -> dict:
    # Routing rule: no mutation op receives fixtures — fixtures route to the
    # scorer (score_population/fixture_score) only, so no op can re-read
    # rubric text.
    if op == "add_guard_step":
        steps = _add_guard_step(ind, failure_sketches, rng)
    elif op == "drop_step":
        steps = _drop_step(ind, rng)
    elif op == "reflect":
        steps = _reflect_step(ind, failure_sketches, rng, llm)
    else:
        raise ValueError(f"unknown mutation op {op!r}; expected one of {MUTATION_OPS}")
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
    failure_sketches: list[dict] | None = None,
    llm: Any | None = None,
) -> dict:
    """Run the minimal evolution loop. `seed_individuals` must be non-empty;
    the population is padded to `population_size` with mutated variants and
    refreshed from the pareto top-k for `generations` rounds. `fixtures`
    score the population only; `failure_sketches` (Tier-2 error evidence)
    feed the guard-step mutation op; `llm` (when given) powers the `reflect`
    mutation op, else its seeded fallback runs. Deterministic given
    (seed_individuals, fixtures, failure_sketches, caps, seed, llm replies).
    """
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
        population.append(
            _mutate(base, failure_sketches, rng, MUTATION_OPS[i % len(MUTATION_OPS)], llm)
        )
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
            refreshed.append(
                _mutate(base, failure_sketches, rng, MUTATION_OPS[j % len(MUTATION_OPS)], llm)
            )
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
