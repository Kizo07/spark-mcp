"""Generative LLM variation with a deterministic fallback (plan §2.B).

The reflective mutation operator spark lacks: one `LLMClient.complete` call
asking for a schema-conformant successor (full C6 field set for hypotheses,
a rewritten step list for skills), parsed with the tolerant balanced-brace
approach used by explore scoring. Any failure — no LLM, LLM raise,
unparsable reply, schema-invalid body — falls back to deterministic
syntactic operators chosen via the run's seeded `rng`, so the degraded
path is bit-reproducible: same (parent, feedback, seed) → same proposal.

Every proposal records `{prompt_sha, model_id, fallback_used}` for
judge-cache-style auditability on the lineage/feedback row. Temperature 0
comes from `OpenAICompatibleLLM` itself; `llm.py` needs no changes.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .. import ledger
from ..errors import ToolError
from ..llm import LLMClient
from .gepa import GUARD_STEP_FALLBACK, _guard_sentence

#: C6 field set a hypothesis successor must carry so `propose_hypothesis`
#: can validate it (dag/adjustment_set additionally iff causal).
HYPOTHESIS_REQUIRED_FIELDS = (
    "mechanism",
    "expected_direction",
    "horizon",
    "scope",
    "falsifier",
    "predicted_ic",
    "predicted_sigma",
)

#: Fields a successor inherits from its parent unless the LLM overrides them.
HYPOTHESIS_INHERITED_FIELDS = (
    "research_family",
    "hypothesis_type",
    "universe",
    "idea",
    "dag",
    "adjustment_set",
)

#: Deterministic fallback operators: field-level edits for hypotheses
#: (each preserving C6 validity), syntactic ops for skill text.
HYPOTHESIS_FALLBACK_OPS = (
    "tighten_horizon",
    "loosen_horizon",
    "negate_direction",
    "narrow_scope",
)
STEPS_FALLBACK_OPS = ("drop_step", "guard_step", "split", "clarify")


@dataclass(frozen=True)
class VariationProposal:
    body: dict | list[str]  # hypothesis fields or procedure steps
    model_id: str | None  # None on deterministic fallback
    prompt_sha: str | None
    fallback_used: bool


def _feedback_text(feedback: dict | str | None) -> str:
    if feedback is None:
        return "No prior feedback recorded."
    if isinstance(feedback, str):
        return feedback
    return json.dumps(feedback, sort_keys=True, default=str)


def _render_hypothesis_prompt(
    parent: dict | None, feedback_text: str, ancestors_render: str
) -> str:
    if parent is None:
        parent_block = (
            "No parent hypothesis: propose a NEW ROOT hypothesis from scratch."
            " Your reply must additionally include research_family,"
            " hypothesis_type ('predictive' unless the hypothesis is causal),"
            " and universe."
        )
    else:
        parent_block = (
            "Parent hypothesis (JSON):\n"
            + json.dumps(parent, sort_keys=True, default=str)
        )
    return (
        "You are evolving a research hypothesis. Propose ONE successor"
        " hypothesis that responds to the feedback below. Reply with a single"
        " JSON object carrying the full C6 field set: mechanism,"
        " expected_direction, horizon, scope, falsifier, predicted_ic,"
        " predicted_sigma (numbers, not strings). Include dag (object) and"
        " adjustment_set (non-empty list) iff the successor is causal."
        " No other text.\n"
        f"{parent_block}\n"
        f"Feedback:\n{feedback_text}\n"
        f"Ancestors:\n{ancestors_render}"
    )


def _render_steps_prompt(
    parent_steps: list[str] | None, feedback_text: str, ancestors_render: str
) -> str:
    if parent_steps is None:
        parent_block = "No parent procedure: draft a NEW procedure from scratch."
    else:
        parent_block = "Parent procedure steps:\n" + "\n".join(
            f"{i + 1}. {s}" for i, s in enumerate(parent_steps)
        )
    return (
        "You are evolving a research procedure. Rewrite the procedure below"
        " into an improved step list that responds to the feedback. Reply"
        " with a single JSON object {\"steps\": [...]} with non-empty string"
        " steps. No other text.\n"
        f"{parent_block}\n"
        f"Feedback:\n{feedback_text}\n"
        f"Ancestors:\n{ancestors_render}"
    )


def _balanced_blocks(text: str, open_ch: str, close_ch: str) -> list[str]:
    blocks: list[str] = []
    depth = 0
    start: int | None = None
    for i, ch in enumerate(text):
        if ch == open_ch:
            if depth == 0:
                start = i
            depth += 1
        elif ch == close_ch and depth:
            depth -= 1
            if depth == 0 and start is not None:
                blocks.append(text[start : i + 1])
                start = None
    return blocks


def _json_candidates(reply: str) -> list[str]:
    """Tolerant extraction (explore `_parse_score_reply` shape): strip markdown
    fences, then offer the full text plus every balanced {...} / [...] block.
    """
    text = reply.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    return [text] + _balanced_blocks(text, "{", "}") + _balanced_blocks(text, "[", "]")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_hypothesis_body(parent: dict | None, parsed: Any) -> dict | None:
    """Overlay an LLM reply onto inherited parent fields; None if the result
    would not pass `propose_hypothesis` C6 validation. Unknown keys are
    dropped (the LLM never sets parent linkage, ids, or client state).
    """
    if not isinstance(parsed, dict):
        return None
    out: dict[str, Any] = {}
    for key in HYPOTHESIS_INHERITED_FIELDS:
        if parent is not None and parent.get(key) is not None:
            out[key] = parent[key]
    for key in HYPOTHESIS_REQUIRED_FIELDS + HYPOTHESIS_INHERITED_FIELDS:
        if key in parsed and parsed[key] is not None:
            out[key] = parsed[key]
    for key in (
        "mechanism",
        "expected_direction",
        "horizon",
        "scope",
        "falsifier",
        "research_family",
        "hypothesis_type",
        "universe",
    ):
        value = out.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
    if not _is_number(out.get("predicted_ic")) or not _is_number(out.get("predicted_sigma")):
        return None
    if out.get("hypothesis_type") not in ledger.HYPOTHESIS_TYPES:
        return None
    if out.get("hypothesis_type") == "causal":
        if not isinstance(out.get("dag"), dict) or not out["dag"]:
            return None
        if not isinstance(out.get("adjustment_set"), list) or not out["adjustment_set"]:
            return None
    required = {k: out[k] for k in HYPOTHESIS_REQUIRED_FIELDS}
    inherited = {k: out.get(k) for k in HYPOTHESIS_INHERITED_FIELDS}
    return {**required, **inherited}


def _validate_steps_body(parsed: Any) -> list[str] | None:
    if isinstance(parsed, dict):
        parsed = parsed.get("steps")
    if not isinstance(parsed, list) or not parsed:
        return None
    steps = []
    for item in parsed:
        if not isinstance(item, str) or not item.strip():
            return None
        steps.append(item.strip())
    return steps


def _fallback_hypothesis(parent: dict, rng: random.Random) -> dict:
    """One field-level edit preserving C6 validity, picked via the seeded rng.
    Returns a new dict; never mutates the parent.
    """
    op = HYPOTHESIS_FALLBACK_OPS[rng.randrange(len(HYPOTHESIS_FALLBACK_OPS))]
    out = {k: parent.get(k) for k in HYPOTHESIS_REQUIRED_FIELDS + HYPOTHESIS_INHERITED_FIELDS}
    horizon = str(out.get("horizon") or "")
    scope = str(out.get("scope") or "")
    if op == "tighten_horizon":
        out["horizon"] = horizon + " (tightened to the front half of the window)"
    elif op == "loosen_horizon":
        out["horizon"] = horizon + " (loosened by one rebalance lag)"
    elif op == "negate_direction":
        direction = str(out.get("expected_direction") or "")
        if direction == "positive":
            out["expected_direction"] = "negative"
        elif direction == "negative":
            out["expected_direction"] = "positive"
        else:
            out["expected_direction"] = f"opposite of {direction}"
        if _is_number(out.get("predicted_ic")):
            out["predicted_ic"] = -float(out["predicted_ic"])
        out["falsifier"] = (
            "Negated-direction variant: no effect opposite to the parent"
            f" expectation (parent falsifier: {out.get('falsifier')})"
        )
    else:  # narrow_scope
        out["scope"] = scope + " (narrowed to the top-ranked half)"
    idea = str(out.get("idea") or "")
    out["idea"] = f"{idea} [evolution variant: {op}]".strip()
    return out


def _fallback_steps(
    parent_steps: list[str], feedback: dict | str | None, rng: random.Random
) -> list[str]:
    """One syntactic edit picked via the seeded rng. Returns new steps; never
    mutates the parent list. Fixed rng consumption order: one draw for the
    operator, then operator-specific draws (`drop_step` only).
    """
    op = STEPS_FALLBACK_OPS[rng.randrange(len(STEPS_FALLBACK_OPS))]
    steps = list(parent_steps)
    if op == "drop_step":
        if len(steps) > 1:
            steps.pop(rng.randrange(len(steps)))
        return steps
    if op == "guard_step":
        sketches = feedback.get("failure_sketches", []) if isinstance(feedback, dict) else []
        classes = [
            str(s.get("error_class", "")).strip()
            for s in sketches
            if isinstance(s, dict) and str(s.get("error_class", "")).strip()
        ]
        if classes:
            error_class, count = Counter(classes).most_common(1)[0]
            steps.append(_guard_sentence(error_class, count))
        else:
            steps.append(GUARD_STEP_FALLBACK)
        return steps
    if op == "split":
        if len(steps) >= 2:
            return steps[: (len(steps) + 1) // 2]
        return steps
    # clarify: degraded paraphrase — normalize whitespace, rotate once.
    clarified = [" ".join(s.split()) for s in steps]
    if len(clarified) >= 2:
        clarified = clarified[1:] + clarified[:1]
    return clarified


def reflect_variation(
    *,
    parent: dict | list[str] | None,
    feedback: dict | str | None,
    ancestors_render: str,
    rng: random.Random,
    llm: LLMClient | None,
    schema: str,
) -> VariationProposal:
    """Propose a successor via one LLM call, or the deterministic fallback.

    `schema='hypothesis'` expects a parent hypothesis dict (or None for a
    new root) and returns a C6-complete hypothesis dict; `schema='skill_steps'`
    expects a parent step list and returns a rewritten step list. `feedback`
    is prior context (a feedback row, free text, or None); `ancestors_render`
    is the already-rendered ancestor trace (component E2 produces it).
    """
    if schema not in ("hypothesis", "skill_steps"):
        raise ValueError(f"unknown schema {schema!r}; want 'hypothesis'|'skill_steps'")
    if schema == "hypothesis":
        if parent is not None and not isinstance(parent, dict):
            raise ValueError("schema='hypothesis' needs a parent hypothesis dict or None")
        parent_steps: list[str] | None = None
        parent_hyp: dict | None = parent  # type: ignore[assignment]
    else:
        if parent is not None and (
            not isinstance(parent, list) or not all(isinstance(s, str) for s in parent)
        ):
            raise ValueError("schema='skill_steps' needs a parent step list or None")
        parent_steps = parent  # type: ignore[assignment]
        parent_hyp = None
    if feedback is not None and not isinstance(feedback, (dict, str)):
        raise ValueError("feedback must be a dict, str, or None")
    if not isinstance(ancestors_render, str):
        raise ValueError("ancestors_render must be a string")

    feedback_text = _feedback_text(feedback)
    if schema == "hypothesis":
        prompt = _render_hypothesis_prompt(parent_hyp, feedback_text, ancestors_render)
    else:
        prompt = _render_steps_prompt(parent_steps, feedback_text, ancestors_render)
    prompt_sha = hashlib.sha1(prompt.encode()).hexdigest()

    if llm is not None:
        try:
            reply = llm.complete(prompt)
            model_id = llm.model_id()
            body: dict | list[str] | None = None
            for candidate in _json_candidates(reply):
                try:
                    parsed = json.loads(candidate)
                except (json.JSONDecodeError, ValueError):
                    continue
                if schema == "hypothesis":
                    body = _validate_hypothesis_body(parent_hyp, parsed)
                else:
                    body = _validate_steps_body(parsed)
                if body is not None:
                    break
            if body is not None:
                return VariationProposal(body, model_id, prompt_sha, False)
        except Exception:
            pass  # every LLM failure degrades to the deterministic fallback

    if parent is None:
        raise ToolError(
            "cannot synthesize a root variation without an LLM: no parent to"
            " edit and no model available (unparsable replies degrade the same"
            " way). Provide an LLM or vary from an existing parent."
        )
    if schema == "hypothesis":
        assert isinstance(parent, dict)
        return VariationProposal(_fallback_hypothesis(parent, rng), None, prompt_sha, True)
    assert isinstance(parent, list)
    return VariationProposal(_fallback_steps(parent, feedback, rng), None, prompt_sha, True)
