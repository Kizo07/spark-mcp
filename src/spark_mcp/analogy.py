"""SME-lite structure-mapping analogy engine (plan §6.1 tool 4) + method
composition (plan §6.1 tool 3).

Mapping representation — the honest minimal version (structure-mapping without
a grammar engine; TRIZ operators and assumption-violation stay deferred, §20):

- A mech tag is a snake_case compound (`cross_section`) or a bare atom
  (`momentum`). Decomposition:
    * compound tag -> atoms split on `_`; the compound names a binary relation
      between its FIRST atom (subject) and LAST atom (object), e.g.
      `regime_detection` -> relation REGIME_DETECTION(subject=regime,
      object=detection). Atoms are opaque roles — the tag vocabulary is
      curated, not grammatical.
    * bare tag -> a unary predicate P(tag): "the method exhibits <tag>".
- Mapping a BASE method's predicate structure onto the TARGET's:
    * mapped_relations: compound tags shared by both sides. Each shared
      compound is one mapped relation (shared relational structure).
    * attribute_only_matches: shared bare tags, plus shared atoms that appear
      inside compounds on BOTH sides but whose compounds themselves are not
      shared (surface similarity without relational structure).
    * inconsistent_relations: a compound present on exactly one side whose
      subject (first atom) equals the subject of a compound on the other
      side — both posit structure over the same subject but disagree on the
      relation.

Systematicity score (Gentner-style; weights are the config-named constants):

    score = |M| / (|M| + |I| + 0.5 * |A|)      in [0, 1]

  Pure relational overlap scores 1.0; attribute-only overlap scores 0.0;
  each inconsistency drags the score down. A pair with no overlap at all is
  undefined -> 0.0 (nothing maps). cfg.research.analogy_systematicity_threshold
  (default 0.5): mappings below the threshold are REFUSED — the result
  records the refusal count, and when nothing passes the tool raises
  AnalogyMappingRefused (the engine refuses low-mapping analogies rather than
  returning them).

Analogy budget cap (§8.3): analogy searches are capped at
cfg.research.analogy_budget_cap (fraction, default 0.20) of total family
exploration trials, tracked by exploration_state.analogy_trials on a
dedicated aggregate row (ANALOGY_TRIALS_FAMILY — accounting, not a research
family, so family-posterior reads exclude it). With zero family trials the
cap cannot bind (nothing to budget against) and the call proceeds. Over the
cap the tool raises AnalogyBudgetExceeded, explaining the cap and that
earning more requires beating the base family hit rate (§8.3 wording).

Falsifiable financial translation: every mapping carries one sentence of the
form "if this analogy holds, then <target domain> should exhibit <testable
implication>". With an LLM the sentence is generated once and cached per
(base, target_class, model_id) in generated_text_cache (model version enters
the key, §4.3). The degraded no-LLM path is a documented template over the
shared tags: "methods sharing <tags> should co-move in <target problem>".

combine_methods composition grammar (2-3 candidates from >=2 method ids;
bases fetched PIT):
    * pipeline: ordered pair (A, B), A != B, bridged by shared mech-compound
      atoms — A's compound vocabulary overlaps B's (A's output feeds B's
      input stage). Best bridge first.
    * ensemble: two same-family methods run in parallel (first repeated
      family in input order).
    * residualizer: pair (A, B) with a visible graph edge between them whose
      predicate is DIAGNOSES or FAILS_WHEN — B is applied to A's residuals.
No candidate kind is forced: kinds without structural support are omitted,
and at least one candidate is returned whenever >=2 ids share any structure.
No returns are computed anywhere in this module (AGENTS.md rule 2).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from . import trajectory
from .config import Config
from .errors import ToolError
from .explore import problem_class
from .storage import Database
from .timeutil import now_iso, validate_iso

# Dedicated exploration_state row carrying the aggregate analogy-trial count
# (§8.3). Excluded from family-posterior reads — accounting, not a family.
ANALOGY_TRIALS_FAMILY = "__analogy__"

# Edge predicates that make (A, B) a residualizer composition (B diagnoses /
# fires on A's residual errors).
_RESIDUAL_PREDICATES = ("DIAGNOSES", "FAILS_WHEN")


class AnalogyMappingRefused(ToolError):
    """No base method's predicate structure mapped above the systematicity
    threshold (§6.1 tool 4): the engine refuses to return below-threshold
    analogies. The message carries the best score seen and the threshold."""


class AnalogyBudgetExceeded(ToolError):
    """§8.3 analogy budget cap: analogy_trials reached the cap fraction of
    family trials. Earning more analogy budget requires beating the base
    family hit rate."""


# ---------------------------------------------------------------------------
# predicate structure decomposition + systematicity


def _compound_parts(tag: str) -> tuple[str, str] | None:
    """(subject, object) = (first atom, last atom) for a compound tag."""
    atoms = [a for a in str(tag).split("_") if a]
    if len(atoms) < 2:
        return None
    return atoms[0], atoms[-1]


def _structure(tags: list[str]) -> dict:
    compounds = [t for t in tags if _compound_parts(t)]
    bare = [t for t in tags if not _compound_parts(t)]
    atoms: set[str] = set()
    for t in compounds:
        atoms.update(a for a in t.split("_") if a)
    return {"compounds": compounds, "bare": bare, "compound_atoms": atoms}


def systematicity(target_tags: list[str], base_tags: list[str]) -> dict:
    """The Gentner-style mapping of base predicate structure onto the target.

    score = |M| / (|M| + |I| + 0.5 * |A|) in [0, 1] — see the module
    docstring for the decomposition of M (mapped relations: shared compounds),
    A (attribute-only matches: shared bare tags + shared compound atoms whose
    compounds do not match) and I (inconsistent relations: one-sided compound
    whose subject collides with a compound subject on the other side).
    """
    t = _structure([str(x) for x in target_tags])
    b = _structure([str(x) for x in base_tags])
    t_comp, b_comp = set(t["compounds"]), set(b["compounds"])
    shared_compounds = sorted(t_comp & b_comp)
    shared_bare = sorted(set(t["bare"]) & set(b["bare"]))
    # Atoms locked inside NON-shared compounds on both sides are surface
    # similarity only (the relational structure around them differs).
    locked_t = {a for c in t_comp - b_comp for a in c.split("_") if a}
    locked_b = {a for c in b_comp - t_comp for a in c.split("_") if a}
    stray_atoms = sorted((locked_t & locked_b) | (locked_t & set(b["bare"]))
                         | (locked_b & set(t["bare"])))
    attribute_matches = shared_bare + stray_atoms

    def _heads(compounds: set[str]) -> set[str]:
        return {c.split("_", 1)[0] for c in compounds}

    mapped = [
        {
            "relation": c,
            "subject": _compound_parts(c)[0],  # type: ignore[index]
            "object": _compound_parts(c)[1],  # type: ignore[index]
        }
        for c in shared_compounds
    ]
    inconsistent = sorted(
        [c for c in t_comp - b_comp if c.split("_", 1)[0] in _heads(b_comp)]
        + [c for c in b_comp - t_comp if c.split("_", 1)[0] in _heads(t_comp)]
    )
    m, i, a = len(mapped), len(inconsistent), len(attribute_matches)
    score = m / (m + i + 0.5 * a) if (m + i + a) > 0 else 0.0
    return {
        "score": float(score),
        "mapped_relations": mapped,
        "attribute_matches": attribute_matches,
        "inconsistent_relations": inconsistent,
    }


# ---------------------------------------------------------------------------
# budget accounting (§8.3)


def _budget_state(db: Database, as_of: str) -> tuple[int, int]:
    """(total family trials, analogy trials) over PIT-visible exploration
    state. The sentinel row contributes analogy trials only (its trials
    counter stays 0, so it never inflates the cap denominator)."""
    rows = db.conn.execute(
        "SELECT trials, analogy_trials FROM exploration_state WHERE updated_at <= ?",
        (as_of,),
    ).fetchall()
    return (
        sum(int(r["trials"]) for r in rows),
        sum(int(r["analogy_trials"]) for r in rows),
    )


def _consume_analogy_budget(conn: Any, as_of: str) -> None:
    """Increment the aggregate analogy-trial counter inside the caller's
    transaction (§8.3)."""
    conn.execute(
        "INSERT INTO exploration_state (family, posterior, analogy_trials, updated_at)"
        " VALUES (?, '{\"alpha\": 1.0, \"beta\": 1.0}', 1, ?)"
        " ON CONFLICT(family) DO UPDATE SET"
        " analogy_trials = analogy_trials + 1, updated_at = excluded.updated_at",
        (ANALOGY_TRIALS_FAMILY, as_of),
    )


# ---------------------------------------------------------------------------
# generated text (LLM with cache; degraded templates documented)


def _cache_key(purpose: str, subject: str, context: str, model_id: str) -> str:
    raw = f"{purpose}|{subject}|{context}|{model_id}".encode()
    return hashlib.sha1(raw).hexdigest()  # noqa: S324 - dedupe key, not security


def _cached_text(
    db: Database, purpose: str, subject_key: str, context_key: str, model_id: str
) -> str | None:
    row = db.conn.execute(
        "SELECT text FROM generated_text_cache WHERE cache_key = ?",
        (_cache_key(purpose, subject_key, context_key, model_id),),
    ).fetchone()
    return str(row["text"]) if row else None


def _store_text(
    db: Database, purpose: str, subject_key: str, context_key: str, model_id: str, text: str
) -> None:
    db.execute(
        "INSERT OR REPLACE INTO generated_text_cache"
        " (cache_key, purpose, subject_key, context_key, model_id, text, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            _cache_key(purpose, subject_key, context_key, model_id),
            purpose,
            subject_key,
            context_key,
            model_id,
            text,
            now_iso(),
        ),
    )


def _degraded_translation(shared_tags: list[str], target_problem: str) -> str:
    """Documented degraded translation (no LLM): a template over the shared
    tags. It is falsifiable as stated — co-movement in the target is the
    testable implication — but carries no domain inference."""
    tags = ", ".join(shared_tags) if shared_tags else "no shared structure"
    return (
        "If this analogy holds, then methods sharing "
        f"{tags} should co-move in {target_problem}."
    )


def _falsifiable_translation(
    db: Database,
    llm: Any | None,
    base_node: dict,
    target_problem: str,
    shared_tags: list[str],
) -> tuple[str, str]:
    """One sentence 'if this analogy holds, then <target domain> should
    exhibit <testable implication>'. LLM path cached per (base, target_class,
    model_id); degraded path is the documented template."""
    pclass = problem_class(target_problem)
    if llm is None:
        return _degraded_translation(shared_tags, target_problem), "degraded_template"
    model_id = llm.model_id()
    cached = _cached_text(db, "analogy_translation", base_node["node_id"], pclass, model_id)
    if cached is not None:
        return cached, "llm_cached"
    prompt = (
        "You are mapping an analogy from a base research method onto a target"
        " problem. Write ONE sentence of the exact form 'If this analogy"
        " holds, then <target domain> should exhibit <testable financial"
        " implication>.' The implication must be falsifiable with market data.\n"
        f"Base method: {base_node['label']}\n"
        f"Shared mechanism structure: {', '.join(shared_tags) or 'none'}\n"
        f"Target problem: {target_problem}\n"
        "Reply with the single sentence only."
    )
    try:
        text = llm.complete(prompt).strip()
    except Exception:
        text = ""
    if not text:
        # LLM failure degrades to the documented template, never refuses.
        return _degraded_translation(shared_tags, target_problem), "degraded_template"
    _store_text(db, "analogy_translation", base_node["node_id"], pclass, model_id, text)
    return text, "llm"


# ---------------------------------------------------------------------------
# tool 4: find_analogies


def _record_effect(db: Database, tool: str, client_id: str, subject_id: str, result: dict) -> None:
    db.execute(
        "INSERT OR IGNORE INTO tool_effects (client_id, tool, subject_id, result, created_at)"
        " VALUES (?, ?, ?, ?, ?)",
        (client_id, tool, subject_id, json.dumps(result, sort_keys=True), now_iso()),
    )


def _replay_effect(db: Database, tool: str, client_id: str) -> dict | None:
    row = db.conn.execute(
        "SELECT result FROM tool_effects WHERE client_id = ? AND tool = ?", (client_id, tool)
    ).fetchone()
    return json.loads(row["result"]) if row else None


def find_analogies(
    db: Database,
    cfg: Config,
    target_problem: str,
    mech_tags: list[str],
    k: int = 5,
    as_of: str | None = None,
    llm: Any | None = None,
    client_id: str | None = None,
) -> dict:
    """Search the method graph for BASE methods whose mech-tag predicate
    structure maps onto the target's (§6.1 tool 4).

    Every returned mapping carries the base method, the systematicity score,
    the shared predicate structure, and a falsifiable financial translation
    (LLM-generated + cached, or the documented degraded template). Mappings
    below cfg.research.analogy_systematicity_threshold are refused (counted in
    `refusals`); when nothing passes, AnalogyMappingRefused is raised. Each
    call consumes one analogy trial against the §8.3 budget cap inside one
    transaction. Idempotent on client_id.
    """
    if client_id:
        existing = _replay_effect(db, "find_analogies", client_id)
        if existing is not None:
            return existing
    if k < 1:
        raise ValueError("k must be >= 1")
    as_of = validate_iso(as_of) if as_of else now_iso()
    target_tags = [str(t) for t in (mech_tags or [])]
    pclass = problem_class(target_problem)
    threshold = float(cfg.research.analogy_systematicity_threshold)

    rows = db.conn.execute(
        "SELECT node_id, label, family, mech_tags FROM graph_nodes"
        " WHERE type = 'technique' AND as_of <= ? ORDER BY node_id",
        (as_of,),
    ).fetchall()

    scored = []
    for r in rows:
        base_tags = [str(t) for t in json.loads(r["mech_tags"] or "[]")]
        mapping = systematicity(target_tags, base_tags)
        mapping["node"] = r
        scored.append(mapping)
    passing = sorted(
        (m for m in scored if m["score"] >= threshold),
        key=lambda m: (-m["score"], m["node"]["node_id"]),
    )
    refusals = len(scored) - len(passing)
    if not passing:
        best = max((m["score"] for m in scored), default=0.0)
        raise AnalogyMappingRefused(
            f"analogy refused: no base method mapped above the systematicity"
            f" threshold {threshold} (best score {best:.3f}, {len(scored)} bases"
            f" searched, {refusals} refused) — the engine does not return"
            " below-threshold analogies (plan §6.1 tool 4)"
        )

    total_trials, analogy_trials = _budget_state(db, as_of)
    cap = float(cfg.research.analogy_budget_cap) * total_trials
    if total_trials > 0 and analogy_trials >= cap:
        raise AnalogyBudgetExceeded(
            f"analogy budget cap reached: {analogy_trials} analogy trials of"
            f" {cap:g} allowed ({float(cfg.research.analogy_budget_cap):.0%} of"
            f" {total_trials} family trials). Earning more analogy budget"
            " requires beating the base family hit rate (plan §8.3)."
        )

    mappings = []
    for m in passing[:k]:
        node = m["node"]
        shared = sorted({r["relation"] for r in m["mapped_relations"]})
        translation, basis = _falsifiable_translation(
            db, llm, dict(node), target_problem, shared
        )
        mappings.append(
            {
                "base_method_id": node["node_id"],
                "label": node["label"],
                "family": node["family"],
                "systematicity": m["score"],
                "shared_predicates": shared,
                "mapped_relations": m["mapped_relations"],
                "attribute_matches": m["attribute_matches"],
                "inconsistent_relations": m["inconsistent_relations"],
                "falsifiable_translation": translation,
                "translation_basis": basis,
            }
        )

    with db.transaction():
        _consume_analogy_budget(db.conn, as_of)
        if llm is not None:
            from . import budgets

            budgets.record_spend(
                db, cfg, family=None, source="find_analogies",
                usd=None, tokens=0, estimated=True, run_id=None,
            )
        trajectory.record(
            db,
            f"analogy:{pclass}",
            "spark",
            "tool_call",
            {
                "tool": "find_analogies",
                "target_problem": target_problem,
                "k": k,
                "as_of": as_of,
                "mappings": [m["base_method_id"] for m in mappings],
                "refusals": refusals,
                "analogy_trials": analogy_trials + 1,
                "client_id": client_id,
            },
        )
    result = {
        "target_problem": target_problem,
        "problem_class": pclass,
        "mech_tags": target_tags,
        "k": k,
        "as_of": as_of,
        "systematicity_threshold": threshold,
        "mappings": mappings,
        "refusals": refusals,
        "budget": {
            "analogy_trials": analogy_trials + 1,
            "cap": cap,
            "family_trials": total_trials,
        },
    }
    if client_id:
        _record_effect(db, "find_analogies", client_id, f"analogy:{pclass}", result)
    return result


# ---------------------------------------------------------------------------
# tool 3: combine_methods


def _compound_atoms(tags: list[str]) -> set[str]:
    atoms: set[str] = set()
    for t in tags:
        if _compound_parts(t):
            atoms.update(a for a in t.split("_") if a)
    return atoms


def _visible_edge(
    db: Database, a: str, b: str, as_of: str
) -> dict | None:
    row = db.conn.execute(
        "SELECT predicate, src, dst FROM graph_edges"
        " WHERE ((src = ? AND dst = ?) OR (src = ? AND dst = ?))"
        " AND predicate IN ('DIAGNOSES', 'FAILS_WHEN')"
        " AND valid_from <= ? AND (valid_to IS NULL OR ? < valid_to) AND as_of <= ?"
        " ORDER BY predicate LIMIT 1",
        (a, b, b, a, as_of, as_of, as_of),
    ).fetchone()
    return dict(row) if row else None


def _combine_rationale(
    db: Database,
    llm: Any | None,
    kind: str,
    subject_key: str,
    target_problem: str,
    degraded: str,
) -> tuple[str, str]:
    """One-line rationale: LLM when present (cached per (methods, kind,
    target_class, model_id)), degraded template otherwise — never refuses."""
    if llm is None:
        return degraded, "degraded_template"
    pclass = problem_class(target_problem)
    model_id = llm.model_id()
    cached = _cached_text(db, "combine_rationale", subject_key, f"{kind}:{pclass}", model_id)
    if cached is not None:
        return cached, "llm_cached"
    prompt = (
        "In one line, explain why combining these research methods as a"
        f" {kind} addresses the target problem. Name the shared mechanism.\n"
        f"Combination kind: {kind}\nSubject: {subject_key}\n"
        f"Target problem: {target_problem}\nReply with one line only."
    )
    try:
        text = llm.complete(prompt).strip()
    except Exception:
        text = ""
    if not text:
        return degraded, "degraded_template"
    _store_text(db, "combine_rationale", subject_key, f"{kind}:{pclass}", model_id, text)
    return text, "llm"


def combine_methods(
    db: Database,
    cfg: Config,
    method_ids: list[str],
    target_problem: str,
    as_of: str | None = None,
    client_id: str | None = None,
    llm: Any | None = None,
) -> dict:
    """Compose 2-3 candidate compositions from method graph nodes (§6.1 tool
    3). Composition grammar: pipeline (A's mech-compound output feeds B's
    input via shared compound atoms), ensemble (same-family parallel), and
    residualizer (B applied to A's residual errors, evidenced by a visible
    DIAGNOSES / FAILS_WHEN edge). Each candidate carries structural-fit notes
    (shared predicates, family bridge, expected interactions), the composed
    mech_tags set, and a one-line rationale (LLM-cached or degraded
    template). No returns are computed. Idempotent on client_id."""
    del cfg  # composition grammar is structural; the knobs live in ResearchConfig readers
    if client_id:
        existing = _replay_effect(db, "combine_methods", client_id)
        if existing is not None:
            return existing
    as_of = validate_iso(as_of) if as_of else now_iso()
    ids = [str(m) for m in (method_ids or [])]
    if len(ids) < 2:
        raise ToolError(
            f"combine_methods needs >= 2 method ids, got {len(ids)} (plan §6.1 tool 3)"
        )
    marks = ",".join("?" for _ in ids)
    rows = db.conn.execute(
        f"SELECT node_id, label, family, mech_tags FROM graph_nodes"
        f" WHERE node_id IN ({marks}) AND as_of <= ?",
        (*ids, as_of),
    ).fetchall()
    by_id = {r["node_id"]: r for r in rows}
    missing = [m for m in ids if m not in by_id]
    if missing:
        raise ToolError(f"unknown or not-yet-visible method_id(s): {missing}")
    nodes = [dict(by_id[m], mech_tags=json.loads(by_id[m]["mech_tags"] or "[]")) for m in ids]
    parsed = {n["node_id"]: n for n in nodes}

    compositions: list[dict] = []

    # -- pipeline: best compound-atom bridge between an ordered pair ----------
    best_pipe: tuple[int, tuple[str, str], list[str]] | None = None
    for i, a in enumerate(nodes):
        for j, b in enumerate(nodes):
            if i == j:
                continue
            bridge = _compound_atoms(a["mech_tags"]) & _compound_atoms(b["mech_tags"])
            if not bridge:
                continue
            cand = (len(bridge), (a["node_id"], b["node_id"]), sorted(bridge))
            if best_pipe is None or cand > best_pipe:
                best_pipe = cand
    if best_pipe is not None:
        _, (a_id, b_id), bridge = best_pipe
        a = parsed[a_id]
        b = parsed[b_id]
        subject_key = f"{a_id},{b_id}"
        degraded = (
            f"Pipeline {a['label']} -> {b['label']}: shared mechanism vocabulary"
            f" ({', '.join(sorted(bridge))}) feeds one method's output into the"
            f" other's input stage for {target_problem}."
        )
        rationale, basis = _combine_rationale(
            db, llm, "pipeline", f"pipeline:{subject_key}", target_problem, degraded
        )
        compositions.append(
            {
                "composition_id": "comp-"
                + _cache_key("pipeline", subject_key, target_problem, "v1")[:8],
                "kind": "pipeline",
                "methods": [
                    {"method_id": a_id, "label": a["label"]},
                    {"method_id": b_id, "label": b["label"]},
                ],
                "structural_fit_notes": {
                    "shared_predicates": sorted(bridge),
                    "family_bridge": (
                        f"same family ({a['family']})" if a["family"] == b["family"]
                        else f"cross-family ({a['family']} -> {b['family']})"
                    ),
                    "expected_interactions": (
                        f"{a['label']}'s {', '.join(sorted(bridge))} stage output"
                        f" conditions {b['label']}'s input; watch error compounding"
                        " across the pipeline."
                    ),
                },
                "composed_mech_tags": sorted(
                    set(a["mech_tags"]) | set(b["mech_tags"])
                ),
                "rationale": rationale,
                "rationale_basis": basis,
            }
        )

    # -- ensemble: first repeated family in input order -----------------------
    seen_family: dict[str, dict] = {}
    ensemble_pair: tuple[dict, dict] | None = None
    for n in nodes:
        fam = n["family"]
        if fam and fam in seen_family:
            ensemble_pair = (seen_family[fam], n)
            break
        if fam:
            seen_family[fam] = n
    if ensemble_pair is not None:
        a, b = ensemble_pair
        subject_key = f"{a['node_id']},{b['node_id']}"
        shared = sorted(set(a["mech_tags"]) & set(b["mech_tags"]))
        degraded = (
            f"Ensemble {a['label']} || {b['label']}: same-family parallel votes"
            f" diversify estimation risk for {target_problem}."
        )
        rationale, basis = _combine_rationale(
            db, llm, "ensemble", f"ensemble:{subject_key}", target_problem, degraded
        )
        compositions.append(
            {
                "composition_id": "comp-"
                + _cache_key("ensemble", subject_key, target_problem, "v1")[:8],
                "kind": "ensemble",
                "methods": [
                    {"method_id": a["node_id"], "label": a["label"]},
                    {"method_id": b["node_id"], "label": b["label"]},
                ],
                "structural_fit_notes": {
                    "shared_predicates": shared,
                    "family_bridge": f"same family ({a['family']})",
                    "expected_interactions": (
                        "Parallel same-family estimates combined by vote/average;"
                        " shared failure modes (family-level crowding) correlate the votes."
                    ),
                },
                "composed_mech_tags": sorted(
                    set(a["mech_tags"]) | set(b["mech_tags"])
                ),
                "rationale": rationale,
                "rationale_basis": basis,
            }
        )

    # -- residualizer: DIAGNOSES / FAILS_WHEN edge between the given methods ---
    resid: tuple[dict, dict, dict] | None = None
    for i, a in enumerate(nodes):
        for b in nodes[i + 1 :]:
            edge = _visible_edge(db, a["node_id"], b["node_id"], as_of)
            if edge is not None:
                diagnoser = a if edge["src"] == a["node_id"] else b
                host = b if diagnoser is a else a
                resid = (host, diagnoser, edge)
                break
        if resid is not None:
            break
    if resid is not None:
        host, diagnoser, edge = resid
        subject_key = f"{host['node_id']},{diagnoser['node_id']}"
        degraded = (
            f"Residualizer {diagnoser['label']} applied to {host['label']}'s"
            f" residuals: the visible {edge['predicate']} edge marks the"
            " diagnoser as a repair stage for the host's errors."
        )
        rationale, basis = _combine_rationale(
            db, llm, "residualizer", f"residualizer:{subject_key}", target_problem, degraded
        )
        compositions.append(
            {
                "composition_id": "comp-"
                + _cache_key("residualizer", subject_key, target_problem, "v1")[:8],
                "kind": "residualizer",
                "methods": [
                    {"method_id": host["node_id"], "label": host["label"]},
                    {"method_id": diagnoser["node_id"], "label": diagnoser["label"]},
                ],
                "structural_fit_notes": {
                    "shared_predicates": sorted(
                        set(host["mech_tags"]) & set(diagnoser["mech_tags"])
                    ),
                    "family_bridge": (
                        f"same family ({host['family']})"
                        if host["family"] == diagnoser["family"]
                        else f"cross-family ({host['family']} <- {diagnoser['family']})"
                    ),
                    "expected_interactions": (
                        f"{host['label']} runs first; {diagnoser['label']} models"
                        f" its residual errors (graph edge: {edge['predicate']})."
                        " Risk: the diagnoser can overfit the host's noise."
                    ),
                },
                "composed_mech_tags": sorted(
                    set(host["mech_tags"]) | set(diagnoser["mech_tags"])
                ),
                "rationale": rationale,
                "rationale_basis": basis,
            }
        )

    if not compositions:
        raise ToolError(
            "no composable structure: the given methods share no mech-compound"
            " vocabulary, no family, and no DIAGNOSES/FAILS_WHEN edge (plan"
            " §6.1 tool 3 composition grammar)"
        )
    trajectory.record(
        db,
        f"combine:{problem_class(target_problem)}",
        "spark",
        "tool_call",
        {
            "tool": "combine_methods",
            "method_ids": ids,
            "kinds": [c["kind"] for c in compositions],
            "as_of": as_of,
            "client_id": client_id,
        },
    )
    result = {
        "target_problem": target_problem,
        "as_of": as_of,
        "method_ids": ids,
        "compositions": compositions,
        "composition_count": len(compositions),
    }
    if client_id:
        _record_effect(
            db,
            "combine_methods",
            client_id,
            result["compositions"][0]["composition_id"],
            result,
        )
    return result


# re-export for tests / callers that want the decomposition helper
__all__ = [
    "ANALOGY_TRIALS_FAMILY",
    "AnalogyBudgetExceeded",
    "AnalogyMappingRefused",
    "combine_methods",
    "find_analogies",
    "systematicity",
]
