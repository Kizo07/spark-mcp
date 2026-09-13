"""explore_methods — candidate scoring & selection (plan §6.1 tool 2, §8.1/§8.2).

    score = w_r·relevance + w_sf·structural_fit + w_n·novelty
          + w_e·exploration_bonus + w_p·prior_success

relevance/structural_fit are LLM-scored only when a key (LLM client) is
configured; otherwise the DeterministicScorer (lexical + graph-proximity)
runs. LLM scores are cached per (method, problem_class, model_id).
exploration_bonus is the Phase 2 uniform constant (bandit posteriors arrive
with the QD archive in Phase 5); prior_success is the family deflated hit
rate (Beta-posterior mean, cfg.research.bandit_prior) from PIT episodes,
default 0.5 with no data. Selection = softmax(temperature by mode) sampled
without replacement into a pool, then MMR(λ) diversification → k
suggestions, each carrying its full decomposition + selection probability.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from . import graph, novelty, retrieval, trajectory
from .config import Config
from .llm import split_model_id
from .storage import Database
from .timeutil import now_iso, validate_iso

TERMINAL_SUCCESS = ("success", "validated")
TERMINAL_FAILURE = ("failure", "failed")


class DeterministicScorer:
    """Degraded-mode scorer (§8.1): IDF-weighted lexical relevance of the node
    text (rare problem tokens like 'causal' outweigh generic ones like
    'returns'), structural fit from graph proximity — neighbors that are
    themselves relevant to the problem, Laplace-smoothed."""

    def __init__(self, db: Database, as_of: str):
        import math
        from collections import Counter

        rows = db.conn.execute(
            "SELECT label, summary, mech_tags FROM graph_nodes WHERE as_of <= ?", (as_of,)
        ).fetchall()
        docs = [
            retrieval.tokens(" ".join([r["label"], r["summary"] or "", r["mech_tags"]]))
            for r in rows
        ]
        n = max(len(docs), 1)
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(d))
        self._idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}

    def _recall(self, problem_tokens: set[str], text_tokens: set[str]) -> float:
        """IDF-weighted share of the problem's vocabulary covered by the text."""
        shared = problem_tokens & text_tokens
        if not shared or not problem_tokens:
            return 0.0
        denom = sum(self._idf.get(t, 1.0) for t in problem_tokens)
        if denom <= 0:
            return 0.0
        return sum(self._idf.get(t, 1.0) for t in shared) / denom

    def relevance(self, problem_tokens: set[str], node: dict) -> float:
        node_tokens = retrieval.tokens(
            " ".join(
                [node["label"], node.get("summary") or "", " ".join(node["mech_tags"])]
            )
        )
        return self._recall(problem_tokens, node_tokens)

    def structural_fit(
        self, db: Database, node: dict, problem_tokens: set[str], as_of: str
    ) -> float:
        rows = db.conn.execute(
            "SELECT n.label AS label, n.summary AS summary,"
            " n.mech_tags AS mech_tags FROM graph_edges e"
            " JOIN graph_nodes n ON n.node_id = e.dst"
            " WHERE e.src = ? AND e.valid_from <= ?"
            " AND (e.valid_to IS NULL OR ? < e.valid_to) AND e.as_of <= ?",
            (node["node_id"], as_of, as_of, as_of),
        ).fetchall()
        if not rows:
            return 0.5  # no graph grounding: neutral, not zero
        support = 0.0
        for r in rows:
            toks = retrieval.tokens(" ".join([r["label"], r["summary"] or ""]))
            support += self._recall(problem_tokens, toks)
        return (1.0 + support) / (2.0 + len(rows))


def problem_class(problem: str) -> str:
    """Deterministic class key for a problem statement (cache + reporting)."""
    toks = sorted(retrieval.tokens(problem))
    return hashlib.sha1(" ".join(toks).encode("utf-8")).hexdigest()[:16]


def _llm_scores(
    db: Database,
    llm: Any,
    node: dict,
    problem: str,
    pclass: str,
) -> tuple[float, float] | None:
    """LLM-scored relevance/structural_fit, cached per (method, problem_class,
    model_id). None on any failure (caller falls back to deterministic)."""
    model_id = llm.model_id()
    key = hashlib.sha1(
        f"{node['node_id']}|{pclass}|{model_id}".encode()
    ).hexdigest()
    row = db.conn.execute(
        "SELECT relevance, structural_fit FROM score_cache WHERE cache_key = ?", (key,)
    ).fetchone()
    if row is not None:
        return float(row["relevance"]), float(row["structural_fit"])
    prompt = (
        "Score how well this research method fits the problem. Reply with"
        ' JSON {"relevance": 0..1, "structural_fit": 0..1} only.\n'
        f"Problem: {problem}\nMethod: {node['label']}\n"
        f"Mechanism tags: {', '.join(node['mech_tags'])}\n"
        f"Summary: {node.get('summary') or 'n/a'}"
    )
    try:
        reply = llm.complete(prompt, json_schema=None)
        parsed = json.loads(reply)
        rel = float(np.clip(float(parsed["relevance"]), 0.0, 1.0))
        fit = float(np.clip(float(parsed["structural_fit"]), 0.0, 1.0))
    except Exception:
        return None
    db.execute(
        "INSERT OR REPLACE INTO score_cache"
        " (cache_key, method_id, problem_class, model_id, relevance,"
        " structural_fit, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (key, node["node_id"], pclass, model_id, rel, fit, now_iso()),
    )
    return rel, fit


def _family_posteriors(
    db: Database, cfg: Config, as_of: str
) -> dict[str, dict[str, float]]:
    """Family deflated hit rates from PIT episodes: Beta(a,b) posterior mean."""
    a, b = (list(cfg.research.bandit_prior) + [1.0, 1.0])[:2]
    rows = db.conn.execute(
        "SELECT research_family AS family,"
        " SUM(CASE WHEN outcome IN ('success','validated') THEN 1 ELSE 0 END) AS wins,"
        " SUM(CASE WHEN outcome IN ('success','validated','failure','failed')"
        " THEN 1 ELSE 0 END) AS trials"
        " FROM episodes WHERE as_of <= ? GROUP BY research_family",
        (as_of,),
    ).fetchall()
    out = {}
    for r in rows:
        trials, wins = int(r["trials"]), int(r["wins"])
        out[r["family"]] = {
            "prior_success": (wins + a) / (trials + a + b),
            "trials": trials,
            "wins": wins,
        }
    return out


def _mmr_select(
    pool: list[dict],
    k: int,
    lam: float,
    sim_matrix: np.ndarray,
) -> list[dict]:
    """Greedy MMR: argmax λ·score − (1−λ)·max_sim_to_selected (§8.2)."""
    selected: list[int] = []
    remaining = list(range(len(pool)))
    while remaining and len(selected) < k:
        best_i, best_val = None, -np.inf
        for i in remaining:
            redundancy = (
                max((sim_matrix[i, j] for j in selected), default=0.0)
            )
            val = lam * pool[i]["score"] - (1.0 - lam) * redundancy
            if val > best_val or (val == best_val and best_i is not None and i < best_i):
                best_i, best_val = i, val
        selected.append(best_i)
        remaining.remove(best_i)
    return [pool[i] for i in selected]


def _similarity_matrix(
    db: Database, pool: list[dict], embedder: Any | None, as_of: str
) -> np.ndarray:
    """Pairwise similarity for MMR: embedding cosine when available, else
    lexical Jaccard of node text. Degraded mode stays deterministic."""
    n = len(pool)
    sim = np.zeros((n, n), dtype=np.float64)
    vecs: list[np.ndarray | None] = [None] * n
    if embedder is not None:
        model_key = split_model_id(embedder.model_id())
        expected_shape = np.asarray(embedder.embed([""])).shape
        for idx, node in enumerate(pool):
            if not db.vec_table_exists():
                break
            rows = db.conn.execute(
                "SELECT embedding, model, model_version FROM vec_embeddings"
                " WHERE subject_type = 'graph_node' AND subject_id = ?",
                (node["method_id"],),
            ).fetchall()
            for r in rows:
                if (r["model"], r["model_version"]) != model_key:
                    continue
                vec = np.frombuffer(bytes(r["embedding"]), dtype=np.float32)
                if vec.shape == expected_shape:
                    vecs[idx] = vec
                    break
    texts = [
        retrieval.tokens(" ".join([p["label"], p.get("summary") or ""]))
        for p in pool
    ]
    for i in range(n):
        for j in range(i + 1, n):
            if vecs[i] is not None and vecs[j] is not None:
                s = retrieval._cosine(vecs[i], vecs[j])
            else:
                s = retrieval.jaccard(texts[i], texts[j])
            sim[i, j] = sim[j, i] = s
    return sim


def explore_methods(
    db: Database,
    cfg: Config,
    problem: str,
    k: int = 5,
    mode: str = "balanced",
    as_of: str | None = None,
    embedder: Any | None = None,
    llm: Any | None = None,
    seed: int | None = None,
) -> dict:
    """Score, sample and diversify method candidates for a problem."""
    as_of = validate_iso(as_of) if as_of else now_iso()
    if k < 1:
        raise ValueError("k must be >= 1")
    temperature = cfg.research.temperatures.get(mode)
    if temperature is None:
        raise ValueError(
            f"unknown mode {mode!r}; expected one of {sorted(cfg.research.temperatures)}"
        )
    lam = float(cfg.research.mmr_lambda)
    weights = (list(cfg.research.scoring_weights) + [0.0] * 5)[:5]
    pclass = problem_class(problem)

    candidates = retrieval.hybrid_recall(
        db,
        problem,
        as_of,
        embedder=embedder,
        k_vec=cfg.research.top_k,
        k_fts=cfg.research.top_k,
        rrf_k=cfg.research.rrf_k,
        node_types=("technique",),
    )
    if not candidates:  # unmatchable problem text: fall back to all visible techniques
        rows = db.conn.execute(
            "SELECT * FROM graph_nodes WHERE type = 'technique' AND as_of <= ?"
            " ORDER BY node_id",
            (as_of,),
        ).fetchall()
        candidates = [dict(r) for r in rows]
        for c in candidates:
            c["mech_tags"] = json.loads(c["mech_tags"])

    posteriors = _family_posteriors(db, cfg, as_of)
    scorer = DeterministicScorer(db, as_of)
    problem_tokens = retrieval.tokens(problem)
    exploration_bonus = float(cfg.research.exploration_bonus_uniform)

    scored = []
    for node in candidates:
        llm_scores = _llm_scores(db, llm, node, problem, pclass) if llm else None
        if llm_scores is not None:
            relevance, structural_fit = llm_scores
        else:
            relevance = scorer.relevance(problem_tokens, node)
            structural_fit = scorer.structural_fit(db, node, problem_tokens, as_of)
        fam_stats = posteriors.get(node.get("family") or "", {})
        prior_success = float(fam_stats.get("prior_success", 0.5))
        nov = novelty.check_novelty(
            db,
            cfg,
            graph.node_text(node),
            family=node.get("family"),
            as_of=as_of,
            embedder=embedder,
            exclude_subject_ids={node["node_id"]},
        )
        decomposition = {
            "relevance": relevance,
            "structural_fit": structural_fit,
            "novelty": nov["N"],
            "exploration_bonus": exploration_bonus,
            "prior_success": prior_success,
        }
        total = float(
            sum(w * v for w, v in zip(weights, decomposition.values(), strict=True))
        )
        scored.append(
            {
                "method_id": node["node_id"],
                "label": node["label"],
                "family": node.get("family"),
                "mech_tags": node["mech_tags"],
                "score": total,
                "decomposition": decomposition,
                "novelty": {
                    "N": nov["N"],
                    "max_cos": nov["max_cos"],
                    "structural_overlap": nov["structural_overlap"],
                    "cell_emptiness": nov["cell_emptiness"],
                },
                "selection_probability": None,  # filled after softmax
                "sources": node.get("sources", []),
                "prior_success_family": {
                    "trials": int(fam_stats.get("trials", 0)),
                    "wins": int(fam_stats.get("wins", 0)),
                },
            }
        )

    if not scored:
        return {
            "problem": problem,
            "problem_class": pclass,
            "mode": mode,
            "scoring_mode": "llm" if llm else "deterministic",
            "temperature": temperature,
            "mmr_lambda": lam,
            "k": k,
            "as_of": as_of,
            "candidate_count": 0,
            "prior_trials": {},
            "families_tried": [],
            "suggestions": [],
            "suggestion_ids": [],
        }

    logits = np.array([s["score"] for s in scored], dtype=np.float64) / max(temperature, 1e-9)
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    for s, p in zip(scored, probs, strict=True):
        s["selection_probability"] = float(p)

    rng = np.random.default_rng(seed)
    n = len(scored)
    pool_size = min(n, max(2 * k, k + 5))
    if n <= pool_size:
        pool = list(scored)
    else:
        drawn = rng.choice(n, size=pool_size, replace=False, p=probs)
        pool = [scored[int(i)] for i in drawn]

    sim_matrix = _similarity_matrix(db, pool, embedder, as_of)
    suggestions = _mmr_select(pool, k, lam, sim_matrix)

    result = {
        "problem": problem,
        "problem_class": pclass,
        "mode": mode,
        "scoring_mode": "llm" if llm else "deterministic",
        "temperature": temperature,
        "mmr_lambda": lam,
        "k": k,
        "as_of": as_of,
        "candidate_count": n,
        "prior_trials": {f: v["trials"] for f, v in posteriors.items()},
        "families_tried": sorted(posteriors),
        "candidates": [
            {
                "method_id": s["method_id"],
                "label": s["label"],
                "family": s["family"],
                "score": s["score"],
                "selection_probability": s["selection_probability"],
            }
            for s in sorted(scored, key=lambda s: (-s["score"], s["method_id"]))
        ],
        "suggestions": suggestions,
        "suggestion_ids": [s["method_id"] for s in suggestions],
    }
    trajectory.record(
        db,
        f"explore:{pclass}",
        "spark",
        "tool_call",
        {
            "tool": "explore_methods",
            "mode": mode,
            "k": k,
            "candidate_count": n,
            "suggestion_ids": result["suggestion_ids"],
            "as_of": as_of,
        },
    )
    return result
