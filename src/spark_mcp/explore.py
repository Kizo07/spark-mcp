"""explore_methods — candidate scoring & selection (plan §6.1 tool 2, §8.1/§8.2).

    score = w_r·relevance + w_sf·structural_fit + w_n·novelty
          + w_e·exploration_bonus + w_p·prior_success

relevance/structural_fit are LLM-scored only when a key (LLM client) is
configured; otherwise the DeterministicScorer runs: IDF-weighted lexical
relevance (best of diluted full-text recall and concentrated mech-tag
evidence), max-normalized across candidates and scaled by a family factor —
the problem's match against each family's graph vocabulary (family names,
ASSUMES/FAILS_WHEN concept atoms, whole mechanism-tag compounds with
affix/part/middle-word discounts, size-normalized so big families can't win
by bulk). The factor applies softly to structural_fit too (edges are the
legitimate cross-family composition route). LLM scores are cached per
(method, problem_class, model_id). exploration_bonus is the Phase 2 uniform
constant (bandit posteriors arrive with the QD archive in Phase 5);
prior_success is the family deflated hit rate (Beta-posterior mean,
cfg.research.bandit_prior) from PIT episodes, default 0.5 with no data.
Selection (§8.2): candidates within selection_score_band of the best score
form the defensible set; the pool is drawn from it with a flattened
(higher-temperature) softmax, then MMR(λ) diversification → k suggestions,
each carrying its full decomposition + selection probability (the reported
selection_probability is always the unflattened softmax).
"""

from __future__ import annotations

import hashlib
import json
import re
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
    text (exact tokens plus discounted affix-bounded derivational matches),
    scaled by a family factor — the problem's match against each family's
    graph vocabulary, max-normalized with a configured floor so cross-family
    composition survives. Family vocabulary: family names + ASSUMES /
    FAILS_WHEN concept labels contribute natural-language ATOMS (middle words
    of compound concepts discounted); mechanism tags stay whole COMPOUNDS
    (inner parts match only at a part-of-compound discount, so
    'normal_returns' does not make 'returns' allocation vocabulary). A
    vocab-space IDF over the family "documents" discounts terms many families
    share ('estimation', 'error'). Structural fit comes from graph proximity;
    the family factor applies to the node's own lexical relevance only, so
    the two channels compose without double-counting."""

    # Derivational affixes bounding a substring match (noise ~ denoising,
    # robust ~ robustness, shift ~ regime_shift). Anything not on these lists
    # must be an exact token match — free substring matching lets 'estimate'
    # match 'estimation_accuracy' and family weights collapse toward 1.
    _AFFIX_PRE = frozenset(
        {"de", "re", "un", "in", "im", "over", "under", "pre", "post", "mis",
         "non", "multi", "counter"}
    )
    _AFFIX_POST = frozenset(
        {"ing", "ed", "es", "s", "er", "est", "ly", "ion", "tion", "ation",
         "sion", "ment", "ness", "ity", "al", "ic", "ive", "ous", "ful",
         "able", "ible", "ize", "ise", "ify"}
    )

    def __init__(self, db: Database, as_of: str, cfg: Config):
        import json as _json
        import math
        from collections import Counter

        self._morph_min = max(int(cfg.research.family_morph_min_len), 3)
        self._part_discount = float(cfg.research.family_part_discount)
        self._affix_discount = float(cfg.research.family_affix_discount)
        self._middle_discount = float(cfg.research.family_middle_discount)
        self._cfg = cfg

        rows = db.conn.execute(
            "SELECT label, summary, mech_tags, family FROM graph_nodes"
            " WHERE as_of <= ? AND type = 'technique'",
            (as_of,),
        ).fetchall()

        # node-text IDF (drives relevance / structural_fit recall)
        docs = [
            retrieval.tokens(" ".join([r["label"], r["summary"] or "", r["mech_tags"]]))
            for r in rows
        ]
        n = max(len(docs), 1)
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(d))
        self._idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}

        # family vocabulary (drives family_weights)
        self._atoms: dict[str, set[str]] = {}
        self._compounds: dict[str, set[str]] = {}
        self._atom_middle: dict[str, set[str]] = {}  # atoms from concept middles

        def _atoms(family: str) -> set[str]:
            return self._atoms.setdefault(family, set())

        def _add_compound(family: str, term: str) -> None:
            self._compounds.setdefault(family, set()).add(term)

        for r in rows:
            family = r["family"] or ""
            _atoms(family).update(retrieval.tokens(family))  # family name
            for tag in _json.loads(r["mech_tags"] or "[]"):
                tag = str(tag)
                if "_" in tag:
                    _add_compound(family, tag)  # inner parts are not atoms
                else:
                    _atoms(family).add(tag)  # bare tag = natural word
            for word in retrieval.TOKEN_RE.findall(r["summary"] or ""):
                if "_" in word:
                    _add_compound(family, word)
        for r in db.conn.execute(
            "SELECT src_node.family AS family, dst.label AS concept"
            " FROM graph_edges e"
            " JOIN graph_nodes src_node ON src_node.node_id = e.src"
            " JOIN graph_nodes dst ON dst.node_id = e.dst"
            " WHERE e.predicate IN ('ASSUMES', 'FAILS_WHEN')"
            " AND e.valid_from <= ? AND (e.valid_to IS NULL OR ? < e.valid_to)"
            " AND e.as_of <= ?",
            (as_of, as_of, as_of),
        ).fetchall():
            family = r["family"] or ""
            words = retrieval.tokens(r["concept"])
            _atoms(family).update(words)
            parts = [w for w in retrieval.TOKEN_RE.findall(r["concept"]) if w in words]
            for w in parts[1:-1]:  # middle words of multi-word concepts
                self._atom_middle.setdefault(family, set()).add(w)

        # vocab-space IDF: each family's vocab (atoms + compounds) is one doc,
        # so terms many families share are weakly discriminative
        fam_count = max(len(self._atoms), 1)
        vocab_df: Counter[str] = Counter()
        for fam in self._atoms:
            for t in self._atoms[fam] | self._compounds.get(fam, set()):
                vocab_df[t] += 1
        self._vidf = {
            t: math.log((fam_count + 1) / (c + 1)) + 1.0 for t, c in vocab_df.items()
        }

    def _affix_bounded(self, short: str, long: str) -> bool:
        """True when `short` sits inside `long` bounded only by derivational
        affixes (plus optional underscores), in either direction."""
        if len(short) > len(long):
            short, long = long, short
        start = 0
        while True:
            i = long.find(short, start)
            if i < 0:
                return False
            pre, post = long[:i], long[i + len(short):]
            pre_ok = pre in ("", "_") or pre.rstrip("_") in self._AFFIX_PRE
            post_ok = post in ("", "_") or post.lstrip("_") in self._AFFIX_POST
            if pre_ok and post_ok:
                return True
            start = i + 1

    def _atom_weight(self, token: str, family: str) -> float:
        if token not in self._atoms.get(family, set()):
            return 0.0
        w = self._vidf.get(token, 1.0)
        if token in self._atom_middle.get(family, set()):
            w *= self._middle_discount
        return w

    def _compound_weight(self, token: str, family: str) -> float:
        """Best weight against the family's compound tags: exact compound >
        exact part > affix-bounded part (detect ~ regime_DETECTION) >
        affix-bounded whole, each discounted."""
        best = 0.0
        for c in self._compounds.get(family, set()):
            if token == c:
                best = max(best, self._vidf.get(c, 1.0))
                continue
            parts = c.split("_")
            if token in parts:
                best = max(best, self._vidf.get(c, 1.0) * self._part_discount)
                continue
            if len(token) < self._morph_min:
                continue
            if any(
                len(p) >= self._morph_min and self._affix_bounded(token, p)
                for p in parts
            ):
                best = max(best, self._vidf.get(c, 1.0) * self._affix_discount)
            elif self._affix_bounded(token, c):
                best = max(best, self._vidf.get(c, 1.0) * self._affix_discount)
        return best

    def _term_weight(self, token: str, family: str) -> float:
        return max(self._atom_weight(token, family), self._compound_weight(token, family))

    def family_weights(self, problem_tokens: set[str]) -> dict[str, float]:
        """How strongly the problem text points at each research family, in
        [0, 1], max-normalized. Raw matches are deflated by vocabulary size
        (large families collect spurious matches by bulk alone) and a problem
        matching nothing returns all-ones (never punish with no signal)."""
        import math

        alpha = float(self._cfg.research.family_size_norm_exponent)
        raw = {}
        for fam in self._atoms:
            vocab_size = len(self._atoms[fam]) + len(self._compounds.get(fam, set()))
            total = sum(
                self._term_weight(t, fam) for t in problem_tokens if len(t) >= 3
            )
            raw[fam] = total / max(1.0, math.sqrt(vocab_size)) ** (2.0 * alpha)
        top = max(raw.values(), default=0.0)
        if top <= 0:
            return {f: 1.0 for f in raw}
        return {f: w / top for f, w in raw.items()}

    def _recall(self, problem_tokens: set[str], text_tokens: set[str]) -> float:
        """IDF-weighted share of the problem's vocabulary covered by the text
        (exact tokens plus affix-bounded derivational matches, discounted)."""
        if not problem_tokens:
            return 0.0
        covered = 0.0
        for t in problem_tokens:
            if t in text_tokens:
                covered += self._idf.get(t, 1.0)
            elif len(t) >= self._morph_min:
                forms = [
                    self._idf.get(u, 1.0)
                    for u in text_tokens
                    if len(u) >= self._morph_min and self._affix_bounded(t, u)
                ]
                if forms:
                    covered += max(forms) * self._affix_discount
        if covered <= 0:
            return 0.0
        denom = sum(self._idf.get(t, 1.0) for t in problem_tokens)
        if denom <= 0:
            return 0.0
        return covered / denom

    def mech_evidence(self, problem_tokens: set[str], node: dict) -> float:
        """Concentrated mechanism-tag evidence: IDF-weighted problem match
        against the node's own mech_tags, with the same affix/part discounts.
        Unlike relevance (a recall diluted by the full problem denominator),
        this rewards a direct hit on the curated mechanism descriptors — a
        method tagged regime_detection IS about detecting regimes even if its
        label/summary never says 'volatility'."""
        compounds = [str(t) for t in node.get("mech_tags", []) if "_" in str(t)]
        atoms = {str(t) for t in node.get("mech_tags", []) if "_" not in str(t)}
        total = 0.0
        for t in problem_tokens:
            if len(t) < 3:
                continue
            best = 0.0
            if t in atoms:
                best = self._idf.get(t, 1.0)
            elif len(t) >= self._morph_min:
                for c in compounds:
                    parts = c.split("_")
                    if t in parts:
                        best = max(best, self._idf.get(c, 1.0) * self._part_discount)
                    elif self._affix_bounded(t, c) or any(
                        len(p) >= self._morph_min and self._affix_bounded(t, p)
                        for p in parts
                    ):
                        best = max(best, self._idf.get(c, 1.0) * self._affix_discount)
            total += best
        return total

    def relevance(self, problem_tokens: set[str], node: dict) -> float:
        """Best of diluted text recall and concentrated tag evidence (max, not
        sum — the tag tokens are already inside the recall text)."""
        node_tokens = retrieval.tokens(
            " ".join(
                [node["label"], node.get("summary") or "", " ".join(node["mech_tags"])]
            )
        )
        return max(self._recall(problem_tokens, node_tokens),
                   self.mech_evidence(problem_tokens, node))

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


def _parse_score_reply(reply: str) -> tuple[float, float] | None:
    """Tolerant parse of the scorer reply: strip markdown fences, then scan
    balanced {...} blocks and return the first that parses with both keys
    (a greedy .* would glue two separate blocks together and fail)."""
    text = reply.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    blocks: list[str] = [text]
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                blocks.append(text[start : i + 1])
                start = None
    for candidate in blocks:
        try:
            parsed = json.loads(candidate)
            rel = float(np.clip(float(parsed["relevance"]), 0.0, 1.0))
            fit = float(np.clip(float(parsed["structural_fit"]), 0.0, 1.0))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        return rel, fit
    return None


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
    with db.locked():  # raw read — threaded scoring pass shares one connection
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
        scores = _parse_score_reply(reply)
    except Exception:
        return None
    if scores is None:
        return None
    rel, fit = scores
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

    full_field = llm is not None and bool(getattr(cfg.research, "scoring_full_field", False))
    if full_field:
        # Semantic coverage: with an LLM scorer, rank the ENTIRE visible
        # technique field, not a lexically-recalled window. Lexical padding
        # demonstrably buries on-target cards (Hamilton regime switching and
        # CUSUM never entered a 30-candidate set for a regime-detection
        # problem). score_cache keys (node, problem_class, model) make the
        # per-problem cost one-time; the scoring pass below is threaded.
        rows = db.conn.execute(
            "SELECT * FROM graph_nodes WHERE type = 'technique' AND as_of <= ?"
            " ORDER BY node_id",
            (as_of,),
        ).fetchall()
        candidates = [dict(r) for r in rows]
        for c in candidates:
            c["mech_tags"] = json.loads(c["mech_tags"])
    else:
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
    scorer = DeterministicScorer(db, as_of, cfg)
    problem_tokens = retrieval.tokens(problem)
    exploration_bonus = float(cfg.research.exploration_bonus_uniform)
    family_weights = scorer.family_weights(problem_tokens)
    family_floor = float(cfg.research.family_weight_floor)

    # Recall floor: hybrid recall can return a handful of nodes for problems
    # whose vocabulary is thin in the corpus (FTS finds few matches, no
    # embedder). A 6-of-6 pool has no real draw in it — repeat queries would
    # return the same k by construction (G1 overlap). Pad the recall set up
    # to selection_recall_pad_to with the best remaining techniques by
    # deterministic relevance, so the §8.2 draw always has room to explore.
    # Padding orders candidates; the LLM/deterministic scorer still ranks all
    # of them — nothing is promoted past the score stage.
    pad_to = int(cfg.research.selection_recall_pad_to)
    if pad_to > 0 and len(candidates) < pad_to:
        have = {c["node_id"] for c in candidates}
        rest = [
            dict(r)
            for r in db.conn.execute(
                "SELECT * FROM graph_nodes WHERE type = 'technique' AND as_of <= ?"
                " ORDER BY node_id",
                (as_of,),
            ).fetchall()
            if r["node_id"] not in have
        ]
        for c in rest:
            c["mech_tags"] = json.loads(c["mech_tags"])
        rest.sort(
            key=lambda c: (-scorer.relevance(problem_tokens, c), c["node_id"])
        )
        candidates = candidates + rest[: pad_to - len(candidates)]

    def _family_factor(node_family: str | None) -> float:
        # on-family candidates keep full lexical relevance; off-family ones
        # keep `family_floor` of it (cross-family composition stays possible)
        return family_floor + (1.0 - family_floor) * family_weights.get(
            node_family or "", 0.0
        )

    fit_exponent = float(cfg.research.family_fit_factor_exponent)

    # Pass 1: raw factored relevance per candidate (LLM or deterministic),
    # plus structural_fit in the same pass when an LLM is present. LLM
    # scoring runs threaded (the cache writes serialize on the db lock); a
    # per-candidate failure falls back to the deterministic scorer and is
    # tallied — scoring_mode alone cannot show a partial fallback.
    def _score_one(node: dict) -> tuple[float, float | None] | None:
        return _llm_scores(db, llm, node, problem, pclass) if llm else None

    if llm is not None and len(candidates) > 1:
        workers = max(int(getattr(cfg.research, "scoring_workers", 1)), 1)
        if workers > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=workers) as pool:
                llm_results = list(pool.map(_score_one, candidates))
        else:
            llm_results = [_score_one(c) for c in candidates]
    else:
        llm_results = [_score_one(c) for c in candidates]

    raw_relevance: list[float] = []
    llm_fit: list[float | None] = []
    llm_fallback_count = 0
    for node, llm_scores in zip(candidates, llm_results, strict=True):
        if llm_scores is not None:
            raw_relevance.append(llm_scores[0])
            llm_fit.append(llm_scores[1])
        else:
            llm_fallback_count += 1 if llm is not None else 0
            raw_relevance.append(
                scorer.relevance(problem_tokens, node) * _family_factor(node.get("family"))
            )
            llm_fit.append(None)
    # Max-normalize the factored relevance across candidates: raw recall is
    # denominator-diluted by unmatched problem tokens, which would let the
    # structural_fit channel dominate the ranking and wash the family factor
    # out. Normalization restores the §8.1 channel balance with the pinned
    # weights; the value reported in the decomposition is the value used.
    rel_max = max(raw_relevance, default=0.0)

    scored = []
    for node, relevance, fit in zip(candidates, raw_relevance, llm_fit, strict=True):
        factor = _family_factor(node.get("family"))
        structural_fit = (
            fit if fit is not None else scorer.structural_fit(db, node, problem_tokens, as_of)
        )
        if fit is None:
            # the family factor applies to lexical evidence directly and to
            # graph-proximity evidence softened (fit_exponent): edges are the
            # legitimate cross-family composition channel, but they cannot
            # fully override a family mismatch either
            if rel_max > 0:
                relevance = relevance / rel_max
            structural_fit = structural_fit * factor**fit_exponent
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
            "llm_fallback_count": 0,
            "temperature": temperature,
            "mmr_lambda": lam,
            "k": k,
            "as_of": as_of,
            "candidate_count": 0,
            "family_weights": family_weights,
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

    # Selection pool = the "defensible set": candidates within
    # selection_score_band of the best score (§8.2 — softmax picks among
    # defensible moves, randomness never promotes an indefensible one).
    # Softmax sampling without replacement inside the set keeps exploration
    # (G1 repeat-overlap); if the band is too tight to sample from, fall back
    # to the top 2k by score so k suggestions are always reachable.
    band = float(cfg.research.selection_score_band)
    obvious_drop = max(int(cfg.research.selection_obvious_drop), 0)
    order = sorted(
        range(len(scored)), key=lambda i: (-scored[i]["score"], scored[i]["method_id"])
    )
    pool_order = order[obvious_drop:]  # selection_obvious_drop > 0 = challenger mode
    best = scored[pool_order[0]]["score"]
    eligible_idx = [i for i in pool_order if scored[i]["score"] >= best - band]
    min_eligible = max(int(cfg.research.selection_min_eligible), k + 1)
    if len(eligible_idx) < min_eligible:  # keep the draw stochastic (G1 overlap)
        eligible_idx = pool_order[: min(min_eligible, len(pool_order))]

    rng = np.random.default_rng(seed)
    pool_size = min(len(eligible_idx), k + max(int(cfg.research.selection_pool_extra), 1))
    if len(eligible_idx) <= pool_size:
        pool = [scored[i] for i in eligible_idx]
    else:
        # Pool draw uses a flattened (higher-temperature) distribution: the
        # reported selection_probability stays the true softmax (§8.2), but
        # exploration across repeats (G1 overlap) needs the draw to stay
        # stochastic even when the score ranking is decisive.
        flatten = max(float(cfg.research.selection_pool_temp_multiplier), 1e-6)
        pool_logits = np.array(
            [scored[i]["score"] for i in eligible_idx], dtype=np.float64
        ) / (max(temperature, 1e-9) * flatten)
        pool_logits -= pool_logits.max()
        pool_probs = np.exp(pool_logits)
        pool_probs /= pool_probs.sum()
        drawn = rng.choice(len(eligible_idx), size=pool_size, replace=False, p=pool_probs)
        pool = [scored[eligible_idx[int(i)]] for i in drawn]

    sim_matrix = _similarity_matrix(db, pool, embedder, as_of)
    suggestions = _mmr_select(pool, k, lam, sim_matrix)

    result = {
        "problem": problem,
        "problem_class": pclass,
        "mode": mode,
        "scoring_mode": "llm" if llm else "deterministic",
        "llm_fallback_count": llm_fallback_count,
        "temperature": temperature,
        "mmr_lambda": lam,
        "k": k,
        "as_of": as_of,
        "candidate_count": len(scored),
        "family_weights": family_weights,
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
            "candidate_count": len(scored),
            "suggestion_ids": result["suggestion_ids"],
            "as_of": as_of,
        },
    )
    return result
