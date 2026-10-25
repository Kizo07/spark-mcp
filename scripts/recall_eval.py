"""Recall-quality eval over the REAL seed corpus (production retrieval path).

One query per technique card (138): mechanism_tags tokens with deterministic
dropout (p=0.3, seeded) + shuffle. The card NAME is held out of the query so
the eval is not a trivial label lookup. Expected-relevant sets (same-card id
+ same-family ids) are recorded BEFORE running retrieval.

Primary metric: same-family recall@10 = |top10 cap family| / min(10, |family|),
macro-averaged over queries, reported per family + overall.
Secondary sanity metric: same-card hit@10 (own card in top 10).

Runs the production `retrieval.hybrid_recall` path. Default is FTS5-only
(embedder None — the current degraded reality). `--with-embedder` uses
`llm.build_embedder(Config())` (hybrid) and fails clearly when unconfigured.

Usage (from repo root):
  .venv/bin/python scripts/recall_eval.py
  .venv/bin/python scripts/recall_eval.py --with-embedder   # needs embeddings endpoint
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import yaml  # noqa: E402

from spark_mcp import retrieval  # noqa: E402
from spark_mcp.graph import find_node_by_label, import_seeds  # noqa: E402
from spark_mcp.storage import open_db  # noqa: E402

SEED = 7
DROPOUT = 0.3
K = 10
AS_OF = "2026-01-01T00:00:00Z"

TOKEN_RE = re.compile(r"[a-z0-9]+")


def query_tokens(card: dict, rng: random.Random) -> list[str]:
    pool = sorted(
        {
            t
            for t in TOKEN_RE.findall(
                " ".join(card.get("mechanism_tags") or []).replace("_", " ").lower()
            )
            if t not in retrieval.STOPWORDS
        }
    )
    keep = [t for t in pool if rng.random() >= DROPOUT] or pool[:2]
    rng.shuffle(keep)
    return keep


def run_eval(seed: int = SEED, embedder=None) -> dict:
    """Build a temp DB from seeds, run all queries, return per-family + overall."""
    repo = os.path.join(os.path.dirname(__file__), "..")
    cards = yaml.safe_load(
        open(os.path.join(repo, "src", "spark_mcp", "seeds", "techniques.yaml"))
    )
    tmp = tempfile.mkdtemp(prefix="recall_eval_")
    db = open_db(os.path.join(tmp, "recall.db"))
    try:
        import_seeds(db, embedder)
        # Provenance recorded BEFORE any retrieval runs.
        card_node = {c["name"]: find_node_by_label(db, c["name"])["node_id"] for c in cards}
        fam_ids: dict[str, set[str]] = {}
        for c in cards:
            fam_ids.setdefault(c["family"], set()).add(card_node[c["name"]])

        rng = random.Random(seed)
        per_family: dict[str, list[float]] = {}
        card_hits = 0
        for c in cards:
            q = " ".join(query_tokens(c, rng))
            hits = [
                n["node_id"]
                for n in retrieval.hybrid_recall(
                    db, q, AS_OF, embedder=embedder, k_vec=50, k_fts=50
                )
            ][:K]
            rel = fam_ids[c["family"]]
            r = len(set(hits) & rel) / min(K, len(rel))
            per_family.setdefault(c["family"], []).append(r)
            if card_node[c["name"]] in hits:
                card_hits += 1
    finally:
        db.close()
    families = {
        f: {"n": len(v), "recall_at_10": sum(v) / len(v)}
        for f, v in sorted(per_family.items())
    }
    all_r = [x for v in per_family.values() for x in v]
    return {
        "seed": seed,
        "families": families,
        "overall_recall_at_10": sum(all_r) / len(all_r),
        "same_card_hit_at_10": card_hits / len(all_r),
        "n_queries": len(all_r),
        "mode": "hybrid" if embedder is not None else "fts5-only",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--with-embedder", action="store_true")
    args = ap.parse_args()
    embedder = None
    if args.with_embedder:
        from spark_mcp.config import Config
        from spark_mcp.llm import build_embedder

        embedder = build_embedder(Config())
        if embedder is None:
            sys.exit(
                "no embedder configured: set llm.embeddings_base_url + "
                "$OPENAI_API_KEY (see docs/recall-eval.md)"
            )
    res = run_eval(seed=args.seed, embedder=embedder)
    print(f"mode={res['mode']} seed={res['seed']} queries={res['n_queries']}")
    print(f"{'family':22s} {'n':>3s}  recall@10")
    for f, d in res["families"].items():
        print(f"{f:22s} {d['n']:3d}  {d['recall_at_10']:.3f}")
    print(f"OVERALL recall@10 = {res['overall_recall_at_10']:.3f}")
    print(f"same-card hit@10  = {res['same_card_hit_at_10']:.3f} (sanity)")


if __name__ == "__main__":
    main()
