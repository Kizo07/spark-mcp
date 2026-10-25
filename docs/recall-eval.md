# Recall eval (seed corpus, production retrieval path)

Date: 2026-10-25. Repo: spark-mcp @c9512bc.

## Method

- Corpus: real seed corpus — 138 technique cards + 263 concept distractors,
  imported via production `graph.import_seeds` into a temp DB.
- Queries: one per technique card (138). Query = card `mechanism_tags` tokens
  (underscores split, stopwords removed) with deterministic dropout p=0.3 and
  shuffle (seed 7, no LLM). The card NAME is held out so the eval is not a
  trivial label lookup.
- Expected-relevant sets (same-card node id + same-family node ids) are
  recorded BEFORE running retrieval.
- Retrieval: production `retrieval.hybrid_recall`, FTS5-only (embedder None).
- Metric: same-family recall@10 = |top10 ∩ family| / min(10, |family|),
  macro-averaged; reported per family + overall. Same-card hit@10 is a
  secondary sanity metric.

## Measured (FTS5-only, seed 7)

| family | n | recall@10 |
|---|---|---|
| causal | 22 | 0.323 |
| econometrics | 32 | 0.388 |
| network_optimization | 10 | 0.300 |
| quant | 51 | 0.576 |
| signal_processing | 23 | 0.317 |
| OVERALL | 138 | **0.429** |

Same-card hit@10 = 0.986 (sanity: own card almost always in top 10).

## Reproduce

From repo root: `.venv/bin/python scripts/recall_eval.py`
Deterministic: two consecutive runs print identical numbers.
Gate: `tests/test_recall_gate.py` asserts overall >= 0.35
(measured 0.429, floor set below with margin). Per-family numbers are
reported but not gated (thin shelves, e.g. network_optimization n=10).

## Hybrid/embeddings: concrete blocker

No hybrid number is reported because `FakeEmbedder` is sha256-seeded
hash-random per text — a hybrid recall figure with it would be meaningless.
Hybrid recall requires a real embedder: `llm.build_embedder` needs
`llm.embeddings_base_url` set AND `$OPENAI_API_KEY` set (default
`api_key_env`). Environment check (values never printed):
`embeddings_base_url`: unset; `$OPENAI_API_KEY`: unset;
`sentence-transformers`: not installed. `build_embedder(Config())` returns
None here, so the FTS5-only path above is the current degraded reality.

Re-run with embeddings once configured:

`.venv/bin/python scripts/recall_eval.py --with-embedder`
