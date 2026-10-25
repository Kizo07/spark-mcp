"""Recall gate: FTS5-only retrieval quality over the seed corpus (see docs/recall-eval.md).

Measured 2026-10-25: overall same-family recall@10 = 0.429 (seed 7, 138 queries).
Floor is 0.35 — below measured with margin so the gate catches regressions,
not noise. Per-family numbers are reported, not gated (thin shelves).
"""

import importlib.util
from pathlib import Path

RECALL_FLOOR = 0.35


def _load_eval():
    path = Path(__file__).resolve().parents[1] / "scripts" / "recall_eval.py"
    spec = importlib.util.spec_from_file_location("recall_eval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_overall_recall_above_floor():
    res = _load_eval().run_eval()
    print({f: round(d["recall_at_10"], 3) for f, d in res["families"].items()})
    assert res["overall_recall_at_10"] >= RECALL_FLOOR, (
        f"overall recall@10 {res['overall_recall_at_10']:.3f} < floor {RECALL_FLOOR}"
    )
