"""Capped, summarized trace rendering for evolution prompts (plan §2.E2).

Renders *ancestors of the selected parent only* (not the whole forest):
concise hypothesis fields plus an in-sample metric table per trial — never
stdout logs, never full history. Trials are numbered oldest-first with
explicit "Trial 1..n" headings (newest last; the order is unit-tested, so
RD's insert(0)+reverse newest-first bug cannot recur silently).

Token budget: `ResearchConfig.evolution_render_max_tokens` with an explicit
truncation policy — drop the oldest non-SOTA ancestors first (SOTA here =
ancestors with a recorded `promoted` feedback row), then the oldest
remaining, keeping at least one trial; the render header records
`truncated_from_n`. Token counts use the explicit word-count estimator
below, never a silent character slice (the audit's judge failure mode).

Held-out exclusion (constraint 5): per-trial metrics come from
`evolution_feedback` rows, which are IS-only by construction (lineage
derives `metrics_json` from the episode's corrected/costs/simulated
allowlist), so OOS values cannot enter a rendered prompt.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from ..config import Config
from ..storage import Database
from ..timeutil import now_iso
from . import lineage

#: Purpose tag for ancestor renders cached in `generated_text_cache` (audit
#: replay of the exact prompt context per generation).
CACHE_PURPOSE = "evolution_ancestors"


@dataclass(frozen=True)
class AncestorRender:
    text: str
    n_ancestors: int
    truncated_from_n: int  # trials considered before truncation (== n when uncut)
    sha: str  # sha1 of text; stored as prompt_sha on feedback rows


def estimate_tokens(text: str) -> int:
    """v1 token estimator: whitespace word count (deterministic, documented).

    A deliberate over/under-estimate either way; what matters is that the
    budget check uses this estimator everywhere instead of a character
    slice. At least 1 for any non-empty text.
    """
    words = text.split()
    return len(words) if words else 0


def _latest_feedback(db: Database, hypothesis_id: str) -> dict | None:
    row = db.conn.execute(
        "SELECT * FROM evolution_feedback WHERE hypothesis_id = ?"
        " ORDER BY rowid DESC LIMIT 1",
        (hypothesis_id,),
    ).fetchone()
    return dict(row) if row else None


def _is_sota(feedback: dict | None) -> bool:
    return feedback is not None and str(feedback.get("decision", "")).startswith("promoted")


def _metric_cells(feedback: dict | None) -> str:
    """One-line IS metric summary for a trial (feedback rows are IS-only)."""
    if feedback is None:
        return "no trial yet"
    try:
        fitness = json.loads(feedback.get("fitness_json") or "{}")
        dsr = float(fitness["dsr_prob"])
        sr = float(fitness["sr"])
        m = int(fitness["M_family"])
        t = int(fitness["T"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return str(feedback.get("decision", "no metrics"))
    return (
        f"{feedback.get('decision', '?')} | dsr_prob={dsr:.4f} sr={sr:.4f}"
        f" M={m} T={t}"
    )


def _trial_block(db: Database, hypothesis_id: str) -> tuple[str, bool]:
    """Concise fields + metric line for one ancestor trial."""
    from .. import ledger

    hyp = ledger.get_hypothesis(db, hypothesis_id) or {}
    feedback = _latest_feedback(db, hypothesis_id)
    lines = [
        f"hypothesis_id={hypothesis_id}",
        f"mechanism={str(hyp.get('mechanism', ''))[:280]}",
        f"direction={hyp.get('expected_direction', '?')}"
        f" horizon={hyp.get('horizon', '?')}"
        f" predicted_ic={hyp.get('predicted_ic', '?')}",
        f"metrics: {_metric_cells(feedback)}",
    ]
    return "\n".join(lines), _is_sota(feedback)


def empty_render() -> AncestorRender:
    """Render for a parentless root: zero ancestors, stable sha."""
    text = "Ancestor trace: no ancestors (new root).\n"
    return AncestorRender(
        text=text,
        n_ancestors=0,
        truncated_from_n=0,
        sha=hashlib.sha1(text.encode()).hexdigest(),
    )


def render_ancestors(
    db: Database, cfg: Config, hypothesis_id: str, *, max_tokens: int | None = None
) -> AncestorRender:
    """Render the selected parent's ancestors oldest-first ("Trial 1..n").

    The chain is the first-parent walk (`walk_to_root`, self excluded) —
    deterministic on DAGs by documented convention. Raises ValueError on an
    unknown id. `max_tokens` defaults to
    `ResearchConfig.evolution_render_max_tokens`.
    """
    chain = lineage.walk_to_root(db, hypothesis_id)  # self-first; raises ValueError
    ancestors = list(reversed(chain[1:]))  # oldest-first, self excluded
    if max_tokens is None:
        max_tokens = int(cfg.research.evolution_render_max_tokens)

    blocks = [_trial_block(db, hyp_id) for hyp_id in ancestors]
    total = len(blocks)

    def _assemble(kept: list[tuple[str, str, bool]]) -> str:
        header = f"Ancestor trace: {len(kept)} trial(s)"
        if len(kept) < total:
            header += f" (truncated from {total}; oldest non-SOTA dropped first)"
        parts = [header + "."]
        for i, (hyp_id, body, _) in enumerate(kept, start=1):
            parts.append(f"--- Trial {i} ({hyp_id}) ---\n{body}")
        return "\n".join(parts) + "\n"

    numbered = [
        (hyp_id, body, sota)
        for hyp_id, (body, sota) in zip(ancestors, blocks, strict=True)
    ]
    text = _assemble(numbered)
    # Truncation policy: drop the oldest non-SOTA trial first; when all
    # remaining are SOTA, drop the oldest; always keep at least one trial.
    while len(numbered) > 1 and estimate_tokens(text) > max_tokens:
        drop = next((i for i, (_, _, sota) in enumerate(numbered) if not sota), 0)
        del numbered[drop]
        text = _assemble(numbered)
    return AncestorRender(
        text=text,
        n_ancestors=len(numbered),
        truncated_from_n=total,
        sha=hashlib.sha1(text.encode()).hexdigest(),
    )


def cache_render(db: Database, render: AncestorRender, model_id: str | None) -> str:
    """Store the full render text in `generated_text_cache` for audit replay.

    Keyed like the analogy cache (`sha1(purpose|subject|context|model_id)`);
    returns the cache key. `model_id` is None on the deterministic path.
    """
    model = model_id or "none"
    raw = f"{CACHE_PURPOSE}|{render.sha}|evolution|{model}".encode()
    key = hashlib.sha1(raw).hexdigest()  # noqa: S324 - dedupe key, not security
    db.execute(
        "INSERT OR REPLACE INTO generated_text_cache"
        " (cache_key, purpose, subject_key, context_key, model_id, text, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (key, CACHE_PURPOSE, render.sha, "evolution", model, render.text, now_iso()),
    )
    return key


def cached_render(db: Database, sha: str, model_id: str | None) -> str | None:
    """Fetch a cached render by its sha (audit replay)."""
    model = model_id or "none"
    raw = f"{CACHE_PURPOSE}|{sha}|evolution|{model}".encode()
    key = hashlib.sha1(raw).hexdigest()  # noqa: S324 - dedupe key, not security
    row = db.conn.execute(
        "SELECT text FROM generated_text_cache WHERE cache_key = ?", (key,)
    ).fetchone()
    return str(row["text"]) if row else None
