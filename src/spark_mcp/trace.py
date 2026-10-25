"""Typed, read-only trace views over the ledger (P1-5).

A *trace* for one hypothesis is the append-only chain stored in the
ledger: the hypothesis row (`hypotheses`), its trial episodes
(`episodes`, with their countable `trials` rows), and the narrated
evolution feedback rows (`evolution_feedback`) — merged into one list
ordered by `created_at` ascending. Every table named here is read
exactly as `storage.py` defines it; no column is invented.

The view models are the concise, prompt-safe projections of those
rows (cf. `evolve/render.py`, which slices mechanism text at 280
chars): every `concise_*` field is a plain `str` deterministically
truncated to at most `CONCISE_LIMIT` characters, derived only from
stored row content. No LLM is consulted and no text is generated.

Pure read-only: this module issues only `SELECT`s. It never writes,
never migrates, and never creates schema. Unknown-hypothesis handling
mirrors the tool-level evolve surface (`evolve/fitness.py`,
`evolve/run.py:evolution_status`): a named `ToolError`. A known
hypothesis with no episodes/feedback yet traces as just its
`HypothesisView` (the trace root); `sota_pointer` on an empty or
ungated ledger returns `None`.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .errors import ToolError
from .storage import Database

#: Deterministic cap for every `concise_*` projection (matches the
#: 280-char mechanism slice in `evolve/render.py`).
CONCISE_LIMIT = 280

__all__ = [
    "CONCISE_LIMIT",
    "ExperimentView",
    "FeedbackView",
    "HypothesisView",
    "TraceEntry",
    "sota_pointer",
    "trace_for_hypothesis",
]


def _concise(value: Any, limit: int = CONCISE_LIMIT) -> str:
    """Deterministic concise projection: `None` -> "", non-strings are
    rendered (dicts/lists as sorted JSON, like the ledger stores them),
    and the result is truncated to at most `limit` chars by plain slice
    (no ellipsis, no word-boundary logic — same input, same output)."""
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    elif isinstance(value, (dict, list)):
        text = json.dumps(value, sort_keys=True, default=str)
    else:
        text = str(value)
    return text[:limit]


def _parse_json(value: Any, default: Any = None) -> Any:
    """Parse a stored JSON TEXT column; pass dicts/lists through, and
    degrade malformed/missing values to `default` (read-only tolerance:
    a trace must not fail on one legacy row)."""
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _as_float(value: Any) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


class _ConciseModel(BaseModel):
    """Base enforcing the concise invariant on direct construction too:
    any `concise_*` field longer than the cap is truncated."""

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def _truncate_concise_fields(self) -> _ConciseModel:
        for name in type(self).model_fields:
            if name.startswith("concise_"):
                value = getattr(self, name)
                if isinstance(value, str) and len(value) > CONCISE_LIMIT:
                    object.__setattr__(self, name, value[:CONCISE_LIMIT])
        return self


class HypothesisView(_ConciseModel):
    """Concise view of one `hypotheses` row."""

    kind: Literal["hypothesis"] = "hypothesis"
    hypothesis_id: str
    research_family: str = ""
    hypothesis_type: str = ""
    status: str = ""
    expected_direction: str = ""
    horizon: str = ""
    universe: str = ""
    predicted_ic: float | None = None
    predicted_sigma: float | None = None
    parent_hypothesis_ids: list[str] = Field(default_factory=list)
    created_at: str = ""
    concise_mechanism: str = ""
    concise_falsifier: str = ""
    concise_scope: str = ""
    concise_summary: str = ""

    @classmethod
    def from_row(cls, row: Any) -> HypothesisView:
        d = dict(row)
        parents = _parse_json(d.get("parent_hypothesis_ids"), [])
        if not isinstance(parents, list):
            parents = []
        mechanism = d.get("mechanism") or ""
        falsifier = d.get("falsifier") or ""
        return cls(
            hypothesis_id=str(d.get("hypothesis_id") or ""),
            research_family=str(d.get("research_family") or ""),
            hypothesis_type=str(d.get("hypothesis_type") or ""),
            status=str(d.get("status") or ""),
            expected_direction=str(d.get("expected_direction") or ""),
            horizon=str(d.get("horizon") or ""),
            universe=str(d.get("universe") or ""),
            predicted_ic=_as_float(d.get("predicted_ic")),
            predicted_sigma=_as_float(d.get("predicted_sigma")),
            parent_hypothesis_ids=[str(p) for p in parents],
            created_at=str(d.get("created_at") or ""),
            concise_mechanism=_concise(mechanism),
            concise_falsifier=_concise(falsifier),
            concise_scope=_concise(d.get("scope")),
            concise_summary=_concise(mechanism),
        )


class ExperimentView(_ConciseModel):
    """Concise view of one `episodes` row (a trial episode / experiment),
    with its `trials` ids and the gate verdict stored by
    `gates.service.significance_gate` in `corrected_metrics`
    (`pass`, `dsr`, `sr`, `p_value`). `dsr_prob` mirrors
    `evolve/fitness.py`: the stored `dsr` probability under its fitness
    name. All are `None` for an ungated episode."""

    kind: Literal["experiment"] = "experiment"
    episode_id: str
    hypothesis_id: str = ""
    research_family: str = ""
    run_kind: str = ""
    status: str = ""
    outcome: str | None = None
    created_at: str = ""
    trial_ids: list[str] = Field(default_factory=list)
    trial_count: int = 0
    gate_pass: bool | None = None
    dsr_prob: float | None = None
    sr: float | None = None
    p_value: float | None = None
    concise_failure_reason: str = ""
    concise_metrics: str = ""
    concise_corrected_metrics: str = ""
    concise_summary: str = ""

    @classmethod
    def from_row(cls, row: Any, trial_ids: list[str] | None = None) -> ExperimentView:
        d = dict(row)
        metrics = _parse_json(d.get("metrics"), {})
        if not isinstance(metrics, dict):
            metrics = {}
        corrected = _parse_json(d.get("corrected_metrics"), {})
        if not isinstance(corrected, dict):
            corrected = {}
        dsr = _as_float(corrected.get("dsr", corrected.get("dsr_prob")))
        gate_pass = corrected.get("pass")
        ids = list(trial_ids or [])
        status = str(d.get("status") or "")
        outcome = d.get("outcome")
        summary_src = d.get("failure_reason") or outcome or status
        return cls(
            episode_id=str(d.get("episode_id") or ""),
            hypothesis_id=str(d.get("hypothesis_id") or ""),
            research_family=str(d.get("research_family") or ""),
            run_kind=str(d.get("run_kind") or ""),
            status=status,
            outcome=str(outcome) if outcome is not None else None,
            created_at=str(d.get("created_at") or ""),
            trial_ids=ids,
            trial_count=len(ids),
            gate_pass=bool(gate_pass) if gate_pass is not None else None,
            dsr_prob=dsr,
            sr=_as_float(corrected.get("sr")),
            p_value=_as_float(corrected.get("p_value")),
            concise_failure_reason=_concise(d.get("failure_reason")),
            concise_metrics=_concise(metrics),
            concise_corrected_metrics=_concise(corrected),
            concise_summary=_concise(summary_src),
        )


class FeedbackView(_ConciseModel):
    """Concise view of one `evolution_feedback` row (the narrated trace
    appended by `evolve/lineage.py:record_feedback`). Narration fields
    are audit text, truncated like every other `concise_*` field."""

    kind: Literal["feedback"] = "feedback"
    feedback_id: str
    run_id: str = ""
    hypothesis_id: str = ""
    episode_id: str = ""
    decision: str = ""
    created_at: str = ""
    concise_observations: str = ""
    concise_hypothesis_evaluation: str = ""
    concise_new_hypothesis: str = ""
    concise_decision: str = ""
    concise_metrics: str = ""
    concise_fitness: str = ""
    concise_summary: str = ""

    @classmethod
    def from_row(cls, row: Any) -> FeedbackView:
        d = dict(row)
        metrics = _parse_json(d.get("metrics_json"), {})
        fitness = _parse_json(d.get("fitness_json"), {})
        observations = d.get("observations") or ""
        return cls(
            feedback_id=str(d.get("feedback_id") or ""),
            run_id=str(d.get("run_id") or ""),
            hypothesis_id=str(d.get("hypothesis_id") or ""),
            episode_id=str(d.get("episode_id") or ""),
            decision=str(d.get("decision") or ""),
            created_at=str(d.get("created_at") or ""),
            concise_observations=_concise(observations),
            concise_hypothesis_evaluation=_concise(d.get("hypothesis_evaluation")),
            concise_new_hypothesis=_concise(d.get("new_hypothesis")),
            concise_decision=_concise(d.get("decision")),
            concise_metrics=_concise(metrics),
            concise_fitness=_concise(fitness),
            concise_summary=_concise(observations),
        )


#: One entry of a hypothesis trace, in `created_at` order.
TraceEntry = HypothesisView | ExperimentView | FeedbackView

# Tie-break priority when several rows share one `created_at` second
# (`timeutil` has 1-second resolution): the hypothesis root first,
# then its experiments, then feedback about them. Remaining ties fall
# back to insertion order (`rowid`, the append-only order).
_KIND_PRIORITY = {"hypothesis": 0, "experiment": 1, "feedback": 2}


def trace_for_hypothesis(
    db: Database, hypothesis_id: str
) -> list[TraceEntry]:
    """Ordered trace for `hypothesis_id`: hypothesis, episodes and
    feedback merged append-only by `created_at` ASC (rowid breaks
    same-second ties, kind priority above breaks cross-table ties).

    Raises `ToolError` on an unknown `hypothesis_id` (consistent with
    `fitness_from_episode` / `evolution_status`). A known hypothesis
    with no episodes/feedback returns `[HypothesisView]` — the trace
    root alone, so filtering that list by kind yields empty experiment
    / feedback lists. Read-only: SELECTs only.
    """
    hyp_row = db.conn.execute(
        "SELECT rowid AS _rowid, * FROM hypotheses WHERE hypothesis_id = ?",
        (hypothesis_id,),
    ).fetchone()
    if hyp_row is None:
        raise ToolError(f"unknown hypothesis_id {hypothesis_id!r}")

    entries: list[tuple[str, int, int, TraceEntry]] = []

    hyp = HypothesisView.from_row(hyp_row)
    entries.append((hyp.created_at, _KIND_PRIORITY["hypothesis"],
                      int(hyp_row["_rowid"]), hyp))

    ep_rows = db.conn.execute(
        "SELECT rowid AS _rowid, * FROM episodes WHERE hypothesis_id = ?"
        " ORDER BY created_at ASC, rowid ASC",
        (hypothesis_id,),
    ).fetchall()
    for row in ep_rows:
        trial_rows = db.conn.execute(
            "SELECT trial_id FROM trials WHERE episode_id = ?"
            " ORDER BY created_at ASC, rowid ASC",
            (row["episode_id"],),
        ).fetchall()
        view = ExperimentView.from_row(row, [str(r["trial_id"]) for r in trial_rows])
        entries.append((view.created_at, _KIND_PRIORITY["experiment"],
                        int(row["_rowid"]), view))

    fb_rows = db.conn.execute(
        "SELECT rowid AS _rowid, * FROM evolution_feedback WHERE hypothesis_id = ?"
        " ORDER BY created_at ASC, rowid ASC",
        (hypothesis_id,),
    ).fetchall()
    for row in fb_rows:
        view = FeedbackView.from_row(row)
        entries.append((view.created_at, _KIND_PRIORITY["feedback"],
                        int(row["_rowid"]), view))

    entries.sort(key=lambda e: (e[0], e[1], e[2]))
    return [view for _, _, _, view in entries]


def sota_pointer(db: Database) -> dict | None:
    """Pointer to the best gated trial in the ledger, or `None`.

    "Gated" uses only verdict fields that actually exist:
    `episodes.corrected_metrics` as written by `significance_gate`
    (`pass`, `dsr`, `sr`, `p_value` — see `gates/service.py`). Rows
    whose stored gate verdict is not `pass is True`, or whose `dsr`
    is missing/unparseable, are not candidates; no column, score, or
    verdict field is invented. Ranking is by stored `dsr` descending
    (reported as `dsr_prob`, the `evolve/fitness.py` name for the same
    stored probability); ties break by earliest `episode_id`, matching
    `evolve/lineage.py:sota_ancestor`. Read-only: SELECTs only.
    """
    rows = db.conn.execute(
        "SELECT * FROM episodes WHERE corrected_metrics IS NOT NULL"
        " ORDER BY created_at ASC, rowid ASC"
    ).fetchall()
    candidates: list[dict] = []
    for row in rows:
        corrected = _parse_json(row["corrected_metrics"], {})
        if not isinstance(corrected, dict) or corrected.get("pass") is not True:
            continue
        dsr = _as_float(corrected.get("dsr", corrected.get("dsr_prob")))
        if dsr is None:
            continue
        candidates.append({
            "hypothesis_id": str(row["hypothesis_id"]),
            "episode_id": str(row["episode_id"]),
            "dsr_prob": dsr,
            "sr": _as_float(corrected.get("sr")),
            "p_value": _as_float(corrected.get("p_value")),
            "created_at": str(row["created_at"] or ""),
        })
    if not candidates:
        return None
    best_dsr = max(c["dsr_prob"] for c in candidates)
    tied = [c for c in candidates if c["dsr_prob"] == best_dsr]
    tied.sort(key=lambda c: c["episode_id"])
    return tied[0]
