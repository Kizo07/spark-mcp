"""DAG lineage + branch points for artifact evolution (plan §2.C1).

RD ideas ported onto spark's ledger: the `hypotheses` table already carries
`parent_hypothesis_ids` (multi-parent capable, no schema change), so this
module adds typed helpers over it plus the two things the ledger lacked —
an explicit "current selection / branch from ancestor" pointer per research
line (`evolution_lines`) and a structured narrated trace
(`evolution_feedback`). RD's `Trace` class is not imported.

Branching: `create_line` inserts a new line whose head points at any
ancestor hypothesis (the branch point). Promotion bookkeeping moves the
prior head to `rollback_*` before setting the new head (`advance_head`),
so a human veto can restore it; the E1 run driver calls that helper, and
`skills.approve(target_type='evolution')` performs the restore.

All helpers are deterministic ledger reads/writes; ULID primary keys come
from `ids.py` (never the run's seeded RNG) and are identity metadata,
excluded from reproducibility comparisons by column (constraint 4).
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3

from .. import ledger
from ..errors import ToolError
from ..ids import new_evolution_feedback_id, new_evolution_line_id, new_evolution_run_id
from ..storage import Database
from ..timeutil import now_iso
from .fitness import FitnessReport

#: Decision prefixes accepted on evolution_feedback rows. The stored value is
#: the prefix plus the computed reason ("promoted: ..."/"rejected: ...").
_DECISION_PREFIXES = ("promoted", "rejected")


def _require_hypothesis(db: Database, hypothesis_id: str) -> dict:
    hypothesis = ledger.get_hypothesis(db, hypothesis_id)
    if hypothesis is None:
        raise ValueError(f"unknown hypothesis_id {hypothesis_id!r}")
    return hypothesis


def _parents_of(hypothesis: dict) -> list[str]:
    parents = hypothesis.get("parent_hypothesis_ids") or []
    return [str(p) for p in parents]


def ancestors(db: Database, hypothesis_id: str) -> list[str]:
    """All ancestor ids of a hypothesis, nearest-first (BFS over the DAG).

    Deterministic: parents are visited in stored list order, each node once
    (cycle-safe). The node itself is excluded. Raises ValueError on an
    unknown id.
    """
    _require_hypothesis(db, hypothesis_id)
    out: list[str] = []
    visited = {hypothesis_id}
    queue = list(_parents_of(ledger.get_hypothesis(db, hypothesis_id) or {}))
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        out.append(current)
        node = ledger.get_hypothesis(db, current)
        if node is not None:
            queue.extend(p for p in _parents_of(node) if p not in visited)
    return out


def children(db: Database, hypothesis_id: str) -> list[str]:
    """Direct children of a hypothesis, ordered by hypothesis_id.

    Raises ValueError on an unknown id.
    """
    _require_hypothesis(db, hypothesis_id)
    rows = db.conn.execute(
        "SELECT hypothesis_id, parent_hypothesis_ids FROM hypotheses ORDER BY hypothesis_id"
    ).fetchall()
    out = []
    for row in rows:
        try:
            parents = json.loads(row["parent_hypothesis_ids"] or "[]")
        except (json.JSONDecodeError, TypeError):
            continue
        if hypothesis_id in parents:
            out.append(row["hypothesis_id"])
    return out


def walk_to_root(db: Database, hypothesis_id: str) -> list[str]:
    """First-parent chain from the node up to a root, self-first.

    For single-parent lineages this is the full chain; on multi-parent DAGs
    the walk follows the first stored parent at each fork (documented,
    deterministic). Cycle-safe. Raises ValueError on an unknown id.
    """
    node = _require_hypothesis(db, hypothesis_id)
    chain = [hypothesis_id]
    visited = {hypothesis_id}
    parents = _parents_of(node)
    while parents:
        nxt = parents[0]
        if nxt in visited:
            break
        visited.add(nxt)
        chain.append(nxt)
        node = ledger.get_hypothesis(db, nxt)
        if node is None:
            break
        parents = _parents_of(node)
    return chain


def _latest_promoted_feedback(db: Database, hypothesis_id: str) -> dict | None:
    row = db.conn.execute(
        "SELECT rowid, episode_id, fitness_json FROM evolution_feedback"
        " WHERE hypothesis_id = ? AND decision LIKE 'promoted%'"
        " ORDER BY rowid DESC LIMIT 1",
        (hypothesis_id,),
    ).fetchone()
    return dict(row) if row else None


def sota_ancestor(db: Database, hypothesis_id: str) -> dict | None:
    """Best promoted ancestor: argmax of stored `dsr_prob` over ancestors with
    a recorded `promoted` feedback row (a computed ledger query, never an LLM
    bit). Each ancestor is ranked by its latest promoted row's fitness;
    ties break by earliest `episode_id` (deterministic). Returns
    `{hypothesis_id, episode_id, dsr_prob}` or None when no ancestor has a
    promoted row. Raises ValueError on an unknown id.
    """
    _require_hypothesis(db, hypothesis_id)
    ranked: list[tuple[str, str, float]] = []
    for ancestor_id in ancestors(db, hypothesis_id):
        row = _latest_promoted_feedback(db, ancestor_id)
        if row is None:
            continue
        try:
            fitness = json.loads(row["fitness_json"] or "{}")
            dsr = float(fitness["dsr_prob"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        ranked.append((ancestor_id, str(row["episode_id"]), dsr))
    if not ranked:
        return None
    best_dsr = max(dsr for _, _, dsr in ranked)
    tied = sorted(ep for _, ep, dsr in ranked if dsr == best_dsr)
    # Re-resolve the winning hypothesis for the earliest tied episode.
    winner_ep = tied[0]
    winner_hyp = next(h for h, ep, _ in ranked if ep == winner_ep)
    return {"hypothesis_id": winner_hyp, "episode_id": winner_ep, "dsr_prob": best_dsr}


def create_line(
    db: Database,
    family: str,
    head_hypothesis_id: str | None = None,
    head_episode_id: str | None = None,
) -> dict:
    """Insert a new (active) evolution line. The head may point at any
    ancestor hypothesis — that is the branch point. A headless line is a
    fresh root: the driver proposes its first hypothesis on this line.
    Raises ValueError when a named head hypothesis/episode does not exist.
    """
    if head_hypothesis_id is not None:
        _require_hypothesis(db, head_hypothesis_id)
    if head_episode_id is not None and ledger.get_episode(db, head_episode_id) is None:
        raise ValueError(f"unknown head_episode_id {head_episode_id!r}")
    line_id = new_evolution_line_id()
    now = now_iso()
    db.execute(
        "INSERT INTO evolution_lines (line_id, family, head_hypothesis_id,"
        " head_episode_id, status, created_at, updated_at)"
        " VALUES (?, ?, ?, ?, 'active', ?, ?)",
        (line_id, family, head_hypothesis_id, head_episode_id, now, now),
    )
    row = get_line(db, line_id)
    assert row is not None
    return row


def get_line(db: Database, line_id: str) -> dict | None:
    """One evolution line by id, or None."""
    row = db.conn.execute(
        "SELECT * FROM evolution_lines WHERE line_id = ?", (line_id,)
    ).fetchone()
    return dict(row) if row else None


def active_line(db: Database, family: str) -> dict | None:
    """Newest active line for a family (branching may leave several active;
    newest wins by insertion order). The run driver tracks its own line_id
    anyway; this is the lookup for fresh runs and status queries.
    """
    # rowid, not created_at: timestamps have 1-second resolution, so two
    # lines created in the same second would tie and fall back to a ULID
    # coin flip. Insertion order is the deterministic "newest wins".
    row = db.conn.execute(
        "SELECT * FROM evolution_lines WHERE family = ? AND status = 'active'"
        " ORDER BY rowid DESC LIMIT 1",
        (family,),
    ).fetchone()
    return dict(row) if row else None


def advance_head(
    db: Database, line_id: str, hypothesis_id: str, episode_id: str
) -> dict:
    """Promotion bookkeeping, atomically: the prior head moves to `rollback_*`
    before the new head is set, so a human veto can restore it (mirrors the
    skill `rollback_target`). Called by the run driver after a computed
    promotion — never a decision input. Raises ValueError on unknown ids.
    """
    line = get_line(db, line_id)
    if line is None:
        raise ValueError(f"unknown line_id {line_id!r}")
    _require_hypothesis(db, hypothesis_id)
    if ledger.get_episode(db, episode_id) is None:
        raise ValueError(f"unknown episode_id {episode_id!r}")
    now = now_iso()
    with db.transaction():
        db.execute(
            "UPDATE evolution_lines SET rollback_hypothesis_id = head_hypothesis_id,"
            " rollback_episode_id = head_episode_id,"
            " head_hypothesis_id = ?, head_episode_id = ?, updated_at = ?"
            " WHERE line_id = ?",
            (hypothesis_id, episode_id, now, line_id),
        )
    row = get_line(db, line_id)
    assert row is not None
    return row


def create_run(
    db: Database, line_id: str, seed: int, caps: dict | None = None
) -> dict:
    """Open a run audit record (`status='running'`). Raises ValueError on an
    unknown line.
    """
    line = get_line(db, line_id)
    if line is None:
        raise ValueError(f"unknown line_id {line_id!r}")
    run_id = new_evolution_run_id()
    db.execute(
        "INSERT INTO evolution_runs (run_id, line_id, seed, caps_json, status,"
        " head_before, started_at) VALUES (?, ?, ?, ?, 'running', ?, ?)",
        (
            run_id,
            line_id,
            int(seed),
            json.dumps(caps or {}, sort_keys=True),
            line.get("head_hypothesis_id"),
            now_iso(),
        ),
    )
    row = db.conn.execute(
        "SELECT * FROM evolution_runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    assert row is not None
    return dict(row)


def finish_run(db: Database, run_id: str, status: str, head_after: str | None) -> dict:
    """Close a run audit record. Raises ValueError on an unknown run."""
    row = db.conn.execute(
        "SELECT * FROM evolution_runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown run_id {run_id!r}")
    db.execute(
        "UPDATE evolution_runs SET status = ?, head_after = ?, finished_at = ?"
        " WHERE run_id = ?",
        (status, head_after, now_iso(), run_id),
    )
    row = db.conn.execute(
        "SELECT * FROM evolution_runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    assert row is not None
    return dict(row)


def _strip_oos(mapping: dict) -> dict:
    """Defensive copy without held-out keys (constraint 5). The inputs below
    should never carry OOS keys in the first place; this makes the exclusion
    structural at the write edge too.
    """
    return {k: v for k, v in mapping.items() if k != "oos" and not k.startswith("oos_")}


def record_feedback(
    db: Database,
    run_id: str,
    hypothesis_id: str,
    episode_id: str,
    parent_ids: list[str],
    narration: dict,
    fitness: FitnessReport | dict,
    decision: str,
    model_id: str | None,
    prompt_sha: str | None,
) -> dict:
    """Append a structured narrated-trace row (the middle form between spark's
    `mutation_op` string and RD's unbounded render).

    `metrics_json` is derived here from the episode row — `corrected_metrics`
    (in-sample gate outputs) + `costs` + the `metrics.simulated` provenance
    bit, the same allowlist as fitness — never from caller-supplied metrics,
    so OOS values can never reach a feedback row (constraint 5). Narration
    fields are write-only audit: recorded after the decision, never inputs
    to any future `promote.decide` call (constraint 2). `as_of`/
    `data_through` are copied from the episode (PIT rule).

    `fitness` is the component-A report (or its dict form); `decision` must
    start with `promoted` or `rejected` (computed reason follows).
    Raises ValueError on unknown ids or a malformed decision.
    """
    run = db.conn.execute(
        "SELECT * FROM evolution_runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    if run is None:
        raise ValueError(f"unknown run_id {run_id!r}")
    _require_hypothesis(db, hypothesis_id)
    episode = ledger.get_episode(db, episode_id)
    if episode is None:
        raise ValueError(f"unknown episode_id {episode_id!r}")
    if not isinstance(decision, str) or not decision.startswith(_DECISION_PREFIXES):
        raise ValueError(
            f"decision must start with one of {_DECISION_PREFIXES}; got {decision!r}"
        )
    if dataclasses.is_dataclass(fitness):
        fitness_dict = dataclasses.asdict(fitness)
    else:
        fitness_dict = dict(fitness)
    corrected = episode.get("corrected_metrics") or {}
    costs = episode.get("costs") or {}
    metrics = episode.get("metrics") or {}
    metrics_json = {
        "corrected": _strip_oos(dict(corrected)),
        "costs": _strip_oos(dict(costs)),
        "simulated": bool(metrics.get("simulated", False)),
    }
    feedback_id = new_evolution_feedback_id()
    try:
        db.execute(
            "INSERT INTO evolution_feedback (feedback_id, run_id, hypothesis_id,"
            " episode_id, parent_ids_json, observations, hypothesis_evaluation,"
            " new_hypothesis, metrics_json, fitness_json, decision, model_id,"
            " prompt_sha, as_of, data_through, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                feedback_id,
                run_id,
                hypothesis_id,
                episode_id,
                json.dumps(list(parent_ids), sort_keys=True),
                narration.get("observations"),
                narration.get("hypothesis_evaluation"),
                narration.get("new_hypothesis"),
                json.dumps(metrics_json, sort_keys=True),
                json.dumps(fitness_dict, sort_keys=True, default=str),
                decision,
                model_id,
                prompt_sha,
                episode.get("as_of"),
                episode.get("data_through"),
                now_iso(),
            ),
        )
    except sqlite3.IntegrityError as exc:
        raise ToolError(f"cannot record evolution feedback: {exc}") from exc
    row = db.conn.execute(
        "SELECT * FROM evolution_feedback WHERE feedback_id = ?", (feedback_id,)
    ).fetchone()
    assert row is not None
    return dict(row)
