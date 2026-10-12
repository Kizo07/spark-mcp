"""Explore/exploit parent selectors over the DAG (plan §2.C2).

Deterministic port of the one good RD selector idea (`SOTAJump`: recent
SOTA hit-rate below threshold → start a new root), mapped onto spark's own
recorded IS-gated promotion verdicts. The coin flip stays behind: these
policies are pure functions of (ledger state, config, seeded rng) and the
current policies consume no rng draws at all — `rng` is reserved for
explicitly documented tie-breaks, of which there are currently none.

Hit-rate derives from recorded promotion verdicts only — OOS values never
enter selector inputs (constraint 5). Selectors never write: on
`sota_jump_new_root` the E1 run driver creates the new line (via
`lineage.create_line`) and proposes the root on it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..config import Config
from ..errors import ToolError
from ..storage import Database
from . import lineage


@dataclass(frozen=True)
class ParentSelection:
    hypothesis_id: str | None  # None = propose a new root
    episode_id: str | None
    policy: str  # 'exploit_head' | 'sota_jump_new_root' | 'jump_back'
    reason: str


def _recent_decisions(db: Database, line_id: str, window: int) -> list[str]:
    """Newest-first decision strings for runs on this line, capped at window."""
    rows = db.conn.execute(
        "SELECT f.decision FROM evolution_feedback f"
        " JOIN evolution_runs r ON f.run_id = r.run_id"
        " WHERE r.line_id = ? ORDER BY f.rowid DESC LIMIT ?",
        (line_id, max(window, 0)),
    ).fetchall()
    return [str(r["decision"]) for r in rows]


def recent_hit_rate(db: Database, line_id: str, window: int) -> float | None:
    """Share of `promoted` verdicts among the last `window` feedback rows on
    this line (fewer when less history exists), or None with no history.
    """
    decisions = _recent_decisions(db, line_id, window)
    if not decisions:
        return None
    promoted = sum(1 for d in decisions if d.startswith("promoted"))
    return promoted / len(decisions)


def select_parent(
    db: Database, cfg: Config, line_id: str, rng: random.Random
) -> ParentSelection:
    """Exploit the line head by default; jump to a new root when the recent
    SOTA hit-rate sags below `evolution_sota_jump_threshold` over the last
    `evolution_sota_jump_window` recorded verdicts (strict `<`: exactly at
    the threshold still exploits). A `window <= 0` disables the jump check.

    The returned `reason` is recorded on the feedback row by the driver.
    Consumes no rng draws (see module docstring). Raises ValueError on an
    unknown line.
    """
    del rng  # reserved for documented tie-breaks; current policies draw nothing
    line = lineage.get_line(db, line_id)
    if line is None:
        raise ValueError(f"unknown line_id {line_id!r}")
    window = int(cfg.research.evolution_sota_jump_window)
    threshold = float(cfg.research.evolution_sota_jump_threshold)
    rate = recent_hit_rate(db, line_id, window) if window >= 1 else None
    if rate is not None and rate < threshold:
        return ParentSelection(
            None,
            None,
            "sota_jump_new_root",
            f"SOTA hit-rate {rate:.3f} over the last {window} verdict(s) is below"
            f" threshold {threshold:.3f}; start a new root on a new line",
        )
    head_hyp = line.get("head_hypothesis_id")
    head_ep = line.get("head_episode_id")
    if head_hyp is None:
        return ParentSelection(
            None,
            None,
            "exploit_head",
            f"line {line_id} has no head yet; propose the root hypothesis on this line",
        )
    rate_txt = f"{rate:.3f}" if rate is not None else "n/a (no verdicts yet)"
    return ParentSelection(
        str(head_hyp),
        str(head_ep) if head_ep is not None else None,
        "exploit_head",
        f"exploit line head {head_hyp} (SOTA hit-rate {rate_txt})",
    )


def jump_back(db: Database, line_id: str, n_sota: int = 2) -> ParentSelection:
    """Resume from the n-th-last passing ancestor of the line head (a branch
    point): passing ancestors are ordered oldest-first by their latest
    promoted feedback row and the n-th from the end is selected. Fewer than
    `n_sota` passing ancestors clamps to the earliest available one (noted
    in the reason). Deterministic ancestor walk; no randomness.

    Raises ValueError on an unknown line or `n_sota < 1`; ToolError when the
    line has no head or no passing ancestor exists.
    """
    if n_sota < 1:
        raise ValueError(f"n_sota must be >= 1; got {n_sota!r}")
    line = lineage.get_line(db, line_id)
    if line is None:
        raise ValueError(f"unknown line_id {line_id!r}")
    head = line.get("head_hypothesis_id")
    if head is None:
        raise ToolError(f"line {line_id} has no head to jump back from")
    passing: list[tuple[int, str, str]] = []  # (recency rowid, hyp, episode)
    for ancestor_id in lineage.ancestors(db, str(head)):
        row = db.conn.execute(
            "SELECT rowid, episode_id FROM evolution_feedback"
            " WHERE hypothesis_id = ? AND decision LIKE 'promoted%'"
            " ORDER BY rowid DESC LIMIT 1",
            (ancestor_id,),
        ).fetchone()
        if row is not None:
            passing.append((int(row["rowid"]), ancestor_id, str(row["episode_id"])))
    if not passing:
        raise ToolError(f"line {line_id} has no passing ancestor to jump back to")
    passing.sort(key=lambda t: t[0])
    idx = max(0, len(passing) - n_sota)
    _, hyp, ep = passing[idx]
    clamped = ""
    if idx == 0 and len(passing) < n_sota:
        clamped = f" (only {len(passing)} passing ancestor(s); clamped to earliest)"
    return ParentSelection(
        hyp,
        ep,
        "jump_back",
        f"resume from the {n_sota}-th-last passing ancestor {hyp} (episode {ep})"
        f" of head {head}{clamped}",
    )
