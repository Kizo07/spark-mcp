"""Minimal curriculum (plan §18.1 Phase 7): consolidation gaps -> exercise
generator -> replay evaluation against the CURRENT CHAMPION.

The v1 scope is deliberately small: an exercise is synthesized from ONE real
gap of a consolidation report (§13 artifact, consumed by id — the same
provenance path suggest_next uses). The task prompt names the gap's family,
evidence, and suggested action; the evaluation maps the gap family to a
seeded replay fixture (seeds/replay_fixtures.yaml, plan §11) and reuses the
skills.run_replay DETERMINISTIC scorer (evolve/gepa.fixture_score — the exact
function run_replay calls with no LLM configured; not a fork) to score the
current champion skill's procedure_steps against that fixture.

Deferral is explicit (G5 accepts "≥1 curriculum exercise generated from a
real gap or explicit deferral"): when the report has no gaps, when the chosen
gap carries no family, or when no replay fixture maps the gap family, the
result is {deferred: true, reason} naming what was missing — never a silent
empty exercise. When no champion skill exists the exercise still generates
with champion_score=None (there is simply no incumbent to beat yet).
"""

from __future__ import annotations

import json
import random
from typing import Any

from . import consolidate as consolidate_mod
from . import skills as skills_mod
from . import trajectory
from .config import Config
from .evolve.gepa import fixture_score
from .ids import new_exercise_id
from .storage import Database
from .timeutil import now_iso

# Gap severities ranked highest-first (same vocabulary as consolidate.py).
_SEVERITY_RANK = {"high": 3, "medium": 2, "info": 1}


def _current_champion(db: Database) -> dict | None:
    """The current champion skill (any name): newest version wins. The
    curriculum v1 scores THE champion — the incumbent the evolution loop
    must beat — not a family-matched one (role/family matching is the
    recorded upgrade path once skills carry family metadata)."""
    row = db.conn.execute(
        "SELECT * FROM skills WHERE status = 'champion' ORDER BY version DESC, created_at DESC"
        " LIMIT 1"
    ).fetchone()
    return dict(row) if row else None


def generate_exercise(
    db: Database,
    cfg: Config,
    gap_report_id: str | None = None,
    seed: int | None = None,
    llm: Any | None = None,
) -> dict:
    """Generate one curriculum exercise from a real consolidation gap and
    replay-score the current champion against it.

    With no gap_report_id a fresh consolidation pass runs first (the natural
    cron flow: consolidate -> teach). Deferral result when no usable gap or
    fixture mapping exists. A trajectory event records every call.
    """
    del llm  # v1 prompt synthesis is template-based; the param reserves the seam
    if gap_report_id is None:
        report = consolidate_mod.consolidate(db, cfg)
        gap_report_id = report["report_id"]
    else:
        report = consolidate_mod.load_report(cfg, gap_report_id)
    gaps = list(report.get("gaps", []))
    if not gaps:
        return {
            "deferred": True,
            "reason": f"consolidation report {gap_report_id} carries no open gaps",
            "gap_report_id": gap_report_id,
        }

    # Choose among the highest-severity family-carrying gaps. The tier is
    # sorted by (family, gap_id) so the seeded draw is stable across runs —
    # gap ids are random ULIDs, so report order alone is not reproducible.
    candidates = [
        g for g in gaps if g.get("family") and _SEVERITY_RANK.get(g["severity"], 0) > 0
    ]
    if not candidates:
        return {
            "deferred": True,
            "reason": "no gap carries a research family to build an exercise on",
            "gap_report_id": gap_report_id,
        }
    candidates.sort(key=lambda g: (str(g["family"]), str(g["gap_id"])))
    top = max(_SEVERITY_RANK.get(g["severity"], 0) for g in candidates)
    tier = [g for g in candidates if _SEVERITY_RANK.get(g["severity"], 0) == top]
    rng = random.Random(seed)
    gap = tier[rng.randrange(len(tier))] if seed is not None else tier[0]
    family = str(gap["family"])

    fixtures = skills_mod.load_fixtures()
    fixture = next((f for f in fixtures if f.get("family") == family), None)
    if fixture is None:
        return {
            "deferred": True,
            "reason": (
                f"no replay fixture maps family {family!r}"
                f" (seeds/replay_fixtures.yaml covers"
                f" {sorted({f['family'] for f in fixtures})})"
            ),
            "gap_report_id": gap_report_id,
            "source_gap_id": gap["gap_id"],
            "family": family,
        }

    exercise_id = new_exercise_id()
    evidence = gap.get("evidence", {})
    task_prompt = (
        f"Research exercise ({family}). A consolidation pass found an open gap"
        f" [{gap['kind']}, severity {gap['severity']}]:"
        f" {gap.get('suggested_action', 'address the gap')}."
        f" Evidence: {evidence.get('claim') or evidence}."
        " Produce a defensible step-by-step research procedure for the mapped"
        f" replay task: {fixture['prompt'].strip()}"
    )
    exercise = {
        "exercise_id": exercise_id,
        "generated_at": now_iso(),
        "gap_report_id": gap_report_id,
        "source_gap_id": gap["gap_id"],
        "family": family,
        "task_prompt": task_prompt,
        "evaluation": {
            "fixture_task_id": fixture["task_id"],
            "rubric": list(fixture["rubric"]),
            "scorer": "spark_mcp.evolve.gepa.fixture_score (deterministic, the"
            " run_replay no-LLM scorer, plan §11)",
        },
    }

    champion = _current_champion(db)
    if champion is None:
        exercise["champion"] = None
        exercise["champion_score"] = None
        exercise["champion_note"] = "no champion skill on record; nothing to beat yet"
    else:
        # procedure_steps live in the lineage JSON (§11 body-storage contract).
        lineage = json.loads(champion["lineage"] or "{}")
        score = fixture_score(lineage["procedure_steps"], fixture)
        exercise["champion"] = {
            "skill_id": champion["skill_id"],
            "name": champion["name"],
            "version": int(champion["version"]),
        }
        exercise["champion_score"] = score

    trajectory.record(
        db,
        exercise_id,
        "spark",
        "tool_call",
        {
            "tool": "generate_exercise",
            "gap_report_id": gap_report_id,
            "source_gap_id": gap["gap_id"],
            "family": family,
            "fixture_task_id": fixture["task_id"],
            "champion_score": exercise["champion_score"],
            "seed": seed,
        },
    )
    return exercise


__all__ = ["generate_exercise"]
