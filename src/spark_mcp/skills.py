"""Skill evolution service (plan §11: tools 16-18/22, §19 M11, §21).

Lifecycle (§11): draft -> challenger -> champion/rejected/retired, with
rollback targets. Key contracts:

- Body storage: the plan §5.1 skills DDL names no body column, and §3.3
  forbids filesystem deployment, so the v1 procedure body lives INSIDE the
  lineage JSON under "procedure_steps"/"references"; body_uri stays a
  registry-internal pointer of the form spark://skills/{name}/{version}
  (never a file path). The Gateway manifest carries the body at promotion.
- Write-path (C7): promotion writes go ONLY to the QuantFlow Gateway
  skill-write API (promote_via_gateway). No endpoint, or a peer that does
  not acknowledge the versioned manifest contract, raises the named error
  GatewayWriteBlocked; the champion stays staged in Spark's registry
  (status='champion', promotion_status='staged_in_spark'), which remains
  the system of record until a Gateway appears (§11 blocked-state handling).
- Human gate (M11): champion promotion is auto-with-human-notify — the gate
  records the promotion AND an approvals row with decided_by='gate'; a human
  may veto via approve (tool 22). Skill ADOPTION into a live book is always
  human-required and out of band: approve(target_type='adoption') records
  the human authorization; the actual book edit happens in QuantFlow.
- Replay discipline (§11/§21): every verdict reports held-in vs held-out
  scores and heldout_gap; a gap over cfg.research.replay_heldout_gap_alarm
  flags the fixture set for refresh review. Held-out fixtures score at
  promotion time, never as evolution feedback.
- Credit assignment (v1 minimal form): on promotion, skills.credit =
  {episode_id: 1/n} over the candidate's source episodes, and the champion
  version is appended to each source episode's skill_versions JSON.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from . import trajectory
from .config import Config
from .errors import ToolError
from .evolve import gepa as gepa_mod
from .ids import new_approval_id, new_candidate_id, new_skill_id
from .llm import build_llm
from .storage import Database
from .timeutil import now_iso

REPLAY_FIXTURES_PATH = Path(__file__).resolve().parent / "seeds" / "replay_fixtures.yaml"

ROLES = ("quant", "ml", "nlp", "risk", "lead")
MUTATION_OPS = ("repair", "split", "merge", "paraphrase", "gepa")
# Verification-first words: the degraded paraphrase operator partitions on
# these to reorder a procedure so checks come after the main flow.
_VERIFY_WORDS = ("verify", "check", "guard", "record", "validate", "confirm", "test")

GATEWAY_MANIFEST_SCHEMA_VERSION = 1
# skills.promotion_status values written by this module.
PROMOTION_STAGED_IN_SPARK = "staged_in_spark"
PROMOTION_DEPLOYED = "deployed_to_gateway"


class GatewayWriteBlocked(ToolError):
    """C7/§11 blocked state: the only skill write-path is the QuantFlow
    Gateway skill-write API. Raised when no endpoint is configured or the
    endpoint does not acknowledge the versioned manifest contract. Carries a
    structured payload; the champion stays staged in Spark's registry."""

    def __init__(self, payload: dict):
        self.payload = payload
        super().__init__(
            "gateway write blocked: " + json.dumps(payload, sort_keys=True)
        )


# -- fixtures -----------------------------------------------------------------


def load_fixtures(path: str | Path | None = None) -> list[dict]:
    """Load the replay fixtures (plan §11: seeded at Phase 6 start, grown from
    real failures thereafter). Each fixture: {task_id, prompt, family,
    held_in, rubric: [...], reference_keywords: [...]}."""
    p = Path(path) if path else REPLAY_FIXTURES_PATH
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"replay fixture file {p} must contain a non-empty YAML list")
    for fixture in data:
        for key in ("task_id", "prompt", "family", "rubric", "reference_keywords"):
            if key not in fixture:
                raise ValueError(f"replay fixture in {p} is missing {key!r}")
    return data


# -- small row helpers --------------------------------------------------------


def _skill_row(db: Database, name: str, version: int) -> sqlite3.Row | None:
    return db.conn.execute(
        "SELECT * FROM skills WHERE name = ? AND version = ?", (name, version)
    ).fetchone()


def _parent_skill(db: Database, skill_name: str) -> sqlite3.Row:
    """The version a challenger derives from: the newest non-rejected,
    non-retired row (champion preferred by construction, else draft or the
    current rollback_target)."""
    row = db.conn.execute(
        "SELECT * FROM skills WHERE name = ? AND status IN ('champion', 'draft',"
        " 'rollback_target') ORDER BY version DESC LIMIT 1",
        (skill_name,),
    ).fetchone()
    if row is None:
        raise ToolError(f"unknown skill {skill_name!r} (no draft/champion/rollback row)")
    return row


def _next_version(db: Database, name: str) -> int:
    row = db.conn.execute(
        "SELECT COALESCE(MAX(version), 0) + 1 AS v FROM skills WHERE name = ?", (name,)
    ).fetchone()
    return int(row["v"])


def _newest_pending_candidate(db: Database, skill_name: str) -> sqlite3.Row | None:
    return db.conn.execute(
        "SELECT * FROM skill_candidates WHERE skill_name = ? AND verdict = 'pending'"
        " ORDER BY rowid DESC LIMIT 1",
        (skill_name,),
    ).fetchone()


def _lineage(row: sqlite3.Row) -> dict:
    return json.loads(row["lineage"] or "{}")


def _challenger_for(
    db: Database, skill_name: str, parent_version: int | None,
    candidate: sqlite3.Row | None = None,
) -> sqlite3.Row | None:
    """The skills row a candidate evaluates. EXACT linkage first (review
    blocker B2): v9+ candidates carry challenger_skill_id, immune to lineage
    forks (v2 retired, v3 challenger from the same parent — a veto naming
    the v2 candidate must never touch the live v3). Fallback for pre-v9
    rows: newest row of the name whose lineage parents from parent_version.
    (Usually parent_version + 1; after a rejected challenger the next free
    version is higher — versions are per-name and never reused.)"""
    if candidate is not None and candidate["challenger_skill_id"] is not None:
        row = db.conn.execute(
            "SELECT * FROM skills WHERE skill_id = ?",
            (candidate["challenger_skill_id"],),
        ).fetchone()
        if row is not None:
            return row
    rows = db.conn.execute(
        "SELECT * FROM skills WHERE name = ? ORDER BY version DESC", (skill_name,)
    ).fetchall()
    matches = [r for r in rows if _lineage(r).get("parent_version") == parent_version]
    if len(matches) > 1:
        # Legacy rows (pre-v10) with no challenger_skill_id can fork: refuse
        # rather than guess (re-review nit — ambiguity must not pick one).
        raise ToolError(
            f"ambiguous lineage for {skill_name!r} parent {parent_version}:"
            f" {len(matches)} candidate rows; resolve via a v10+ candidate id"
        )
    return matches[0] if matches else None


def _record_effect(db: Database, tool: str, client_id: str, subject_id: str, result: dict) -> None:
    db.execute(
        "INSERT OR IGNORE INTO tool_effects (client_id, tool, subject_id, result, created_at)"
        " VALUES (?, ?, ?, ?, ?)",
        (client_id, tool, subject_id, json.dumps(result, sort_keys=True), now_iso()),
    )


def _replay_effect(db: Database, tool: str, client_id: str) -> dict | None:
    row = db.conn.execute(
        "SELECT result FROM tool_effects WHERE client_id = ? AND tool = ?", (client_id, tool)
    ).fetchone()
    return json.loads(row["result"]) if row else None


def _has_trajectory_evidence(db: Database, episode_id: str) -> bool:
    """Tier-1 events key by run_id = episode_id; Tier-2 hook summaries may key
    by their QuantFlow run_id and only reference the episode inside the
    payload — either satisfies the distill evidence check (C10: Tier-1 +
    Tier-2, Tier-2-only allowed)."""
    row = db.conn.execute(
        "SELECT 1 FROM trajectory_events WHERE run_id = ? OR payload LIKE ? LIMIT 1",
        (episode_id, f'%"{episode_id}"%'),
    ).fetchone()
    return row is not None


def _check_episode_evidence(db: Database, episode_ids: list[str]) -> None:
    for episode_id in episode_ids:
        row = db.conn.execute(
            "SELECT 1 FROM episodes WHERE episode_id = ?", (episode_id,)
        ).fetchone()
        if row is None:
            raise ToolError(f"unknown episode_id {episode_id!r}")
        if not _has_trajectory_evidence(db, episode_id):
            raise ToolError(
                f"episode {episode_id!r} carries no trajectory evidence"
                " (Tier-1 events or a Tier-2 hook summary are required to distill)"
            )


def _tier2_failure_sketches(db: Database, episode_ids: list[str]) -> list[dict]:
    """Failure sketches from the Tier-2 hook summaries of the given episodes."""
    if not episode_ids:
        return []
    clauses = " OR ".join("(run_id = ? OR payload LIKE ?)" for _ in episode_ids)
    params: list[str] = []
    for episode_id in episode_ids:
        params += [episode_id, f'%"{episode_id}"%']
    rows = db.conn.execute(
        f"SELECT payload FROM trajectory_events WHERE kind = 'hook_summary' AND ({clauses})",
        params,
    ).fetchall()
    sketches: list[dict] = []
    for row in rows:
        payload = json.loads(row["payload"])
        sketches.extend(payload.get("failure_sketches") or [])
    return sketches


# -- tool 16: distill_skill -----------------------------------------------------


def distill_skill(
    db: Database,
    cfg: Config,
    episode_ids: list[str],
    name: str,
    role: str | None,
    procedure_steps: list[str],
    references: list[str] | None = None,
    client_id: str | None = None,
) -> dict:
    """Draft candidate from validated trajectories (plan §6.3 tool 16, C10).
    Every episode must exist and carry trajectory evidence (Tier-1 events or
    a Tier-2 hook summary). Writes a skills row status='draft' at the next
    free version for `name`; credit stays {} until promotion (§11). The body
    (procedure_steps/references) lives in the lineage JSON — see module
    docstring. Idempotent on client_id."""
    if client_id:
        existing = _replay_effect(db, "distill_skill", client_id)
        if existing:
            return existing
    if not isinstance(name, str) or not name.strip():
        raise ToolError("skill name must be a non-empty string")
    if role is not None and role not in ROLES:
        raise ToolError(f"role must be one of {ROLES}, got {role!r}")
    if (
        not isinstance(procedure_steps, list)
        or not procedure_steps
        or any(not isinstance(s, str) or not s.strip() for s in procedure_steps)
    ):
        raise ToolError("procedure_steps must be a non-empty list of non-empty strings")
    references = references or []
    if any(not isinstance(r, str) or not r.strip() for r in references):
        raise ToolError("references must be a list of non-empty strings")
    if not episode_ids:
        raise ToolError("episode_ids must be non-empty")
    _check_episode_evidence(db, list(episode_ids))

    version = _next_version(db, name)
    skill_id = new_skill_id()
    lineage = {
        "source_episode_ids": list(episode_ids),
        "procedure_steps": list(procedure_steps),
        "references": list(references),
        "distilled_at": now_iso(),
    }
    with db.transaction():
        db.execute(
            "INSERT INTO skills ("
            " skill_id, name, version, role, body_uri, lineage, status, credit,"
            " benchmark_history, promotion_status, created_at"
            ") VALUES (?, ?, ?, ?, ?, ?, 'draft', '{}', '[]', 'auto', ?)",
            (
                skill_id,
                name,
                version,
                role,
                f"spark://skills/{name}/{version}",
                json.dumps(lineage, sort_keys=True),
                now_iso(),
            ),
        )
        trajectory.record(
            db,
            episode_ids[0],
            "spark",
            "tool_call",
            {"tool": "distill_skill", "skill_id": skill_id, "name": name, "version": version},
        )
    result = {
        "skill_id": skill_id,
        "name": name,
        "version": version,
        "role": role,
        "status": "draft",
        "body_uri": f"spark://skills/{name}/{version}",
        "source_episode_ids": list(episode_ids),
    }
    if client_id:
        _record_effect(db, "distill_skill", client_id, skill_id, result)
    return result


# -- tool 17: propose_skill_challenger ------------------------------------------


def _op_repair(
    db: Database, parent: sqlite3.Row, source_episode_ids: list[str]
) -> tuple[list[str], dict]:
    """repair: append one step addressing the most common failure_sketch
    error_class across the source episodes' Tier-2 hook events. Refuses when
    there is no Tier-2 failure evidence to repair from."""
    sketches = _tier2_failure_sketches(db, source_episode_ids)
    classes = [s["error_class"].strip() for s in sketches if s.get("error_class", "").strip()]
    if not classes:
        raise ToolError(
            "repair needs Tier-2 failure sketches (hook_summary events with"
            " non-empty error_class) on the source episodes; none found"
        )
    error_class, count = Counter(classes).most_common(1)[0]
    steps = list(_lineage(parent)["procedure_steps"]) + [
        f"Guard against {error_class}: check data coverage, preconditions, and"
        f" inputs before executing this procedure (observed in {count} source"
        " episode(s))."
    ]
    return steps, {"repaired_error_class": error_class, "occurrences": count}


def _op_split(parent: sqlite3.Row) -> tuple[list[str], dict]:
    """split: break a multi-goal procedure into the first coherent
    sub-procedure — deterministically the first ceil(n/2) steps. Refuses a
    single-step procedure (nothing to split)."""
    steps = list(_lineage(parent)["procedure_steps"])
    if len(steps) < 2:
        raise ToolError("split needs a multi-step procedure (>= 2 steps)")
    half = (len(steps) + 1) // 2
    return steps[:half], {"dropped_steps": len(steps) - half}


def _op_merge(db: Database, parent: sqlite3.Row) -> tuple[list[str], dict]:
    """merge: combine the parent's steps with the newest other skill sharing
    its role (any other live skill when no role-mate exists), deduped
    case-insensitively, parent order first."""
    parent_lin = _lineage(parent)
    partner = db.conn.execute(
        "SELECT * FROM skills WHERE name != ? AND status != 'retired'"
        " AND role IS NOT NULL AND role = ? ORDER BY version DESC LIMIT 1",
        (parent["name"], parent["role"]),
    ).fetchone()
    if partner is None:
        partner = db.conn.execute(
            "SELECT * FROM skills WHERE name != ? AND status != 'retired'"
            " ORDER BY version DESC LIMIT 1",
            (parent["name"],),
        ).fetchone()
    if partner is None:
        raise ToolError("merge needs a second live skill to combine with; none exists")
    seen: set[str] = set()
    merged: list[str] = []
    for step in list(parent_lin["procedure_steps"]) + list(_lineage(partner)["procedure_steps"]):
        key = " ".join(step.lower().split())
        if key not in seen:
            seen.add(key)
            merged.append(step)
    return merged, {"merged_with": f"{partner['name']}:{partner['version']}"}


def _op_paraphrase(parent: sqlite3.Row, llm: Any | None) -> tuple[list[str], dict]:
    """paraphrase: LLM rewrite when an LLM is configured; otherwise the
    documented degraded path — a deterministic reorder+clarify pass:
    verification steps (first word in _VERIFY_WORDS) move to the end with
    order otherwise stable, and every step is whitespace-normalized with a
    trailing period."""
    steps = list(_lineage(parent)["procedure_steps"])
    if llm is not None:
        steps_text = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
        prompt = (
            "Rewrite each step of this research procedure more precisely and"
            " concretely, preserving its meaning and number of steps.\n"
            f"{steps_text}\n"
            "Return a JSON array of strings, one rewritten step in the original order."
        )
        reply = llm.complete(prompt)
        try:
            start, end = reply.index("["), reply.rindex("]") + 1
            parsed = json.loads(reply[start:end])
            if (
                isinstance(parsed, list)
                and parsed
                and all(isinstance(s, str) and s.strip() for s in parsed)
            ):
                return [s.strip() for s in parsed], {"paraphrase_mode": "llm"}
        except (ValueError, json.JSONDecodeError):
            pass  # fall through to the degraded path, documented above
    main = [s for s in steps if s.split()[0].lower().strip(",.:") not in _VERIFY_WORDS]
    checks = [s for s in steps if s.split()[0].lower().strip(",.:") in _VERIFY_WORDS]
    clarified = [s.strip().rstrip(".") + "." for s in main + checks]
    return clarified, {"paraphrase_mode": "degraded_reorder_clarify"}


def _op_gepa(
    cfg: Config, parent: sqlite3.Row, source_episode_ids: list[str]
) -> tuple[list[str], dict]:
    """gepa: run the native minimal GEPA loop (evolve/gepa.py) seeded with the
    parent's procedure, over the HELD-IN fixtures only (held-out fixtures are
    promotion-time evidence, never evolution feedback, §21). Deterministic:
    the RNG seed is derived from (skill_name, parent_version)."""
    fixtures = [f for f in load_fixtures() if f.get("held_in", True)]
    parent_lin = _lineage(parent)
    # Seed derived from the parent's content, not its name: the same
    # procedure evolves identically wherever it appears.
    seed = int.from_bytes(
        hashlib.sha1(
            json.dumps(parent_lin["procedure_steps"], sort_keys=True).encode()
        ).digest()[:8],
        "big",
    )
    result = gepa_mod.run_gepa(
        [
            {
                "name": parent["name"],
                "procedure_steps": list(parent_lin["procedure_steps"]),
                "references": list(parent_lin.get("references") or []),
            }
        ],
        fixtures,
        population_size=cfg.research.gepa_population_size,
        generations=cfg.research.gepa_generations,
        pareto_top_k=cfg.research.gepa_pareto_top_k,
        seed=seed,
    )
    return list(result["best"]["procedure_steps"]), {
        "gepa": {
            "held_in_score": result["best"]["held_in_score"],
            "pareto_front_size": len(result["pareto_front"]),
            "population_size": result["population_size"],
            "generations": result["generations"],
            "lineage_episode_ids": list(source_episode_ids),
        }
    }


def propose_skill_challenger(
    db: Database,
    cfg: Config,
    skill_name: str,
    op: str,
    source_episode_ids: list[str] | None = None,
    client_id: str | None = None,
    llm: Any | None = None,
) -> dict:
    """Challenger via mutation operators / mini-GEPA (plan §6.3 tool 17).
    Operators (deterministic): repair, split, merge, paraphrase, gepa — see
    the _op_* docstrings. Writes the challenger skills row (status=
    'challenger', version = next free for the name) and a pending
    skill_candidates row.
    Idempotent on client_id."""
    if client_id:
        existing = _replay_effect(db, "propose_skill_challenger", client_id)
        if existing:
            return existing
    if op not in MUTATION_OPS:
        raise ToolError(f"unknown op {op!r}; expected one of {MUTATION_OPS}")
    parent = _parent_skill(db, skill_name)
    parent_lin = _lineage(parent)
    source = list(source_episode_ids or parent_lin.get("source_episode_ids") or [])
    pending = db.conn.execute(
        "SELECT 1 FROM skill_candidates WHERE skill_name = ? AND parent_version = ?"
        " AND verdict = 'pending'",
        (skill_name, int(parent["version"])),
    ).fetchone()
    if pending is not None:
        raise ToolError(
            f"a challenger is already pending for {skill_name}:{parent['version']}"
            " (decide it via run_replay + decide_promotion before proposing another)"
        )

    if llm is None and op == "paraphrase":
        llm = build_llm(cfg)

    if op == "repair":
        steps, extra = _op_repair(db, parent, source)
    elif op == "split":
        steps, extra = _op_split(parent)
    elif op == "merge":
        steps, extra = _op_merge(db, parent)
    elif op == "paraphrase":
        steps, extra = _op_paraphrase(parent, llm)
    else:
        steps, extra = _op_gepa(cfg, parent, source)

    # Version = next free for the name (never reused: a rejected challenger
    # keeps its version, so parent+1 is not always free).
    version = _next_version(db, skill_name)
    skill_id = new_skill_id()
    candidate_id = new_candidate_id()
    lineage = {
        "parent_version": int(parent["version"]),
        "mutation_op": op,
        "source_episode_ids": source,
        "procedure_steps": steps,
        "references": list(parent_lin.get("references") or []),
        **extra,
    }
    with db.transaction():
        db.execute(
            "INSERT INTO skills ("
            " skill_id, name, version, role, body_uri, lineage, status, credit,"
            " benchmark_history, promotion_status, created_at"
            ") VALUES (?, ?, ?, ?, ?, ?, 'challenger', '{}', '[]', 'auto', ?)",
            (
                skill_id,
                skill_name,
                version,
                parent["role"],
                f"spark://skills/{skill_name}/{version}",
                json.dumps(lineage, sort_keys=True),
                now_iso(),
            ),
        )
        db.execute(
            "INSERT INTO skill_candidates ("
            " candidate_id, skill_name, parent_version, source_episode_ids,"
            " mutation_op, verdict, challenger_skill_id"
            ") VALUES (?, ?, ?, ?, ?, 'pending', ?)",
            (
                candidate_id,
                skill_name,
                int(parent["version"]),
                json.dumps(source),
                op,
                skill_id,
            ),
        )
        trajectory.record(
            db,
            skill_name,
            "spark",
            "tool_call",
            {
                "tool": "propose_skill_challenger",
                "candidate_id": candidate_id,
                "skill_id": skill_id,
                "op": op,
                "parent_version": int(parent["version"]),
            },
        )
    result = {
        "candidate_id": candidate_id,
        "skill_id": skill_id,
        "skill_name": skill_name,
        "version": version,
        "parent_version": int(parent["version"]),
        "op": op,
        "procedure_steps": steps,
        **extra,
    }
    if client_id:
        _record_effect(db, "propose_skill_challenger", client_id, candidate_id, result)
    return result


# -- tool 18: run_replay ---------------------------------------------------------


def _cache_key(skill_name: str, version: int, task_id: str, model_id: str) -> str:
    raw = f"{skill_name}|{version}|{task_id}|{model_id}".encode()
    return hashlib.sha1(raw).hexdigest()  # noqa: S324 - dedupe key, not security


def _fixture_score_cached(
    db: Database, row: sqlite3.Row, fixture: dict, llm: Any | None
) -> float:
    if llm is None:
        return gepa_mod.fixture_score(_lineage(row)["procedure_steps"], fixture)
    model_id = llm.model_id()
    key = _cache_key(row["name"], int(row["version"]), fixture["task_id"], model_id)
    cached = db.conn.execute(
        "SELECT score FROM replay_score_cache WHERE cache_key = ?", (key,)
    ).fetchone()
    if cached is not None:
        return float(cached["score"])
    score = gepa_mod.fixture_score(_lineage(row)["procedure_steps"], fixture, llm=llm)
    db.execute(
        "INSERT OR IGNORE INTO replay_score_cache ("
        " cache_key, skill_name, version, task_id, model_id, score, created_at"
        ") VALUES (?, ?, ?, ?, ?, ?, ?)",
        (key, row["name"], int(row["version"]), fixture["task_id"], model_id, score, now_iso()),
    )
    return score


def _mean(scores: dict[str, float]) -> float:
    return sum(scores.values()) / len(scores) if scores else 0.0


def run_replay(
    db: Database,
    cfg: Config,
    skill_name: str | None = None,
    suite: str | None = None,
    task_ids: list[str] | None = None,
    seed: int | None = None,
    llm: Any | None = None,
) -> dict:
    """Replay-suite verdicts (plan §6.3 tool 18, §11): scores skill versions
    against the seeded fixtures and reports the held-in/held-out split with
    heldout_gap in every verdict. Without an LLM the score is deterministic
    keyword/rubric coverage (evolve/gepa.py); with one, an LLM judge 0-1 per
    rubric item, cached per (skill_version, task_id, model_id). Scores land
    on the newest pending candidate for each skill. When no pending candidate
    exists the latest version (usually the champion) is scored as a
    regression run and no candidate row is touched. `seed` is accepted for
    run provenance — deterministic mode is seed-stable by construction."""
    fixtures = load_fixtures()
    if task_ids:
        wanted = set(task_ids)
        selected = [f for f in fixtures if f["task_id"] in wanted]
        unknown = wanted - {f["task_id"] for f in selected}
        if unknown:
            raise ToolError(f"unknown replay task_id(s): {sorted(unknown)}")
    elif suite:
        selected = [f for f in fixtures if f["family"] == suite]
        if not selected:
            raise ToolError(f"suite {suite!r} matches no replay fixtures")
    else:
        selected = list(fixtures)
    held_in = [f for f in selected if f.get("held_in", True)]
    held_out = [f for f in selected if not f.get("held_in", True)]

    if skill_name is not None:
        names = [skill_name]
    else:
        names = [
            r["skill_name"]
            for r in db.conn.execute(
                "SELECT DISTINCT skill_name FROM skill_candidates WHERE verdict = 'pending'"
                " ORDER BY skill_name"
            ).fetchall()
        ]
        if not names:
            raise ToolError("no pending skill candidates to replay")

    if llm is None:
        llm = build_llm(cfg)

    verdicts: dict[str, dict] = {}
    for name in names:
        candidate = _newest_pending_candidate(db, name)
        if candidate is not None:
            row = _challenger_for(db, name, candidate["parent_version"], candidate=candidate)
            scored_as = "challenger"
        else:
            row = db.conn.execute(
                "SELECT * FROM skills WHERE name = ? ORDER BY version DESC LIMIT 1", (name,)
            ).fetchone()
            scored_as = "latest"
        if row is None:
            raise ToolError(f"unknown skill {name!r}")
        version = int(row["version"])
        in_scores = {
            f["task_id"]: _fixture_score_cached(db, row, f, llm) for f in held_in
        }
        out_scores = {
            f["task_id"]: _fixture_score_cached(db, row, f, llm) for f in held_out
        }
        gap = abs(_mean(in_scores) - _mean(out_scores))
        verdict = {
            "skill_name": name,
            "version": version,
            "scored_as": scored_as,
            "held_in": in_scores,
            "held_out": out_scores,
            "n_held_in": len(in_scores),
            "n_held_out": len(out_scores),
            "heldout_gap": gap,
            "gap_alarm": gap > cfg.research.replay_heldout_gap_alarm,
            "seed": seed,
        }
        if candidate is not None:
            with db.transaction():
                db.execute(
                    "UPDATE skill_candidates SET eval_scores = ?, heldout_gap = ?"
                    " WHERE candidate_id = ?",
                    (
                        json.dumps(
                            {"held_in": in_scores, "held_out": out_scores}, sort_keys=True
                        ),
                        gap,
                        candidate["candidate_id"],
                    ),
                )
        verdicts[name] = verdict

    if skill_name is not None:
        return verdicts[skill_name]
    return {"seed": seed, "skills": verdicts}


# -- promotion gate + Gateway write-path (C7) -----------------------------------


def _block_and_stage(db: Database, row: sqlite3.Row, payload: dict) -> GatewayWriteBlocked:
    """§11 blocked-state: the champion stays staged in Spark's registry and
    the registry remains the system of record until a Gateway appears."""
    with db.transaction():
        db.execute(
            "UPDATE skills SET promotion_status = ? WHERE skill_id = ?",
            (PROMOTION_STAGED_IN_SPARK, row["skill_id"]),
        )
        trajectory.record(
            db,
            row["name"],
            "spark",
            "tool_call",
            {"tool": "promote_via_gateway", "blocked": True, "payload": payload},
        )
    return GatewayWriteBlocked(payload)


def promote_via_gateway(
    db: Database,
    cfg: Config,
    candidate: sqlite3.Row | dict,
    *,
    transport: Any | None = None,
) -> dict:
    """The ONLY write-path to a live registry (C7): POST the versioned skill
    manifest to the QuantFlow Gateway skill-write API and require a manifest
    acknowledgement (HTTP 200/201 with body acknowledging the same
    skill_name+version). Any block (no endpoint, transport error, wrong
    contract) raises GatewayWriteBlocked with a structured payload; the
    champion stays staged_in_spark. No filesystem write is ever attempted —
    body_uri stays spark:// (§3.3). `transport` exists so tests can inject an
    httpx.MockTransport."""
    cand = dict(candidate)
    skill_name = cand["skill_name"]
    row = _challenger_for(db, skill_name, cand["parent_version"], candidate=cand)
    base_payload = {
        "skill_name": skill_name,
        "candidate_id": cand["candidate_id"],
    }
    if row is None:
        raise GatewayWriteBlocked({**base_payload, "reason": "challenger_row_missing"})
    version = int(row["version"])
    base_payload["version"] = version
    gw = cfg.peers.gateway
    if not gw.enabled or not gw.url:
        raise _block_and_stage(
            db, row, {**base_payload, "reason": "no_gateway_endpoint_configured"}
        )

    lineage = _lineage(row)
    manifest = {
        "schema_version": GATEWAY_MANIFEST_SCHEMA_VERSION,
        "skill_name": skill_name,
        "version": version,
        "role": row["role"],
        "status": "champion",
        "body_uri": row["body_uri"],
        "procedure_steps": lineage.get("procedure_steps") or [],
        "references": lineage.get("references") or [],
        "lineage": {k: v for k, v in lineage.items() if k not in ("procedure_steps", "references")},
        "credit": json.loads(row["credit"] or "{}"),
        "replay": {
            "eval_scores": json.loads(cand["eval_scores"]) if cand.get("eval_scores") else None,
            "heldout_gap": cand.get("heldout_gap"),
        },
        "staged_at": now_iso(),
    }
    headers: dict[str, str] = {}
    token = os.environ.get(gw.token_env)
    if token:
        headers["Authorization"] = f"Bearer {token}"

    import httpx

    client = httpx.Client(timeout=15.0, transport=transport)
    try:
        resp = client.post(gw.url, json={"manifest": manifest}, headers=headers)
    except httpx.TransportError as exc:
        raise _block_and_stage(
            db, row, {**base_payload, "reason": "transport_error", "detail": str(exc)}
        ) from exc
    finally:
        client.close()
    if resp.status_code not in (200, 201):
        raise _block_and_stage(
            db,
            row,
            {**base_payload, "reason": "unexpected_status", "status_code": resp.status_code},
        )
    try:
        ack = resp.json()
    except json.JSONDecodeError:
        raise _block_and_stage(
            db, row, {**base_payload, "reason": "invalid_response_body"}
        ) from None
    acked = (
        isinstance(ack, dict)
        and bool(ack.get("manifest_ack"))
        and isinstance(ack.get("manifest"), dict)
        and ack["manifest"].get("skill_name") == skill_name
        and int(ack["manifest"].get("version")) == version
    )
    if not acked:
        raise _block_and_stage(
            db, row, {**base_payload, "reason": "manifest_not_acknowledged"}
        )
    with db.transaction():
        db.execute(
            "UPDATE skills SET promotion_status = ? WHERE skill_id = ?",
            (PROMOTION_DEPLOYED, row["skill_id"]),
        )
        trajectory.record(
            db,
            skill_name,
            "spark",
            "tool_call",
            {
                "tool": "promote_via_gateway",
                "blocked": False,
                "version": version,
                "gateway_url": gw.url,
            },
        )
    return ack


def _assign_credit(db: Database, row: sqlite3.Row, source_episode_ids: list[str]) -> dict:
    """v1 minimal credit (§11): uniform weight 1/n over the candidate's source
    episodes; the champion version is appended to each episode's
    skill_versions JSON when missing."""
    n = len(source_episode_ids)
    credit = {ep: (1.0 / n if n else 0.0) for ep in source_episode_ids}
    with db.transaction():
        db.execute(
            "UPDATE skills SET credit = ? WHERE skill_id = ?",
            (json.dumps(credit, sort_keys=True), row["skill_id"]),
        )
        for ep in source_episode_ids:
            ep_row = db.conn.execute(
                "SELECT skill_versions FROM episodes WHERE episode_id = ?", (ep,)
            ).fetchone()
            if ep_row is None:
                continue
            versions = json.loads(ep_row["skill_versions"] or "{}")
            if row["name"] not in versions:
                versions[row["name"]] = str(row["version"])
                db.execute(
                    "UPDATE episodes SET skill_versions = ? WHERE episode_id = ?",
                    (json.dumps(versions, sort_keys=True), ep),
                )
    return credit


def decide_promotion(
    db: Database,
    cfg: Config,
    skill_name: str,
    candidate_id: str,
    decision: str,
    rationale: str,
    decided_by: str,
) -> dict:
    """The promotion gate (plan §11), used by the auto path and by human
    ratification. decision in {'promoted', 'rejected', 'auto'}: 'auto' applies
    the documented gate criterion — promote iff the challenger beats the
    incumbent champion's last held-in mean (first promotion: held-in mean >=
    replay_promotion_floor) AND heldout_gap <= replay_heldout_gap_alarm;
    otherwise reject. Promotion requires FULL-suite replay evidence (review
    major M1): every held-in AND every held-out fixture must be scored on
    the candidate — a cherry-picked subset cannot satisfy the gate.
    decision='promoted': the challenger becomes champion, the prior champion
    becomes the rollback_target, credit is assigned, and — per M11
    auto-with-human-notify — when the decider is the gate an approvals row
    (decision='approved', decided_by='gate') is written as the human-veto
    surface, plus a trajectory notify event. The Gateway write is then
    attempted (C7); a block is reported in the result (gateway.status=
    'blocked') and the champion stays staged_in_spark."""
    if decision not in ("promoted", "rejected", "auto"):
        raise ToolError("decision must be 'promoted', 'rejected', or 'auto'")
    if not rationale or not rationale.strip():
        raise ToolError("rationale is required")
    candidate = db.conn.execute(
        "SELECT * FROM skill_candidates WHERE candidate_id = ? AND skill_name = ?",
        (candidate_id, skill_name),
    ).fetchone()
    if candidate is None:
        raise ToolError(f"unknown candidate {candidate_id!r} for skill {skill_name!r}")
    if candidate["verdict"] != "pending":
        raise ToolError(
            f"candidate {candidate_id} already decided: {candidate['verdict']}"
        )
    if candidate["eval_scores"] is None:
        raise ToolError(
            "promotion requires replay evidence: run run_replay for this skill first"
        )
    challenger = _challenger_for(db, skill_name, candidate["parent_version"], candidate=candidate)
    if challenger is None:  # pragma: no cover - candidate without its challenger row is a bug
        raise ToolError(f"internal: no challenger row for candidate {candidate_id}")
    version = int(challenger["version"])

    # M1: full-suite coverage — every fixture (held-in and held-out) must be
    # scored; filtered run_replay(task_ids=...) subsets cannot satisfy the
    # gate. Enforced only when the decision is 'promoted' (re-review nit:
    # rejecting a candidate does not need full evidence) and resolved AFTER
    # the 'auto' criterion so auto-rejections skip the check.
    eval_scores = json.loads(candidate["eval_scores"])

    source_episode_ids = json.loads(candidate["source_episode_ids"] or "[]")
    old_champion = db.conn.execute(
        "SELECT * FROM skills WHERE name = ? AND status = 'champion' AND version < ?"
        " ORDER BY version DESC LIMIT 1",
        (skill_name, version),
    ).fetchone()

    if decision == "auto":
        challenger_scores = eval_scores.get("held_in") or {}
        ch_mean = (
            sum(challenger_scores.values()) / len(challenger_scores)
            if challenger_scores else 0.0
        )
        gap = candidate["heldout_gap"]
        alarm = float(cfg.research.replay_heldout_gap_alarm)
        if gap is not None and gap > alarm:
            decision = "rejected"
        elif old_champion is not None:
            old_hist = json.loads(old_champion["benchmark_history"] or "[]")
            old_scores = None
            for entry in reversed(old_hist):
                if entry.get("eval_scores", {}).get("held_in"):
                    old_scores = entry["eval_scores"]["held_in"]
                    break
            old_mean = (
                sum(old_scores.values()) / len(old_scores) if old_scores else 0.0
            )
            decision = "promoted" if ch_mean >= old_mean else "rejected"
        else:
            floor = float(getattr(cfg.research, "replay_promotion_floor", 0.5))
            decision = "promoted" if ch_mean >= floor else "rejected"

    if decision == "promoted":
        fixtures = load_fixtures()
        in_ids = {f["task_id"] for f in fixtures if f.get("held_in", True)}
        out_ids = {f["task_id"] for f in fixtures if not f.get("held_in", True)}
        have_in = set((eval_scores.get("held_in") or {}).keys())
        have_out = set((eval_scores.get("held_out") or {}).keys())
        missing = (in_ids - have_in) | (out_ids - have_out)
        if missing:
            raise ToolError(
                "promotion requires FULL replay-suite evidence; missing fixtures: "
                + ", ".join(sorted(missing))
            )

    with db.transaction():
        db.execute(
            "UPDATE skill_candidates SET verdict = ?, decided_at = ?, decided_by = ?"
            " WHERE candidate_id = ?",
            (decision, now_iso(), decided_by, candidate_id),
        )
        if decision == "promoted":
            if old_champion is not None:
                db.execute(
                    "UPDATE skills SET status = 'rollback_target' WHERE skill_id = ?",
                    (old_champion["skill_id"],),
                )
            db.execute(
                "UPDATE skills SET status = 'champion' WHERE skill_id = ?",
                (challenger["skill_id"],),
            )
            history = json.loads(challenger["benchmark_history"] or "[]")
            history.append(
                {
                    "at": now_iso(),
                    "candidate_id": candidate_id,
                    "decision": decision,
                    "heldout_gap": candidate["heldout_gap"],
                    "decided_by": decided_by,
                    # full evidence snapshot so the NEXT auto gate compares
                    # against this champion's actual held-in scores
                    "eval_scores": eval_scores,
                }
            )
            db.execute(
                "UPDATE skills SET benchmark_history = ? WHERE skill_id = ?",
                (json.dumps(history, sort_keys=True), challenger["skill_id"]),
            )
            if decided_by == "gate":
                # M11 auto-with-human-notify: the approvals row is the
                # human-veto surface (see approve()); the trajectory event is
                # the notify.
                db.execute(
                    "INSERT INTO approvals ("
                    " approval_id, target_type, target_id, decision, rationale,"
                    " decided_by, created_at"
                    ") VALUES (?, 'skill', ?, 'approved', ?, 'gate', ?)",
                    (new_approval_id(), candidate_id, rationale, now_iso()),
                )
            trajectory.record(
                db,
                skill_name,
                "spark",
                "tool_call",
                {
                    "tool": "decide_promotion",
                    "candidate_id": candidate_id,
                    "decision": decision,
                    "version": version,
                    "decided_by": decided_by,
                    "notify": decided_by == "gate",
                    "rationale": rationale,
                },
            )
        else:
            db.execute(
                "UPDATE skills SET status = 'rejected' WHERE skill_id = ?",
                (challenger["skill_id"],),
            )
            trajectory.record(
                db,
                skill_name,
                "spark",
                "tool_call",
                {
                    "tool": "decide_promotion",
                    "candidate_id": candidate_id,
                    "decision": decision,
                    "version": version,
                    "decided_by": decided_by,
                },
            )

    credit: dict[str, float] = {}
    if decision == "promoted":
        credit = _assign_credit(db, challenger, source_episode_ids)
        try:
            ack = promote_via_gateway(db, cfg, candidate)
            gateway = {"status": "acknowledged", "ack": ack}
        except GatewayWriteBlocked as blocked:
            gateway = {"status": "blocked", "payload": blocked.payload}
    else:
        gateway = {"status": "not_attempted"}

    return {
        "candidate_id": candidate_id,
        "skill_name": skill_name,
        "version": version,
        "decision": decision,
        "decided_by": decided_by,
        "champion_version": version if decision == "promoted" else None,
        "rollback_target_version": (
            int(old_champion["version"]) if decision == "promoted" and old_champion else None
        ),
        "credit": credit,
        "gateway": gateway,
    }


# -- tool 22: approve (M11 human gate) -------------------------------------------


def approve(
    db: Database,
    cfg: Config,
    target_type: str,
    target_id: str,
    decision: str,
    rationale: str,
    decided_by: str = "human:operator",
) -> dict:
    """Human gate (plan §6.3 tool 22, M11). target_type in {belief, skill,
    adoption}; decision in {approved, rejected}. Idempotent per actor class
    on (target_type, target_id, decision): a duplicate from the same class
    (gate vs human) returns the existing approval row with no side effects,
    while a human ratification is recorded alongside the gate's auto-notify
    row rather than colliding with it.

    - belief: target_id = belief_id; the belief must be promotion_status=
      'pending_human' (§12.3); approval resolves it.
    - skill: target_id = candidate_id of a PROMOTED candidate; decision=
      'rejected' vetoes the champion — it goes to 'rejected' and the prior
      rollback_target champion is restored (§11 rollback). decision=
      'approved' ratifies the staged champion.
    - adoption: target_id = skill_id of a champion; records the human
      authorization for adoption into a live book. The book edit itself is
      out of band in QuantFlow — Spark never touches live books (§11).
    """
    if target_type not in ("belief", "skill", "adoption"):
        raise ToolError("target_type must be one of belief|skill|adoption")
    if decision not in ("approved", "rejected"):
        raise ToolError("decision must be 'approved' or 'rejected'")
    if not rationale or not rationale.strip():
        raise ToolError("rationale is required")

    # Idempotency is matched per actor class ('gate' vs human): a human
    # ratification is recorded alongside the gate's auto-notify row rather
    # than colliding with it, while a repeated decision from the same class
    # returns the existing approval untouched.
    if decided_by == "gate":
        existing_sql = (
            "SELECT * FROM approvals WHERE target_type = ? AND target_id = ?"
            " AND decision = ? AND decided_by = 'gate'"
        )
    else:
        existing_sql = (
            "SELECT * FROM approvals WHERE target_type = ? AND target_id = ?"
            " AND decision = ? AND decided_by != 'gate'"
        )
    existing = db.conn.execute(
        existing_sql, (target_type, target_id, decision)
    ).fetchone()
    if existing is not None:
        return {
            "approval_id": existing["approval_id"],
            "target_type": target_type,
            "target_id": target_id,
            "decision": decision,
            "rationale": existing["rationale"],
            "decided_by": existing["decided_by"],
            "duplicated": True,
        }

    applied: dict[str, Any] = {}
    with db.transaction():
        if target_type == "belief":
            row = db.conn.execute(
                "SELECT * FROM beliefs WHERE belief_id = ?", (target_id,)
            ).fetchone()
            if row is None:
                raise ToolError(f"unknown belief {target_id!r}")
            if row["promotion_status"] != "pending_human":
                raise ToolError(
                    f"belief {target_id} is promotion_status={row['promotion_status']!r},"
                    " not 'pending_human'; nothing for a human to resolve"
                )
            db.execute(
                "UPDATE beliefs SET promotion_status = ? WHERE belief_id = ?",
                (decision, target_id),
            )
            applied = {"belief_id": target_id, "promotion_status": decision}
        elif target_type == "skill":
            candidate = db.conn.execute(
                "SELECT * FROM skill_candidates WHERE candidate_id = ?", (target_id,)
            ).fetchone()
            if candidate is None:
                raise ToolError(f"unknown skill candidate {target_id!r}")
            if candidate["verdict"] != "promoted":
                raise ToolError(
                    f"candidate {target_id} is verdict={candidate['verdict']!r}; only a"
                    " promoted (staged) champion can be ratified or vetoed"
                )
            version_row = _challenger_for(
                db, candidate["skill_name"], candidate["parent_version"], candidate=candidate
            )
            if version_row is None or version_row["status"] != "champion":
                raise ToolError(
                    f"skill {candidate['skill_name']} (candidate {target_id}) is not a"
                    " live champion (already vetoed or retired?)"
                )
            champion = version_row
            version = int(champion["version"])
            if decision == "rejected":
                db.execute(
                    "UPDATE skills SET status = 'rejected' WHERE skill_id = ?",
                    (champion["skill_id"],),
                )
                rollback = db.conn.execute(
                    "SELECT * FROM skills WHERE name = ? AND status = 'rollback_target'"
                    " ORDER BY version DESC LIMIT 1",
                    (candidate["skill_name"],),
                ).fetchone()
                restored = None
                if rollback is not None:
                    db.execute(
                        "UPDATE skills SET status = 'champion' WHERE skill_id = ?",
                        (rollback["skill_id"],),
                    )
                    restored = int(rollback["version"])
                applied = {
                    "vetoed_version": version,
                    "restored_version": restored,
                }
            else:
                applied = {"ratified_version": version}
        else:  # adoption
            row = db.conn.execute(
                "SELECT * FROM skills WHERE skill_id = ?", (target_id,)
            ).fetchone()
            if row is None:
                raise ToolError(f"unknown skill {target_id!r} (adoption target_id = skill_id)")
            if row["status"] != "champion":
                raise ToolError(
                    f"skill {row['name']}:{row['version']} is status={row['status']!r};"
                    " adoption requires a live champion"
                )
            applied = {
                "adoption": "recorded_out_of_band",
                "skill": f"{row['name']}:{row['version']}",
            }
        approval_id = new_approval_id()
        db.execute(
            "INSERT INTO approvals ("
            " approval_id, target_type, target_id, decision, rationale, decided_by, created_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)",
            (approval_id, target_type, target_id, decision, rationale, decided_by, now_iso()),
        )
        trajectory.record(
            db,
            target_id,
            "spark",
            "tool_call",
            {
                "tool": "approve",
                "target_type": target_type,
                "target_id": target_id,
                "decision": decision,
                "decided_by": decided_by,
            },
        )
    return {
        "approval_id": approval_id,
        "target_type": target_type,
        "target_id": target_id,
        "decision": decision,
        "rationale": rationale,
        "decided_by": decided_by,
        "duplicated": False,
        "applied": applied,
    }


# -- retirement + rollback -------------------------------------------------------


def retire_skill(db: Database, name: str, version: int, rationale: str) -> dict:
    """Retire a skill version (§11 lifecycle). When the retired row was the
    champion, the newest lower version that could still serve as a fallback
    (status champion/rejected/rollback_target) becomes the rollback_target
    so a veto/restore path exists."""
    if not rationale or not rationale.strip():
        raise ToolError("rationale is required")
    row = _skill_row(db, name, version)
    if row is None:
        raise ToolError(f"unknown skill {name}:{version}")
    if row["status"] == "retired":
        raise ToolError(f"skill {name}:{version} is already retired")
    restored = None
    with db.transaction():
        db.execute(
            "UPDATE skills SET status = 'retired' WHERE skill_id = ?", (row["skill_id"],)
        )
        if row["status"] == "champion":
            prev = db.conn.execute(
                "SELECT * FROM skills WHERE name = ? AND version < ?"
                " AND status IN ('champion', 'rejected', 'rollback_target')"
                " ORDER BY version DESC LIMIT 1",
                (name, version),
            ).fetchone()
            if prev is not None:
                db.execute(
                    "UPDATE skills SET status = 'rollback_target' WHERE skill_id = ?",
                    (prev["skill_id"],),
                )
                restored = int(prev["version"])
        trajectory.record(
            db,
            name,
            "spark",
            "tool_call",
            {
                "tool": "retire_skill",
                "name": name,
                "version": version,
                "was_status": row["status"],
                "rollback_target_version": restored,
                "rationale": rationale,
            },
        )
    return {
        "name": name,
        "version": version,
        "status": "retired",
        "was_champion": row["status"] == "champion",
        "rollback_target_version": restored,
    }
