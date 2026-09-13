"""QuantFlow Tier-2 run-summary hook adapter (plan §15.1 C9/C10).

QuantFlow may emit an optional run-summary payload at task close; this module
is the Spark-side receiver. Emission is best-effort peer behavior — the
supported close-out path remains the manual `log_outcome` tool (§15.1 change
list fallback row 1) — so a missing hook never blocks a desk run, and a bad
payload never corrupts ledger state: `validate_run_summary` runs before any
write.

Two failure classes are distinguished deliberately:
- schema violations raise `HookSchemaError` before any write (no trajectory
  event either — a malformed payload is not evidence);
- a *valid* payload referencing an unknown `episode_id` is evidence and is
  recorded as a Tier-2 trajectory event, but the episode is not enriched:
  the result carries `enriched=False` and a `warning`. This keeps the
  peer-side emission path crash-free (§15.1 failure semantics: neither peer
  is a single point of failure for the other).

Payload contract (schema_version 1, extra keys ignored for forward compat):
    {episode_id?, run_id, task_family?, tool_call_count,
     failure_sketches: [{stage, error_class, resolution}],
     skills_activated: {name: version}, tokens, wall_sec,
     outcome_summary?, fingerprint?, fingerprint_spec_version?,
     schema_version}
"""

from __future__ import annotations

import json
from typing import Any

from . import trajectory
from .config import Config
from .errors import ToolError
from .storage import Database

RUN_SUMMARY_SCHEMA_VERSION = 1
# Tier-2 markers (plan §5.1 trajectory_events DDL comments): the actor/kind
# pair marks hook summaries distinctly from Tier-1 tool_call/observation/gate
# events, and the payload repeats {"tier": 2} so tier reads off the row alone.
RUN_SUMMARY_ACTOR = "quantflow-hook"
RUN_SUMMARY_KIND = "hook_summary"
# §15.1 change-list fallback row 2: fingerprint present without a spec version.
UNVERSIONED = "unversioned"

_REQUIRED_FIELDS = (
    "run_id",
    "tool_call_count",
    "failure_sketches",
    "skills_activated",
    "tokens",
    "wall_sec",
    "schema_version",
)
_OPTIONAL_STR_FIELDS = (
    "episode_id",
    "task_family",
    "outcome_summary",
    "fingerprint",
    "fingerprint_spec_version",
)
_SKETCH_FIELDS = ("stage", "error_class", "resolution")


class HookSchemaError(ToolError):
    """A run-summary payload violated the §15.1 contract. Named error so the
    MCP client and tests share one stable type; always raised before any
    write (failure semantics, §15.1)."""


def _reject(message: str) -> None:
    raise HookSchemaError(f"invalid run summary: {message}")


def _check_count(payload: dict[str, Any], key: str) -> int:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        _reject(f"{key!r} must be an integer, got {type(value).__name__} ({value!r})")
    if value < 0:
        _reject(f"{key!r} must be >= 0, got {value}")
    return value


def validate_run_summary(payload: dict) -> dict:
    """Strict validation of a Tier-2 run summary. Returns the payload when
    valid; raises HookSchemaError with a precise message otherwise. Extra
    keys are ignored (forward compat); schema_version != 1 is rejected with
    the received version named in the message — newer payloads are refused,
    never guessed.
    """
    if not isinstance(payload, dict):
        _reject(f"payload must be a JSON object, got {type(payload).__name__}")
    for key in _REQUIRED_FIELDS:
        if key not in payload:
            _reject(f"missing required field {key!r}")

    version = payload["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        _reject(
            f"'schema_version' must be an integer, got {type(version).__name__} ({version!r})"
        )
    if version != RUN_SUMMARY_SCHEMA_VERSION:
        _reject(
            f"unsupported 'schema_version' {version}; this receiver speaks"
            f" schema_version {RUN_SUMMARY_SCHEMA_VERSION}"
            " (newer payloads are rejected, not guessed)"
        )

    run_id = payload["run_id"]
    if not isinstance(run_id, str) or not run_id.strip():
        _reject(f"'run_id' must be a non-empty string, got {run_id!r}")
    for key in _OPTIONAL_STR_FIELDS:
        value = payload.get(key)
        if value is not None and not isinstance(value, str):
            _reject(f"{key!r} must be a string, got {type(value).__name__} ({value!r})")

    _check_count(payload, "tool_call_count")
    _check_count(payload, "tokens")
    wall_sec = payload["wall_sec"]
    if isinstance(wall_sec, bool) or not isinstance(wall_sec, (int, float)):
        _reject(f"'wall_sec' must be a number, got {type(wall_sec).__name__} ({wall_sec!r})")
    if wall_sec < 0:
        _reject(f"'wall_sec' must be >= 0, got {wall_sec}")

    sketches = payload["failure_sketches"]
    if not isinstance(sketches, list):
        _reject(
            f"'failure_sketches' must be an array, got {type(sketches).__name__}"
        )
    for i, sketch in enumerate(sketches):
        if not isinstance(sketch, dict):
            _reject(f"'failure_sketches[{i}]' must be an object, got {type(sketch).__name__}")
        for field in _SKETCH_FIELDS:
            value = sketch.get(field)
            if not isinstance(value, str):
                _reject(
                    f"'failure_sketches[{i}].{field}' must be a string,"
                    f" got {type(value).__name__} ({value!r})"
                )

    skills = payload["skills_activated"]
    if not isinstance(skills, dict):
        _reject(f"'skills_activated' must be an object, got {type(skills).__name__}")
    for name, skill_version in skills.items():
        if not isinstance(name, str) or not isinstance(skill_version, str):
            _reject(
                "'skills_activated' must map skill names to version strings;"
                f" got {name!r}: {skill_version!r}"
            )
    return payload


def ingest_run_summary(db: Database, cfg: Config, payload: dict) -> dict:
    """Validate, then record + enrich. Never raises on extra keys or on an
    unknown episode_id (see module docstring); raises HookSchemaError on
    schema violations before any write.

    Enrichment (episode_id present and resolving): agent_assembly_fingerprint
    + fingerprint_spec_version (default 'unversioned' when a fingerprint
    arrives without one, §15.1 fallback row 2) and skill_versions (default {},
    row 3). An absent fingerprint never wipes a previously recorded one
    (NULL = no new information). Retrospective difficulty per v1 §6:
    tool_call_count > cfg.research.hook_tool_calls_threshold OR
    failures_overcome > 0 OR tokens > cfg.research.hook_tokens_threshold,
    where failures_overcome counts sketches with a non-empty resolution.
    """
    summary = validate_run_summary(payload)

    run_id = summary["run_id"]
    # Empty strings are treated as absent throughout enrichment.
    episode_id = summary.get("episode_id") or None
    fingerprint = summary.get("fingerprint") or None
    spec_version = summary.get("fingerprint_spec_version") or None
    if fingerprint and not spec_version:
        spec_version = UNVERSIONED
    skills = dict(summary["skills_activated"])
    failures_overcome = sum(1 for s in summary["failure_sketches"] if s["resolution"])

    research = cfg.research
    retrospective_triggered = (
        summary["tool_call_count"] > research.hook_tool_calls_threshold
        or failures_overcome > 0
        or summary["tokens"] > research.hook_tokens_threshold
    )
    difficulty = {
        "tool_call_count": summary["tool_call_count"],
        "failures_overcome": failures_overcome,
        "tokens": summary["tokens"],
        "retrospective_triggered": retrospective_triggered,
    }
    event = {
        "tier": 2,
        "run_id": run_id,
        "episode_id": episode_id,
        "task_family": summary.get("task_family"),
        "tool_call_count": summary["tool_call_count"],
        "failure_sketches": summary["failure_sketches"],
        "skills_activated": skills,
        "tokens": summary["tokens"],
        "wall_sec": summary["wall_sec"],
        "outcome_summary": summary.get("outcome_summary"),
        "fingerprint": fingerprint,
        "fingerprint_spec_version": spec_version,
        "difficulty": difficulty,
    }

    warning = None
    enriched = False
    with db.transaction():
        # Tier-1 events key by episode/hypothesis id; a Tier-2 summary joins
        # that trajectory when its episode is known, else keys by the
        # QuantFlow run_id (which is preserved inside the payload either way).
        trajectory.record(db, episode_id or run_id, RUN_SUMMARY_ACTOR, RUN_SUMMARY_KIND, event)
        if episode_id:
            row = db.conn.execute(
                "SELECT 1 FROM episodes WHERE episode_id = ?", (episode_id,)
            ).fetchone()
            if row is None:
                warning = (
                    f"unknown episode_id {episode_id!r}: Tier-2 event recorded,"
                    " episode not enriched"
                )
            else:
                db.execute(
                    "UPDATE episodes SET"
                    " agent_assembly_fingerprint = COALESCE(?, agent_assembly_fingerprint),"
                    " fingerprint_spec_version = COALESCE(?, fingerprint_spec_version),"
                    " skill_versions = ?"
                    " WHERE episode_id = ?",
                    (
                        fingerprint,
                        spec_version,
                        json.dumps(skills, sort_keys=True),
                        episode_id,
                    ),
                )
                enriched = True

    result = {
        "run_id": run_id,
        "episode_id": episode_id,
        "enriched": enriched,
        "fingerprint": fingerprint,
        "fingerprint_spec_version": spec_version,
        "skill_versions": skills,
        "difficulty": difficulty,
    }
    if warning:
        result["warning"] = warning
    return result


def telemetry_push(db: Database, cfg: Config, payload: dict) -> dict:
    """Extension point for the §15.2 optional alpha_engine telemetry row.

    Deferred explicitly: the alpha_engine peer on this machine does not speak
    the §15.2 telemetry contract yet (recorded in STATUS.md), so Phase 4 ships
    the hook adapter only. Implement the push here once the peer exposes the
    contract; both the CLI and MCP paths must call this one function (N13).
    """
    raise NotImplementedError(
        "telemetry_push is deferred until alpha_engine speaks the §15.2 contract"
    )
