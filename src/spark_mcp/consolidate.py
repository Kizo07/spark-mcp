"""Consolidation runtime (plan §13, §12.3, §14): one service function shared
by the `consolidate` MCP tool and `spark consolidate --since …` (N13 — CLI and
MCP never fork logic; intended for cron/systemd timers).

One pass, six gap sources, every item emitted as
    {gap_id, kind, family?, severity, evidence, suggested_action}:

  (a) belief decay application (§12.3): effective_conf = confidence *
      exp(−(as_of − last_validated)/halflife_days) * regime_factor. v1 pins
      regime_factor = 1.0 (no regime model yet — recorded here so the factor
      has one home). The computed factor is WRITTEN to beliefs.decay; the raw
      confidence column is untouched; retrieval always multiplies (memory.py).
      Beliefs with last_validated NULL are skipped (nothing to measure
      staleness against) and counted.
  (b) validation SLO (§12.3): beliefs whose last_validated predates
      as_of − belief_validation_slo_days are listed as gaps.
  (c) archive gap analysis: families with zero occupied QD cells, killed
      families (exploration_state.killed), and predictor_error_map outliers
      (family mean |predicted − realized| above the config floor, plan §21).
  (d) family posterior review: stored Beta posteriors with enough trials
      whose mean sits below 0.5 (using informed_prior_after_episodes as the
      "enough trials" line).
  (e) skill-pipeline review: pending skill_candidates whose challenger row is
      older than skill_candidate_stale_days (stale evolution pipeline).
  (f) fixture-suite review (evolution upgrade plan §2.F4): the replay suite
      still counts below replay_min_fixtures (auto-promotion stays weak until
      fixtures grow from real failures). Alarm only, never blocks promotion.

Budget governance (§13/§14): the per-run cap cfg.budgets.per_consolidation_usd
is enforced DURING the pass. LLM usage is the gap-annotation step (one
`complete` call per gap, estimated at budgets.llm_call_estimate_usd each —
providers report no usage on this seam). The annotation pass stops mid-pass
the moment the next call would push the run over the cap: `warn` (default)
returns a PARTIAL report with budget_exhausted=True (ung annotated gaps keep
their template suggested_action); `fail_closed` mirrors the daily/episode
pattern — the whole plan is checked before any write and the run refuses
whole (ToolError, no decay writes, no spend rows, no consolidation_runs row).
Spend lands in spend_ledger with source='consolidate', run_id=report_id.

`since` (the CLI's --since) makes a run INCREMENTAL: rows whose activity
marker (beliefs.last_validated, skills.created_at, exploration_state /
qd_cells.updated_at) predates `since` are out of scope — cron runs
`spark consolidate --since <last run>` to re-check only what moved. A full
run (since=None) sees everything. Tradeoff recorded: a belief that goes stale
WITHOUT new activity is caught by full runs, not by incremental ones.

Persistence: a consolidation_runs row (run_id=report_id, scope, wall time,
llm_spend_usd, report_uri=spark://consolidation/{report_id}) plus the gap
report as an immutable JSON artifact at
cfg.artifacts_dir()/consolidation/{report_id}.json. Reports are write-once —
suggest_next consumes them by id (plan §13, G5), so provenance is trivially
PIT-correct: a report can never change under a past as_of. Every run records
a trajectory event with actor='consolidator'.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from . import archive, budgets, trajectory
from .config import Config
from .errors import ToolError
from .ids import new_consolidation_id, new_gap_id
from .storage import Database
from .timeutil import ISO_FORMAT, now_iso, validate_iso

REGIME_FACTOR_V1 = 1.0  # §12.3 regime_factor: no regime model in v1 (recorded)

SEVERITY = {"high": 3, "medium": 2, "info": 1}

_GAP_KINDS = (
    "belief_stale",
    "empty_family",
    "killed_family",
    "predictor_error_outlier",
    "family_posterior_weak",
    "skill_candidate_stale",
    "fixture_suite_small",
)


def _parse_iso(ts: str) -> datetime:
    return datetime.strptime(ts, ISO_FORMAT)


def report_path(cfg: Config, report_id: str) -> Path:
    """The immutable gap-report artifact location (§13)."""
    return cfg.artifacts_dir() / "consolidation" / f"{report_id}.json"


def load_report(cfg: Config, report_id: str) -> dict:
    """Load a consolidation gap report artifact by id (suggest_next's
    gap_report_id consumption path, plan §13). Reports are immutable, so the
    loaded copy is exactly what the run produced. Unknown ids raise the
    named ToolError."""
    path = report_path(cfg, report_id)
    if not path.exists():
        raise ToolError(
            f"unknown gap_report_id {report_id!r}: no consolidation report"
            f" artifact at {path} (run consolidate first; reports are"
            " immutable artifacts consumed by id, plan §13)"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _gap(kind: str, family: str | None, severity: str, evidence: dict, action: str) -> dict:
    return {
        "gap_id": new_gap_id(),
        "kind": kind,
        "family": family,
        "severity": severity,
        "evidence": evidence,
        "suggested_action": action,
    }


def _days_between(later: str, earlier: str) -> float:
    """Whole-second-exact day count between two ISO timestamps (can be
    negative — callers clamp where the direction matters)."""
    return (_parse_iso(later) - _parse_iso(earlier)).total_seconds() / 86400.0


# ---------------------------------------------------------------------------
# pass phases: (a) decay + (b) SLO


def _decay_and_slo(
    db: Database, cfg: Config, as_of: str, since: str | None
) -> tuple[list[tuple[str, float]], dict, list[dict], int]:
    """Returns (decay updates [(belief_id, decay)], decay stats, slo gaps,
    skipped-null count). One PIT query; both phases read the same rows."""
    halflife = float(cfg.research.belief_halflife_days)
    slo_days = int(cfg.research.belief_validation_slo_days)
    rows = db.conn.execute(
        "SELECT belief_id, claim, confidence, decay, last_validated, scope"
        " FROM beliefs"
        " WHERE valid_from <= ? AND (valid_to IS NULL OR ? < valid_to) AND as_of <= ?"
        " ORDER BY belief_id",
        (as_of, as_of, as_of),
    ).fetchall()
    updates: list[tuple[str, float]] = []
    gaps: list[dict] = []
    skipped_null = 0
    skipped_since = 0
    for r in rows:
        lv = r["last_validated"]
        if lv is None:
            skipped_null += 1
            continue
        if since is not None and lv < since:
            skipped_since += 1
            continue
        days = max(_days_between(as_of, lv), 0.0)
        decay = min(1.0, math.exp(-days / halflife)) * REGIME_FACTOR_V1
        updates.append((r["belief_id"], decay))
        if days > slo_days:
            gaps.append(
                _gap(
                    "belief_stale",
                    r["scope"],
                    "medium",
                    {
                        "belief_id": r["belief_id"],
                        "claim": r["claim"],
                        "confidence": r["confidence"],
                        "decay": decay,
                        "last_validated": lv,
                        "days_unvalidated": round(days, 2),
                        "slo_days": slo_days,
                    },
                    "re-validate the belief with a fresh episode, or let its"
                    " decay retire it (memory_recall already ranks by"
                    " effective confidence, §12.3)",
                )
            )
    stats = {
        "considered": len(rows),
        "updated": len(updates),
        "skipped_no_last_validated": skipped_null,
        "skipped_since_filter": skipped_since,
        "halflife_days": halflife,
        "regime_factor": REGIME_FACTOR_V1,
    }
    return updates, stats, gaps, skipped_null


# ---------------------------------------------------------------------------
# pass phases: (c) archive + (d) posteriors + (e) skill pipeline


def _archive_gaps(
    db: Database, cfg: Config, as_of: str, since: str | None, scope: str | None
) -> list[dict]:
    gaps: list[dict] = []
    arch = archive.Archive(db, cfg)
    coverage = arch.coverage(as_of)
    families = [f for f in cfg.research.qd_family_bins if scope is None or f == scope]
    for fam in families:
        if coverage["per_family"].get(fam, 0) == 0:
            gaps.append(
                _gap(
                    "empty_family",
                    fam,
                    "high",
                    {"occupied_cells": 0, "as_of": as_of},
                    f"occupy the {fam!r} region: propose and register a hypothesis"
                    " there (suggest_next with this gap report boosts it, §13)",
                )
            )
    state_rows = db.conn.execute(
        "SELECT family, killed, posterior, trials, successes, updated_at"
        " FROM exploration_state WHERE updated_at <= ?"
        + (" AND updated_at >= ?" if since is not None else "")
        + " ORDER BY family",
        (as_of, since) if since is not None else (as_of,),
    ).fetchall()
    for r in state_rows:
        fam = r["family"]
        if fam == "__analogy__" or (scope is not None and fam != scope):
            continue
        if int(r["killed"]):
            gaps.append(
                _gap(
                    "killed_family",
                    fam,
                    "high",
                    {"posterior": json.loads(r["posterior"]), "updated_at": r["updated_at"]},
                    "family auto-tombstoned by the kill rule; reassess its mechanism"
                    " vocabulary or leave it dark (§6 kill rule)",
                )
            )
    for fam, err in archive.predictor_error_map(db, as_of).items():
        if scope is not None and fam != scope:
            continue
        if err > float(cfg.research.predictor_error_outlier):
            gaps.append(
                _gap(
                    "predictor_error_outlier",
                    fam,
                    "medium",
                    {"mean_abs_predicted_minus_realized": err, "as_of": as_of},
                    "family predictions drift from realizations; widen predicted_sigma"
                    " or re-estimate the mapping from idea to IC (plan §21)",
                )
            )
    return gaps


def _posterior_gaps(
    db: Database, cfg: Config, as_of: str, scope: str | None
) -> list[dict]:
    gaps: list[dict] = []
    min_trials = int(cfg.research.informed_prior_after_episodes)
    for fam, post in archive.family_posteriors(db, as_of).items():
        if scope is not None and fam != scope:
            continue
        if post["trials"] >= min_trials and post["mean"] < 0.5:
            gaps.append(
                _gap(
                    "family_posterior_weak",
                    fam,
                    "info",
                    {
                        "alpha": post["alpha"],
                        "beta": post["beta"],
                        "mean": post["mean"],
                        "trials": post["trials"],
                        "threshold": "mean < 0.5 after "
                        f"{min_trials}+ trials",
                    },
                    "deflated hit rate is weak; favor refutation or mechanism"
                    " transfer over further same-family refinement",
                )
            )
    return gaps


def _skill_pipeline_gaps(db: Database, cfg: Config, as_of: str, since: str | None) -> list[dict]:
    """Pending candidates whose challenger skills row is older than
    skill_candidate_stale_days (activity marker: the challenger's
    created_at; candidates table has no timestamp of its own)."""
    stale_days = int(cfg.research.skill_candidate_stale_days)
    rows = db.conn.execute(
        "SELECT c.candidate_id, c.skill_name, c.parent_version, s.created_at, s.version"
        " FROM skill_candidates c"
        " JOIN skills s ON s.name = c.skill_name"
        " AND json_extract(s.lineage, '$.parent_version') = c.parent_version"
        " WHERE c.verdict = 'pending'"
        " AND s.created_at <= ?"
        + (" AND s.created_at >= ?" if since is not None else "")
        + " ORDER BY c.candidate_id",
        (as_of, since) if since is not None else (as_of,),
    ).fetchall()
    gaps = []
    for r in rows:
        age = _days_between(as_of, r["created_at"])
        if age <= stale_days:
            continue
        gaps.append(
            _gap(
                "skill_candidate_stale",
                None,
                "medium",
                {
                    "candidate_id": r["candidate_id"],
                    "skill_name": r["skill_name"],
                    "challenger_version": r["version"],
                    "age_days": round(age, 2),
                    "stale_after_days": stale_days,
                },
                "decide the pending challenger: run run_replay, then"
                " decide_promotion (promote/reject) — a stalled pipeline hides"
                " evolution debt (plan §11)",
            )
        )
    return gaps


def _fixture_growth_gaps(cfg: Config) -> list[dict]:
    """The F4 alarm hook: while the replay suite counts below
    replay_min_fixtures, report fixture growth as an open gap (severity
    medium, alarm only — the gate never reads this). The suite is global, so
    the hook takes no scope/since filter. The skills import is function-local
    to keep this module's import surface narrow."""
    from .skills import load_fixtures

    have = len(load_fixtures())
    floor = int(cfg.research.replay_min_fixtures)
    if have >= floor:
        return []
    return [
        _gap(
            "fixture_suite_small",
            None,
            "medium",
            {"fixture_count": have, "min_fixtures": floor},
            "grow the replay fixture suite from real failures (seed docstring's"
            " own direction): auto-promotion stays weak on a hand-written suite"
            " (evolution upgrade plan §2.F4)",
        )
    ]


# ---------------------------------------------------------------------------
# the pass


def consolidate(
    db: Database,
    cfg: Config,
    scope: str | None = None,
    as_of: str | None = None,
    since: str | None = None,
    llm: Any | None = None,
) -> dict:
    """Run one consolidation pass; returns the gap report (and persists its
    artifact + a consolidation_runs row). See the module docstring for the
    five gap sources, the budget-governance policy, and `since` semantics."""
    as_of = validate_iso(as_of) if as_of else now_iso()
    if since is not None:
        since = validate_iso(since)
    started_at = now_iso()
    report_id = new_consolidation_id()
    cap = float(cfg.budgets.per_consolidation_usd)
    policy = cfg.budgets.policy
    call_cost = float(cfg.budgets.llm_call_estimate_usd)

    # -- gap collection (no writes) -------------------------------------------
    decay_updates, decay_stats, slo_gaps, _ = _decay_and_slo(db, cfg, as_of, since)
    gaps = (
        slo_gaps
        + _archive_gaps(db, cfg, as_of, since, scope)
        + _posterior_gaps(db, cfg, as_of, scope)
        + _skill_pipeline_gaps(db, cfg, as_of, since)
        + _fixture_growth_gaps(cfg)
    )
    gaps.sort(key=lambda g: (-SEVERITY[g["severity"]], g["kind"], g["gap_id"]))

    # -- LLM annotation with the per-run cap enforced DURING the pass ----------
    llm_spend = 0.0
    completed_calls = 0
    budget_exhausted = False
    planned_calls = len(gaps) if llm is not None else 0
    if llm is not None and planned_calls * call_cost > cap:
        if policy == "fail_closed":
            raise ToolError(
                f"consolidation budget exceeded: {planned_calls} gap annotation(s) x"
                f" ${call_cost:.4f} = ${planned_calls * call_cost:.4f} would exceed the"
                f" per-consolidation cap ${cap:.2f} (policy fail_closed) — the run is"
                " refused before any write (plan §13/§14); raise"
                " budgets.per_consolidation_usd or run without an LLM"
            )
    if llm is not None:
        for gap in gaps:
            if llm_spend + call_cost > cap:
                budget_exhausted = True
                break  # stop mid-pass: partial report, remaining gaps keep templates
            prompt = (
                "You are a research-desk consolidator. In one sentence, sharpen the"
                " suggested action for this gap into a concrete next step.\n"
                f"Gap kind: {gap['kind']}\nFamily: {gap.get('family')}\n"
                f"Evidence: {json.dumps(gap['evidence'], sort_keys=True)}\n"
                f"Current action: {gap['suggested_action']}\nReply with one sentence."
            )
            try:
                note = llm.complete(prompt).strip()
            except Exception:
                note = ""
            if note:
                gap["llm_note"] = note
            llm_spend += call_cost
            completed_calls += 1

    finished_at = now_iso()
    report = {
        "report_id": report_id,
        "run_id": report_id,
        "schema": "spark.consolidation_report/v1",
        "started_at": started_at,
        "finished_at": finished_at,
        "scope": scope,
        "as_of": as_of,
        "since": since,
        "gap_count": len(gaps),
        "gaps": gaps,
        "decay": decay_stats,
        "budget": {
            "policy": policy,
            "cap_usd": cap,
            "llm_call_estimate_usd": call_cost,
            "planned_llm_calls": planned_calls,
            "completed_llm_calls": completed_calls,
            "llm_spend_usd": llm_spend,
            "budget_exhausted": budget_exhausted,
        },
    }

    # -- persistence: artifact first (the runs row must always resolve), then
    # the DB transaction (decay writes + spend rows + audit row + trajectory).
    path = report_path(cfg, report_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    with db.transaction():
        for belief_id, decay in decay_updates:
            db.execute(
                "UPDATE beliefs SET decay = ? WHERE belief_id = ?", (decay, belief_id)
            )
        if llm_spend > 0.0:
            budgets.record_spend(
                db,
                cfg,
                family=None,
                source="consolidate",
                usd=llm_spend,
                tokens=0,
                estimated=True,
                run_id=report_id,
            )
        db.execute(
            "INSERT INTO consolidation_runs ("
            " run_id, started_at, finished_at, scope, llm_spend_usd, report_uri, report_id"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                report_id,
                started_at,
                finished_at,
                scope,
                llm_spend,
                f"spark://consolidation/{report_id}",
                report_id,
            ),
        )
        trajectory.record(
            db,
            report_id,
            "consolidator",
            "tool_call",
            {
                "tool": "consolidate",
                "scope": scope,
                "as_of": as_of,
                "since": since,
                "gap_count": len(gaps),
                "kinds": sorted({g["kind"] for g in gaps}),
                "decay_updated": decay_stats["updated"],
                "llm_spend_usd": llm_spend,
                "budget_exhausted": budget_exhausted,
            },
        )
    return {
        **report,
        "report_uri": f"spark://consolidation/{report_id}",
        "artifact_path": str(path),
    }


__all__ = ["REGIME_FACTOR_V1", "consolidate", "load_report", "report_path"]
