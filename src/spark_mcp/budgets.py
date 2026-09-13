"""Budget v1 (plan §14 / M15): spend ledger, rate-card estimation, enforcement.

Every spend event appends to `spend_ledger`; `report_write` includes a cost
section built from it here, and the Phase 5 `archive_status` exposes the same
`spend_summary` as its `spend` field (plan §6.3 tool 19). Spend rows carry
`run_id` (v5): the episode that caused the spend, or the consolidation run id
(§13) — the per-episode and per-consolidation caps key on it; NULL means
unattributed (manual/ad-hoc rows). The QD archive's efficiency reward term
consumes the same ledger (archive.score_components).

Enforcement policy (config `budgets.policy`), identical for the daily,
per-episode, and per-consolidation caps:
- `warn` (default): over-limit spend logs a trajectory event and proceeds.
- `fail_closed`: raises ToolError. `backtest_run` records the pending trial +
  spend row first and enforces inside the same transaction, before dispatch:
  a raise rolls the trial and spend rows back together and never dispatches —
  the end state matches a pre-check (no trial, no dispatch, no spend).
  `log_outcome` enforces the same way: under fail_closed an over-cap outcome
  is refused whole (episode update + trial accounting + spend roll back), on
  the principle that fail_closed never partially commits an over-budget
  event — an out-of-band run that already happened is re-logged once the cap
  is raised, or under `warn`.
"""

from __future__ import annotations

from typing import Any

from . import trajectory
from .config import Config
from .errors import ToolError
from .storage import Database
from .timeutil import now_iso


def estimate_usd(cfg: Config, *, tokens: int, model: str | None = None) -> float:
    """Rate-card estimate for a spend event (§14): per-1K-token rates looked
    up by model. Unknown model or zero tokens -> 0.0 (caller marks estimated).
    """
    if tokens <= 0:
        return 0.0
    model_key = model or cfg.llm.model
    rates = cfg.budgets.rates.get(model_key)
    if not rates:
        return 0.0
    per_1k = float(rates.get("input_per_1k", 0.0)) + float(rates.get("output_per_1k", 0.0))
    return tokens / 1000.0 * per_1k


def record_spend(
    db: Database,
    cfg: Config,
    *,
    family: str | None,
    source: str,
    usd: float | None = None,
    tokens: int = 0,
    estimated: bool = False,
    run_id: str | None = None,
) -> dict:
    """Append one spend_ledger row. `usd` missing -> rate-card estimate
    (0.0 when the model is unknown), always flagged estimated. `run_id`
    attributes the spend (episode_id / consolidation run id) for the
    per-episode and per-consolidation caps. Returns the recorded row."""
    if usd is None:
        usd = estimate_usd(cfg, tokens=tokens)
        estimated = True
    row = {
        "ts": now_iso(),
        "family": family,
        "source": source,
        "usd": float(usd),
        "tokens": int(tokens),
        "estimated": bool(estimated),
        "run_id": run_id,
    }
    db.execute(
        "INSERT INTO spend_ledger (ts, family, source, usd, tokens, estimated, run_id)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (row["ts"], family, source, row["usd"], row["tokens"], int(row["estimated"]), run_id),
    )
    return row


def spend_summary(db: Database) -> dict[str, float]:
    """{today_usd, total_usd} across the ledger — the §14 spend summary shape
    archive_status will return (and report_write renders as a cost section)."""
    today = now_iso()[:10] + "T00:00:00Z"  # ISO text compares lexicographically
    today_usd = db.conn.execute(
        "SELECT COALESCE(SUM(usd), 0.0) AS s FROM spend_ledger WHERE ts >= ?", (today,)
    ).fetchone()["s"]
    total_usd = db.conn.execute("SELECT COALESCE(SUM(usd), 0.0) AS s FROM spend_ledger").fetchone()[
        "s"
    ]
    return {"today_usd": float(today_usd), "total_usd": float(total_usd)}


def enforce_daily_budget(
    db: Database, cfg: Config, *, run_id: str, source: str, hard: bool = False
) -> list[str]:
    """Check the daily total against cfg.budgets.daily_usd. Returns warning
    strings (policy `warn` logs a trajectory event and proceeds); raises
    ToolError when `hard` is set (backtest_run pre-dispatch under fail_closed:
    the raise rolls the trial row back — no trial, no dispatch) or when the
    configured policy is `fail_closed`."""
    summary = spend_summary(db)
    limit = float(cfg.budgets.daily_usd)
    if summary["today_usd"] <= limit:
        return []
    message = (
        f"daily budget exceeded: ${summary['today_usd']:.4f} spent today "
        f"(limit ${limit:.2f}, source {source})"
    )
    if hard or cfg.budgets.policy == "fail_closed":
        raise ToolError(message)
    trajectory.record(
        db,
        run_id,
        "spark",
        "observation",
        {"kind": "budget_warning", "message": message, "today_usd": summary["today_usd"]},
    )
    return [message]


def run_spend(db: Database, run_id: str) -> float:
    """Total spend attributed to one run_id — one episode (backtest_run /
    log_outcome) or one consolidation run (§13) — across the ledger."""
    row = db.conn.execute(
        "SELECT COALESCE(SUM(usd), 0.0) AS s FROM spend_ledger WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    return float(row["s"])


def enforce_episode_budget(
    db: Database, cfg: Config, *, episode_id: str, source: str, hard: bool = False
) -> list[str]:
    """Check the episode's total (all spend attributed to it) against
    cfg.budgets.per_episode_usd (§14). Same warn/fail_closed policy as
    enforce_daily_budget; callers run it inside the same transaction as the
    spend/trial write, so a fail_closed raise rolls both back together.
    """
    spent = run_spend(db, episode_id)
    limit = float(cfg.budgets.per_episode_usd)
    if spent <= limit:
        return []
    message = (
        f"episode budget exceeded: ${spent:.4f} spent on episode {episode_id} "
        f"(limit ${limit:.2f}, source {source})"
    )
    if hard or cfg.budgets.policy == "fail_closed":
        raise ToolError(message)
    trajectory.record(
        db,
        episode_id,
        "spark",
        "observation",
        {"kind": "budget_warning", "message": message, "episode_usd": spent},
    )
    return [message]


def enforce_consolidation_budget(
    db: Database, cfg: Config, *, run_id: str, source: str, hard: bool = False
) -> list[str]:
    """Per-run cap for consolidate (plan §13): LLM spend attributed to one
    consolidation run vs cfg.budgets.per_consolidation_usd — over the cap the
    run stops mid-pass rather than overrunning. Phase 7's consolidate passes
    its run id; the enforcement lives here (budgets v1) so both policies are
    tested before the tool lands. Same policy/transaction rule as the daily
    and per-episode caps."""
    spent = run_spend(db, run_id)
    limit = float(cfg.budgets.per_consolidation_usd)
    if spent <= limit:
        return []
    message = (
        f"consolidation budget exceeded: ${spent:.4f} spent on run {run_id} "
        f"(limit ${limit:.2f}, source {source})"
    )
    if hard or cfg.budgets.policy == "fail_closed":
        raise ToolError(message)
    trajectory.record(
        db,
        run_id,
        "consolidator",
        "observation",
        {"kind": "budget_warning", "message": message, "run_usd": spent},
    )
    return [message]


def spend_from_costs(cfg: Config, costs: dict[str, Any] | None) -> tuple[float, int, bool]:
    """(usd, tokens, estimated) for a costs dict: reported usd wins; otherwise
    rate-card estimate from the token count."""
    costs = costs or {}
    usd = costs.get("usd")
    tokens = int(costs.get("tokens", 0) or 0)
    if usd is not None:
        return float(usd), tokens, bool(costs.get("estimated", False))
    return estimate_usd(cfg, tokens=tokens), tokens, True
