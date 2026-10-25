"""Trial governor + gates service (plan §9.1, §9.2, §12.2, §14): backtest_run,
significance_gate, causal_check, risk_decompose, kill rule.

Trial accounting (§9.2, AGENTS.md rule 4): the PENDING trial row lands before
provider dispatch inside one atomic transaction; UNIQUE(episode_id, idem_key)
makes retries free — racing writers resolve to one winner whose committed row
the losers read. Out-of-band outcomes stay on the ledger's reconciled path.

The tombstone pre-check (§12.2) is mandatory before any dispatch: embed the
hypothesis+spec text, cosine vs tombstone embeddings (lexical fallback in
degraded mode), block / review-band adjudicate / miss, and always write the
outcome to episodes.tombstone_check — the logging is what makes
"≥1 tombstone-blocked re-spend" gradeable.

C6 wording is honored verbatim in gate results: no causal promotion without
DAG; no promotion of any kind without falsifier.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any

import numpy as np
from pydantic import ValidationError

from .. import budgets, graph, ledger, memory, trajectory
from ..config import Config
from ..dispatch import DispatchContext, build_pipeline
from ..errors import ToolError
from ..ids import new_trial_id
from ..providers.protocols import BacktestSpec, BookSpec, CausalSpec
from ..storage import Database
from ..timeutil import now_iso
from .dsr import GateRefusal, dsr, min_backtest_length, psr

# ---------------------------------------------------------------------------
# trial accounting


def M_family(db: Database, family: str, as_of: str | None = None) -> int:
    """M_family = COUNT(trials WHERE family AND status IN ('completed','failed'))
    (§9.2). `as_of` filters on created_at for PIT audit views (trials PIT =
    audit horizon, §5.0 matrix)."""
    sql = (
        "SELECT COUNT(*) AS n FROM trials"
        " WHERE family = ? AND status IN ('completed', 'failed')"
    )
    params: list[Any] = [family]
    if as_of is not None:
        sql += " AND created_at <= ?"
        params.append(as_of)
    return int(db.conn.execute(sql, params).fetchone()["n"])


def spec_hash(spec: BacktestSpec) -> str:
    """sha256 over the canonical JSON of the validated spec (§5.0 hashing)."""
    canonical = json.dumps(
        spec.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _idem_key(episode_id: str, hash_of_spec: str, seed: int) -> str:
    raw = f"{episode_id}|{hash_of_spec}|{seed}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()  # noqa: S324 (plan-mandated sha1)


def _trial_dict(row: Any) -> dict:
    return dict(row)


def _replay_trial(db: Database, episode_id: str, winner: Any) -> dict:
    """Idempotent retry: return the existing (winner) trial without dispatch.
    The episode's stored metrics ride along when the trial finished."""
    episode = ledger.get_episode(db, episode_id) or {}
    return {
        "trial_id": winner["trial_id"],
        "episode_id": episode_id,
        "status": winner["status"],
        "metrics": episode.get("metrics"),
        "provider": winner["provider"],
        "reused": True,
        "M_family": M_family(db, winner["family"]),
    }


# ---------------------------------------------------------------------------
# tombstone pre-check (§12.2, mandatory)


def _tokens(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9]+", text.lower())


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def tombstone_candidates(
    db: Database, embedder: Any, text: str
) -> list[dict[str, Any]]:
    """Cosine of `text` vs every PIT-visible tombstone, descending.

    Embedding basis when an embedder + tombstone embeddings exist (§12.2);
    deterministic lexical cosine fallback keeps the pre-check mandatory in
    degraded no-embedder mode (basis is always reported).
    """
    now = now_iso()
    rows = db.conn.execute(
        "SELECT * FROM tombstones WHERE as_of <= ? ORDER BY tombstone_id", (now,)
    ).fetchall()
    if not rows:
        return []
    if embedder is not None:
        query_vec = np.asarray(embedder.embed([text]), dtype=np.float64)[0]
        embedded = dict(graph.fetch_embeddings(db, embedder.model_id(), "tombstone"))
        if embedded:
            out = [
                {
                    "tombstone_id": r["tombstone_id"],
                    "summary": r["summary"],
                    "match_keys": json.loads(r["match_keys"]),
                    "cosine": _cosine(query_vec, embedded[r["tombstone_id"]]),
                    "basis": "embedding",
                }
                for r in rows
                if r["tombstone_id"] in embedded
            ]
            if out:
                out.sort(key=lambda c: c["cosine"], reverse=True)
                return out
    # lexical fallback (degraded mode): bag-of-words cosine vs summary+keywords
    q_tokens = set(_tokens(text))
    out = []
    for r in rows:
        keys = json.loads(r["match_keys"])
        hay = " ".join(
            [r["summary"], " ".join(keys.get("keywords", [])), " ".join(keys.get("mech_tags", []))]
        )
        h_tokens = set(_tokens(hay))
        cos = (
            len(q_tokens & h_tokens) / ((len(q_tokens) * len(h_tokens)) ** 0.5)
            if q_tokens and h_tokens
            else 0.0
        )
        out.append(
            {
                "tombstone_id": r["tombstone_id"],
                "summary": r["summary"],
                "match_keys": keys,
                "cosine": float(cos),
                "basis": "lexical",
            }
        )
    out.sort(key=lambda c: c["cosine"], reverse=True)
    return out


def tombstone_precheck(
    db: Database,
    cfg: Config,
    text: str,
    *,
    embedder: Any = None,
    llm: Any = None,
    acknowledge_tombstone: dict | None = None,
) -> tuple[dict, ToolError | None]:
    """§12.2 pre-check. Returns (check, block_error). The caller writes the
    check dict to episodes.tombstone_check in all cases (hit, miss, ack)."""
    check: dict[str, Any] = {
        "status": "miss",
        "matched_id": None,
        "cosine": None,
        "basis": None,
        "adjudicated": False,
        "thresholds": {
            "block": cfg.research.tombstone_block,
            "review_low": cfg.research.tombstone_review_low,
        },
        "checked_at": now_iso(),
    }
    candidates = tombstone_candidates(db, embedder, text)
    if not candidates:
        return check, None
    best = candidates[0]
    check.update(matched_id=best["tombstone_id"], cosine=best["cosine"], basis=best["basis"])
    block_at = cfg.research.tombstone_block
    review_at = cfg.research.tombstone_review_low

    if best["cosine"] >= block_at:
        ack_id = (acknowledge_tombstone or {}).get("tombstone_id")
        if ack_id == best["tombstone_id"]:
            check["status"] = "ack"
            check["rationale"] = (acknowledge_tombstone or {}).get("rationale")
            return check, None
        check["status"] = "hit"
        return check, _tombstone_block_error(best, "cosine >= tombstone_block threshold")
    if best["cosine"] >= review_at:
        if llm is not None:
            verdict = llm.complete(
                "Tombstone pre-check adjudication. A proposed backtest resembles a "
                f"tombstoned (failed) idea.\nTombstone: {best['summary']}\nCosine: "
                f"{best['cosine']:.3f}\nReply with exactly ALLOW or BLOCK."
            )
            check["adjudicated"] = True
            # N10: exact match only — a substring test ("ALLOW" in ...) would
            # let a "DISALLOW" reply through.
            if verdict.strip().upper() == "ALLOW":
                check["status"] = "miss"
                return check, None
            check["status"] = "hit"
            return check, _tombstone_block_error(
                best, f"LLM adjudication blocked in review band (cosine {best['cosine']:.3f})"
            )
        # no key configured: conservative block (§12.2 step 3)
        check["status"] = "hit"
        return check, _tombstone_block_error(
            best,
            f"conservative block in review band (cosine {best['cosine']:.3f}, no LLM key)",
        )
    return check, None


def _tombstone_block_error(candidate: dict, why: str) -> ToolError:
    evidence = {
        "tombstone_id": candidate["tombstone_id"],
        "summary": candidate["summary"],
        "match_keys": candidate["match_keys"],
        "cosine": candidate["cosine"],
        "basis": candidate["basis"],
    }
    return ToolError(
        "tombstone pre-check blocked dispatch (§12.2): "
        f"{why}. matched tombstone: {json.dumps(evidence, sort_keys=True)}. "
        "Resolution: revise the hypothesis, or re-attempt with "
        "acknowledge_tombstone={tombstone_id, rationale} (logged on the episode)."
    )


def _write_tombstone_check(db: Database, episode_id: str, check: dict) -> None:
    db.execute(
        "UPDATE episodes SET tombstone_check = ? WHERE episode_id = ?",
        (json.dumps(check, sort_keys=True), episode_id),
    )


# ---------------------------------------------------------------------------
# kill rule (§8.3 / exploration §6) + family counters


def _record_family_outcome_impl(
    db: Database,
    cfg: Config,
    family: str,
    *,
    success: bool,
    episode_id: str,
    failure_reason: str | None = None,
) -> dict:
    """Maintain per-family exploration counters; the kill rule fires when
    consecutive deflated failures reach cfg.research.kill_rule_failures:
    one auto-tombstone (dedupe-keyed, kind='failed_hypothesis', evidence =
    triggering episode) and the family's `killed` flag.

    Callers must be genuine evaluation/gate outcomes only: a backtest that
    completed, a significance_gate verdict, or a log_outcome (§8.3). Provider
    exceptions/transport errors are infra, not hypothesis evidence, and never
    reach this function (N6) — they persist a failed trial (§9.2) without
    ticking the kill rule. The §8.3 family posterior penalty still rides the
    kill-rule tombstone; the Beta posterior itself updates here (Phase 5):
    success → alpha+1, failure → beta+1, initialized from
    cfg.research.bandit_prior (§5.1 exploration_state.posterior).

    `trials`/`successes` are bandit observation-event counters (§8.3
    visit/reward aggregates), one increment per genuine family outcome — they
    are NOT the §9.2 trial count, which lives in the trials table
    (M_family). Success resets consecutive_failures (§8.3).

    Safe to call inside an outer transaction (savepoint-nested) and
    concurrently (row UPDATE is atomic; tombstone insert is INSERT OR IGNORE
    on the dedupe key)."""
    now = now_iso()
    prior_a, prior_b = (list(cfg.research.bandit_prior) + [1.0, 1.0])[:2]
    with db.transaction():
        db.execute(
            "INSERT INTO exploration_state (family, posterior, updated_at)"
            " VALUES (?, ?, ?)"
            " ON CONFLICT(family) DO NOTHING",
            (family, json.dumps({"alpha": prior_a, "beta": prior_b}), now),
        )
        db.execute(
            "UPDATE exploration_state SET"
            " trials = trials + 1,"
            " successes = successes + ?,"
            " consecutive_failures = CASE WHEN ? THEN 0 ELSE consecutive_failures + 1 END,"
            " updated_at = ?"
            " WHERE family = ?",
            (1 if success else 0, 1 if success else 0, now, family),
        )
        state = db.conn.execute(
            "SELECT consecutive_failures, killed, posterior FROM exploration_state"
            " WHERE family = ?",
            (family,),
        ).fetchone()
        failures = int(state["consecutive_failures"])
        killed = bool(state["killed"])
        try:
            post = json.loads(state["posterior"])
            a, b = float(post["alpha"]), float(post["beta"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            a, b = prior_a, prior_b
        a, b = (a + 1.0, b) if success else (a, b + 1.0)
        db.execute(
            "UPDATE exploration_state SET posterior = ? WHERE family = ?",
            (json.dumps({"alpha": a, "beta": b}), family),
        )
        tombstone_id = None
        if failures >= cfg.research.kill_rule_failures and not killed:
            tombstone = memory.add_tombstone(
                db,
                kind="failed_hypothesis",
                summary=(
                    f"[kill-rule] family {family!r} reached {failures} consecutive"
                    f" deflated failures"
                    + (f": {failure_reason}" if failure_reason else "")
                ),
                match_keys={"family": family, "mech_tags": [family], "keywords": _tokens(family)},
                evidence_episode_ids=[episode_id],
                as_of=now,
                dedupe_key=hashlib.sha1(f"kill-rule|{family}".encode()).hexdigest(),
            )
            tombstone_id = tombstone["tombstone_id"]
            killed = True
            db.execute(
                "UPDATE exploration_state SET killed = 1 WHERE family = ?", (family,)
            )
        trajectory.record(
            db,
            episode_id,
            "spark",
            "gate",
            {
                "kind": "family_outcome",
                "family": family,
                "success": success,
                "consecutive_failures": failures,
                "killed": killed,
                "tombstone_id": tombstone_id,
                "posterior": {"alpha": a, "beta": b},
            },
        )
    return {
        "family": family,
        "consecutive_failures": failures,
        "killed": killed,
        "tombstone_id": tombstone_id,
    }


# ---------------------------------------------------------------------------
# dispatch pipeline steps (Fix D): the _backtest_run_impl sequence factored
# into the canonical dispatch chain — TombstonePreCheck -> BudgetCheck ->
# TrialInsert -> ProviderDispatch -> VerdictRecord — executed via
# dispatch.Pipeline. Each step delegates to the existing helpers above
# (tombstone_precheck, budgets.*, _replay_trial, provider.backtest); no
# ledger or budget logic is duplicated here.
#
# Two deliberate order notes vs the pre-pipeline body:
# - BudgetCheck runs before TrialInsert as a read-only enforce (no spend
#   row yet). backtest_run's own spend is always a $0.0 estimate, so the
#   pre-check is exactly equivalent to the old post-spend enforce, and the
#   fail_closed end state still matches a pre-check (no trial, no spend,
#   no dispatch). The spend row still lands in TrialInsert right after the
#   pending insert, inside the same transaction as dispatch.
# - The tombstone step is non-transactional: its episode write must
#   persist even when it blocks. The other four steps share one
#   transaction (pending-row-before-dispatch exactly-once, §9.2).


TOMBSTONE_STEP = "tombstone_precheck"
BUDGET_STEP = "budget_check"
TRIAL_STEP = "trial_insert"
DISPATCH_STEP = "provider_dispatch"
VERDICT_STEP = "verdict_record"


def _dispatch_text(hypothesis: dict, bt: BacktestSpec) -> str:
    return " ".join(
        str(p)
        for p in (
            hypothesis.get("mechanism"),
            hypothesis.get("scope"),
            bt.signal_ref,
            json.dumps(bt.params, sort_keys=True),
        )
        if p
    )


def _step_tombstone_precheck(ctx: DispatchContext) -> None:
    """§12.2 pre-check (always logged on the episode in enforce mode)."""
    db, cfg = ctx.db, ctx.cfg
    llm = None if ctx.mode != "enforce" else ctx.llm  # assess: no external calls
    check, block_error = tombstone_precheck(
        db,
        cfg,
        _dispatch_text(ctx.hypothesis, ctx.bt),
        embedder=ctx.embedder,
        llm=llm,
        acknowledge_tombstone=ctx.acknowledge_tombstone,
    )
    ctx.check = check
    if ctx.mode != "enforce":
        ctx.would_block[TOMBSTONE_STEP] = block_error is not None
        return
    _write_tombstone_check(db, ctx.episode_id, check)
    if block_error is not None:
        trajectory.record(
            db,
            ctx.episode_id,
            "spark",
            "gate",
            {
                "kind": "tombstone_block",
                "matched_id": check["matched_id"],
                "cosine": check["cosine"],
                "status": check["status"],
            },
        )
        raise block_error


def _step_budget_check(ctx: DispatchContext) -> None:
    """§14 enforcement before any write (fail_closed raises: no trial,
    no spend, no dispatch). Warn policy logs trajectory events and the
    warnings ride on ctx."""
    db, cfg = ctx.db, ctx.cfg
    if ctx.mode != "enforce":
        over_day = budgets.spend_summary(db)["today_usd"] > float(cfg.budgets.daily_usd)
        over_ep = budgets.run_spend(db, ctx.episode_id) > float(cfg.budgets.per_episode_usd)
        ctx.would_block[BUDGET_STEP] = bool(over_day or over_ep)
        return
    hard = cfg.budgets.policy == "fail_closed"
    ctx.warnings.extend(
        budgets.enforce_daily_budget(
            db, cfg, run_id=ctx.episode_id, source="backtest_run", hard=hard
        )
    )
    ctx.warnings.extend(
        budgets.enforce_episode_budget(
            db, cfg, episode_id=ctx.episode_id, source="backtest_run", hard=hard
        )
    )


def _step_trial_insert(ctx: DispatchContext) -> None:
    """§9.2 pending trial row before dispatch; UNIQUE(episode_id, idem_key)
    makes retries replay the winner (ctx.halted short-circuits dispatch).
    The spend row lands right after the insert, same transaction."""
    db, cfg = ctx.db, ctx.cfg
    bt = ctx.bt
    ctx.hash_of_spec = spec_hash(bt)
    ctx.idem_key = _idem_key(ctx.episode_id, ctx.hash_of_spec, bt.seed)
    ctx.trial_id = new_trial_id()
    now = now_iso()
    ctx.family = ctx.episode["research_family"]
    try:
        db.execute(
            "INSERT INTO trials ("
            " trial_id, episode_id, idem_key, family, source, status, provider,"
            " created_at"
            ") VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)",
            (
                ctx.trial_id,
                ctx.episode_id,
                ctx.idem_key,
                ctx.family,
                ctx.source,
                _provider_name(ctx.provider),
                now,
            ),
        )
    except sqlite3.IntegrityError:
        # §9.2: racing writers resolve to one winner; losers read its row.
        winner = db.conn.execute(
            "SELECT * FROM trials WHERE episode_id = ? AND idem_key = ?",
            (ctx.episode_id, ctx.idem_key),
        ).fetchone()
        if winner is None:
            raise
        result = _replay_trial(db, ctx.episode_id, winner)
        if ctx.client_id:
            _record_effect(db, "backtest_run", ctx.client_id, ctx.episode_id, result)
        ctx.result = result
        ctx.halted = True
        return
    budgets.record_spend(
        db,
        cfg,
        family=ctx.family,
        source="backtest_run",
        usd=None,
        tokens=0,
        estimated=True,
        run_id=ctx.episode_id,
    )


def _step_provider_dispatch(ctx: DispatchContext) -> None:
    """Provider backtest inside the trial transaction (the pending row is
    visible to the call). Failures persist the failed trial (§9.2) and are
    captured in ctx.error, raised only after the transaction commits —
    raising here would roll the failure update back. N6: provider errors
    are infra, never kill-rule evidence."""
    db = ctx.db
    try:
        ctx.metrics = ctx.provider.backtest(ctx.bt)
    except NotImplementedError as e:
        # a provider without backtest support is a config error, not a
        # deflated hypothesis: the trial fails but the kill rule does not tick.
        db.execute(
            "UPDATE trials SET status = 'failed', completed_at = ? WHERE trial_id = ?",
            (now_iso(), ctx.trial_id),
        )
        trajectory.record(
            db, ctx.episode_id, "spark", "tool_call",
            {"tool": "backtest_run", "trial_id": ctx.trial_id, "error": "not_implemented"},
        )
        ctx.error = ToolError(
            f"provider {_provider_name(ctx.provider)!r} does not support backtest: {e}"
        )
    except Exception as e:
        db.execute(
            "UPDATE trials SET status = 'failed', completed_at = ? WHERE trial_id = ?",
            (now_iso(), ctx.trial_id),
        )
        trajectory.record(
            db, ctx.episode_id, "spark", "tool_call",
            {"tool": "backtest_run", "trial_id": ctx.trial_id, "error": repr(e)},
        )
        ctx.error = ToolError(f"provider {_provider_name(ctx.provider)!r} backtest failed: {e}")


def _step_verdict_record(ctx: DispatchContext) -> None:
    """On success: close the trial, flip the episode to testing, record
    metrics + family outcome. Skipped when dispatch captured an error."""
    if ctx.error is not None:
        return
    db, cfg = ctx.db, ctx.cfg
    bt = ctx.bt
    metrics = ctx.metrics or {}
    completed = now_iso()
    costs = {
        "bps": bt.costs.bps,
        "slippage": bt.costs.slippage,
        "borrow": bt.costs.borrow,
        "usd": 0.0,
        "tokens": 0,
        "estimated": True,
    }
    db.execute(
        "UPDATE trials SET status = 'completed', completed_at = ? WHERE trial_id = ?",
        (completed, ctx.trial_id),
    )
    db.execute(
        "UPDATE episodes SET"
        " status = 'testing', metrics = ?, corrected_metrics = ?,"
        " costs = ?, spec_hash = ?, seed = ?, params = ?"
        " WHERE episode_id = ?",
        (
            json.dumps(metrics, sort_keys=True),
            json.dumps(
                {"psr": None, "dsr": None, "p_value": None, "status": "pending_gate"},
                sort_keys=True,
            ),
            json.dumps(costs, sort_keys=True),
            ctx.hash_of_spec,
            bt.seed,
            json.dumps(bt.params, sort_keys=True),
            ctx.episode_id,
        ),
    )
    _record_family_outcome_impl(db, cfg, ctx.family, success=True, episode_id=ctx.episode_id)
    trajectory.record(
        db,
        ctx.episode_id,
        "spark",
        "tool_call",
        {
            "tool": "backtest_run",
            "trial_id": ctx.trial_id,
            "provider": _provider_name(ctx.provider),
            "sr": metrics.get("sr"),
            "simulated": metrics.get("simulated", False),
            "client_id": ctx.client_id,
        },
    )


_DISPATCH_STEPS = {
    TOMBSTONE_STEP: _step_tombstone_precheck,
    BUDGET_STEP: _step_budget_check,
    TRIAL_STEP: _step_trial_insert,
    DISPATCH_STEP: _step_provider_dispatch,
    VERDICT_STEP: _step_verdict_record,
}


def resolve_dispatch_step(name: str):
    """Bind a dispatch step name to its service callable (KeyError if unknown)."""
    try:
        return _DISPATCH_STEPS[name]
    except KeyError:
        raise KeyError(
            f"unknown dispatch step {name!r} (have {sorted(_DISPATCH_STEPS)})"
        ) from None


def build_dispatch_pipeline(profile: str = "gated-backtest"):
    """Compose the dispatch Pipeline for `profile` with the real service
    steps (KeyError on unknown profile or step)."""
    return build_pipeline(profile, resolve_dispatch_step)


# ---------------------------------------------------------------------------
# tools


def _backtest_run_impl(
    db: Database,
    cfg: Config,
    provider: Any,
    episode_id: str,
    spec: dict,
    *,
    client_id: str | None = None,
    acknowledge_tombstone: dict | None = None,
    embedder: Any = None,
    llm: Any = None,
) -> dict:
    """Dispatch a backtest under the trial governor (§9.2 + §12.2 + §14).

    Order: validate BacktestSpec -> dispatch pipeline (TombstonePreCheck ->
    BudgetCheck -> TrialInsert -> ProviderDispatch -> VerdictRecord). The
    budget pre-check raises fail_closed before any write; the pending trial
    row + spend row land before dispatch inside one transaction with the
    dispatch itself: exactly one winner per idem key, one provider call,
    losers read the winner's committed row.
    """
    if client_id:
        row = db.conn.execute(
            "SELECT result FROM tool_effects WHERE client_id = ? AND tool = 'backtest_run'",
            (client_id,),
        ).fetchone()
        if row:
            return json.loads(row["result"])

    episode = ledger.get_episode(db, episode_id)
    if episode is None:
        raise ToolError(f"unknown episode_id {episode_id!r}")

    try:
        bt = BacktestSpec.model_validate(spec)
    except ValidationError as e:
        raise ToolError(f"invalid BacktestSpec: {e}") from e
    if bt.hypothesis_id != episode["hypothesis_id"]:
        raise ToolError(
            f"spec.hypothesis_id {bt.hypothesis_id!r} does not match the episode's"
            f" hypothesis {episode['hypothesis_id']!r}"
        )
    hypothesis = ledger.get_hypothesis(db, bt.hypothesis_id)
    if hypothesis is None:
        raise ToolError(f"unknown hypothesis_id {bt.hypothesis_id!r}")

    # -- dispatch pipeline: TombstonePreCheck -> BudgetCheck -> TrialInsert
    # -> ProviderDispatch -> VerdictRecord (steps above; the tombstone step
    # runs outside the transaction so its episode write survives a block,
    # the rest share one transaction for §9.2 exactly-once).
    ctx = DispatchContext(
        db=db,
        cfg=cfg,
        provider=provider,
        episode_id=episode_id,
        bt=bt,
        episode=episode,
        hypothesis=hypothesis,
        client_id=client_id,
        acknowledge_tombstone=acknowledge_tombstone,
        embedder=embedder,
        llm=llm,
    )
    build_dispatch_pipeline("gated-backtest").run(ctx)
    if ctx.halted:
        return ctx.result  # idempotent replay: winner's row, no new dispatch
    if ctx.error is not None:
        raise ctx.error

    result = {
        "trial_id": ctx.trial_id,
        "episode_id": episode_id,
        "status": "testing",
        "provider": _provider_name(provider),
        "metrics": ctx.metrics,
        "spec_hash": ctx.hash_of_spec,
        "idem_key": ctx.idem_key,
        "reused": False,
        "tombstone_check": ctx.check,
        "M_family": M_family(db, ctx.family),
    }
    if client_id:
        _record_effect(db, "backtest_run", client_id, episode_id, result)
    return result


def _significance_gate_impl(db: Database, cfg: Config, episode_id: str) -> dict:
    """Type-A gate (§9.1): DSR on the episode's net-of-costs metrics, MinBTL
    pre-check, status flip to validated/inconclusive. SR is always reported
    paired with DSR (sr_reported_with_dsr); refusing paths surface
    needs-more-data as ToolError. C6 blockers are reported verbatim."""
    episode = ledger.get_episode(db, episode_id)
    if episode is None:
        raise ToolError(f"unknown episode_id {episode_id!r}")
    metrics = episode.get("metrics") or {}
    missing = [k for k in ("sr", "T", "skew", "kurt") if k not in metrics]
    if missing:
        raise ToolError(
            f"episode {episode_id} has no gateable metrics (missing {missing});"
            " run backtest_run or log_outcome with metrics first"
        )
    sr = float(metrics["sr"])
    T = int(metrics["T"])
    skew = float(metrics.get("skew", 0.0))
    kurt = float(metrics.get("kurt", 3.0))
    family = episode["research_family"]
    M = M_family(db, family)
    alpha = float(cfg.research.dsr_alpha)

    min_t_needed = min_backtest_length(sr, M, skew, kurt, alpha=alpha, min_t=cfg.research.min_T)
    try:
        # psr()/dsr() return the Bailey–LdP probability Phi(z) (confidence
        # form); the plan's promotion rule "DSR p < dsr_alpha" is the p-value
        # form, p = 1 - Phi(z). The two are equivalent transforms; reporting
        # both keeps SR paired with DSR without inverting the decision.
        prob_psr, ok_psr = psr(sr, 0.0, T, skew, kurt)
        prob_dsr, ok_dsr = dsr(sr, T, skew, kurt, M, min_t=cfg.research.min_T)
    except GateRefusal as e:
        raise ToolError(str(e)) from e
    p_value = 1.0 - prob_dsr

    heavy_tail = abs(skew) > 3.0
    degenerate = not (ok_psr and ok_dsr)
    # §9.1: SR_hat <= 0 fails regardless of p (belt and braces: p >= 0.5 there
    # already, but the protocol wording is explicit).
    passed = bool(ok_dsr and sr > 0.0 and p_value < alpha)

    hypothesis = ledger.get_hypothesis(db, episode["hypothesis_id"]) or {}
    promotion_blockers = []
    if hypothesis.get("hypothesis_type") == "causal" and not hypothesis.get("dag"):
        promotion_blockers.append("no causal promotion without DAG (C6)")
    if not hypothesis.get("falsifier"):
        promotion_blockers.append("no promotion of any kind without falsifier (C6)")

    failed_episode = episode["outcome"] == "failed" or episode["status"] == "failed"
    new_status = "validated" if (passed and not failed_episode) else "inconclusive"
    corrected = {
        "sr": sr,
        "psr": prob_psr,
        "dsr": prob_dsr,
        "p_value": p_value,
        "M_family": M,
        "min_T": min_t_needed,
        "alpha": alpha,
        "T": T,
        "skew": skew,
        "kurt": kurt,
        "heavy_tail_caveat": heavy_tail,
        "degenerate": degenerate,
        "pass": passed,
    }
    with db.transaction():
        db.execute(
            "UPDATE episodes SET status = ?, corrected_metrics = ? WHERE episode_id = ?",
            (new_status, json.dumps(corrected, sort_keys=True), episode_id),
        )
        _record_family_outcome_impl(db, cfg, family, success=passed, episode_id=episode_id)
        trajectory.record(
            db,
            episode_id,
            "spark",
            "gate",
            {
                "tool": "significance_gate",
                "sr": sr,
                "dsr": prob_dsr,
                "p_value": p_value,
                "M_family": M,
                "pass": passed,
                "status": new_status,
            },
        )
    return {
        "episode_id": episode_id,
        "status": new_status,
        "sr": sr,
        "psr": prob_psr,
        "dsr": prob_dsr,
        "p": p_value,
        "M_family": M,
        "min_T": min_t_needed,
        "alpha": alpha,
        "T": T,
        "pass": passed,
        "placebo": episode["run_kind"] != "primary",
        "sr_reported_with_dsr": True,
        "heavy_tail_caveat": heavy_tail,
        "degenerate": degenerate,
        "promotion_blockers": promotion_blockers,
    }


def _causal_check_impl(
    db: Database,
    cfg: Config,
    provider: Any,
    episode_id: str,
    spec: dict,
    *,
    client_id: str | None = None,
) -> dict:
    """Type-B causal probe (§9.3, C6 scoped): validate CausalSpec, delegate to
    the provider, store under episodes.metrics.causal, return verbatim.
    Providers that do not implement causal raise NotImplementedError -> ToolError."""
    del cfg
    if client_id:
        row = db.conn.execute(
            "SELECT result FROM tool_effects WHERE client_id = ? AND tool = 'causal_check'",
            (client_id,),
        ).fetchone()
        if row:
            return json.loads(row["result"])
    episode = ledger.get_episode(db, episode_id)
    if episode is None:
        raise ToolError(f"unknown episode_id {episode_id!r}")
    try:
        cs = CausalSpec.model_validate(spec)
    except ValidationError as e:
        raise ToolError(f"invalid CausalSpec: {e}") from e
    if cs.episode_id != episode_id:
        raise ToolError(
            f"spec.episode_id {cs.episode_id!r} does not match episode_id {episode_id!r}"
        )
    try:
        result = provider.causal(cs)
    except NotImplementedError as e:
        raise ToolError(
            f"provider {_provider_name(provider)!r} does not support causal analysis: {e}"
        ) from e
    _store_analysis(db, episode_id, episode, "causal", result)
    if client_id:
        _record_effect(db, "causal_check", client_id, episode_id, result)
    return result


def _risk_decompose_impl(
    db: Database,
    cfg: Config,
    provider: Any,
    episode_id: str,
    spec: dict,
    *,
    client_id: str | None = None,
) -> dict:
    """Book risk decomposition (§15.2): validate BookSpec (exactly one source),
    delegate, store on episodes.risk_exposures, return verbatim."""
    del cfg
    if client_id:
        row = db.conn.execute(
            "SELECT result FROM tool_effects WHERE client_id = ? AND tool = 'risk_decompose'",
            (client_id,),
        ).fetchone()
        if row:
            return json.loads(row["result"])
    episode = ledger.get_episode(db, episode_id)
    if episode is None:
        raise ToolError(f"unknown episode_id {episode_id!r}")
    try:
        bs = BookSpec.model_validate(spec)
    except ValidationError as e:
        raise ToolError(f"invalid BookSpec: {e}") from e
    try:
        result = provider.risk(bs)
    except NotImplementedError as e:
        raise ToolError(
            f"provider {_provider_name(provider)!r} does not support risk decomposition: {e}"
        ) from e
    _store_analysis(db, episode_id, episode, "risk", result, column="risk_exposures")
    if client_id:
        _record_effect(db, "risk_decompose", client_id, episode_id, result)
    return result


def _store_analysis(
    db: Database,
    episode_id: str,
    episode: dict,
    key: str,
    result: dict,
    *,
    column: str = "metrics",
) -> None:
    existing = episode.get(column) if column == "metrics" else None
    if column == "metrics":
        merged = {**(existing or {}), key: result}
        db.execute(
            "UPDATE episodes SET metrics = ? WHERE episode_id = ?",
            (json.dumps(merged, sort_keys=True), episode_id),
        )
    else:
        db.execute(
            "UPDATE episodes SET risk_exposures = ? WHERE episode_id = ?",
            (json.dumps(result, sort_keys=True), episode_id),
        )
    trajectory.record(
        db,
        episode_id,
        "spark",
        "tool_call",
        {"tool": key + "_check", "provider": result.get("provider"), "stored_in": column},
    )


def _provider_name(provider: Any) -> str:
    return str(getattr(provider, "name", type(provider).__name__))


def _record_effect(db: Database, tool: str, client_id: str, subject_id: str, result: dict) -> None:
    db.execute(
        "INSERT OR IGNORE INTO tool_effects (client_id, tool, subject_id, result, created_at)"
        " VALUES (?, ?, ?, ?, ?)",
        (client_id, tool, subject_id, json.dumps(result, sort_keys=True), now_iso()),
    )


# ---------------------------------------------------------------------------
# thread-safety wrappers: one shared connection, so every public entry point
# serializes reads AND writes across MCP worker threads (storage.locked).


def backtest_run(
    db: Database,
    cfg: Config,
    provider: Any,
    episode_id: str,
    spec: dict,
    *,
    client_id: str | None = None,
    acknowledge_tombstone: dict | None = None,
    embedder: Any = None,
    llm: Any = None,
) -> dict:
    with db.locked():
        return _backtest_run_impl(
            db,
            cfg,
            provider,
            episode_id,
            spec,
            client_id=client_id,
            acknowledge_tombstone=acknowledge_tombstone,
            embedder=embedder,
            llm=llm,
        )


def significance_gate(db: Database, cfg: Config, episode_id: str) -> dict:
    with db.locked():
        return _significance_gate_impl(db, cfg, episode_id)


def causal_check(
    db: Database,
    cfg: Config,
    provider: Any,
    episode_id: str,
    spec: dict,
    *,
    client_id: str | None = None,
) -> dict:
    with db.locked():
        return _causal_check_impl(db, cfg, provider, episode_id, spec, client_id=client_id)


def risk_decompose(
    db: Database,
    cfg: Config,
    provider: Any,
    episode_id: str,
    spec: dict,
    *,
    client_id: str | None = None,
) -> dict:
    with db.locked():
        return _risk_decompose_impl(db, cfg, provider, episode_id, spec, client_id=client_id)


def record_family_outcome(
    db: Database,
    cfg: Config,
    family: str,
    *,
    success: bool,
    episode_id: str,
    failure_reason: str | None = None,
) -> dict:
    with db.locked():
        return _record_family_outcome_impl(
            db, cfg, family, success=success, episode_id=episode_id, failure_reason=failure_reason
        )
