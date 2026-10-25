"""Ordered dispatch pipeline (Fix D): composable pre-dispatch stages.

The gates service path already embodies an ordered dispatch sequence
(tombstone pre-check -> trial insert -> budget enforcement -> provider
dispatch -> verdict record); this module makes that order an explicit,
testable abstraction instead of a fork: the real step callables live in
gates/service.py and delegate to the existing ledger/budget/trajectory
helpers, while this module owns only ordering, the profile registry, and
transaction grouping.

Step chain (canonical order):
    TombstonePreCheck -> BudgetCheck -> TrialInsert -> ProviderDispatch
    -> VerdictRecord

Profiles (step-name lists; service.py binds names to callables):
    gated-backtest: all five steps, enforcing mode (the backtest_run path).
    read-only:      tombstone + budget assessment only, no writes, no raise.
    evolution-step: all five steps, enforcing mode, flagged for the
                    evolution runner (reserved ctx.evolution flag).

Transaction grouping: Pipeline.run executes consecutive transactional
steps inside ONE database transaction and non-transactional steps
outside it. The tombstone step is non-transactional because its check
must persist on the episode even when it blocks; the remaining gated
steps share one transaction so pending-row-before-dispatch exactly-once
semantics (UNIQUE(episode_id, idem_key), budget rollback, failed-trial
persistence) are preserved byte-for-byte.

Server routing: every MCP tool handler in server.py additionally executes
as a single TOOL_CALL step of its declared tool profile ("tool-gated" /
"tool-evolution" / "tool-standard"). Those profiles contain ONLY the
tool-call step — no tombstone/budget/trial side effects are invented for
tools whose services do not already have them — and the step is
non-transactional, so services keep owning their own transactions. The
tool handler runs inside Pipeline.run with byte-identical behaviour.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import groupby
from typing import Any

# ---------------------------------------------------------------------------
# step names (canonical chain order)

TOMBSTONE_PRECHECK = "tombstone_precheck"
BUDGET_CHECK = "budget_check"
TRIAL_INSERT = "trial_insert"
PROVIDER_DISPATCH = "provider_dispatch"
VERDICT_RECORD = "verdict_record"

STEP_ORDER: tuple[str, ...] = (
    TOMBSTONE_PRECHECK,
    BUDGET_CHECK,
    TRIAL_INSERT,
    PROVIDER_DISPATCH,
    VERDICT_RECORD,
)

# Server-routing step: the MCP tool handler itself, bound by server.py.
# Appended AFTER the gated chain so the canonical relative order of the
# five gated steps is preserved byte-for-byte (STEP_ORDER is frozen;
# validation accepts the extended order).
TOOL_CALL = "tool_call"

EXTENDED_STEP_ORDER: tuple[str, ...] = STEP_ORDER + (TOOL_CALL,)

# ---------------------------------------------------------------------------
# profile registry: profile name -> ordered step names (pure data; no
# service imports here so this module stays cycle-free)

PROFILES: dict[str, tuple[str, ...]] = {
    "gated-backtest": STEP_ORDER,
    "read-only": (TOMBSTONE_PRECHECK, BUDGET_CHECK),
    "evolution-step": STEP_ORDER,
}


def register_profile(name: str, steps: tuple[str, ...] | list[str]) -> None:
    """Add or replace a profile's step list. Step names should come from
    STEP_ORDER (order is validated); unknown names raise KeyError."""
    ordered = tuple(steps)
    unknown = [s for s in ordered if s not in EXTENDED_STEP_ORDER]
    if unknown:
        raise KeyError(f"unknown dispatch step(s): {unknown}")
    if tuple(sorted(ordered, key=EXTENDED_STEP_ORDER.index)) != ordered:
        raise ValueError(f"profile {name!r} steps violate canonical order: {ordered}")
    PROFILES[name] = ordered


# Server-routing profiles: single TOOL_CALL step each (see module
# docstring). Names document the tool family; all three route the handler
# through Pipeline.run with no added side effects.
register_profile("tool-gated", (TOOL_CALL,))  # gates-service tools
register_profile("tool-evolution", (TOOL_CALL,))  # evolution + skill lifecycle
register_profile("tool-standard", (TOOL_CALL,))  # every other tool


# ---------------------------------------------------------------------------
# context + steps + pipeline


@dataclass
class DispatchContext:
    """Mutable state threaded through the pipeline.

    Inputs (set by the caller): db, cfg, provider, episode_id, spec/bt,
    episode, hypothesis, client_id, acknowledge_tombstone, embedder, llm,
    mode ("enforce" | "assess"), source (trial-row source label).
    Outputs (set by steps): check, trial_id, metrics, error, result,
    warnings, would_block, halted, events.
    """

    db: Any
    cfg: Any
    provider: Any = None
    episode_id: str = ""
    bt: Any = None  # validated BacktestSpec (set by the service prologue)
    episode: dict = field(default_factory=dict)
    hypothesis: dict = field(default_factory=dict)
    client_id: str | None = None
    acknowledge_tombstone: dict | None = None
    embedder: Any = None
    llm: Any = None
    mode: str = "enforce"  # "enforce" (write+raise) or "assess" (read-only)
    source: str = "spark"
    evolution: bool = False  # set for the evolution-step profile

    check: dict = field(default_factory=dict)
    hash_of_spec: str = ""
    idem_key: str = ""
    trial_id: str = ""
    family: str = ""
    metrics: dict | None = None
    error: Any = None  # captured provider failure (raised after commit)
    result: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    would_block: dict = field(default_factory=dict)
    halted: bool = False  # set by TrialInsert on idempotent replay
    events: list = field(default_factory=list)  # step-execution record


@dataclass(frozen=True)
class Step:
    """One composable pipeline stage: a name plus its callable."""

    name: str
    func: Callable[[DispatchContext], None]
    transactional: bool = True


class Pipeline:
    """Runs steps in order, recording each execution in ctx.events.

    Consecutive transactional steps share one db transaction; a step that
    raises aborts the pipeline (its transaction rolls back); a step may
    set ctx.halted to short-circuit the rest (idempotent replay).
    """

    def __init__(self, steps: list[Step] | tuple[Step, ...]) -> None:
        names = [s.name for s in steps]
        unknown = [n for n in names if n not in EXTENDED_STEP_ORDER]
        if unknown:
            raise KeyError(f"unknown dispatch step(s): {unknown}")
        if tuple(sorted(names, key=EXTENDED_STEP_ORDER.index)) != tuple(names):
            raise ValueError(f"pipeline steps violate canonical order: {names}")
        self.steps = list(steps)

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.steps]

    def run(self, ctx: DispatchContext) -> DispatchContext:
        for transactional, group in groupby(self.steps, key=lambda s: s.transactional):
            chunk = list(group)
            if transactional:
                with ctx.db.transaction():
                    self._run_chunk(ctx, chunk)
            else:
                self._run_chunk(ctx, chunk)
            if ctx.halted:
                break
        return ctx

    @staticmethod
    def _run_chunk(ctx: DispatchContext, chunk: list[Step]) -> None:
        for step in chunk:
            if ctx.halted:
                break
            ctx.events.append(step.name)
            step.func(ctx)


def build_pipeline(
    profile: str,
    resolve: Callable[[str], Callable[[DispatchContext], None]],
    *,
    transactional: Callable[[str], bool] | None = None,
) -> Pipeline:
    """Compose a Pipeline for `profile`, binding each step name via
    `resolve`. Unknown profiles raise KeyError. `transactional`
    optionally overrides the per-step transaction flag (default: every
    step transactional except the tombstone pre-check, whose episode
    write must survive a block, and the tool-call step, which is pure
    routing — services own their own transactions)."""
    try:
        names = PROFILES[profile]
    except KeyError:
        raise KeyError(f"unknown dispatch profile {profile!r} (have {sorted(PROFILES)})") from None
    is_txn = transactional or (lambda name: name not in (TOMBSTONE_PRECHECK, TOOL_CALL))
    return Pipeline([Step(name=n, func=resolve(n), transactional=is_txn(n)) for n in names])
