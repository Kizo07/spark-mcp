"""Execution-grounded fitness for artifact evolution (plan §2.A).

Score candidates by *executed* provider/backtest outcomes, replacing
fixture-text coverage as the artifact-evolution fitness. The fitness record
is composed from the episode's `corrected_metrics` (written by
`gates.service.significance_gate`; never computed here — spark never
computes returns), the episode's cost basis, and one provenance bit.

Held-out exclusion is structural, not caller discipline: fitness reads
exactly three inputs — `corrected_metrics` (in-sample gate outputs; no OOS
keys exist there), `episodes.costs`, and the single `metrics.simulated`
provenance bit (a refusal input, never a score input). It never reads any
other `metrics.*` key, so present-or-future OOS keys (`oos_ic`, `oos_*`)
are invisible to fitness by construction.

Pure function of (episode row, config): trivially deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .. import ledger
from ..config import Config
from ..errors import ToolError
from ..storage import Database

#: corrected_metrics keys the fitness record requires (all written by the
#: significance gate; the pending_gate stub carries none of them).
_REQUIRED_CORRECTED_KEYS = ("sr", "dsr", "p_value", "M_family", "T")


@dataclass(frozen=True)
class FitnessReport:
    episode_id: str
    sr: float
    dsr_prob: float
    p_value: float
    M_family: int
    T: int
    skew: float
    kurt: float
    heavy_tail_caveat: bool
    degenerate: bool
    simulated: bool
    costs_basis: dict  # echoed CostModel, must sum > 0


def fitness_from_episode(db: Database, cfg: Config, episode_id: str) -> FitnessReport:
    """Compose the fitness record for a gated episode (raises ToolError).

    Refuses when `corrected_metrics` are missing or still `status:
    pending_gate` (run `significance_gate` first), when `simulated` is true
    and `evolution_require_live_provider` forbids simulated fitness, or when
    the cost basis is zero (fitness on gross numbers is refused — the
    with/without-cost skew must not enter selection).
    """
    episode = ledger.get_episode(db, episode_id)
    if episode is None:
        raise ToolError(f"unknown episode_id {episode_id!r}")
    corrected = episode.get("corrected_metrics") or {}
    if corrected.get("status") == "pending_gate" or not corrected:
        raise ToolError(
            f"episode {episode_id} has no gated corrected_metrics"
            " (status pending_gate); run significance_gate first"
        )
    missing = [k for k in _REQUIRED_CORRECTED_KEYS if corrected.get(k) is None]
    if missing:
        raise ToolError(
            f"episode {episode_id} has incomplete corrected_metrics"
            f" (missing {missing}); run significance_gate first"
        )

    # Provenance bit only — this is the single metrics.* key fitness may
    # read. Every other metrics key (ic_mean, gross_sr, oos_*, ...) is
    # invisible here by construction (see module docstring).
    metrics: dict[str, Any] = episode.get("metrics") or {}
    simulated = bool(metrics.get("simulated", False))
    if simulated and cfg.research.evolution_require_live_provider:
        raise ToolError(
            f"episode {episode_id} is simulated and evolution_require_live_provider"
            " is set; simulated fitness is refused"
        )

    # Cost basis comes from episodes.costs (the {bps, slippage, borrow, ...}
    # JSON written by backtest_run), never from metrics.*.
    costs: dict[str, Any] = episode.get("costs") or {}
    bps = float(costs.get("bps", 0.0))
    slippage = float(costs.get("slippage", 0.0))
    borrow = float(costs.get("borrow", 0.0))
    if bps + slippage + borrow <= 0.0:
        raise ToolError(
            f"episode {episode_id} has a zero cost basis; fitness on gross"
            " (pre-cost) numbers is refused"
        )

    return FitnessReport(
        episode_id=episode_id,
        sr=float(corrected["sr"]),
        dsr_prob=float(corrected["dsr"]),
        p_value=float(corrected["p_value"]),
        M_family=int(corrected["M_family"]),
        T=int(corrected["T"]),
        skew=float(corrected.get("skew", 0.0)),
        kurt=float(corrected.get("kurt", 3.0)),
        heavy_tail_caveat=bool(corrected.get("heavy_tail_caveat", False)),
        degenerate=bool(corrected.get("degenerate", False)),
        simulated=simulated,
        costs_basis={"bps": bps, "slippage": slippage, "borrow": borrow},
    )
