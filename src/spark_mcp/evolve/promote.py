"""Computed promotion decisions for artifact evolution (plan §2.E3).

`decide` promotes iff (a) the challenger's significance-gate verdict passes
AND its C6 `promotion_blockers` list is empty, and (b) the numeric margin
over the incumbent clears the configured thresholds. The decision bit is
always computed from numbers — narration text is not an input (the
signature takes episode IDs only), and no LLM boolean is consulted.

The margin check is a pure comparison on two *stored* `corrected_metrics`
rows, using each row's stored gate-time `M_family` (never recomputed —
trials recorded between gate and decide must not shift the comparison).

Held-out at promotion mirrors the skill-gate shape (held-out reported +
gap-alarmed, never the comparison input): the decision carries an
`oos_report` advisory with any `oos_*` keys found on the challenger's
metrics, but the promote boolean's inputs are in-sample significance +
margin only. No provider emits OOS metric keys today, so `oos_report` is
`{available: False}` until an OOS producer exists.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import ledger
from ..config import Config
from ..storage import Database
from .fitness import FitnessReport, fitness_from_episode


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    reason: str
    challenger: FitnessReport
    incumbent: FitnessReport | None
    dsr_margin: float
    sr_margin: float
    oos_report: dict  # advisory only: {available: False} until an OOS
    # producer exists; never an input to `promote`


def _promotion_blockers(db: Database, episode: dict) -> list[str]:
    """C6 blockers, verbatim from gates/service.py::_significance_gate_impl.

    The gate reports these without flipping `pass`, so `decide` must check
    them itself. The strings are pinned to match the gate implementation.
    """
    hypothesis = ledger.get_hypothesis(db, episode["hypothesis_id"]) or {}
    blockers: list[str] = []
    if hypothesis.get("hypothesis_type") == "causal" and not hypothesis.get("dag"):
        blockers.append("no causal promotion without DAG (C6)")
    if not hypothesis.get("falsifier"):
        blockers.append("no promotion of any kind without falsifier (C6)")
    return blockers


def _oos_report(episode: dict) -> dict:
    """Advisory OOS evidence carried alongside (never into) the decision."""
    metrics = episode.get("metrics") or {}
    oos = {k: v for k, v in metrics.items() if k == "oos" or k.startswith("oos_")}
    if not oos:
        return {"available": False}
    return {"available": True, "metrics": oos}


def decide(
    db: Database,
    cfg: Config,
    challenger_episode_id: str,
    incumbent_episode_id: str | None,
) -> PromotionDecision:
    """Compute the promotion decision for a challenger episode.

    First promotion (no incumbent): gate pass + blockers empty +
    `evolution_first_promotion_floor` on the DSR probability. Later
    promotions: gate pass + blockers empty + strict margins over the
    incumbent (`evolution_dsr_margin`, `evolution_sr_margin`). Ties never
    promote (strict `>` on both margins).
    """
    challenger = fitness_from_episode(db, cfg, challenger_episode_id)
    challenger_episode = ledger.get_episode(db, challenger_episode_id)
    assert challenger_episode is not None  # fitness_from_episode already refused otherwise
    corrected = challenger_episode.get("corrected_metrics") or {}
    gate_pass = bool(corrected.get("pass"))
    blockers = _promotion_blockers(db, challenger_episode)
    oos_report = _oos_report(challenger_episode)

    if incumbent_episode_id is None:
        floor = float(cfg.research.evolution_first_promotion_floor)
        dsr_margin = challenger.dsr_prob - floor
        sr_margin = challenger.sr  # margin over the SR > 0 requirement
        if not gate_pass:
            return PromotionDecision(False, "rejected: challenger failed the significance gate",
                                     challenger, None, dsr_margin, sr_margin, oos_report)
        if blockers:
            return PromotionDecision(False, f"rejected: C6 blockers: {'; '.join(blockers)}",
                                     challenger, None, dsr_margin, sr_margin, oos_report)
        if challenger.dsr_prob < floor:
            return PromotionDecision(
                False,
                f"rejected: DSR prob {challenger.dsr_prob:.4f} below first-promotion"
                f" floor {floor:.4f}",
                challenger, None, dsr_margin, sr_margin, oos_report)
        return PromotionDecision(True,
                                 f"promoted: first promotion, DSR prob {challenger.dsr_prob:.4f}"
                                 f" at/above floor {floor:.4f}",
                                 challenger, None, dsr_margin, sr_margin, oos_report)

    incumbent = fitness_from_episode(db, cfg, incumbent_episode_id)
    dsr_margin = challenger.dsr_prob - incumbent.dsr_prob
    sr_margin = challenger.sr - incumbent.sr
    dsr_req = float(cfg.research.evolution_dsr_margin)
    sr_req = float(cfg.research.evolution_sr_margin)
    if not gate_pass:
        return PromotionDecision(False, "rejected: challenger failed the significance gate",
                                 challenger, incumbent, dsr_margin, sr_margin, oos_report)
    if blockers:
        return PromotionDecision(False, f"rejected: C6 blockers: {'; '.join(blockers)}",
                                 challenger, incumbent, dsr_margin, sr_margin, oos_report)
    if not dsr_margin > dsr_req:
        return PromotionDecision(
            False,
            f"rejected: DSR margin {dsr_margin:.4f} not above required {dsr_req:.4f}",
            challenger, incumbent, dsr_margin, sr_margin, oos_report)
    if not sr_margin > sr_req:
        return PromotionDecision(
            False,
            f"rejected: SR margin {sr_margin:.4f} not above required {sr_req:.4f}",
            challenger, incumbent, dsr_margin, sr_margin, oos_report)
    return PromotionDecision(
        True,
        f"promoted: DSR margin {dsr_margin:.4f} and SR margin {sr_margin:.4f}"
        " clear the required margins",
        challenger, incumbent, dsr_margin, sr_margin, oos_report)
