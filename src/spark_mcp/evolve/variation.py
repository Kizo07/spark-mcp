"""Two-level variation for artifact evolution (plan §2.D).

Separates *what to try next* from *how to implement it correctly* (RD's
HypothesisGen vs CoSTEER split, without the in-place-yield contract —
every function returns new objects, never mutates inputs):

- Level 1 (`vary_hypothesis`): successor hypothesis dict via component B
  (LLM reflective) or its deterministic field-edit fallback. The caller
  inserts the dict via `ledger.propose_hypothesis` — the C6 refusal rule is
  the schema gate, and a refused proposal is recorded as a failed variation
  by the run driver, never retried silently.
- Level 2 (`vary_spec`): `BacktestSpec` variant over a fixed configured base
  (`ResearchConfig.evolution_base_spec`): params inherit the parent
  episode's and perturb `lookback` + `type` from the run's seeded `rng` in
  a fixed consumption order; the spec seed derives from
  `sha1(parent_spec_seed|round)` (hash-based, consumes no rng state).

Costs are NEVER perturbed: inheriting (or defaulting) fixed costs closes
the with/without-cost selection skew — varying costs would let evolution
"improve" by assuming cheaper trading. Signal/universe/dates/splits stay on
the configured base: the plan's universe/date-window moves need a
data-availability oracle to stay PIT-safe, so they are deferred — fixed
configured windows are trivially PIT-safe. Executor/runner agents stay
QuantFlow-side; this module only varies the spec, never executes code.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

from ..config import Config
from ..llm import LLMClient
from ..providers.protocols import BacktestSpec
from ..storage import Database
from .reflect import reflect_variation

#: Base-template keys `vary_spec` requires (missing → ValueError naming the
#: key; pydantic validates shapes below that).
_BASE_REQUIRED_KEYS = ("signal_ref", "params", "universe_pit", "dates", "costs", "splits")


def vary_hypothesis(
    db: Database,
    cfg: Config,
    parent: dict | None,
    ancestors_render: str,
    feedback: dict | None,
    rng: random.Random,
    llm: LLMClient | None,
) -> dict:
    """Level-1 variation: successor hypothesis dict for `parent` (or a new
    root when `parent` is None — root synthesis needs an LLM; the
    deterministic fallback cannot invent a hypothesis from nothing).

    `feedback` is the latest evolution_feedback row (or None); `db`/`cfg`
    are reserved for future template roots and currently unused. Returns a
    fresh dict (never mutates `parent`); the caller inserts it via
    `ledger.propose_hypothesis` with the appropriate `parent_hypothesis_ids`.
    """
    del db, cfg  # reserved for future template roots; variation is parent+feedback only
    if parent is not None and not isinstance(parent, dict):
        raise ValueError("parent must be a hypothesis dict or None")
    if feedback is not None and not isinstance(feedback, dict):
        raise ValueError("feedback must be an evolution_feedback row dict or None")
    proposal = reflect_variation(
        parent=parent,
        feedback=feedback,
        ancestors_render=ancestors_render,
        rng=rng,
        llm=llm,
        schema="hypothesis",
    )
    assert isinstance(proposal.body, dict)
    return dict(proposal.body)


def _base_template(cfg: Config) -> dict:
    base = cfg.research.evolution_base_spec
    if not isinstance(base, dict):
        raise ValueError("evolution_base_spec must be a mapping")
    missing = [k for k in _BASE_REQUIRED_KEYS if k not in base]
    if missing:
        raise ValueError(f"evolution_base_spec is missing keys: {missing}")
    return base


def _spec_seed(hypothesis_id: str, parent_seed: Any, round_no: int) -> int:
    """`sha1(parent_spec_seed|round)` (hash-based, consumes no rng state);
    parentless round-0 specs hash the hypothesis id instead (unique per
    hypothesis, so distinct runs never share a seed stream).
    """
    if parent_seed is None:
        material = f"{hypothesis_id}|{round_no}"
    else:
        material = f"{parent_seed}|{round_no}"
    digest = hashlib.sha1(material.encode()).digest()
    # 63-bit: the seed is stored in a SQLite INTEGER column (signed 64-bit).
    return int.from_bytes(digest[:8], "little") & 0x7FFF_FFFF_FFFF_FFFF


def vary_spec(
    db: Database,
    cfg: Config,
    hypothesis: dict,
    parent_episode: dict | None,
    round_no: int,
    rng: random.Random,
) -> BacktestSpec:
    """Level-2 variation: a dispatchable `BacktestSpec` for `hypothesis`.

    Params start from the parent episode's (or the configured base params
    when parentless / the parent has none), then `lookback` and `type` are
    resampled from `evolution_lookback_grid` / `evolution_type_grid` — fixed
    rng consumption order: lookback first, type second (one `randrange` each).
    Holding/turnover inherit unperturbed; costs inherit the parent episode's
    (or base costs) and are never perturbed. Validated by pydantic before
    return; a `ValidationError` propagates for the driver to record as a
    failed variation. Never mutates its inputs.
    """
    del db  # no ledger reads: the spec derives from (hypothesis, parent, cfg, rng)
    if not isinstance(hypothesis, dict) or not hypothesis.get("hypothesis_id"):
        raise ValueError("hypothesis must be a dict carrying a hypothesis_id")
    if parent_episode is not None and not isinstance(parent_episode, dict):
        raise ValueError("parent_episode must be an episode dict or None")
    if isinstance(round_no, bool) or not isinstance(round_no, int) or round_no < 0:
        raise ValueError(f"round_no must be an int >= 0; got {round_no!r}")
    base = _base_template(cfg)

    lookback_grid = list(cfg.research.evolution_lookback_grid or [])
    if not lookback_grid or not all(
        isinstance(v, int) and not isinstance(v, bool) and v >= 1 for v in lookback_grid
    ):
        raise ValueError("evolution_lookback_grid must be a non-empty list of ints >= 1")
    type_grid = list(cfg.research.evolution_type_grid or [])
    if not type_grid or not all(isinstance(v, str) and v for v in type_grid):
        raise ValueError("evolution_type_grid must be a non-empty list of names")

    base_params = dict(base["params"]) if isinstance(base["params"], dict) else {}
    parent_params = (parent_episode or {}).get("params")
    params = dict(base_params)
    if isinstance(parent_params, dict):
        params.update(parent_params)
    # Fixed consumption order: lookback, then type.
    params["lookback"] = lookback_grid[rng.randrange(len(lookback_grid))]
    params["type"] = type_grid[rng.randrange(len(type_grid))]

    parent_costs = (parent_episode or {}).get("costs")
    if isinstance(parent_costs, dict) and parent_costs:
        costs = dict(parent_costs)
    else:
        costs = dict(base["costs"]) if isinstance(base["costs"], dict) else {}

    hypothesis_id = str(hypothesis["hypothesis_id"])
    parent_seed = (parent_episode or {}).get("seed")
    return BacktestSpec.model_validate(
        {
            "hypothesis_id": hypothesis_id,
            "signal_ref": base["signal_ref"],
            "params": params,
            "universe_pit": base["universe_pit"],
            "dates": base["dates"],
            "costs": costs,
            "splits": base["splits"],
            "seed": _spec_seed(hypothesis_id, parent_seed, round_no),
        }
    )
