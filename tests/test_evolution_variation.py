"""Two-level variation (evolution upgrade plan §2.D, §5).

Level 1 emits C6-valid successor hypotheses (insertable via
propose_hypothesis); level 2 emits dispatchable BacktestSpecs with
hash-derived seeds and fixed-order rng perturbations. Neither level
mutates its inputs; costs are never perturbed.
"""

import copy
import hashlib
import json
import random

import pytest
from pydantic import ValidationError

from spark_mcp import ledger
from spark_mcp.config import Config
from spark_mcp.evolve import variation as variation_mod
from spark_mcp.gates import service as gates
from spark_mcp.llm import FakeLLM
from spark_mcp.providers.simulated import _SIGNAL_TYPES as PROVIDER_SIGNAL_TYPES
from spark_mcp.providers.simulated import SimulatedProvider

FAMILY = "toy_momentum"

PROPOSE = dict(
    research_family=FAMILY,
    hypothesis_type="predictive",
    mechanism="12-1 momentum in toy universe",
    expected_direction="positive",
    horizon="21d",
    scope="cross-section",
    falsifier="IC <= 0 net of costs",
    predicted_ic=0.03,
    predicted_sigma=0.01,
    universe="toy",
)


def _parent_row(db, cfg, **overrides):
    hyp_id = ledger.propose_hypothesis(db, cfg, **{**PROPOSE, **overrides})["hypothesis_id"]
    return ledger.get_hypothesis(db, hyp_id)


def test_level1_outputs_insert_via_propose(db, cfg):
    """Property-style: 20 seeded parents (predictive + causal) all yield
    successor dicts that pass propose_hypothesis validation."""
    causal = dict(
        PROPOSE,
        hypothesis_type="causal",
        dag={"nodes": ["a", "b"], "edges": [["a", "b"]]},
        adjustment_set=["z"],
    )
    for seed in range(20):
        base = causal if seed % 2 else PROPOSE
        parent_id = ledger.propose_hypothesis(db, cfg, **base)["hypothesis_id"]
        parent = ledger.get_hypothesis(db, parent_id)
        body = variation_mod.vary_hypothesis(
            db, cfg, parent, "Trial 1: ...", None, random.Random(seed), None
        )
        child = ledger.propose_hypothesis(
            db, cfg, parent_hypothesis_ids=[parent_id], **body
        )
        stored = ledger.get_hypothesis(db, child["hypothesis_id"])
        assert stored["parent_hypothesis_ids"] == [parent_id]
        assert stored["mechanism"] == body["mechanism"]


def test_level1_llm_path_and_root(db, cfg):
    parent = _parent_row(db, cfg)
    reply = {
        "mechanism": "m", "expected_direction": "positive", "horizon": "5d",
        "scope": "s", "falsifier": "f", "predicted_ic": 0.01, "predicted_sigma": 0.02,
    }
    body = variation_mod.vary_hypothesis(
        db, cfg, parent, "", None, random.Random(0),
        FakeLLM({"successor": json.dumps(reply)}),
    )
    assert body["horizon"] == "5d"
    root_reply = dict(
        reply, research_family=FAMILY, hypothesis_type="predictive", universe="toy"
    )
    root = variation_mod.vary_hypothesis(
        db, cfg, None, "", None, random.Random(0),
        FakeLLM({"successor": json.dumps(root_reply)}),
    )
    inserted = ledger.propose_hypothesis(db, cfg, parent_hypothesis_ids=[], **root)
    assert ledger.get_hypothesis(db, inserted["hypothesis_id"])["parent_hypothesis_ids"] == []


def test_level1_never_mutates_and_validates_inputs(db, cfg):
    parent = _parent_row(db, cfg)
    feedback = {"decision": "rejected: x", "n": [1]}
    snapshot = (copy.deepcopy(parent), copy.deepcopy(feedback))
    variation_mod.vary_hypothesis(
        db, cfg, parent, "anc", feedback, random.Random(3), None
    )
    assert (parent, feedback) == snapshot
    with pytest.raises(ValueError, match="hypothesis dict or None"):
        variation_mod.vary_hypothesis(db, cfg, "hyp_x", "", None, random.Random(0), None)
    with pytest.raises(ValueError, match="feedback row dict or None"):
        variation_mod.vary_hypothesis(db, cfg, parent, "", "txt", random.Random(0), None)


def _episode_row(db, cfg, hyp_id, seed=11, params=None, costs=None):
    ep_id = ledger.register_hypothesis(db, cfg, hyp_id, {"signal": "mom"}, seed)["episode_id"]
    db.execute(
        "UPDATE episodes SET params = ?, costs = ? WHERE episode_id = ?",
        (
            json.dumps(params if params is not None else {"type": "momentum", "lookback": 20}),
            json.dumps(costs if costs is not None else {"bps": 5.0, "slippage": 2.0}),
            ep_id,
        ),
    )
    return ledger.get_episode(db, ep_id)


def test_level2_shape_inheritance_and_seed_derivation(db, cfg):
    hyp_id = ledger.propose_hypothesis(db, cfg, **PROPOSE)["hypothesis_id"]
    hypothesis = ledger.get_hypothesis(db, hyp_id)
    parent = _episode_row(
        db, cfg, hyp_id, seed=11,
        params={"type": "momentum", "lookback": 20, "holding": 5, "turnover": 0.5},
        costs={"bps": 7.0, "slippage": 1.0, "borrow": 0.5},
    )
    spec = variation_mod.vary_spec(db, cfg, hypothesis, parent, 3, random.Random(0))
    assert spec.hypothesis_id == hyp_id
    assert spec.params["lookback"] in cfg.research.evolution_lookback_grid
    assert spec.params["type"] in cfg.research.evolution_type_grid
    # holding/turnover inherit unperturbed; costs inherit exactly
    assert spec.params["holding"] == 5 and spec.params["turnover"] == 0.5
    assert (spec.costs.bps, spec.costs.slippage, spec.costs.borrow) == (7.0, 1.0, 0.5)
    # sha1(parent_spec_seed|round), hash-based, masked to SQLite INTEGER
    want = int.from_bytes(hashlib.sha1(b"11|3").digest()[:8], "little") & 0x7FFF_FFFF_FFFF_FFFF
    assert spec.seed == want
    # parentless: sha1(hypothesis_id|round)
    orphan = variation_mod.vary_spec(db, cfg, hypothesis, None, 0, random.Random(0))
    want_root = (
        int.from_bytes(hashlib.sha1(f"{hyp_id}|0".encode()).digest()[:8], "little")
        & 0x7FFF_FFFF_FFFF_FFFF
    )
    assert orphan.seed == want_root
    assert orphan.costs.bps == cfg.research.evolution_base_spec["costs"]["bps"]


def test_level2_fixed_consumption_order(db, cfg):
    hyp_id = ledger.propose_hypothesis(db, cfg, **PROPOSE)["hypothesis_id"]
    hypothesis = ledger.get_hypothesis(db, hyp_id)
    look_grid = cfg.research.evolution_lookback_grid
    type_grid = cfg.research.evolution_type_grid
    for seed in (0, 1, 42):
        spec = variation_mod.vary_spec(db, cfg, hypothesis, None, 0, random.Random(seed))
        check = random.Random(seed)
        look = look_grid[check.randrange(len(look_grid))]
        typ = type_grid[check.randrange(len(type_grid))]
        assert (spec.params["lookback"], spec.params["type"]) == (look, typ)
    # rerun determinism incl. fresh identical rngs
    a = variation_mod.vary_spec(db, cfg, hypothesis, None, 1, random.Random(9))
    b = variation_mod.vary_spec(db, cfg, hypothesis, None, 1, random.Random(9))
    assert a.model_dump() == b.model_dump()


def test_level2_grids_come_from_config(db, cfg):
    hyp_id = ledger.propose_hypothesis(db, cfg, **PROPOSE)["hypothesis_id"]
    hypothesis = ledger.get_hypothesis(db, hyp_id)
    cfg.research.evolution_lookback_grid = [33]
    cfg.research.evolution_type_grid = ["reversal"]
    spec = variation_mod.vary_spec(db, cfg, hypothesis, None, 0, random.Random(0))
    assert (spec.params["lookback"], spec.params["type"]) == (33, "reversal")


def test_level2_default_type_grid_within_provider_set(cfg):
    assert set(cfg.research.evolution_type_grid) <= set(PROVIDER_SIGNAL_TYPES)


def test_level2_spec_dispatches_end_to_end(db, cfg):
    """The emitted spec runs through the real trial governor + DSR gate."""
    hyp_id = ledger.propose_hypothesis(db, cfg, **PROPOSE)["hypothesis_id"]
    hypothesis = ledger.get_hypothesis(db, hyp_id)
    spec = variation_mod.vary_spec(db, cfg, hypothesis, None, 0, random.Random(2))
    reg = ledger.register_hypothesis(db, cfg, hyp_id, {"signal": "evo"}, spec.seed)
    provider = SimulatedProvider(cfg.artifacts_dir())
    gates.backtest_run(db, cfg, provider, reg["episode_id"], spec.model_dump())
    verdict = gates.significance_gate(db, cfg, reg["episode_id"])
    assert "pass" in verdict


def test_level2_never_mutates_inputs(db, cfg):
    hyp_id = ledger.propose_hypothesis(db, cfg, **PROPOSE)["hypothesis_id"]
    hypothesis = ledger.get_hypothesis(db, hyp_id)
    parent = _episode_row(db, cfg, hyp_id)
    snapshot = (copy.deepcopy(hypothesis), copy.deepcopy(parent), cfg.model_dump())
    variation_mod.vary_spec(db, cfg, hypothesis, parent, 0, random.Random(0))
    assert copy.deepcopy(hypothesis) == snapshot[0]
    assert copy.deepcopy(parent) == snapshot[1]
    assert cfg.model_dump() == snapshot[2]


def test_level2_refusals(db, cfg):
    hyp_id = ledger.propose_hypothesis(db, cfg, **PROPOSE)["hypothesis_id"]
    hypothesis = ledger.get_hypothesis(db, hyp_id)
    with pytest.raises(ValueError, match="hypothesis_id"):
        variation_mod.vary_spec(db, cfg, {}, None, 0, random.Random(0))
    with pytest.raises(ValueError, match="round_no"):
        variation_mod.vary_spec(db, cfg, hypothesis, None, -1, random.Random(0))
    with pytest.raises(ValueError, match="episode dict or None"):
        variation_mod.vary_spec(db, cfg, hypothesis, "ep_x", 0, random.Random(0))
    broken = dict(cfg.research.evolution_base_spec)
    del broken["dates"]
    cfg.research.evolution_base_spec = broken
    with pytest.raises(ValueError, match="missing keys.*dates"):
        variation_mod.vary_spec(db, cfg, hypothesis, None, 0, random.Random(0))
    bad_dates = copy.deepcopy(Config().research.evolution_base_spec)
    bad_dates["dates"]["is_start"] = "not-a-date"
    cfg.research.evolution_base_spec = bad_dates
    with pytest.raises(ValidationError):
        variation_mod.vary_spec(db, cfg, hypothesis, None, 0, random.Random(0))
    cfg.research.evolution_base_spec = Config().research.evolution_base_spec
    cfg.research.evolution_lookback_grid = []
    with pytest.raises(ValueError, match="lookback_grid"):
        variation_mod.vary_spec(db, cfg, hypothesis, None, 0, random.Random(0))
