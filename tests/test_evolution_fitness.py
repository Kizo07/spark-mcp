"""Execution-grounded fitness (evolution upgrade plan §2.A, §5).

Fitness scores candidates by executed provider/backtest outcomes: the
in-sample `corrected_metrics` written by significance_gate, the episode's
cost basis, and the `metrics.simulated` provenance bit only. Held-out keys
are structurally invisible to fitness.
"""

import json

import pytest

from spark_mcp import ledger
from spark_mcp.errors import ToolError
from spark_mcp.evolve import fitness as fitness_mod
from spark_mcp.gates import service as gates
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


def make_spec(hypothesis_id: str, seed: int) -> dict:
    return {
        "hypothesis_id": hypothesis_id,
        "signal_ref": "artifacts://signals/mom.json",
        "params": {"type": "momentum", "lookback": 20},
        "universe_pit": "TOYA,TOYB,TOYC,TOYD,TOYE,TOYF,TOYG,TOYH@2019-01-01",
        "dates": {
            "is_start": "2019-01-01",
            "is_end": "2019-12-31",
            "oos_start": "2020-01-01",
            "oos_end": "2020-06-30",
        },
        "costs": {"bps": 5, "slippage": 2},
        "splits": {"purge": 5, "embargo": 5},
        "seed": seed,
    }


def _gated_episode(db, cfg, seed=11):
    """A real episode through SimulatedProvider + significance_gate."""
    hyp = ledger.propose_hypothesis(db, cfg, **PROPOSE)
    reg = ledger.register_hypothesis(db, cfg, hyp["hypothesis_id"], {"signal": "mom"}, seed)
    provider = SimulatedProvider(cfg.artifacts_dir())
    gates.backtest_run(db, cfg, provider, reg["episode_id"], make_spec(hyp["hypothesis_id"], seed))
    gates.significance_gate(db, cfg, reg["episode_id"])
    return reg["episode_id"]


def test_fitness_reads_gate_outputs(db, cfg):
    ep = _gated_episode(db, cfg)
    episode = ledger.get_episode(db, ep)
    corrected = episode["corrected_metrics"]
    report = fitness_mod.fitness_from_episode(db, cfg, ep)
    assert report.episode_id == ep
    assert report.sr == pytest.approx(corrected["sr"])
    assert report.dsr_prob == pytest.approx(corrected["dsr"])
    assert report.p_value == pytest.approx(corrected["p_value"])
    assert report.p_value == pytest.approx(1.0 - corrected["dsr"])
    assert report.M_family == corrected["M_family"]
    assert report.T == corrected["T"]
    assert report.skew == pytest.approx(corrected["skew"])
    assert report.kurt == pytest.approx(corrected["kurt"])
    assert report.heavy_tail_caveat == corrected["heavy_tail_caveat"]
    assert report.degenerate == corrected["degenerate"]
    assert report.simulated is True
    assert report.costs_basis == {"bps": 5.0, "slippage": 2.0, "borrow": 0.0}


def test_fitness_refuses_pending_gate(db, cfg):
    hyp = ledger.propose_hypothesis(db, cfg, **PROPOSE)
    reg = ledger.register_hypothesis(db, cfg, hyp["hypothesis_id"], {"signal": "mom"}, 11)
    provider = SimulatedProvider(cfg.artifacts_dir())
    gates.backtest_run(db, cfg, provider, reg["episode_id"], make_spec(hyp["hypothesis_id"], 11))
    with pytest.raises(ToolError, match="pending_gate"):
        fitness_mod.fitness_from_episode(db, cfg, reg["episode_id"])


def test_fitness_refuses_missing_corrected_metrics(db, cfg):
    hyp = ledger.propose_hypothesis(db, cfg, **PROPOSE)
    reg = ledger.register_hypothesis(db, cfg, hyp["hypothesis_id"], {"signal": "mom"}, 11)
    with pytest.raises(ToolError, match="no gated corrected_metrics"):
        fitness_mod.fitness_from_episode(db, cfg, reg["episode_id"])


def test_fitness_refuses_unknown_episode(db, cfg):
    with pytest.raises(ToolError, match="unknown episode_id"):
        fitness_mod.fitness_from_episode(db, cfg, "ep_doesnotexist")


def test_fitness_refuses_zero_cost_basis(db, cfg):
    ep = _gated_episode(db, cfg)
    db.execute(
        "UPDATE episodes SET costs = ? WHERE episode_id = ?",
        (json.dumps({"bps": 0, "slippage": 0, "borrow": 0}), ep),
    )
    with pytest.raises(ToolError, match="zero cost basis"):
        fitness_mod.fitness_from_episode(db, cfg, ep)


def test_fitness_refuses_simulated_when_live_required(db, cfg):
    ep = _gated_episode(db, cfg)
    cfg.research.evolution_require_live_provider = True
    try:
        with pytest.raises(ToolError, match="simulated"):
            fitness_mod.fitness_from_episode(db, cfg, ep)
    finally:
        cfg.research.evolution_require_live_provider = False
    # Default config accepts simulated fitness (CI/demo provider).
    assert fitness_mod.fitness_from_episode(db, cfg, ep).simulated is True


def test_fitness_ignores_oos_keys(db, cfg):
    """Constraint 5, structural: an episode carrying oos_* keys yields a
    fitness record identical to the same episode without them."""
    ep = _gated_episode(db, cfg)
    before = fitness_mod.fitness_from_episode(db, cfg, ep)
    episode = ledger.get_episode(db, ep)
    metrics = dict(episode["metrics"])
    metrics.update({"oos_ic": 0.99, "oos_sr": 9.9, "oos_note": "held-out"})
    db.execute(
        "UPDATE episodes SET metrics = ? WHERE episode_id = ?",
        (json.dumps(metrics, sort_keys=True), ep),
    )
    after = fitness_mod.fitness_from_episode(db, cfg, ep)
    assert after == before


def test_fitness_is_deterministic(db, cfg):
    ep = _gated_episode(db, cfg)
    assert fitness_mod.fitness_from_episode(db, cfg, ep) == fitness_mod.fitness_from_episode(
        db, cfg, ep
    )
