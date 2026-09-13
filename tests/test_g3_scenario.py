"""G3 — the Phase 5 program gate (plan §18.0): coverage + behavior.

Scripted scenario on the toy universe with the deterministic SimulatedProvider:
  1. >=3 families occupied in the QD archive (propose -> register -> backtest
     -> log_outcome success per family; each occupies a distinct cell);
  2. >=1 underexplored cell suggested by suggest_next AND acted on (an
     episode is logged for the suggested cell's family -> posterior update +
     archive insertion into that exact cell);
  3. >=1 tombstone-blocked re-spend with the pre-check logged (backtest_run on
     a tombstoned idea is blocked; the acknowledge_tombstone path then
     dispatches);
  4. suggestions carry decomposition + selection_probability.

The gate evidence summary is printed from the test (G3 gradeable artifacts).
"""

from __future__ import annotations

import json

import pytest

from spark_mcp import archive, ledger
from spark_mcp.errors import ToolError
from spark_mcp.explore import explore_methods, suggest_next
from spark_mcp.gates import service as g
from spark_mcp.providers.simulated import SimulatedProvider

AS_OF = "2024-06-01T00:00:00Z"

PROPOSE = dict(
    expected_direction="positive",
    scope="cross_section",
    falsifier="IC <= 0 net of costs",
    predicted_ic=0.03,
    predicted_sigma=0.01,
    universe="toy",
)

# family -> (hypothesis_type, horizon, numeric turnover for the spec params)
OCCUPY = {
    "quant": ("predictive", "20d", 0.1),
    "econometrics": ("predictive", "60d", 0.5),
    "causal": ("causal", "120d", 0.2),
}

TURNOVER_FOR_BIN = {"low": 0.1, "medium": 0.5, "high": 1.0, "very_high": 2.0}

# The tombstoned idea + its re-spend share the exact signal fingerprint
# (mechanism + scope + signal_ref + params tokens) — the failure_reason below
# names that fingerprint verbatim, so the §12.2 lexical pre-check cosine is
# high enough to block the identical re-spend for real (no mocks).
TB_MECHANISM = "momentum reversal small cap overreaction drift"
TB_SCOPE = "cross section"
TB_FAILURE_REASON = (
    "momentum reversal small cap overreaction drift cross section"
    " g3.json lookback 10 type momentum"
)


def make_spec(hypothesis_id: str, seed: int, turnover: float | None = None) -> dict:
    params = {"type": "momentum", "lookback": 10}
    if turnover is not None:
        params["turnover"] = turnover  # QD cell insertion reads this
    return {
        "hypothesis_id": hypothesis_id,
        "signal_ref": "g3.json",
        "params": params,
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


def _propose(db, cfg, family, hypothesis_type, mechanism, horizon, seed):
    return ledger.propose_hypothesis(
        db,
        cfg,
        research_family=family,
        hypothesis_type=hypothesis_type,
        mechanism=mechanism,
        idea=f"g3 {family}",
        horizon=horizon,
        **PROPOSE,
        **({"dag": {"nodes": ["a", "b"], "edges": []},
            "adjustment_set": ["market_conditions"]} if hypothesis_type == "causal" else {}),
        client_id=f"g3-propose-{family}-{seed}",
    )


def _occupy_cell(db, cfg, provider, family, hypothesis_type, horizon, turnover, seed):
    """Full loop propose -> register -> backtest -> success outcome; the QD
    trigger in log_outcome inserts the hypothesis into the family's cell."""
    hyp = _propose(db, cfg, family, hypothesis_type, f"g3 {family} mechanism", horizon, seed)
    reg = ledger.register_hypothesis(db, cfg, hyp["hypothesis_id"], {"g3": family}, seed)
    g.backtest_run(
        db, cfg, provider, reg["episode_id"],
        make_spec(hyp["hypothesis_id"], seed, turnover), embedder=None,
    )
    metrics = {"oos_ic": 0.04, "robustness": 0.6, "orthogonality": 0.5, "novelty": 0.6}
    if hypothesis_type == "causal":
        metrics["stability"] = 0.8
    out = ledger.log_outcome(
        db, cfg, reg["episode_id"], "success", metrics=metrics,
        costs={"usd": 0.10, "tokens": 50},
    )
    return hyp, reg, out


def test_g3_coverage_and_behavior(seeded, cfg, embedder):
    db = seeded  # seed corpus so explore_methods has candidates (G3 condition 4)
    provider = SimulatedProvider(cfg.artifacts_dir())

    # -- 1. >=3 families occupied in the archive --------------------------------
    occupied = {}
    for family, (htype, horizon, turnover) in OCCUPY.items():
        hyp, reg, out = _occupy_cell(db, cfg, provider, family, htype, horizon, turnover, 100)
        occupied[family] = hyp["hypothesis_id"]
        assert out["status"] == "validated"
    status0 = archive.status(db, cfg)
    cov0 = status0["coverage"]
    assert cov0["occupied_cells"] >= 3
    assert sum(1 for f, n in cov0["per_family"].items() if n > 0) >= 3

    # -- 2. underexplored cell suggested AND acted on ---------------------------
    suggestion = suggest_next(db, cfg, n=3, as_of=AS_OF, seed=2024)
    underexplored = [s for s in suggestion["suggestions"] if s["cell_trials"] == 0]
    assert underexplored, "G3 requires >=1 underexplored cell suggestion"
    pick = underexplored[0]
    cell = pick["cell_key"]
    fam = pick["family"]
    horizon = pick["descriptors"]["horizon"]
    turnover = TURNOVER_FOR_BIN[pick["descriptors"]["turnover"]]
    pre_trials = db.conn.execute(
        "SELECT trials FROM exploration_state WHERE family = ?", (fam,)
    ).fetchone()
    pre_trials = int(pre_trials["trials"]) if pre_trials else 0

    hyp, reg, out = _occupy_cell(db, cfg, provider, fam, "predictive", horizon, turnover, 200)
    assert out["status"] == "validated"
    state = db.conn.execute(
        "SELECT * FROM exploration_state WHERE family = ?", (fam,)
    ).fetchone()
    # two genuine outcome events on the fresh family: backtest_run completion
    # + log_outcome (§8.3 visit/reward aggregates — documented multiplicity)
    assert int(state["trials"]) == pre_trials + 2  # posterior ticked by the act
    post = json.loads(state["posterior"])
    row = db.conn.execute(
        "SELECT * FROM qd_cells WHERE cell_key = ?", (cell,)
    ).fetchone()
    assert row is not None, f"acted-on cell {cell} was inserted"
    assert row["family"] == fam
    assert row["elite_hypothesis_id"] == hyp["hypothesis_id"]
    acted = {
        "cell": cell, "family": fam, "posterior": post,
        "posterior_trials": int(state["trials"]),
    }

    # -- 3. tombstone-blocked re-spend with the pre-check logged ----------------
    fail_hyp = _propose(db, cfg, "econometrics", "predictive", TB_MECHANISM, "20d", 300)
    fail_reg = ledger.register_hypothesis(db, cfg, fail_hyp["hypothesis_id"], {"g3": "fail"}, 300)
    fail_out = ledger.log_outcome(
        db, cfg, fail_reg["episode_id"], "failure", failure_reason=TB_FAILURE_REASON,
    )
    tombstone_id = fail_out["tombstone_id"]
    assert tombstone_id is not None

    retry_hyp = _propose(db, cfg, "econometrics", "predictive", TB_MECHANISM, "20d", 301)
    retry_reg = ledger.register_hypothesis(
        db, cfg, retry_hyp["hypothesis_id"], {"g3": "retry"}, 301
    )
    with pytest.raises(ToolError, match="tombstone pre-check blocked"):
        g.backtest_run(
            db, cfg, provider, retry_reg["episode_id"],
            make_spec(retry_hyp["hypothesis_id"], 301), embedder=None,
        )
    blocked_ep = ledger.get_episode(db, retry_reg["episode_id"])
    assert blocked_ep["tombstone_check"]["status"] == "hit"
    assert blocked_ep["tombstone_check"]["matched_id"] == tombstone_id
    # the pre-check is logged on the trajectory (gradeable evidence)
    block_events = db.conn.execute(
        "SELECT payload FROM trajectory_events WHERE run_id = ? AND kind = 'gate'",
        (retry_reg["episode_id"],),
    ).fetchall()
    kinds = [json.loads(r["payload"]).get("kind") for r in block_events]
    assert "tombstone_block" in kinds
    # no trial was created for the blocked dispatch (M must not move)
    M_blocked = g.M_family(db, "econometrics")

    acked = g.backtest_run(
        db, cfg, provider, retry_reg["episode_id"],
        make_spec(retry_hyp["hypothesis_id"], 301), embedder=None,
        acknowledge_tombstone={"tombstone_id": tombstone_id, "rationale": "g3 ack path"},
    )
    assert acked["status"] == "testing"
    acked_ep = ledger.get_episode(db, retry_reg["episode_id"])
    assert acked_ep["tombstone_check"]["status"] == "ack"
    assert g.M_family(db, "econometrics") == M_blocked + 1

    # -- 4. suggestions carry decomposition + selection_probability -------------
    explored = explore_methods(
        db, cfg, "post earnings drift reversal in small caps", k=3,
        as_of=AS_OF, embedder=None, seed=99,
    )
    assert len(explored["suggestions"]) == 3
    for s in explored["suggestions"]:
        assert set(s["decomposition"]) == {
            "relevance", "structural_fit", "novelty", "exploration_bonus",
            "prior_success",
        }
        assert 0.0 < s["selection_probability"] <= 1.0

    # -- gate evidence summary ---------------------------------------------------
    status = archive.status(db, cfg)
    print("\n=== G3 EVIDENCE (Phase 5 gate, plan §18.0) ===")
    cov = status["coverage"]
    print(f"coverage: {cov['occupied_cells']}/{cov['total_cells']} cells;"
          f" per-family occupied: {cov['per_family']}")
    print(f"suggest_next(seed=2024) underexplored picks: "
          f"{[s['cell_key'] for s in underexplored]}")
    print(f"acted-on cell: {acted['cell']} (family {acted['family']}) ->"
          f" posterior {acted['posterior']} after {acted['posterior_trials']} trial(s);"
          f" qd_cells row inserted: yes")
    print(f"tombstone block: matched {tombstone_id} cosine "
          f"{blocked_ep['tombstone_check']['cosine']:.3f} (basis "
          f"{blocked_ep['tombstone_check']['basis']}); pre-check logged: hit;"
          f" ack path: {acked_ep['tombstone_check']['status']} -> trial {acked['trial_id']}")
    print(f"family posteriors: {status['family_posteriors']}")
    print("explore_methods: decomposition + selection_probability present on all 3 suggestions")
    print("=== G3 PASS: >=3 families occupied; >=1 suggested-and-acted cell;"
          " >=1 tombstone-blocked re-spend with logged pre-check ===")
