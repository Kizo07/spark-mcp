"""G5 — the Phase 7 program gate (plan §18.0): creativity + consolidation.

Scripted scenario on the seeded corpus with the deterministic SimulatedProvider:
  1. >=1 analogy-derived hypothesis through the gate: find_analogies on a
     target problem -> a mapping's falsifiable translation becomes the
     hypothesis mechanism -> propose -> register -> backtest_run -> forced
     positive metrics (G2/G3 pattern) -> significance_gate -> validated;
  2. suggest_next consumes a consolidation gap report BY ID (the report id is
     recorded and the report's gap families get the utility boost);
  3. >=1 curriculum exercise generated from a REAL gap of that report and
     replay-scored against the current champion (deterministic scorer).

Gate evidence is printed from the test (G5 gradeable artifacts).
"""

from __future__ import annotations

import json

from spark_mcp import consolidate as consolidate_mod
from spark_mcp import curriculum, ledger
from spark_mcp.analogy import find_analogies
from spark_mcp.evolve.gepa import fixture_score
from spark_mcp.explore import suggest_next
from spark_mcp.gates import service as g
from spark_mcp.providers.simulated import SimulatedProvider
from spark_mcp.skills import load_fixtures

PROPOSE = dict(
    expected_direction="positive",
    scope="cross_section",
    falsifier="IC <= 0 net of costs",
    predicted_ic=0.03,
    predicted_sigma=0.01,
    universe="toy",
)

# Strong-signal metrics pin (same as the G3/gates-service positive pins:
# sr=3.5, T=1200 passes DSR at M=1 with min_T = cfg.research.min_T).
GATE_METRICS = {"sr": 3.5, "T": 1200, "skew": 0.0, "kurt": 3.0}

OCCUPY = {  # every QD bin family except signal_processing (its fixture is held out)
    "quant": "20d",
    "econometrics": "60d",
    "causal": "120d",
    "network_optimization": "5d",
    "nlp": "1d",
    "macro": "long",
    "portfolio": "20d",
}

CHAMPION_STEPS = [
    "Select a spectral estimation method suited to uneven sampling.",
    "Estimate significance of peaks against a null spectrum via FFT windows.",
    "Check stability of the dominant frequencies across windows.",
]


def make_spec(hypothesis_id: str, seed: int) -> dict:
    return {
        "hypothesis_id": hypothesis_id,
        "signal_ref": "g5.json",
        "params": {"type": "momentum", "lookback": 10, "turnover": 0.1},
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


def _occupy_family(db, cfg, provider, family, horizon, seed):
    hyp = ledger.propose_hypothesis(
        db, cfg, research_family=family, hypothesis_type="predictive",
        mechanism=f"g5 occupy {family}", idea=f"g5 {family}", horizon=horizon, **PROPOSE,
        client_id=f"g5-occupy-{family}",
    )
    reg = ledger.register_hypothesis(db, cfg, hyp["hypothesis_id"], {"g5": family}, seed)
    g.backtest_run(db, cfg, provider, reg["episode_id"], make_spec(hyp["hypothesis_id"], seed),
                   embedder=None)
    return ledger.log_outcome(
        db, cfg, reg["episode_id"], "success",
        metrics={"oos_ic": 0.04, "robustness": 0.6, "orthogonality": 0.5, "novelty": 0.6},
        costs={"usd": 0.05, "tokens": 25},
    )


def test_g5_creativity_and_consolidation(seeded, cfg, embedder):
    db = seeded
    provider = SimulatedProvider(cfg.artifacts_dir())

    # -- (1) analogy-derived hypothesis through the gate -----------------------
    analogies = find_analogies(
        db, cfg,
        "regime-aware sizing of trend signals before volatility shifts",
        ["regime_switching", "regime_aware", "volatility"], k=3,
    )
    mapping = analogies["mappings"][0]
    translation = mapping["falsifiable_translation"]
    assert translation.startswith("If this analogy holds, then")
    hyp = ledger.propose_hypothesis(
        db, cfg,
        research_family="quant",
        hypothesis_type="predictive",
        mechanism=translation,  # the analogy's falsifiable translation IS the mechanism
        idea="g5 analogy-derived hypothesis",
        horizon="20d",
        **PROPOSE,
        client_id="g5-analogy-hypothesis",
    )
    reg = ledger.register_hypothesis(db, cfg, hyp["hypothesis_id"], {"g5": "analogy"}, 700)
    bt = g.backtest_run(
        db, cfg, provider, reg["episode_id"],
        make_spec(hyp["hypothesis_id"], 700), embedder=None,
    )
    assert bt["reused"] is False and bt["M_family"] >= 1
    out = ledger.log_outcome(
        db, cfg, reg["episode_id"], "success", metrics=dict(GATE_METRICS),
        costs={"usd": 0.05, "tokens": 25},
    )
    assert out["status"] == "validated"
    verdict = g.significance_gate(db, cfg, reg["episode_id"])
    assert verdict["pass"] is True
    assert verdict["status"] == "validated"
    assert verdict["sr_reported_with_dsr"] is True
    ep = ledger.get_episode(db, reg["episode_id"])
    assert ep["status"] == "validated"

    # -- occupy every family except signal_processing so the consolidation
    #    report's high-severity gaps target exactly the fixture-mapped family
    for family, horizon in OCCUPY.items():
        _occupy_family(db, cfg, provider, family, horizon, 800)

    # -- (2) suggest_next consumes the consolidation gap report by id ----------
    report = consolidate_mod.consolidate(db, cfg)
    gap_families = {g_["family"] for g_ in report["gaps"] if g_.get("family")}
    assert gap_families, "expected family-carrying gaps"
    base = suggest_next(db, cfg, n=8, seed=2027)
    consumed = suggest_next(db, cfg, n=8, seed=2027, gap_report_id=report["report_id"])
    assert consumed["gap_report_id"] == report["report_id"]
    assert set(consumed["gap_boosted_families"]) == gap_families
    boost_proof = {}
    for fam in gap_families:
        p0 = base["family_selection_probabilities"][fam]
        p1 = consumed["family_selection_probabilities"][fam]
        assert p1 > p0, f"gap family {fam} was not boosted"
        boost_proof[fam] = (p0, p1)
    traj = db.conn.execute(
        "SELECT payload FROM trajectory_events WHERE run_id = 'suggest_next'"
        " ORDER BY seq DESC LIMIT 1"
    ).fetchone()
    assert json.loads(traj["payload"])["gap_report_id"] == report["report_id"]

    # -- (3) curriculum exercise from a real gap, replay-scored ----------------
    db.execute(
        "INSERT INTO skills (skill_id, name, version, body_uri, lineage, status,"
        " created_at) VALUES ('skill_g5', 'spectral_audit', 1,"
        " 'spark://skills/spectral_audit/1', ?, 'champion',"
        " '2026-01-01T00:00:00Z')",
        (json.dumps({"procedure_steps": CHAMPION_STEPS}),),
    )
    exercise = curriculum.generate_exercise(db, cfg, gap_report_id=report["report_id"])
    assert exercise.get("deferred") is not True, exercise.get("reason")
    assert exercise["gap_report_id"] == report["report_id"]
    assert exercise["source_gap_id"] in {g_["gap_id"] for g_ in report["gaps"]}
    assert exercise["evaluation"]["fixture_task_id"] == "replay_spectral_micro"
    fixture = next(f for f in load_fixtures() if f["task_id"] == "replay_spectral_micro")
    assert exercise["champion_score"] == fixture_score(CHAMPION_STEPS, fixture)
    assert 0.0 < exercise["champion_score"] <= 1.0

    # -- gate evidence summary ---------------------------------------------------
    print("\n=== G5 EVIDENCE (Phase 7 gate, plan §18.0) ===")
    print(f"(i) analogy: base={mapping['label']!r} systematicity"
          f" {mapping['systematicity']:.3f} (threshold"
          f" {cfg.research.analogy_systematicity_threshold}); translation"
          f" {translation!r}")
    print(f"    hypothesis {hyp['hypothesis_id']} -> episode {reg['episode_id']}:"
          f" backtest trial {bt['trial_id']} (M_family={bt['M_family']}),"
          f" DSR gate pass={verdict['pass']} p={verdict['p']:.2e}"
          f" -> status {ep['status']}")
    print(f"(ii) consolidation report {report['report_id']}:"
          f" {report['gap_count']} gaps, families {sorted(gap_families)}")
    for fam, (p0, p1) in sorted(boost_proof.items()):
        print(f"    suggest_next boost {fam}: selection probability"
              f" {p0:.4f} -> {p1:.4f} (x{cfg.research.gap_boost_factor} utility boost;"
              f" report id recorded on trajectory)")
    print(f"(iii) exercise {exercise['exercise_id']} from gap"
          f" {exercise['source_gap_id']} (family {exercise['family']}) -> fixture"
          f" {exercise['evaluation']['fixture_task_id']}; champion"
          f" {exercise['champion']['name']}:{exercise['champion']['version']}"
          f" replay score {exercise['champion_score']:.3f}")
    print("=== G5 PASS: analogy-derived hypothesis validated through the gate;"
          " gap report consumed by id with boosted families; curriculum"
          " exercise from a real gap replay-scored ===")
