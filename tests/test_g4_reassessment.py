"""G4 reassessment artifact (plan §18.0: "else reassess (exploration §6 kill
rule)"). G4 evaluated RED on the G1 judge half; this scenario is the §6
reassessment evidence: the kill-rule machinery is healthy — a family driven
to consecutive genuine failures is auto-tombstoned, its posterior collapses,
and suggest_next redirects exploration away from it. The remaining G4
blocker is NOT exploration hygiene; it is the G1 judge half's corpus-depth
ceiling (diagnosis in STATUS.md).
"""
import json

from spark_mcp import archive, explore, ledger
from spark_mcp.config import Config
from spark_mcp.storage import open_db


def _run_episode(db, cfg, family: str, i: int, outcome: str) -> None:
    prop = ledger.propose_hypothesis(
        db, cfg,
        research_family=family,
        hypothesis_type="predictive",
        mechanism=f"reassessment probe {family} #{i}",
        expected_direction="positive",
        horizon="20d", scope="cross_section",
        falsifier="IC <= 0",
        predicted_ic=0.02, predicted_sigma=0.01,
        universe="toy",
    )
    reg = ledger.register_hypothesis(db, cfg, prop["hypothesis_id"], {"i": i}, i)
    ledger.log_outcome(
        db, cfg, reg["episode_id"], outcome,
        metrics={"realized_ic": -0.02 if outcome == "failure" else 0.02},
        costs={"usd": 0.01, "tokens": 5},
        reward=0.0 if outcome == "failure" else 1.0,
    )


def test_g4_reassessment_kill_rule_posterior_and_redirect(tmp_path):
    cfg = Config(storage={
        "db_path": f"{tmp_path}/spark.db", "artifacts_dir": f"{tmp_path}/a"
    })
    db = open_db(cfg.db_path())
    K = cfg.research.kill_rule_failures

    # healthy family: mix of outcomes -> live posterior
    for i, o in enumerate(["success", "failure", "success", "success"]):
        _run_episode(db, cfg, "quant", i, o)

    # sick family: K consecutive genuine failures
    for i in range(K):
        _run_episode(db, cfg, "network_optimization", i, "failure")

    # one more failure AFTER the kill to prove the counter still ticks
    _run_episode(db, cfg, "network_optimization", K, "failure")

    rows = {r["family"]: r for r in db.conn.execute(
        "SELECT family, consecutive_failures, killed, posterior"
        " FROM exploration_state"
    ).fetchall()}
    sick = rows["network_optimization"]
    assert int(sick["consecutive_failures"]) >= K
    assert int(sick["killed"]) == 1  # auto-tombstoned by the kill rule

    # posterior collapsed toward 0
    post = json.loads(sick["posterior"])
    mean = post["alpha"] / (post["alpha"] + post["beta"])
    assert mean < 0.25

    # a tombstone row was written for the family
    tomb = db.conn.execute(
        "SELECT COUNT(*) AS n FROM tombstones"
        " WHERE kind = 'failed_hypothesis'"
        " AND summary LIKE '%network_optimization%'"
    ).fetchone()["n"]
    assert tomb >= 1

    # suggest_next still runs and prefers the healthy family over the killed
    # one at equal emptiness (utility = emptiness x Thompson draw; the killed
    # family's collapsed posterior dominates)
    out = explore.suggest_next(db, cfg, n=2, seed=7)
    fams = [s["family"] for s in out["suggestions"]]
    assert "network_optimization" not in fams

    # the healthy family's posterior is live (non-degenerate)
    healthy = json.loads(rows["quant"]["posterior"])
    h_mean = healthy["alpha"] / (healthy["alpha"] + healthy["beta"])
    assert 0.3 < h_mean < 0.9

    # archive_status reflects the world coherently
    status = archive.status(db, cfg)
    assert status["family_posteriors"]["network_optimization"]["mean"] < 0.25
    assert status["family_M"]["quant"] >= 4
