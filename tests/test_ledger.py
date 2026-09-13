"""Ledger service tests: propose/register/log round-trip, C6 refusal,
client_id idempotency, protocol_hash, C4 reconciliation, cost defaults."""

import hashlib
import json

import pytest

import spark_mcp
from spark_mcp import ledger
from spark_mcp.config import Config
from spark_mcp.storage import open_db


@pytest.fixture
def cfg():
    return Config()


@pytest.fixture
def db(tmp_path):
    handle = open_db(tmp_path / "spark.db")
    yield handle
    handle.close()


def _propose(db, cfg, **overrides):
    fields = dict(
        research_family="momentum",
        hypothesis_type="predictive",
        mechanism="intermediate-horizon momentum reversal driven by forced rebalancing",
        expected_direction="negative",
        horizon="20d",
        scope="cross_section",
        falsifier="reversal disappears net of transaction costs",
        predicted_ic=0.03,
        predicted_sigma=0.01,
        universe="us_equity",
    )
    fields.update(overrides)
    return ledger.propose_hypothesis(db, cfg, **fields)


def test_propose_register_log_round_trip(db, cfg):
    hyp = _propose(db, cfg)
    assert hyp["hypothesis_id"].startswith("hyp_")
    # Phase 2: empty archive ⇒ maximally novel (N = w·(1,1,1) = 1.0); S stays
    # None until a realized_ic exists.
    assert hyp["N"] == pytest.approx(1.0)
    assert hyp["S"] is None

    reg = ledger.register_hypothesis(db, cfg, hyp["hypothesis_id"], {"signal": "mom"}, seed=42)
    assert reg["episode_id"].startswith("ep_")
    assert reg["status"] == "registered"

    hyp_row = ledger.get_hypothesis(db, hyp["hypothesis_id"])
    assert hyp_row["status"] == "registered"
    assert hyp_row["research_family"] == "momentum"

    out = ledger.log_outcome(
        db, cfg, reg["episode_id"], "success", metrics={"net_sharpe": 1.2}, reward=0.5
    )
    assert out["status"] == "validated"
    ep = ledger.get_episode(db, reg["episode_id"])
    assert ep["status"] == "validated"
    assert ep["metrics"] == {"net_sharpe": 1.2}
    assert ep["reward"] == 0.5


def test_propose_idempotent_on_client_id(db, cfg):
    r1 = _propose(db, cfg, client_id="prop-1")
    r2 = _propose(db, cfg, client_id="prop-1")
    assert r1["hypothesis_id"] == r2["hypothesis_id"]
    n = db.conn.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0]
    assert n == 1


def test_register_idempotent_on_client_id(db, cfg):
    hyp = _propose(db, cfg)
    r1 = ledger.register_hypothesis(db, cfg, hyp["hypothesis_id"], {"a": 1}, 7, "reg-1")
    r2 = ledger.register_hypothesis(db, cfg, hyp["hypothesis_id"], {"a": 1}, 7, "reg-1")
    assert r1["episode_id"] == r2["episode_id"]
    n = db.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    assert n == 1


def test_causal_requires_dag_and_adjustment_set_iff_causal(db, cfg):
    # non-causal without dag: fine
    _propose(db, cfg)
    # causal without dag: refused (C6)
    with pytest.raises(ValueError, match="C6"):
        _propose(db, cfg, hypothesis_type="causal", dag=None, adjustment_set=None)
    # causal with dag but no adjustment_set: refused
    with pytest.raises(ValueError, match="adjustment_set"):
        _propose(db, cfg, hypothesis_type="causal", dag={"nodes": ["x"]})
    # causal with both: accepted
    ok = _propose(
        db,
        cfg,
        hypothesis_type="causal",
        dag={"nodes": ["a", "b"], "edges": [["a", "b"]]},
        adjustment_set=["c"],
    )
    assert ok["hypothesis_id"]
    row = ledger.get_hypothesis(db, ok["hypothesis_id"])
    assert row["dag"] == {"nodes": ["a", "b"], "edges": [["a", "b"]]}
    assert row["adjustment_set"] == ["c"]
    # non-causal must not require dag: proposing with dag is still legal
    _propose(db, cfg, hypothesis_type="predictive", dag={"nodes": []})


def test_c6_requires_core_fields_for_all_types(db, cfg):
    with pytest.raises(ValueError, match="falsifier"):
        _propose(db, cfg, falsifier="")
    with pytest.raises(ValueError, match="predicted_sigma"):
        _propose(db, cfg, predicted_sigma=None)


def test_protocol_hash_is_stable_and_sensitive(db, cfg):
    hyp = _propose(db, cfg)
    reg1 = ledger.register_hypothesis(db, cfg, hyp["hypothesis_id"], {"signal": "mom"}, 42)
    reg2 = ledger.register_hypothesis(db, cfg, hyp["hypothesis_id"], {"signal": "mom"}, 42)
    assert reg1["protocol_hash"] == reg2["protocol_hash"]

    reg3 = ledger.register_hypothesis(db, cfg, hyp["hypothesis_id"], {"signal": "rev"}, 42)
    assert reg3["protocol_hash"] != reg1["protocol_hash"]

    canonical = json.dumps(
        {
            "hypothesis_id": hyp["hypothesis_id"],
            "spec": {"signal": "mom"},
            "seed": 42,
            "code_version": spark_mcp.__version__,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    assert reg1["protocol_hash"] == hashlib.sha256(canonical).hexdigest()


def test_reconciliation_inserts_reconciled_trial(db, cfg):
    hyp = _propose(db, cfg)
    reg = ledger.register_hypothesis(db, cfg, hyp["hypothesis_id"], {"a": 1}, 3)
    # no trial row exists for this episode (backtest_run arrives in Phase 3)
    out = ledger.log_outcome(db, cfg, reg["episode_id"], "success", metrics={"ic": 0.02})
    assert out["trial_source"] == "reconciled"
    ep = ledger.get_episode(db, reg["episode_id"])
    assert ep["reconciled"] == 1
    trials = db.conn.execute(
        "SELECT source, status, family FROM trials WHERE episode_id = ?",
        (reg["episode_id"],),
    ).fetchall()
    assert len(trials) == 1
    assert trials[0]["source"] == "reconciled"
    assert trials[0]["status"] == "completed"
    assert trials[0]["family"] == "momentum"


def test_log_outcome_idempotent_on_client_id(db, cfg):
    hyp = _propose(db, cfg)
    reg = ledger.register_hypothesis(db, cfg, hyp["hypothesis_id"], {"a": 1}, 3)
    r1 = ledger.log_outcome(
        db, cfg, reg["episode_id"], "success", metrics={"ic": 0.02}, client_id="log-1"
    )
    r2 = ledger.log_outcome(
        db, cfg, reg["episode_id"], "success", metrics={"ic": 0.02}, client_id="log-1"
    )
    assert r1 == r2
    n = db.conn.execute(
        "SELECT COUNT(*) FROM trials WHERE episode_id = ?", (reg["episode_id"],)
    ).fetchone()[0]
    assert n == 1  # retried outcome never double-counts M


def test_log_outcome_marks_supplied_costs_as_actual(db, cfg):
    hyp = _propose(db, cfg)
    reg = ledger.register_hypothesis(db, cfg, hyp["hypothesis_id"], {"a": 1}, 3)
    out = ledger.log_outcome(
        db,
        cfg,
        reg["episode_id"],
        "success",
        metrics={"ic": 0.01},
        costs={"usd": 1.25, "tokens": 900},
    )
    assert out["costs"] == {"usd": 1.25, "tokens": 900, "estimated": False}
    ep = ledger.get_episode(db, reg["episode_id"])
    assert ep["costs"]["usd"] == 1.25


def test_costs_default_to_zero_and_estimated(db, cfg):
    hyp = _propose(db, cfg)
    reg = ledger.register_hypothesis(db, cfg, hyp["hypothesis_id"], {"a": 1}, 3)
    out = ledger.log_outcome(db, cfg, reg["episode_id"], "inconclusive")
    assert out["costs"] == {"usd": 0.0, "tokens": 0, "estimated": True}
    assert out["reward"] == 0.0  # reward non-null from Phase 1 (M13)
    ep = ledger.get_episode(db, reg["episode_id"])
    assert ep["status"] == "inconclusive"
    assert ep["costs"]["estimated"] is True
    # metrics None -> no trial row, no reconciliation flag
    assert ep["reconciled"] == 0
    assert out["trial_id"] is None


def test_failure_outcome_writes_tombstone(db, cfg):
    hyp = _propose(db, cfg)
    reg = ledger.register_hypothesis(db, cfg, hyp["hypothesis_id"], {"a": 1}, 3)
    out = ledger.log_outcome(
        db, cfg, reg["episode_id"], "failure", failure_reason="sharpe collapsed net of costs"
    )
    assert out["status"] == "failed"
    assert out["tombstone_id"] is not None
    tomb = db.conn.execute(
        "SELECT * FROM tombstones WHERE tombstone_id = ?", (out["tombstone_id"],)
    ).fetchone()
    assert tomb["kind"] == "failed_hypothesis"
    assert "sharpe collapsed" in tomb["summary"]
    assert json.loads(tomb["evidence_episode_ids"]) == [reg["episode_id"]]


def test_trajectory_records_tool_calls(db, cfg):
    hyp = _propose(db, cfg)
    reg = ledger.register_hypothesis(db, cfg, hyp["hypothesis_id"], {"a": 1}, 3)
    ledger.log_outcome(db, cfg, reg["episode_id"], "success", metrics={"ic": 0.02})
    rows = db.conn.execute(
        "SELECT kind, run_id FROM trajectory_events ORDER BY seq"
    ).fetchall()
    assert [r["kind"] for r in rows] == ["tool_call"] * len(rows)
    assert {hyp["hypothesis_id"], reg["episode_id"]} <= {r["run_id"] for r in rows}


def _concurrent(fn, n=8):
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(n) as pool:
        return list(pool.map(fn, range(n)))


def test_racing_proposers_resolve_to_one_winner(tmp_path, cfg):
    """§9.2: racing writers on the same client_id resolve to one winner,
    the losers read the winner's row."""
    path = tmp_path / "race.db"
    open_db(path).close()  # create + migrate

    def go(_):
        handle = open_db(path)
        try:
            return ledger.propose_hypothesis(
                handle, cfg, research_family="f", hypothesis_type="predictive",
                mechanism="m", expected_direction="pos", horizon="20d", scope="s",
                falsifier="fals", predicted_ic=0.1, predicted_sigma=0.1,
                universe="u", client_id="race-1",
            )
        finally:
            handle.close()

    results = _concurrent(go)
    assert len({r["hypothesis_id"] for r in results}) == 1
    db = open_db(path)
    assert db.conn.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0] == 1
    db.close()


def test_racing_log_outcomes_insert_exactly_one_reconciled_trial(tmp_path, cfg):
    """Trial accounting under concurrency: N parallel log_outcome calls with
    metrics and no prior trial -> exactly one reconciled trial row."""
    path = tmp_path / "race.db"
    db = open_db(path)
    hyp = _propose(db, cfg)
    reg = ledger.register_hypothesis(db, cfg, hyp["hypothesis_id"], {"a": 1}, 3)
    db.close()

    def go(_):
        handle = open_db(path)
        try:
            return ledger.log_outcome(
                handle, cfg, reg["episode_id"], "success", metrics={"ic": 0.02}
            )
        finally:
            handle.close()

    results = _concurrent(go)
    trial_ids = {r["trial_id"] for r in results}
    assert len(trial_ids) == 1
    db = open_db(path)
    rows = db.conn.execute(
        "SELECT trial_id, source, status FROM trials WHERE episode_id = ?",
        (reg["episode_id"],),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["trial_id"] == trial_ids.pop()
    assert rows[0]["source"] == "reconciled" and rows[0]["status"] == "completed"
    ep = ledger.get_episode(db, reg["episode_id"])
    assert ep["reconciled"] == 1
    db.close()
