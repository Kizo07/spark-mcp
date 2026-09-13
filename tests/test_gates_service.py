"""Trial governor + gates service (plan §9.2, §12.2, §14, §17.8).

Covers: pending-row-before-dispatch, idempotent retry, the Phase-1 reconciled
path (unchanged), the mandatory tombstone pre-check (block / acknowledge /
miss / review band with and without a key), the kill rule, gate status flips,
C6 messaging, budget v1, and the §17.8 concurrency test.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from spark_mcp import ledger, memory
from spark_mcp.errors import ToolError
from spark_mcp.gates import service as g
from spark_mcp.llm import FakeEmbedder, FakeLLM
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
    idea="momentum",
)


def sim(cfg):
    return SimulatedProvider(cfg.artifacts_dir())


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


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def provider(cfg):
    return sim(cfg)


@pytest.fixture
def episode(db, cfg):
    hyp = ledger.propose_hypothesis(db, cfg, **PROPOSE)
    reg = ledger.register_hypothesis(db, cfg, hyp["hypothesis_id"], {"signal": "mom"}, 7)
    return reg["episode_id"], hyp["hypothesis_id"]


class SpyProvider:
    """Counts dispatches; can assert the pending row exists before the call."""

    def __init__(self, inner, db):
        self.inner = inner
        self.db = db
        self.calls = 0
        self.pending_seen_before_dispatch = []

    @property
    def name(self):
        return getattr(self.inner, "name", "spy")

    def backtest(self, spec):
        row = self.db.conn.execute(
            "SELECT status FROM trials WHERE episode_id = ? ORDER BY created_at DESC LIMIT 1",
            (self.episode_id,),
        ).fetchone()
        self.pending_seen_before_dispatch.append(row["status"] if row else None)
        self.calls += 1
        return self.inner.backtest(spec)

    def causal(self, spec):
        return self.inner.causal(spec)

    def risk(self, spec):
        return self.inner.risk(spec)


class FailingProvider:
    name = "failing"

    def backtest(self, spec):
        raise RuntimeError("alpha_engine exploded")

    def causal(self, spec):
        raise NotImplementedError

    def risk(self, spec):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# §9.2 pending row / idempotency


def test_pending_row_exists_before_dispatch(db, cfg, episode, embedder):
    ep_id, hyp_id = episode
    spy = SpyProvider(sim(cfg), db)
    spy.episode_id = ep_id
    result = g.backtest_run(db, cfg, spy, ep_id, make_spec(hyp_id, 11), embedder=embedder)
    assert result["status"] == "testing"
    assert spy.calls == 1
    assert spy.pending_seen_before_dispatch == ["pending"]
    row = db.conn.execute(
        "SELECT * FROM trials WHERE episode_id = ?", (ep_id,)
    ).fetchone()
    assert row["status"] == "completed"
    assert row["source"] == "spark"


def test_idempotent_retry_no_second_dispatch_no_double_count(db, cfg, episode, embedder):
    ep_id, hyp_id = episode
    spy = SpyProvider(sim(cfg), db)
    spy.episode_id = ep_id
    spec = make_spec(hyp_id, 11)
    r1 = g.backtest_run(db, cfg, spy, ep_id, spec, embedder=embedder)
    r2 = g.backtest_run(db, cfg, spy, ep_id, spec, embedder=embedder)
    assert spy.calls == 1
    assert r2["reused"] is True
    assert r2["trial_id"] == r1["trial_id"]
    assert g.M_family(db, FAMILY) == 1
    n = db.conn.execute("SELECT COUNT(*) AS n FROM trials").fetchone()["n"]
    assert n == 1


def test_same_seed_different_episode_is_a_new_trial(db, cfg, episode, embedder):
    ep_id, hyp_id = episode
    hyp2 = ledger.propose_hypothesis(db, cfg, **{**PROPOSE, "idea": "momentum v2"})
    reg2 = ledger.register_hypothesis(db, cfg, hyp2["hypothesis_id"], {"signal": "mom"}, 8)
    g.backtest_run(db, cfg, sim(cfg), ep_id, make_spec(hyp_id, 11), embedder=embedder)
    g.backtest_run(
        db, cfg, sim(cfg), reg2["episode_id"],
        make_spec(hyp2["hypothesis_id"], 11), embedder=embedder,
    )
    assert g.M_family(db, FAMILY) == 2


def test_reconciled_path_unchanged_from_phase1(db, cfg, episode):
    """Out-of-band log_outcome with metrics and no trial -> reconciled (C4)."""
    ep_id, _ = episode
    out = ledger.log_outcome(db, cfg, ep_id, "success", metrics={"sr": 0.8, "T": 500})
    assert out["trial_source"] == "reconciled"
    assert out["reconciled"] == 1
    assert g.M_family(db, FAMILY) == 1
    trial = db.conn.execute(
        "SELECT * FROM trials WHERE episode_id = ?", (ep_id,)
    ).fetchone()
    assert trial["source"] == "reconciled" and trial["status"] == "completed"


def test_backtest_run_updates_existing_trial_instead_of_reconciled(db, cfg, episode, embedder):
    """backtest_run creates the trial; log_outcome closes the loop in place —
    no reconciled row, no double count."""
    ep_id, hyp_id = episode
    r = g.backtest_run(db, cfg, sim(cfg), ep_id, make_spec(hyp_id, 11), embedder=embedder)
    out = ledger.log_outcome(db, cfg, ep_id, "success", metrics={"sr": 0.8, "T": 500})
    assert out["trial_id"] == r["trial_id"]
    assert out["trial_source"] == "spark"
    assert out["reconciled"] == 0
    assert g.M_family(db, FAMILY) == 1


def test_spec_validation_refusals(db, cfg, episode, embedder):
    ep_id, hyp_id = episode
    bad = make_spec(hyp_id, 11)
    bad["costs"] = {"bps": 0, "slippage": 0, "borrow": 0}  # zero-cost default
    with pytest.raises(ToolError, match="invalid BacktestSpec"):
        g.backtest_run(db, cfg, sim(cfg), ep_id, bad, embedder=embedder)
    bad = make_spec("hyp_nope", 11)
    with pytest.raises(ToolError, match="does not match"):
        g.backtest_run(db, cfg, sim(cfg), ep_id, bad, embedder=embedder)
    assert db.conn.execute("SELECT COUNT(*) AS n FROM trials").fetchone()["n"] == 0


def test_provider_failure_fails_trial_and_leaves_episode(db, cfg, episode, embedder):
    ep_id, hyp_id = episode
    with pytest.raises(ToolError, match="backtest failed"):
        g.backtest_run(db, cfg, FailingProvider(), ep_id, make_spec(hyp_id, 11), embedder=embedder)
    trial = db.conn.execute(
        "SELECT * FROM trials WHERE episode_id = ?", (ep_id,)
    ).fetchone()
    assert trial["status"] == "failed"
    ep = ledger.get_episode(db, ep_id)
    assert ep["status"] == "registered"  # episode status unchanged (§9.2 build spec)
    assert g.M_family(db, FAMILY) == 1  # failed trials count toward M


# ---------------------------------------------------------------------------
# §12.2 tombstone pre-check


def _blocked_setup(db, embedder, cosine=0.95, basis="embedding"):
    tomb = memory.add_tombstone(
        db,
        kind="failed_hypothesis",
        summary="momentum failed OOS after costs in the toy universe",
        match_keys={"keywords": ["momentum"], "mech_tags": [FAMILY]},
        evidence_episode_ids=[],
        embedder=embedder,
    )

    def fake_candidates(db_, embedder_, text):
        return [
            {
                "tombstone_id": tomb["tombstone_id"],
                "summary": tomb["summary"],
                "match_keys": tomb["match_keys"],
                "cosine": cosine,
                "basis": basis,
            }
        ]

    return tomb, fake_candidates


def test_tombstone_block_stops_dispatch_and_logs_check(db, cfg, episode, embedder, monkeypatch):
    ep_id, hyp_id = episode
    tomb, fake = _blocked_setup(db, embedder)
    monkeypatch.setattr(g, "tombstone_candidates", fake)
    spy = SpyProvider(sim(cfg), db)
    spy.episode_id = ep_id
    with pytest.raises(ToolError, match="tombstone pre-check blocked") as exc_info:
        g.backtest_run(db, cfg, spy, ep_id, make_spec(hyp_id, 11), embedder=embedder)
    assert tomb["tombstone_id"] in str(exc_info.value)  # matched id + evidence carried
    assert spy.calls == 0
    assert db.conn.execute("SELECT COUNT(*) AS n FROM trials").fetchone()["n"] == 0
    ep = ledger.get_episode(db, ep_id)
    check = ep["tombstone_check"]
    assert check["status"] == "hit"
    assert check["matched_id"] == tomb["tombstone_id"]
    assert check["cosine"] == 0.95


def test_tombstone_acknowledge_path_proceeds_and_logs(db, cfg, episode, embedder, monkeypatch):
    ep_id, hyp_id = episode
    tomb, fake = _blocked_setup(db, embedder)
    monkeypatch.setattr(g, "tombstone_candidates", fake)
    result = g.backtest_run(
        db, cfg, sim(cfg), ep_id, make_spec(hyp_id, 11),
        embedder=embedder,
        acknowledge_tombstone={"tombstone_id": tomb["tombstone_id"], "rationale": "new cost model"},
    )
    assert result["status"] == "testing"
    ep = ledger.get_episode(db, ep_id)
    assert ep["tombstone_check"]["status"] == "ack"
    assert ep["tombstone_check"]["rationale"] == "new cost model"
    # a wrong tombstone id does NOT satisfy the acknowledgement
    ep2_hyp = ledger.propose_hypothesis(db, cfg, **{**PROPOSE, "idea": "v2"})
    reg2 = ledger.register_hypothesis(db, cfg, ep2_hyp["hypothesis_id"], {"signal": "mom"}, 3)
    with pytest.raises(ToolError, match="tombstone pre-check blocked"):
        g.backtest_run(
            db, cfg, sim(cfg), reg2["episode_id"],
            make_spec(ep2_hyp["hypothesis_id"], 12), embedder=embedder,
            acknowledge_tombstone={"tombstone_id": "tomb_wrong", "rationale": "x"},
        )


def test_tombstone_miss_proceeds(db, cfg, episode, embedder, monkeypatch):
    ep_id, hyp_id = episode
    _, fake = _blocked_setup(db, embedder, cosine=0.12)
    monkeypatch.setattr(g, "tombstone_candidates", fake)
    result = g.backtest_run(
        db, cfg, sim(cfg), ep_id, make_spec(hyp_id, 11), embedder=embedder
    )
    assert result["status"] == "testing"
    ep = ledger.get_episode(db, ep_id)
    assert ep["tombstone_check"]["status"] == "miss"
    assert ep["tombstone_check"]["cosine"] == 0.12


def test_review_band_conservative_block_without_key(db, cfg, episode, embedder, monkeypatch):
    ep_id, hyp_id = episode
    _, fake = _blocked_setup(db, embedder, cosine=0.75)
    monkeypatch.setattr(g, "tombstone_candidates", fake)
    with pytest.raises(ToolError, match="conservative block"):
        g.backtest_run(
            db, cfg, sim(cfg), ep_id, make_spec(hyp_id, 11),
            embedder=embedder, llm=None,
        )
    ep = ledger.get_episode(db, ep_id)
    assert ep["tombstone_check"]["status"] == "hit"
    assert ep["tombstone_check"]["adjudicated"] is False


def test_review_band_llm_adjudication(db, cfg, episode, embedder, monkeypatch):
    ep_id, hyp_id = episode
    _, fake = _blocked_setup(db, embedder, cosine=0.75)
    monkeypatch.setattr(g, "tombstone_candidates", fake)
    allow = FakeLLM({"Tombstone pre-check adjudication": "ALLOW"})
    result = g.backtest_run(
        db, cfg, sim(cfg), ep_id, make_spec(hyp_id, 11),
        embedder=embedder, llm=allow,
    )
    assert result["status"] == "testing"
    ep = ledger.get_episode(db, ep_id)
    assert ep["tombstone_check"]["adjudicated"] is True

    ep2_hyp = ledger.propose_hypothesis(db, cfg, **{**PROPOSE, "idea": "v3"})
    reg2 = ledger.register_hypothesis(db, cfg, ep2_hyp["hypothesis_id"], {"signal": "mom"}, 4)
    block = FakeLLM({"Tombstone pre-check adjudication": "BLOCK"})
    with pytest.raises(ToolError, match="adjudication blocked"):
        g.backtest_run(
            db, cfg, sim(cfg), reg2["episode_id"],
            make_spec(ep2_hyp["hypothesis_id"], 13), embedder=embedder, llm=block,
        )


def test_tombstone_candidates_embedding_cosine(db, embedder):
    """The real (unmocked) candidate path: identical text -> cosine 1.0,
    disjoint text -> near 0, basis reported."""
    tomb = memory.add_tombstone(
        db,
        kind="failed_hypothesis",
        summary="momentum crashes after costs in toy universe backtest",
        match_keys={"keywords": []},  # empty: stored embedding text == summary
        evidence_episode_ids=[],
        embedder=embedder,
    )
    query = "momentum crashes after costs in toy universe backtest"
    near = g.tombstone_candidates(db, embedder, query)
    assert near[0]["tombstone_id"] == tomb["tombstone_id"]
    assert near[0]["cosine"] == pytest.approx(1.0)
    assert near[0]["basis"] == "embedding"
    far = g.tombstone_candidates(db, embedder, "completely unrelated bond carry idea")
    assert far[0]["cosine"] < 0.5


def test_tombstone_candidates_lexical_fallback_without_embedder(db):
    memory.add_tombstone(
        db,
        kind="failed_hypothesis",
        summary="momentum crashes after costs",
        match_keys={"keywords": ["momentum"]},
        evidence_episode_ids=[],
    )
    out = g.tombstone_candidates(db, None, "momentum crashes after costs")
    assert out[0]["basis"] == "lexical"
    assert out[0]["cosine"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# kill rule (§8.3)


def test_kill_rule_fires_at_k_and_creates_tombstone(db, cfg, episode, embedder, monkeypatch):
    cfg.research.kill_rule_failures = 3
    monkeypatch.setattr(g, "tombstone_candidates", lambda db_, e, t: [])
    ep_id, hyp_id = episode
    failing = FailingProvider()
    for i in range(3):
        hyp = ledger.propose_hypothesis(db, cfg, **{**PROPOSE, "idea": f"kill{i}"})
        reg = ledger.register_hypothesis(db, cfg, hyp["hypothesis_id"], {"signal": "mom"}, 100 + i)
        with pytest.raises(ToolError):
            g.backtest_run(
                db, cfg, failing, reg["episode_id"],
                make_spec(hyp["hypothesis_id"], 100 + i), embedder=embedder,
            )
        state = db.conn.execute(
            "SELECT * FROM exploration_state WHERE family = ?", (FAMILY,)
        ).fetchone()
        assert state["consecutive_failures"] == i + 1
        assert state["killed"] == (1 if i == 2 else 0)
    tomb = db.conn.execute(
        "SELECT * FROM tombstones WHERE kind = 'failed_hypothesis' AND summary LIKE '%kill-rule%'"
    ).fetchone()
    assert tomb is not None
    assert FAMILY in tomb["summary"]
    assert json.loads(tomb["evidence_episode_ids"])  # triggering episode ids
    # exactly one tombstone despite the family hitting the rule once
    n = db.conn.execute(
        "SELECT COUNT(*) AS n FROM tombstones WHERE summary LIKE '%kill-rule%'"
    ).fetchone()["n"]
    assert n == 1


def test_success_resets_consecutive_failures(db, cfg, episode, embedder, monkeypatch):
    cfg.research.kill_rule_failures = 3
    monkeypatch.setattr(g, "tombstone_candidates", lambda db_, e, t: [])
    failing = FailingProvider()
    for i in range(2):
        hyp = ledger.propose_hypothesis(db, cfg, **{**PROPOSE, "idea": f"reset{i}"})
        reg = ledger.register_hypothesis(db, cfg, hyp["hypothesis_id"], {"signal": "mom"}, 200 + i)
        with pytest.raises(ToolError):
            g.backtest_run(
                db, cfg, failing, reg["episode_id"],
                make_spec(hyp["hypothesis_id"], 200 + i), embedder=embedder,
            )
    # a success on a FRESH attempt (new episode + seed, else the retry returns
    # the existing failed trial untouched) resets the counter
    hyp = ledger.propose_hypothesis(db, cfg, **{**PROPOSE, "idea": "reset-winner"})
    reg = ledger.register_hypothesis(db, cfg, hyp["hypothesis_id"], {"signal": "mom"}, 999)
    g.backtest_run(
        db, cfg, sim(cfg), reg["episode_id"],
        make_spec(hyp["hypothesis_id"], 999), embedder=embedder,
    )
    state = db.conn.execute(
        "SELECT consecutive_failures FROM exploration_state WHERE family = ?", (FAMILY,)
    ).fetchone()
    assert state["consecutive_failures"] == 0
    assert db.conn.execute(
        "SELECT COUNT(*) AS n FROM tombstones WHERE summary LIKE '%kill-rule%'"
    ).fetchone()["n"] == 0


# ---------------------------------------------------------------------------
# significance gate


def _set_metrics(db, ep_id, sr, T, skew=0.0, kurt=3.0):
    db.execute(
        "UPDATE episodes SET metrics = ? WHERE episode_id = ?",
        (json.dumps({"sr": sr, "T": T, "skew": skew, "kurt": kurt}), ep_id),
    )


def test_gate_flips_validated_on_strong_signal(db, cfg, episode, monkeypatch):
    ep_id, _ = episode
    monkeypatch.setattr(g, "tombstone_candidates", lambda db_, e, t: [])
    g.backtest_run(
        db, cfg, sim(cfg), ep_id,
        make_spec(ledger.get_episode(db, ep_id)["hypothesis_id"], 11),
        embedder=None,
    )
    _set_metrics(db, ep_id, sr=3.5, T=1200, skew=0.0, kurt=3.0)  # M=1 here
    result = g.significance_gate(db, cfg, ep_id)
    assert result["pass"] is True
    assert result["status"] == "validated"
    assert result["sr_reported_with_dsr"] is True
    assert result["placebo"] is False
    assert result["p"] < cfg.research.dsr_alpha
    assert result["dsr"] > 1 - cfg.research.dsr_alpha  # probability form
    assert result["min_T"] == cfg.research.min_T
    ep = ledger.get_episode(db, ep_id)
    assert ep["status"] == "validated"
    assert ep["corrected_metrics"]["p_value"] == result["p"]


def test_gate_inconclusive_on_deflated_signal(db, cfg, episode, monkeypatch):
    ep_id, _ = episode
    monkeypatch.setattr(g, "tombstone_candidates", lambda db_, e, t: [])
    # M=1: SR0=0, sr=0.5 with T=60 gives p ~= 1 - Phi(3.74) < 0.05 -> pass;
    # use sr below the M=... threshold: with M=1, sr=0.2, T=60 -> z=1.49 -> p~0.068.
    g.backtest_run(
        db, cfg, sim(cfg), ep_id,
        make_spec(ledger.get_episode(db, ep_id)["hypothesis_id"], 11), embedder=None,
    )
    _set_metrics(db, ep_id, sr=0.2, T=60, skew=0.0, kurt=3.0)
    result = g.significance_gate(db, cfg, ep_id)
    assert result["pass"] is False
    assert result["status"] == "inconclusive"
    assert result["p"] > cfg.research.dsr_alpha


def test_gate_nonpositive_sr_fails_regardless_of_p(db, cfg, episode, monkeypatch):
    ep_id, _ = episode
    monkeypatch.setattr(g, "tombstone_candidates", lambda db_, e, t: [])
    g.backtest_run(
        db, cfg, sim(cfg), ep_id,
        make_spec(ledger.get_episode(db, ep_id)["hypothesis_id"], 11), embedder=None,
    )
    for sr in (0.0, -2.0):
        _set_metrics(db, ep_id, sr=sr, T=5000, skew=0.0, kurt=3.0)
        result = g.significance_gate(db, cfg, ep_id)
        assert result["pass"] is False, f"sr={sr} must never pass"
        assert result["status"] == "inconclusive"


def test_gate_heavy_tail_caveat_flag(db, cfg, episode, monkeypatch):
    ep_id, _ = episode
    monkeypatch.setattr(g, "tombstone_candidates", lambda db_, e, t: [])
    g.backtest_run(
        db, cfg, sim(cfg), ep_id,
        make_spec(ledger.get_episode(db, ep_id)["hypothesis_id"], 11), embedder=None,
    )
    _set_metrics(db, ep_id, sr=4.0, T=2000, skew=3.5, kurt=12.0)
    result = g.significance_gate(db, cfg, ep_id)
    assert result["heavy_tail_caveat"] is True
    _set_metrics(db, ep_id, sr=4.0, T=2000, skew=0.5, kurt=4.0)
    result = g.significance_gate(db, cfg, ep_id)
    assert result["heavy_tail_caveat"] is False


def test_gate_refusals(db, cfg, episode, monkeypatch):
    ep_id, _ = episode
    monkeypatch.setattr(g, "tombstone_candidates", lambda db_, e, t: [])
    with pytest.raises(ToolError, match="no gateable metrics"):
        g.significance_gate(db, cfg, ep_id)
    _set_metrics(db, ep_id, sr=1.0, T=10, skew=0.0, kurt=3.0)
    with pytest.raises(ToolError, match="needs-more-data"):
        g.significance_gate(db, cfg, ep_id)
    with pytest.raises(ToolError, match="unknown episode_id"):
        g.significance_gate(db, cfg, "ep_nope")


def test_gate_placebo_flagged(db, cfg, episode, monkeypatch):
    ep_id, _ = episode
    monkeypatch.setattr(g, "tombstone_candidates", lambda db_, e, t: [])
    g.backtest_run(
        db, cfg, sim(cfg), ep_id,
        make_spec(ledger.get_episode(db, ep_id)["hypothesis_id"], 11), embedder=None,
    )
    db.execute("UPDATE episodes SET run_kind = 'placebo' WHERE episode_id = ?", (ep_id,))
    _set_metrics(db, ep_id, sr=3.5, T=1200, skew=0.0, kurt=3.0)
    result = g.significance_gate(db, cfg, ep_id)
    assert result["placebo"] is True  # noted; placebo runs are first-class trials in M


def test_gate_c6_messaging(db, cfg, monkeypatch):
    monkeypatch.setattr(g, "tombstone_candidates", lambda db_, e, t: [])
    # causal hypothesis without DAG + hypothesis without falsifier, inserted
    # directly (propose_hypothesis refuses them at the door per C6).
    db.execute(
        "INSERT INTO hypotheses (hypothesis_id, parent_hypothesis_ids, research_family,"
        " hypothesis_type, mechanism, expected_direction, horizon, scope, dag,"
        " adjustment_set, falsifier, predicted_ic, predicted_sigma, universe,"
        " as_of, data_through, status, client_id, created_at)"
        " VALUES ('hyp_causal_nodag', '[]', ?, 'causal', 'x causes y', 'positive', '1d',"
        " 'scope', NULL, NULL, 'falsifier present', 0.1, 0.1, 'toy',"
        " '2020-01-01T00:00:00Z', '2020-01-01T00:00:00Z', 'registered', NULL,"
        " '2020-01-01T00:00:00Z')",
        (FAMILY,),
    )
    db.execute(
        "INSERT INTO hypotheses (hypothesis_id, parent_hypothesis_ids, research_family,"
        " hypothesis_type, mechanism, expected_direction, horizon, scope, dag,"
        " adjustment_set, falsifier, predicted_ic, predicted_sigma, universe,"
        " as_of, data_through, status, client_id, created_at)"
        " VALUES ('hyp_nofalsifier', '[]', ?, 'predictive', 'plain signal', 'positive', '1d',"
        " 'scope', NULL, NULL, '', 0.1, 0.1, 'toy',"
        " '2020-01-01T00:00:00Z', '2020-01-01T00:00:00Z', 'registered', NULL,"
        " '2020-01-01T00:00:00Z')",
        (FAMILY,),
    )
    for hyp_id, expected_blocker in (
        ("hyp_causal_nodag", "no causal promotion without DAG (C6)"),
        ("hyp_nofalsifier", "no promotion of any kind without falsifier (C6)"),
    ):
        reg = ledger.register_hypothesis(db, cfg, hyp_id, {"signal": "s"}, 5)
        g.backtest_run(
            db, cfg, sim(cfg), reg["episode_id"],
            make_spec(hyp_id, 11), embedder=None,
        )
        _set_metrics(db, reg["episode_id"], sr=3.5, T=1200, skew=0.0, kurt=3.0)
        result = g.significance_gate(db, cfg, reg["episode_id"])
        assert expected_blocker in result["promotion_blockers"]
        db.execute("DELETE FROM trials WHERE episode_id = ?", (reg["episode_id"],))


def test_gate_failed_outcome_never_validates(db, cfg, episode, monkeypatch):
    ep_id, _ = episode
    monkeypatch.setattr(g, "tombstone_candidates", lambda db_, e, t: [])
    g.backtest_run(
        db, cfg, sim(cfg), ep_id,
        make_spec(ledger.get_episode(db, ep_id)["hypothesis_id"], 11), embedder=None,
    )
    db.execute(
        "UPDATE episodes SET outcome = 'failed' WHERE episode_id = ?", (ep_id,)
    )
    _set_metrics(db, ep_id, sr=3.5, T=1200, skew=0.0, kurt=3.0)
    result = g.significance_gate(db, cfg, ep_id)
    assert result["pass"] is True  # the statistic passes...
    assert result["status"] == "inconclusive"  # ...but the episode stays inconclusive


def test_causal_and_risk_probe_storage(db, cfg, episode, provider, monkeypatch):
    ep_id, _ = episode
    monkeypatch.setattr(g, "tombstone_candidates", lambda db_, e, t: [])
    g.backtest_run(
        db, cfg, provider, ep_id,
        make_spec(ledger.get_episode(db, ep_id)["hypothesis_id"], 11), embedder=None,
    )
    causal = g.causal_check(
        db, cfg, provider, ep_id,
        {
            "episode_id": ep_id,
            "targets": ["TOYA", "TOYB"],
            "conditioning": ["mkt"],
            "method": "granger",
            "lags": {"min_lag": 1, "max_lag": 5},
        },
    )
    assert causal["simulated"] is True
    ep = ledger.get_episode(db, ep_id)
    assert ep["metrics"]["causal"]["method"] == "granger"
    risk = g.risk_decompose(
        db, cfg, provider, ep_id, {"book_id": "book1", "factor_model": "ff5+momentum"}
    )
    assert risk["simulated"] is True
    ep = ledger.get_episode(db, ep_id)
    assert ep["risk_exposures"]["factor_model"] == "ff5+momentum"
    # metrics.causal must merge, not clobber, the backtest metrics
    assert ep["metrics"]["sr"] is not None


def test_causal_risk_spec_validation_and_unsupported(db, cfg, episode, monkeypatch):
    ep_id, _ = episode
    monkeypatch.setattr(g, "tombstone_candidates", lambda db_, e, t: [])
    g.backtest_run(
        db, cfg, sim(cfg), ep_id,
        make_spec(ledger.get_episode(db, ep_id)["hypothesis_id"], 11), embedder=None,
    )
    with pytest.raises(ToolError, match="invalid CausalSpec"):
        g.causal_check(db, cfg, sim(cfg), ep_id, {"episode_id": ep_id})
    with pytest.raises(ToolError, match="invalid BookSpec"):
        g.risk_decompose(
            db, cfg, sim(cfg), ep_id,
            {"book_id": "b1", "returns_uri": "artifacts://r.json"},  # two sources
        )
    with pytest.raises(ToolError, match="does not support causal"):
        g.causal_check(
            db, cfg, FailingProvider(), ep_id,
            {
                "episode_id": ep_id,
                "targets": ["TOYA"],
                "conditioning": [],
                "method": "granger",
                "lags": {"min_lag": 1, "max_lag": 3},
            },
        )
    with pytest.raises(ToolError, match="does not support risk"):
        g.risk_decompose(db, cfg, FailingProvider(), ep_id, {"book_id": "b1"})


# ---------------------------------------------------------------------------
# budget v1 (§14)


def test_spend_ledger_appends_on_backtest_and_log_outcome(db, cfg, episode, monkeypatch):
    ep_id, _ = episode
    monkeypatch.setattr(g, "tombstone_candidates", lambda db_, e, t: [])
    g.backtest_run(
        db, cfg, sim(cfg), ep_id,
        make_spec(ledger.get_episode(db, ep_id)["hypothesis_id"], 11), embedder=None,
    )
    ledger.log_outcome(
        db, cfg, ep_id, "success",
        metrics={"sr": 1.0, "T": 500}, costs={"usd": 0.25, "tokens": 100},
    )
    rows = db.conn.execute(
        "SELECT source, usd, tokens, estimated FROM spend_ledger ORDER BY seq"
    ).fetchall()
    assert [r["source"] for r in rows] == ["backtest_run", "log_outcome"]
    assert rows[1]["usd"] == 0.25 and rows[1]["tokens"] == 100
    assert rows[1]["estimated"] == 0
    assert rows[0]["estimated"] == 1  # backtest spend is a rate-card estimate


def test_rate_card_estimate_for_unknown_costs(db, cfg, episode):
    cfg.budgets.rates = {"your-model-id": {"input_per_1k": 1.0, "output_per_1k": 1.0}}
    ep_id, _ = episode
    ledger.log_outcome(
        db, cfg, ep_id, "success", metrics={"sr": 1.0, "T": 500}, costs={"tokens": 500}
    )
    row = db.conn.execute(
        "SELECT usd, estimated FROM spend_ledger WHERE source='log_outcome'"
    ).fetchone()
    assert row["usd"] == pytest.approx(1.0)  # 500 tokens * (1+1)/1k
    assert row["estimated"] == 1


def test_budget_warn_policy_logs_trajectory_event(db, cfg, episode, monkeypatch):
    from spark_mcp import budgets

    cfg.budgets.daily_usd = 0.5
    ep_id, _ = episode
    budgets.record_spend(db, cfg, family=FAMILY, source="manual", usd=1.0)  # already over
    monkeypatch.setattr(g, "tombstone_candidates", lambda db_, e, t: [])
    result = g.backtest_run(
        db, cfg, sim(cfg), ep_id,
        make_spec(ledger.get_episode(db, ep_id)["hypothesis_id"], 11), embedder=None,
    )
    assert result["status"] == "testing"  # warn proceeds
    event = db.conn.execute(
        "SELECT payload FROM trajectory_events WHERE kind = 'observation' ORDER BY seq DESC LIMIT 1"
    ).fetchone()
    assert json.loads(event["payload"])["kind"] == "budget_warning"


def test_budget_fail_closed_blocks_before_dispatch(db, cfg, episode, monkeypatch):
    from spark_mcp import budgets

    cfg.budgets.daily_usd = 0.5
    cfg.budgets.policy = "fail_closed"
    budgets.record_spend(db, cfg, family=FAMILY, source="manual", usd=1.0)  # already over
    ep_id, hyp_id = episode
    monkeypatch.setattr(g, "tombstone_candidates", lambda db_, e, t: [])
    spy = SpyProvider(sim(cfg), db)
    spy.episode_id = ep_id
    with pytest.raises(ToolError, match="daily budget exceeded"):
        g.backtest_run(db, cfg, spy, ep_id, make_spec(hyp_id, 11), embedder=None)
    assert spy.calls == 0  # no dispatch
    assert db.conn.execute("SELECT COUNT(*) AS n FROM trials").fetchone()["n"] == 0  # no trial row
    # the blocked run's own spend row rolled back with its transaction
    rows = db.conn.execute("SELECT source FROM spend_ledger").fetchall()
    assert [r["source"] for r in rows] == ["manual"]


def test_spend_summary_shape(db, cfg, episode):
    ep_id, _ = episode
    ledger.log_outcome(
        db, cfg, ep_id, "success",
        metrics={"sr": 1.0, "T": 500}, costs={"usd": 1.5, "tokens": 10},
    )
    from spark_mcp.budgets import spend_summary

    summary = spend_summary(db)
    assert summary == {"today_usd": 1.5, "total_usd": 1.5}


# ---------------------------------------------------------------------------
# §17.8 concurrency: N racing writers -> one trial, one dispatch


def test_concurrent_backtest_run_single_trial_single_dispatch(db, cfg, episode, monkeypatch):
    ep_id, hyp_id = episode
    monkeypatch.setattr(g, "tombstone_candidates", lambda db_, e, t: [])
    spy = SpyProvider(sim(cfg), db)
    spy.episode_id = ep_id
    spec = make_spec(hyp_id, 42)

    def call(_):
        return g.backtest_run(db, cfg, spy, ep_id, spec, embedder=None)

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(call, range(8)))

    assert spy.calls == 1
    trials = db.conn.execute("SELECT * FROM trials WHERE episode_id = ?", (ep_id,)).fetchall()
    assert len(trials) == 1
    assert trials[0]["status"] == "completed"
    winner_trial = results[0]["trial_id"]
    for r in results:
        assert r["trial_id"] == winner_trial
        assert r["metrics"] == results[0]["metrics"]  # everyone gets the winner's result
    assert g.M_family(db, FAMILY) == 1


def test_concurrent_failure_log_outcomes_create_one_tombstone(db, cfg, episode):
    """Phase 1 review nit, closed: 8 racing failure log_outcomes -> 1 tombstone."""
    cfg.research.kill_rule_failures = 100  # isolate the dedupe from the kill rule
    ep_id, _ = episode

    def call(i):
        return ledger.log_outcome(
            db, cfg, ep_id, "failure", failure_reason=f"sharpe collapsed {i}"
        )

    with ThreadPoolExecutor(8) as pool:
        outcomes = list(pool.map(call, range(8)))
    assert len({o["tombstone_id"] for o in outcomes}) == 1
    n = db.conn.execute("SELECT COUNT(*) AS n FROM tombstones").fetchone()["n"]
    assert n == 1
