"""Memory service tests: PIT leakage (blocking per AGENTS.md rule 3), recall
ordering by effective confidence, belief promotion thresholds, lexical
contradiction, and tombstone retrieval before re-spend."""

import pytest

from spark_mcp import memory
from spark_mcp.config import Config
from spark_mcp.storage import open_db
from spark_mcp.timeutil import now_iso

FUTURE = "2099-01-01T00:00:00Z"


@pytest.fixture
def cfg():
    return Config()


@pytest.fixture
def db(tmp_path):
    handle = open_db(tmp_path / "spark.db")
    yield handle
    handle.close()


def _promote(db, cfg, claim, confidence=0.9, as_of=None, episodes=None, **kw):
    return memory.promote_belief(
        db,
        cfg,
        episode_ids=episodes or ["ep_test1"],
        claim=claim,
        scope="cross_section",
        rationale="test",
        confidence=confidence,
        as_of=as_of,
        **kw,
    )


def test_future_dated_belief_is_invisible_at_earlier_as_of(db, cfg):
    """The leakage test: a belief valid_from/ as_of in the future must never
    surface in recall at an earlier as_of (AGENTS.md rule 3)."""
    _promote(db, cfg, "momentum persists in us equities", as_of="2020-01-01T00:00:00Z")
    _promote(db, cfg, "future regime breaks momentum", as_of=FUTURE)

    now = now_iso()
    claims = [b["claim"] for b in memory.recall(db, "momentum", now, k=10)]
    assert "momentum persists in us equities" in claims
    assert "future regime breaks momentum" not in claims

    past = "2021-01-01T00:00:00Z"
    claims_past = [b["claim"] for b in memory.recall(db, "momentum", past, k=10)]
    assert claims_past == ["momentum persists in us equities"]


def test_invalidated_belief_invisible_after_valid_to(db, cfg):
    _promote(db, cfg, "carry works in fx", as_of="2020-06-01T00:00:00Z")
    bid = db.conn.execute("SELECT belief_id FROM beliefs").fetchone()[0]
    db.execute("UPDATE beliefs SET valid_to = '2021-01-01T00:00:00Z' WHERE belief_id = ?", (bid,))
    at = "2022-01-01T00:00:00Z"
    assert memory.recall(db, "carry", at, k=10) == []
    assert memory.recall(db, "carry", "2020-07-01T00:00:00Z", k=10) != []


def test_recall_orders_by_effective_confidence(db, cfg):
    _promote(db, cfg, "low volatility anomaly stable", confidence=0.5)
    high = _promote(db, cfg, "quality minus junk premium", confidence=0.9)
    # simulate decay written by consolidate (Phase 7): 0.9 * 0.4 = 0.36 < 0.5
    db.execute("UPDATE beliefs SET decay = 0.4 WHERE belief_id = ?", (high["belief_id"],))

    hits = memory.recall(db, "", now_iso(), k=10)
    assert [h["belief_id"] for h in hits] == [h["belief_id"] for h in sorted(
        hits, key=lambda h: -h["effective_confidence"]
    )]
    assert hits[0]["claim"] == "low volatility anomaly stable"
    assert hits[0]["effective_confidence"] == pytest.approx(0.5)


def test_recall_returns_episode_pointers(db, cfg):
    r = _promote(db, cfg, "earnings drift is tradeable", episodes=["ep_a", "ep_b"])
    hits = memory.recall(db, "earnings", now_iso(), k=5)
    assert hits[0]["belief_id"] == r["belief_id"]
    assert hits[0]["supporting_runs"] == ["ep_a", "ep_b"]
    assert hits[0]["contradicting_runs"] == []


def test_promote_auto_at_threshold_and_pending_below(db, cfg):
    threshold = cfg.research.belief_auto_promote_threshold  # 0.8
    at = _promote(db, cfg, "term structure carry predicts bond returns", confidence=threshold)
    assert at["promotion_status"] == "auto"
    below = _promote(
        db, cfg, "sentiment breadth predicts reversal", confidence=threshold - 0.01
    )
    assert below["promotion_status"] == "pending_human"


def test_promote_idempotent_on_client_id(db, cfg):
    r1 = _promote(db, cfg, "book pressure predicts short term returns", client_id="m-1")
    r2 = _promote(db, cfg, "book pressure predicts short term returns", client_id="m-1")
    assert r1["belief_id"] == r2["belief_id"]
    assert db.conn.execute("SELECT COUNT(*) FROM beliefs").fetchone()[0] == 1


def test_lexical_contradiction_blocks_auto_promotion(db, cfg):
    first = _promote(
        db, cfg, "momentum predicts positive equity returns", confidence=0.9
    )
    assert first["promotion_status"] == "auto"

    # new claim shares >= 0.6 of its tokens with the existing claim
    second = _promote(
        db,
        cfg,
        "momentum predicts strong equity returns",  # 3/4 tokens overlap
        confidence=0.95,
        episodes=["ep_new"],
    )
    assert second["promotion_status"] == "pending_human"
    assert second["contradictions"] == [first["belief_id"]]

    row = db.conn.execute(
        "SELECT supporting_runs, contradicting_runs FROM beliefs WHERE belief_id = ?",
        (second["belief_id"],),
    ).fetchone()
    import json

    assert json.loads(row["supporting_runs"]) == []
    assert json.loads(row["contradicting_runs"]) == ["ep_new"]


def test_no_contradiction_when_tokens_differ(db, cfg):
    _promote(db, cfg, "momentum predicts positive equity returns", confidence=0.9)
    other = _promote(db, cfg, "weather surprises drive energy volatility", confidence=0.9)
    assert other["contradictions"] == []
    assert other["promotion_status"] == "auto"


def test_research_history_returns_tombstones_before_respend(db, cfg):
    """The money-saver check: after a failure, a mechanism-keyword query must
    surface the tombstone so the family is not re-spent blindly (§12.2)."""
    from spark_mcp import ledger

    hyp = ledger.propose_hypothesis(
        db,
        cfg,
        research_family="momentum",
        hypothesis_type="predictive",
        mechanism="forced rebalancing drives reversal",
        expected_direction="negative",
        horizon="20d",
        scope="cross_section",
        falsifier="reversal disappears net of costs",
        predicted_ic=0.02,
        predicted_sigma=0.01,
        universe="us_equity",
    )
    reg = ledger.register_hypothesis(db, cfg, hyp["hypothesis_id"], {"s": 1}, 9)
    ledger.log_outcome(
        db, cfg, reg["episode_id"], "failure", failure_reason="drawdown exceeded kill bound"
    )

    now = now_iso()
    hist = memory.research_history(db, "drawdown kill bound", now, k=5)
    assert len(hist["tombstones"]) == 1
    assert hist["tombstones"][0]["kind"] == "failed_hypothesis"
    # beliefs exist too but don't match this query; episodes match on failure_reason
    assert hist["beliefs"] == []
    assert any(
        ep["episode_id"] == reg["episode_id"] for ep in hist["episodes"]
    )  # failure_reason LIKE hit


def test_research_history_pit_filters_future_episodes(db, cfg):
    from spark_mcp import ledger

    hyp = ledger.propose_hypothesis(
        db,
        cfg,
        research_family="carry",
        hypothesis_type="predictive",
        mechanism="fx carry unwind clustering",
        expected_direction="negative",
        horizon="60d",
        scope="fx",
        falsifier="carry flat in crisis window",
        predicted_ic=0.02,
        predicted_sigma=0.01,
        universe="fx_majors",
    )
    ledger.register_hypothesis(db, cfg, hyp["hypothesis_id"], {"s": 1}, 9)

    assert memory.research_history(db, "carry", now_iso(), k=5)["episodes"]
    assert memory.research_history(db, "carry", "2000-01-01T00:00:00Z", k=5)["episodes"] == []


def test_research_history_kinds_selection_and_validation(db, cfg):
    _promote(db, cfg, "value spread mean reversion", as_of="2020-01-01T00:00:00Z")
    hist = memory.research_history(
        db, "value", "2021-01-01T00:00:00Z", k=5, kinds=["beliefs"]
    )
    assert set(hist) == {"beliefs"}
    with pytest.raises(ValueError, match="unknown history kinds"):
        memory.research_history(db, "value", now_iso(), kinds=["bogus"])
