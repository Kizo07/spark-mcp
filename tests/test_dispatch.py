"""Dispatch pipeline (Fix D): order, profiles, and exactly-once preservation.

Proves: canonical step order via recording fakes; the profile registry;
trial-insert-before-dispatch, tombstone-block-before-trial-insert, and
budget-block-before-dispatch through the REAL service steps; idempotent
replay short-circuit; read-only assessment without writes; and the
transaction grouping (tombstone write survives a later rollback).
"""

from __future__ import annotations

import pytest

from spark_mcp import dispatch as d
from spark_mcp import ledger
from spark_mcp.dispatch import DispatchContext, Pipeline, Step
from spark_mcp.errors import ToolError
from spark_mcp.gates import service as g
from spark_mcp.llm import FakeEmbedder
from spark_mcp.providers.protocols import BacktestSpec
from spark_mcp.providers.simulated import SimulatedProvider

FAMILY = "dispatch_momentum"

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
def episode(db, cfg):
    hyp = ledger.propose_hypothesis(db, cfg, **PROPOSE)
    reg = ledger.register_hypothesis(db, cfg, hyp["hypothesis_id"], {"signal": "mom"}, 7)
    return reg["episode_id"], hyp["hypothesis_id"]


def make_ctx(db, cfg, episode, provider, seed=11, **kw):
    ep_id, hyp_id = episode
    ep = ledger.get_episode(db, ep_id)
    hyp = ledger.get_hypothesis(db, hyp_id)
    return DispatchContext(
        db=db,
        cfg=cfg,
        provider=provider,
        episode_id=ep_id,
        bt=BacktestSpec.model_validate(make_spec(hyp_id, seed)),
        episode=ep,
        hypothesis=hyp,
        embedder=kw.pop("embedder", FakeEmbedder()),
        **kw,
    )


class SpyProvider:
    """Sees the pending trial row iff insert precedes dispatch in one txn."""

    def __init__(self, inner, db):
        self.inner = inner
        self.db = db
        self.episode_id = None
        self.calls = 0
        self.seen = []

    @property
    def name(self):
        return getattr(self.inner, "name", "spy")

    def backtest(self, spec):
        row = self.db.conn.execute(
            "SELECT status FROM trials WHERE episode_id = ? ORDER BY created_at DESC LIMIT 1",
            (self.episode_id,),
        ).fetchone()
        self.seen.append(row["status"] if row else None)
        self.calls += 1
        return self.inner.backtest(spec)


# ---------------------------------------------------------------------------
# order + registry (recording fakes)


def test_fake_steps_run_in_canonical_order(db, cfg):
    calls = []

    def fake(name):
        def run(ctx):
            calls.append(name)

        return run

    pipe = Pipeline([Step(name=n, func=fake(n), transactional=False) for n in d.STEP_ORDER])
    ctx = DispatchContext(db=db, cfg=cfg)
    pipe.run(ctx)
    assert calls == list(d.STEP_ORDER)
    assert ctx.events == list(d.STEP_ORDER)


def test_canonical_order_is_enforced(db, cfg):
    with pytest.raises(ValueError, match="canonical order"):
        Pipeline(
            [
                Step(name=d.TRIAL_INSERT, func=lambda c: None),
                Step(name=d.TOMBSTONE_PRECHECK, func=lambda c: None),
            ]
        )
    with pytest.raises(KeyError, match="unknown dispatch step"):
        Pipeline([Step(name="nope", func=lambda c: None)])
    with pytest.raises(KeyError, match="unknown dispatch profile"):
        d.build_pipeline("nope", lambda n: lambda c: None)


def test_profile_registry_contents():
    assert d.PROFILES["gated-backtest"] == d.STEP_ORDER
    assert d.PROFILES["evolution-step"] == d.STEP_ORDER
    assert d.PROFILES["read-only"] == (d.TOMBSTONE_PRECHECK, d.BUDGET_CHECK)
    assert g.build_dispatch_pipeline("gated-backtest").names == list(d.STEP_ORDER)
    assert g.build_dispatch_pipeline("read-only").names == [
        d.TOMBSTONE_PRECHECK,
        d.BUDGET_CHECK,
    ]


def test_a_raise_aborts_later_steps(db, cfg):
    def boom(ctx):
        raise ToolError("stop here")

    pipe = Pipeline(
        [
            Step(name=d.TOMBSTONE_PRECHECK, func=lambda c: None, transactional=False),
            Step(name=d.BUDGET_CHECK, func=boom, transactional=False),
            Step(name=d.TRIAL_INSERT, func=lambda c: None, transactional=False),
        ]
    )
    ctx = DispatchContext(db=db, cfg=cfg)
    with pytest.raises(ToolError, match="stop here"):
        pipe.run(ctx)
    assert ctx.events == [d.TOMBSTONE_PRECHECK, d.BUDGET_CHECK]


def test_transaction_grouping_keeps_non_txn_write_on_rollback(db, cfg):
    def marker(ctx):
        ctx.db.execute(
            "INSERT INTO tool_effects (client_id, tool, subject_id, result, created_at)"
            " VALUES ('m1', 'probe', 's', '{}', 'now')"
        )

    def boom(ctx):
        ctx.db.execute(
            "INSERT INTO tool_effects (client_id, tool, subject_id, result, created_at)"
            " VALUES ('m2', 'probe', 's', '{}', 'now')"
        )
        raise ToolError("rollback me")

    pipe = Pipeline(
        [
            Step(name=d.TOMBSTONE_PRECHECK, func=marker, transactional=False),
            Step(name=d.BUDGET_CHECK, func=boom, transactional=True),
        ]
    )
    with pytest.raises(ToolError, match="rollback me"):
        pipe.run(DispatchContext(db=db, cfg=cfg))
    rows = {
        r["client_id"] for r in db.conn.execute("SELECT client_id FROM tool_effects").fetchall()
    }
    assert rows == {"m1"}  # non-txn write survives; txn chunk rolled back


# ---------------------------------------------------------------------------
# real steps: exactly-once semantics preserved


def test_trial_inserted_before_provider_dispatch(db, cfg, episode):
    spy = SpyProvider(SimulatedProvider(cfg.artifacts_dir()), db)
    spy.episode_id = episode[0]
    ctx = make_ctx(db, cfg, episode, spy)
    g.build_dispatch_pipeline("gated-backtest").run(ctx)
    assert ctx.events == list(d.STEP_ORDER)
    assert spy.calls == 1
    assert spy.seen == ["pending"]
    assert ctx.error is None and not ctx.halted
    row = db.conn.execute("SELECT * FROM trials WHERE episode_id = ?", (episode[0],)).fetchone()
    assert row["status"] == "completed" and row["source"] == "spark"


def test_tombstone_blocks_before_trial_insert(db, cfg, episode, monkeypatch):
    from spark_mcp import memory

    tomb = memory.add_tombstone(
        db,
        kind="failed_hypothesis",
        summary="momentum failed OOS after costs",
        match_keys={"keywords": ["momentum"], "mech_tags": [FAMILY]},
        evidence_episode_ids=[],
        embedder=FakeEmbedder(),
    )

    def fake_candidates(db_, embedder_, text):
        return [
            {
                "tombstone_id": tomb["tombstone_id"],
                "summary": tomb["summary"],
                "match_keys": tomb["match_keys"],
                "cosine": 0.95,
                "basis": "embedding",
            }
        ]

    monkeypatch.setattr(g, "tombstone_candidates", fake_candidates)
    spy = SpyProvider(SimulatedProvider(cfg.artifacts_dir()), db)
    spy.episode_id = episode[0]
    ctx = make_ctx(db, cfg, episode, spy)
    with pytest.raises(ToolError, match="tombstone pre-check blocked"):
        g.build_dispatch_pipeline("gated-backtest").run(ctx)
    assert ctx.events == [d.TOMBSTONE_PRECHECK]  # blocked before trial insert
    assert spy.calls == 0
    assert db.conn.execute("SELECT COUNT(*) AS n FROM trials").fetchone()["n"] == 0
    assert db.conn.execute("SELECT COUNT(*) AS n FROM spend_ledger").fetchone()["n"] == 0
    ep = ledger.get_episode(db, episode[0])
    assert ep["tombstone_check"]["status"] == "hit"  # block is still logged


def test_budget_block_before_dispatch(db, cfg, episode):
    from spark_mcp import budgets

    cfg.budgets.daily_usd = 0.5
    cfg.budgets.policy = "fail_closed"
    budgets.record_spend(db, cfg, family=FAMILY, source="manual", usd=1.0)
    spy = SpyProvider(SimulatedProvider(cfg.artifacts_dir()), db)
    spy.episode_id = episode[0]
    ctx = make_ctx(db, cfg, episode, spy)
    with pytest.raises(ToolError, match="daily budget exceeded"):
        g.build_dispatch_pipeline("gated-backtest").run(ctx)
    assert ctx.events == [d.TOMBSTONE_PRECHECK, d.BUDGET_CHECK]
    assert spy.calls == 0
    assert db.conn.execute("SELECT COUNT(*) AS n FROM trials").fetchone()["n"] == 0
    rows = db.conn.execute("SELECT source FROM spend_ledger").fetchall()
    assert [r["source"] for r in rows] == ["manual"]


def test_idempotent_replay_halts_before_dispatch(db, cfg, episode):
    inner = SimulatedProvider(cfg.artifacts_dir())
    spy = SpyProvider(inner, db)
    spy.episode_id = episode[0]
    ctx1 = make_ctx(db, cfg, episode, spy)
    g.build_dispatch_pipeline("gated-backtest").run(ctx1)
    assert not ctx1.halted and spy.calls == 1
    ctx2 = make_ctx(db, cfg, episode, spy)  # same episode + seed
    g.build_dispatch_pipeline("gated-backtest").run(ctx2)
    assert ctx2.halted is True
    assert ctx2.result["reused"] is True
    assert ctx2.result["trial_id"] == ctx1.trial_id
    assert ctx2.events == [d.TOMBSTONE_PRECHECK, d.BUDGET_CHECK, d.TRIAL_INSERT]
    assert spy.calls == 1  # no second dispatch
    assert db.conn.execute("SELECT COUNT(*) AS n FROM trials").fetchone()["n"] == 1


def test_backtest_run_uses_pipeline_and_reports_full_order(db, cfg, episode, monkeypatch):
    seen_events = []
    real_run = Pipeline.run

    def spy_run(self, ctx):
        out = real_run(self, ctx)
        seen_events.extend(ctx.events)
        return out

    monkeypatch.setattr(Pipeline, "run", spy_run)
    ep_id, hyp_id = episode
    result = g.backtest_run(
        db,
        cfg,
        SimulatedProvider(cfg.artifacts_dir()),
        ep_id,
        make_spec(hyp_id, 11),
        embedder=FakeEmbedder(),
    )
    assert result["status"] == "testing"
    assert seen_events == list(d.STEP_ORDER)


# ---------------------------------------------------------------------------
# read-only profile: assessment without writes


def test_read_only_profile_assesses_without_writes(db, cfg, episode, monkeypatch):
    from spark_mcp import budgets

    cfg.budgets.daily_usd = 0.5  # over budget already
    budgets.record_spend(db, cfg, family=FAMILY, source="manual", usd=1.0)
    monkeypatch.setattr(g, "tombstone_candidates", lambda db_, e, t: [])
    ctx = make_ctx(db, cfg, episode, SimulatedProvider(cfg.artifacts_dir()), mode="assess")
    out = d.build_pipeline("read-only", g.resolve_dispatch_step).run(ctx)
    assert out.events == [d.TOMBSTONE_PRECHECK, d.BUDGET_CHECK]
    assert out.would_block == {d.TOMBSTONE_PRECHECK: False, d.BUDGET_CHECK: True}
    assert db.conn.execute("SELECT COUNT(*) AS n FROM trials").fetchone()["n"] == 0
    assert db.conn.execute("SELECT COUNT(*) AS n FROM spend_ledger").fetchone()["n"] == 1
    ep = ledger.get_episode(db, episode[0])
    assert ep["tombstone_check"] is None  # assess mode never persists the check
