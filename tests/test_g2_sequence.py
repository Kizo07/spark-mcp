"""G2 — the Phase 3 program gate (plan §18.0): full loop + 100% accounting.

Scripted 20-run sequence on the toy universe:
  * 15 normal backtest_runs across 3 families (propose -> register -> backtest
    -> DSR gate per run, the full propose->backtest->DSR->verdict loop);
  * 3 duplicate calls (same episode_id + spec + seed -> no new trial, no new
    dispatch);
  * 2 out-of-band log_outcome-with-metrics runs (no trial -> reconciled, C4).

Pass conditions: (counted + reconciled) / attested == 1.0 exactly, M_family
counts correct per family, and the timed loop stays under the plan's <15 min
(900s) SLO on this machine.
"""

from __future__ import annotations

import time

from spark_mcp import ledger
from spark_mcp.gates import service as g
from spark_mcp.providers.simulated import SimulatedProvider

FAMILIES = ("g2_momentum", "g2_mean_reversion", "g2_carry")
SPECS_PER_FAMILY = 5
RECONCILED_FAMILIES = ("g2_momentum", "g2_carry")  # one reconciled run each
G2_TIME_SLO = 900.0

PROPOSE = dict(
    hypothesis_type="predictive",
    mechanism="toy-universe signal evaluated by the simulated provider",
    expected_direction="positive",
    horizon="21d",
    scope="cross-section",
    falsifier="IC <= 0 net of costs",
    predicted_ic=0.03,
    predicted_sigma=0.01,
    universe="toy",
    idea="g2 scripted sequence",
)


class CountingProvider:
    """Wraps SimulatedProvider; counts real dispatches (G2's one-dispatch rule)."""

    def __init__(self, inner):
        self.inner = inner
        self.dispatches = 0

    @property
    def name(self):
        return self.inner.name

    def backtest(self, spec):
        self.dispatches += 1
        return self.inner.backtest(spec)

    def causal(self, spec):
        return self.inner.causal(spec)

    def risk(self, spec):
        return self.inner.risk(spec)


def make_spec(hypothesis_id: str, seed: int) -> dict:
    return {
        "hypothesis_id": hypothesis_id,
        "signal_ref": "artifacts://signals/g2.json",
        "params": {"type": "momentum", "lookback": 15 + seed % 10},
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


def test_g2_scripted_20_run_sequence(db, cfg):
    provider = CountingProvider(SimulatedProvider(cfg.artifacts_dir()))
    started = time.perf_counter()

    hypotheses = {}
    for family in FAMILIES:
        hyp = ledger.propose_hypothesis(
            db, cfg, research_family=family, **{**PROPOSE, "idea": f"g2 {family}"}
        )
        hypotheses[family] = hyp["hypothesis_id"]

    # -- 15 normal runs: 5 per family, full loop each --------------------------
    run_records = []  # (episode_id, spec) for the duplicate calls
    for family in FAMILIES:
        for i in range(SPECS_PER_FAMILY):
            seed = 1000 + i
            reg = ledger.register_hypothesis(db, cfg, hypotheses[family], {"g2": family}, seed)
            spec = make_spec(hypotheses[family], seed)
            result = g.backtest_run(db, cfg, provider, reg["episode_id"], spec, embedder=None)
            assert result["reused"] is False
            verdict = g.significance_gate(db, cfg, reg["episode_id"])  # DSR verdict
            assert verdict["sr_reported_with_dsr"] is True
            run_records.append((reg["episode_id"], spec))

    # -- 3 duplicate calls: identical (episode, spec, seed) --------------------
    for ep_id, spec in run_records[:3]:
        dup = g.backtest_run(db, cfg, provider, ep_id, spec, embedder=None)
        assert dup["reused"] is True

    # -- 2 out-of-band runs: metrics arrive with no trial (C4 reconciled) ------
    reconciled_episodes = []
    for family in RECONCILED_FAMILIES:
        reg = ledger.register_hypothesis(db, cfg, hypotheses[family], {"g2": "oob"}, 7777)
        out = ledger.log_outcome(
            db,
            cfg,
            reg["episode_id"],
            "success",
            metrics={"sr": 1.1, "T": 750, "skew": 0.1, "kurt": 3.5},
            costs={"usd": 0.10, "tokens": 50},
        )
        assert out["trial_source"] == "reconciled"
        reconciled_episodes.append(reg["episode_id"])

    elapsed = time.perf_counter() - started
    print(f"\nG2 loop time: {elapsed:.2f}s (SLO < {G2_TIME_SLO:.0f}s)")

    # -- G2 accounting: (counted + reconciled) / attested == 1.0 exactly --------
    counted = db.conn.execute(
        "SELECT COUNT(*) AS n FROM trials WHERE source = 'spark' AND status = 'completed'"
    ).fetchone()["n"]
    reconciled = db.conn.execute(
        "SELECT COUNT(*) AS n FROM trials WHERE source = 'reconciled'"
    ).fetchone()["n"]
    attested = 15 + 2  # scripted run intents: 15 dispatched + 2 out-of-band
    assert counted == 15, f"counted={counted}"
    assert reconciled == 2, f"reconciled={reconciled}"
    assert (counted + reconciled) / attested == 1.0

    # exactly one dispatch per distinct spec — duplicates never re-dispatched
    assert provider.dispatches == 15

    # M_family: 5 counted per family + 1 reconciled for the two OOB families
    for family in FAMILIES:
        expected = SPECS_PER_FAMILY + (1 if family in RECONCILED_FAMILIES else 0)
        assert g.M_family(db, family) == expected, f"{family}: M={g.M_family(db, family)}"

    # episodes reconciled carry the flag; gate verdicts exist for the 15
    flagged = db.conn.execute(
        "SELECT COUNT(*) AS n FROM episodes WHERE reconciled = 1"
    ).fetchone()["n"]
    assert flagged == 2
    for ep_id, _ in run_records:
        ep = ledger.get_episode(db, ep_id)
        assert ep["corrected_metrics"]["p_value"] is not None
        assert ep["status"] in ("validated", "inconclusive")

    # the loop meets the plan's <15 min SLO with orders of magnitude to spare
    assert elapsed < G2_TIME_SLO
