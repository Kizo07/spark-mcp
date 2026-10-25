"""QD archive (plan §10) and exploration posterior wiring (§8.3): cell mapping
and named refusals, elite replacement, per-type reward sets, the
component_defaults honesty field, SQL-durable rebuild, posterior init/update,
exploration-bonus sourcing, archive_status, and suggest_next."""

import asyncio
import json

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError as ClientToolError

from spark_mcp import archive, budgets, ledger, novelty
from spark_mcp.config import Config
from spark_mcp.errors import ToolError
from spark_mcp.explore import explore_methods, suggest_next
from spark_mcp.gates import service as g
from spark_mcp.llm import FakeEmbedder
from spark_mcp.providers.simulated import SimulatedProvider
from spark_mcp.server import create_server
from spark_mcp.timeutil import now_iso

AS_OF = "2024-06-01T00:00:00Z"

DESCRIPTORS = {"family": "quant", "horizon": "20d", "turnover": "low"}

FULL_METRICS = {
    "oos_ic": 0.8,
    "dsr": 0.5,
    "robustness": 0.4,
    "orthogonality": 0.6,
    "novelty": 0.7,
}


def _propose(db, cfg, family="quant", hypothesis_type="predictive", **overrides):
    fields = dict(
        research_family=family,
        hypothesis_type=hypothesis_type,
        mechanism="intermediate-horizon momentum reversal driven by forced rebalancing",
        expected_direction="negative",
        horizon="20d",
        scope="cross_section",
        falsifier="reversal disappears net of transaction costs",
        predicted_ic=0.03,
        predicted_sigma=0.01,
        universe="us_equity",
        idea="archive test hypothesis",
    )
    fields.update(overrides)
    return ledger.propose_hypothesis(db, cfg, **fields)


def _insert(db, cfg, hypothesis_id="hyp_x", metrics=None, reward_type="predictive", **kw):
    return archive.Archive(db, cfg).insert(
        hypothesis_id, DESCRIPTORS, metrics if metrics is not None else FULL_METRICS,
        reward_type, **kw,
    )


def _spec(hypothesis_id: str, seed: int) -> dict:
    return {
        "hypothesis_id": hypothesis_id,
        "signal_ref": "artifacts://signals/t.json",
        "params": {"type": "momentum", "lookback": 15, "turnover": 0.1},
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


def _cfg_for(tmp_path):
    return Config(
        storage={"db_path": f"{tmp_path}/spark.db", "artifacts_dir": f"{tmp_path}/art"}
    )


def _server(tmp_path):
    return create_server(_cfg_for(tmp_path), embedder=FakeEmbedder())


# ---------------------------------------------------------------------------
# cell mapping & refusals (§10.1)


def test_cell_mapping_resolves_known_categories(db, cfg):
    arch = archive.Archive(db, cfg)
    assert arch.cell_index(DESCRIPTORS) == (0, 2, 0)
    # case-insensitive, whitespace-tolerant
    assert arch.cell_index({"family": " QUANT ", "horizon": "20D", "turnover": "Low"}) == (0, 2, 0)
    # every bin family maps into the grid
    for i, fam in enumerate(cfg.research.qd_family_bins):
        assert arch.cell_index({**DESCRIPTORS, "family": fam})[0] == i


def test_numeric_turnover_maps_through_config_thresholds(db, cfg):
    arch = archive.Archive(db, cfg)
    assert arch.cell_index({**DESCRIPTORS, "turnover": 0.10})[2] == 0  # low
    assert arch.cell_index({**DESCRIPTORS, "turnover": 0.50})[2] == 1  # medium
    assert arch.cell_index({**DESCRIPTORS, "turnover": 1.50})[2] == 2  # high
    assert arch.cell_index({**DESCRIPTORS, "turnover": 9.99})[2] == 3  # very_high


def test_unknown_category_refused_with_named_error(db, cfg):
    arch = archive.Archive(db, cfg)
    with pytest.raises(ToolError, match="unknown 'family' category 'astrology'"):
        arch.cell_index({**DESCRIPTORS, "family": "astrology"})
    with pytest.raises(ToolError, match="unknown 'horizon' category '17m'"):
        arch.cell_index({**DESCRIPTORS, "horizon": "17m"})
    with pytest.raises(ToolError, match="missing 'turnover' descriptor"):
        arch.cell_index({"family": "quant", "horizon": "20d"})


def test_grid_shape_must_match_bin_lists(db, cfg):
    cfg.research.qd_grid = [7, 6, 4]
    with pytest.raises(ValueError, match="grid size 7"):
        archive.Archive(db, cfg)


# ---------------------------------------------------------------------------
# insertion, elite replacement, SQL durability (§10.1)


def test_insert_writes_sql_durable_truth_and_rebuilds(db, cfg):
    out = _insert(db, cfg, "hyp_first")
    assert out["elite_replaced"] is True
    assert out["cell_key"] == "0:2:0"
    # a FRESH archive rebuilt from qd_cells sees the same elite (M7)
    arch2 = archive.Archive(db, cfg)
    cov = arch2.coverage()
    assert cov["occupied_cells"] == 1
    assert cov["per_family"]["quant"] == 1
    elites = arch2.elites(10)
    assert elites[0]["elite_hypothesis_id"] == "hyp_first"
    assert elites[0]["elite_score"] == pytest.approx(out["elite_score"])
    assert elites[0]["descriptors"]["family"] == "quant"  # self-describing row
    assert elites[0]["descriptors"]["horizon"] == "20d"
    row = db.conn.execute("SELECT * FROM qd_cells WHERE cell_key = '0:2:0'").fetchone()
    assert row["descriptor_spec_version"] == 1
    assert row["family"] == "quant"
    assert row["reward_type"] == "predictive"


def test_elite_replacement_rules(db, cfg):
    arch = archive.Archive(db, cfg)
    arch.insert("hyp_a", DESCRIPTORS, FULL_METRICS, "predictive")
    # higher score replaces
    better = arch.insert("hyp_b", DESCRIPTORS, {**FULL_METRICS, "oos_ic": 0.95}, "predictive")
    assert better["elite_replaced"] is True
    assert better["elite_hypothesis_id"] == "hyp_b"
    assert better["trials"] == 2
    # lower score does not
    worse = arch.insert("hyp_c", DESCRIPTORS, {**FULL_METRICS, "oos_ic": 0.10}, "predictive")
    assert worse["elite_replaced"] is False
    assert worse["elite_hypothesis_id"] == "hyp_b"
    assert worse["trials"] == 3
    # tie keeps the incumbent
    tied = arch.insert("hyp_d", DESCRIPTORS, {**FULL_METRICS, "oos_ic": 0.95}, "predictive")
    assert tied["elite_replaced"] is False
    assert tied["elite_hypothesis_id"] == "hyp_b"
    arch2 = archive.Archive(db, cfg)
    top = arch2.elites(1)[0]
    assert top["elite_hypothesis_id"] == "hyp_b"
    assert top["trials"] == 4


def test_elite_score_uses_spend_ledger_for_efficiency(db, cfg):
    budgets.record_spend(db, cfg, family="quant", source="test", usd=1.0, run_id="ep_x")
    out = _insert(db, cfg, metrics=FULL_METRICS, episode_id="ep_x")
    # efficiency = 1/(1+usd) = 0.5; the rest per FULL_METRICS
    expected = (
        0.30 * 0.8 + 0.20 * 0.5 + 0.15 * 0.4 + 0.15 * 0.6 + 0.10 * 0.7 + 0.10 * 0.5
    )
    assert out["components"]["efficiency"] == pytest.approx(0.5)
    assert out["elite_score"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# per-type reward sets (§10.2) + component_defaults honesty field


def test_reward_weights_per_type(db, cfg):
    arch = archive.Archive(db, cfg)
    full = arch.reward_weights("predictive")
    assert full == cfg.research.qd_reward_weights
    assert sum(full.values()) == pytest.approx(1.0)
    for rtype in ("portfolio", "risk"):
        assert arch.reward_weights(rtype) == full
    causal = arch.reward_weights("causal")
    assert "stability" in causal
    assert sum(causal.values()) == pytest.approx(1.0)
    # stability is drawn from the same budget: others scaled by (1 - 0.10)
    assert causal["stability"] == pytest.approx(0.10)
    assert causal["oos"] == pytest.approx(full["oos"] * 0.9)


def test_descriptive_execution_excluded_unless_opted_in(db, cfg):
    arch = archive.Archive(db, cfg)
    for rtype in ("descriptive", "execution"):
        with pytest.raises(ToolError, match="excluded from QD insertion by default"):
            arch.insert("hyp_d", DESCRIPTORS, FULL_METRICS, rtype)
        out = arch.insert("hyp_d", DESCRIPTORS, FULL_METRICS, rtype, allow_descriptive=True)
        reduced = cfg.research.qd_reduced_reward_weights
        assert set(out["components"]) == set(reduced)
        assert sum(reduced.values()) == pytest.approx(1.0)
        # reduced set ignores oos/dsr entirely: score from the non-Sharpe parts
        assert out["elite_score"] == pytest.approx(
            reduced["robustness"] * 0.4 + reduced["orthogonality"] * 0.6
            + reduced["novelty"] * 0.7 + reduced["efficiency"] * 1.0
        )


def test_unknown_reward_type_refused(db, cfg):
    with pytest.raises(ToolError, match="unknown reward_type 'witchcraft'"):
        _insert(db, cfg, reward_type="witchcraft")


def test_component_defaults_records_missing_components(db, cfg):
    out = _insert(db, cfg, metrics={"oos_ic": 0.8})
    assert out["components"]["oos"] == pytest.approx(0.8)
    assert set(out["component_defaults"]) == {
        "deflated_significance", "robustness", "orthogonality", "novelty",
    }
    row = db.conn.execute("SELECT descriptors FROM qd_cells WHERE cell_key = '0:2:0'").fetchone()
    recorded = json.loads(row["descriptors"])
    assert recorded["component_defaults"] == out["component_defaults"]
    # nothing missing -> honesty field empty
    out2 = _insert(
        db, cfg, "hyp_y",
        metrics={**FULL_METRICS, "stability": 0.9}, reward_type="causal",
    )
    assert out2["component_defaults"] == []


def test_deflated_significance_prefers_stored_gate_fields(db, cfg):
    metrics = {**FULL_METRICS, "dsr": None,
               "corrected_metrics": {"dsr": 0.9, "psr": 0.8, "p_value": 0.1}}
    out = _insert(db, cfg, metrics=metrics)
    assert out["components"]["deflated_significance"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# posterior wiring (§8.3)


def test_posterior_initialized_at_prior_and_updated(db, cfg):
    g.record_family_outcome(db, cfg, "quant", success=True, episode_id="ep_1")
    post = archive.family_posteriors(db)["quant"]
    assert (post["alpha"], post["beta"]) == (2.0, 1.0)
    g.record_family_outcome(db, cfg, "quant", success=False, episode_id="ep_2")
    g.record_family_outcome(db, cfg, "quant", success=False, episode_id="ep_3")
    post = archive.family_posteriors(db)["quant"]
    assert (post["alpha"], post["beta"]) == (2.0, 3.0)
    assert post["mean"] == pytest.approx(0.4)


def test_posterior_honors_configured_prior(db, cfg):
    cfg.research.bandit_prior = [2.0, 3.0]
    g.record_family_outcome(db, cfg, "quant", success=True, episode_id="ep_1")
    post = archive.family_posteriors(db)["quant"]
    assert (post["alpha"], post["beta"]) == (3.0, 3.0)
    assert post["mean"] == pytest.approx(0.5)  # (a,b) mean at the prior == 0.5


def test_family_posteriors_stored_row_wins(db, cfg):
    """The stored exploration_state posterior is authoritative; the
    episode-derived path is the reconciliation fallback for pre-Phase-5 data
    (episodes with outcomes, no posterior row — e.g. a v6 database upgraded
    to v7 whose outcomes never ticked a stored row)."""
    hyp = _propose(db, cfg, family="quant")
    reg = ledger.register_hypothesis(db, cfg, hyp["hypothesis_id"], {"a": 1}, 3)
    ledger.log_outcome(db, cfg, reg["episode_id"], "success", metrics={"ic": 0.02})
    g.record_family_outcome(db, cfg, "causal", success=True, episode_id="ep_x")
    # simulate the legacy state: outcomes exist but the posterior row is gone
    db.conn.execute("DELETE FROM exploration_state WHERE family = 'quant'")
    from spark_mcp.explore import _family_posteriors

    posteriors = _family_posteriors(db, cfg, now_iso())
    # stored (2,1) row wins even though 'causal' has no episodes at all
    assert posteriors["causal"]["posterior_source"] == "stored"
    assert posteriors["causal"]["prior_success"] == pytest.approx(2 / 3)
    # 'quant' reconciles from its PIT episodes (1 success at prior (1,1))
    assert posteriors["quant"]["posterior_source"] == "episodes"
    assert posteriors["quant"]["prior_success"] == pytest.approx(2 / 3)
    # PIT: rows updated after as_of are invisible
    posteriors_past = _family_posteriors(db, cfg, "2001-01-01T00:00:00Z")
    assert "causal" not in posteriors_past


def test_exploration_bonus_is_posterior_mean_uniform_at_prior(seeded, cfg, embedder):
    """Phase 5: the bonus is the Thompson posterior mean when a stored row
    exists; with no outcomes (the (1,1) prior) it equals the old uniform 0.5,
    so empty-state decomposition outputs are unchanged."""
    now = now_iso()
    out = explore_methods(
        seeded, cfg, "detect regime shifts in volatility", k=3,
        as_of=now, embedder=embedder, seed=5,
    )
    assert out["suggestions"], "seeded corpus visible at now"
    assert all(s["decomposition"]["exploration_bonus"] == 0.5 for s in out["suggestions"])

    hyp = _propose(seeded, cfg, family="causal", hypothesis_type="causal",
                   dag={"nodes": ["a", "b"], "edges": []},
                   adjustment_set=["x"])
    reg = ledger.register_hypothesis(seeded, cfg, hyp["hypothesis_id"], {"a": 1}, 3)
    ledger.log_outcome(seeded, cfg, reg["episode_id"], "success", metrics={"ic": 0.02})
    # fresh as_of AFTER the outcome: the stored row's updated_at must be <= as_of
    out2 = explore_methods(
        seeded, cfg, "estimate the causal effect of a policy shock", k=5,
        as_of=now_iso(), embedder=embedder, seed=5,
    )
    bonuses = {s["family"]: s["decomposition"]["exploration_bonus"] for s in out2["suggestions"]}
    assert bonuses.get("causal") == pytest.approx(2 / 3)  # (2,1) mean


# ---------------------------------------------------------------------------
# cell_emptiness: archive mode vs degraded approximation (§7)


def test_cell_emptiness_archive_mode_slice_formula(db, cfg, embedder):
    arch = archive.Archive(db, cfg)
    arch.insert("hyp_q1", DESCRIPTORS, FULL_METRICS, "predictive")
    arch.insert("hyp_q2", DESCRIPTORS, FULL_METRICS, "predictive")
    # only the quant slice is occupied: max_slice_trials = 2
    assert arch.cell_emptiness("quant") == pytest.approx(0.0)
    assert arch.cell_emptiness("econometrics") == 1.0  # empty slice
    out = novelty.check_novelty(
        db, cfg, "anything", family="quant", embedder=embedder,
        archive=arch,  # occupancy stamped now -> read at the current view
    )
    assert out["cell_emptiness"] == pytest.approx(0.0)
    assert out["cell_emptiness_basis"] == "qd_archive"
    out2 = novelty.check_novelty(
        db, cfg, "anything", family="econometrics", embedder=embedder,
        archive=arch,
    )
    assert out2["cell_emptiness"] == 1.0
    # a family outside the bin vocabulary degrades to the approximation
    out3 = novelty.check_novelty(
        db, cfg, "anything", family="voodoo", as_of=AS_OF, embedder=embedder,
        archive=arch,
    )
    assert out3["cell_emptiness_basis"] == "family_count_approximation"
    # no archive injected -> degraded approximation (pinned Phase 2 behavior)
    out4 = novelty.check_novelty(
        db, cfg, "anything", family="quant", as_of=AS_OF, embedder=embedder,
    )
    assert out4["cell_emptiness_basis"] == "family_count_approximation"
    assert out4["cell_emptiness"] == 1.0  # no families tried


def test_cell_emptiness_scaled_by_occupancy(db, cfg):
    arch = archive.Archive(db, cfg)
    arch.insert("hyp_q1", DESCRIPTORS, FULL_METRICS, "predictive")
    arch.insert("hyp_q2", DESCRIPTORS, FULL_METRICS, "predictive")
    # one occupied cell in the econometrics slice -> 1 - 1/2
    arch.insert("hyp_e1", {**DESCRIPTORS, "family": "econometrics"}, FULL_METRICS, "predictive")
    assert arch.cell_emptiness("econometrics") == pytest.approx(0.5)
    assert 0.0 <= arch.cell_emptiness("quant") <= 1.0


# ---------------------------------------------------------------------------
# archive_status (plan §6.3 tool 19)


def test_archive_status_fields(db, cfg):
    hyp = _propose(db, cfg, predicted_ic=0.03)
    reg = ledger.register_hypothesis(db, cfg, hyp["hypothesis_id"], {"a": 1}, 3)
    provider = SimulatedProvider(cfg.artifacts_dir())
    g.backtest_run(db, cfg, provider, reg["episode_id"], _spec(hyp["hypothesis_id"], 3),
                   embedder=None)
    ledger.log_outcome(
        db, cfg, reg["episode_id"], "success",
        metrics={"oos_ic": 0.01, "robustness": 0.5, "orthogonality": 0.5,
                 "novelty": 0.5},
        costs={"usd": 0.25, "tokens": 10},
    )
    out = archive.status(db, cfg)
    assert out["coverage"]["occupied_cells"] == 1
    assert out["coverage"]["total_cells"] == 8 * 6 * 4
    assert out["coverage"]["per_family"]["quant"] == 1
    assert out["elites"][0]["elite_hypothesis_id"] == hyp["hypothesis_id"]
    assert out["family_M"] == {"quant": 1}
    assert out["spend"]["total_usd"] == pytest.approx(0.25)
    # two genuine outcome events (backtest_run + log_outcome): posterior (3,1)
    post = out["family_posteriors"]["quant"]
    assert (post["alpha"], post["beta"]) == (3.0, 1.0)
    # |predicted_ic - realized_ic| = |0.03 - 0.01|
    assert out["predictor_error_map"]["quant"] == pytest.approx(0.02)


def test_archive_status_tool_over_mcp(tmp_path):
    async def run():
        async with Client(_server(tmp_path)) as client:
            return json.loads(
                (await client.call_tool("archive_status", {})).content[0].text
            )

    out = asyncio.run(run())
    for key in ("coverage", "elites", "family_M", "spend", "family_posteriors",
                "predictor_error_map"):
        assert key in out, f"archive_status missing {key}"
    assert out["coverage"]["occupied_cells"] == 0
    assert out["coverage"]["total_cells"] == 192
    assert out["spend"]["today_usd"] == 0.0


# ---------------------------------------------------------------------------
# suggest_next (plan §6.1 tool 6)


def test_suggest_next_seeding_reproducible_and_trajectory(db, cfg, seeded):
    a = suggest_next(seeded, cfg, n=2, as_of=AS_OF, seed=42)
    b = suggest_next(seeded, cfg, n=2, as_of=AS_OF, seed=42)
    assert a == b
    c = suggest_next(seeded, cfg, n=2, as_of=AS_OF, seed=43)
    assert [s["cell_key"] for s in a["suggestions"]] != [
        s["cell_key"] for s in c["suggestions"]
    ] or a["family_selection_probabilities"] != c["family_selection_probabilities"]
    probs = a["family_selection_probabilities"]
    assert sum(probs.values()) == pytest.approx(1.0)
    for s in a["suggestions"]:
        assert s["cell_trials"] == 0  # empty archive: everything underexplored
        assert 0.0 < s["selection_probability"] <= 1.0
        assert s["descriptors"]["family"] == s["family"]
        assert s["methods"], "seeded corpus supplies candidate methods"
        for m in s["methods"]:
            assert m["selection_probability"] == pytest.approx(s["selection_probability"])
    events = seeded.conn.execute(
        "SELECT payload FROM trajectory_events WHERE run_id = 'suggest_next'"
        " ORDER BY seq"
    ).fetchall()
    assert len(events) == 3  # one trajectory event per call
    payload = json.loads(events[0]["payload"])
    assert payload["tool"] == "suggest_next"
    assert payload["seed"] == 42


def test_suggest_next_prefers_emptiest_cells(db, cfg):
    # occupy the (quant, 1d, low) cell so the family's suggestion moves
    # elsewhere; the occupancy is stamped at AS_OF so the PIT read at AS_OF
    # sees it (nit-1 fix: archive reads are as_of-filtered).
    db.execute(
        "INSERT INTO qd_cells (cell_key, descriptors, descriptor_spec_version,"
        " elite_hypothesis_id, dsr, oos_ic, family, trials, reward_type,"
        " updated_at) VALUES ('0:0:0', ?, 1, 'hyp_q', 0.9, 0.5, 'quant', 1,"
        " 'predictive', ?)",
        ('{"family": "quant", "horizon": "1d", "turnover": "low",'
         ' "elite_score": 0.9}', AS_OF),
    )
    out = suggest_next(db, cfg, n=8, as_of=AS_OF, seed=7)
    quant_suggestions = [s for s in out["suggestions"] if s["family"] == "quant"]
    assert quant_suggestions  # n=8 covers all 8 bin families
    for s in quant_suggestions:
        assert s["cell_key"] != "0:0:0"
        assert s["cell_trials"] == 0


def test_suggest_next_gap_report_id_unknown_refused(db, cfg):
    # Phase 7 (§13/G5): gap_report_id is consumed — an unknown report id is
    # a named ToolError from the artifact loader (placeholder removed).
    with pytest.raises(ToolError, match="unknown gap_report_id"):
        suggest_next(db, cfg, gap_report_id="rep_gap_123")


def test_suggest_next_tool_over_mcp(tmp_path):
    async def run():
        server = _server(tmp_path)
        async with Client(server) as client:
            out = json.loads(
                (await client.call_tool("suggest_next", {"n": 2, "seed": 11})).content[0].text
            )
            with pytest.raises(ClientToolError, match="unknown gap_report_id"):
                await client.call_tool("suggest_next", {"gap_report_id": "rep_gap_1"})
            return out

    out = asyncio.run(run())
    assert len(out["suggestions"]) == 2
    assert set(out["family_selection_probabilities"]) == set(
        Config().research.qd_family_bins
    )


def test_numeric_turnover_out_of_range_refused(db, cfg):
    import math

    import pytest

    from spark_mcp.errors import ToolError

    arch = archive.Archive(db, cfg)
    base = {"family": "quant", "horizon": "20d"}
    for bad in (-1.0, float("nan"), float("inf"), True):
        with pytest.raises(ToolError, match="turnover"):
            arch.insert(
                hypothesis_id="hyp_x", descriptors={**base, "turnover": bad},
                metrics={}, reward_type="predictive",
            )
    # boundary: 0.0 is a valid 'low'
    out = arch.insert(
        hypothesis_id="hyp_ok", descriptors={**base, "turnover": 0.0},
        metrics={"oos_ic": 0.5}, reward_type="predictive",
    )
    assert out["cell_key"].endswith(":0")
    assert not math.isnan(out["elite_score"])


def test_non_replacing_insert_reports_incumbent_score(db, cfg):
    arch = archive.Archive(db, cfg)
    d = {"family": "quant", "horizon": "20d", "turnover": "low"}
    first = arch.insert(
        hypothesis_id="hyp_a", descriptors=d,
        metrics={"oos_ic": 0.8}, reward_type="predictive",
    )
    weaker = arch.insert(
        hypothesis_id="hyp_b", descriptors=d,
        metrics={"oos_ic": 0.1}, reward_type="predictive",
    )
    assert weaker["elite_replaced"] is False
    assert weaker["incumbent_elite_score"] == first["elite_score"]
    # a replacing insert has no incumbent to report
    stronger = arch.insert(
        hypothesis_id="hyp_c", descriptors=d,
        metrics={"oos_ic": 0.95}, reward_type="predictive",
    )
    assert stronger["elite_replaced"] is True
    assert stronger["incumbent_elite_score"] is None


# ---------------------------------------------------------------------------
# PIT as_of threading (Phase-5 review nit 1): future-dated archive/posterior
# state must never leak into past reads (plan hard rule 3).

PAST = "2020-01-01T00:00:00Z"
FUTURE = "2999-01-01T00:00:00Z"


def _future_cell(db, cfg):
    """Write a qd_cells row directly with a future updated_at."""
    db.execute(
        "INSERT INTO qd_cells (cell_key, descriptors, descriptor_spec_version,"
        " elite_hypothesis_id, dsr, oos_ic, family, trials, reward_type,"
        " updated_at) VALUES (?, ?, 1, 'hyp_future', 0.9, 0.5, 'quant', 7,"
        " 'predictive', ?)",
        ("0:0:0", '{"family": "quant", "horizon": "1d", "turnover": "low",'
        ' "elite_score": 0.9}', FUTURE),
    )


def test_future_cell_hidden_from_past_archive_reads(db, cfg):
    arch = archive.Archive(db, cfg)
    assert arch.coverage()["occupied_cells"] == 0
    assert arch.coverage(PAST)["occupied_cells"] == 0
    _future_cell(db, cfg)
    arch2 = archive.Archive(db, cfg)
    assert arch2.coverage()["occupied_cells"] == 1          # now sees it
    assert arch2.coverage(PAST)["occupied_cells"] == 0      # past does not
    assert arch2.elites(5, PAST) == []
    assert arch2.cell_emptiness("quant", PAST) == 1.0       # slice empty then
    assert arch2.cell_emptiness("quant") < 1.0              # occupied now
    assert arch2.emptiest_cell("quant", PAST)["trials"] == 0


def test_future_posterior_hidden_from_past_reads(db, cfg):
    # row stamped between PAST and now: a 2020 read must not see it, the
    # current read must.
    db.execute(
        "INSERT INTO exploration_state (family, posterior, trials, successes,"
        " updated_at) VALUES ('quant', '{\"alpha\": 9, \"beta\": 2}', 11, 9,"
        " '2021-01-01T00:00:00Z')",
    )
    assert "quant" in archive.family_posteriors(db)
    assert "quant" not in archive.family_posteriors(db, PAST)
    past_status = archive.status(db, cfg, PAST)
    assert "quant" not in past_status["family_posteriors"]
    now_status = archive.status(db, cfg)
    assert now_status["family_posteriors"]["quant"]["mean"] > 0.8


def test_suggest_next_past_view_uses_prior_not_future_posterior(db, cfg):
    from spark_mcp import explore

    db.execute(
        "INSERT INTO exploration_state (family, posterior, trials, successes,"
        " updated_at) VALUES ('quant', '{\"alpha\": 30, \"beta\": 2}', 32, 30, ?)",
        (FUTURE,),
    )
    out = explore.suggest_next(db, cfg, n=3, as_of=PAST, seed=11)
    assert out["as_of"] == PAST
    # every suggestion carries selection_probability + resolved descriptors
    assert all(0.0 < s["selection_probability"] <= 1.0 for s in out["suggestions"])


# ---------------------------------------------------------------------------
# shelf-depth report (P2-1): cards per shelf/family + QD occupancy per family
# from the real tables, PIT at as_of.


def _seed_cells_at(db, as_of):
    for key, fam, hor in (("0:2:0", "quant", "20d"), ("2:1:1", "causal", "5d")):
        db.execute(
            "INSERT INTO qd_cells (cell_key, descriptors, descriptor_spec_version,"
            " elite_hypothesis_id, dsr, oos_ic, family, trials, reward_type,"
            " updated_at) VALUES (?, ?, 1, ?, 0.9, 0.5, ?, 1, 'predictive', ?)",
            (
                key,
                json.dumps(
                    {"family": fam, "horizon": hor, "turnover": "low",
                     "elite_score": 0.9}
                ),
                f"hyp_{fam}",
                fam,
                as_of,
            ),
        )


def test_shelf_depth_counts_match_seeded_rows(db, cfg, seeded):
    """archive_status.shelf_depth mirrors the real tables: technique cards per
    family shelf (zero-filled on the bins, non-bin families as-is, NULL
    families unassigned — never attributed) + QD occupancy per family."""
    db.execute(
        "INSERT INTO graph_nodes (node_id, type, label, family, mech_tags,"
        " summary, as_of, created_at) VALUES ('node_zz1', 'technique',"
        " 'orphan card', NULL, '[]', 'no family', ?, ?)",
        (AS_OF, AS_OF),
    )
    db.execute(
        "INSERT INTO graph_nodes (node_id, type, label, family, mech_tags,"
        " summary, as_of, created_at) VALUES ('node_zz2', 'technique',"
        " 'outlier card', 'astrology', '[]', 'non-bin', ?, ?)",
        (AS_OF, AS_OF),
    )
    _seed_cells_at(seeded, AS_OF)

    expected_cards = {
        r["family"]: int(r["n"])
        for r in seeded.conn.execute(
            "SELECT family, COUNT(*) AS n FROM graph_nodes"
            " WHERE type = 'technique' AND as_of <= ? GROUP BY family",
            (AS_OF,),
        ).fetchall()
        if r["family"]
    }
    out = archive.status(seeded, cfg, as_of=AS_OF)
    shelf = out["shelf_depth"]
    assert shelf["as_of_basis"] == AS_OF
    for fam, n in expected_cards.items():
        assert shelf["cards_per_family"][fam] == n, fam
    for fam in cfg.research.qd_family_bins:  # zero-filled, joinable
        assert fam in shelf["cards_per_family"]
    assert shelf["cards_per_family"]["astrology"] == 1  # real data kept
    assert shelf["unassigned_cards"] == 1
    assert shelf["total_cards"] == sum(expected_cards.values()) + 1
    assert shelf["qd_occupancy_per_family"]["quant"] == 1
    assert shelf["qd_occupancy_per_family"]["causal"] == 1
    assert shelf["qd_occupied_cells"] == 2
    assert shelf["qd_total_cells"] == 8 * 6 * 4
    assert set(shelf["qd_occupancy_per_family"]) == set(cfg.research.qd_family_bins)
    # PIT: timeless seeds stay visible in 2001, but the AS_OF-stamped cells
    # and extra cards do not.
    past = archive.status(seeded, cfg, as_of="2001-01-01T00:00:00Z")["shelf_depth"]
    assert past["qd_occupied_cells"] == 0
    assert past["unassigned_cards"] == 0
    assert "astrology" not in past["cards_per_family"]
    assert past["total_cards"] == sum(expected_cards.values()) - 1  # minus astrology
