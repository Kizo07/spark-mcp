"""Adversarial pins for GEPA / fitness / promotion anti-gaming guarantees.

Attacks the five surfaces a gaming challenger would try, at the layer that
is supposed to stop it:

1. rubric leak — mutation/reflect must never read or emit fixture rubric
   or reference-keyword text (the retired ``add_check_step`` channel);
2. ties — promotion margins are strict (``>``) at both layers
   (artifact ``evolve.promote.decide``, skill ``skills.decide_promotion``);
3. keyword stuffing — an empty / keyword-soup challenger scores ~0 at the
   deterministic scorer itself (substance pre-filter: no leading
   instruction verb, no coverage) and loses to an honest baseline at the
   gate as well (held-out gap alarm, strict margins, scorer reads steps
   only);
4. min-fixtures guard — promotion requires FULL replay-suite evidence
   and the suite-size floor alarms below ``replay_min_fixtures``;
5. reflect — reflective variation rewrites text only; it cannot inject
   or alter the execution-grounded fitness record.
"""

import inspect
import json
import random
import re
from pathlib import Path

import pytest

from spark_mcp import ledger, skills
from spark_mcp.consolidate import _fixture_growth_gaps
from spark_mcp.errors import ToolError
from spark_mcp.evolve import fitness as fitness_mod
from spark_mcp.evolve import gepa
from spark_mcp.evolve import promote as promote_mod
from spark_mcp.evolve import reflect as reflect_mod
from spark_mcp.gates import service as gates
from spark_mcp.llm import FakeLLM
from spark_mcp.providers.simulated import SimulatedProvider

# -- honest baseline vs keyword soup (real replay fixtures) -------------------

HONEST_STEPS = [
    "States the universe and point-in-time membership rules before ranking on 12-1 momentum.",
    "Applies explicit transaction cost and slippage assumptions.",
    "Reports sharpe paired with a deflated significance check.",
    "Checks turnover and capacity before trusting the result.",
    "Names the regime segmentation method and its parameters.",
    "Validates regime labels out of sample before using them out-of-sample.",
    "Checks parameter stability across the labelled regimes.",
    "Draws an explicit causal graph before estimating anything with a dag.",
    "Names the adjustment set and the conditioning variables.",
    "Runs a placebo or negative control to challenge the effect.",
    "Verifies data availability as of each decision timestamp.",
    "Checks feature construction for lookahead windows.",
    "Documents the as_of policy used by the pipeline for point-in-time safety.",
    "Chooses a panel estimator with fixed effects for firm and time.",
    "Clusters standard errors at the firm level.",
    "Tests residuals for cross-sectional dependence.",
    "Selects a spectral estimation method suited to uneven sampling.",
    "Estimates significance of peaks against a null spectrum using fft.",
    "Checks stability of the dominant frequencies across windows.",
]

NEUTRAL_STEPS = ["Rank securities by past returns.", "Size positions equally."]


def _stuffed_steps(fixtures):
    """One-step keyword soup covering every held-in rubric/reference word."""
    words = set()
    for fixture in fixtures:
        if not fixture.get("held_in", True):
            continue
        for item in fixture["rubric"]:
            words.update(item.lower().split())
        for keyword in fixture["reference_keywords"]:
            words.update(keyword.lower().split("-"))
    return [" ".join(sorted(words))]


def _replay_scores(steps):
    fixtures = skills.load_fixtures()
    held_in = {
        f["task_id"]: gepa.fixture_score(steps, f)
        for f in fixtures
        if f.get("held_in", True)
    }
    held_out = {
        f["task_id"]: gepa.fixture_score(steps, f)
        for f in fixtures
        if not f.get("held_in", True)
    }
    gap = abs(sum(held_in.values()) / len(held_in) - sum(held_out.values()) / len(held_out))
    return held_in, held_out, gap


def _mean(scores):
    return sum(scores.values()) / len(scores) if scores else 0.0


# -- DB helpers (hand-set rows, mirroring test_evolution_promote/test_skills) --

PROPOSE = dict(
    research_family="toy_momentum",
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


def _registered_episode(db, cfg):
    hyp = ledger.propose_hypothesis(db, cfg, **PROPOSE)
    reg = ledger.register_hypothesis(db, cfg, hyp["hypothesis_id"], {"signal": "mom"}, 7)
    return reg["episode_id"]


def _set_corrected(db, episode_id, *, sr=1.2, dsr=0.97, passed=True):
    corrected = {
        "sr": sr,
        "psr": dsr,
        "dsr": dsr,
        "p_value": 1.0 - dsr,
        "M_family": 1,
        "min_T": 30,
        "alpha": 0.05,
        "T": 252,
        "skew": 0.1,
        "kurt": 3.2,
        "heavy_tail_caveat": False,
        "degenerate": False,
        "pass": passed,
    }
    db.execute(
        "UPDATE episodes SET corrected_metrics = ?, metrics = ?, costs = ?"
        " WHERE episode_id = ?",
        (
            json.dumps(corrected, sort_keys=True),
            json.dumps({"sr": sr, "T": 252, "simulated": True}, sort_keys=True),
            json.dumps({"bps": 5, "slippage": 2, "borrow": 0}),
            episode_id,
        ),
    )


def _next_free(db, name):
    row = db.conn.execute(
        "SELECT COALESCE(MAX(version),0) AS v FROM skills WHERE name = ?", (name,)
    ).fetchone()
    return int(row["v"]) + 1


def _insert_challenger(db, name, held_in, held_out, gap, *, parent_version=1, steps=None):
    sid = skills.new_skill_id()
    cid = skills.new_candidate_id()
    version = _next_free(db, name)
    lineage = {"parent_version": parent_version, "procedure_steps": steps or []}
    db.execute(
        "INSERT INTO skills (skill_id, name, version, role, body_uri, lineage,"
        " status, credit, benchmark_history, promotion_status, created_at)"
        " VALUES (?, ?, ?, NULL, ?, ?, 'challenger', '{}', '[]', 'auto', ?)",
        (sid, name, version, f"spark://skills/{name}/{version}",
         json.dumps(lineage), skills.now_iso()),
    )
    db.execute(
        "INSERT INTO skill_candidates (candidate_id, skill_name, parent_version,"
        " source_episode_ids, mutation_op, eval_scores, heldout_gap, verdict,"
        " challenger_skill_id) VALUES (?, ?, ?, '[]', 'split', ?, ?, 'pending', ?)",
        (cid, name, parent_version,
         json.dumps({"held_in": held_in, "held_out": held_out}), gap, sid),
    )
    return cid, version


def _insert_champion(db, name, held_in):
    sid = skills.new_skill_id()
    version = _next_free(db, name)
    db.execute(
        "INSERT INTO skills (skill_id, name, version, role, body_uri, lineage,"
        " status, credit, benchmark_history, promotion_status, created_at)"
        " VALUES (?, ?, ?, NULL, ?, '{}', 'champion', '{}', ?, 'auto', ?)",
        (sid, name, version, f"spark://skills/{name}/{version}",
         json.dumps([{"eval_scores": {"held_in": held_in}}]), skills.now_iso()),
    )
    return version


def _skill_status(db, name, version):
    row = db.conn.execute(
        "SELECT status FROM skills WHERE name = ? AND version = ?", (name, version)
    ).fetchone()
    return row["status"] if row else None


# -- 1. rubric leak ------------------------------------------------------------


def test_mutation_ops_are_exactly_the_nonleaking_set():
    assert gepa.MUTATION_OPS == ("add_guard_step", "drop_step", "reflect")
    assert "add_check_step" not in gepa.MUTATION_OPS
    assert not hasattr(gepa, "_add_check_step")
    src = Path(__file__).resolve().parents[1] / "src" / "spark_mcp"
    offenders = [
        py.name
        for py in src.rglob("*.py")
        if "_add_check_step" in py.read_text(encoding="utf-8")
        or '"add_check_step"' in py.read_text(encoding="utf-8")
    ]
    assert not offenders, f"leaking operator survives in: {offenders}"


def test_guard_steps_carry_no_rubric_or_reference_text():
    fixtures = skills.load_fixtures()
    assert len(fixtures) >= 6
    ind = {"name": "s", "procedure_steps": list(NEUTRAL_STEPS)}
    candidates = [
        gepa._add_guard_step(
            ind, [{"error_class": "DataGap"}, {"error_class": "DataGap"}], random.Random(0)
        )[-1],
        gepa._add_guard_step(ind, None, random.Random(0))[-1],
        gepa.GUARD_STEP_FALLBACK,
    ]
    for step in candidates:
        lowered = step.lower()
        for fixture in fixtures:
            for item in fixture["rubric"]:
                assert item.lower() not in lowered, (step, item)
            for keyword in fixture["reference_keywords"]:
                assert keyword.lower() not in lowered, (step, keyword)


def test_guard_sentence_sanitizes_hostile_error_class():
    fixtures = skills.load_fixtures()
    # A rubric-shaped error class cannot be copied verbatim: whitespace and
    # non-identifier characters are stripped, so no multi-word rubric item
    # survives, and the interpolated token matches the identifier allowlist.
    hostile = "applies explicit transaction cost and slippage assumptions"
    step = gepa._guard_sentence(hostile, 1)
    assert step.startswith("Guard against appliesexplicittransactioncost")
    token = step.removeprefix("Guard against ").split(":")[0]
    assert re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", token), token
    lowered = step.lower()
    for fixture in fixtures:
        for item in fixture["rubric"]:
            assert item.lower() not in lowered, (step, item)
    # Length is capped; a class with no identifier content at all degrades
    # to the generic fallback instead of emitting an empty guard.
    capped = gepa._guard_sentence("D" * 200, 3)
    assert capped.removeprefix("Guard against ").split(":")[0] == "D" * 64
    assert gepa._guard_sentence("!!!", 1) == gepa.GUARD_STEP_FALLBACK
    # The sanitizer is a no-op on genuine classes (pinned template intact).
    assert gepa._guard_sentence("DataGap", 2) == (
        "Guard against DataGap: check data coverage, preconditions, and"
        " inputs before executing this procedure (observed in 2 source"
        " episode(s))."
    )
    # ... including through the public mutation entry point.
    ind = {"name": "s", "procedure_steps": list(NEUTRAL_STEPS)}
    out = gepa._add_guard_step(ind, [{"error_class": hostile}], random.Random(0))
    assert out[-1] == step


def test_mutate_signature_and_outputs_never_gain_rubric():
    params = inspect.signature(gepa._mutate).parameters
    assert "fixtures" not in params and "fixture" not in params
    assert "failure_sketches" in params
    fixtures = skills.load_fixtures()
    ind = {"name": "s", "procedure_steps": list(NEUTRAL_STEPS)}
    for op in gepa.MUTATION_OPS:
        out = gepa._mutate(ind, [{"error_class": "DataGap"}], random.Random(3), op)
        assert out["procedure_steps"]
        blob = " ".join(out["procedure_steps"]).lower()
        for fixture in fixtures:
            for item in fixture["rubric"]:
                # parent is rubric-neutral, so any verbatim rubric is a leak
                if item.lower() not in " ".join(NEUTRAL_STEPS).lower():
                    assert item.lower() not in blob, (op, item)
    assert ind["procedure_steps"] == NEUTRAL_STEPS  # never in-place


def test_reflect_prompt_and_output_carry_no_fixture_rubric():
    fixtures = skills.load_fixtures()
    llm = FakeLLM(
        {"procedure": json.dumps({"steps": list(NEUTRAL_STEPS)})}
    )
    proposal = reflect_mod.reflect_variation(
        parent=list(NEUTRAL_STEPS),
        feedback={"failure_sketches": [{"error_class": "DataGap"}]},
        ancestors_render="",
        rng=random.Random(0),
        llm=llm,
        schema="skill_steps",
    )
    assert proposal.body == NEUTRAL_STEPS
    assert len(llm.calls) == 1
    prompt = llm.calls[0].lower()
    for fixture in fixtures:
        for item in fixture["rubric"]:
            assert item.lower() not in prompt, item
        for keyword in fixture["reference_keywords"]:
            assert keyword.lower() not in prompt, keyword


def test_run_gepa_returned_steps_gain_no_verbatim_rubric():
    fixtures = skills.load_fixtures()
    seeds = [{"name": "s", "procedure_steps": list(NEUTRAL_STEPS), "references": []}]
    result = gepa.run_gepa(
        seeds,
        [f for f in fixtures if f.get("held_in", True)],
        population_size=6,
        generations=2,
        pareto_top_k=3,
        seed=11,
        failure_sketches=[{"error_class": "DataGap"}],
    )
    returned = [result["best"], *result["pareto_front"]]
    for individual in returned:
        blob = " ".join(individual["procedure_steps"]).lower()
        for fixture in fixtures:
            for item in fixture["rubric"]:
                if item.lower() not in " ".join(NEUTRAL_STEPS).lower():
                    assert item.lower() not in blob, item


# -- 2. ties never promote (strict margins, both layers) -----------------------


def test_evolve_tie_on_dsr_margin_only_rejects(db, cfg):
    cfg.research.evolution_dsr_margin = 0.25
    cfg.research.evolution_sr_margin = 0.5
    ep_inc = _registered_episode(db, cfg)
    _set_corrected(db, ep_inc, dsr=0.5, sr=1.0)
    ep_ch = _registered_episode(db, cfg)
    _set_corrected(db, ep_ch, dsr=0.75, sr=2.0)  # DSR exactly at req, SR clear
    decision = promote_mod.decide(db, cfg, ep_ch, ep_inc)
    assert decision.dsr_margin == 0.25 and decision.sr_margin == 1.0
    assert decision.promote is False and "DSR margin" in decision.reason


def test_evolve_tie_on_sr_margin_only_rejects(db, cfg):
    cfg.research.evolution_dsr_margin = 0.25
    cfg.research.evolution_sr_margin = 0.5
    ep_inc = _registered_episode(db, cfg)
    _set_corrected(db, ep_inc, dsr=0.5, sr=1.0)
    ep_ch = _registered_episode(db, cfg)
    _set_corrected(db, ep_ch, dsr=0.875, sr=1.5)  # SR exactly at req, DSR clear
    decision = promote_mod.decide(db, cfg, ep_ch, ep_inc)
    assert decision.dsr_margin == 0.375 and decision.sr_margin == 0.5
    assert decision.promote is False and "SR margin" in decision.reason


def test_evolve_identical_scores_never_promote_but_clear_margin_does(db, cfg):
    ep_inc = _registered_episode(db, cfg)
    _set_corrected(db, ep_inc, dsr=0.96, sr=1.0)
    ep_same = _registered_episode(db, cfg)
    _set_corrected(db, ep_same, dsr=0.96, sr=1.0)
    same = promote_mod.decide(db, cfg, ep_same, ep_inc)
    assert same.dsr_margin == 0.0 and same.sr_margin == 0.0 and same.promote is False
    ep_over = _registered_episode(db, cfg)
    _set_corrected(db, ep_over, dsr=0.99, sr=1.1)
    assert promote_mod.decide(db, cfg, ep_over, ep_inc).promote is True


def test_skill_auto_tie_and_exact_margin_reject_against_incumbent(db, cfg):
    cfg.research.replay_promotion_margin = 0.25
    fixtures = skills.load_fixtures()
    in_ids = [f["task_id"] for f in fixtures if f.get("held_in", True)]
    out_ids = [f["task_id"] for f in fixtures if not f.get("held_in", True)]

    def mk(value):
        return {t: value for t in in_ids}, {t: 0.5 for t in out_ids}

    name = "adv-tie-incumbent"
    champion_version = _insert_champion(db, name, {"t1": 0.625})
    for label, value, expect in (
        ("exact tie", 0.625, "rejected"),
        ("exactly margin above", 0.875, "rejected"),
        ("clearly above margin", 1.0, "promoted"),
    ):
        held_in, held_out = mk(value)
        cid, version = _insert_challenger(
            db, name, held_in, held_out, 0.0, parent_version=champion_version
        )
        out = skills.decide_promotion(db, cfg, name, cid, "auto", label, "gate")
        assert out["decision"] == expect, (label, out)
        if expect == "rejected":
            assert _skill_status(db, name, version) == "rejected"
            assert _skill_status(db, name, champion_version) == "champion"


def test_skill_auto_first_promotion_floor_ties_reject(db, cfg):
    assert cfg.research.replay_promotion_floor == 0.5
    cfg.research.replay_promotion_margin = 0.25
    fixtures = skills.load_fixtures()
    in_ids = [f["task_id"] for f in fixtures if f.get("held_in", True)]
    out_ids = [f["task_id"] for f in fixtures if not f.get("held_in", True)]
    for label, value, expect in (
        ("exactly floor", 0.5, "rejected"),
        ("exactly margin above floor", 0.75, "rejected"),
        ("above margin", 0.875, "promoted"),
    ):
        name = f"adv-tie-floor-{value}"
        cid, _version = _insert_challenger(
            db, name, {t: value for t in in_ids}, {t: 0.5 for t in out_ids}, 0.0
        )
        out = skills.decide_promotion(db, cfg, name, cid, "auto", label, "gate")
        assert out["decision"] == expect, (label, out)


# -- 3. keyword-stuffed empty challenger loses to honest baseline --------------


def test_empty_and_whitespace_steps_score_zero_everywhere():
    fixtures = skills.load_fixtures()
    honest_in, _, _ = _replay_scores(HONEST_STEPS)
    assert _mean(honest_in) > 0.5
    for steps in ([], [""], ["   "]):
        assert all(gepa.fixture_score(steps, f) == 0.0 for f in fixtures)
        population = [{"name": "empty", "procedure_steps": steps, "references": []}]
        assert gepa.score_population(population, fixtures) == [0.0]


def test_scorer_ignores_keywords_stuffed_outside_steps():
    fixtures = skills.load_fixtures()
    held_in, held_out, _gap = _replay_scores(HONEST_STEPS)
    assert _mean(held_in) > 0.5 and _mean(held_out) > 0.5
    all_words = _stuffed_steps(fixtures) + HONEST_STEPS
    stuffed_empty = {
        "name": " ".join(all_words),  # keywords in the name
        "procedure_steps": [],  # ... but no procedure at all
        "references": all_words,  # ... and stuffed references
    }
    honest = {"name": "honest", "procedure_steps": HONEST_STEPS, "references": []}
    scores = gepa.score_population([stuffed_empty, honest], fixtures)
    assert scores[0] == 0.0 and scores[1] > scores[0]


def test_keyword_soup_scores_low_at_scorer_level_honest_unchanged():
    """Scorer-level defence: the held-in keyword soup that once scored
    0.9375 now scores 0.0 per fixture (no substantive step), while the
    honest baseline keeps its exact pre-defence scores."""
    fixtures = skills.load_fixtures()
    stuffed = _stuffed_steps(fixtures)
    s_in, s_out, _ = _replay_scores(stuffed)
    assert all(v == 0.0 for v in s_in.values()), s_in
    assert all(v == 0.0 for v in s_out.values()), s_out
    assert _mean(s_in) == 0.0
    h_in, h_out, h_gap = _replay_scores(HONEST_STEPS)
    assert _mean(h_in) == pytest.approx(0.96875)
    assert _mean(h_out) == pytest.approx(1.0)
    assert h_gap == pytest.approx(0.03125)


def test_substance_filter_drops_keyword_only_and_keeps_legitimate():
    fixtures = skills.load_fixtures()
    assert all(gepa._is_substantive_step(s) for s in HONEST_STEPS)
    assert all(gepa._is_substantive_step(s) for s in NEUTRAL_STEPS)
    guard = gepa._guard_sentence("DataGap", 2)
    assert gepa._is_substantive_step(guard)
    assert gepa._is_substantive_step(gepa.GUARD_STEP_FALLBACK)
    # Keyword-only steps carry the rubric words but no instruction verb or
    # structure: zero coverage on every fixture despite the words.
    soup_step = _stuffed_steps(fixtures)[0]
    assert "sharpe" in soup_step and "turnover" in soup_step  # words present
    assert not gepa._is_substantive_step(soup_step)
    for steps in (
        [soup_step],
        ["momentum turnover costs sharpe"],
        ["Check"],  # verb but no complement
        ["a b"],
        [""],
        ["   "],
    ):
        assert all(gepa.fixture_score(steps, f) == 0.0 for f in fixtures), steps
    # A substantive step keeps its exact coverage: filtering never edits a
    # kept step, it only drops insubstantial ones.
    mixed = [soup_step, HONEST_STEPS[1]]
    for fixture in fixtures:
        assert gepa.fixture_score(mixed, fixture) == gepa.fixture_score(
            [HONEST_STEPS[1]], fixture
        )


def test_adaptive_soup_variants_score_zero_honest_unchanged():
    """Adaptive stuffing must not revive the soup: a leading instruction
    verb, splitting the word list across steps, or appending numbers all
    still score 0.0, and the honest baseline keeps its pinned scores."""
    fixtures = skills.load_fixtures()
    words = _stuffed_steps(fixtures)[0].split()
    variants = (
        ["Check " + " ".join(words)],  # verb-led soup
        ["Check " + " ".join(words[i::4]) for i in range(4)],  # split soup
        [" ".join(words) + " 20 60 120"],  # numbers appended
    )
    for steps in variants:
        assert all(not gepa._is_substantive_step(s) for s in steps), steps
        assert all(gepa.fixture_score(steps, f) == 0.0 for f in fixtures), steps
    h_in, h_out, _ = _replay_scores(HONEST_STEPS)
    assert _mean(h_in) == pytest.approx(0.96875)
    assert _mean(h_out) == pytest.approx(1.0)


def test_keyword_soup_and_empty_lose_to_honest_at_gate(db, cfg):
    fixtures = skills.load_fixtures()
    stuffed = _stuffed_steps(fixtures)
    s_in, s_out, s_gap = _replay_scores(stuffed)
    h_in, h_out, h_gap = _replay_scores(HONEST_STEPS)
    e_in, e_out, e_gap = _replay_scores([])
    # Preconditions: soup is stopped at the scorer (0.0 everywhere, gap 0 —
    # the gate never even sees an inflated mean); honest clears floor +
    # margin with a quiet gap; empty scores nothing anywhere.
    floor = cfg.research.replay_promotion_floor
    margin = cfg.research.replay_promotion_margin
    alarm = cfg.research.replay_heldout_gap_alarm
    assert _mean(s_in) == 0.0 and s_gap == 0.0
    assert _mean(h_in) - floor > margin and h_gap <= alarm
    assert _mean(e_in) == 0.0

    cases = (
        ("adv-gate-honest", HONEST_STEPS, h_in, h_out, h_gap, "promoted"),
        ("adv-gate-stuffed", stuffed, s_in, s_out, s_gap, "rejected"),
        ("adv-gate-empty", [], e_in, e_out, e_gap, "rejected"),
    )
    for name, steps, held_in, held_out, gap, expect in cases:
        cid, version = _insert_challenger(db, name, held_in, held_out, gap, steps=steps)
        out = skills.decide_promotion(db, cfg, name, cid, "auto", "adversarial", "gate")
        assert out["decision"] == expect, (name, out)
        assert _skill_status(db, name, version) == (
            "champion" if expect == "promoted" else "rejected"
        )


def test_keyword_soup_cannot_displace_honest_incumbent(db, cfg):
    fixtures = skills.load_fixtures()
    stuffed = _stuffed_steps(fixtures)
    s_in, s_out, s_gap = _replay_scores(stuffed)
    h_in, _h_out, _h_gap = _replay_scores(HONEST_STEPS)
    assert _mean(s_in) < _mean(h_in)  # soup does not even beat honest held-in

    name = "adv-soup-vs-incumbent"
    champion_version = _insert_champion(db, name, h_in)
    cid, version = _insert_challenger(
        db, name, s_in, s_out, s_gap, parent_version=champion_version, steps=stuffed
    )
    out = skills.decide_promotion(db, cfg, name, cid, "auto", "soup attack", "gate")
    assert out["decision"] == "rejected"
    assert _skill_status(db, name, version) == "rejected"
    assert _skill_status(db, name, champion_version) == "champion"


# -- 4. min-fixtures guard -------------------------------------------------------


def test_promotion_refuses_cherry_picked_subset(db, cfg):
    fixtures = skills.load_fixtures()
    one_in = {fixtures[0]["task_id"]: 1.0}  # single cherry-picked fixture
    cid, _v = _insert_challenger(db, "adv-subset", one_in, {}, 0.0)
    with pytest.raises(ToolError, match="FULL replay-suite"):
        skills.decide_promotion(db, cfg, "adv-subset", cid, "promoted", "cherry", "gate")
    # auto would promote on the subset mean, so it must hit the same guard
    cid2, _v2 = _insert_challenger(db, "adv-subset-auto", one_in, {}, 0.0)
    with pytest.raises(ToolError, match="FULL replay-suite"):
        skills.decide_promotion(db, cfg, "adv-subset-auto", cid2, "auto", "cherry", "gate")
    row = db.conn.execute(
        "SELECT verdict FROM skill_candidates WHERE candidate_id = ?", (cid2,)
    ).fetchone()
    assert row["verdict"] == "pending"  # guard fires before any decision write


def test_promotion_with_no_fixture_evidence_auto_rejects(db, cfg):
    cid, version = _insert_challenger(db, "adv-no-evidence", {}, {}, 0.0)
    out = skills.decide_promotion(db, cfg, "adv-no-evidence", cid, "auto", "empty", "gate")
    assert out["decision"] == "rejected"
    assert _skill_status(db, name="adv-no-evidence", version=version) == "rejected"


def test_no_fixtures_means_no_positive_evolution_score():
    seeds = [{"name": "s", "procedure_steps": list(NEUTRAL_STEPS), "references": []}]
    assert gepa.score_population(seeds, []) == [0.0]
    result = gepa.run_gepa(
        seeds, [], population_size=3, generations=1, pareto_top_k=2, seed=1
    )
    assert result["best"]["held_in_score"] == 0.0


def test_fixture_suite_floor_and_loader_guards(db, cfg, tmp_path):
    fixtures = skills.load_fixtures()
    assert len(fixtures) == 6
    assert cfg.research.replay_min_fixtures == 6
    assert _fixture_growth_gaps(cfg) == []  # at floor: quiet
    cfg.research.replay_min_fixtures = len(fixtures) + 1
    gaps = _fixture_growth_gaps(cfg)
    assert len(gaps) == 1 and gaps[0]["kind"] == "fixture_suite_small"
    assert gaps[0]["evidence"] == {
        "fixture_count": len(fixtures),
        "min_fixtures": len(fixtures) + 1,
    }
    empty = tmp_path / "empty.yaml"
    empty.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        skills.load_fixtures(empty)
    incomplete = tmp_path / "incomplete.yaml"
    incomplete.write_text("- {task_id: x, prompt: p}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        skills.load_fixtures(incomplete)


# -- 5. reflect cannot alter computed fitness -----------------------------------


def _gated_episode(db, cfg, seed=11):
    spec = {
        "hypothesis_id": None,  # filled below
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
    hyp = ledger.propose_hypothesis(db, cfg, **PROPOSE)
    spec["hypothesis_id"] = hyp["hypothesis_id"]
    reg = ledger.register_hypothesis(db, cfg, hyp["hypothesis_id"], {"signal": "mom"}, seed)
    provider = SimulatedProvider(cfg.artifacts_dir())
    gates.backtest_run(db, cfg, provider, reg["episode_id"], spec)
    gates.significance_gate(db, cfg, reg["episode_id"])
    return reg["episode_id"]


def test_reflect_drops_fitness_injection_keys():
    parent = {
        "research_family": "toy_momentum",
        "hypothesis_type": "predictive",
        "mechanism": "12-1 momentum in toy universe",
        "expected_direction": "positive",
        "horizon": "21d",
        "scope": "cross-section",
        "falsifier": "IC <= 0 net of costs",
        "predicted_ic": 0.03,
        "predicted_sigma": 0.01,
        "universe": "toy",
        "idea": "momentum persists",
        "dag": None,
        "adjustment_set": None,
    }
    evil = {
        **parent,
        "sr": 999.0,
        "dsr": 1.0,
        "dsr_prob": 1.0,
        "p_value": 0.0,
        "M_family": 0,
        "fitness": 999.0,
        "corrected_metrics": {"sr": 999.0, "dsr": 1.0, "pass": True},
    }
    llm = FakeLLM({"successor": json.dumps(evil)})
    proposal = reflect_mod.reflect_variation(
        parent=dict(parent), feedback=None, ancestors_render="",
        rng=random.Random(0), llm=llm, schema="hypothesis",
    )
    assert proposal.fallback_used is False
    for key in ("sr", "dsr", "dsr_prob", "p_value", "M_family", "fitness",
                "corrected_metrics"):
        assert key not in proposal.body, key
    # steps schema likewise returns steps only, never a fitness mapping
    llm2 = FakeLLM({"procedure": json.dumps({"sr": 999.0, "steps": ["Do the work."]})})
    steps_proposal = reflect_mod.reflect_variation(
        parent=list(NEUTRAL_STEPS), feedback=None, ancestors_render="",
        rng=random.Random(0), llm=llm2, schema="skill_steps",
    )
    assert steps_proposal.body == ["Do the work."]


def test_reflect_has_no_fitness_channel():
    params = inspect.signature(reflect_mod.reflect_variation).parameters
    assert set(params) == {"parent", "feedback", "ancestors_render", "rng", "llm", "schema"}
    source = inspect.getsource(reflect_mod)
    assert "fitness_from_episode" not in source
    assert "corrected_metrics" not in source
    # Behavioral invariant (not a whole-file "fitness" substring pin, which
    # breaks on any comment mentioning fitness): reflect's own params and
    # rendered prompts carry no fitness values. Captured prompts over
    # fitness-free inputs contain no fitness field at all, on both schemas.
    fitness_keys = ("sr", "dsr", "dsr_prob", "p_value", "M_family",
                    "fitness", "corrected_metrics")
    clean_parent = {
        "research_family": "toy_momentum",
        "hypothesis_type": "predictive",
        "mechanism": "12-1 momentum in toy universe",
        "expected_direction": "positive",
        "horizon": "21d",
        "scope": "cross-section",
        "falsifier": "IC <= 0 net of costs",
        "predicted_ic": 0.03,
        "predicted_sigma": 0.01,
        "universe": "toy",
    }
    llm = FakeLLM({"successor": json.dumps(clean_parent)})
    reflect_mod.reflect_variation(
        parent=dict(clean_parent), feedback={"note": "plain feedback"},
        ancestors_render="", rng=random.Random(0), llm=llm, schema="hypothesis",
    )
    assert len(llm.calls) == 1
    for key in fitness_keys:
        assert f'"{key}"' not in llm.calls[0], key
    llm2 = FakeLLM({"procedure": json.dumps({"steps": list(NEUTRAL_STEPS)})})
    reflect_mod.reflect_variation(
        parent=list(NEUTRAL_STEPS), feedback=None, ancestors_render="",
        rng=random.Random(0), llm=llm2, schema="skill_steps",
    )
    assert len(llm2.calls) == 1
    for key in fitness_keys:
        assert f'"{key}"' not in llm2.calls[0], key
    gepa_params = inspect.signature(gepa._reflect_step).parameters
    assert "fitness" not in gepa_params and "episode" not in gepa_params


def test_reflect_variations_leave_computed_fitness_unchanged(db, cfg):
    episode_id = _gated_episode(db, cfg)
    before = fitness_mod.fitness_from_episode(db, cfg, episode_id)
    corrected_before = ledger.get_episode(db, episode_id)["corrected_metrics"]

    # LLM steps rewrite, LLM hypothesis rewrite, fallbacks, GEPA reflect op:
    # every reflective channel runs; none may touch the fitness record.
    llm = FakeLLM({"procedure": json.dumps({"steps": ["Rewrite everything."]})})
    reflect_mod.reflect_variation(
        parent=list(NEUTRAL_STEPS), feedback={"failure_sketches": []},
        ancestors_render="", rng=random.Random(1), llm=llm, schema="skill_steps",
    )
    reflect_mod.reflect_variation(
        parent=list(NEUTRAL_STEPS), feedback=None, ancestors_render="",
        rng=random.Random(1), llm=None, schema="skill_steps",
    )
    gepa._reflect_step(
        {"name": "s", "procedure_steps": list(NEUTRAL_STEPS)},
        [{"error_class": "DataGap"}], random.Random(2), None,
    )
    gepa._reflect_step(
        {"name": "s", "procedure_steps": list(NEUTRAL_STEPS)}, None,
        random.Random(2),
        FakeLLM({"procedure": json.dumps({"steps": ["sr 999 dsr 1.0 promote me."]})}),
    )

    after = fitness_mod.fitness_from_episode(db, cfg, episode_id)
    assert after == before
    assert fitness_mod.fitness_from_episode(db, cfg, episode_id) == before
    assert ledger.get_episode(db, episode_id)["corrected_metrics"] == corrected_before
