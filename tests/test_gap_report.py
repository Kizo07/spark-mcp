"""Gap-report consumption + the Phase-7 report_write HTML decision.

suggest_next (plan §6.1 tool 6, §13/G5): consuming a consolidation gap report
by id boosts the named families' utility and records the report id on the
trajectory + result. report_write (plan §6.3 tool 21, N5): HTML is served
by the built-in renderer; a config pinning
research.report_html_renderer != 'builtin' still rejects format='html'
with a message citing the decision record."""

from __future__ import annotations

import asyncio
import json

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError as ClientToolError

from spark_mcp import consolidate as consolidate_mod
from spark_mcp import ledger
from spark_mcp.config import Config
from spark_mcp.consolidate import report_path
from spark_mcp.explore import suggest_next
from spark_mcp.server import create_server
from spark_mcp.storage import open_db


def _write_report(cfg, gaps, report_id="cons_gap_boost"):
    report = {
        "report_id": report_id,
        "schema": "spark.consolidation_report/v1",
        "gap_count": len(gaps),
        "gaps": gaps,
    }
    path = report_path(cfg, report_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report), encoding="utf-8")
    return report_id


def _family_gap(family):
    return {
        "gap_id": f"gap_{family}",
        "kind": "empty_family",
        "family": family,
        "severity": "high",
        "evidence": {"occupied_cells": 0},
        "suggested_action": f"occupy {family}",
    }


def test_suggest_next_boosts_gap_families(db, cfg):
    report_id = _write_report(cfg, [_family_gap("nlp")])
    base = suggest_next(db, cfg, n=8, as_of="2024-06-01T00:00:00Z", seed=11)
    boosted = suggest_next(
        db, cfg, n=8, as_of="2024-06-01T00:00:00Z", seed=11, gap_report_id=report_id
    )
    assert boosted["gap_report_id"] == report_id
    assert boosted["gap_boosted_families"] == ["nlp"]
    # multiplicative 1.5x on the Thompson x emptiness utility: the boosted
    # family's softmax share strictly increases at the same seed, and every
    # non-boosted family's share strictly decreases (renormalized softmax)
    p_base = base["family_selection_probabilities"]["nlp"]
    p_boosted = boosted["family_selection_probabilities"]["nlp"]
    assert p_boosted > p_base
    for fam, p in base["family_selection_probabilities"].items():
        if fam == "nlp":
            continue
        assert boosted["family_selection_probabilities"][fam] < p
    # the boosted suggestion's reported utility reflects the boost
    sugg = next(s for s in boosted["suggestions"] if s["family"] == "nlp")
    base_sugg = next((s for s in base["suggestions"] if s["family"] == "nlp"), None)
    if base_sugg is not None:
        assert sugg["utility"] == pytest.approx(base_sugg["utility"] * 1.5)


def test_suggest_next_records_report_on_trajectory(db, cfg):
    report_id = _write_report(cfg, [_family_gap("macro")])
    suggest_next(db, cfg, n=1, seed=3, gap_report_id=report_id)
    row = db.conn.execute(
        "SELECT payload FROM trajectory_events WHERE run_id = 'suggest_next'"
        " ORDER BY seq DESC LIMIT 1"
    ).fetchone()
    payload = json.loads(row["payload"])
    assert payload["gap_report_id"] == report_id
    assert payload["gap_boosted_families"] == ["macro"]
    assert payload["gap_boost_factor"] == 1.5


def test_suggest_next_real_consolidation_report_boosts_its_families(db, cfg):
    report = consolidate_mod.consolidate(db, cfg)  # fresh db: 8 empty families
    boosted = suggest_next(db, cfg, n=1, seed=5, gap_report_id=report["report_id"])
    report_families = {g["family"] for g in report["gaps"] if g.get("family")}
    assert set(boosted["gap_boosted_families"]) == report_families


def test_suggest_next_no_report_default_path_unchanged(db, cfg):
    out = suggest_next(db, cfg, n=2, seed=9)
    assert out["gap_report_id"] is None
    assert out["gap_boosted_families"] == []


# ---------------------------------------------------------------------------
# report_write Phase-7 HTML decision (N5)


def _server(tmp_path):
    return create_server(
        Config(
            storage={
                "db_path": f"{tmp_path}/spark.db",
                "artifacts_dir": f"{tmp_path}/art",
            }
        ),
        seeds=False,
    )


def test_report_write_rejection_names_the_decision_record(tmp_path):
    deferred = Config(
        storage={
            "db_path": f"{tmp_path}/spark.db",
            "artifacts_dir": f"{tmp_path}/art",
        },
        research={"report_html_renderer": "deferred"},
    )

    async def run():
        async with Client(create_server(deferred, seeds=False)) as client:
            with pytest.raises(ClientToolError) as excinfo:
                await client.call_tool(
                    "report_write", {"episode_id": "ep_x", "format": "html"}
                )
            message = str(excinfo.value)
            assert "report_html_renderer" in message
            assert "deferred" in message.lower()
            assert "builtin" in message  # names the enabling value
            assert "N5" in message  # plan §6.3 tool 21 N5

    asyncio.run(run())


def test_report_write_md_still_works(tmp_path):
    cfg = Config(
        storage={"db_path": f"{tmp_path}/spark.db", "artifacts_dir": f"{tmp_path}/art"}
    )
    db = open_db(cfg.db_path())
    hyp = ledger.propose_hypothesis(
        db, cfg, research_family="quant", hypothesis_type="predictive",
        mechanism="m", expected_direction="positive", horizon="20d", scope="s",
        falsifier="f", predicted_ic=0.03, predicted_sigma=0.01, universe="toy",
        idea="i",
    )
    reg = ledger.register_hypothesis(db, cfg, hyp["hypothesis_id"], {"t": 1}, 5)
    db.close()

    async def run():
        async with Client(_server(tmp_path)) as client:
            out = json.loads(
                (await client.call_tool("report_write", {"episode_id": reg["episode_id"]}))
                .content[0]
                .text
            )
            assert out["format"] == "md"
            assert "Spark ledger report" in out["markdown"]

    asyncio.run(run())
