"""Embed backfill (Fix F): idempotent `seed backfill` over graph-node +
hypothesis subjects lacking vec_embeddings rows; degraded no-op."""

from typer.testing import CliRunner

from spark_mcp import graph, ledger
from spark_mcp.cli import app
from spark_mcp.storage import open_db

HYP = dict(
    research_family="momentum",
    hypothesis_type="descriptive",
    mechanism="volatility clustering persists after earnings",
    expected_direction="positive",
    horizon="21d",
    scope="us-large-cap",
    falsifier="ic turns negative for two consecutive quarters",
    predicted_ic=0.05,
    predicted_sigma=0.02,
    universe="sp500",
)


def _unembedded_db(db, cfg):
    graph.add_node(db, "technique", "Backfill Tech A", summary="alpha signal")
    graph.add_node(db, "technique", "Backfill Tech B", summary="beta signal")
    graph.add_node(db, "concept", "Backfill Concept", summary="gamma idea")
    ledger.propose_hypothesis(db, cfg, **HYP)
    return db


def _vec_count(db):
    if not db.vec_table_exists():
        return 0
    return db.conn.execute("SELECT COUNT(*) FROM vec_embeddings").fetchone()[0]


def test_backfill_first_run_embeds_all_missing(db, cfg, embedder):
    _unembedded_db(db, cfg)
    assert _vec_count(db) == 0
    stats = graph.backfill_embeddings(db, embedder)
    assert stats["degraded"] is False
    assert stats["embedded"] == 4
    assert stats["embedded_graph_nodes"] == 3
    assert stats["embedded_hypotheses"] == 1
    assert _vec_count(db) == 4


def test_backfill_second_run_embeds_zero(db, cfg, embedder):
    _unembedded_db(db, cfg)
    first = graph.backfill_embeddings(db, embedder)
    assert first["embedded"] > 0
    rows_before = _vec_count(db)
    second = graph.backfill_embeddings(db, embedder)
    assert second["embedded"] == 0
    assert second["skipped_existing"] == 4
    assert _vec_count(db) == rows_before


def test_backfill_degraded_writes_nothing(tmp_path):
    db = open_db(tmp_path / "degraded.db")
    graph.add_node(db, "technique", "Lonely Tech", summary="no embedder")
    stats = graph.backfill_embeddings(db, None)
    assert stats["degraded"] is True
    assert stats["embedded"] == 0
    assert "degraded" in stats["reason"]
    assert not db.vec_table_exists(), "degraded backfill must write nothing"
    db.close()


def test_cli_backfill_degraded_exits_nonzero(tmp_path):
    db_path = tmp_path / "cli.db"
    db = open_db(db_path)
    graph.add_node(db, "technique", "CLI Tech", summary="no embedder")
    db.close()
    cfg_path = tmp_path / "spark.yaml"
    cfg_path.write_text(
        f"storage:\n  db_path: {db_path}\n  artifacts_dir: {tmp_path}/art\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["seed", "backfill", "--config", str(cfg_path)])
    assert result.exit_code == 1
    assert "degraded" in result.output
    db = open_db(db_path)
    assert not db.vec_table_exists(), "degraded CLI backfill must write nothing"
    db.close()
