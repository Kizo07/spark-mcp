"""Storage: WAL, migrations via user_version, capability probes."""

from spark_mcp.storage import open_db


def test_migrate_creates_meta_and_version(tmp_path):
    db = open_db(tmp_path / "spark.db")
    version = db.conn.execute("PRAGMA user_version").fetchone()[0]
    assert version == 4
    name = db.conn.execute(
        "SELECT value FROM spark_meta WHERE key='schema_name'"
    ).fetchone()[0]
    assert name == "spark"
    db.close()


def test_migrations_are_idempotent(tmp_path):
    path = tmp_path / "spark.db"
    db = open_db(path)
    db.close()
    db2 = open_db(path)  # second open: no pending migrations, no error
    assert db2.conn.execute("PRAGMA user_version").fetchone()[0] == 4
    db2.close()


def test_phase1_tables_exist(tmp_path):
    db = open_db(tmp_path / "spark.db")
    tables = {
        r[0]
        for r in db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    for t in (
        "spark_meta",
        "hypotheses",
        "episodes",
        "trials",
        "beliefs",
        "tombstones",
        "skills",
        "trajectory_events",
        "tool_effects",
        "beliefs_fts",
        "tombstones_fts",
        "graph_nodes",
        "graph_edges",
        "graph_nodes_fts",
        "score_cache",
        "exploration_state",
        "spend_ledger",
    ):
        assert t in tables, f"missing table {t}"
    db.close()


def test_vec_embeddings_table_created_on_first_embed(tmp_path):
    db = open_db(tmp_path / "spark.db")
    assert db.vec_available is True  # sqlite-vec present in the dev venv
    row = db.conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = 'vec_embeddings'"
    ).fetchone()
    assert row is None  # lazy: created on first embed write
    db.ensure_vec_embeddings(64)
    row = db.conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = 'vec_embeddings'"
    ).fetchone()
    assert row is not None
    db.close()


def test_wal_and_busy_timeout(tmp_path):
    db = open_db(tmp_path / "spark.db")
    mode = db.conn.execute("PRAGMA journal_mode").fetchone()[0]
    timeout = db.conn.execute("PRAGMA busy_timeout").fetchone()[0]
    assert mode == "wal"
    assert timeout == 5000
    db.close()


def test_fts5_probe(tmp_path):
    db = open_db(tmp_path / "spark.db")
    assert db.fts5_available() is True
    db.close()


def test_execute_does_not_commit_inside_outer_transaction(tmp_path):
    """Phase 1 review nit, closed: execute() inside an open transaction must
    not commit partial work when it returns."""
    import pytest

    db = open_db(tmp_path / "spark.db")
    with pytest.raises(RuntimeError):
        with db.transaction():
            db.execute("INSERT INTO spark_meta (key, value) VALUES ('outer', '1')")
            db.execute("INSERT INTO spark_meta (key, value) VALUES ('inner', '2')")
            raise RuntimeError("boom")
    rows = db.conn.execute(
        "SELECT key FROM spark_meta WHERE key IN ('outer', 'inner')"
    ).fetchall()
    assert rows == []  # both rolled back: no early commit from inner execute()
    db.close()


def test_transaction_nests_via_savepoints(tmp_path):
    db = open_db(tmp_path / "spark.db")
    with db.transaction():
        db.execute("INSERT INTO spark_meta (key, value) VALUES ('outer', '1')")
        try:
            with db.transaction():
                db.execute("INSERT INTO spark_meta (key, value) VALUES ('inner', '2')")
                raise ValueError("nested failure")
        except ValueError:
            pass
        db.execute("INSERT INTO spark_meta (key, value) VALUES ('outer2', '3')")
    keys = {r[0] for r in db.conn.execute("SELECT key FROM spark_meta").fetchall()}
    assert {"outer", "outer2"} <= keys and "inner" not in keys
    db.close()
