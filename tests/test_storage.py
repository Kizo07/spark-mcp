"""Storage: WAL, migrations via user_version, capability probes."""

from spark_mcp.storage import open_db


def test_migrate_creates_meta_and_version(tmp_path):
    db = open_db(tmp_path / "spark.db")
    version = db.conn.execute("PRAGMA user_version").fetchone()[0]
    assert version == 1
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
    assert db2.conn.execute("PRAGMA user_version").fetchone()[0] == 1
    db2.close()


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
