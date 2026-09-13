"""SQLite storage: WAL, busy_timeout, user_version migrations, sqlite-vec.

Storage discipline (plan §5.0 / M6): WAL on, busy_timeout 5000, all mutations
in explicit transactions, idempotency via UNIQUE constraints, migrations as
versioned SQL applied through PRAGMA user_version.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1

# Versioned migrations. Key = user_version the schema moves TO.
MIGRATIONS: dict[int, str] = {
    1: """
    -- Phase 0: meta table only. Domain tables land in Phases 1-3
    -- (see spark_v3_master_plan.md §5.1).
    CREATE TABLE IF NOT EXISTS spark_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    INSERT OR REPLACE INTO spark_meta (key, value) VALUES ('schema_name', 'spark');
    """,
}


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            # check_same_thread=False: MCP servers execute tool handlers on a
            # worker thread while the Database object lives in the main thread.
            # All access is serialized by busy_timeout + explicit transactions.
            self._conn = sqlite3.connect(str(self.path), timeout=10, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def migrate(self) -> int:
        """Apply pending migrations; returns the schema version reached."""
        with self.conn:  # single transaction for the whole migration run
            current = self.conn.execute("PRAGMA user_version").fetchone()[0]
            for version in sorted(MIGRATIONS):
                if version > current:
                    self.conn.executescript(MIGRATIONS[version])
                    self.conn.execute(f"PRAGMA user_version={version}")
            return self.conn.execute("PRAGMA user_version").fetchone()[0]

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self.conn:
            return self.conn.execute(sql, params)

    # -- capability probes (doctor / N14) -----------------------------------

    def fts5_available(self) -> bool:
        try:
            self.conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS _probe_fts USING fts5(x)"
            )
            self.conn.execute("DROP TABLE IF EXISTS _probe_fts")
            return True
        except sqlite3.Error:
            return False

    def sqlite_vec_available(self) -> bool:
        try:
            import sqlite_vec

            sqlite_vec.load(self.conn)
            self.conn.execute("SELECT vec_version()").fetchone()
            return True
        except Exception:
            return False


def open_db(path: str | Path) -> Database:
    db = Database(path)
    db.migrate()
    return db
