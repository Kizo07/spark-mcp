"""SQLite storage: WAL, busy_timeout, user_version migrations, sqlite-vec.

Storage discipline (plan §5.0 / M6): WAL on, busy_timeout 5000, all mutations
in explicit transactions, idempotency via UNIQUE constraints, migrations as
versioned SQL applied through PRAGMA user_version.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 2

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
    2: """
    -- Phase 1: ledger & memory (plan §5.1). skills is created now deliberately
    -- so episodes record skill_versions from day one (M13); its lifecycle
    -- machinery arrives in Phase 6.

    -- Hypotheses: the idea DAG. One hypothesis : N trial episodes. (C3)
    CREATE TABLE IF NOT EXISTS hypotheses (
        hypothesis_id TEXT PRIMARY KEY,
        parent_hypothesis_ids TEXT NOT NULL DEFAULT '[]',  -- JSON array; idea lineage
        research_family TEXT NOT NULL,
        hypothesis_type TEXT NOT NULL,      -- predictive|causal|descriptive|
                                            -- portfolio|risk|execution
        mechanism TEXT NOT NULL,
        expected_direction TEXT NOT NULL,
        horizon TEXT NOT NULL,
        scope TEXT NOT NULL,
        dag TEXT,                           -- JSON; REQUIRED iff hypothesis_type='causal'
        adjustment_set TEXT,                -- REQUIRED iff causal
        falsifier TEXT NOT NULL,
        predicted_ic REAL NOT NULL,         -- (C8) required at propose time
        predicted_sigma REAL NOT NULL,      -- (C8) floor applied at scoring time
        universe TEXT NOT NULL,
        as_of TEXT NOT NULL,
        data_through TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'proposed',  -- proposed|registered|testing|retired
        client_id TEXT UNIQUE,
        created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_hypotheses_family ON hypotheses(research_family);

    -- Episodes: trial records. append-only. (C3)
    CREATE TABLE IF NOT EXISTS episodes (
        episode_id TEXT PRIMARY KEY,
        hypothesis_id TEXT NOT NULL REFERENCES hypotheses,
        run_kind TEXT NOT NULL DEFAULT 'primary',   -- primary|placebo|shuffled|reconciled
        research_family TEXT NOT NULL,              -- denormalized for M_family counts
        agent_assembly_fingerprint TEXT,
        fingerprint_spec_version TEXT,              -- (C9) QuantFlow change list v1
        skill_versions TEXT NOT NULL DEFAULT '{}',  -- JSON; recorded from day one (M13)
        protocol_hash TEXT,
        spec_hash TEXT,
        seed INTEGER,
        tool_calls INTEGER,
        datasets TEXT,
        params TEXT,
        metrics TEXT,
        corrected_metrics TEXT,
        risk_exposures TEXT,
        costs TEXT,                          -- required-or-estimated (§14)
        reward REAL,                         -- §6.3 tool 15: non-null from Phase 1 (M13)
        outcome TEXT,
        failure_reason TEXT,
        status TEXT NOT NULL DEFAULT 'registered',  -- registered|testing|failed|
                                                    -- inconclusive|validated|promoted|
                                                    -- tombstoned
        tombstone_check TEXT,        -- JSON {status: hit|miss|ack, matched_id, rationale}
        reconciled INTEGER NOT NULL DEFAULT 0,   -- (C4) out-of-band reconciliation flag
        as_of TEXT NOT NULL,
        data_through TEXT NOT NULL,
        client_id TEXT UNIQUE,
        created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_episodes_hypothesis ON episodes(hypothesis_id);
    CREATE INDEX IF NOT EXISTS idx_episodes_status ON episodes(status);
    CREATE INDEX IF NOT EXISTS idx_episodes_family ON episodes(research_family);

    -- Trials: the countable unit for M. Inserted PENDING before dispatch. (C2/C4)
    CREATE TABLE IF NOT EXISTS trials (
        trial_id TEXT PRIMARY KEY,
        episode_id TEXT NOT NULL REFERENCES episodes,
        idem_key TEXT NOT NULL,        -- sha1(episode_id|spec_hash|seed)
        family TEXT NOT NULL,
        source TEXT NOT NULL,          -- spark|reconciled
        status TEXT NOT NULL,          -- pending|completed|failed
        provider TEXT NOT NULL,        -- alpha_engine|simulated|...
        created_at TEXT NOT NULL,
        completed_at TEXT,
        UNIQUE(episode_id, idem_key)
    );
    CREATE INDEX IF NOT EXISTS idx_trials_episode ON trials(episode_id);
    CREATE INDEX IF NOT EXISTS idx_trials_family_status ON trials(family, status);

    -- Beliefs: semantic memory. Promotion via gate + §12.3 lifecycle. (M9)
    CREATE TABLE IF NOT EXISTS beliefs (
        belief_id TEXT PRIMARY KEY,
        claim TEXT NOT NULL,
        confidence REAL NOT NULL,            -- raw; §12.3 decay writes `decay`
        decay REAL NOT NULL DEFAULT 1.0,     -- multiplicative staleness factor
        direction TEXT,
        horizon TEXT,
        regime_scope TEXT,
        supporting_runs TEXT NOT NULL DEFAULT '[]',   -- JSON (N6)
        contradicting_runs TEXT NOT NULL DEFAULT '[]',
        scope TEXT,
        last_validated TEXT,
        valid_from TEXT NOT NULL,
        valid_to TEXT,                       -- invalidation, not deletion
        as_of TEXT NOT NULL,
        data_through TEXT NOT NULL,
        promotion_status TEXT NOT NULL DEFAULT 'auto', -- auto|pending_human|approved|
                                                       -- rejected
        created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_beliefs_validity ON beliefs(valid_from, valid_to);

    -- Negative knowledge (§12.2). Tombstones are written by log_outcome
    -- (failure outcomes), memory_promote contradiction findings, consolidate.
    CREATE TABLE IF NOT EXISTS tombstones (
        tombstone_id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        summary TEXT NOT NULL,               -- FTS5-indexed
        match_keys TEXT NOT NULL,            -- JSON: keywords + mech_tags + embedding ref
        evidence_episode_ids TEXT NOT NULL DEFAULT '[]',
        as_of TEXT NOT NULL,
        supersedes TEXT,
        created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_tombstones_as_of ON tombstones(as_of);

    -- Procedural memory (M13: created P1 to record skill_versions on
    -- episodes; lifecycle machinery arrives Phase 6).
    CREATE TABLE IF NOT EXISTS skills (
        skill_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        version INTEGER NOT NULL,
        role TEXT,                           -- quant|ml|nlp|risk|lead
        body_uri TEXT NOT NULL,
        lineage TEXT NOT NULL DEFAULT '{}',
        status TEXT NOT NULL DEFAULT 'draft', -- draft|challenger|champion|rejected|
                                              -- retired|rollback_target
        credit TEXT NOT NULL DEFAULT '{}',
        benchmark_history TEXT NOT NULL DEFAULT '[]',
        promotion_status TEXT NOT NULL DEFAULT 'auto',
        created_at TEXT NOT NULL,
        UNIQUE(name, version)
    );

    -- Tier-1 trajectory (C10): Spark's own tool-call/observation/gate events.
    CREATE TABLE IF NOT EXISTS trajectory_events (
        seq INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        ts TEXT NOT NULL,
        actor TEXT NOT NULL,                 -- 'spark' | 'consolidator' | 'quantflow-hook'
        kind TEXT NOT NULL,                  -- tool_call|observation|gate|hook_summary|...
        payload TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_trajectory_run_seq ON trajectory_events(run_id, seq);

    -- Uniform client_id idempotency for mutating effects that have no natural
    -- client_id column in the plan DDL (log_outcome, memory_promote):
    -- reuse returns the recorded result with no side effects (N2).
    CREATE TABLE IF NOT EXISTS tool_effects (
        client_id TEXT PRIMARY KEY,
        tool TEXT NOT NULL,
        subject_id TEXT NOT NULL,
        result TEXT NOT NULL,                -- JSON of the returned record
        created_at TEXT NOT NULL
    );

    -- FTS5 (degraded-mode retrieval, §4.3/§12.1). External-content tables kept
    -- in sync by insert triggers; queries PIT-filter before ranking.
    CREATE VIRTUAL TABLE IF NOT EXISTS beliefs_fts
        USING fts5(claim, content='beliefs', content_rowid='rowid');
    CREATE VIRTUAL TABLE IF NOT EXISTS tombstones_fts
        USING fts5(summary, content='tombstones', content_rowid='rowid');

    CREATE TRIGGER IF NOT EXISTS beliefs_ai AFTER INSERT ON beliefs BEGIN
        INSERT INTO beliefs_fts(rowid, claim) VALUES (new.rowid, new.claim);
    END;
    CREATE TRIGGER IF NOT EXISTS tombstones_ai AFTER INSERT ON tombstones BEGIN
        INSERT INTO tombstones_fts(rowid, summary) VALUES (new.rowid, new.summary);
    END;
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
