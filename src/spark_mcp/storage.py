"""SQLite storage: WAL, busy_timeout, user_version migrations, sqlite-vec.

Storage discipline (plan §5.0 / M6): WAL on, busy_timeout 5000, all mutations
in explicit transactions, idempotency via UNIQUE constraints, migrations as
versioned SQL applied through PRAGMA user_version.

Concurrency (Phase 3 hardening, Phase 1 review nit): a single connection is
shared across MCP worker threads (check_same_thread=False), so a re-entrant
lock serializes transactions at the Python level while busy_timeout covers
cross-process contention. `Database.transaction()` opens BEGIN IMMEDIATE and
nests via SAVEPOINTs; `execute()` never wraps `with conn:` while a transaction
is already active (the old behavior committed outer partial work early).
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA_VERSION = 10

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
        reward REAL NOT NULL DEFAULT 0.0,    -- §6.3 tool 15: non-null from Phase 1 (M13)
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
    3: """
    -- Phase 2: method graph & novelty (plan §5.1). Predicates restricted to the
    -- 6 active at launch (§2.7); enforced in code (graph.PREDICATES).
    --
    -- graph_nodes carries `as_of` in addition to the plan DDL so nodes obey the
    -- same PIT visibility rule as hypotheses/episodes (§5.0 matrix): a row is
    -- visible at T iff as_of <= T. `family` is additive to the plan DDL — the
    -- §7 novelty term needs family-scoped archive lookups.
    CREATE TABLE IF NOT EXISTS graph_nodes (
        node_id TEXT PRIMARY KEY,
        type TEXT NOT NULL,              -- technique|concept|domain|...
        label TEXT NOT NULL,
        family TEXT,                     -- additive to plan DDL (§7 family scoping)
        mech_tags TEXT NOT NULL DEFAULT '[]',  -- JSON array
        summary TEXT,
        as_of TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_graph_nodes_type ON graph_nodes(type);
    CREATE INDEX IF NOT EXISTS idx_graph_nodes_family ON graph_nodes(family);

    CREATE TABLE IF NOT EXISTS graph_edges (
        edge_id TEXT PRIMARY KEY,
        src TEXT NOT NULL REFERENCES graph_nodes(node_id),
        dst TEXT NOT NULL REFERENCES graph_nodes(node_id),
        predicate TEXT NOT NULL,         -- SOLVES|FAILS_WHEN|COMPOSES_WITH|
                                         -- DIAGNOSES|ASSUMES|DERIVED_FROM
        weight REAL,
        evidence_episode_ids TEXT NOT NULL DEFAULT '[]',  -- JSON (N6)
        valid_from TEXT NOT NULL,
        valid_to TEXT,                   -- invalidation, not deletion
        as_of TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_graph_edges_src ON graph_edges(src);
    CREATE INDEX IF NOT EXISTS idx_graph_edges_dst ON graph_edges(dst);
    CREATE INDEX IF NOT EXISTS idx_graph_edges_predicate ON graph_edges(predicate);

    -- FTS5 on graph_nodes (§5.0/N11 names label; mech_tags + summary are
    -- indexed too so degraded-mode recall is not blind to mechanism
    -- vocabulary — labels alone never carry it); insert trigger keeps sync.
    CREATE VIRTUAL TABLE IF NOT EXISTS graph_nodes_fts
        USING fts5(label, mech_tags, summary,
                   content='graph_nodes', content_rowid='rowid');
    CREATE TRIGGER IF NOT EXISTS graph_nodes_ai AFTER INSERT ON graph_nodes BEGIN
        INSERT INTO graph_nodes_fts(rowid, label, mech_tags, summary)
        VALUES (new.rowid, new.label, new.mech_tags, new.summary);
    END;

    -- LLM score cache (§8.1): cached per (method, problem_class, model_id) —
    -- model version enters the key so a model change re-scores, never re-scales.
    CREATE TABLE IF NOT EXISTS score_cache (
        cache_key TEXT PRIMARY KEY,      -- sha1(method_id|problem_class|model_id)
        method_id TEXT NOT NULL,
        problem_class TEXT NOT NULL,
        model_id TEXT NOT NULL,
        relevance REAL NOT NULL,
        structural_fit REAL NOT NULL,
        created_at TEXT NOT NULL
    );
    -- vec_embeddings is created dynamically by Database.ensure_vec_embeddings:
    -- a sqlite-vec vec0 virtual table when the extension loads, else a plain
    -- BLOB table with identical columns (degraded storage, numpy similarity).
    """,
    4: """
    -- Phase 3: gates & backtest seam (plan §5.1, §8.3, §14).
    --
    -- exploration_state: per-family exploration counters + bandit posterior.
    -- Phase 3 maintains trials/successes/consecutive_failures (+ the kill-rule
    -- flag); Phase 5's QD archive consumes the rest (posterior, windows).
    -- trials/successes are bandit observation-EVENT counters (§8.3
    -- visit/reward aggregates): +1 per genuine family outcome (a completed
    -- backtest, a significance_gate verdict, a log_outcome) — not the §9.2
    -- trial count, which lives in the trials table (M_family). Provider/infra
    -- errors deliberately do not feed them (they are not hypothesis evidence;
    -- Phase 3 review nit N6).
    CREATE TABLE IF NOT EXISTS exploration_state (
        family TEXT PRIMARY KEY,
        trials INTEGER NOT NULL DEFAULT 0,
        successes INTEGER NOT NULL DEFAULT 0,
        consecutive_failures INTEGER NOT NULL DEFAULT 0,
        analogy_trials INTEGER NOT NULL DEFAULT 0,
        posterior TEXT NOT NULL DEFAULT '{"alpha": 1.0, "beta": 1.0}',
        window_trials INTEGER NOT NULL DEFAULT 0,
        window_successes INTEGER NOT NULL DEFAULT 0,
        window_start TEXT,
        last_reward REAL,
        regime_context TEXT,
        killed INTEGER NOT NULL DEFAULT 0,
        updated_at TEXT NOT NULL
    );

    -- spend_ledger (§14): every spend event appends here; budget v1 reads it.
    CREATE TABLE IF NOT EXISTS spend_ledger (
        seq INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        family TEXT,
        source TEXT NOT NULL,            -- backtest_run | log_outcome | ...
        usd REAL NOT NULL,
        tokens INTEGER NOT NULL DEFAULT 0,
        estimated INTEGER NOT NULL DEFAULT 0
    );
    CREATE INDEX IF NOT EXISTS idx_spend_ledger_ts ON spend_ledger(ts);
    CREATE INDEX IF NOT EXISTS idx_spend_ledger_family ON spend_ledger(family);

    -- Phase 1 review nit: concurrent same-episode failure log_outcomes
    -- double-inserted tombstones. A UNIQUE dedupe key + INSERT OR IGNORE
    -- collapses racing writers to one row (NULLs are distinct in SQLite
    -- unique indexes, so tombstones without a key stay unaffected).
    ALTER TABLE tombstones ADD COLUMN dedupe_key TEXT;
    CREATE UNIQUE INDEX IF NOT EXISTS idx_tombstones_dedupe
        ON tombstones(dedupe_key);
    """,
    5: """
    -- Phase 3 review nits (N4/N8): attribute spend to the run that caused it
    -- (episode_id for backtest_run/log_outcome, consolidation run id for
    -- consolidate, §13) so per-episode and per-consolidation caps (§14) can
    -- be enforced from the ledger. NULL = unattributed (manual/ad-hoc rows).
    ALTER TABLE spend_ledger ADD COLUMN run_id TEXT;
    CREATE INDEX IF NOT EXISTS idx_spend_ledger_run ON spend_ledger(run_id);
    """,
    6: """
    -- Phase 4 (§15.1 C9): the QuantFlow hook-enrichment columns on episodes —
    -- agent_assembly_fingerprint, fingerprint_spec_version, skill_versions —
    -- already exist since migration 2 (plan §5.1 put them in the day-one DDL,
    -- M13), so an ALTER here would fail with a duplicate-column error on every
    -- database that ran migration 2, i.e. all of them. This marker pins the
    -- Phase-4 schema state; any future additive episode columns land in
    -- migration 7+.
    """,
    7: """
    -- Phase 5: QD archive (plan §5.1, §10.1). SQL is the durable source of
    -- truth (M7): the in-memory grid in archive.py is rebuilt from these rows
    -- at construction; insert() writes through to this table inside the
    -- caller's transaction. descriptor_spec_version bumps start a fresh grid
    -- when new dimensions land (§10.1) — history stays under the old version.
    CREATE TABLE IF NOT EXISTS qd_cells (
        cell_key TEXT NOT NULL,              -- 'f:h:t' = family:horizon:turnover idx
        descriptors TEXT NOT NULL,           -- JSON resolved names + score detail
        descriptor_spec_version INTEGER NOT NULL DEFAULT 1,
        elite_hypothesis_id TEXT, dsr REAL, oos_ic REAL,
        family TEXT, trials INTEGER NOT NULL DEFAULT 0,
        reward_type TEXT NOT NULL,           -- per-type rewards (M12)
        updated_at TEXT NOT NULL,
        PRIMARY KEY (descriptor_spec_version, cell_key)
    );
    CREATE INDEX IF NOT EXISTS idx_qd_cells_family ON qd_cells(family);
    """,
    10: """
    -- Phase 6 re-review (review 2026-09-13_phase-6, blocker B2): exact
    -- candidate -> challenger linkage. approve()/promote_via_gateway used to
    -- re-derive the challenger row from (name, lineage.parent_version),
    -- which after a lineage fork could veto/promote the WRONG live version.
    -- Nullable: pre-v9 rows keep the lineage-based fallback.
    ALTER TABLE skill_candidates ADD COLUMN challenger_skill_id TEXT;
    """,
    8: """
    -- Phase 6: skill evolution (plan §5.1, §11, M11). skill_candidates carries
    -- the replay evidence for one challenger; approvals is the M11 human-gate
    -- audit surface (also recording auto-with-human-notify gate decisions,
    -- decided_by='gate'); replay_score_cache holds LLM judge scores keyed by
    -- (skill_name, version, task_id, model_id) so a model swap re-scores
    -- rather than silently re-scales (§21).
    CREATE TABLE IF NOT EXISTS skill_candidates (
        candidate_id TEXT PRIMARY KEY,
        skill_name TEXT NOT NULL,
        parent_version INTEGER,
        source_episode_ids TEXT NOT NULL,   -- JSON array
        mutation_op TEXT,
        eval_scores TEXT,                   -- JSON {held_in: {...}, held_out: {...}}
        heldout_gap REAL,                   -- replay-gaming alarm (§21)
        verdict TEXT NOT NULL DEFAULT 'pending',  -- promoted|rejected|pending
        decided_at TEXT,
        decided_by TEXT                     -- 'gate' | 'human:…'
    );
    CREATE INDEX IF NOT EXISTS idx_skill_candidates_name
        ON skill_candidates(skill_name, verdict);

    CREATE TABLE IF NOT EXISTS approvals (
        approval_id TEXT PRIMARY KEY,
        target_type TEXT NOT NULL,          -- belief|skill|adoption
        target_id TEXT NOT NULL,
        decision TEXT NOT NULL,             -- approved|rejected
        rationale TEXT NOT NULL,
        decided_by TEXT NOT NULL,
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS replay_score_cache (
        cache_key TEXT PRIMARY KEY,         -- sha1(skill_name|version|task_id|model_id)
        skill_name TEXT NOT NULL,
        version INTEGER NOT NULL,
        task_id TEXT NOT NULL,
        model_id TEXT NOT NULL,
        score REAL NOT NULL,
        created_at TEXT NOT NULL
    );
    """,
    9: """
    -- Phase 7: consolidation runtime (plan §5.1 M10, §13) + SME-lite analogy
    -- engine text cache (§6.1 tools 3-4). consolidation_runs is the
    -- background-job audit log; the gap report itself is an immutable JSON
    -- artifact at {artifacts_dir}/consolidation/{report_id}.json, so reports
    -- are PIT-safe by construction (suggest_next consumes them by id, §13).
    CREATE TABLE IF NOT EXISTS consolidation_runs (
        run_id TEXT PRIMARY KEY,            -- == report_id
        started_at TEXT NOT NULL,
        finished_at TEXT,
        scope TEXT,
        llm_spend_usd REAL NOT NULL DEFAULT 0,
        report_uri TEXT NOT NULL,           -- spark://consolidation/{report_id}
        report_id TEXT NOT NULL
    );

    -- LLM-generated one-liners (analogy falsifiable translations cached per
    -- (base, target_class, model_id); combine rationales per (methods, target
    -- class, model_id)). Model version enters the key: a model swap re-
    -- generates, never silently re-scales (same rule as score_cache, §4.3).
    CREATE TABLE IF NOT EXISTS generated_text_cache (
        cache_key TEXT PRIMARY KEY,         -- sha1(purpose|subject|context|model_id)
        purpose TEXT NOT NULL,              -- analogy_translation | combine_rationale
        subject_key TEXT NOT NULL,
        context_key TEXT NOT NULL,          -- target problem class
        model_id TEXT NOT NULL,
        text TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    """,
}


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._vec_available = False
        self._vec_table_ready = False
        # Serializes transactions across MCP worker threads sharing this
        # handle; re-entrant so nested execute()/transaction() in one thread
        # never self-deadlocks. Cross-process contention is covered by
        # busy_timeout (WAL).
        self._lock = threading.RLock()
        self._txn_depth = 0

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
            self._load_sqlite_vec()
        return self._conn

    def _load_sqlite_vec(self) -> None:
        """Load the sqlite-vec extension once per connection; guard failures
        (N14: the package must run where the extension is missing)."""
        self._vec_available = False
        try:
            import sqlite_vec

            self._conn.enable_load_extension(True)  # load() is gated by default
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
            self._conn.execute("SELECT vec_version()").fetchone()
            self._vec_available = True
        except Exception:
            self._vec_available = False

    @property
    def vec_available(self) -> bool:
        """sqlite-vec usable on this connection (False → BLOB fallback table)."""
        _ = self.conn  # ensure the connection (and probe) has run
        return self._vec_available

    def vec_table_exists(self) -> bool:
        """vec_embeddings is created lazily on the first embed write."""
        row = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'vec_embeddings'"
        ).fetchone()
        return row is not None

    def ensure_vec_embeddings(self, dim: int) -> None:
        """Create vec_embeddings if absent. sqlite-vec present → vec0 virtual
        table (plan §5.1, N3: no BLOB columns on base tables); absent → plain
        BLOB table with identical columns so embed-on-write never loses data.
        Similarity search runs in numpy either way (archive scale is small),
        which keeps this dimension-agnostic across embedders."""
        if self._vec_table_ready:
            return
        row = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'vec_embeddings'"
        ).fetchone()
        if row is None:
            with self.conn:
                if self._vec_available:
                    self.conn.execute(
                        "CREATE VIRTUAL TABLE vec_embeddings USING vec0("
                        " subject_type TEXT, subject_id TEXT, model TEXT,"
                        f" model_version TEXT, embedding FLOAT[{dim}])"
                    )
                else:
                    self.conn.execute(
                        "CREATE TABLE vec_embeddings ("
                        " subject_type TEXT NOT NULL, subject_id TEXT NOT NULL,"
                        " model TEXT NOT NULL, model_version TEXT NOT NULL,"
                        " embedding BLOB NOT NULL)"
                    )
                    self.conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_vec_embeddings_subject"
                        " ON vec_embeddings(subject_type, subject_id)"
                    )
        self._vec_table_ready = True

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

    @contextmanager
    def locked(self) -> Iterator[sqlite3.Connection]:
        """Hold the instance lock for the whole block: every statement (reads
        included) is serialized across MCP worker threads sharing one
        connection. Re-entrant; combine freely with transaction()."""
        with self._lock:
            yield self.conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Explicit transaction: BEGIN IMMEDIATE at the outer level, SAVEPOINTs
        when nested. Commits on clean exit, rolls back (to the savepoint) and
        re-raises on any exception. Serialized across threads via RLock.

        Every multi-write mutation path must run inside this rather than
        `with conn:` so partial work is never committed early (Phase 1 review
        nit) and racing writers resolve through UNIQUE constraints inside one
        atomic unit.
        """
        with self._lock:
            conn = self.conn
            depth = self._txn_depth
            self._txn_depth += 1
            try:
                if depth == 0:
                    conn.execute("BEGIN IMMEDIATE")
                    try:
                        yield conn
                    except BaseException:
                        conn.rollback()
                        raise
                    conn.commit()
                else:
                    name = f"spark_sp_{depth}"
                    conn.execute(f"SAVEPOINT {name}")
                    try:
                        yield conn
                    except BaseException:
                        conn.execute(f"ROLLBACK TO {name}")
                        conn.execute(f"RELEASE {name}")
                        raise
                    conn.execute(f"RELEASE {name}")
            finally:
                self._txn_depth -= 1

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute one statement. Outside an explicit transaction this commits
        (backwards compatible); inside `transaction()` it joins the open
        transaction and never commits early."""
        with self._lock:
            if self.conn.in_transaction:
                return self.conn.execute(sql, params)
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
        return self.vec_available


def open_db(path: str | Path) -> Database:
    db = Database(path)
    db.migrate()
    return db
