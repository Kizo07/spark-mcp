"""Concurrency probe (P2-5): N=4 parallel read/write tool-level operations
against ONE temp SQLite database, measured 2026-10-25 (docs/gates.md).

Connection patterns — which is real, documented from storage.py/server.py:

* Production MCP pattern: `create_server` builds ONE `Services` with ONE
  shared `Database` handle. `Database` owns a single sqlite3 connection
  (`check_same_thread=False`, WAL, `busy_timeout=5000`) and serializes access
  across MCP worker threads with a re-entrant lock — but only code that goes
  through `Database.locked()` / `transaction()` / `execute()` is serialized.
  `ledger.log_outcome` wraps its whole body in `db.locked()`, so it is safe
  on the shared handle (test 1). `ledger.register_hypothesis` /
  `propose_hypothesis` do their initial reads via bare `db.conn.execute`
  (ledger.get_hypothesis) outside any lock, so concurrent calls on the
  shared handle intermittently corrupt each other's parameter binding /
  result rows (sqlite3.InterfaceError, spurious "unknown hypothesis_id",
  wrong-row reads) — never SQLITE_BUSY, since one connection has one
  writer. The envelope for them is serialise (test 2 asserts exactly-once
  after serial re-drive, and that failures are only that known class).
* Separate-connections pattern (the repo's own racing tests in
  tests/test_ledger.py): one `Database` per thread on the same file. Every
  call is then single-threaded per connection; WAL + busy_timeout=5000
  coordinate writers. Probed in test 3: zero SQLITE_BUSY observed; the one
  hazard is `ensure_vec_embeddings` check-then-create racing across
  connections on simultaneous first embeds, recovered by serial re-drive.

All workers are barrier-synchronized, use distinct client_ids/seeds, and
assert every row present exactly once. No sleeps; outcome is deterministic
even though the intermittent races above may or may not fire on a run —
their observed counts are printed, never hidden.

Test 4 (fix B): bare `ledger.get_hypothesis` / `get_episode` /
`_replay_effect` reads now self-serialize via `db.locked()`, so direct
(unlocked-caller) concurrent reads racing register/log_outcome writers on
the shared handle must be clean with correct rows.
"""

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

from spark_mcp import ledger
from spark_mcp.config import Config
from spark_mcp.storage import open_db

N = 4

# Error classes the shared-handle register race is known to produce
# (measured 2026-10-25): shared-connection parameter/result interleaving,
# never lock contention.
_REGISTER_RACE_TYPES = (sqlite3.InterfaceError, sqlite3.IntegrityError)


def _propose_fields(i: int) -> dict:
    return dict(
        research_family="momentum",
        hypothesis_type="predictive",
        mechanism=f"concurrency probe mechanism {i} momentum reversal rebalancing",
        expected_direction="negative",
        horizon="20d",
        scope="cross_section",
        falsifier="reversal disappears net of transaction costs",
        predicted_ic=0.03,
        predicted_sigma=0.01,
        universe="us_equity",
    )


def _is_busy(exc: BaseException) -> bool:
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    msg = str(exc).lower()
    return "busy" in msg or "locked" in msg


def _is_vec_create_race(exc: BaseException) -> bool:
    """Cross-connection first-embed race: two connections both find
    vec_embeddings absent and both CREATE it (storage.ensure_vec_embeddings
    check-then-create is only serialized within one Database instance)."""
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    msg = str(exc).lower()
    return "vec_embeddings" in msg and "already exists" in msg


def _is_register_race(exc: BaseException) -> bool:
    if isinstance(exc, _REGISTER_RACE_TYPES):
        return True
    # get_hypothesis raced to an empty/wrong result on the shared connection.
    return isinstance(exc, ValueError) and "unknown hypothesis_id" in str(exc)


def test_shared_handle_n4_parallel_log_outcomes(db, cfg, embedder):
    """Production envelope: 4 threads share ONE Database handle (conftest
    `db`) and run log_outcome (which self-serializes via db.locked()) on
    distinct pre-registered episodes, plus locked reads. Must be clean:
    no SQLITE_BUSY, no errors, exactly-once trial accounting."""
    episode_ids = []
    for i in range(N):
        prop = ledger.propose_hypothesis(db, cfg, embedder=embedder, **_propose_fields(i))
        reg = ledger.register_hypothesis(
            db, cfg, prop["hypothesis_id"], {"signal": f"mom{i}"}, seed=100 + i
        )
        episode_ids.append(reg["episode_id"])

    barrier = threading.Barrier(N)
    results: list = [None] * N
    errors: list = [None] * N

    def worker(i: int) -> None:
        try:
            barrier.wait(timeout=10)
            out = ledger.log_outcome(
                db,
                cfg,
                episode_ids[i],
                "success",
                metrics={"ic": 0.01 * i},
                client_id=f"probe-log-{i}",
                embedder=embedder,
            )
            with db.locked():  # serialized read on the shared connection
                ep = ledger.get_episode(db, episode_ids[i])
            assert ep is not None and ep["status"] == "validated"
            results[i] = (out["status"], out["trial_id"])
        except BaseException as exc:  # recorded, asserted below
            errors[i] = exc

    with ThreadPoolExecutor(N) as pool:
        list(pool.map(worker, range(N)))

    busy = [e for e in errors if e is not None and _is_busy(e)]
    assert busy == [], f"SQLITE_BUSY under shared handle: {busy!r}"
    assert errors == [None] * N, f"parallel log_outcome failed: {errors!r}"
    assert all(r[0] == "validated" for r in results)
    assert len({r[1] for r in results}) == N  # distinct trials

    # Exactly once: N episodes, one reconciled trial each, all validated.
    assert db.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == N
    assert db.conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0] == N
    per_episode = db.conn.execute(
        "SELECT episode_id, COUNT(*) AS c FROM trials GROUP BY episode_id"
    ).fetchall()
    assert len(per_episode) == N and all(r["c"] == 1 for r in per_episode)
    statuses = {
        r[0] for r in db.conn.execute("SELECT DISTINCT status FROM episodes").fetchall()
    }
    assert statuses == {"validated"}


def test_shared_handle_register_serialise_envelope(db, cfg, embedder):
    """N=4 concurrent registrations on the ONE shared handle must all
    succeed: ledger.register_hypothesis/propose_hypothesis are serialised
    on the Database instance lock (previously bare reads raced). Exactly
    once per client_id; no SQLITE_BUSY, no re-drive needed."""
    hyps = [
        ledger.propose_hypothesis(db, cfg, embedder=embedder, **_propose_fields(i))[
            "hypothesis_id"
        ]
        for i in range(N)
    ]
    barrier = threading.Barrier(N)
    results: list = [None] * N
    errors: list = [None] * N

    def register_one(i: int, handle) -> dict:
        return ledger.register_hypothesis(
            handle,
            cfg,
            hyps[i],
            {"signal": f"mom{i}"},
            seed=300 + i,
            client_id=f"probe-reg-{i}",
        )

    def worker(i: int) -> None:
        try:
            barrier.wait(timeout=10)
            results[i] = register_one(i, db)
        except BaseException as exc:  # recorded, asserted below
            errors[i] = exc

    with ThreadPoolExecutor(N) as pool:
        list(pool.map(worker, range(N)))

    busy = [e for e in errors if e is not None and _is_busy(e)]
    assert busy == [], f"unexpected SQLITE_BUSY on one connection: {busy!r}"
    assert errors == [None] * N, f"concurrent register failed: {errors!r}"
    assert all(r is not None for r in results)
    assert len({r["episode_id"] for r in results}) == N
    assert db.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == N
    dupes = db.conn.execute(
        "SELECT COUNT(*) FROM (SELECT client_id FROM episodes"
        " GROUP BY client_id HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    assert dupes == 0  # every row present exactly once


def test_separate_connections_n4_parallel_tool_ops(tmp_path, embedder):
    """Envelope pattern: 4 threads each open their OWN Database (separate
    connections) on the same file and run propose -> register -> read ->
    log_outcome on distinct hypotheses. WAL + busy_timeout=5000 is the only
    coordination. Unexpected errors always fail; SQLITE_BUSY and the
    vec_embeddings creation race, if observed, are recovered by the
    documented serialise envelope and still end exactly-once (both counts
    are printed, never hidden)."""
    cfg = Config(
        storage={
            "db_path": f"{tmp_path}/probe-separate.db",
            "artifacts_dir": f"{tmp_path}/art-sep",
        }
    )
    path = cfg.db_path()
    open_db(path).close()  # create + migrate before the race

    barrier = threading.Barrier(N)
    results: list = [None] * N
    busy_errors: list = [None] * N
    vec_race_errors: list = [None] * N
    other_errors: list = [None] * N

    def drive(handle, i: int):
        prop = ledger.propose_hypothesis(
            handle,
            cfg,
            embedder=embedder,
            client_id=f"probe-sep-{i}",
            **_propose_fields(i),
        )
        reg = ledger.register_hypothesis(
            handle,
            cfg,
            prop["hypothesis_id"],
            {"signal": f"mom{i}"},
            seed=200 + i,
            client_id=f"probe-reg-sep-{i}",
        )
        hyp = ledger.get_hypothesis(handle, prop["hypothesis_id"])
        assert hyp is not None and hyp["hypothesis_id"] == prop["hypothesis_id"]
        out = ledger.log_outcome(
            handle,
            cfg,
            reg["episode_id"],
            "success",
            metrics={"ic": 0.01 * i},
            client_id=f"probe-log-sep-{i}",
            embedder=embedder,
        )
        return (prop["hypothesis_id"], reg["episode_id"], out["status"])

    def worker(i: int) -> None:
        handle = open_db(path)
        try:
            barrier.wait(timeout=10)
            results[i] = drive(handle, i)
        except BaseException as exc:  # recorded, asserted below
            if _is_busy(exc):
                busy_errors[i] = exc
            elif _is_vec_create_race(exc):
                vec_race_errors[i] = exc
            else:
                other_errors[i] = exc
        finally:
            handle.close()

    with ThreadPoolExecutor(N) as pool:
        list(pool.map(worker, range(N)))

    n_busy = sum(e is not None for e in busy_errors)
    n_vec_race = sum(e is not None for e in vec_race_errors)
    print(
        f"[concurrency-probe] separate-connections busy_errors={n_busy}/{N}"
        f" vec_create_races={n_vec_race}/{N}"
    )
    assert other_errors == [None] * N, f"unexpected failures: {other_errors!r}"

    # Documented envelope: serialise any race-failed workers, then verify.
    # Re-driven calls replay on client_id (raced inserts already committed),
    # so exactly-once is preserved.
    for i in range(N):
        if busy_errors[i] is not None or vec_race_errors[i] is not None:
            handle = open_db(path)
            try:
                results[i] = drive(handle, i)
            finally:
                handle.close()

    assert all(r is not None for r in results), f"missing results: {results!r}"
    assert len({r[0] for r in results}) == N  # distinct hypotheses
    assert len({r[1] for r in results}) == N  # distinct episodes
    assert all(r[2] == "validated" for r in results)

    db = open_db(path)
    try:
        assert db.conn.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0] == N
        assert db.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == N
        assert db.conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0] == N
        dupes = db.conn.execute(
            "SELECT COUNT(*) FROM (SELECT client_id FROM hypotheses"
            " GROUP BY client_id HAVING COUNT(*) > 1)"
        ).fetchone()[0]
        assert dupes == 0  # every row present exactly once
    finally:
        db.close()


def test_shared_handle_concurrent_reads_race_writers(db, cfg, embedder):
    """Fix-B envelope: reader threads call ledger.get_hypothesis directly
    (no outer lock — exercises the db.locked() inside the read helpers)
    while writer threads race register_hypothesis + log_outcome on the ONE
    shared handle. Must be clean: no errors, every read returns the
    correct row, writers land exactly once."""
    n_readers = N
    n_writers = N
    reads_per_thread = 25

    read_ids = [
        ledger.propose_hypothesis(db, cfg, embedder=embedder, **_propose_fields(i))[
            "hypothesis_id"
        ]
        for i in range(n_readers)
    ]
    expected_mechanism = {
        hid: _propose_fields(i)["mechanism"] for i, hid in enumerate(read_ids)
    }
    write_ids = [
        ledger.propose_hypothesis(
            db, cfg, embedder=embedder, **_propose_fields(100 + i)
        )["hypothesis_id"]
        for i in range(n_writers)
    ]

    barrier = threading.Barrier(n_readers + n_writers)
    read_errors: list = [None] * n_readers
    write_errors: list = [None] * n_writers
    read_counts: list = [0] * n_readers
    write_results: list = [None] * n_writers

    def reader(i: int) -> None:
        try:
            barrier.wait(timeout=10)
            for _ in range(reads_per_thread):
                for hid in read_ids:
                    # Direct unlocked-caller read: correct only because the
                    # helper self-serializes on the shared connection.
                    hyp = ledger.get_hypothesis(db, hid)
                    assert hyp is not None, f"missing row for {hid}"
                    assert hyp["hypothesis_id"] == hid, "wrong-row read"
                    assert hyp["mechanism"] == expected_mechanism[hid], "wrong-row read"
                    read_counts[i] += 1
        except BaseException as exc:  # recorded, asserted below
            read_errors[i] = exc

    def writer(i: int) -> None:
        try:
            barrier.wait(timeout=10)
            reg = ledger.register_hypothesis(
                db,
                cfg,
                write_ids[i],
                {"signal": f"racemom{i}"},
                seed=400 + i,
                client_id=f"probe-race-reg-{i}",
            )
            out = ledger.log_outcome(
                db,
                cfg,
                reg["episode_id"],
                "success",
                metrics={"ic": 0.02 * i},
                client_id=f"probe-race-log-{i}",
                embedder=embedder,
            )
            write_results[i] = (reg["episode_id"], out["status"])
        except BaseException as exc:  # recorded, asserted below
            write_errors[i] = exc

    with ThreadPoolExecutor(n_readers + n_writers) as pool:
        # Submit ALL workers before waiting: readers block on the shared
        # barrier (parties = readers + writers), so sequential pool.map
        # calls would deadlock. (reader/writer record their own errors.)
        futs = [pool.submit(reader, i) for i in range(n_readers)]
        futs += [pool.submit(writer, i) for i in range(n_writers)]
        for f in futs:
            f.result()

    assert read_errors == [None] * n_readers, f"reader failures: {read_errors!r}"
    assert write_errors == [None] * n_writers, f"writer failures: {write_errors!r}"
    assert read_counts == [reads_per_thread * n_readers] * n_readers
    assert all(r is not None and r[1] == "validated" for r in write_results)
    assert len({r[0] for r in write_results}) == n_writers

    # Exactly once: pre-proposed hypotheses + writer episodes/trials.
    assert (
        db.conn.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0]
        == n_readers + n_writers
    )
    assert db.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == n_writers
    assert db.conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0] == n_writers
