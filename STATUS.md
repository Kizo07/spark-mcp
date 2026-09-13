# STATUS — Spark Phases 0–3 (Goal 1: ledger+gates MVP)

Per-phase verdicts from independent milestone reviews (muse-spark-1.3-contributor,
max reasoning effort, read-only). A phase is done only on PASS or
PASS-WITH-NITS without blockers (AGENTS.md rule 5).

| Phase | Commit | Review file | Verdict | Open items |
|---|---|---|---|---|
| 0 — Skeleton | `559910f` → fixes in `03f6b19` | reviews/2026-09-12_phase-0.md (FAIL, superseded) + reviews/2026-09-13_phase-0-rereview.md (**PASS-WITH-NITS**) | **PASS-WITH-NITS** | nits below |
| 1 — Ledger & Memory | `bc511a3` | reviews/2026-09-12_phase-1.md | **PASS-WITH-NITS** | nits carried below |
| 2 — Method Graph & Novelty | `1344d89` | reviews/2026-09-12_phase-2.md | pending | G1 judge half pending |

## Phase 0 review blockers → fixes (commit `review-fixes`)

1. Independence-gate AST scan was vacuous (`parents[1]` → nonexistent path;
   rglob silently scanned 0 files). Fixed: `parents[2]` + assertion that
   scanned-file count > 0. tests/independence/test_independence_gate.py
2. `python -m spark_mcp` broken (no `__main__.py`) though it is the plan's
   canonical entrypoint (§3.1, §15.1). Fixed: src/spark_mcp/__main__.py;
   verified MCP initialize handshake over stdio.
3. FakeEmbedder seeded from per-process-salted builtin hash() — broke its
   determinism contract. Fixed earlier in Phase 2 work (sha256-based).

## Phase 0 review nits addressed

- `FastMCP("spark", version=__version__)` (stdio serverInfo now reports spark's
  version, not FastMCP's).
- `ResearchConfig.informed_prior_after_episodes` (=20) added (plan §16).
- Degraded-mode tests now force `monkeypatch.delenv(OPENAI_API_KEY)` —
  assertions no longer depend on ambient env (test_server.py, gate).
- sqlite-vec probe fixed (enable_load_extension) during Phase 2; doctor now
  reports `sqlite_vec: True`.
- Contradiction threshold moved from module constant to
  `ResearchConfig.contradiction_overlap` (=0.6) (AGENTS.md config rule).
- `episodes.reward REAL NOT NULL DEFAULT 0.0` (was nullable; §6.3 non-null).
- protocol_hash canonical bytes now stored under
  `<artifacts>/protocols/<hash>.json` (§5.0/N8 audit requirement).
- Duplicate trajectory write on hypothesis_propose removed (ledger records it).

## Environment deviations (disclosed, not papered over)

- **pyribs absent from this machine's package index** (only a 0.0.2 stub is
  published; verified against the index's own JSON API). pyribs moved from
  core deps to optional `[qd]` extra. Phase 5 (outside this goal) must ship a
  native numpy MAP-Elites fallback. README + pyproject note it; the
  independence gate forbids core from importing pyribs.
- venv pinned to Python 3.12 (3.13 works for core, but pyribs metadata caps
  <3.13 — moot while pyribs is an extra; 3.12 chosen for headroom).

## Phase 1 carried nits (accepted or scheduled)

- `storage.Database.execute()` commits via nested `with conn:` — weakens
  atomicity of outer transactions; fix before money-path concurrency
  (Phase 3 hardening).
- Concurrent same-client_id `memory_promote` / failure-tombstone paths can
  double-insert rows (M stays correct); formal dedup hardening = Phase 3.
- Rate-card cost estimation (§14) not yet implemented — costs default to
  zeros with `estimated: true`; Phase 3 with budgets.
- Contradiction handling does not yet record the pointer on the target
  belief's contradicting_runs / set valid_to — deferred to Phase 7
  LLM adjudication per §12.3 (documented in code).

## Program gates

| Gate | Status | Evidence |
|---|---|---|
| G1 (end Phase 2) | half met | repeat-query overlap mean 0.18 < 0.30 measured over 5 repeats (deterministic path); judge half (≥2 suggestions/problem ≥3/5 by a quant judge) pending muse-judge run |
| G2 (end Phase 3) | pending | — |

*Updated after each milestone review. Next update: Phase 0 re-review verdict,
Phase 2 review verdict, G1 judge evidence.*
