Re-review complete at HEAD `03f6b19`. All verification ran read-only against the repo (`PYTHONDONTWRITEBYTECODE=1 -p no:cacheprovider`); adversarial/bare-install probes staged from `git archive` under `/tmp` and removed. `git status` clean at end.

## Blockers

1. **Gate AST scan vacuous — VERIFIED.** [test_independence_gate.py](./tests/independence/test_independence_gate.py:17) uses `parents[2]` and [asserts non-empty scan](./tests/independence/test_independence_gate.py:44). Live: resolves to `src/spark_mcp`, scans **16 files**. Adversarial: injected `import dspy` into a scratch copy → gate FAILS with `forbidden imports found: [('novelty.py', ['dspy'])]`; clean scratch passes.
2. **`python -m spark_mcp` broken — VERIFIED.** [__main__.py](./src/spark_mcp/__main__.py:1) exists, calls `server.main()`. Piped `initialize` over stdio → `serverInfo:{"name":"spark","version":"0.1.0"}` in project venv **and** in a bare venv (`uv venv` + install from `git archive`, then cleaned).
3. **FakeEmbedder `hash()` seeding — VERIFIED.** [llm.py](./src/spark_mcp/llm.py:63) uses `sha256(text␀model)`; two separate processes produced identical vectors (digest `5a827a76…`); no builtin `hash()` remains outside comments.

## Claimed nits

- **FastMCP version passed — VERIFIED.** [server.py](./src/spark_mcp/server.py:68) `FastMCP("spark", version=__version__)`; handshake reports `0.1.0`, not FastMCP's `4.0.3`.
- **`informed_prior_after_episodes` — VERIFIED (present, unwired).** [config.py](./src/spark_mcp/config.py:65) `= 20`, runtime default confirmed; no other references — fine for Phase 0, must be wired before bandit work claims it.
- **`monkeypatch.delenv` — VERIFIED.** [test_server.py](./tests/test_server.py:17), [gate](./tests/independence/test_independence_gate.py:22); both files pass with `OPENAI_API_KEY=fake` in ambient env.
- **`contradiction_overlap` in ResearchConfig — VERIFIED.** [config.py](./src/spark_mcp/config.py:66) `= 0.6`; [memory.py](./src/spark_mcp/memory.py:156) reads it; `CONTRADICTION_OVERLAP` gone repo-wide; threshold is live (0.7-overlap claim found at 0.6, not at 0.8).
- **`episodes.reward NOT NULL DEFAULT 0.0` — VERIFIED.** [storage.py](./src/spark_mcp/storage.py:75); fresh-DB DDL contains the clause; `log_outcome` without `reward` yields `0.0` float.
- **Protocol canonical bytes — VERIFIED.** [ledger.py](./src/spark_mcp/ledger.py:301) writes `<artifacts>/protocols/<hash>.json`; file exists, `sha256(bytes) == protocol_hash`, keys `{code_version, hypothesis_id, seed, spec}`.
- **Duplicate trajectory write removed — VERIFIED (with side effect).** Server no longer imports/calls trajectory ([server.py](./src/spark_mcp/server.py:153) is comment-only); server-level propose now yields exactly **1** event (was 2). Side effect: `idea` text is no longer persisted anywhere — the removed write was its only store (ledger payload has no `idea` key). Minor audit-trail regression; carry as nit.
- **STATUS.md created — VERIFIED**, but contains placeholder hashes: `a1b2c3d` ([STATUS.md](./STATUS.md:9)) and `` `review-fixes` `` (:13) instead of `03f6b19`. Doc nit.

## Suite / regressions

- 82 tests collected (matches orchestrator). **3 of 4 full-suite runs green; 1 failed** on `test_repeat_query_overlap_below_g1_threshold` (`0.34 >= 0.30`) — the test is unseeded/stochastic ([test_explore.py](./tests/test_explore.py:68)), so the suite is flaky, not reliably green. Ruff clean. `sqlite_vec: True` in bare venv (probe fix holds).

## Verdict: PASS-WITH-NITS

All three Phase 0 blockers and all claimed nits are real and verified by execution, not diff-reading. Remaining nits: flaky G1 overlap test, `idea` dropped from trajectory, STATUS.md placeholder hashes, unwired `informed_prior_after_episodes`.

**Go/no-go for Phase 3:** Go — Phase 0 is closed and needs no further review. But Phase 3 (final phase of this goal) must open by stabilizing the G1 overlap test (seed it or assert on distribution with retries), because a gate test that fails ~1 in 4 runs cannot anchor the G2 milestone; also re-persist `idea` (trajectory payload or hypotheses-adjacent store) so the dedup fix stops dropping user input, and correct the STATUS.md hashes so the audit trail points at `03f6b19`.
