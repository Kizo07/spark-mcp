# STATUS — Spark Phases 0–3 (Goal 1: ledger+gates MVP)

Per-phase verdicts from independent milestone reviews (muse-spark-1.3-contributor,
max reasoning effort, read-only). A phase is done only on PASS or
PASS-WITH-NITS without blockers (AGENTS.md rule 5).

## Phase verdicts

| Phase | Commit | Review file | Verdict | Notes |
|---|---|---|---|---|
| 0 — Skeleton | `559910f` → fixes `03f6b19` | reviews/2026-09-12_phase-0.md (FAIL) + 2026-09-13_phase-0-rereview.md | **PASS-WITH-NITS** | 3 blockers fixed (gate scan path, `__main__`, embedder determinism); adversarially re-verified |
| 1 — Ledger & Memory | `bc511a3` | reviews/2026-09-12_phase-1.md | **PASS-WITH-NITS** | PIT-before-ranking, exact-once trials, reconciliation independently reproduced |
| 2 — Method Graph & Novelty | `1344d89` | reviews/2026-09-13_phase-2.md | **PASS-WITH-NITS** | 153-node corpus, RRF retrieval, bounded N/S bit-identical cross-process; post-review scorer change at `1473f38` (see G1) |
| 3 — Gates & Backtest Seam | `a85c21a` | reviews/2026-09-13_phase-3.md | **PASS-WITH-NITS** | Golden pins re-derived by third machinery (~1e-12); G2 reproduced 0.44s; judge: GO for goal completion |

## Program gates — final

| Gate | Status | Evidence |
|---|---|---|
| **G1** (end Phase 2) | **RED** | overlap half: **GREEN** — reviewer-reproduced mean ≈0.14 over 5 problems × 5 repeats (<0.30; test now seeded, deterministic). judge half: **RED** — blind quant-judge round 1 (2026-09-13_g1-judge-round1.md): per-problem avg [2.8, 1.4, 1.8, 2.2, 1.6], count≥3 [4,1,1,1,0]. After one genuine improvement cycle (family-aware deterministic relevance + obvious-champion exclusion, `1473f38`), blind round 2 (2026-09-13_g1-judge-round2.md): avg [2.6, 1.8, 1.4, 1.0, 2.2], count≥3 [4,1,0,0,2], count≥4 [0,0,0,0,1]. Criterion ≥2/problem ≥3/5 unmet on 4/5 problems; high draw variance (ULID tie-breaking) noted. |
| **G2** (end Phase 3) | **GREEN (standalone)** | loop 0.44s vs 900s SLO; (counted+reconciled)/attested = 17/17 = 1.0 exactly incl. 3 retries + 2 out-of-band; 100% counted; C6 enforced at entry and gate; SR always paired with DSR. Live-peer half: peer exists (`../alpha_engine`) but its tools do not speak plan §15.2 contract (no `run_backtest`/`causal_test`/`risk_decompose`); RecordedProvider cassettes cover the seam per plan §17.4. **Post-goal follow-up: live integration once the peer exposes §15.2 tools.** |

## Why G1 was not forced (stop rule invoked)

One improvement cycle was attempted and honestly measured — the deterministic
lexical relevance path has a real ceiling, and the plan's intended remedy
(LLM-scored relevance, §8.1/M4) requires an OpenAI-compatible endpoint that
does not exist in this environment (muse is a CLI, not an HTTP endpoint).
Further cycles would tune against a high-variance blind draw — the
overfitting pattern this project exists to prevent. Per the goal's stop rule,
the goal stops and reports with G1 red rather than forcing a pass.

## Carried nits / follow-ups (for the next goal or Phase 4-7 work)

Post-review nit-fix pass (this working tree): **N5, N6, N7, N8, N10 fixed;
Phase-2 unresolved-refs nit fixed** (details inline below). Still open:

- **Live alpha_engine integration**: needs peer-side §15.2 tools (out of this
  goal's scope: no alpha_engine changes permitted).
- **LLM-scored relevance path**: implemented and unit-reachable; only the
  deterministic path is exercised live (no endpoint). Wire + re-run G1 when a
  key exists; G4 go/no-go (≥50 logged outcomes) already gates Phase 6.
- **N5 (FIXED)**: committed failed-retry regression test —
  `tests/test_gates_service.py::test_failed_trial_retry_returns_existing_row_without_dispatch`
  (retry of a FAILED trial replays the failed row without dispatch; new seed =
  new trial).
- **N6 (FIXED)**: kill-rule counter now ticks only on genuine evaluation/gate
  failures (`significance_gate` verdict, `log_outcome`); provider exceptions
  persist a failed trial (§9.2) but never tick (`gates/service.py` dispatch
  error path). Posterior penalty remains a Phase-5 deferral (`killed` flag
  only, noted in `_record_family_outcome_impl`).
- **N7 (FIXED, document-and-adjust)**: `exploration_state.trials` is a bandit
  observation-event counter, not the §9.2 trial count — invariant now stated
  at the DDL (`storage.py` v4 comment) and in `_record_family_outcome_impl`;
  N6 stops provider errors from feeding it. `M_family` remains the trial count.
- **N8 (FIXED)**: per-episode and per-consolidation caps enforced
  (`budgets.enforce_episode_budget` / `enforce_consolidation_budget`, same
  warn/fail_closed policy as daily, raise inside the caller's transaction so
  spend+trial roll back together). Wired into `backtest_run` + `log_outcome`;
  per-run consolidation cap (plan §13) is live and tested at the budgets
  level for Phase 7's consolidate to call. Schema v5 adds
  `spend_ledger.run_id` attribution.
- **N10 (FIXED)**: review-band verdict requires exact match
  (`verdict.strip().upper() == "ALLOW"`); "DISALLOW" blocks (test committed).
- Phase 2 review nits: structural-overlap term degenerate in tool paths (needs
  free-text→graph grounding design before Phase 5); graph-node archive not
  family-scoped (recorded decision: method graph is shared vocabulary);
  **226/524 seed concept refs (FIXED)**: concept layer grew by 195 cards so
  every referenced concept resolves — import now reports
  `unresolved_refs == 0` (348 nodes, 988 edges; pinned by
  `test_seed_import_resolves_every_reference`). One duplicate ref
  canonicalized (`computation_cost` → `computational_cost`); pool-size
  constant `min(n, max(2k, k+5))` should move to config (explore.py, pending
  the file owner).
- **Re-pin needed (off-limits test file)**: the 3 `tests/test_explore.py`
  assertions over the deterministic scorer's corpus profile
  (`test_family_factor_floors_off_family_relevance` expected causal weight
  exactly 1.0, now 0.9130 — econometrics gains 'effect' +
  causal_estimation/causal_inference vocabulary through newly-resolved
  econometrics→concept edges; `test_causal_problem_surfaces_causal_methods`
  finds 1 causal candidate where ≥2 expected; `test_cross_family_composition_survives_via_edges`
  finds 0 signal_processing candidates in the recall pool). Root cause is the
  mandated seed-corpus growth, not a scorer regression; re-pin by the
  explore-test owner.
- `idea` text now persisted in the propose trajectory payload.
- G1 scorer design flag for review in the next goal: the top-1-scored
  candidate is categorically excluded from suggestions (champion = the
  obvious move). Assess whether this product behavior is right when the LLM
  scorer arrives.

## Environment record

- pyribs absent from this machine's package index → `[qd]` extra; Phase 5 must
  ship a native MAP-Elites fallback.
- venv: Python 3.12 (see README).
- Test counts: 15 → 44 → 82 → 160 → 166 → 178 (nit-fix pass adds 8; 175
  green at HEAD with 3 explore assertions awaiting re-pin — see carried nits).
  Ruff clean.

*Goal 1 ends here per its stop rule: Phases 0–3 complete and independently
reviewed; G2 green; G1 red on the judge half after one measured improvement
cycle. Final commit records this file and all review artifacts.*
