# Program gates G1–G5 — original vs amended/current criteria

This file is the citation source for gate status. STATUS.md remains the verdict log; counts below are copied from STATUS.md, the gate tests, and the review files listed as evidence — no new counts are introduced here.

Canonical G1 citation (use verbatim): **G1 — judge RED [5,1,1,1,1] single-draw; GREEN only under Tier-3 union-of-5 amendment.**

## G1 (end Phase 2) — suggestion quality

- **Original criterion (§18.0, single-draw):** two halves in one blind draw per problem: (a) repeat-overlap at/under the 0.30 SLO, and (b) judge half — ≥2 suggestions/problem rated ≥3/5 on ALL 5 canonical problems.
- **Amended/current criterion (Tier-3, user-authorized plan §18.0 change, 2026-09-13):** overlap half unchanged (≤0.30). Judge half evaluates the suggestion *distribution*: the union of top-5 across 5 independent balanced draws must contain ≥2 distinct methods/problem rated ≥3/5.
- **Current status:** overlap GREEN; judge RED [5,1,1,1,1] single-draw; GREEN only under Tier-3 union-of-5 amendment.
- **Evidence:** STATUS.md “Program gates — final” (round 5: overlap mean 0.212, per-problem [0.22, 0.18, 0.20, 0.16, 0.30]; judge rounds r1 [4,1,1,1,0], r2 [4,1,0,0,2], r3 [5,0,1,1,0], r4 [5,0,1,1,0], r5 [5,1,1,1,1]); `reviews/2026-09-13_g1-judge-round{1..5}.md`; `reviews/2026-09-13_g1-round{3,5}-candidates.json`; corpus-saturation round 6 judge [5,1,2,1,2] in `reviews/2026-09-13_g1-judge-round6.md` (STATUS.md “G1 round 6 + knob-frontier measurement”); knob-frontier infeasibility in `reviews/2026-09-13_g1-knob-frontier.md` (no in-scope §8.2 setting satisfies both halves); amended round 7 in `reviews/2026-09-13_g1-judge-round7-union.md` (overlap mean 0.236, per-problem [0.12, 0.30, 0.24, 0.24, 0.28]; union distinct ≥3/5 per problem [17, 3, 8, 3, 3]).

## G2 (end Phase 3) — full loop + accounting

- **Original criterion (§18.0):** full propose→backtest→DSR→verdict loop under the 900s (15 min) SLO, plus (counted+reconciled)/attested = 1.0 exactly with 100% counted, no causal promotion without DAG, no promotion without falsifier, and SR always paired with DSR. The plan text also named a live-peer half against the local alpha_engine.
- **Amended/current criterion:** no threshold change. The milestone closed as **GREEN (standalone)** on the scripted sequence; the live-peer half was initially deferred (peer present but not speaking the §15.2 contract, RecordedProvider cassettes covering the seam per §17.4) and was later landed post-goal (2026-09-15) once alpha_engine exposed the §15.2 tools.
- **Evidence:** STATUS.md G2 row (loop 0.44s vs 900s SLO; 17/17 = 1.0 incl. 3 retries + 2 out-of-band; C6 at entry and gate); `tests/test_g2_sequence.py`; `reviews/2026-09-13_phase-3.md` (standalone VERIFIED, live-peer NOT MET at review time with contract mismatch as blocker); STATUS.md “Live alpha_engine integration: LANDED 2026-09-15” (backtest/causal/risk return `provider=alpha_engine, simulated=false`).

## G3 (end Phase 5) — QD coverage + behavior

- **Original criterion (§18.0, scripted scenario):** (1) ≥3 families occupied in the QD archive; (2) ≥1 underexplored cell suggested by `suggest_next` AND acted on; (3) ≥1 tombstone-blocked re-spend with the pre-check logged and the acknowledge path dispatching; (4) suggestions carry decomposition + `selection_probability`.
- **Amended/current criterion:** unchanged; no amendment recorded for G3.
- **Current status:** GREEN.
- **Evidence:** STATUS.md Phase 5 (4/192 cells across 4 families; cell `7:0:0` suggested AND acted on; tombstone block at cosine 0.931); `tests/test_g3_scenario.py`; `reviews/2026-09-13_phase-5.md` (§5: all four §18.0 conditions independently reproduced).

## G4 (before Phase 6) — go/no-go

- **Original criterion (§18.0):** G1–G3 green + ≥50 logged outcomes with non-degenerate family posteriors, else reassess per the §6 kill rule.
- **Amended/current criterion:** unchanged. G4 therefore tracked G1: it evaluated **RED** while G1 was single-draw RED, with the §6 reassessment artifact retained, and re-evaluated **GREEN** only after the Tier-3 union-of-5 amendment made G1 amended-GREEN.
- **Current status:** GREEN (on the amended G1 basis above; the red period is retained in STATUS.md).
- **Evidence:** STATUS.md “G4 go/no-go evaluation (2026-09-13, before Phase 6)” (G1 NOT green [5,1,1,1,1]; G2 GREEN standalone; G3 GREEN; ≥50-outcomes leg PASS — 60 outcomes across 5 families, Beta(10,4) everywhere; verdict RED — reassessed per §6); `tests/test_g4_reassessment.py` (family driven to K consecutive failures is auto-tombstoned, posterior collapses below 0.25, `suggest_next` redirects); STATUS.md “G1 — judge RED [5,1,1,1,1] single-draw; GREEN only under Tier-3 union-of-5 amendment — G4 re-evaluation GREEN” (≥50-outcomes leg previously demonstrated).

## G5 (end Phase 7) — creativity + consolidation

- **Original criterion (§18.0, scripted scenario):** (1) ≥1 analogy-derived hypothesis carried through the DSR gate to validated; (2) `suggest_next` consumes a consolidation gap report BY ID with the report’s gap families boosted; (3) ≥1 curriculum exercise generated from a REAL gap of that report and replay-scored against the current champion.
- **Amended/current criterion:** unchanged; no amendment recorded for G5.
- **Current status:** GREEN.
- **Evidence:** STATUS.md Phase 7 (analogy-derived hypothesis validated through the DSR gate; gap report consumed by id; curriculum exercise from a real gap replay-scored); `tests/test_g5_scenario.py`; `reviews/2026-09-13_phase-7.md` (PASS-WITH-NITS, GO).

## Concurrency envelope (measured 2026-10-25)

Probe: `tests/test_concurrency_probe.py` (P2-5), N=4 barrier-synchronized threads, tool-level ledger operations (propose/register/log_outcome + reads) against one temp SQLite DB per pattern, distinct client_ids/seeds, runtime ~0.6s per full file run. Measured over 10 consecutive `pytest tests/test_concurrency_probe.py -q` runs (all green, final state asserted exactly-once in every run):

- **Shared handle — locked paths (production pattern):** ONE `Database` shared across threads (as `Services.db` in server.py), workers running `log_outcome` (self-serializes via `Database.locked()`) on distinct episodes plus reads wrapped in `db.locked()`. Outcome: 0 errors, 0 SQLITE_BUSY in 10/10 runs; 4 episodes / 4 trials, exactly once, all `validated`.
- **Shared handle — `register_hypothesis`:** concurrent registrations originally raced (bare pre-reads outside the lock; `InterfaceError` / spurious unknown-hypothesis in characterization runs). Fixed 2026-10-25: `propose_hypothesis`/`register_hypothesis` now serialise on `Database.locked()`; the probe asserts 4/4 concurrent registrations succeed exactly-once.
- **Separate connections:** one `Database` per thread on the same file, each running propose→register→read→log_outcome. Outcome: 0 SQLITE_BUSY (WAL + `busy_timeout=5000`); the `vec_embeddings` check-then-create race on simultaneous first embeds is fixed in `ensure_vec_embeddings` (an `already exists` loser proceeds).

**Envelope summary (2026-10-25, post-fix):** safe under N=4 for tool-level ledger operations on the shared handle (all access via `Database.locked()`/`transaction()`/`execute()`) and for separate connections including first embeds. Bare `db.conn.execute` reads outside the lock remain outside the envelope.
