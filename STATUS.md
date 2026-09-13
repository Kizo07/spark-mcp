# STATUS — Spark Phases 0–3 (Goal 1: ledger+gates MVP) + G1 hardening pass

Per-phase verdicts from independent milestone reviews (muse-spark-1.3-contributor,
max reasoning effort, read-only). A phase is done only on PASS or
PASS-WITH-NITS without blockers (AGENTS.md rule 5).

## Phase verdicts

| Phase | Commit | Review file | Verdict | Notes |
|---|---|---|---|---|
| 0 — Skeleton | `559910f` → fixes `03f6b19` | reviews/2026-09-12_phase-0.md (FAIL) + 2026-09-13_phase-0-rereview.md | **PASS-WITH-NITS** | 3 blockers fixed (gate scan path, `__main__`, embedder determinism); adversarially re-verified |
| 1 — Ledger & Memory | `bc511a3` | reviews/2026-09-12_phase-1.md | **PASS-WITH-NITS** | PIT-before-ranking, exact-once trials, reconciliation independently reproduced |
| 2 — Method Graph & Novelty | `1344d89` | reviews/2026-09-13_phase-2.md | **PASS-WITH-NITS** | RRF retrieval, bounded N/S bit-identical cross-process; post-review scorer change at `1473f38` (see G1) |
| 3 — Gates & Backtest Seam | `a85c21a` | reviews/2026-09-13_phase-3.md | **PASS-WITH-NITS** | Golden pins re-derived by third machinery (~1e-12); G2 reproduced 0.44s |
| G1 hardening delta | `39543dc..HEAD` (`e16980e`) | reviews/2026-09-13_delta-hardening.md | **pending** | LLM wiring + nit fixes + selection repair; verdict filled when the review lands |

## Program gates — final

| Gate | Status | Evidence |
|---|---|---|
| **G1** (end Phase 2) | **OVERLAP GREEN / JUDGE RED (corpus-bound)** | **Overlap half: GREEN.** Round 3 (LLM scorer, `8e81e8c`): mean 0.204 over 5 problems × 5 repeats, all problems ≤0.28 (SLO 0.30). Round 4 (`e16980e`, champion eligible): mean 0.26; 12-seed Monte Carlo of the same code path puts true per-problem rates at 0.20–0.21 (the 5-call gate sample is high-variance; two problems landed at 0.32). **Judge half: RED** after two measured LLM-scorer cycles (same rubric/judge all rounds): count≥3/5 per problem — r1 [4,1,1,1,0], r2 [4,1,0,0,2], r3 [5,0,1,1,0], r4 [5,0,1,1,0] (reviews/2026-09-13_g1-judge-round{1..4}.md). Criterion ≥2/problem ≥3/5 on all 5: met only on P1 (5/5) and unmet on P2/P3/P5. **Diagnosis (see below): corpus coverage, not mechanism.** |
| **G2** (end Phase 3) | **GREEN (standalone)** | loop 0.44s vs 900s SLO; (counted+reconciled)/attested = 17/17 = 1.0 exactly incl. 3 retries + 2 out-of-band; 100% counted; C6 enforced at entry and gate; SR always paired with DSR. Live-peer half: peer exists (`../alpha_engine`) but its tools do not speak plan §15.2 contract; RecordedProvider cassettes cover the seam per plan §17.4. **Post-goal follow-up: live integration once the peer exposes §15.2 tools.** |

## G1 judge half: why it stays red (and why we stopped)

With the LLM scorer live, relevance *ranking* is no longer the binding
constraint — for P2 the scored candidate list is exactly right (2SLS 0.757,
doubly-robust 0.697, front-door 0.620, negative-controls 0.608, PCMCI 0.601,
backdoor 0.578, DiD 0.500, PC 0.498, synthetic control 0.496). The judge
half is **corpus-bound**:

- **P5 (microstructure denoising):** the corpus holds ~1 method the judge
  would rate ≥3 (Kalman filter; the judge's own ICA rationale says "Kalman/
  wavelets dominate this"). Criterion needs 2. No scorer can rank methods
  that do not exist.
- **P3 (vol-regime early detection):** vol-timing coverage tops out at the
  GARCH trio (judge: ≤2, backward-looking) + option-implied skew (3). Max
  achievable ≈ 1–2.
- **P2 (macro causal identification):** ~12 causal cards exist; the strict
  judge rates identification-fit for rate→sector questions at 1–2 for all
  but the canonical estimators (which cap at 3).

The remedy is plan-scoped **Phase-4 corpus ingestion** (§16.1 seeds are v0;
ingestion pipelines are later-phase work). Growing the seed corpus to target
this judge would be gaming the eval. Selection knobs were iterated exactly
once on principle (champion-exclusion reversal, flagged for reassessment by
the Phase-2 review) and not tuned against the blind draw.

### G1 mechanism history (all rounds, same 5 problems / rubric / judge)

1. Rounds 1–2 (deterministic lexical scorer): overlap GREEN 0.14; judge
   RED. Diagnosis then: lexical relevance ceiling → plan remedy is the
   LLM-scored path (§8.1/M4), which needed an endpoint.
2. User provided a z.ai coding-plan key (OpenAI-compatible,
   `https://api.z.ai/api/coding/paas/v4`, chat models only — no embeddings
   route on this key). `OpenAICompatibleLLM` wired (`19ba51d`): temperature
   0, retry on transient 429/5xx, tolerant JSON parsing (fences/prose) so a
   chatty scorer cannot silently fall back.
3. Round 3: overlap RED 0.54 (sharper LLM scores concentrated the pool
   draw; thin recall sets of 7–20 made some pools a trivial 6-of-6). Two
   mechanism repairs, both §8.2 design space (`8e81e8c`): recall floor
   `selection_recall_pad_to=30` (pad with best-remaining techniques by
   deterministic relevance; padding only orders, the scorer still ranks)
   and `selection_pool_temp_multiplier` 3→6 (12-seed Monte Carlo
   calibration: 0.54→0.20, suggestion quality unchanged). Round 3 rerun:
   overlap GREEN 0.204.
4. Round 3 judge [5,0,1,1,0] with the excluded-champion pattern visible
   (judge named the excluded champion as the right answer). Reversed
   `selection_obvious_drop` 1→0 (`e16980e`): the exclusion's overlap
   rationale died with the recall floor (simulated 0.202→0.198 with the
   champion included) and hiding the best-scoring candidate removed the
   pick most likely to be plausible. Round 4: overlap mean 0.26; judge
   [5,0,1,1,0] — corpus-bound, stop per the goal's stop rule.

## Carried nits / follow-ups

Post-review nit-fix pass (commits `84bd915`, `e16980e`): **N5, N6, N7, N8,
N10 fixed; Phase-2 unresolved-refs nit fixed; pool-size nit already
satisfied** (the flagged literal moved to `selection_pool_extra` config in
the Phase-2 selection redesign). Still open:

- **Live alpha_engine integration**: needs peer-side §15.2 tools (out of
  this goal's scope: no alpha_engine changes permitted).
- **Corpus growth (P2/P3/P5 coverage)** — the G1 judge-half remedy; scoped
  to Phase 4+ ingestion (docling pipeline, peer harvest). Do NOT grow seeds
  to target the judge.
- **G4 go/no-go** (≥50 logged outcomes with non-degenerate family
  posteriors) already gates Phase 6.
- Phase 2 deferred-by-design: structural-overlap term degenerate in tool
  paths (needs free-text→graph grounding design before Phase 5);
  graph-node archive shared across families (recorded decision).
- N1 (crash between dispatch and commit) — architectural, C4 reconciled
  backstop judged defensible by the Phase-3 review.
- N2 (first-class placebo episode creation) — feature plumbing, Phase 4+.
- Embeddings: this key has no embeddings route; `build_embedder` degrades
  to None (FTS5 + lexical MMR). Wire an embedding endpoint or the
  `embeddings-local` extra when available.
- `idea` text now persisted in the propose trajectory payload.

## Environment record

- LLM: z.ai coding plan via zcode config (`api.z.ai/api/coding/paas/v4`),
  scorer `glm-5.3` (temperature 0). `spark doctor` reports "full (llm key
  present)" when `llm.base_url` + key env are configured.
- pyribs absent from this machine's package index → `[qd]` extra; Phase 5
  must ship a native MAP-Elites fallback.
- venv: Python 3.12 (see README).
- Test counts: 15 → 44 → 82 → 160 → 166 → 180 (hardening pass: +14). All
  green at HEAD; ruff clean.

*Goal 1 ended at `39543dc` per its stop rule; the 2026-09-13 hardening pass
(user-provided key) wired the LLM path, fixed all code-level nits, and took
G1 through four blind rounds: overlap half green, judge half red with a
diagnosed, plan-scoped remedy. Final commit records this file and all review
artifacts.*
