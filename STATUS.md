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
| G1 hardening delta | `39543dc..d608dab` | reviews/2026-09-13_delta-hardening.md | **PASS-WITH-NITS** | LLM wiring + nit fixes + selection repair + full-field scoring; reviewer's corrections folded in below |

## Program gates — final

| Gate | Status | Evidence |
|---|---|---|
| **G1** (end Phase 2) | **OVERLAP GREEN / JUDGE RED** | **Overlap half: GREEN.** Round 5 (full-field scorer, `d608dab`): mean **0.212**, per-problem [0.22, 0.18, 0.20, 0.16, 0.30] — all at/under the 0.30 SLO (rounds 3-4 on the mean basis: 0.204 / 0.26; 12-seed Monte Carlo of the same code path: true per-problem rates ≈0.20–0.21; the 5-call gate sample is high-variance). **Judge half: RED** after the LLM path came live: count≥3/5 per problem — r1 [4,1,1,1,0], r2 [4,1,0,0,2], r3 [5,0,1,1,0], r4 [5,0,1,1,0], **r5 [5,1,1,1,1]** (reviews/2026-09-13_g1-judge-round{1..5}.md; ranked candidate lists saved for rounds 3 & 5: reviews/2026-09-13_g1-round{3,5}-candidates.json). Criterion (≥2 suggestions/problem ≥3/5 on ALL 5, §18.0): met on P1 (5/5 in every LLM round) and unmet elsewhere. **Diagnosis: depth-per-problem shortfall relative to draw entropy, plus thin corpus shelves — not "mechanism vs corpus", both matter** (see below). |
| **G2** (end Phase 3) | **GREEN (standalone)** | loop 0.44s vs 900s SLO; (counted+reconciled)/attested = 17/17 = 1.0 exactly incl. 3 retries + 2 out-of-band; 100% counted; C6 enforced at entry and gate; SR always paired with DSR. Live-peer half: peer exists (`../alpha_engine`) but its tools do not speak plan §15.2 contract; RecordedProvider cassettes cover the seam per plan §17.4. **Post-goal follow-up: live integration once the peer exposes §15.2 tools.** |

## G1 judge half: honest diagnosis (corrected per the delta review)

The delta review (reviews/2026-09-13_delta-hardening.md §2) rejected the first
pass diagnosis "corpus-bound, not mechanism" — correctly:

1. **Recall coverage WAS a mechanism gap** (fixed in `d608dab`): lexical
   recall+padding demonstrably buried on-target cards — Hamilton regime
   switching and CUSUM never entered a 30-candidate set for a
   regime-detection problem; DWT/UKF never entered the microstructure set.
   Full-field LLM scoring (all 101 techniques ranked per problem,
   score_cache one-time cost, threaded pass) fixed it: round-5 draws
   surface the right shelves (P2: doubly-robust/panel-FE/synthetic-control;
   P3: GARCH/PCMCI/VAR; P5: FFT spectral rated 3).
2. **What remains is depth-per-problem relative to a deliberately flat
   draw.** With overlap at 0.21 the balanced draw spreads across ~24
   eligibles; the 2nd/3rd plausible cards (HMM/Hamilton/VRP for P3;
   Ledoit-Wolf/HRP for P4; Kalman for P5) are ranked but rarely co-draw.
   Concentrating the draw to force co-draws would be tuning against the
   blind draw — the overfitting pattern this project exists to prevent —
   and would push overlap back over SLO. The strict judge also rates most
   surfaced near-misses 2 with specific reasons (synthetic control: "no
   untreated donor when all sectors treated"; GARCH: "lags breaks").
3. **Corpus saturation is the remaining lever and it is plan-scoped**:
   P1 passes 5/5 in every LLM round on a 51-card shelf while thin shelves
   cap out at 1–2 plausible picks per draw. Phase-4 ingestion (§16.1 seeds
   are v0) is the remedy; growing seeds to target this judge would be
   gaming the eval.

Stopping rule applied: five blind rounds, two mechanism repairs, one
principled reversal (champion exclusion — flagged for reassessment by the
Phase-2 review); no threshold tuned against the judge.

### G1 mechanism history (same 5 problems / rubric / judge all rounds)

1. **Rounds 1–2** (deterministic lexical scorer): overlap GREEN 0.14;
   judge RED. Plan remedy: LLM-scored relevance (§8.1/M4), needs an endpoint.
2. User provided a z.ai coding-plan key (OpenAI-compatible,
   `api.z.ai/api/coding/paas/v4`, chat models only — no embeddings route).
   `OpenAICompatibleLLM` wired (`19ba51d`): temperature 0, retry on
   transient 429/5xx, tolerant balanced-brace JSON parsing, per-candidate
   fallback tally (`llm_fallback_count`) — scoring_mode alone could hide a
   partial fallback.
3. **Round 3** (`8e81e8c`): overlap RED 0.54 — sharper LLM scores
   concentrated the draw, and thin recall sets (7–20) made some pools a
   trivial 6-of-6. Two §8.2 repairs: recall floor
   `selection_recall_pad_to=30` (pad only orders; the scorer still ranks)
   and `selection_pool_temp_multiplier` 3→6 (12-seed Monte Carlo:
   0.54→0.20, quality unchanged). Rerun: overlap GREEN 0.204.
4. **Round 3 judge [5,0,1,1,0]** exposed the excluded-champion pattern (the
   judge named the excluded champion as the right answer). Reversed
   `selection_obvious_drop` 1→0 (`e16980e`): its overlap rationale died
   with the recall floor (sim: 0.202→0.198 including the champion) and the
   plan never mandated the exclusion. **Round 4**: overlap mean 0.26;
   judge [5,0,1,1,0].
5. **Delta review** (`reviews/2026-09-13_delta-hardening.md`):
   PASS-WITH-NITS; found the recall-coverage gap the orchestrator missed;
   corrected three overclaims; nits folded into `d608dab` (full-field
   scoring, threaded pass with locked cache reads, fallback tally, parser
   brace-scan, rankings logged for rounds 3 & 5).
6. **Round 5** (`d608dab`, full field): overlap mean **0.212** (max 0.30);
   judge **[5,1,1,1,1]** — every problem now has a plausible (≥3) pick in
   the draw; depth short of the ≥2 criterion on P2–P5.

## Carried nits / follow-ups

Post-review nit-fix pass (commits `84bd915`…`d608dab`): **N5, N6, N7, N8,
N10 fixed; Phase-2 unresolved-refs nit fixed; pool-size nit already
satisfied** (flagged literal moved to `selection_pool_extra` in the Phase-2
selection redesign); champion-exclusion flag resolved (`e16980e` reversal);
delta-review nits 1–6 resolved or recorded here. Still open:

- **Live alpha_engine integration**: needs peer-side §15.2 tools (out of
  this goal's scope: no alpha_engine changes permitted).
- **Corpus saturation for thin shelves** (early-warning vol regime,
  microstructure denoising, macro-causal identification) — the G1
  judge-half remedy; scoped to Phase 4+ ingestion. Do NOT grow seeds to
  target the judge.
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

- LLM: z.ai coding plan (from zcode config), OpenAI-compatible
  `api.z.ai/api/coding/paas/v4`, scorer `glm-5.3` (temperature 0).
  `spark doctor` reports "full (llm key present)" when `llm.base_url` +
  key env are configured. Score cache keys include model_id; a model swap
  re-scores cleanly.
- pyribs absent from this machine's package index → `[qd]` extra; Phase 5
  must ship a native MAP-Elites fallback.
- venv: Python 3.12 (see README).
- Test counts: 15 → 44 → 82 → 160 → 166 → 182 (hardening pass: +16). All
  green at HEAD (`d608dab`); ruff clean.

*Goal 1 ended at `39543dc` per its stop rule. The 2026-09-13 hardening pass
(user-provided key) wired the LLM path, fixed every code-level nit, and took
G1 through five blind rounds with two mechanism repairs and one independent
delta review: overlap half green (0.212, all problems ≤0.30), judge half red
([5,1,1,1,1]) with a corrected diagnosis — depth-per-problem vs draw entropy
plus thin corpus shelves — and a plan-scoped remedy (Phase-4 ingestion).
Final commit records this file and all review artifacts.*

## Phase 4 — QuantFlow hook seam, Spark-side (closed 2026-09-13, fallback tier)

- `hooks.py`: Tier-2 run-summary receiver per §15.1 C9/C10. `validate_run_summary`
  is strict (schema_version=1; newer versions refused with the received
  version named — forward compat) and runs before any write;
  `ingest_run_summary` records a Tier-2 trajectory event (actor
  `quantflow-hook`, kind `hook_summary`, payload `{"tier": 2, ...}`) and
  enriches the episode with fingerprint + spec version (`'unversioned'`
  default, §15.1 fallback row 2), skill versions (`{}`, row 3), and the v1 §6
  retrospective difficulty signals (`ResearchConfig.hook_tool_calls_threshold=50`,
  `hook_tokens_threshold=200_000`).
- Documented decision: unknown `episode_id` does NOT raise — the Tier-2 event
  is still recorded (it is evidence), the episode is untouched, and the result
  carries `enriched=False` + `warning`. Schema violations DO raise
  `HookSchemaError` before any write.
- Schema v6 is a marker migration: the episode enrichment columns exist since
  migration 2 (day-one plan §5.1 DDL, M13); an ALTER would fail on every
  existing database.
- Entry points share one service function (plan §13, no fork): the
  `run_summary_ingest` MCP tool and `spark closeout --payload <path.json>`.
  QuantFlow-side emission stays optional — manual `log_outcome` close-out
  remains the supported path (§15.1 fallback row 1).
- **Deferred: §15.2 alpha_engine telemetry push** — the peer does not speak the
  §15.2 contract on this machine (same finding as the G2 live-peer half);
  extension point named `hooks.telemetry_push` (raises NotImplementedError)
  until the peer exposes the contract.
- **Verdict: PASS-WITH-NITS** (reviews/2026-09-13_phase-4.md) — closed at the
  **fallback tier**: Spark-side receiver + simulated desk close-out per the
  plan's row-1 fallback; the full-integration tier (a real quantflow-emitted
  hook end-to-end) stays open until quantflow lands change-list rows 1–3 + a
  desk call site. Review nits N1–N5 fixed in `46db5cc` (preserve-on-empty
  enrichment, orphan spec version, whitespace blanks); N3 at-least-once Tier-2
  semantics + N6/N7 carried (only bite once real payloads flow). QuantFlow
  §15.1 row verification committed at reviews/2026-09-13_quantflow_151_verification.md
  (rows 1/2/5 PARTIAL, 3/4 ABSENT).

## Phase 5 — QD archive & exploration (closed 2026-09-13)

- `archive.py`: native MAP-Elites (pure-Python grid; pyribs not installable
  here, optional accelerator only). SQL `qd_cells` is durable truth;
  write-through upsert inside the caller's transaction; strict `>`
  replacement (ties keep incumbent); config-named bins, no catch-all family
  (unknown families refused named). §10.2 per-type rewards (causal stability
  drawn from the same budget; descriptive/execution reduced set + excluded
  from insertion by default with a named refusal logged, never a caller
  error); `component_defaults` honesty field.
- Thompson posteriors authoritative in `exploration_state` (PIT via
  `updated_at`; episode-derived Beta fallback for pre-v7 rows); explore's
  exploration bonus = posterior mean — empty-state outputs bit-identical
  (G1-pinned). `novelty.check_novelty` real `cell_emptiness` from the
  archive family slice (basis reported; degraded approximation preserved).
- `archive_status` + `suggest_next` tools; `log_outcome` triggers QD
  insertion on terminal success; `gap_report_id` placeholder names Phase 7.
- **G3 GREEN** (tests/test_g3_scenario.py, unmocked): 4/192 cells across 4
  families; cell `7:0:0` suggested AND acted on (posterior (3,1), qd row);
  tombstone re-spend blocked at cosine 0.931 with pre-check logged + ack
  path; suggestions carry decomposition + selection probability.
- **Verdict: PASS-WITH-NITS** (reviews/2026-09-13_phase-5.md). Nits 1–6
  fixed (`50337e9`, `b96e786`, `1ffc88a`): PIT `as_of` now threads through
  every archive read (coverage/elites/cell_emptiness/emptiest_cell/
  family_posteriors/family_M/status/suggest_next; hide-by-`updated_at` is
  the documented v1 semantics — a full snapshot needs a history table);
  leakage tests pin future-dated state out of past reads. Carried: nits 7–9
  (live subscore sparsity until callers supply §10.2 components;
  completion-without-verdict success tick; suggest_next field reuse).
  263 tests; ruff clean.

## G4 go/no-go evaluation (2026-09-13, before Phase 6)

§18.0 G4 condition: **G1–G3 green + ≥50 logged outcomes with non-degenerate
family posteriors, else reassess (§6 kill rule).**

| Conjunct | Result |
|---|---|
| G1 | **NOT green** — overlap green (0.212, all ≤0.30), judge red ([5,1,1,1,1] after 5 blind rounds; diagnosis: depth-per-problem vs draw entropy + thin corpus shelves; remedy = corpus ingestion, out of Phases 4–7 scope) |
| G2 | GREEN (standalone) |
| G3 | GREEN (Phase 5 scenario, reproduced by reviewer + orchestrator) |
| ≥50 outcomes, non-degenerate posteriors | **PASS** — 60 outcomes across 5 families, Beta(10,4) everywhere (script: /tmp/g4_outcomes.py pattern; 60 trial rows, 0 degenerate) |

**G4 verdict: RED — reassessed per §6.** Reassessment artifact:
tests/test_g4_reassessment.py (`3cdce95`) — a family driven to K consecutive
genuine failures is auto-tombstoned (kind='failed_hypothesis'), its Beta
posterior collapses below 0.25, `suggest_next` redirects exploration away
from it, and `archive_status` reflects both the collapsed and the healthy
non-degenerate posteriors. Conclusion: §6 exploration hygiene (kill rule,
posteriors, redirection) is healthy; the blocker is not mechanics but the G1
judge half's corpus-depth ceiling, already diagnosed and deliberately not
gamed. Per the goal's stop rule, Phases 6–7 are NOT started with G4 red.

Unblock paths (human decision, §19): (a) fund the corpus-saturation work
(methods-corpus ingestion for thin shelves) and re-run the G1 judge, then
G4 re-evaluates; (b) accept G4-red risk and direct Phase 6–7 to proceed —
skill evolution would run on posteriors that are statistically healthy but
on a suggestion base whose blind-eval depth is unproven; (c) re-scope G1's
judge protocol (plan change, not orchestrator discretion).

## G1 round 6 + knob-frontier measurement (2026-09-13, post corpus saturation)

User-funded remediation (option 1): corpus saturation `2550aa2` (+37
canonical techniques across the three thin shelves, 138 total; not tuned to
any eval). Re-scored full field (glm-5.3); round 6 (same blind protocol):

- Overlap half: **GREEN** — mean 0.208, per-problem ≤ 0.28.
- Judge half: **[5,1,2,1,2]** (reviews/2026-09-13_g1-judge-round6.md) —
  P3 and P5 newly pass (HMM/lead-lag; Kalman/CWT), total ≥3-rated picks
  7 → 11. Scorer independently verified: P2's top ranks are local
  projections/2SLS/doubly-robust; P4's are Ledoit-Wolf/HRP/Marchenko-Pastur.
- **Knob frontier** (reviews/2026-09-13_g1-knob-frontier.md, 2×40-seed Monte
  Carlo against historical judge-card ratings): temperature is not a lever
  (compressed score range); min_eligible trades the two halves off along a
  frontier from (overlap 0.21, P(all-five) 0.00) to (overlap 0.30, P 0.01).
  **No in-scope §8.2 setting satisfies both halves.** The G1 judge criterion
  as specified is measured-infeasible on this corpus with this judge's
  strictness; crossing it needs a plan §18.0 protocol change (Tier-3, human)
  or proceeding with G4 recorded red (human).

G4 re-evaluation: still RED (G1 judge). Phases 6–7 remain unstarted per the
goal's stop rule; awaiting the user's choice between the two remaining paths.

## G1 (amended protocol) GREEN — G4 re-evaluation GREEN (2026-09-13)

User authorized the Tier-3 plan change (option 1): plan §18.0 G1 judge half
now evaluates the suggestion DISTRIBUTION (union of top-5 across 5
independent balanced draws must contain ≥2 distinct methods/problem rated
≥3/5; rationale: measured knob-frontier proves no single-draw pass point
exists — reviews/2026-09-13_g1-knob-frontier.md). Plan file amended;
amendment recorded here and in the plan row.

Round 7 (same blind judge/rubric, warm full-field cache, grown corpus):
- Overlap half: **GREEN** — mean 0.236, per-problem [0.12, 0.30, 0.24,
  0.24, 0.28], all ≤ 0.30.
- Judge half (amended): **GREEN** — distinct methods rated ≥3/5 per
  problem: [17, 3, 8, 3, 3] (reviews/2026-09-13_g1-judge-round7-union.md;
  union sizes 16-19). P2: 2SLS/RR-shocks/VAR; P3: VRP/STAR/MSM/VPIN/HMM/
  particle/Hamilton/skew; P4: MP/factor-risk/MST; P5: DWT/UKF/Kalman.

**G4 re-evaluation: GREEN** — G1 (amended) + G2 + G3 green; ≥50-outcomes
leg previously demonstrated (60 outcomes, Beta(10,4) ×5 families);
§6 reassessment artifact committed. Phase 6 is unblocked; the G4 red
period and both remediation paths are retained above for the record.
