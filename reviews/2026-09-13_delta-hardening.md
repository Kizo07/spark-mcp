# Delta Re-Review: 39543dc..HEAD (G1 hardening pass)

**Verdict: PASS-WITH-NITS.** The delta is honest, well-tested engineering that does what its commit messages claim — with one substantive finding the orchestrator's diagnosis misses, plus documentation nits. Details below; all work was read-only.

**On the proposed STATUS line:** I endorse recording **overlap GREEN / judge RED**, but **not** the unqualified parenthetical **"(corpus-bound; Phase-4 remedy)"**. The accurate record is: judge RED — *depth-per-problem shortfall relative to draw entropy, compounded by a demonstrated recall-coverage gap (on-target cards excluded from the candidate set) that should be repaired before/during Phase 4*. The orchestrator's three specific sub-claims don't all survive contact with the evidence (§2). No re-tuning against the blind draw is warranted; the one mechanism fix I identify needs no judge feedback at all.

Review environment note: no LLM API key exists in this session, so live-LLM figures (overlap rounds, 12-seed MC, judge draws) are corroborated for plausibility — not reproduced. Everything else was verified by reading code, running the suite, and an independent deterministic-path probe.

---

## 1. Delta verification

**Suite: 179 passed, ruff clean.** Measured with the project venv (`.venv/bin/python -m pytest`: `179 passed in 27.78s`; `.venv/bin/ruff check src tests`: `All checks passed!`). The suite includes the independence gate (2 tests, green). One doc nit: commit `8e81e8c`'s message says "179 → 180" but HEAD collects 179 (167 test functions + parametrization; `git grep` counts 154→158→166→167→167 across the five commits, consistent with +4 LLM/parse, +8 nit, +1 recall-floor, +0 knob-rewrite). Trivial, but the pending STATUS update should say 179, not 180.

**Diff walk (all four commits reviewed in full):**
- `19ba51d` LLM wiring: `OpenAICompatibleLLM` ([llm.py](./src/spark_mcp/llm.py:89)) POSTs `{base}/chat/completions`, temp 0, retries transient 429/5xx only, key used solely in the `Authorization` header (never logged/printed — grepped). `build_llm` ([llm.py](./src/spark_mcp/llm.py:175)) degrades to `None` without key *or* endpoint; `build_embedder` stays `None` with the z.ai rationale documented. Matches plan §4.3/M4 (protocol + `model_id()` in cache keys — [explore.py](./src/spark_mcp/explore.py:347)).
- `84bd915` nits N5–N8/N10 + seeds: each nit has code + committed test, all passing — N5 ([test_gates_service.py](./tests/test_gates_service.py:216)), N6 ([test_gates_service.py](./tests/test_gates_service.py:480) plus kill-rule-via-genuine-outcomes rewrites), N7 (DDL comment [storage.py](./src/spark_mcp/storage.py:279)), N8 (episode/consolidation tests [test_gates_service.py](./tests/test_gates_service.py:825) + v5 `run_id` migration [storage.py](./src/spark_mcp/storage.py:327), applied by the version-stepping `migrate()`), N10 (DISALLOW test [test_gates_service.py](./tests/test_gates_service.py:369)). Seeds: concepts 52→247 (+195 ✓), techniques 101→101 with one canonicalization, `unresolved_refs == 0` pinned by [test_graph.py](./tests/test_graph.py:93). The `dsr.py` hunk is comment-only. The three re-pinned `test_explore.py` assertions are honest mechanism-invariant relaxations (corpus dilution hurt the lexical path — anti-gaming evidence, not goalpost-moving).
- `8e81e8c` recall floor + temp 3→6; `e16980e` obvious_drop 1→0 + round-3 file. Both verified claim-by-claim next.

**Selection mechanics — the three commented claims all check out:**
1. *"Padding only orders"* — TRUE ([explore.py](./src/spark_mcp/explore.py:518)). Padding appends best-remaining-by-deterministic-relevance up to 30, then *all* candidates go through Pass 1 LLM/deterministic scoring ([explore.py](./src/spark_mcp/explore.py:558)) — no score is assigned at pad time. PIT-correct (`as_of <= ?`, [explore.py](./src/spark_mcp/explore.py:532)), deterministic tie-break. Pinned by [test_explore.py](./tests/test_explore.py:380). *Nuance (see §2): padding selects the candidate SET, and that set demonstrably excludes on-target cards — the comment is accurate about scoring but the recall decision itself leaks coverage.*
2. *"Champion eligible at default"* — TRUE. Default `selection_obvious_drop = 0` ([config.py](./src/spark_mcp/config.py:91)); `pool_order = order[0:]` ([explore.py](./src/spark_mcp/explore.py:669)).
3. *"Knob still works"* — TRUE. [test_explore.py](./tests/test_explore.py:313) pins both directions over 4 seeds: champion surfaces at default, never under `obvious_drop=1`.
- Temp 6.0 direction is correct (flatter draw → less overlap; reported probabilities stay unflattened, [explore.py](./src/spark_mcp/explore.py:651)). The "0.54→0.20" calibration and "0.198 vs 0.202" figures are LLM-live and unverifiable in this session — plausible, flagged as such, not as fact.

**Independent probe (deterministic path, FTS-only to match live G1's no-embedder condition, temp DBs):** native recall P1–P5 = 18/9/19/20/**7** — corroborates "as few as 7" exactly; all pad to 30. Deterministic 5-seed overlap means 0.22–0.24 with pairwise values spanning 0.0–0.4 — corroborates the high-varianceDefense of the 5-call sample. Overlap arithmetic checks: (0.32+0.32+0.22+0.24+0.20)/5 = **0.26** ✓.

---

## 2. Honesty of the corpus-coverage diagnosis — PARTIALLY SUPPORTED, overstated in specifics

I read all four round files and programmatically verified every summary (counts, avgs, n=25 — all match; no tampering). What the rationales *do* support: the judge is strict, specific, and consistent in kind ("portfolio construction, not a causal estimator", "annual factor, irrelevant to intraday", "requires cross-sectional overlap; invalid for universal time-series macro shock"). P1 passes 4/4 rounds on corpus saturation (51-card quant family) while thin problems fail — the mechanism *can* pass when depth exists. Judge variance also exists (Negative controls for P2: 3 in R1 → 2 in R3), which honestly cuts against ever passing a fixed-draw criterion.

But three specifics fail:

| Orchestrator claim | Evidence |
|---|---|
| "P5 has ~1 card the judge would rate ≥3" | Corpus holds Kalman, DWT denoising, UKF, CWT scalogram — and the judge's own R3 rationale says "Kalman/**wavelets** dominate" (plural). ~1 is true of the *candidate set*, not the corpus (see below). |
| "P3's vol-timing coverage tops out at ~2" | Contradicted by demonstrated ratings: VRP=3 (R1), Delta-hedged=**4** (R3), skew=3 (R4). Three distinct ≥3 cards exist; they just never co-draw. |
| "P2's cards get rated 1–2" | 2SLS got 3 (R2); Negative controls got 3 (R1). The accurate claim is "too few ≥3 cards to reliably co-draw 2", not absence. |

**The missed mechanism fix (no blind-draw tuning required):** my probe shows the recall floor *excludes* on-target cards from the 30-candidate set on the deterministic path — and the candidate set is identical under the LLM scorer, since padding precedes scoring ([explore.py](./src/spark_mcp/explore.py:540)):
- P3: Hamilton regime-switching, HMM regimes, CUSUM **all NOT in candidates** (only option-implied proxies made it).
- P5: DWT, UKF, CWT **NOT in candidates** (only Kalman, rank 15/30).
- P2 (contrast): 2SLS/DiD/Synthetic all present (DiD rank 2) — yet never suggested in R3/R4, implicating the LLM rank + flat draw there.

Padding by weak lexical relevance guarantees a lexically-near set, not a semantically-covering one. Principled repairs exist in-plan and need zero judge feedback: exhaustive cached LLM scoring over all 101 techniques (one-time cost via `score_cache`), family-stratified padding floors, or graph-proximity padding — verifiable from corpus family/mechanism labels alone. This is the one genuine mechanism item the orchestrator missed. Caveat: it may still not pass G1 (a strict judge may rate surfaced cards 2), so it does not overturn RED — but "corpus-bound, not mechanism" is wrong as stated.

**On the obvious_drop reversal vs. tuning:** borderline but acceptable. It used R3 feedback, but it *removes* a deviation from plan §8.2 (the plan never excludes the top scorer), applies uniformly, keeps the knob tested both ways, and yielded zero gain (R4 identical to R3) — openly reported. One overclaim: `e16980e` says rationales named "(Kalman filter, 2SLS)" — only Kalman is in R3; 2SLS appears solely in R2. The 2SLS-champion assertion is unverifiable because **candidate rankings were never logged for any round** — the review's top process nit.

**Otherwise correctly left alone:** sharpening the near-uniform draw (T_eff = 0.7×6 = 4.2 over 24 eligibles) or touching MMR λ against blind draws *would* be tuning, and would break overlap. The overlap↔judge tension (diversity vs. concentration) is real; saturation via Phase-4 ingestion is the right lever. Stopping after one reversal was disciplined.

---

## 3. Evidence handling — no cherry-picking

All four rounds preserved as separate files with sequential timestamps (R3 11:45, R4 11:51); R1+R2 committed at `39543dc`, R3 at `e16980e`, R4 present untracked. No deleted review files, no hidden judge/overlap artifacts. The reported history *hurts* the author's case in exactly the places a cherry-picker would hide: R3 shows a P2 regression (1→0), R4 shows zero gain from the champion reversal, and round-4 overlap discloses two problems above 0.30. Internal arithmetic of all four files verified programmatically. The only unverifiable numbers are LLM-live simulations (flagged, not impugned).

---

## 4. Standing constraints — all intact

- **Independence:** gate green; no new runtime deps (`httpx` pre-existing); no reference-repo contact in the delta.
- **Spark-never-computes-returns:** delta adds only scoring/selection/transport/budget-accounting code. No backtest/causal/risk math.
- **PIT:** padding query is `as_of`-filtered; retrieval-before-ranking untouched. (Pre-existing, out of scope: `score_cache` keys lack `as_of` — stale scores could survive corpus time-travel. Worth a Phase-5 note.)
- **Trial accounting:** M semantics pinned by passing tests — failed retries replay without dispatch or M increment (N5); provider failures count toward M but don't tick the kill rule (N6, `M_family == 4` after 4 infra failures); fail_closed rolls trial+spend back *before* dispatch inside the caller's transaction ([service.py](./src/spark_mcp/gates/service.py:432)), matching pre-check end state. The `log_outcome` fail_closed whole-refusal is a documented, tested, opt-in tradeoff (default `warn` unaffected).

---

## 5. Nits (for the pending STATUS update and next goal)

1. **Recall-coverage repair** (substantive, §2): measure per-problem family/mechanism coverage of the 30-candidate set from corpus labels; repair padding (exhaustive cached scoring or stratified floors). No judge data needed.
2. **Log candidate rankings** for every future G1 round (the 2SLS-champion claim is currently unverifiable).
3. **Per-candidate scoring provenance:** `scoring_mode: "llm"` is global ([explore.py](./src/spark_mcp/explore.py:702)) while per-candidate deterministic fallback is silent — report fallback counts in G1 evidence.
4. **`_parse_score_reply` greedy regex** ([explore.py](./src/spark_mcp/explore.py:324)): two `{...}` blocks in one reply grab across both → parse fail → silent fallback. Use a first-valid-block scan.
5. **Stale STATUS lines to fix now:** test counts ("178…175 green, 3 awaiting re-pin" — actually 179 green, re-pin done in-delta), the "Re-pin needed" section, the champion-exclusion design flag (resolved by `e16980e`), and the LLM-path "no endpoint" paragraph.
6. **Record overlap honestly:** GREEN on the established mean basis, but note 0.26 vs 0.30 is thin, P1/P2 exceed per-problem, and the 5-call sample is high-variance — with per-problem values in STATUS, not just the mean.

**Bottom line:** a clean, honest delta (PASS-WITH-NITS). Record G1 as overlap GREEN (thin, mean-basis) / judge RED, with the cause stated as *depth-per-problem shortfall relative to draw entropy plus a demonstrated recall-coverage gap* — not "corpus coverage, not mechanism." Phase-4 ingestion remains the right remedy; the recall repair in nit 1 should ride with it.
