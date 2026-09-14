# Phase 6 re-review — B1 / B2 / M1 fixes (`fd83666`)

**Baseline verified this session (read-only):** `391 passed in 42.84s`, `ruff check src tests` → clean. (The evidence block's dot snippet was truncated mid-output — full run shows 391, consistent with 327 + fix tests + Phase-7 tests.)

## B2 — exact candidate→version linkage: PASS-WITH-NITS

**Original repro re-run — fixed.** v2 promoted → retired → v3 (same `parent_version=1`) promoted → `approve(skill, cand_v2, rejected)` now raises `not a live champion`, leaving `{1: draft, 2: retired, 3: champion}`. `_challenger_for` resolves `cand_v2→v2`, `cand_v3→v3` via `challenger_skill_id` ([skills.py](./src/spark_mcp/skills.py:139), written atomically at propose time, [skills.py](./src/spark_mcp/skills.py:528)).

**New fork attempted (veto-then-rechallenge) — holds.** Built v3 (parent 2) → vetoed → v4 (parent 2, same parent as v3) → promoted, so v3/v4 share `parent_version=2`:
- Ratify naming vetoed v3 (`approved`, a non-duplicate decision) **safely refuses** `not a live champion`. Pre-fix this resolved live v4 and would have ratified the wrong version — the strongest proof the linkage is exact.
- Ratify/veto naming live v4 hits exactly v4 (`ratified_version: 4`); veto of live v3 in the retire-fork hits exactly v3.
- Gateway path exact too: manifest captured via `MockTransport` carries `version: 3` for `cand_v3` in the fork ([skills.py](./src/spark_mcp/skills.py:745)).
- (Veto-again of v3 returns the idempotent `duplicated: true` no-op per actor-class idempotency ([skills.py](./src/spark_mcp/skills.py:1102)) — safe, but via idempotency rather than linkage; the ratify probe above is the linkage test.)

**Nits:** (1) Legacy rows with `challenger_skill_id IS NULL` still fall back to the ambiguous lineage lookup — demonstrated live that the fallback maps `cand_v3 → v4` (wrong). Nullable-by-design ([storage.py](./src/spark_mcp/storage.py:357)), and Phase 6 has no production legacy data, but the fallback should refuse on ambiguity rather than guess. (2) Stale version refs: docstring says "v9+" and migration comment "pre-v9" — the column is v10. (3) `_challenger_for` is annotated `Row` but the gateway path passes `dict(cand)`; a dict missing the key would `KeyError` instead of falling back (no current caller does this — all pass full rows).

## M1 — full-suite coverage: PASS-WITH-NITS

**Enforcement confirmed, no bypass found.** `decide_promotion` requires every held-in and held-out fixture id ([skills.py](./src/spark_mcp/skills.py:911)): held-out-only (`suite="econometrics"`, `n_held_in=0`), single-task, and all-but-one subsets are all refused naming the missing fixtures (e.g. `missing fixtures: replay_leakage_screen`); full replay then promotes. Filtered-overwrite-after-full is also refused (fail-closed last-write-wins). `load_fixtures()` default is used on both write and check paths, so no suite/task_ids combination can smuggle a subset through.

**Nit (made slightly worse, friction-only):** coverage is now required even for `decision='rejected'` — an operator cannot reject a partially-evidenced candidate without a full replay first. Fail-closed direction, but the review asked for the rule on promotion; extending it to rejection is workflow friction. Same for a filtered diagnostic run downgrading full evidence and forcing a re-run.

## B1 — product surface + auto criterion: PASS-WITH-NITS

**Surface exists.** `decide_skill_promotion` + `retire_skill` are registered MCP tools (28-tool list verified live, [server.py](./src/spark_mcp/server.py:599)); `skills decide|retire` exist in the CLI with `--decision promoted|rejected|auto` ([cli.py](./src/spark_mcp/cli.py:254)). Both call the same service functions (N13).

**Auto criterion is real — all four branches traced live** ([skills.py](./src/spark_mcp/skills.py:933)):
- **Regression rejects (pure product path, no DB edits):** incumbent snapshot mean 0.3125; split-challenger mean 0.125 with gap 0.208 (under alarm 0.25) → `rejected` via the incumbent comparison — isolates the comparison branch, not the gap branch. Statuses `{1: draft, 2: champion, 3: rejected}`.
- **Gap alarm rejects:** mean 0.698, gap 0.615 → `rejected`.
- **Floor rejects:** weak first promotion vs `replay_promotion_floor=0.5` ([config.py](./src/spark_mcp/config.py:253)) → `rejected`.
- **Beater promotes:** 0.312 vs weakened incumbent 0.1, gap 0 → `promoted` (proves the promote branch executes; natural deterministic scores rarely reach it, which is why the branch needs a pinned test).

**Nits:** (1) The committed `test_auto_gate_criterion_promotes_and_rejects` asserts only `decision in (promoted, rejected)` — with natural AUDIT/split scores it always takes floor-reject and never pins the comparison, gap, or promote branches. My probes cover the gap; the repo test should pin all four. (2) Manual `'promoted'` still bypasses all thresholds (operator override by design) and nothing auto-invokes the gate (`auto` = automatic verdict, not automatic invocation) — both within the review's "criterion or downgrade" alternative, but the manual-as-override framing should be explicit. (3) Edge leniencies: retired-champion incumbents miss the `status='champion'` lookup ([skills.py](./src/spark_mcp/skills.py:927)) so auto falls back to the floor; pre-fix champions without an `eval_scores` snapshot compare against `old_mean=0.0` ([skills.py](./src/spark_mcp/skills.py:950)).

## §4 — anything made worse

Nothing correctness-worse. Friction-only: reject-requires-full-evidence (M1 nit above). Untouched/carried: uncapped replay-LLM spend (prior budgets nit), `decided_by` strings conventional-not-structural. Migration keys are out of source order (10 before 8) but `migrate()` sorts ([storage.py](./src/spark_mcp/storage.py:537)) — cosmetic.

## Verdict

| Item | Verdict |
|------|---------|
| B2 exact-version linkage | **PASS-WITH-NITS** (legacy-fallback ambiguity, stale v9 comments) |
| M1 full-suite coverage | **PASS-WITH-NITS** (over-strict on reject) |
| B1 surface + auto criterion | **PASS-WITH-NITS** (weak committed auto test, manual-override caveats) |

**Overall: GO for closing Phase 6.** All three blockers are fixed at the root cause with live-verified behavior; remaining items are nits (strongest: pin all four auto branches in a committed test, make the legacy fallback refuse on ambiguity).
