# Strategy/prompt evolution upgrade plan (RD-shaped loop, natively reimplemented)

Date: 2026-10-12. Status: plan (not implemented).
Implements the recommendation of
`/home/fire/Documents/Audit/2026-10-12-spark-gepa-vs-rd-evolution-audit.md` §7:
give trading-strategy / research-artifact evolution to an RD-shaped outer loop
(hypothesis → experiment → executed backtest → feedback → trace), ported as a
*pattern onto spark's existing ledger*, with spark's gate stack as the
fitness/promotion layer. GEPA stays only for skill/procedure text, after repair.

All paths below are relative to the spark-mcp root unless absolute.
RD-Agent (`/home/fire/Documents/RD-Agent`) is reference only — see constraint 1.

## 0. Hard constraints (from the task; restated as acceptance criteria)

1. **Native reimplementation.** No `rdagent` imports, no vendored RD code.
   Add `rdagent` to the `tests/independence/` forbidden scan so CI enforces it.
2. **Spark's gate stack stays the fitness/promotion layer**: trial governor
   (pending-trial-before-dispatch, idempotent M), DSR `significance_gate`,
   challenger/champion/rollback, human veto via `approve`. Never adopt RD's
   LLM "Replace Best Result" / `Decision` boolean as ground truth; the
   decision bit is always computed from numbers.
3. **No human-in-the-loop inside the evolution run itself.** The run is a
   closed loop; the existing `approve` veto and adoption gates stay outside it.
4. **Deterministic mode stays bit-reproducible.** One seeded RNG per run;
   no unseeded randomness on any evolution path (no `random`/`np.random`
   globals, no coin flips, no wall-clock in scored state). LLM paths record
   model id + rendered prompt (judge-cache style) for audit.
   Reproducibility scope: decisions, line heads (by hypothesis content +
   episode metrics), and feedback narration inputs are byte-identical
   across reruns; ULID primary keys (`ids.py` path, never the run's seeded
   RNG) and wall-clock timestamps are identity/audit metadata, never scored
   state, and are excluded from the byte-identity assertion by column.
5. **Held-out fixtures stay promotion-only evidence**, never evolution
   feedback (`score_population` docstring rule extends to the new fitness:
   evolution sees held-in / in-sample metrics only; held-out / OOS metrics
   enter only at the promotion gate).
6. **Fix the rubric-leaking `add_check_step` operator**: no verbatim rubric
   text injected as fitness. See component F.

Additionally (repo rules from AGENTS.md): spark never computes returns —
all backtest math stays behind provider protocols; PIT correctness on new
rows; experiment-affecting constants live in `ResearchConfig`; storage via
`PRAGMA user_version` migrations; milestone closes only with a review.

## 1. Architecture: what gets built

Two tracks, one shared discipline:

- **Track A — artifact evolution (new):** an RD-shaped outer loop over
  hypotheses/episodes, scored by *executed* provider results. New code under
  `src/spark_mcp/evolve/`, new MCP tool(s) in `server.py`, new tables via
  migration. This is the bulk of the plan (components A–E).
- **Track B — GEPA-for-skills repair (edits):** fix the rubric leak, retire
  the no-op operator, and strengthen the skill auto gate with a strict
  margin (component F). GEPA's genome stays procedure text; it is never
  pointed at trading strategies.

Shared discipline: every trial the loop dispatches goes through
`gates/service.py::backtest_run` (tombstone pre-check, trial governor,
budgets) and every promotion decision through `significance_gate` +
margin + the champion/rollback/approve machinery. The LLM may narrate
(observations, evaluation, `new_hypothesis` seed) but never decides.

## 2. Component breakdown with target files

### A. Execution-grounded fitness — `src/spark_mcp/evolve/fitness.py` (new)

Score candidates by executed results via the existing providers/backtest
seam, replacing fixture-text coverage as the artifact-evolution fitness.

- `fitness_from_episode(db, cfg, episode_id) -> FitnessReport`: reads the
  episode's executed `corrected_metrics` (written by `significance_gate`;
  never computed here per AGENTS.md rule 2) and composes the fitness record:
  `{sr, dsr_prob, p_value, M_family, T, skew, kurt, heavy_tail_caveat,
  degenerate, costs_basis}`, where `dsr_prob` is `corrected["dsr"]` (the
  Bailey–LdP probability form) and `p_value` is `corrected["p_value"]`
  (`1 - dsr`). Refuses (named `ToolError`) when `corrected_metrics` are
  missing or still `status: pending_gate`, when `simulated` is true and
  config forbids simulated fitness (new
  `ResearchConfig.evolution_require_live_provider`, default false for
  tests), or when costs are zero (reuse `CostModel` invariant — fitness on
  gross numbers is refused, closing the audit's with/without-cost skew).
  The cost-basis check reads `episodes.costs` (the `{bps, slippage,
  borrow, ...}` JSON written by `backtest_run`), not `metrics.*`.
- Held-out exclusion is structural (constraint 5), not caller discipline:
  fitness reads exactly three inputs — `corrected_metrics` (IS gate
  outputs; no OOS keys exist there), `episodes.costs`, and the single
  `metrics.simulated` provenance bit (a refusal input, never a score
  input). It never reads any other `metrics.*` key, so present-or-future
  OOS keys (`oos_ic`, `oos_*`) are invisible to fitness by construction.
  The same allowlist scopes `evolution_feedback.metrics_json` /
  `fitness_json`: IS-only, so LLM narration can never carry OOS values
  into the next variation.
- Pure function of (episode row, config): trivially deterministic.

Depends on: `ledger.get_episode`, `gates/service.py` (writes the inputs),
`providers/protocols.py` (spec discipline). No provider imports directly —
dispatch stays in the loop driver (component E) via `backtest_run`.

### B. Generative LLM variation — `src/spark_mcp/evolve/reflect.py` (new)

Reflective mutation (the honest GEPA/RD operator spark lacks) with a
deterministic fallback.

- `reflect_variation(*, parent, feedback, ancestors_render, rng, llm) ->
  VariationProposal`: when `llm` is available, one `LLMClient.complete`
  call (temperature 0 via `OpenAICompatibleLLM`; JSON-object response
  parsed with the tolerant balanced-brace approach used by explore
  scoring) asking for a *schema-conformant* successor: for hypotheses the
  full C6 field set (mechanism, direction, horizon, scope, falsifier,
  `predicted_ic`/`predicted_sigma`, DAG for causal) so `propose_hypothesis`
  can validate it; for skills a rewritten step list.
- **Deterministic fallback** (no LLM / parse failure / LLM raise): the
  existing syntactic operators (`drop_step`, failure-sketch `repair`,
  `split`, degraded `paraphrase`) selected round-robin from the seeded
  `rng`. Degraded path must be bit-reproducible: same (parent, feedback,
  seed) → same proposal.
- Every proposal records `{prompt_rendered, model_id, fallback_used,
  seed}` on the lineage/feedback row (judge-cache-style auditability).
- `llm.py` needs no changes (reuse the `LLMClient` protocol + `FakeLLM`).

### C. DAG lineage + branch points + parent selectors

RD ideas ported onto spark's ledger (audit seam 1 + 5); RD's `Trace` class
is *not* imported.

**C1. Lineage storage — `src/spark_mcp/storage.py` migration v11 + `src/spark_mcp/evolve/lineage.py` (new)**

- The `hypotheses` table already carries `parent_hypothesis_ids` (JSON
  list) — multi-parent capable with no schema change. `lineage.py` adds
  typed helpers over it: `ancestors(db, hypothesis_id)`,
  `children(db, hypothesis_id)`, `walk_to_root(db, hypothesis_id)`,
  `sota_ancestor(db, hypothesis_id)` — the argmax of stored `dsr_prob`
  over ancestors with a recorded `decision='promoted'` feedback row (a
  computed ledger query, never an LLM bit); ties break by earliest
  `episode_id` (deterministic).
- New table `evolution_lines` (the "current selection / branch from
  ancestor" pointer spark lacks):
  `line_id TEXT PK (elif_ ULID), family TEXT, head_hypothesis_id TEXT,
  head_episode_id TEXT, rollback_hypothesis_id TEXT,
  rollback_episode_id TEXT, status (active|abandoned|merged), created_at,
  updated_at`. One active line per family by default; branching inserts a
  new line whose head points at any ancestor (branch point). The rollback
  columns mirror the skill `rollback_target` status: on every promotion the
  prior head moves there before the new head is set, so a human veto can
  restore it (see E3).
- New table `evolution_runs` (run audit record):
  `run_id TEXT PK (erun_ ULID), line_id, seed, caps_json {population or
  beam width, max_rounds, ...}, started_at, finished_at, status,
  head_before/after`. Plus new ids in `src/spark_mcp/ids.py`
  (`new_evolution_line_id`, `new_evolution_run_id`).
- New table `evolution_feedback` (structured narrated trace, the middle
  form between spark's `mutation_op` string and RD's unbounded render):
  `feedback_id PK, run_id, hypothesis_id, episode_id, parent_ids_json,
  observations TEXT, hypothesis_evaluation TEXT, new_hypothesis TEXT,
  metrics_json (the IS numbers the narration interpreted — same allowlist
  as component A, never OOS), fitness_json (component A report),
  decision TEXT (promoted|rejected + computed reason), model_id,
  prompt_sha TEXT, created_at`. Narration fields are write-only audit: they
  are recorded after the decision and are never inputs to any future
  `promote.decide` call (constraint 2).
- All new rows carry `as_of`/`data_through` where data-dependent (PIT rule).

**C2. Parent selectors — `src/spark_mcp/evolve/selectors.py` (new)**

Deterministic explore/exploit policies over the DAG (port of the one good
RD selector idea; the coin flip stays behind):

- `select_parent(db, cfg, line, rng) -> ParentSelection`: exploit by
  default (head = latest passing champion-ancestor); explore when the
  recent SOTA hit-rate sags — `SOTAJump` rule mapped onto spark's own
  per-family stats (`exploration_state` consecutive_failures / kill-rule
  counters / posteriors, all already tracked): hit-rate below
  `ResearchConfig.evolution_sota_jump_threshold` over the last
  `evolution_sota_jump_window` rounds → start a new root (new
  `evolution_lines` row, `parent_hypothesis_ids=[]`). Hit-rate derives
  from recorded IS-gated promotion verdicts only — OOS values never enter
  selector inputs (constraint 5). The `rng` is consumed only for
  explicitly documented tie-breaks, in a fixed order.
- `jump_back(db, line, n_sota=2)`: resume from the n-th-last passing
  ancestor (deterministic ancestor walk; no randomness).
- All policies are pure functions of (ledger state, config, seeded rng)
  and return a reason string recorded on the feedback row. No time-boxed
  sub-trace accounting until a multi-branch runner exists (audit: skip).

### D. Two-level variation — `src/spark_mcp/evolve/variation.py` (new)

Separates *what to try next* from *how to implement it correctly* (RD's
HypothesisGen vs CoSTEER split, without the in-place-yield contract —
every function returns new objects, never mutates inputs):

- **Level 1 — idea level** (`vary_hypothesis`): takes the selected parent
  hypothesis + rendered ancestors + feedback and emits a successor
  hypothesis dict via component B (LLM reflective) or deterministic
  fallback (field-level edits: tighten/loosen horizon, negate direction
  with new falsifier, narrow scope — each preserving C6 validity).
  Inserted via `ledger.propose_hypothesis` (the C6 refusal rule is the
  schema gate; a refused proposal is recorded as a failed variation, not
  retried silently).
- **Level 2 — implementation level** (`vary_spec`): takes a hypothesis +
  parent episode and emits a `BacktestSpec` variant: param perturbations
  (`lookback`, `type` within the provider's supported set) drawn from the
  run's seeded `rng` in a fixed consumption order, spec-seed derivation by
  `sha1(parent_spec_seed|round)` (hash-based, consumes no rng state),
  universe/date-window moves within PIT discipline. Both streams are
  deterministic given (run seed, round); the hash derivation mirrors the
  `_op_gepa` content-seed precedent. Validated by pydantic
  (`BacktestSpec.model_validate`) before dispatch; invalid → recorded
  failure, loop continues. This is where "most iterations die in
  implementation" is handled: per-task execution errors from the provider
  feed the next level-2 variation (bounded retries, then escalate to a
  level-1 variation), mirroring CoSTEER's per-task evaluate→repair shape
  with spark's trial accounting (every dispatch is a counted trial; infra
  errors fail the trial without ticking the kill rule, per N6).
- Executor/runner agents stay QuantFlow-side; this module only varies the
  *spec*, never executes code itself.

### E. Loop driver + capped rendering + gate wiring

**E1. `src/spark_mcp/evolve/run.py` (new)** — the RD-shaped outer loop:

```
for round in range(max_rounds):
    parent   = selectors.select_parent(...)        # explore/exploit
    hyp      = variation.vary_hypothesis(...)       # level 1
    spec     = variation.vary_spec(...)             # level 2
    outcome  = gates.backtest_run(...)              # trial governor + budgets
    verdict  = gates.significance_gate(...)         # DSR, computed
    fitness  = fitness.fitness_from_episode(...)    # component A
    decision = promote.decide(...)                  # margin + significance (E3)
    lineage.record_feedback(...)                    # narrated trace row
    update line head / champion / rollback_target
```

One `random.Random(seed)` threads through all rounds. Provider
determinism follows from the spec: `vary_spec` sets every dispatched
`BacktestSpec.seed` from the run seed (hash derivation, §2.D), and
`SimulatedProvider`/`RecordedProvider` consume only the spec seed
(`_spec_seed`; no wall-clock, no global RNG) — same (seed, round) replays
identical metrics. Stopping rules: `max_rounds`, consecutive
computed-rejections ≥ `evolution_max_consecutive_rejections`, kill-rule
family death, budget exhaustion (budgets raise → graceful run close with
status). No human prompt inside (constraint 3): the driver never calls
`approve` and never blocks on input; veto applies post-run at the line
head (see E3). Every step writes trajectory events (event timestamps are
audit-only, never scored state).

**E2. `src/spark_mcp/evolve/render.py` (new)** — capped, summarized trace
rendering for prompts (audit seam 6):

- Renders *ancestors of the selected parent only* (not the whole forest),
  concise fields + metric tables (no stdout logs), newest-last with
  explicit "Trial 1..n" ordering (do not repeat RD's insert(0)+reverse
  bug — order is unit-tested).
- Token budget: `ResearchConfig.evolution_render_max_tokens` with an
  explicit truncation policy (drop oldest non-SOTA ancestors first, record
  `truncated_from_n` in the render header). Budgets count tokens by the
  configured estimator, never a silent character slice (the audit's judge
  failure mode).
- `rendered_sha` stored on each feedback row; full prompt text stored in
  `generated_text_cache` (existing table) keyed by sha for audit replay.

**E3. Margin + significance gating — `src/spark_mcp/gates/service.py`
(extend) + `src/spark_mcp/evolve/promote.py` (new)**

- `promote.decide(db, cfg, challenger_episode_id, incumbent_episode_id |
  None) -> PromotionDecision`: promote iff (a) the challenger's
  `significance_gate` verdict has `pass` True (the impl's own conjunction
  `ok_dsr ∧ SR > 0 ∧ p < dsr_alpha`; `T`/`min_T` reported alongside) AND
  its `promotion_blockers` list is empty — checked by `decide`, since the
  impl reports C6 blockers verbatim without flipping `pass` — AND (b)
  numeric margin over incumbent: `challenger.dsr_prob -
  incumbent.dsr_prob ≥ evolution_dsr_margin` (default 0.02) AND
  `challenger.sr - incumbent.sr ≥ evolution_sr_margin` (default 0.0 —
  documented; DSR margin is the binding constraint). First promotion (no
  incumbent): gate pass + blockers empty +
  `evolution_first_promotion_floor` on DSR prob. Ties never promote
  (strict `>` on both margins). The signature takes episode IDs only:
  narration text is not an input (constraint 2, structural).
- Reuses `_significance_gate_impl` verbatim (no second statistic to drift);
  the margin check is a pure comparison on two stored `corrected_metrics`
  rows, using each row's stored gate-time `M_family` (never recomputed —
  trials recorded between gate and decide must not shift the comparison).
- Held-out at promotion (constraint 5) mirrors the skill-gate shape
  (held-out reported + gap-alarmed, never the comparison input): the
  decision's `oos_report` advisory carries any `oos_*` keys found on the
  challenger's `metrics` (e.g. `oos_ic`), but the promote boolean's inputs
  are IS significance + margin only. No provider emits OOS metric keys
  today (`SimulatedProvider` returns IS-only metrics), so `oos_report` is
  `{available: False}` until an OOS producer exists; engaging a future gap
  veto then requires a new `ResearchConfig` constant per AGENTS.md, and
  OOS values still never reach `evolution_feedback` rows. Rationale (same
  as F3): the loop observes promotion history, so gating the boolean on
  OOS would make held-out an optimization target.
- Champion/rollback/approve flow mirrors `skills.decide_promotion`, with
  line-head bookkeeping instead of skill rows: on promote, the prior head
  moves to `evolution_lines.rollback_*` before the new head is set (same
  transaction); a direct `approvals` INSERT with
  `target_type='evolution'`, `target_id=line_id`, `decision='approved'`,
  `decided_by='gate'` is written as the human-notify surface (mirroring
  the decide_promotion gate-notify INSERT — the table accepts any
  `target_type` TEXT, but its `storage.py` comment and the `approve`
  target check must be extended). Veto path (new `elif` branch in
  `skills.approve`, same file as the skill/belief branches):
  `decision='rejected'` on an `evolution` target restores the rollback
  head and marks the vetoed head abandoned; `decision='approved'`
  ratifies. `server.py`'s `approve` docstring enumeration gains
  `evolution`. Adoption into live books stays a separate human approval;
  the Gateway/adoption contracts are unchanged.

**E4. MCP surface — `src/spark_mcp/server.py` (extend, thin wrappers)**

- `evolution_run(family, hypothesis_seed_id?, caps?, seed?)` → starts a
  bounded run (E1), returns `{run_id, line_id, rounds, head, decisions}`.
- `evolution_status(run_id)` → run record + per-round decisions + current
  head/SOTA. (`research_history`-style read; no new write path.)
- No new write path to live systems: Gateway/adoption contracts unchanged.

### F. GEPA-for-skills repair (Track B)

- **F1. `add_check_step` (constraint 6)** — `src/spark_mcp/evolve/gepa.py`:
  delete the operator including its rubric-text read (the
  `f"Verify: {uncovered[0]}."` verbatim injection); the symbol
  `_add_check_step`/`"add_check_step"` must not survive anywhere in `src/`
  (regression test greps for it). Replacement operator `add_guard_step`:
  appends a guard step conditioned on *failure evidence* — the most common
  Tier-2 `failure_sketches` error class for the skill's source episodes,
  reusing `_op_repair`'s exact sentence template
  (`"Guard against {error_class}: check data coverage, preconditions, and
  inputs before executing this procedure (observed in {count} source
  episode(s))."`), falling back to the existing generic audit sentence
  `"Record the inputs, assumptions, and outputs of this step for audit."`
  (pinned as a module constant) when no failure evidence exists.
  Signature change: `_add_check_step(ind, fixtures, rng)` →
  `_add_guard_step(ind, failure_sketches, rng)`; `run_gepa` takes
  `failure_sketches: list[dict] | None = None` (default keeps call-sites
  compiling; `_op_gepa` in skills.py passes
  `_tier2_failure_sketches(db, source)`). Routing rule: `_mutate` never
  passes `fixtures` to any mutation op — fixtures route to the scorer
  (`score_population`/`fixture_score`) only — so no future op can re-read
  rubric text. `MUTATION_OPS` becomes
  `("add_guard_step", "drop_step", "reflect")` — see F2.
- **F2. `rotate_steps`**: dead weight under an order-insensitive scorer.
  Replace the slot with `reflect` (component B applied to skill text: LLM
  rewrite when available, degraded reorder+clarify otherwise — reusing
  `_op_paraphrase`'s degraded path as the fallback body). Do not make the
  scorer order-sensitive instead: word-order sensitivity would be a new
  semantic claim needing its own validation; the reflective op is the
  audit-blessed direction.
- **F3. Skill auto-gate margin** — `src/spark_mcp/skills.py`
  `decide_promotion(decision='auto')`: replace `ch_mean >= old_mean` with
  `ch_mean - old_mean > replay_promotion_margin` (new
  `ResearchConfig.replay_promotion_margin`, default 0.05) and
  `ch_mean >= floor` with `ch_mean - floor > replay_promotion_margin`.
  Ties no longer promote. Held-in comparison stays (held-out is promotion
  *evidence*, reported + gap-alarmed, but the audit's "gate on held-out
  mean" repair is NOT adopted: gating on held-out would make held-out an
  optimization target, violating constraint 5 — the margin + gap alarm is
  the correct combination).
- **F4. Fixture growth note**: the suite is still 6 hand-written items.
  Auto-promotion stays weak until fixtures grow from real failures (seed
  docstring's own direction). Track as follow-up work, not this plan's
  gate: add `docs` note + a `consolidate` gap-report hook listing
  fixture-growth as an open gap when `len(fixtures) < replay_min_fixtures`
  (new config, default 6 — alarm only, never blocks).

### G. Config — `src/spark_mcp/config.py` (`ResearchConfig` additions)

All new knobs with defaults (AGENTS.md: no literals in algorithm code):

```
evolution_max_rounds: int = 8
evolution_max_consecutive_rejections: int = 3
evolution_dsr_margin: float = 0.02
evolution_sr_margin: float = 0.0
evolution_first_promotion_floor: float = 0.95   # DSR prob floor, first promotion
evolution_sota_jump_window: int = 6
evolution_sota_jump_threshold: float = 0.2      # hit-rate below → new root
evolution_render_max_tokens: int = 4000
evolution_require_live_provider: bool = False   # True in prod: refuse simulated fitness
replay_promotion_margin: float = 0.05           # F3
replay_min_fixtures: int = 6                    # F4 alarm threshold
```

### H. Independence gate — `tests/independence/` (extend)

Add `"rdagent"` to `FORBIDDEN_MODULES` in
`tests/independence/test_independence_gate.py`: that one entry enforces
constraint 1 through both existing blocking tests (the AST import scan and
the `sys.modules` runtime test), mirroring the dspy/hermes entries. The
current gate has no path-read/subprocess/dependency scan, so add a third
blocking test in the same file: `test_no_rdagent_references_in_source`
fails on any case-insensitive `rdagent`/`RD-Agent` literal under `src/`
(no allowlist), and asserts `rdagent` appears in neither
`pyproject.toml`'s dependencies nor any lockfile. Vendored RD code would
trip the literal scan by its module docstrings/imports.

## 3. APIs and data models

### 3.1 New public functions (service layer; MCP tools wrap these)

```python
# evolve/fitness.py
@dataclass(frozen=True)
class FitnessReport:
    episode_id: str; sr: float; dsr_prob: float; p_value: float
    M_family: int; T: int; skew: float; kurt: float
    heavy_tail_caveat: bool; degenerate: bool; simulated: bool
    costs_basis: dict  # echoed CostModel, must sum > 0

def fitness_from_episode(db, cfg, episode_id) -> FitnessReport  # raises ToolError

# evolve/selectors.py
@dataclass(frozen=True)
class ParentSelection:
    hypothesis_id: str | None  # None = new root
    episode_id: str | None
    policy: str                # 'exploit_head' | 'sota_jump_new_root' | 'jump_back'
    reason: str

def select_parent(db, cfg, line_id, rng) -> ParentSelection
def jump_back(db, line_id, n_sota=2) -> ParentSelection

# evolve/variation.py
def vary_hypothesis(db, cfg, parent, ancestors_render, feedback, rng, llm) -> dict
def vary_spec(db, cfg, hypothesis, parent_episode, round_no, rng) -> BacktestSpec

# evolve/reflect.py
@dataclass(frozen=True)
class VariationProposal:
    body: dict | list[str]     # hypothesis fields or procedure steps
    model_id: str | None       # None on deterministic fallback
    prompt_sha: str | None
    fallback_used: bool

def reflect_variation(*, parent, feedback, ancestors_render, rng, llm,
                      schema: str) -> VariationProposal  # schema: 'hypothesis' | 'skill_steps'

# evolve/render.py
@dataclass(frozen=True)
class AncestorRender:
    text: str; n_ancestors: int; truncated_from_n: int; sha: str

def render_ancestors(db, cfg, hypothesis_id, *, max_tokens) -> AncestorRender

# evolve/promote.py
@dataclass(frozen=True)
class PromotionDecision:
    promote: bool; reason: str
    challenger: FitnessReport; incumbent: FitnessReport | None
    dsr_margin: float; sr_margin: float
    oos_report: dict  # advisory only: {available: False} until an OOS
                      # producer exists; never an input to `promote`

def decide(db, cfg, challenger_episode_id, incumbent_episode_id | None) -> PromotionDecision

# evolve/run.py
def run_evolution(db, cfg, provider, *, family, seed_hypothesis_id=None,
                  max_rounds=None, seed=0, llm=None) -> dict
def evolution_status(db, run_id) -> dict

# evolve/lineage.py
def ancestors(db, hypothesis_id) -> list[str]
def children(db, hypothesis_id) -> list[str]
def sota_ancestor(db, hypothesis_id) -> dict | None
def create_line(db, family, head_hypothesis_id=None, head_episode_id=None) -> dict
def record_feedback(db, run_id, hypothesis_id, episode_id, parent_ids,
                    narration, fitness, decision, model_id, prompt_sha) -> dict
```

### 3.2 Storage (migration v11, additive only)

- `evolution_lines`, `evolution_runs`, `evolution_feedback` per §2.C1.
- No changes to existing tables' columns. `hypotheses.parent_hypothesis_ids`
  already supports DAG parents; `episodes` needs no new columns (fitness
  inputs already on `metrics`/`corrected_metrics`).
- `ids.py`: `elif_` and `erun_` prefixed ULIDs (convention: type prefixes).

### 3.3 MCP tools (server.py, thin wrappers per N13)

- `evolution_run(family, seed_hypothesis_id?, max_rounds?, seed?,
  client_id?)` — idempotent on `client_id` via `tool_effects` (repo
  convention). Returns run summary; full trace via `evolution_status`.
- `evolution_status(run_id)` — read-only.
- Tool contracts documented in server.py docstrings mirroring existing
  tool style (refusal rules, determinism notes).

## 4. Migration and back-compat

- **Schema**: v11 migration creates three new tables; existing DBs migrate
  forward with zero data changes. No backfill needed (no new lineage is
  inferred for pre-existing hypotheses; their `parent_hypothesis_ids`
  already parse).
- **`run_gepa` signature**: additive kwarg `failure_sketches=None`; old
  call-sites behave as before except the operator rename
  (`add_check_step` → `add_guard_step`; with `failure_sketches=None` the
  op appends the generic audit sentence — the old "everything covered"
  branch). `MUTATION_OPS` tuple changes: any caller iterating it by name
  (only `_mutate` + tests) is updated in the same commit.
- **`replay_score_cache`**: unaffected — `fixture_score` semantics don't
  change, only mutation behavior does, so cached scores stay valid.
- **Skill auto gate (F3)**: behavior change by design (margin + strict
  inequality). Existing tests asserting tie-promotion (`>=`) must be
  updated to the new rule — flagged explicitly in review, not snuck in:
  list each touched assertion in the commit message.
- **Config**: all new fields have defaults; old YAML files validate
  unchanged (`config.example.yaml` gains a commented evolution section).
- **`approve` extension**: no schema change (the `approvals` table takes
  any `target_type` TEXT); the `storage.py` column comment, the
  `skills.approve` target check, and the `server.py` docstring enumeration
  gain `evolution`. Existing `{belief, skill, adoption}` behavior is
  untouched.
- **No MCP tool removals or signature changes**; two tools added.

## 5. Test strategy (mirror `test_gepa.py` / `test_skills.py` conventions)

New files: `tests/test_evolution_lineage.py`, `tests/test_evolution_selectors.py`,
`tests/test_evolution_fitness.py`, `tests/test_evolution_variation.py`,
`tests/test_evolution_run.py`, `tests/test_evolution_promote.py`, plus
extensions to `tests/test_gepa.py` (F1/F2) and `tests/test_skills.py` (F3).

- **Determinism**: every run-level test executes twice (fresh DBs, same
  seed) and asserts byte-identical decisions/heads *excluding ULID primary
  keys and timestamp columns* (identity/audit metadata, §0.4); a dedicated
  test runs with `PYTHONHASHSEED` varied. `FakeLLM` scripted replies for
  the reflective path; fallback path tested with `llm=None`.
- **Fitness (A)**: refusal cases — missing/`pending_gate`
  `corrected_metrics`, zero-cost basis, simulated-with-`require_live`;
  IS/OOS split test (an episode carrying `oos_ic` + `oos_*` keys yields a
  fitness record identical to the same episode without them; only
  `metrics.simulated` among `metrics.*` is ever read).
- **Selectors (C2)**: exploit-head default; sota-jump triggers exactly at
  the threshold boundary (window/hit-rate edge cases); `jump_back(2)`
  lands on the 2nd-last passing ancestor; new-root creates a second
  active line; `sota_ancestor` tie-breaks by earliest `episode_id`; all
  policies deterministic across reruns.
- **Variation (D)**: level-1 output always passes `propose_hypothesis`
  validation or is recorded refused (property-style: N seeded parents);
  level-2 output always validates as `BacktestSpec`; no in-place mutation
  (parent objects deep-compared before/after); provider-error →
  level-2 retry → level-1 escalation sequence with trial rows counted.
- **Promote (E3)**: all branches — first-promotion floor, margin pass,
  margin fail, tie (no promote), DSR fail, C6 blockers (a passing verdict
  with non-empty `promotion_blockers` must not promote), degenerate
  inputs; stored gate-time `M_family` used (insert counted trials between
  gate and decide, assert the comparison still uses the stored M);
  `oos_report` advisory present with `{available: False}` and never
  affecting the boolean; veto test — `approve(target_type='evolution',
  decision='rejected')` restores the rollback head, `'approved'` ratifies,
  and existing `{belief, skill, adoption}` approve paths are unaffected.
- **Run (E1)**: happy path on `SimulatedProvider` (deterministic via
  `_spec_seed`) and on `RecordedProvider` cassettes; stopping rules
  (max_rounds, consecutive rejections, kill-rule, budget fail_closed);
  feedback rows written with prompt_sha + model_id and IS-only
  `metrics_json` (assert no `oos_*` key survives on any feedback row even
  when the episode carries them); trajectory events emitted; no human
  interaction (run with no stdin/approvals available; driver never calls
  `approve`).
- **Render (E2)**: ancestor-only (siblings excluded), newest-last order
  pinned, truncation policy (oldest non-SOTA dropped first,
  `truncated_from_n` exact), budget respected.
- **GEPA repair (F)**: `add_guard_step` never emits rubric substrings
  (assert against every fixture rubric + reference keyword — the
  regression test for the leak — including the pinned fallback sentence);
  a source grep test fails if `_add_check_step`/`"add_check_step"` survives
  anywhere under `src/`; `_mutate` routing test (no op receives `fixtures`);
  `reflect` fallback path deterministic; skill auto-gate margin branches
  incl. tie-rejects; full-suite M1 rule still enforced on promote.
- **Independence (H)**: `FORBIDDEN_MODULES` gains `rdagent` + the new
  literal/dependency scan test; full suite `python -m pytest -q` +
  `ruff check src tests` green.
- **Leakage/PIT**: new-table rows carry `as_of`; retrieval-before-ranking
  unaffected (no new retrieval path). Reuse existing leakage tests.

## 6. Non-goals (explicitly out)

1. No dependency on, import of, or vendored code from RD-Agent (constraint 1).
2. No LLM-as-selection-function: no "Replace Best" boolean, no LLM SOTA
   pick, no submit-time LLM selector (constraint 2).
3. No human-in-the-loop inside a run: no approval prompts, no interactive
   hypothesis editing (constraint 3).
4. No GEPA genome change: GEPA still evolves procedure text only; never
   pointed at strategies/factors/models.
5. No CoSTEER literal port: no in-place-yield contracts, no multiprocessing
   evolving strategy, no untested knowledge-base RAG. Success/failure
   knowledge stays as spark tombstones + `evolution_feedback` rows.
6. No unbounded full-history prompts; no stdout-log rendering; no
   hardcoded curriculum strings.
7. No new live-system write path: Gateway manifest + adoption approvals
   unchanged; spark still never touches live books.
8. No fixture-suite growth beyond the F4 alarm hook (follow-up work).
9. No backtest-math changes: providers own all returns/IC/risk math
   (AGENTS.md rule 2); fitness only reads provider outputs.
10. No `BackJumpCKPSelector` coin flip, no time-boxed sub-trace accounting,
    no multi-branch parallel runner (single-threaded loop; `db.locked()`
    convention inherited from the gates service).

## 7. Suggested build order (dependency order, reviewable slices)

1. H (independence scan) + G (config knobs) + v11 migration + `ids.py` —
   no behavior change; proves the scaffolding.
2. C1 lineage helpers + C2 selectors + render (E2) — pure functions over
   the ledger; unit-testable without providers.
3. A fitness + E3 promote (incl. the `approve(target_type='evolution')`
   veto branch + rollback columns) — the computed decision layer; needs
   only `SimulatedProvider` episodes in tests.
4. B reflect + D variation (level 1, then level 2) — generative core.
5. E1 run driver + E4 MCP tools — wiring + stopping rules.
6. F GEPA repair + skill-gate margin — behavior changes, own commit with
   per-assertion test updates listed.
7. Full suite + `milestone-review` skill → verdict in STATUS.md → commit
   (AGENTS.md rule 5).

## 8. Self-check revision note (2026-10-12; reviewer unavailable)

The assigned reviewer was unavailable, so this plan was self-checked
against the six hard constraints by re-reading the cited repo seams
(`gates/service.py` backtest/gate impl, `skills.py` gate/approve ops,
`evolve/gepa.py` operators, `providers/simulated.py` metrics,
`tests/independence/test_independence_gate.py`). Findings, all fixed
in place above:

1. (Constraint 1) §H overstated the existing gate: the test file has only
   an AST import scan + a `sys.modules` test — no path-read/subprocess/
   dependency scan. Fixed: `rdagent` joins `FORBIDDEN_MODULES` (covers
   both existing tests) plus a specified new literal/dependency scan test.
2. (Constraint 2) E3's pass parenthetical did not match the impl:
   `_significance_gate_impl` reports C6 blockers verbatim without flipping
   `pass` (`pass = ok_dsr ∧ SR > 0 ∧ p < alpha`). Fixed: `decide` checks
   `verdict["pass"]` AND empty `promotion_blockers`; narration is
   structurally excluded (episode-IDs-only signature, write-only audit
   fields). The veto path was hand-waved while `approve` rejects unknown
   target types — fixed with explicit `rollback_*` columns, the
   `target_type='evolution'` branch, and the comment/check/docstring
   updates. M is pinned to the stored gate-time value.
3. (Constraint 3) The closed-loop claim is now testable: the driver never
   calls `approve` and never blocks on input; veto applies post-run at
   the line head (§2.E1, §5).
4. (Constraint 4) Reproducibility scope was unstateable as written: ULID
   keys and timestamps would break byte-identity. Fixed: scope excludes
   id/time columns by column (§0.4, §5); rng-vs-hash responsibilities in
   selectors/variation are pinned; provider determinism is derived from
   the spec seed.
5. (Constraint 5) The IS/OOS split was caller discipline with no producer
   analysis. Fixed: fitness reads only `corrected_metrics` +
   `episodes.costs` + the `metrics.simulated` provenance bit (structural
   OOS invisibility, incl. a seeded-`oos_ic` identity test); feedback rows
   and selector stats are IS-only; promotion carries an `oos_report`
   advisory (`{available: False}` — no provider emits OOS keys today)
   that never feeds the boolean, mirroring the skill held-out discipline.
6. (Constraint 6) F1 now deletes the symbol (grep regression test),
   reuses `_op_repair`'s exact guard template, pins the fallback sentence
   as a module constant, and bars `fixtures` from every mutation op via
   the `_mutate` routing rule.

No open items: every constraint maps to plan text plus a named test.
