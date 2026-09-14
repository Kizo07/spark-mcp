# QuantFlow §15.1 change-list verification (read-only, 2026-09-13)

Verified against <peer: quantflow> by read-only
inspection (no quantflow files modified; repo AGENTS.md respected). This
record satisfies review nit N8 (tier evidence must live in-repo).

| # | Row | Verdict | Evidence |
|---|---|---|---|
| 1 | Task-lifecycle `run_completed` hook event (schema + version) with difficulty signals | **PARTIAL** | Extension task-lifecycle exists: `TaskLifecycleContributor.on_task_start/on_task_stop` (backend/packages/extension-api/deerflow_extension_api/contracts.py:70-86), dispatched from the run worker (backend/packages/harness/deerflow/runtime/runs/worker.py:941-955, 1579-1589), timeout-guarded fan-out (backend/packages/harness/deerflow/extensions/notify.py:271-310); internal `on_run_completed` (worker.py:588, 1562-1566). But no event named `run_completed`, payloads (`TaskInfo` contracts.py:57-67, `TaskOutcome` contracts.py:51-54) carry no schema/version, and `difficulty` has zero matches in the backend. |
| 2 | Hook payload: `agent_assembly_fingerprint` + spec version | **PARTIAL** | Literal absent (0 matches). Fingerprint infra exists: `AgentAssemblyDescriptor.fingerprint` (backend/packages/extension-api/deerflow_extension_api/assembly.py:66-113), built at backend/packages/harness/deerflow/agents/assembly_descriptor.py:441-502, delivered via `on_agent_assembled` — but only at assembly time; task-lifecycle payloads do not carry it; no spec version anywhere (nearest: `prompt_template_id="deerflow-lead-agent-v1"`). |
| 3 | Hook payload: active skill versions (`skills_activated`) | **ABSENT** | 0 matches. `Skill` dataclass has no version field (backend/packages/harness/deerflow/skills/types.py:41-57); `version` is an allowed SKILL.md frontmatter key (skills/frontmatter.py:25) the model never parses; assembly records `enabled_skills` as bare names + `skill_catalog_hash` (assembly_descriptor.py:485,499). |
| 4 | `quant-desk` skill call sites for the four Spark tools | **EXISTS (skill) / ABSENT (call sites)** | Skill on disk: skills/custom/quant-desk/SKILL.md (calls `alpha_engine`/`kizonlp`/`reportforge` tools only). `explore_methods`/`suggest_next`/`register_hypothesis`/`log_outcome`: 0 matches repo-wide. |
| 5 | Gateway skill-write API (versioned, with manifest) | **PARTIAL** | Write API exists: backend/app/gateway/routers/skills.py — POST /api/skills/install (:283), install/upload (:300), PUT /api/skills/custom/{name} with scan + history (:413), DELETE (:453), rollback (:509). Not versioned (no version field on any model); no skill-package manifest (the `_MANIFEST_VERSION = 1` in skills/projection.py:35 is an internal projection, not a write-API manifest). |

**Consumption path for a future full tier:** quantflow already has a general
MCP client (backend/packages/harness/deerflow/mcp/session_pool.py uses
`langchain_mcp_adapters`, pinned at backend/packages/harness/pyproject.toml:24);
a Spark registration in `extensions_config.json` needs no new client
machinery.

**Consequence:** every row currently resolves to the plan's Spark-side
fallback — Phase 4 closes at the *fallback tier* (Spark-side receiver +
manual close-out), with the full-integration tier (a real quantflow-emitted
hook observed end-to-end) open until quantflow lands rows 1-3 and a desk
call site (row 4). Row 5 matters again at Phase 6 (Gateway write-path).
