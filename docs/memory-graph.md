# Memory graph — node & edge types (as implemented)

This document describes the method graph **as it exists in code today**
(`src/spark_mcp/graph.py` + the DDL in `src/spark_mcp/storage.py`). It is
documentation only: no storage, schema, or behaviour is changed by it.
Plan references (§5.1, §2.7, §16.1) are the design anchors named in the
source docstrings.

## Storage

Two tables (plus an FTS5 shadow) in the SQLite store:

### `graph_nodes`

| Column | Type | Notes |
|---|---|---|
| `node_id` | `TEXT PRIMARY KEY` | Minted by `ids.new_node_id()` unless the caller supplies one (`add_node(..., node_id=...)`). |
| `type` | `TEXT NOT NULL` | Node kind — see *Node types* below. Indexed (`idx_graph_nodes_type`). |
| `label` | `TEXT NOT NULL` | Human-readable name; the FTS5-indexed field. Matched case/space-insensitively via `normalize_name` (lower-cased `[a-z0-9]+` tokens joined by `_`). |
| `family` | `TEXT` (nullable) | Research family for family-scoped novelty (§7). Technique cards carry their card `family`; concept nodes are inserted with `family=None`. Indexed (`idx_graph_nodes_family`). Additive to the plan DDL. |
| `mech_tags` | `TEXT NOT NULL DEFAULT '[]'` | JSON array of mechanism tags. On read (`get_node`) it is decoded back to a Python `list`. |
| `summary` | `TEXT` (nullable) | Free-text summary; part of the embedding/lexical text (`node_text`). |
| `as_of` | `TEXT NOT NULL` | PIT visibility timestamp (`YYYY-MM-DDTHH:MM:SSZ`): a row is visible at `T` iff `as_of <= T`. Seeds use `1000-01-01T00:00:00Z`. |
| `created_at` | `TEXT NOT NULL` | Wall-clock insertion time (`now_iso()`). |

### `graph_edges`

| Column | Type | Notes |
|---|---|---|
| `edge_id` | `TEXT PRIMARY KEY` | Minted by `ids.new_edge_id()`. |
| `src` | `TEXT NOT NULL REFERENCES graph_nodes(node_id)` | Source node; must exist or `add_edge` raises `ValueError("unknown endpoint …")`. Indexed. |
| `dst` | `TEXT NOT NULL REFERENCES graph_nodes(node_id)` | Destination node; same existence check. Indexed. |
| `predicate` | `TEXT NOT NULL` | Edge kind — see *Edge types (predicates)* below. Indexed. |
| `weight` | `REAL` (nullable) | Optional edge weight; `None` when not supplied. |
| `evidence_episode_ids` | `TEXT NOT NULL DEFAULT '[]'` | JSON array of episode ids carrying provenance for the edge (M5/N6). Decoded to a list on read (`get_edge`). |
| `valid_from` | `TEXT NOT NULL` | Start of temporal validity (set from the edge `as_of`). |
| `valid_to` | `TEXT` (nullable) | Invalidation time; `NULL` = still valid. Invalidation is by timestamp, not deletion. |
| `as_of` | `TEXT NOT NULL` | PIT timestamp of the edge assertion. |
| `created_at` | `TEXT NOT NULL` | Insertion time (`now_iso()`). |

### `graph_nodes_fts` (FTS5)

Virtual table `fts5(label, mech_tags, summary, content='graph_nodes')`
kept in sync by the `graph_nodes_ai` AFTER INSERT trigger. It exists so
degraded-mode (no-embedder) recall is not blind to mechanism vocabulary —
labels alone never carry it.

### Embeddings (N3)

Embeddings never land on the base tables. `graph.embed_text` writes to
the `vec_embeddings` virtual/fallback table keyed by
`(subject_type, subject_id, model, model_version)` with
`subject_type='graph_node'` for graph nodes, so a model change can never
silently re-scale novelty or cached scores. `node_text(node)` — label +
summary + `mech_tags` joined by spaces — is the embedded/scored text.

## Node types

`add_node(db, type, label, ...)` accepts **any string** for `type`; there
is no enum or CHECK constraint in code or DDL. The DDL comment records
the intended vocabulary as `technique|concept|domain|...` (open-ended).

The types actually produced and consumed today are:

| `type` value | Produced by | Meaning |
|---|---|---|
| `technique` | `import_seeds`: one node per card in `seeds/techniques.yaml` | A method/technique card. Carries the card `family`, merged `mech_tags` (the card's `mechanism_tags` plus its `assumptions`, `failure_modes`, and `analog_domains` values), and a `summary` synthesised as `assumptions: …; failure_modes: …` when present. This is the candidate type scored by `explore_methods`/`suggest_next` and the only type the analogy engine scans. |
| `concept` | `import_seeds`: one node per card in `seeds/concepts.yaml` | A shared mechanism-concept atom (e.g. *Stationarity*, *Trend persistence*). Inserted with `family=None`, the card's `mechanism_tags` as `mech_tags`, and the card `summary` verbatim. Concept nodes are the usual `dst` of `ASSUMES`/`FAILS_WHEN` edges and ground structural-overlap novelty. |

No other `type` value is created by current code. Consumers filter
explicitly: `retrieval` defaults to `node_types=("technique", "concept")`,
while `explore` and `analogy` query `type='technique'` only. A future
`domain` (or any other) type would be stored without code changes but
would be invisible to those consumers until they name it.

Node helpers in `graph.py`:

- `add_node(...) -> dict` — insert + optional embed-on-write; returns the
  stored row via `get_node` (with `mech_tags` decoded).
- `get_node(db, node_id) -> dict` — raises `ValueError` for an unknown id.
- `find_node_by_label(db, label) -> dict | None` — `normalize_name`
  equality over all nodes; used for seed-import idempotence and
  reference resolution.
- `normalize_name(text)` — `"_".join(re.findall(r"[a-z0-9]+", text.lower()))`.
- `node_text(node)` — label + summary + mech tags, space-joined.

## Edge types (predicates)

The plan (§2.7) defines 12 predicate types; **6 are active at launch**.
The active set is the module constant in `graph.py`:

```python
PREDICATES = frozenset({
    "SOLVES", "FAILS_WHEN", "COMPOSES_WITH",
    "DIAGNOSES", "ASSUMES", "DERIVED_FROM",
})
```

| Predicate | Meaning (as used) | Created by seed import? |
|---|---|---|
| `SOLVES` | Source method solves / addresses the destination (problem/concept). | No — available via `add_edge` only. |
| `FAILS_WHEN` | Source fails under the destination condition/concept. | Yes — from each technique card's `failure_modes` (technique → concept). |
| `COMPOSES_WITH` | Source composes with the destination method; treated as symmetric — seed import inserts **both** directions `(src, dst)` and `(dst, src)`. | Yes — from each technique card's `composes_with` (technique → technique, both ways). |
| `DIAGNOSES` | Source diagnoses the destination. | No — `add_edge` only. |
| `ASSUMES` | Source assumes the destination concept. | Yes — from each technique card's `assumptions` (technique → concept). |
| `DERIVED_FROM` | Source is derived from the destination. | No — `add_edge` only. |

The other six plan predicates are **not** in `PREDICATES` and are
rejected: `add_edge` raises
`ValueError("unknown predicate …; active predicates are [...]")` naming
the sorted active list. The tests pin this (`CAUSES`,
`correlates_with` are refused) and pin that exactly the six above are
accepted.

Edge helpers in `graph.py`:

- `add_edge(db, src, dst, predicate, weight=None, evidence_episode_ids=None, as_of=None) -> dict`
  — validates the predicate against `PREDICATES`, validates that both
  endpoints exist, then inserts with `valid_from = as_of`,
  `valid_to = NULL`.
- `get_edge(db, edge_id) -> dict` — raises `ValueError` for an unknown
  id; `evidence_episode_ids` decoded to a list.
- `edge_profile(db, node_id, as_of) -> set[tuple[str, str]]` — the
  structural fingerprint used by §7 novelty: the set of
  `(predicate, dst_type)` pairs over edges visible at `as_of`
  (`valid_from <= as_of AND (valid_to IS NULL OR as_of < valid_to)
  AND e.as_of <= as_of`). Empty when the node is ungrounded.

## Seed import (`import_seeds`)

`import_seeds(db, embedder=None, seeds_dir=None, as_of=None)` loads the
seed corpus (§16.1) into the two tables above:

1. `seeds/concepts.yaml` cards → `type='concept'` nodes.
2. `seeds/techniques.yaml` cards → `type='technique'` nodes.
3. Edges derived from card fields, resolved by `normalize_name`
   against the nodes just created (plus any pre-existing nodes):
   `composes_with` → `COMPOSES_WITH` (both directions),
   `assumptions` → `ASSUMES`, `failure_modes` → `FAILS_WHEN`.

References that do not resolve are **counted, not fatal**
(`stats["unresolved_refs"]`). Import is idempotent: a card whose
normalised label already exists is skipped (`skipped_existing`) and
reused as an edge endpoint. Every created node is embedded on write
when an embedder is configured (`stats["embeddings"]`). The returned
stats dict is
`{techniques, concepts, nodes, edges, embeddings, unresolved_refs, skipped_existing}`.
