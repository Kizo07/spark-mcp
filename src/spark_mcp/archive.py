"""QD archive — native MAP-Elites (pure-Python grid; plan §10; pyribs is not
installable on this machine's index and is an optional accelerator only).
on this machine, so the grid is native).

Encoding (§10.1, descriptors v1): `family × horizon × turnover`, categorical,
integer indices into the config-named bin lists in ResearchConfig
(qd_family_bins × qd_horizon_bins × qd_turnover_bins, default 8×6×4 = 192
cells; qd_grid pins the shape). `cell_key = "f:h:t"`; the `descriptors` JSON
on every qd_cells row stores the resolved names so rows are self-describing.

SQL is the durable source of truth (M7): an Archive is rebuilt from qd_cells
at construction and held in memory per process; insert() writes through to
qd_cells inside the caller's transaction boundary (storage.transaction) — no
async twin-writer divergence path exists.

Elite score (§10.2, M12): per-type reward sets, weights in ResearchConfig —
    full (predictive/portfolio/risk): oos .30 + deflated_significance .20
        + robustness .15 + orthogonality .15 + novelty .10 + efficiency .10
    causal: adds stability, drawn from the SAME budget — the six full-set
        weights are rescaled by (1 - qd_causal_stability_weight) so the set
        still sums to 1.0.
    reduced (descriptive/execution): non-Sharpe subset
        (robustness .30 + orthogonality .30 + novelty .20 + efficiency .20);
        these types are excluded from insertion by default.
Elite replacement: the new elite wins iff its score is STRICTLY greater —
ties keep the incumbent. Every insert bumps the cell's trial counter.

Metric sourcing honors AGENTS.md rule 2 (Spark never computes returns):
every component arrives from the caller/provider — oos_ic from the episode
metrics, deflated significance from the significance_gate's stored dsr/psr
(corrected_metrics), robustness/orthogonality/novelty/stability as metrics
subscores, efficiency = 1/(1+usd) from spend_ledger. Missing components are
treated as 0 and named in the row's component_defaults honesty field.

cell_emptiness (§7, archive mode): a check_novelty candidate carries a family
but no horizon/turnover, so the family slice is the finest archive
granularity it can address: emptiness = 1.0 when the candidate family's
slice has zero trials, else 1 − slice_trials / max_slice_trials over the
grid, bounded [0, 1]. Degraded no-archive mode keeps the Phase 2
approximation (novelty._cell_emptiness).
"""

from __future__ import annotations

import json
import math
from typing import Any

from . import budgets
from .config import Config
from .errors import ToolError
from .storage import Database
from .timeutil import now_iso

DESCRIPTOR_SPEC_VERSION = 1

# Component -> candidate metric keys, first non-None wins. deflated
# significance reads the significance_gate's stored fields (corrected_metrics
# carries psr/dsr as probabilities Phi(z); the higher the more significant).
_SIGNIFICANCE_KEYS = ("corrected_metrics.dsr", "dsr", "corrected_metrics.psr", "psr")


def _clip01(value: Any) -> float | None:
    """float in [0,1], or None when missing/non-numeric."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, v))


def _parse_cell_key(cell_key: str) -> tuple[int, int, int]:
    parts = str(cell_key).split(":")
    if len(parts) != 3:
        raise ValueError(f"malformed cell_key {cell_key!r}")
    return int(parts[0]), int(parts[1]), int(parts[2])


class Archive:
    """Per-process MAP-Elites grid rebuilt from qd_cells (M7)."""

    def __init__(self, db: Database, cfg: Config):
        self._db = db
        self._cfg = cfg
        research = cfg.research
        self._grid = [int(g) for g in research.qd_grid]
        self._dims = list(research.qd_descriptors)
        self._bins: list[list[str]] = [
            [str(b) for b in research.qd_family_bins],
            [str(b) for b in research.qd_horizon_bins],
            [str(b) for b in research.qd_turnover_bins],
        ]
        if len(self._grid) != 3 or len(self._dims) != 3 or len(self._bins) != 3:
            raise ValueError("qd_grid/qd_descriptors must name exactly 3 dimensions")
        if self._dims[0] != "family":
            raise ValueError(
                "qd_descriptors v1 must start with 'family' — coverage per_family,"
                " cell_emptiness slices, and the qd_cells.family column all key on it"
            )
        for dim, (size, bins) in enumerate(zip(self._grid, self._bins, strict=True)):
            if size != len(bins):
                raise ValueError(
                    f"qd dimension {self._dims[dim]!r}: grid size {size} !="
                    f" {len(bins)} configured bins"
                )
        self._cells: dict[tuple[int, int, int], dict] = {}
        rows = db.conn.execute(
            "SELECT * FROM qd_cells WHERE descriptor_spec_version = ?",
            (DESCRIPTOR_SPEC_VERSION,),
        ).fetchall()
        for r in rows:
            idx = _parse_cell_key(r["cell_key"])
            descriptors = json.loads(r["descriptors"])
            self._cells[idx] = {
                "cell_key": r["cell_key"],
                "descriptors": descriptors,
                "descriptor_spec_version": int(r["descriptor_spec_version"]),
                "elite_hypothesis_id": r["elite_hypothesis_id"],
                "dsr": r["dsr"],
                "oos_ic": r["oos_ic"],
                "family": r["family"],
                "trials": int(r["trials"]),
                "reward_type": r["reward_type"],
                "updated_at": r["updated_at"],
            }

    # -- descriptor -> cell mapping (§10.1) -----------------------------------

    def _bin_index(self, dim: int, value: Any) -> int:
        name = self._dims[dim]
        bins = self._bins[dim]
        if value is None:
            raise ToolError(
                f"QD insertion refused: missing {name!r} descriptor;"
                f" configured bins: {bins}"
            )
        if name == "turnover":
            numeric = _clip01(value) is not None and isinstance(value, (int, float))
            if numeric:
                if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0:
                    raise ToolError(
                        "QD insertion refused: numeric turnover must be a finite"
                        f" value >= 0, got {value!r}"
                    )
                thresholds = [
                    float(t) for t in self._cfg.research.qd_turnover_thresholds
                ]
                if len(thresholds) != len(bins) - 1:
                    raise ValueError(
                        "qd_turnover_thresholds must have len(qd_turnover_bins) - 1"
                        " entries"
                    )
                v = float(value)
                for i, bound in enumerate(thresholds):
                    if v <= bound:
                        return i
                return len(bins) - 1
        needle = str(value).strip().lower()
        for i, b in enumerate(bins):
            if b.lower() == needle:
                return i
        raise ToolError(
            f"QD insertion refused: unknown {name!r} category {value!r};"
            f" configured bins: {bins}"
        )

    def cell_index(self, descriptors: dict) -> tuple[int, int, int]:
        """Resolve descriptor names to integer cell indices; unknown
        categories are refused with a named ToolError naming the value."""
        return (
            self._bin_index(0, descriptors.get(self._dims[0])),
            self._bin_index(1, descriptors.get(self._dims[1])),
            self._bin_index(2, descriptors.get(self._dims[2])),
        )

    @staticmethod
    def cell_key(idx: tuple[int, int, int]) -> str:
        return f"{idx[0]}:{idx[1]}:{idx[2]}"

    def descriptor_names(self, idx: tuple[int, int, int]) -> dict[str, str]:
        return {name: self._bins[d][idx[d]] for d, name in enumerate(self._dims)}

    # -- elite score (§10.2) ---------------------------------------------------

    def reward_weights(
        self, reward_type: str, *, allow_descriptive: bool = False
    ) -> dict[str, float]:
        """Weight set for a hypothesis_type (via qd_reward_type_map).
        descriptive/execution use the reduced non-Sharpe set and are excluded
        from insertion unless allow_descriptive=True (named refusal)."""
        research = self._cfg.research
        set_name = research.qd_reward_type_map.get(reward_type)
        if set_name is None:
            raise ToolError(
                f"QD insertion refused: unknown reward_type {reward_type!r};"
                f" configured mapping: {research.qd_reward_type_map}"
            )
        if set_name == "reduced" and not allow_descriptive:
            raise ToolError(
                f"QD insertion refused: reward_type {reward_type!r} uses the reduced"
                " non-Sharpe set and is excluded from QD insertion by default"
                " (plan §10.2); pass allow_descriptive=True to record it anyway"
            )
        if set_name == "causal":
            stability = float(research.qd_causal_stability_weight)
            scale = 1.0 - stability
            weights = {k: v * scale for k, v in research.qd_reward_weights.items()}
            weights["stability"] = stability
            return weights
        if set_name == "reduced":
            return dict(research.qd_reduced_reward_weights)
        return dict(research.qd_reward_weights)

    def _lookup(self, metrics: dict, dotted: str) -> Any:
        src: Any = metrics
        for part in dotted.split("."):
            if not isinstance(src, dict):
                return None
            src = src.get(part)
        return src

    def score_components(
        self, metrics: dict | None, *, episode_id: str | None = None
    ) -> tuple[dict[str, float], list[str]]:
        """Elite-score components, each in [0,1]. Missing/non-numeric inputs
        default to 0.0 and are named in the returned component_defaults list
        (honesty field recorded on the cell row)."""
        metrics = metrics or {}
        components: dict[str, float] = {}
        defaults: list[str] = []

        def take(name: str, *chain: Any) -> None:
            value = None
            for candidate in chain:
                value = _clip01(candidate)
                if value is not None:
                    break
            if value is None:
                value = 0.0
                defaults.append(name)
            components[name] = value

        take("oos", metrics.get("oos_ic"))
        take("deflated_significance", *(self._lookup(metrics, k) for k in _SIGNIFICANCE_KEYS))
        take("robustness", metrics.get("robustness"))
        take("orthogonality", metrics.get("orthogonality"))
        take("novelty", metrics.get("novelty"))
        take("stability", metrics.get("stability"))
        # efficiency: 1/(1+usd) from the spend ledger for this episode
        # (§14: the efficiency term consumes the same ledger as the budgets).
        usd = budgets.run_spend(self._db, episode_id) if episode_id else 0.0
        components["efficiency"] = 1.0 / (1.0 + max(float(usd), 0.0))
        return components, defaults

    # -- insertion -------------------------------------------------------------

    def insert(
        self,
        hypothesis_id: str,
        descriptors: dict,
        metrics: dict | None,
        reward_type: str,
        *,
        allow_descriptive: bool = False,
        episode_id: str | None = None,
    ) -> dict:
        """Insert an elite candidate. Unknown descriptor categories and the
        descriptive/execution exclusion are named ToolErrors. Elite
        replacement iff the new score is strictly greater (ties keep the
        incumbent). Writes through to qd_cells in the caller's transaction."""
        idx = self.cell_index(descriptors)
        weights = self.reward_weights(reward_type, allow_descriptive=allow_descriptive)
        components, defaults = self.score_components(metrics, episode_id=episode_id)
        # honesty field names only the components the active weight set
        # actually consumes (stability defaults only matter for causal).
        defaults = [d for d in defaults if d in weights]
        score = float(sum(w * components.get(k, 0.0) for k, w in weights.items()))

        key = self.cell_key(idx)
        names = self.descriptor_names(idx)
        incumbent = self._cells.get(idx)
        replaced = incumbent is None or score > float(
            incumbent["descriptors"].get("elite_score") or 0.0
        )
        trials = (incumbent["trials"] if incumbent else 0) + 1
        now = now_iso()
        if replaced:
            row_descriptors = {
                **names,
                "components": {k: components[k] for k in weights},
                "component_defaults": defaults,
                "elite_score": score,
            }
            stored_dsr = components["deflated_significance"]
            stored_oos = components["oos"]
            elite_hypothesis_id = hypothesis_id
            reward_type_stored = reward_type
        else:
            # non-replacing insert: the incumbent elite's descriptors, score,
            # and reward_type stand; only trials/updated_at advance.
            row_descriptors = incumbent["descriptors"]  # type: ignore[index]
            stored_dsr = incumbent["dsr"]  # type: ignore[index]
            stored_oos = incumbent["oos_ic"]  # type: ignore[index]
            elite_hypothesis_id = incumbent["elite_hypothesis_id"]  # type: ignore[index]
            reward_type_stored = incumbent["reward_type"]  # type: ignore[index]
        self._db.execute(
            "INSERT INTO qd_cells ("
            " cell_key, descriptors, descriptor_spec_version, elite_hypothesis_id,"
            " dsr, oos_ic, family, trials, reward_type, updated_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(descriptor_spec_version, cell_key) DO UPDATE SET"
            " descriptors = excluded.descriptors,"
            " elite_hypothesis_id = excluded.elite_hypothesis_id,"
            " dsr = excluded.dsr, oos_ic = excluded.oos_ic,"
            " family = excluded.family, trials = excluded.trials,"
            " reward_type = excluded.reward_type, updated_at = excluded.updated_at",
            (
                key,
                json.dumps(row_descriptors, sort_keys=True),
                DESCRIPTOR_SPEC_VERSION,
                elite_hypothesis_id,
                stored_dsr,
                stored_oos,
                names.get("family"),
                trials,
                reward_type_stored,
                now,
            ),
        )
        self._cells[idx] = {
            "cell_key": key,
            "descriptors": row_descriptors,
            "descriptor_spec_version": DESCRIPTOR_SPEC_VERSION,
            "elite_hypothesis_id": elite_hypothesis_id,
            "dsr": stored_dsr,
            "oos_ic": stored_oos,
            "family": names.get("family"),
            "trials": trials,
            "reward_type": reward_type_stored,
            "updated_at": now,
        }
        return {
            "cell_key": key,
            "descriptors": names,
            "elite_score": score,
            "components": {k: components[k] for k in weights},
            "component_defaults": defaults,
            "reward_type": reward_type,
            "elite_replaced": replaced,
            "elite_hypothesis_id": elite_hypothesis_id,
            # Non-replacing inserts return the CHALLENGER's score above; this
            # field disambiguates by exposing the incumbent that held the cell.
            "incumbent_elite_score": (
                None if replaced else float(incumbent["descriptors"].get("elite_score") or 0.0)  # type: ignore[index]
            ),
            "trials": trials,
        }

    # -- reads -----------------------------------------------------------------

    def total_cells(self) -> int:
        total = 1
        for g in self._grid:
            total *= g
        return total

    def coverage(self) -> dict:
        """occupied/total cells + per-family occupied counts."""
        per_family = {b: 0 for b in self._bins[0]}
        for idx in self._cells:
            per_family[self._bins[0][idx[0]]] += 1
        return {
            "occupied_cells": len(self._cells),
            "total_cells": self.total_cells(),
            "per_family": per_family,
        }

    def elites(self, top_n: int) -> list[dict]:
        """Top-N cells by elite score (desc; cell_key asc on ties)."""
        rows = sorted(
            self._cells.values(),
            key=lambda r: (-float(r["descriptors"].get("elite_score") or 0.0), r["cell_key"]),
        )
        out = []
        for r in rows[: max(int(top_n), 0)]:
            out.append(
                {
                    "cell_key": r["cell_key"],
                    "descriptors": r["descriptors"],
                    "elite_hypothesis_id": r["elite_hypothesis_id"],
                    "elite_score": float(r["descriptors"].get("elite_score") or 0.0),
                    "dsr": r["dsr"],
                    "oos_ic": r["oos_ic"],
                    "family": r["family"],
                    "trials": r["trials"],
                    "reward_type": r["reward_type"],
                    "updated_at": r["updated_at"],
                }
            )
        return out

    def cell_emptiness(self, family: str) -> float:
        """§7 archive-mode cell emptiness for a candidate family:
        1.0 when the family's slice has zero trials, else
        1 − slice_trials / max_slice_trials over the grid, in [0,1].
        Unknown families are refused (named ToolError) — novelty falls back
        to the degraded approximation for those."""
        fam_idx = self._bin_index(0, family)
        slice_trials = [0] * self._grid[0]
        for idx, cell in self._cells.items():
            slice_trials[idx[0]] += cell["trials"]
        own = slice_trials[fam_idx]
        if own == 0:
            return 1.0
        top = max(slice_trials)
        if top <= 0:
            return 1.0
        return max(0.0, min(1.0, 1.0 - own / top))

    def emptiest_cell(self, family: str) -> dict:
        """The family's least-tried cell (empty first; ties -> lowest
        horizon/turnover index), for underexplored-region suggestions."""
        fam_idx = self._bin_index(0, family)
        best: tuple[int, int, int] | None = None
        for h in range(self._grid[1]):
            for t in range(self._grid[2]):
                idx = (fam_idx, h, t)
                if idx not in self._cells:
                    best = idx
                    break
                if best is None or self._cells[idx]["trials"] < self._cells[best]["trials"]:
                    best = idx
            if best is not None and best not in self._cells:
                break
        assert best is not None
        cell = self._cells.get(best)
        return {
            "cell_key": self.cell_key(best),
            "descriptors": self.descriptor_names(best),
            "trials": cell["trials"] if cell else 0,
        }


# ---------------------------------------------------------------------------
# archive_status (plan §6.3 tool 19)


def family_posteriors(db: Database) -> dict[str, dict]:
    """Stored Beta(a,b) per family from exploration_state (§5.1, §8.3)."""
    rows = db.conn.execute(
        "SELECT family, posterior, trials, successes FROM exploration_state"
        " ORDER BY family"
    ).fetchall()
    out = {}
    for r in rows:
        try:
            post = json.loads(r["posterior"])
            a, b = float(post["alpha"]), float(post["beta"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            a, b = 1.0, 1.0
        out[r["family"]] = {
            "alpha": a,
            "beta": b,
            "mean": a / (a + b) if (a + b) > 0 else 0.5,
            "trials": int(r["trials"]),
            "successes": int(r["successes"]),
        }
    return out


def family_M(db: Database) -> dict[str, int]:
    """M_family per §9.2 from the trials table (counted unit semantics:
    status IN ('completed','failed'), all families present in the ledger)."""
    rows = db.conn.execute(
        "SELECT family, COUNT(*) AS n FROM trials"
        " WHERE status IN ('completed','failed') GROUP BY family ORDER BY family"
    ).fetchall()
    return {r["family"]: int(r["n"]) for r in rows}


def predictor_error_map(db: Database, as_of: str) -> dict[str, float]:
    """Per-family mean |predicted_ic − realized_ic| over PIT episodes that
    carry both (predictor error tracking, plan §21): realized_ic is read from
    episode metrics as metrics.realized_ic -> metrics.oos_ic -> metrics.ic,
    first non-None wins."""
    rows = db.conn.execute(
        "SELECT e.research_family AS family, h.predicted_ic AS predicted,"
        " e.metrics AS metrics"
        " FROM episodes e JOIN hypotheses h ON h.hypothesis_id = e.hypothesis_id"
        " WHERE e.metrics IS NOT NULL AND e.as_of <= ?",
        (as_of,),
    ).fetchall()
    errors: dict[str, list[float]] = {}
    for r in rows:
        metrics = json.loads(r["metrics"])
        realized = None
        for key in ("realized_ic", "oos_ic", "ic"):
            value = _clip01(metrics.get(key))
            if value is not None:
                realized = float(metrics[key])
                break
        if realized is None:
            continue
        err = abs(float(r["predicted"]) - realized)
        errors.setdefault(r["family"], []).append(err)
    return {
        family: float(sum(vals) / len(vals))
        for family, vals in sorted(errors.items())
        if vals
    }


def status(db: Database, cfg: Config, as_of: str | None = None) -> dict:
    """archive_status tool payload (plan §6.3 tool 19): coverage, elites,
    family_M, spend (§14 spend_summary), family posteriors, and the
    predictor error map."""
    from .budgets import spend_summary

    arch = Archive(db, cfg)
    return {
        "as_of": as_of or now_iso(),
        "descriptor_spec_version": DESCRIPTOR_SPEC_VERSION,
        "coverage": arch.coverage(),
        "elites": arch.elites(cfg.research.qd_archive_status_top_n),
        "family_M": family_M(db),
        "spend": spend_summary(db),
        "family_posteriors": family_posteriors(db),
        "predictor_error_map": predictor_error_map(db, as_of or now_iso()),
    }
