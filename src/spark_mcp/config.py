"""Configuration loading: YAML file + environment overrides (plan §16)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = Path.home() / ".spark" / "config.yaml"


class StorageConfig(BaseModel):
    db_path: str = "~/.spark/spark.db"
    artifacts_dir: str = "~/.spark/artifacts"


class LLMConfig(BaseModel):
    provider: str = "openai-compatible"
    model: str = "your-model-id"
    embedder_model: str = "your-embedder-id"
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    degraded_mode: str = "auto"  # auto | require-key | never-llm


class PeerConfig(BaseModel):
    enabled: bool = False
    transport: str = "stdio"  # stdio | http
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    url: str | None = None


class PeersConfig(BaseModel):
    quantflow: PeerConfig = Field(default_factory=PeerConfig)
    alpha_engine: PeerConfig = Field(default_factory=PeerConfig)
    documents: PeerConfig = Field(default_factory=PeerConfig)


class ResearchConfig(BaseModel):
    """All experiment-affecting weights/thresholds live here (plan §16).

    Defaults mirror the v3 master plan; every value is overridable via YAML.
    """

    novelty_weights: list[float] = Field(default=[0.5, 0.3, 0.2])
    sigma_floor: float = 0.01
    s_cap: float = 5.0
    # cell_emptiness v1 normalizer: QD grid lands in Phase 5, so emptiness is
    # approximated as 1 − distinct_families_tried / family_count_norm (§7 note).
    family_count_norm: int = 20
    # Phase 2 uniform exploration bonus; Phase 5 uses the family Thompson
    # posterior mean for stored rows (§8.3) and keeps this as the fallback for
    # families with no stored posterior (empty-state behavior stays 0.5).
    exploration_bonus_uniform: float = 0.5
    # Family-aware relevance (degraded/deterministic mode): candidates from a
    # family the problem doesn't point at keep this fraction of their lexical
    # relevance (cross-family composition stays possible, just penalized).
    family_weight_floor: float = 0.05
    # Minimum token length for substring (derivational) matches when matching
    # problem tokens against the family vocabulary (noise ~ denoising).
    family_morph_min_len: int = 4
    # Discounts inside the family-vocabulary match: part-of-compound tags
    # ('returns' inside 'normal_returns'), affix-bounded derivations
    # (robust ~ robustness), and middle words of compound concepts
    # ('estimate' inside 'Edge estimate accuracy').
    family_part_discount: float = 0.3
    family_affix_discount: float = 0.5
    family_middle_discount: float = 0.5
    # Family-weight size normalization: raw family matches are deflated by
    # vocab_size^exponent so large families (51 quant cards) cannot win by
    # bulk alone against focused families (10 signal cards).
    family_size_norm_exponent: float = 0.5
    # Selection: candidates within this score band of the best candidate form
    # the "defensible set" the softmax draws from (§8.2); wider = more
    # exploration (lower repeat-overlap), narrower = tighter tracking of the
    # score ranking. The set is never smaller than selection_min_eligible so
    # the draw stays stochastic (G1-grade repeat-overlap), and the pool is
    # k + selection_pool_extra so MMR picks k from a meaningfully random draw.
    # selection_obvious_drop > 0 excludes the top-N scorers from the draw
    # ("challenger mode"): with the recall floor in place its original
    # repeat-overlap rationale is obsolete (simulated: inclusion moves mean
    # overlap 0.20 -> 0.198), and hiding the single best-scoring candidate
    # removes the pick most likely to be a plausible fit — so the default is
    # 0 (champion eligible). Round-3 blind-judge rationales independently
    # pointed at the excluded champion as the correct answer.
    selection_score_band: float = 0.12
    selection_min_eligible: int = 24
    selection_pool_extra: int = 0
    selection_obvious_drop: int = 0
    # Recall floor for explore_methods (0 disables): when hybrid recall
    # returns fewer candidates than this, pad with the best remaining
    # techniques by deterministic relevance so the §8.2 pool draw has real
    # choice — a 6-of-6 pool would repeat identically on every call (G1).
    selection_recall_pad_to: int = 30
    # With an LLM scorer, rank the ENTIRE visible technique field instead of
    # a lexically-recalled window (degraded mode keeps the recall+pad path).
    # Lexical recall demonstrably buries on-target cards (Hamilton regime
    # switching, CUSUM, DWT never entered a 30-candidate set for problems
    # squarely in their domain). score_cache makes the per-problem cost
    # one-time; scoring_workers threads the per-candidate calls.
    scoring_full_field: bool = True
    scoring_workers: int = 8
    # Softened family factor exponent for the structural_fit channel: edges
    # are the legitimate cross-family composition route, so fit is penalized
    # by factor^exponent rather than the full factor.
    family_fit_factor_exponent: float = 0.5
    # Pool draws use temperature x this multiplier (higher = flatter draws =
    # more repeat-query exploration); the reported selection_probability is
    # always the unflattened softmax. Calibrated with the LLM scorer live
    # (G1 simulation over the 5 canonical problems, 12-seed Monte Carlo):
    # sharper LLM scores concentrate the draw, so the balanced-mode default
    # sits at 6.0 (mean repeat-overlap ~0.20 vs SLO 0.30); the deterministic
    # path is insensitive to this knob (flat scores already explore).
    selection_pool_temp_multiplier: float = 6.0
    scoring_weights: list[float] = Field(default=[0.35, 0.20, 0.20, 0.15, 0.10])
    temperatures: dict[str, float] = Field(
        default={"safe": 0.3, "balanced": 0.7, "exploratory": 1.2}
    )
    mmr_lambda: float = 0.5
    top_k: int = 50
    rrf_k: int = 60
    bandit_prior: list[float] = Field(default=[1.0, 1.0])
    informed_prior_after_episodes: int = 20
    contradiction_overlap: float = 0.6
    budget: dict[str, float] = Field(
        default={"refine": 0.60, "new_cell": 0.25, "long_shot": 0.15}
    )
    budget_override_band: float = 0.15
    budget_window_days: int = 7
    analogy_cap: float = 0.20
    ucb_c: float = 1.4
    tombstone_block: float = 0.85
    tombstone_review_low: float = 0.70
    belief_auto_promote_threshold: float = 0.8
    belief_halflife_days: float = 180.0
    belief_validation_slo_days: int = 90
    contradiction_cos: float = 0.75
    qd_grid: list[int] = Field(default=[8, 6, 4])
    qd_descriptors: list[str] = Field(default=["family", "horizon", "turnover"])
    # QD descriptor bins (Phase 5, plan §10.1): categorical bin lists, one per
    # qd_descriptors dimension; len(qd_*_bins) must equal the matching qd_grid
    # entry. Family bins 1-5 mirror the Phase-2 seed corpus families
    # (seeds/techniques.yaml: quant/econometrics/causal/signal_processing/
    # network_optimization); nlp reserves the §15.3 documents desk; macro and
    # portfolio reserve adjacent desks named on the plan's tool surface.
    # A descriptor value outside its bin list is refused at insertion (named
    # ToolError) — there is deliberately no catch-all bin.
    qd_family_bins: list[str] = Field(
        default=[
            "quant", "econometrics", "causal", "signal_processing",
            "network_optimization", "nlp", "macro", "portfolio",
        ]
    )
    # Horizon bins are trading-day categories matching the corpus horizon
    # vocabulary (hypotheses declare horizons like '20d'; the long_horizon
    # corpus tag maps to 'long'). Case-insensitive exact match, else refused.
    qd_horizon_bins: list[str] = Field(
        default=["1d", "5d", "20d", "60d", "120d", "long"]
    )
    # Turnover bins are daily-turnover categories. A numeric params.turnover
    # (BacktestSpec) maps through qd_turnover_thresholds = upper bounds for
    # low/medium/high (above the last bound -> very_high); a string value must
    # match a bin name directly.
    qd_turnover_bins: list[str] = Field(
        default=["low", "medium", "high", "very_high"]
    )
    qd_turnover_thresholds: list[float] = Field(default=[0.25, 0.75, 1.5])
    # Elite score weights (plan §10.2, M12), full predictive set.
    qd_reward_weights: dict[str, float] = Field(
        default={
            "oos": 0.30, "deflated_significance": 0.20, "robustness": 0.15,
            "orthogonality": 0.15, "novelty": 0.10, "efficiency": 0.10,
        }
    )
    # Causal adds stability, drawn FROM the same budget: the six full-set
    # weights are rescaled by (1 - qd_causal_stability_weight) so the causal
    # set still sums to 1.0 (default -> stability 0.10, others x0.9).
    qd_causal_stability_weight: float = 0.10
    # Descriptive/execution reduced non-Sharpe set (no oos_ic / deflated
    # significance — a descriptive study is not scored like an alpha factor,
    # §10.2); also used for the allow_descriptive=True opt-in insertion.
    qd_reduced_reward_weights: dict[str, float] = Field(
        default={
            "robustness": 0.30, "orthogonality": 0.30,
            "novelty": 0.20, "efficiency": 0.20,
        }
    )
    # hypothesis_type -> reward-set name, single mapping dict (§10.2).
    qd_reward_type_map: dict[str, str] = Field(
        default={
            "predictive": "full", "portfolio": "full", "risk": "full",
            "causal": "causal", "descriptive": "reduced", "execution": "reduced",
        }
    )
    # archive_status elites list length.
    qd_archive_status_top_n: int = 10
    dsr_alpha: float = 0.05
    kill_rule_failures: int = 5
    min_T: int = 30
    # Tier-2 run-summary difficulty heuristics (plan §15.1, v1 §6 defaults):
    # a QuantFlow hook summary triggers a retrospective when it exceeds either
    # threshold or overcame at least one failure (resolution non-empty).
    hook_tool_calls_threshold: int = 50
    hook_tokens_threshold: int = 200_000


class BudgetsConfig(BaseModel):
    daily_usd: float = 25.0
    per_episode_usd: float = 5.0
    per_consolidation_usd: float = 3.0
    policy: str = "warn"  # warn | fail_closed
    rates: dict[str, dict[str, float]] = Field(default_factory=dict)


class Config(BaseModel):
    storage: StorageConfig = Field(default_factory=StorageConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    peers: PeersConfig = Field(default_factory=PeersConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    budgets: BudgetsConfig = Field(default_factory=BudgetsConfig)

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> Config:
        p = Path(path) if path else DEFAULT_CONFIG_PATH
        if not p.exists():
            return cls()
        data: dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)

    def db_path(self) -> Path:
        return Path(os.path.expanduser(self.storage.db_path))

    def artifacts_dir(self) -> Path:
        return Path(os.path.expanduser(self.storage.artifacts_dir))

    def llm_api_key(self) -> str | None:
        return os.environ.get(self.llm.api_key_env) or None

    def active_mode(self) -> str:
        """Degraded-mode report for doctor (plan §4.3, M4)."""
        has_key = bool(self.llm_api_key())
        if self.llm.degraded_mode == "never-llm":
            return "deterministic (never-llm configured)"
        if has_key:
            return "full (llm key present)"
        if self.llm.degraded_mode == "require-key":
            return "unusable (require-key but no key)"
        return "degraded (no llm key: deterministic scoring, fts5-only retrieval)"
