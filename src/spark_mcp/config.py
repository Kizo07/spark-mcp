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


class GatewayConfig(BaseModel):
    """QuantFlow Gateway skill-write peer (plan §11 C7). Default absent: the
    promotion write-path blocks with GatewayWriteBlocked and the champion
    stays staged in Spark's registry, which remains the system of record."""

    enabled: bool = False
    url: str | None = None
    token_env: str = "QUANTFLOW_GATEWAY_TOKEN"


class PeersConfig(BaseModel):
    quantflow: PeerConfig = Field(default_factory=PeerConfig)
    alpha_engine: PeerConfig = Field(default_factory=PeerConfig)
    documents: PeerConfig = Field(default_factory=PeerConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)


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
    # Phase 7 SME-lite analogy engine (plan §6.1 tool 4, §8.3): analogy
    # searches are capped at this FRACTION of total family exploration trials
    # (tracked by exploration_state.analogy_trials); earning more requires
    # beating the base family hit rate (§8.3 wording, refusal message).
    analogy_budget_cap: float = 0.20
    # Gentner-style systematicity floor (formula in analogy.py docstring):
    # mappings scoring below are refused (named ToolError when none pass).
    analogy_systematicity_threshold: float = 0.5
    # suggest_next gap-report consumption (plan §13/G5): families named in an
    # open consolidation gap report get this multiplicative boost on the
    # Thompson x cell-emptiness utility.
    gap_boost_factor: float = 1.5
    # Consolidate (§13): pending skill candidates whose challenger row is older
    # than this many days are listed on the gap report.
    skill_candidate_stale_days: int = 14
    # Consolidate archive-gap analysis: a family whose mean |predicted -
    # realized| predictor error exceeds this is an outlier gap (plan §21).
    predictor_error_outlier: float = 0.05
    # Phase-7 decision record (plan §6.3 tool 21 N5 + §18.1): the report_write
    # HTML renderer is DEFERRED — md-only output until a renderer is named by
    # a later phase. The flag pins the decision in config; report_write
    # rejects format != 'md' citing this record (STATUS.md, Phase 7).
    report_html_renderer: str = "deferred"
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
    # vocabulary (hypotheses declare horizons like '20d'). Case-insensitive
    # EXACT match only — no aliases (a 'long_horizon' corpus tag is refused,
    # not mapped; callers normalize before insertion).
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
    # Phase 6 GEPA caps (plan §11, §21 cost-blowup row): population and
    # generation caps bound the native evolution loop; pareto_top_k is the
    # parent pool the population refreshes from each generation.
    gepa_population_size: int = 6
    gepa_generations: int = 2
    gepa_pareto_top_k: int = 3
    # Replay gaming alarm (plan §11/§21): every replay verdict reports
    # heldout_gap; a gap above this flags the fixture set for refresh review
    # (challengers overfitting held-in tasks must surface).
    replay_heldout_gap_alarm: float = 0.25
    # Auto-promotion floor (plan §11 'the gate decides'): a FIRST promotion
    # (no incumbent champion to beat) requires the challenger's held-in mean
    # to exceed this floor by more than replay_promotion_margin; later
    # promotions require beating the incumbent's last held-in mean by more
    # than the margin, and heldout_gap must be under the alarm everywhere.
    # Ties never promote (strict inequality).
    replay_promotion_floor: float = 0.5
    # Skill auto-gate margin (evolution upgrade plan §2.F3): the challenger's
    # held-in mean must exceed the incumbent mean (or the first-promotion
    # floor) by MORE than this. DSR margin is the binding constraint at the
    # artifact layer; this is its skill-text analogue.
    replay_promotion_margin: float = 0.05
    # Fixture-growth alarm threshold (plan §2.F4): consolidate reports an
    # open fixture-growth gap while len(fixtures) is below this. Alarm only,
    # never blocks promotion.
    replay_min_fixtures: int = 6
    # Artifact-evolution loop (evolution upgrade plan §2.G): all knobs for
    # the execution-grounded fitness + computed-promotion layer.
    evolution_max_rounds: int = 8
    evolution_max_consecutive_rejections: int = 3
    evolution_dsr_margin: float = 0.02
    evolution_sr_margin: float = 0.0
    evolution_first_promotion_floor: float = 0.95  # DSR prob floor, first promotion
    evolution_sota_jump_window: int = 6
    evolution_sota_jump_threshold: float = 0.2  # hit-rate below → new root
    evolution_render_max_tokens: int = 4000
    evolution_require_live_provider: bool = False  # True in prod: refuse simulated fitness
    # Level-2 variation base template (component 2): vary_spec perturbs
    # params/seed over this fixed base. Universe/dates/signal/splits stay on
    # the base (window moves need a data-availability oracle to stay
    # PIT-safe, so fixed configured windows are the safe default). Costs here
    # apply when no parent episode costs exist; costs are never perturbed.
    evolution_base_spec: dict[str, Any] = Field(
        default={
            "signal_ref": "artifacts://signals/evo-base.json",
            "params": {"type": "momentum", "lookback": 20, "holding": 1, "turnover": 1.0},
            "universe_pit": "TOYA,TOYB,TOYC,TOYD,TOYE,TOYF,TOYG,TOYH@2019-01-01",
            "dates": {
                "is_start": "2019-01-01",
                "is_end": "2019-12-31",
                "oos_start": "2020-01-01",
                "oos_end": "2020-06-30",
            },
            "costs": {"bps": 5.0, "slippage": 2.0, "borrow": 0.0},
            "splits": {"purge": 5, "embargo": 5},
        }
    )
    # Level-2 perturbation grids: lookback/type are resampled from these via
    # the run's seeded rng. The default type grid is the provider-supported
    # real-signal subset ("noise" is a null control, not an evolution target).
    evolution_lookback_grid: list[int] = Field(default=[5, 10, 20, 40, 60, 120])
    evolution_type_grid: list[str] = Field(default=["momentum", "reversal"])


class BudgetsConfig(BaseModel):
    daily_usd: float = 25.0
    per_episode_usd: float = 5.0
    per_consolidation_usd: float = 3.0
    policy: str = "warn"  # warn | fail_closed
    rates: dict[str, dict[str, float]] = Field(default_factory=dict)
    # §14 estimate for one llm complete call when the provider reports no
    # usage: consolidate's per-run cap counts complete calls x this rate.
    llm_call_estimate_usd: float = 0.02


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
