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
    scoring_weights: list[float] = Field(default=[0.35, 0.20, 0.20, 0.15, 0.10])
    temperatures: dict[str, float] = Field(
        default={"safe": 0.3, "balanced": 0.7, "exploratory": 1.2}
    )
    mmr_lambda: float = 0.5
    top_k: int = 50
    rrf_k: int = 60
    bandit_prior: list[float] = Field(default=[1.0, 1.0])
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
    dsr_alpha: float = 0.05
    kill_rule_failures: int = 5
    min_T: int = 30


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
