"""Provider request schemas (plan §15.2, C5): BacktestSpec / CausalSpec /
BookSpec as pydantic models with the plan's fields and validators.

Spark validates and holds verdicts; the provider never sees unvalidated
input. Cost discipline: no zero-cost default — bps + slippage + borrow must
sum positive.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

CAUSAL_METHODS = (
    "granger",
    "pcmci-parcorr",
    "pcmci-gpdc",
    "pcmci-cmiknn",
)


def _check_iso_date(value: str, field: str) -> str:
    """Bar-granular ISO dates (YYYY-MM-DD) for range fields (§5.0 N4)."""
    import re

    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError(f"{field} must be a YYYY-MM-DD date, got {value!r}")
    return value


class DateRange(BaseModel):
    """In-sample and out-of-sample windows (§15.2). OOS untouched by fitting."""

    is_start: str
    is_end: str
    oos_start: str
    oos_end: str

    @field_validator("*")
    @classmethod
    def _dates(cls, v: str) -> str:
        return _check_iso_date(v, "date")

    @model_validator(mode="after")
    def _ordered(self) -> DateRange:
        if not (self.is_start <= self.is_end < self.oos_start <= self.oos_end):
            raise ValueError(
                "dates must satisfy is_start <= is_end < oos_start <= oos_end"
            )
        return self


class CostModel(BaseModel):
    """Trading-cost model (§15.2): bps + slippage + borrow; total must be > 0 —
    the plan forbids a zero-cost default (§9.1 splits/costs contract)."""

    bps: float = 0.0
    slippage: float = 0.0
    borrow: float = 0.0

    @model_validator(mode="after")
    def _nonzero(self) -> CostModel:
        if self.bps + self.slippage + self.borrow <= 0.0:
            raise ValueError(
                "costs must be > 0 in total (no zero-cost default; plan §9.1/§15.2)"
            )
        return self


class SplitSpec(BaseModel):
    """Purged/embargoed split discipline (§9.1). Lengths in days/bars, >= 0."""

    purge: int = Field(default=0, ge=0)
    embargo: int = Field(default=0, ge=0)


class BacktestSpec(BaseModel):
    """What to test, how, and under which discipline (§15.2 verbatim fields)."""

    hypothesis_id: str
    signal_ref: str  # artifacts/ URI or alpha_engine signal id
    params: dict = Field(default_factory=dict)
    universe_pit: str  # PIT universe id + as_of ("SYM1,SYM2@YYYY-MM-DD" for SimulatedProvider)
    dates: DateRange
    costs: CostModel
    splits: SplitSpec
    seed: int


class LagRange(BaseModel):
    min_lag: int = Field(ge=0)
    max_lag: int = Field(ge=1)

    @model_validator(mode="after")
    def _ordered(self) -> LagRange:
        if self.min_lag > self.max_lag:
            raise ValueError("min_lag must be <= max_lag")
        return self


class CausalSpec(BaseModel):
    """Type-B causal probe (§9.3, §15.2). DAG lives on the hypothesis row
    (C6: no causal promotion without DAG); this spec names targets/conditioning."""

    episode_id: str
    targets: list[str] = Field(min_length=1)
    conditioning: list[str] = Field(default_factory=list)
    method: Literal[*CAUSAL_METHODS] = "granger"
    lags: LagRange


class BookSpec(BaseModel):
    """Book for risk decomposition (§15.2): exactly one of the three sources."""

    book_id: str | None = None
    holdings_uri: str | None = None  # exactly one of the three
    returns_uri: str | None = None
    factor_model: str = "ff5+momentum"

    @model_validator(mode="after")
    def _exactly_one(self) -> BookSpec:
        provided = [
            name
            for name, value in (
                ("book_id", self.book_id),
                ("holdings_uri", self.holdings_uri),
                ("returns_uri", self.returns_uri),
            )
            if value
        ]
        if len(provided) != 1:
            raise ValueError(
                f"exactly one of book_id / holdings_uri / returns_uri is required; got {provided}"
            )
        return self


def spec_fingerprint(spec: BaseModel) -> str:
    """sha1 over the canonical JSON of a validated spec — the RecordedProvider
    cassette key (§5.0 hashing conventions; sha1 per the trials idem_key rule).
    """
    import hashlib
    import json

    canonical = json.dumps(
        spec.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha1(canonical).hexdigest()  # noqa: S324 (plan-mandated sha1)
