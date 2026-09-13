"""Backtest seam providers (plan §15.2, C5).

`protocols` holds the validated request schemas; `base` the provider
protocol; `simulated`/`recorded`/`alpha_engine` the three implementations.
Spark validates and holds verdicts — providers do the returns math.
"""

from .base import BacktestProvider
from .protocols import (
    BacktestSpec,
    BookSpec,
    CausalSpec,
    CostModel,
    DateRange,
    LagRange,
    SplitSpec,
    spec_fingerprint,
)
from .recorded import RecordedProvider
from .simulated import SimulatedProvider

__all__ = [
    "BacktestProvider",
    "BacktestSpec",
    "BookSpec",
    "CausalSpec",
    "CostModel",
    "DateRange",
    "LagRange",
    "RecordedProvider",
    "SimulatedProvider",
    "SplitSpec",
    "spec_fingerprint",
]
