"""Backtest provider protocol (plan §15.2, C5).

Spark holds verdicts, never returns: providers do all backtest/causal/risk
math (AGENTS.md hard rule 2). A provider that does not implement an optional
method raises NotImplementedError; the gates service converts that to a
ToolError with a clear message.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .protocols import BacktestSpec, BookSpec, CausalSpec


@runtime_checkable
class BacktestProvider(Protocol):
    #: provider name recorded on trials rows (alpha_engine | simulated | recorded | ...)

    @property
    def name(self) -> str: ...

    def backtest(self, spec: BacktestSpec) -> dict: ...

    def causal(self, spec: CausalSpec) -> dict:
        """Type-B causal probe (§9.3). Raise NotImplementedError if unsupported."""
        ...

    def risk(self, spec: BookSpec) -> dict:
        """Book risk decomposition. Raise NotImplementedError if unsupported."""
        ...
