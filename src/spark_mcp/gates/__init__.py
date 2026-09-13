"""Statistical gates + trial governor (plan §9, §12.2).

`dsr` is the pure-statistics module (PSR / DSR / MinBTL, Bailey & López de
Prado 2014, no subtraction step). `service` is the trial governor: backtest
dispatch under §9.2 accounting, the mandatory §12.2 tombstone pre-check, the
Type-A significance gate, Type-B causal/risk probes, and the §8.3 kill rule.
"""

from .dsr import (
    DEFAULT_MIN_T,
    EULER_GAMMA,
    GateRefusal,
    dsr,
    expected_max_sharpe,
    min_backtest_length,
    psr,
)
from .service import (
    M_family,
    backtest_run,
    causal_check,
    record_family_outcome,
    risk_decompose,
    significance_gate,
    spec_hash,
    tombstone_candidates,
    tombstone_precheck,
)

__all__ = [
    "DEFAULT_MIN_T",
    "EULER_GAMMA",
    "GateRefusal",
    "M_family",
    "backtest_run",
    "causal_check",
    "dsr",
    "expected_max_sharpe",
    "min_backtest_length",
    "psr",
    "record_family_outcome",
    "risk_decompose",
    "significance_gate",
    "spec_hash",
    "tombstone_candidates",
    "tombstone_precheck",
]
