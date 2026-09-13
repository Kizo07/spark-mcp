"""Type-A statistical gates (plan §9.1, C1 corrected): PSR, DSR, MinBTL.

Bailey & López de Prado (2014), "The Deflated Sharpe Ratio: Correcting for
Selection Bias, Backtest Overfitting and Non-Normality":

    PSR(SR_hat; SR0) = Phi( (SR_hat - SR0) * sqrt(T-1)
                            / sqrt(1 - skew*SR_hat + (kurt-1)/4 * SR_hat^2) )

    DSR = PSR( SR_hat ; SR0 = E[max SR | H0, M, skew, kurt, T] )

There is NO subtraction step anywhere: the null threshold SR0 replaces the
fixed 0 in the PSR formula.

E[max SR | H0] uses the False Strategy Theorem (Thm 1) Euler–Mascheroni
approximation to the expected maximum of M independent standard-normal trials,

    E[max_M] ~= (1-gamma) * Phi^-1(1 - 1/M) + gamma * Phi^-1(1 - 1/(M*e)),

scaled by sqrt(V), the non-normality variance correction of the SR estimator
evaluated at the null per-trial SR scale sr0_e = sqrt(1/(T-1)) (under H0 the
standard error of an SR estimate with T observations is 1/sqrt(T-1), so this
is the SR magnitude a pure-noise trial is expected to show):

    V = 1 - skew*sr0_e + (kurt-1)/4 * sr0_e^2
    E[max SR|H0] = sqrt(V) * E[max_M]

Formula choice note: the paper evaluates the variance correction at the
estimated max SR itself in its worked recursion; the closed-form lecture
variant used here (and requested by the plan §9 build spec) evaluates V at
the fixed null scale sr0_e. At that scale the correction term is O(1/T), so
the two forms agree to within O(1/sqrt(T)^2) for T >> 1; the form here is
deterministic (no fixed-point iteration) and documented.

MinBTL (minimum backtest length, same paper family): the smallest T at which
a strategy with the target SR clears DSR p < alpha. Solved by monotone search
over our own dsr() (no second formula to drift out of agreement); the §9.1
hard floor min_T is part of the answer: a T below the floor can never be
promoted, so MinBTL reports at least min_T.

Pure stdlib (math + statistics.NormalDist); scipy is intentionally not used.
"""

from __future__ import annotations

import math
from statistics import NormalDist

#: Hard floor for T (plan §9.1 "T < 30 -> gate refuses needs-more-data";
#: config value cfg.research.min_T is passed by the service layer).
DEFAULT_MIN_T = 30

#: Euler–Mascheroni constant.
EULER_GAMMA = 0.577215664901532860606512090082402431

_STD_NORMAL = NormalDist(0.0, 1.0)


class GateRefusal(Exception):
    """A gate refuses to evaluate (§9.1 needs-more-data, degenerate inputs).

    Raised as a value-level signal so the service layer can convert it into a
    ToolError with the protocol wording intact.
    """


def _phi(x: float) -> float:
    return _STD_NORMAL.cdf(x)


def _phi_inv(p: float) -> float:
    return _STD_NORMAL.inv_cdf(p)


def psr(sr_hat: float, sr0: float, T: int, skew: float, kurt: float) -> tuple[float, bool]:
    """Probabilistic Sharpe Ratio (Def 1). Returns (value, ok); ok=False when
    the variance term 1 - skew*SR_hat + (kurt-1)/4*SR_hat^2 is non-positive
    (the normal approximation has broken down) — value is clamped to 0.0."""
    if T < 2:
        raise GateRefusal(f"needs-more-data: T={T} < 2 (no variance estimate exists)")
    var_term = 1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat * sr_hat
    if var_term <= 0.0:
        return 0.0, False
    z = (sr_hat - sr0) * math.sqrt(T - 1.0) / math.sqrt(var_term)
    return _phi(z), True


def expected_max_sharpe(M: int, skew: float, kurt: float, T: int) -> float:
    """E[max SR | H0, M trials, skew/kurt correction, T observations].

    M=1 -> 0.0 exactly (one trial has no selection bias; DSR reduces to
    PSR(SR_hat; 0), §9.1). T < 2 -> GateRefusal (no null scale exists).
    """
    if M <= 1:
        return 0.0
    if T < 2:
        raise GateRefusal(f"needs-more-data: T={T} < 2 (no null SR scale exists)")
    sr0_e = 1.0 / math.sqrt(T - 1.0)  # null per-trial SR scale (see module docstring)
    V = 1.0 - skew * sr0_e + (kurt - 1.0) / 4.0 * sr0_e * sr0_e
    if V <= 0.0:
        # Unreachable for |skew| <= 3 at any realistic T; guard so callers get
        # a defined value instead of a math domain error. No deflation is the
        # conservative-leaning choice (it never promotes).
        return 0.0
    e_max = (1.0 - EULER_GAMMA) * _phi_inv(1.0 - 1.0 / M) + EULER_GAMMA * _phi_inv(
        1.0 - 1.0 / (M * math.e)
    )
    return math.sqrt(V) * e_max


def dsr(
    sr_hat: float,
    T: int,
    skew: float,
    kurt: float,
    M: int,
    *,
    min_t: int = DEFAULT_MIN_T,
) -> tuple[float, bool]:
    """Deflated Sharpe Ratio = PSR(SR_hat; SR0 = E[max SR|H0]). NO subtraction.

    Refuses (GateRefusal) when T < min_t (§9.1 needs-more-data). M=1 reduces
    to PSR(SR_hat; 0). SR_hat <= 0 always yields p <= 0.5 (automatic fail at
    any sane alpha); the service layer makes that explicit.
    """
    if T < min_t:
        raise GateRefusal(f"needs-more-data: T={T} < min_T={min_t}")
    sr0 = expected_max_sharpe(M, skew, kurt, T)
    return psr(sr_hat, sr0, T, skew, kurt)


def min_backtest_length(
    target_sr: float,
    M: int,
    skew: float,
    kurt: float,
    *,
    alpha: float = 0.05,
    min_t: int = DEFAULT_MIN_T,
    max_t: int = 10_000_000,
) -> int | float:
    """MinBTL: smallest T (>= min_t) with DSR(target_sr; T) p < alpha.

    Solved by binary search over T (monotone in T: sqrt(T-1) grows while
    the null scale sr0_e = 1/sqrt(T-1) shrinks toward its asymptote). The
    `clears` predicate inlines dsr()'s z-inequality — dsr p < alpha is
    algebraically z > z_alpha since Phi is monotone — instead of calling
    dsr() per probe; the golden pins in test_dsr.py pin the result either
    way. Returns math.inf when no finite T clears alpha (target at or below
    the asymptotic null max — no amount of data makes it significant).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    z_alpha = _phi_inv(1.0 - alpha)

    def clears(t: int) -> bool:
        var_term = 1.0 - skew * target_sr + (kurt - 1.0) / 4.0 * target_sr * target_sr
        if var_term <= 0.0:
            return False
        sr0 = expected_max_sharpe(M, skew, kurt, t) if t >= 2 else 0.0
        return (target_sr - sr0) * math.sqrt(t - 1.0) > z_alpha * math.sqrt(var_term)

    if not clears(max_t):
        return math.inf
    lo, hi = min_t, max_t
    while lo < hi:
        mid = (lo + hi) // 2
        if clears(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo
