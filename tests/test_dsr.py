"""Golden + property tests for the Type-A gates (plan §9.1).

Golden-value derivation (independent of the implementation under test):
`spark_mcp.gates.dsr` computes Phi / Phi^-1 with `statistics.NormalDist`.
The expected values below were derived by RE-implementing the formulas from
the Bailey & López de Prado (2014) definitions with a different machinery —

    Phi(x)     = 0.5 * (1 + math.erf(x / sqrt(2)))          (no NormalDist)
    Phi^-1(p)  = bisection on the erf form, 200 iterations  (no NormalDist)
    E[max_M]   = (1-gamma)*Phi^-1(1-1/M) + gamma*Phi^-1(1-1/(M*e)),
                 gamma = 0.57721566490153286060... (Euler–Mascheroni)

— and pinned here at 1e-6 tolerance. No constant was copied from the paper
or any lecture note; each is reproducible from the derivation comments.

Worked derivations (skew = gamma_3, kurt = Pearson gamma_4, normal -> 0 / 3):

  PSR(1; 0, T=1000, 0, 3):
      z = (1-0)*sqrt(999)/sqrt(1 + (3-1)/4*1^2) = sqrt(999/1.5)
        = 25.80697580112788...  ->  Phi(z) = 1 - ~1e-147 -> 1.0 in float64.

  PSR(0.5; 0.2, T=120, skew=-0.5, kurt=4):
      var = 1 - (-0.5)(0.5) + (3/4)(0.5^2) = 1.4375
      z   = (0.3)*sqrt(119)/sqrt(1.4375) = 2.7295484576406945...

  E[max SR|H0] (M=10, 0, 3, T=1000):  sr0_e = 1/sqrt(999)
      V   = 1 + 0.5*sr0_e^2 = 1.0005005005005005
      E   = sqrt(V)*[(1-g)*Phi^-1(0.9) + g*Phi^-1(1-1/(10e))]
          = 1.5749922956722824...   (cf. exact E[max of 10 N(0,1)] = 1.53875;
                                     the Euler-Mascheroni form is an approx)

  DSR(1.5; T=1000, 0, 3, M=10):  SR0 = 1.5749922... > SR_hat
      ->  z = (1.5-1.57499...)*sqrt(999)/sqrt(1+0.5*2.25) < 0
      ->  Phi(z) = 0.05197507293757708...  (SR 1.5 FAILS at alpha 0.05 after
                                             only 10 trials — the point of DSR)
"""

from __future__ import annotations

import math

import pytest

from spark_mcp.gates.dsr import (
    EULER_GAMMA,
    GateRefusal,
    dsr,
    expected_max_sharpe,
    min_backtest_length,
    psr,
)

SQRT2 = math.sqrt(2.0)


def _phi_erf(x: float) -> float:
    """Independent Phi via math.erf (no statistics.NormalDist)."""
    return 0.5 * (1.0 + math.erf(x / SQRT2))


def _phi_inv_bisect(p: float) -> float:
    lo, hi = -12.0, 12.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _phi_erf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _psr_ind(sr_hat, sr0, T, skew, kurt):
    var = 1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat * sr_hat
    z = (sr_hat - sr0) * math.sqrt(T - 1.0) / math.sqrt(var)
    return _phi_erf(z)


def _emax_ind(M, skew, kurt, T):
    sr0_e = 1.0 / math.sqrt(T - 1.0)
    V = 1.0 - skew * sr0_e + (kurt - 1.0) / 4.0 * sr0_e * sr0_e
    e = (1 - EULER_GAMMA) * _phi_inv_bisect(1 - 1.0 / M) + EULER_GAMMA * _phi_inv_bisect(
        1 - 1.0 / (M * math.e)
    )
    return math.sqrt(V) * e


TOL = 1e-6


def test_normaldist_matches_independent_erf_machinery():
    """Sanity: the module's statistics.NormalDist and this file's erf/bisection
    machinery agree to <1e-9, so the pins discriminate the formula, not the
    CDF library."""
    from statistics import NormalDist

    nd = NormalDist()
    for x in (-3.5, -1.0, -0.25, 0.0, 0.73, 2.7295484576406945, 8.0):
        assert abs(nd.cdf(x) - _phi_erf(x)) < 1e-9
    for p in (0.01, 0.05, 0.5, 0.9, 0.95, 0.99, 1 - 1 / (10 * math.e)):
        assert abs(nd.inv_cdf(p) - _phi_inv_bisect(p)) < 1e-9


# --------------------------------------------------------------------------
# PSR golden values (derivation comments above)


@pytest.mark.parametrize(
    ("sr_hat", "sr0", "T", "skew", "kurt", "expected"),
    [
        # z = 25.80697580112788 -> saturated at 1.0 in float64
        (1.0, 0.0, 1000, 0.0, 3.0, 1.0),
        # z = 2.7295484576406945 -> 0.9968289436241629
        (0.5, 0.2, 120, -0.5, 4.0, 0.9968289436241629),
        # z = 4.8826658222650146 -> 0.999999476694202
        (0.3, 0.0, 250, 0.5, 5.0, 0.999999476694202),
        # z = 3.8886002778907645 -> 0.9999495879939089
        (0.8, 0.6, 500, 0.0, 3.0, 0.9999495879939089),
        # z = -3.42425031153951 -> 0.00030824903877529897
        (-0.2, 0.0, 300, 0.0, 3.0, 0.00030824903877529897),
    ],
)
def test_psr_golden(sr_hat, sr0, T, skew, kurt, expected):
    value, ok = psr(sr_hat, sr0, T, skew, kurt)
    assert ok
    assert abs(value - expected) < TOL
    # and it matches the independent recomputation exactly
    assert abs(value - _psr_ind(sr_hat, sr0, T, skew, kurt)) < 1e-9


def test_psr_matches_independent_implementation_grid():
    for sr_hat in (-1.0, -0.3, 0.0, 0.4, 1.1, 2.0):
        for T in (31, 120, 1000):
            for skew, kurt in ((0.0, 3.0), (-0.8, 4.5), (1.2, 6.0)):
                var = 1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat * sr_hat
                if var <= 0:
                    continue
                value, ok = psr(sr_hat, 0.1, T, skew, kurt)
                assert ok
                assert abs(value - _psr_ind(sr_hat, 0.1, T, skew, kurt)) < 1e-9


# --------------------------------------------------------------------------
# expected max Sharpe / DSR golden values


@pytest.mark.parametrize(
    ("M", "skew", "kurt", "T", "expected"),
    [
        # V = 1.0005005005005005, E_std = 1.5745983013457496
        (10, 0.0, 3.0, 1000, 1.5749922956722824),
        (20, 0.5, 4.0, 500, 1.8807598165989505),
        (50, -1.0, 6.0, 750, 2.3193820635516826),
        (5, 0.0, 3.0, 60, 1.1976367042720644),
        # M=1 -> exactly 0 (no selection bias; §9.1)
        (1, 0.0, 3.0, 1000, 0.0),
        (0, 0.0, 3.0, 1000, 0.0),
    ],
)
def test_expected_max_sharpe_golden(M, skew, kurt, T, expected):
    assert abs(expected_max_sharpe(M, skew, kurt, T) - expected) < TOL


def test_expected_max_sharpe_matches_independent_grid():
    for M in (2, 3, 7, 25, 100):
        for skew, kurt in ((0.0, 3.0), (-1.5, 8.0), (0.9, 4.0)):
            got = expected_max_sharpe(M, skew, kurt, 500)
            assert abs(got - _emax_ind(M, skew, kurt, 500)) < 1e-9


@pytest.mark.parametrize(
    ("sr_hat", "T", "skew", "kurt", "M", "expected"),
    [
        # SR0 = 1.5749922956722824 > SR_hat -> Phi(z) = 0.05197507293757708
        (1.5, 1000, 0.0, 3.0, 10, 0.05197507293757708),
        # M=1 -> DSR = PSR(SR_hat; 0): z = 38.72... -> 1.0
        (1.5, 1000, 0.0, 3.0, 1, 1.0),
        (2.5, 500, 0.0, 3.0, 20, 0.999999999976622),
        # SR0 = 2.0365... > SR_hat -> z << 0 -> Phi underflows to 0.0
        (1.0, 750, 1.0, 5.0, 30, 0.0),
    ],
)
def test_dsr_golden(sr_hat, T, skew, kurt, M, expected):
    value, ok = dsr(sr_hat, T, skew, kurt, M)
    assert ok
    assert abs(value - expected) < TOL


# --------------------------------------------------------------------------
# edge behavior (§9.1)


def test_dsr_refuses_below_min_t():
    with pytest.raises(GateRefusal, match="needs-more-data"):
        dsr(1.0, 29, 0.0, 3.0, 1)
    # custom floor honored
    with pytest.raises(GateRefusal, match="needs-more-data"):
        dsr(1.0, 99, 0.0, 3.0, 1, min_t=100)
    # floor is inclusive
    value, ok = dsr(1.0, 30, 0.0, 3.0, 1)
    assert ok and 0.0 <= value <= 1.0


def test_psr_refuses_t_below_2():
    with pytest.raises(GateRefusal):
        psr(1.0, 0.0, 1, 0.0, 3.0)


def test_psr_degenerate_denominator_returns_zero_with_flag():
    # var = 1 - 1*5 + (1-1)/4*25 = -4 <= 0
    value, ok = psr(5.0, 0.0, 1000, 1.0, 1.0)
    assert value == 0.0
    assert ok is False


def test_expected_max_sharpe_refuses_t_below_2():
    with pytest.raises(GateRefusal):
        expected_max_sharpe(5, 0.0, 3.0, 1)


def test_nonpositive_sr_can_not_pass():
    # even with absurdly favorable T, SR_hat <= 0 keeps p >= 0.5
    for sr_hat in (0.0, -0.5):
        value, ok = dsr(sr_hat, 100_000, 0.0, 3.0, 1)
        assert ok
        assert 1.0 - value >= 0.5


def test_bounds_and_monotonicity_in_m():
    # bounds: outputs are probabilities
    for M in (1, 2, 5, 10, 25, 50, 100):
        for sr_hat in (0.25, 0.9, 1.5, 2.5):
            value, ok = dsr(sr_hat, 500, 0.2, 4.0, M)
            assert ok
            assert 0.0 <= value <= 1.0
    # the anti-p-hacking property: at fixed SR/T, deflation grows with M —
    # DSR is non-increasing in M and strictly decreasing across this range.
    prev = math.inf
    for M in range(1, 51):
        value, _ = dsr(1.5, 500, 0.0, 3.0, M)
        assert value <= prev + 1e-12
        prev = value
    assert prev < dsr(1.5, 500, 0.0, 3.0, 1)[0]


def test_heavy_tail_skew_computes_with_caveat_worthy_values():
    # |skew| > 3: the gate must still compute (caveat attaches at service level)
    value, ok = dsr(2.0, 750, 3.5, 8.0, 10)
    assert ok and 0.0 <= value <= 1.0
    value, ok = dsr(2.0, 750, -3.5, 8.0, 10)
    assert ok and 0.0 <= value <= 1.0


# --------------------------------------------------------------------------
# MinBTL


def test_min_backtest_length_golden():
    # (target=2, M=10): derived by monotone search — T=48 is the first T with
    # (2 - SR0(T)) * sqrt(T-1) > Phi^-1(0.95) * sqrt(1 + 0.5*4) = 1.645*1.732.
    assert min_backtest_length(2.0, 10, 0.0, 3.0) == 48
    # below the asymptotic null max: unreachable at any T
    assert min_backtest_length(1.0, 10, 0.0, 3.0) == math.inf
    assert min_backtest_length(0.4, 10, 0.0, 3.0) == math.inf
    # strong signal: just above the §9.1 floor
    assert min_backtest_length(3.0, 50, 0.0, 3.0) == 31
    # M=1: no deflation, floor dominates
    assert min_backtest_length(0.5, 1, 0.0, 3.0) == 30


def test_min_backtest_length_is_consistent_with_dsr():
    """The reported MinBTL must agree with the gate: dsr p < alpha exactly at
    T = min_T (and not at min_T - 1, when above the floor)."""
    target, M = 2.2, 12
    min_t = min_backtest_length(target, M, 0.0, 3.0, alpha=0.05)
    assert isinstance(min_t, int) and min_t >= 30
    sr0 = expected_max_sharpe(M, 0.0, 3.0, min_t)
    var = 1.0 + 0.5 * target * target
    z = (target - sr0) * math.sqrt(min_t - 1) / math.sqrt(var)
    assert _phi_erf(z) > 1 - 0.05  # p < alpha at min_T
    if min_t > 30:
        sr0_prev = expected_max_sharpe(M, 0.0, 3.0, min_t - 1)
        z_prev = (target - sr0_prev) * math.sqrt(min_t - 2) / math.sqrt(var)
        assert _phi_erf(z_prev) <= 1 - 0.05  # not at min_T - 1
