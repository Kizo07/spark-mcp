"""SimulatedProvider (plan §15.2): deterministic seeded synthetic backtests.

Default provider for CI/demo (no live alpha_engine peer). All randomness flows
from sha256(spec.seed | universe | signal_ref | params) -> numpy default_rng,
so identical specs replay bit-identically across processes (no builtin hash —
it is salted per process).

Spark never computes returns (AGENTS.md hard rule 2): this provider IS the
compute. It synthesizes a toy price panel for the PIT universe, builds the
signal named in `params.type` (momentum | reversal | noise), measures an IC
stream, and nets the Sharpe ratio against the spec's cost model.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np

from .protocols import BacktestSpec, BookSpec, CausalSpec

DEFAULT_TOY_UNIVERSE = (
    "TOYA",
    "TOYB",
    "TOYC",
    "TOYD",
    "TOYE",
    "TOYF",
    "TOYG",
    "TOYH",
)

_SIGNAL_TYPES = ("momentum", "reversal", "noise")
_TRADING_DAYS = 252


def _spec_seed(spec: BacktestSpec | CausalSpec | BookSpec) -> int:
    """Deterministic 64-bit seed chain: builtin hash() is salted per process
    and must never seed persisted computation."""
    if isinstance(spec, BacktestSpec):
        raw = json.dumps(
            {
                "seed": spec.seed,
                "universe_pit": spec.universe_pit,
                "signal_ref": spec.signal_ref,
                "params": spec.params,
            },
            sort_keys=True,
        )
    else:
        raw = json.dumps(spec.model_dump(mode="json"), sort_keys=True)
    digest = hashlib.sha256(raw.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def _parse_universe(universe_pit: str) -> list[str]:
    """`universe_pit` is "PIT universe id + as_of" (§15.2). SimulatedProvider
    accepts "SYM1,SYM2,...[@as_of]"; anything unparseable falls back to the
    toy universe so demos/G2 always have breadth."""
    symbols = [s.strip() for s in universe_pit.split("@", 1)[0].split(",") if s.strip()]
    return symbols or list(DEFAULT_TOY_UNIVERSE)


def _business_days(d0: str, d1: str) -> int:
    start = date.fromisoformat(d0)
    end = date.fromisoformat(d1)
    if end < start:
        return 0
    # busday_count is half-open: extend `end` by one day to include it.
    next_day = date.fromordinal(end.toordinal() + 1).isoformat()
    return int(np.busday_count(start.isoformat(), next_day))


def _sample_moments(x: np.ndarray) -> tuple[float, float]:
    """(skewness, Pearson kurtosis) of a sample; normal -> (0, 3) as the PSR
    formula expects (§9.1 uses (kurt-1)/4 with Pearson kurtosis)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 4 or np.std(x) == 0.0:
        return 0.0, 3.0
    m2 = float(np.mean((x - x.mean()) ** 2))
    m3 = float(np.mean((x - x.mean()) ** 3))
    m4 = float(np.mean((x - x.mean()) ** 4))
    return m3 / m2**1.5, m4 / m2**2


class SimulatedProvider:
    """Deterministic synthetic provider. `simulated: true` marks every result."""

    name = "simulated"

    def __init__(self, artifacts_dir: str | Path | None = None):
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else None

    # -- backtest --------------------------------------------------------------

    def backtest(self, spec: BacktestSpec) -> dict:
        rng = np.random.default_rng(_spec_seed(spec))
        symbols = _parse_universe(spec.universe_pit)
        n = len(symbols)
        if n < 3:
            raise ValueError(
                f"universe {spec.universe_pit!r} resolves to {n} symbols; need >= 3 for IC"
            )
        T = _business_days(spec.dates.is_start, spec.dates.oos_end)
        if T < 10:
            raise ValueError(
                f"date range {spec.dates.is_start}..{spec.dates.oos_end} gives {T} business"
                " days; need >= 10"
            )

        # Synthetic panel: one common factor + idiosyncratic noise per symbol.
        vol = float(spec.params.get("vol", 0.02))
        beta = 0.5 + rng.uniform(0.0, 0.5, size=n)
        factor = rng.normal(0.0, vol * 0.6, size=T)
        idio = rng.normal(0.0, vol * 0.8, size=(T, n))
        rets = beta[None, :] * factor[:, None] + idio

        # Signal from params.type (frozen by the same rng: deterministic).
        lookback = max(1, int(spec.params.get("lookback", 20)))
        signal_type = str(spec.params.get("type", "momentum"))
        if signal_type not in _SIGNAL_TYPES:
            raise ValueError(
                f"unknown signal params.type {signal_type!r}; supported: {list(_SIGNAL_TYPES)}"
            )
        csum = np.vstack([np.zeros((1, n)), np.cumsum(rets, axis=0)])
        # signal[t] = sum of returns (t-lookback+1 .. t); rows t < lookback-1 are NaN.
        mom = csum[lookback:] - csum[:-lookback]
        pad = np.full((lookback - 1, n), np.nan)
        signal = np.vstack([pad, mom])
        if signal_type == "reversal":
            signal = -signal
        elif signal_type == "noise":
            signal = rng.normal(0.0, 1.0, size=(T, n))

        # Forward returns over the holding period: rets[t+1 .. t+holding]
        # (strictly after the signal date — sharing rets[t] would fabricate
        # mechanical IC).
        holding = max(1, int(spec.params.get("holding", 1)))
        fwd = np.full((T, n), np.nan)
        csum_r = np.vstack([np.zeros((1, n)), np.cumsum(rets, axis=0)])
        fwd[: T - holding] = csum_r[holding + 1 :] - csum_r[1 : T - holding + 1]

        # IC stream: cross-sectional correlation signal vs forward, per day.
        valid = ~np.isnan(signal).any(axis=1)
        valid[T - holding :] = False
        idx = np.where(valid)[0]
        ic = np.array(
            [
                np.corrcoef(signal[t], fwd[t])[0, 1]
                if np.std(signal[t]) > 0 and np.std(fwd[t]) > 0
                else 0.0
                for t in idx
            ]
        )
        # Strategy daily returns: z-scored signal tilt x forward return.
        strat = np.array(
            [
                float(np.mean(_zscore(signal[t]) * fwd[t]))
                if np.std(signal[t]) > 0
                else 0.0
                for t in idx
            ]
        )

        gross_sr = _annualized_sr(strat)
        cost_rate = (spec.costs.bps + spec.costs.slippage + spec.costs.borrow) / 1.0e4
        turnover = float(spec.params.get("turnover", 1.0))
        net = strat - cost_rate * turnover
        sr = _annualized_sr(net)
        skew, kurt = _sample_moments(net)

        ic_uri = self._write_ic_series(spec, idx, ic)
        return {
            "sr": float(sr),
            "T": int(net.size),
            "skew": float(skew),
            "kurt": float(kurt),
            "gross_sr": float(gross_sr),
            "cost_drag": float(gross_sr - sr),
            "ic_mean": float(np.mean(ic)) if ic.size else 0.0,
            "ic_series_uri": ic_uri,
            "symbols": n,
            "simulated": True,
            "provider": self.name,
        }

    def _write_ic_series(self, spec: BacktestSpec, idx: np.ndarray, ic: np.ndarray) -> str | None:
        if self.artifacts_dir is None:
            return None
        out_dir = self.artifacts_dir / "ic_series"
        out_dir.mkdir(parents=True, exist_ok=True)
        name = f"{_spec_seed(spec):016x}.json"
        path = out_dir / name
        payload = {
            "signal_ref": spec.signal_ref,
            "days": [int(t) for t in idx],
            "ic": [float(v) for v in ic],
            "simulated": True,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return f"artifacts://ic_series/{name}"

    # -- causal / risk (trivial synthetic versions so the loop runs end-to-end) --

    def causal(self, spec: CausalSpec) -> dict:
        rng = np.random.default_rng(_spec_seed(spec))
        n_series = max(2, len(spec.targets))
        t = 300
        eps = rng.normal(0.0, 1.0, size=(t, n_series))
        x = np.zeros((t, n_series))
        for i in range(1, n_series):
            x[:, i] = 0.4 * np.roll(x[:, i - 1], 2) + eps[:, i]
        x[0:2, :] = eps[0:2, :]
        p_values: dict[str, float] = {}
        for i in range(n_series):
            for j in range(n_series):
                if i != j:
                    # synthetic stand-in for granger/pcmci: lagged regression F-test
                    lag = max(1, spec.lags.min_lag)
                    y = x[lag:, j]
                    xlag = x[: t - lag, i]
                    b = float(np.dot(xlag, y) / (np.dot(xlag, xlag) + 1e-12))
                    resid = y - b * xlag
                    f_stat = (np.var(y) - np.var(resid)) / (np.var(resid) + 1e-12) * (t - 2 * lag)
                    p_values[f"{spec.targets[i]}->{spec.targets[j]}"] = float(
                        np.clip(np.exp(-0.5 * max(f_stat, 0.0)), 0.0, 1.0)
                    )
        return {
            "method": spec.method,
            "targets": list(spec.targets),
            "conditioning": list(spec.conditioning),
            "lags": spec.lags.model_dump(),
            "p_values": p_values,
            "simulated": True,
            "provider": self.name,
            "note": "synthetic coupling stand-in; real Granger/PCMCI runs via alpha_engine",
        }

    def risk(self, spec: BookSpec) -> dict:
        rng = np.random.default_rng(_spec_seed(spec))
        factors = [f.strip() for f in spec.factor_model.split("+") if f.strip()]
        exposures = {f: float(rng.normal(0.0, 0.5)) for f in factors}
        return {
            "factor_model": spec.factor_model,
            "book": spec.book_id or spec.holdings_uri or spec.returns_uri,
            "exposures": exposures,
            "specific_risk": float(abs(rng.normal(0.0, 0.1))),
            "simulated": True,
            "provider": self.name,
        }


def _zscore(x: np.ndarray) -> np.ndarray:
    sd = float(np.std(x))
    if sd == 0.0:
        return np.zeros_like(x)
    return (x - float(np.mean(x))) / sd


def _annualized_sr(returns: np.ndarray) -> float:
    returns = np.asarray(returns, dtype=np.float64)
    sd = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    if sd == 0.0 or returns.size < 2:
        return 0.0
    return float(np.mean(returns) / sd * np.sqrt(_TRADING_DAYS))
