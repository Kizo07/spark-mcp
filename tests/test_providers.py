"""Providers (plan §15.2): schema validation, SimulatedProvider determinism,
RecordedProvider cassette replay + fingerprint stability, AlphaEngineProvider
spec mapping with a fake transport (no live peer in CI).
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from spark_mcp.errors import ToolError
from spark_mcp.providers.alpha_engine import (
    TOOL_BACKTEST,
    TOOL_CAUSAL,
    TOOL_RISK,
    AlphaEngineProvider,
    map_spec,
)
from spark_mcp.providers.protocols import (
    BacktestSpec,
    BookSpec,
    CausalSpec,
    DateRange,
    spec_fingerprint,
)
from spark_mcp.providers.recorded import RecordedProvider
from spark_mcp.providers.simulated import SimulatedProvider


def make_spec(seed: int = 11, **overrides) -> BacktestSpec:
    base = {
        "hypothesis_id": "hyp_x",
        "signal_ref": "artifacts://signals/mom.json",
        "params": {"type": "momentum", "lookback": 20},
        "universe_pit": "TOYA,TOYB,TOYC,TOYD,TOYE,TOYF,TOYG,TOYH@2019-01-01",
        "dates": {
            "is_start": "2019-01-01",
            "is_end": "2019-12-31",
            "oos_start": "2020-01-01",
            "oos_end": "2020-06-30",
        },
        "costs": {"bps": 5, "slippage": 2},
        "splits": {"purge": 5, "embargo": 5},
        "seed": seed,
    }
    base.update(overrides)
    return BacktestSpec.model_validate(base)


# ---------------------------------------------------------------------------
# schema validation (§15.2)


def test_cost_model_rejects_zero_cost_default():
    with pytest.raises(ValidationError, match="costs must be > 0"):
        BacktestSpec.model_validate(
            {
                "hypothesis_id": "hyp_x",
                "signal_ref": "s",
                "params": {},
                "universe_pit": "u",
                "dates": {
                    "is_start": "2019-01-01",
                    "is_end": "2019-06-30",
                    "oos_start": "2019-07-01",
                    "oos_end": "2019-12-31",
                },
                "costs": {"bps": 0, "slippage": 0, "borrow": 0},
                "splits": {"purge": 0, "embargo": 0},
                "seed": 1,
            }
        )
    # borrow alone is enough to satisfy the contract
    spec = make_spec()
    assert spec.costs.bps == 5


def test_date_range_must_be_ordered():
    with pytest.raises(ValidationError, match="is_start <= is_end < oos_start"):
        DateRange(
            is_start="2019-06-01",
            is_end="2019-01-01",
            oos_start="2019-07-01",
            oos_end="2019-12-31",
        )


def test_book_spec_exactly_one_source():
    with pytest.raises(ValidationError, match="exactly one of"):
        BookSpec(factor_model="ff5")  # none given
    with pytest.raises(ValidationError, match="exactly one of"):
        BookSpec(book_id="b1", returns_uri="artifacts://r.json")
    spec = BookSpec(holdings_uri="artifacts://h.json")
    assert spec.factor_model == "ff5+momentum"  # plan default
    assert BookSpec(book_id="b1").book_id == "b1"


def test_causal_spec_method_vocabulary():
    CausalSpec(
        episode_id="ep_1",
        targets=["a"],
        conditioning=[],
        method="pcmci-parcorr",
        lags={"min_lag": 1, "max_lag": 3},
    )
    with pytest.raises(ValidationError):
        CausalSpec(
            episode_id="ep_1",
            targets=["a"],
            method="deep-learning-magic",
            lags={"min_lag": 1, "max_lag": 3},
        )


# ---------------------------------------------------------------------------
# SimulatedProvider


def test_simulated_provider_deterministic_per_seed(tmp_path):
    import json as _json

    provider = SimulatedProvider(tmp_path)
    r1 = provider.backtest(make_spec(seed=7))
    r2 = provider.backtest(make_spec(seed=7))
    assert r1 == r2  # bit-identical replay across calls
    r3 = provider.backtest(make_spec(seed=8))
    assert r3["sr"] != r1["sr"] or r3["ic_series_uri"] != r1["ic_series_uri"]

    # cross-process determinism: the seed chain never uses builtin hash()
    # (salted per process), so a fresh interpreter reproduces the metrics.
    import os
    import subprocess
    import sys

    import spark_mcp

    src_root = os.path.dirname(os.path.dirname(os.path.abspath(spark_mcp.__file__)))
    script = (
        "import json;"
        "from spark_mcp.providers.simulated import SimulatedProvider;"
        "from spark_mcp.providers.protocols import BacktestSpec;"
        f"spec = BacktestSpec.model_validate(json.loads({make_spec(seed=7).model_dump_json()!r}));"
        f"print(json.dumps(SimulatedProvider({str(tmp_path)!r}).backtest(spec), sort_keys=True))"
    )
    env = {**os.environ, "PYTHONPATH": src_root}
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, env=env
    )
    assert _json.loads(out.stdout) == r1


def test_simulated_provider_returns_gateable_metrics(tmp_path):
    result = SimulatedProvider(tmp_path).backtest(make_spec(seed=3))
    for key in ("sr", "T", "skew", "kurt", "ic_series_uri", "simulated"):
        assert key in result
    assert result["simulated"] is True
    assert result["T"] >= 30
    assert abs(result["skew"]) < 5.0
    assert 1.0 <= result["kurt"] <= 20.0
    artifact = tmp_path / "ic_series" / result["ic_series_uri"].rsplit("/", 1)[-1]
    assert artifact.exists()  # IC stream persisted as an artifact
    payload = json.loads(artifact.read_text())
    assert len(payload["ic"]) == result["T"]


def test_simulated_provider_causal_and_risk_are_synthetic(tmp_path):
    provider = SimulatedProvider(tmp_path)
    causal = provider.causal(
        CausalSpec(
            episode_id="ep_1",
            targets=["TOYA", "TOYB"],
            conditioning=["mkt"],
            method="granger",
            lags={"min_lag": 1, "max_lag": 3},
        )
    )
    assert causal["simulated"] is True and causal["method"] == "granger"
    risk = provider.risk(BookSpec(book_id="b1"))
    assert risk["simulated"] is True and risk["exposures"]


# ---------------------------------------------------------------------------
# RecordedProvider


def test_recorded_provider_replays_and_is_stable(tmp_path):
    cassette = tmp_path / "cassette.json"
    recorder = RecordedProvider(cassette)
    spec = make_spec(seed=21)
    response = {"sr": 1.23, "T": 500, "skew": 0.1, "kurt": 3.2, "simulated": False}
    fp = recorder.record(spec, response)

    # fingerprint is sha1 of the canonical spec: stable across construction
    assert fp == spec_fingerprint(spec)
    rebuilt = BacktestSpec.model_validate(json.loads(spec.model_dump_json()))
    assert spec_fingerprint(rebuilt) == fp

    player = RecordedProvider(cassette)
    result = player.backtest(make_spec(seed=21))
    assert result["sr"] == 1.23
    assert result["recorded"] is True
    assert result["provider"] == "recorded"


def test_recorded_provider_unknown_fingerprint_fails_closed(tmp_path):
    player = RecordedProvider(tmp_path / "empty.json")
    with pytest.raises(ToolError, match="no cassette entry"):
        player.backtest(make_spec(seed=99))


def test_recorded_provider_roundtrips_causal_and_risk(tmp_path):
    cassette = tmp_path / "cassette.json"
    recorder = RecordedProvider(cassette)
    causal_spec = CausalSpec(
        episode_id="ep_1", targets=["a"], conditioning=[], method="granger",
        lags={"min_lag": 1, "max_lag": 2},
    )
    book_spec = BookSpec(book_id="b1")
    recorder.record(causal_spec, {"p_values": {"a->a": 0.03}})
    recorder.record(book_spec, {"exposures": {"mkt": 1.1}})
    player = RecordedProvider(cassette)
    assert player.causal(causal_spec)["p_values"]["a->a"] == 0.03
    assert player.risk(book_spec)["exposures"]["mkt"] == 1.1


# ---------------------------------------------------------------------------
# AlphaEngineProvider (spec mapping with a fake transport — no live peer)


class FakeTransport:
    def __init__(self):
        self.calls = []

    def call(self, tool: str, payload: dict) -> dict:
        self.calls.append((tool, payload))
        return {"ok": True, "tool": tool, "echo": payload}


def test_map_spec_tool_names_and_payloads():
    bs = make_spec(seed=1)
    tool, payload = map_spec("backtest", bs)
    assert tool == TOOL_BACKTEST
    assert payload == bs.model_dump(mode="json")  # validated fields, unchanged names

    cs = CausalSpec(
        episode_id="ep_1", targets=["a"], method="pcmci-parcorr",
        lags={"min_lag": 1, "max_lag": 2},
    )
    tool, payload = map_spec("causal", cs)
    assert tool == TOOL_CAUSAL and payload == cs.model_dump(mode="json")

    book = BookSpec(returns_uri="artifacts://r.json")
    tool, payload = map_spec("risk", book)
    assert tool == TOOL_RISK and payload == book.model_dump(mode="json")

    with pytest.raises(ToolError, match="cannot map"):
        map_spec("causal", bs)


def test_alpha_engine_provider_uses_injected_transport():
    from spark_mcp.config import PeerConfig

    transport = FakeTransport()
    provider = AlphaEngineProvider(PeerConfig(enabled=True, command="/bin/false"), transport)
    result = provider.backtest(make_spec(seed=5))
    assert result["ok"] is True
    tool, payload = transport.calls[0]
    assert tool == TOOL_BACKTEST
    assert payload["seed"] == 5
    provider.causal(
        CausalSpec(
            episode_id="ep_1", targets=["a"], method="granger",
            lags={"min_lag": 1, "max_lag": 2},
        )
    )
    provider.risk(BookSpec(book_id="b1"))
    assert [t for t, _ in transport.calls] == [TOOL_BACKTEST, TOOL_CAUSAL, TOOL_RISK]
