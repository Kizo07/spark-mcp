"""RecordedProvider (plan §15.2): replays contract cassettes.

A cassette is JSON {request_fingerprint: response}; the fingerprint is the
sha1 of the canonical validated spec (protocols.spec_fingerprint), so Phase 3
contract tests replay a recorded alpha_engine session without a live peer.
Unknown fingerprints raise ToolError — a cassette must never silently
fabricate a response (that would let backtests pass unaudited).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..errors import ToolError
from .protocols import BacktestSpec, BookSpec, CausalSpec, spec_fingerprint


class RecordedProvider:
    name = "recorded"

    def __init__(self, cassette_path: str | Path):
        self.cassette_path = Path(cassette_path)
        if self.cassette_path.exists():
            self._cassette: dict[str, Any] = json.loads(
                self.cassette_path.read_text(encoding="utf-8")
            )
        else:
            self._cassette = {}

    def record(self, spec: BacktestSpec | CausalSpec | BookSpec, response: dict) -> str:
        """Append (spec -> response) to the cassette; returns the fingerprint."""
        key = spec_fingerprint(spec)
        self._cassette[key] = response
        self.cassette_path.parent.mkdir(parents=True, exist_ok=True)
        self.cassette_path.write_text(
            json.dumps(self._cassette, indent=2, sort_keys=True), encoding="utf-8"
        )
        return key

    def _replay(self, spec: BacktestSpec | CausalSpec | BookSpec, method: str) -> dict:
        key = spec_fingerprint(spec)
        if key not in self._cassette:
            raise ToolError(
                f"RecordedProvider: no cassette entry for {method} fingerprint {key} "
                f"({self.cassette_path})"
            )
        response = dict(self._cassette[key])
        response.setdefault("recorded", True)
        response["provider"] = self.name
        return response

    def backtest(self, spec: BacktestSpec) -> dict:
        return self._replay(spec, "backtest")

    def causal(self, spec: CausalSpec) -> dict:
        return self._replay(spec, "causal")

    def risk(self, spec: BookSpec) -> dict:
        return self._replay(spec, "risk")
