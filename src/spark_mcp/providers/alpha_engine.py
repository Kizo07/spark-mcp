"""AlphaEngineProvider (plan §15.2): config-driven stdio MCP client wrapper.

Speaks MCP over stdio to the alpha_engine peer (`peers.alpha_engine`:
command/args or http url) via fastmcp's Client. Spark holds verdicts, never
returns — this class only maps Spark specs onto alpha_engine tool calls and
returns the peer's payload verbatim.

The transport is injectable: `AlphaEngineProvider(config, transport=fake)`
keeps unit tests off the network/stdio entirely (no live peer in CI). The
fastmcp import stays inside `call` so importing this module never requires
a usable MCP client (AGENTS.md: peers are optional, never dependencies).
"""

from __future__ import annotations

from typing import Any, Protocol

from ..config import PeerConfig
from ..errors import ToolError
from .protocols import BacktestSpec, BookSpec, CausalSpec

#: alpha_engine tool names for the three §15.2 methods (mapped, overridable via config env).
TOOL_BACKTEST = "run_backtest"
TOOL_CAUSAL = "causal_test"
TOOL_RISK = "risk_decompose"


class _Transport(Protocol):
    def call(self, tool: str, payload: dict) -> dict: ...


class _FastMCPTransport:
    """stdio (or http) MCP client transport built from the peer config."""

    def __init__(self, peer: PeerConfig):
        self._peer = peer

    def _client_target(self) -> Any:
        """Transport target for fastmcp's Client: url string or server-config
        dict. A fresh Client is built from this per call — fastmcp 4 clients
        bind to the event loop that opens them, so caching one across the
        per-call ``asyncio.run`` loops below dies with 'Event loop is closed'
        on the second use."""
        if self._peer.transport == "http" and self._peer.url:
            return self._peer.url
        if self._peer.command:
            import shlex

            args = self._peer.args or []
            if len(args) == 1 and " " in args[0]:
                args = shlex.split(args[0])
            return {
                "mcpServers": {
                    "alpha_engine": {"command": self._peer.command, "args": list(args)}
                }
            }
        raise ToolError(
            "alpha_engine peer enabled but neither url nor command configured"
        )

    def call(self, tool: str, payload: dict) -> dict:
        import asyncio

        from fastmcp import Client  # lazy: importing this module stays cheap

        async def _invoke() -> Any:
            # fastmcp's Client is an async context manager (the sync ``with``
            # shim was removed in fastmcp 4); each call opens and closes one
            # stdio connection inside a fresh event loop.
            async with Client(self._client_target()) as client:
                return await client.call_tool(tool, payload)

        result = asyncio.run(_invoke())
        if getattr(result, "is_error", False):
            raise ToolError(
                f"alpha_engine tool {tool!r} returned an error: {result.content!r}"
            )
        content = result.content
        if isinstance(content, list) and content:
            # MCP text-content JSON payloads; structured content passes through.
            text = getattr(content[0], "text", None)
            if text is not None:
                import json

                return json.loads(text)
        return {"content": content}


def map_spec(method: str, spec: BacktestSpec | CausalSpec | BookSpec) -> tuple[str, dict]:
    """The spec-mapping contract: Spark schema -> (tool_name, payload).

    Pure function — unit-tested with a fake transport, no live peer needed.
    Payloads are the spec's validated field dicts (mode="json"), so the peer
    receives exactly what Spark validated, key names unchanged.
    """
    payload = spec.model_dump(mode="json")
    if method == "backtest" and isinstance(spec, BacktestSpec):
        return TOOL_BACKTEST, payload
    if method == "causal" and isinstance(spec, CausalSpec):
        return TOOL_CAUSAL, payload
    if method == "risk" and isinstance(spec, BookSpec):
        return TOOL_RISK, payload
    raise ToolError(f"cannot map {type(spec).__name__} onto alpha_engine method {method!r}")


class AlphaEngineProvider:
    name = "alpha_engine"

    def __init__(self, peer: PeerConfig, transport: _Transport | None = None):
        self._peer = peer
        self._transport = transport if transport is not None else _FastMCPTransport(peer)

    def backtest(self, spec: BacktestSpec) -> dict:
        tool, payload = map_spec("backtest", spec)
        return self._transport.call(tool, payload)

    def causal(self, spec: CausalSpec) -> dict:
        tool, payload = map_spec("causal", spec)
        return self._transport.call(tool, payload)

    def risk(self, spec: BookSpec) -> dict:
        tool, payload = map_spec("risk", spec)
        return self._transport.call(tool, payload)
