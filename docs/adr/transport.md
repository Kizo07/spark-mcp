# ADR — MCP transport: stay on stdio

Status: accepted (documents current behavior; no code change in this ADR).

## Decision

Spark's MCP server stays on **stdio**. Do not add a Streamable HTTP listener now. Add Streamable HTTP only when a **named hosted consumer** requires it — the consumer must be named in the change that introduces HTTP (product/host and why stdio cannot serve it). An unnamed “might need HTTP later” is not sufficient.

## Current facts (as built)

- Entrypoints: `python -m spark_mcp` (`src/spark_mcp/__main__.py`: stdio MCP server entrypoint) and `spark serve` (`src/spark_mcp/cli.py`: “Run the MCP server over stdio”) both call `create_server(...).run()` with no transport argument (`src/spark_mcp/server.py` `main()`); FastMCP's default run is the stdio transport. README's MCP example spawns the process over stdio.
- `PeerConfig.transport: stdio | http` (`src/spark_mcp/config.py`, `config.example.yaml` `peers.alpha_engine.transport: stdio`) is the *outbound* peer-client choice (Spark→alpha_engine via `_FastMCPTransport` in `src/spark_mcp/providers/alpha_engine.py`), not an inbound server listener. It is not precedent for serving Spark over HTTP.

## If HTTP is ever added (named-consumer trigger only)

- **Shared dispatch, no fork:** the HTTP surface must serve the same `create_server` tool registry and the same service functions the stdio server and CLI use (plan §13: MCP tools and CLI call one service function — e.g. `hooks.ingest_run_summary` backs both the `run_summary_ingest` tool and `spark closeout`). No HTTP-only handlers, no duplicated business logic, no second verdict path.
- **Token handling and redaction (required, not optional):** secrets stay environment-variable references only, never values in files (`llm.api_key_env`, `peers.gateway.token_env`; README: “Secrets are environment-variable references only”). Any HTTP layer must: read bearer/API tokens only from the configured env var at request time; send them only in the `Authorization` header (as `skills.promote_via_gateway` does for `QUANTFLOW_GATEWAY_TOKEN`); never log, echo, persist, or include token values (or raw `Authorization` headers) in logs, error messages, trajectory payloads, spend rows, or HTTP error bodies — redact to a fixed placeholder on every error path, including upstream transport errors. The same redaction applies to the LLM key. A change that cannot guarantee redaction does not ship.
- Keep stdio working unchanged alongside any future HTTP listener; existing stdio consumers must not need to migrate.

## Consequences

Local agent harnesses keep spawning Spark as a subprocess with no port, auth, or network surface to operate. The cost is that no hosted/remote consumer can attach until the trigger above is met and the shared-dispatch + redaction requirements are implemented and reviewed.
