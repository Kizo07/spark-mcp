"""P2-2: MCP conformance + secret-redaction checks.

- create_server lists tools; every tool carries a non-empty description
  and a JSON-schema (object with properties) for its parameters.
- Error shape: a bad call surfaces as a ToolError with a message (the
  FastMCP client harness raises ToolError; we assert shape, not success).
- Redaction: spark_doctor (and spark_health) output must never contain
  the *values* of QUANTFLOW_GATEWAY_TOKEN / OPENAI_API_KEY when those
  env vars are set to sentinel values.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError as ClientToolError

from spark_mcp.config import Config
from spark_mcp.server import create_server

GW_SENTINEL = "SENTINEL_GW_TOKEN_9f8e7d6c5b4a"
OAI_SENTINEL = "SENTINEL_OAI_KEY_1a2b3c4d5e6f"


def _cfg(tmp_path):
    return Config(
        storage={"db_path": f"{tmp_path}/spark.db", "artifacts_dir": f"{tmp_path}/art"}
    )


def _server(tmp_path):
    return create_server(_cfg(tmp_path), seeds=False)


def _input_schema(tool) -> dict:
    # FastMCP/MCP SDK v2 renamed inputSchema -> input_schema; accept either.
    schema = getattr(tool, "input_schema", None)
    if schema is None:
        schema = getattr(tool, "inputSchema", None)
    assert schema is not None, f"{tool.name}: no input schema attribute"
    if not isinstance(schema, dict):
        schema = schema.model_dump() if hasattr(schema, "model_dump") else dict(schema)
    return schema


def test_list_tools_every_tool_has_description_and_schema(tmp_path):
    async def run():
        async with Client(_server(tmp_path)) as client:
            tools = await client.list_tools()
            assert len(tools) >= 20, f"expected >=20 tools, got {len(tools)}"
            names = [t.name for t in tools]
            assert len(names) == len(set(names)), "tool names must be unique"
            for tool in tools:
                assert tool.name and tool.name.strip(), "tool name empty"
                assert tool.description and tool.description.strip(), (
                    f"{tool.name}: missing/empty description"
                )
                schema = _input_schema(tool)
                assert isinstance(schema, dict), f"{tool.name}: schema not a dict"
                assert schema.get("type") == "object", (
                    f"{tool.name}: schema type != object: {schema.get('type')!r}"
                )
                assert "properties" in schema, f"{tool.name}: schema missing properties"
                assert isinstance(schema["properties"], dict), (
                    f"{tool.name}: schema properties not a dict"
                )
            # spot-check core tools are present
            assert "spark_health" in names
            assert "spark_doctor" in names
            assert "report_write" in names

    asyncio.run(run())


def test_error_shape_bad_call_raises_tool_error(tmp_path):
    async def run():
        async with Client(_server(tmp_path)) as client:
            # missing the exactly-one id -> named ToolError, not a crash/empty
            with pytest.raises(ClientToolError) as excinfo:
                await client.call_tool("report_write", {})
            assert "exactly one" in str(excinfo.value)

            with pytest.raises(ClientToolError) as excinfo2:
                await client.call_tool(
                    "report_write", {"hypothesis_id": "h", "episode_id": "e"}
                )
            assert "exactly one" in str(excinfo2.value)

            # unknown tool name -> ToolError mentioning the tool
            with pytest.raises(ClientToolError) as excinfo3:
                await client.call_tool("no_such_tool_xyz", {})
            assert "no_such_tool_xyz" in str(excinfo3.value)

    asyncio.run(run())


def test_doctor_output_redacts_env_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTFLOW_GATEWAY_TOKEN", GW_SENTINEL)
    monkeypatch.setenv("OPENAI_API_KEY", OAI_SENTINEL)

    async def run():
        async with Client(_server(tmp_path)) as client:
            doctor_text = (await client.call_tool("spark_doctor", {})).content[0].text
            health_text = (await client.call_tool("spark_health", {})).content[0].text
            # payloads are valid JSON
            doctor = json.loads(doctor_text)
            assert doctor["version"]
            assert "mode" in doctor
            # sentinel *values* must never appear anywhere in the output
            assert GW_SENTINEL not in doctor_text
            assert OAI_SENTINEL not in doctor_text
            assert GW_SENTINEL not in health_text
            assert OAI_SENTINEL not in health_text
            # mode may report key *presence* but never the value
            assert GW_SENTINEL not in str(doctor.get("mode", ""))
            assert OAI_SENTINEL not in str(doctor.get("mode", ""))

    asyncio.run(run())


def test_cli_doctor_output_redacts_env_secrets(tmp_path, monkeypatch, capsys):
    """The `spark doctor` CLI command (cli.doctor) shares the redaction duty."""
    monkeypatch.setenv("QUANTFLOW_GATEWAY_TOKEN", GW_SENTINEL)
    monkeypatch.setenv("OPENAI_API_KEY", OAI_SENTINEL)
    monkeypatch.setenv("SPARK_CONFIG", str(tmp_path / "nope.yaml"))

    from spark_mcp import cli

    # cli.doctor loads the default config (missing file -> defaults) and
    # opens the default db path; point storage at tmp via a config file.
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"storage:\n  db_path: {tmp_path}/spark.db\n  artifacts_dir: {tmp_path}/art\n",
        encoding="utf-8",
    )
    cli.doctor(config=str(cfg_path))
    out = capsys.readouterr().out
    assert GW_SENTINEL not in out
    assert OAI_SENTINEL not in out
