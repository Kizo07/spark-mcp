"""MCP server shell: the Phase-0 tools callable over the real protocol."""

import asyncio
import json
import tempfile

import pytest
from fastmcp import Client

from spark_mcp.config import Config
from spark_mcp.server import create_server


@pytest.fixture(autouse=True)
def _no_ambient_llm_key(monkeypatch):
    """Degraded-mode assertions must not depend on the ambient environment."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def _call(tool: str, args: dict | None = None):
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(
            storage={"db_path": f"{tmp}/spark.db", "artifacts_dir": f"{tmp}/art"}
        )
        server = create_server(cfg)

        async def _run():
            async with Client(server) as client:
                return await client.call_tool(tool, args or {})

        result = asyncio.run(_run())
    return result.content[0].text  # JSON payload


def test_spark_health():
    payload = json.loads(_call("spark_health"))
    assert payload["service"] == "spark"
    assert payload["db_exists"] is True
    assert payload["schema_version"] == 4


def test_spark_doctor_reports_degraded_mode_and_capabilities():
    payload = json.loads(_call("spark_doctor"))
    assert "degraded" in payload["mode"]  # no API key in test env
    assert payload["fts5"] is True
    assert payload["peers"]["alpha_engine"]["enabled"] is False
