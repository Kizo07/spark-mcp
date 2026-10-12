"""The independence gate (plan §17.6, blocking).

Proves spark-mcp works installed alone: no reference-repo imports anywhere in
src/, no reference module enters sys.modules at runtime, the MCP server boots
and answers in a degraded (no-key) environment.
"""

import ast
import sys
from pathlib import Path

import pytest

# tests/independence/test_independence_gate.py -> parents[2] is the repo root.
# (parents[1] silently points at a nonexistent tests/src/spark_mcp and the scan
# goes vacuous — caught by the Phase 0 milestone review.)
SRC = Path(__file__).resolve().parents[2] / "src" / "spark_mcp"


@pytest.fixture(autouse=True)
def _no_ambient_llm_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

FORBIDDEN_MODULES = {
    "hermes", "hermes_agent", "docling", "langmem", "graphiti",
    "dspy", "mlflow", "aflow", "openhands", "OpenHands", "qdax",
    "pyribs",  # optional [qd] extra only; core must not require it
    "rdagent",  # evolution upgrade plan §0.1/H: native reimplementation only
}


def _imports_in_file(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    return found


def test_no_reference_repo_imports_in_source():
    scanned = list(SRC.rglob("*.py"))
    assert scanned, f"scan found no source files under {SRC} (path drift?)"
    offenders = []
    for py in scanned:
        bad = _imports_in_file(py) & FORBIDDEN_MODULES
        if bad:
            offenders.append((py.name, sorted(bad)))
    assert not offenders, f"forbidden imports found: {offenders}"


def test_no_rdagent_references_in_source():
    """Evolution upgrade plan §H: no vendored/copied reference-loop code.

    Fails on any case-insensitive `rdagent` / `RD-Agent` literal under src/
    (no allowlist — vendored code would trip this via its module docstrings
    and imports), and asserts `rdagent` appears in neither pyproject.toml's
    dependencies nor any lockfile present in the repo root.
    """
    import re

    scanned = list(SRC.rglob("*.py"))
    assert scanned, f"scan found no source files under {SRC} (path drift?)"
    offenders = []
    for py in scanned:
        text = py.read_text(encoding="utf-8")
        if re.search(r"rdagent", text, re.IGNORECASE) or re.search(
            r"rd-agent", text, re.IGNORECASE
        ):
            offenders.append(py.name)
    assert not offenders, f"rdagent references found in src/: {offenders}"

    root = SRC.parents[1]
    project = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert "rdagent" not in project.lower(), "rdagent in pyproject.toml dependencies"
    for lock in ("uv.lock", "poetry.lock", "Pipfile.lock", "requirements.txt"):
        lock_path = root / lock
        if lock_path.exists():
            assert "rdagent" not in lock_path.read_text(encoding="utf-8").lower(), (
                f"rdagent in {lock}"
            )


def test_reference_modules_never_enter_sysmodules():
    import asyncio
    import json
    import tempfile

    from fastmcp import Client

    from spark_mcp.config import Config
    from spark_mcp.server import create_server

    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(
            storage={"db_path": f"{tmp}/spark.db", "artifacts_dir": f"{tmp}/art"}
        )
        server = create_server(cfg)

        async def _run():
            async with Client(server) as client:
                health = json.loads((await client.call_tool("spark_health", {})).content[0].text)
                doctor = json.loads((await client.call_tool("spark_doctor", {})).content[0].text)
                return health, doctor

        health, doctor = asyncio.run(_run())

    loaded = {m.split(".")[0] for m in sys.modules} & FORBIDDEN_MODULES
    assert not loaded, f"reference modules entered sys.modules: {loaded}"
    assert health["service"] == "spark"
    assert "degraded" in doctor["mode"]
