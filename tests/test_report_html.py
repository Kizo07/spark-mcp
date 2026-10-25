"""P2-6: report_write format='html' built-in single-file renderer.

The md path must stay byte-identical in behaviour; html is a new branch
gated on research.report_html_renderer='builtin' (the new default).
All interpolated ledger text is escaped, and the document carries a
provenance stamp (generated-at + spark-mcp version).
"""

from __future__ import annotations

import asyncio
import json
import re

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError as ClientToolError

from spark_mcp import __version__, ledger
from spark_mcp.config import Config
from spark_mcp.server import create_server
from spark_mcp.storage import open_db


def _cfg(tmp_path, **research_overrides):
    research = {}
    if research_overrides:
        research = research_overrides
    return Config(
        storage={"db_path": f"{tmp_path}/spark.db", "artifacts_dir": f"{tmp_path}/art"},
        research=research,  # type: ignore[arg-type]
    )


def _server(tmp_path, **kw):
    return create_server(_cfg(tmp_path, **kw), seeds=False)


def _seed_episode(tmp_path, mechanism="plain mechanism"):
    cfg = _cfg(tmp_path)
    db = open_db(cfg.db_path())
    hyp = ledger.propose_hypothesis(
        db,
        cfg,
        research_family="quant",
        hypothesis_type="predictive",
        mechanism=mechanism,
        expected_direction="positive",
        horizon="20d",
        scope="toy-scope",
        falsifier="falsified if IC < 0",
        predicted_ic=0.03,
        predicted_sigma=0.01,
        universe="toy",
        idea="toy idea",
    )
    reg = ledger.register_hypothesis(db, cfg, hyp["hypothesis_id"], {"t": 1}, 5)
    db.close()
    return hyp, reg


def _call(tmp_path, tool, args, **server_kw):
    async def run():
        async with Client(_server(tmp_path, **server_kw)) as client:
            result = await client.call_tool(tool, args)
            return result.content[0].text

    return asyncio.run(run())


def test_html_episode_report_is_single_file_with_provenance(tmp_path):
    _hyp, reg = _seed_episode(tmp_path)
    payload = json.loads(
        _call(tmp_path, "report_write", {"episode_id": reg["episode_id"], "format": "html"})
    )
    assert payload["format"] == "html"
    html_doc = payload["html"]
    assert html_doc.startswith("<!DOCTYPE html>")
    assert '<html lang="en">' in html_doc
    assert '<meta charset="utf-8">' in html_doc
    # single-file: inline style, no external resources
    assert "<style>" in html_doc
    assert "http://" not in html_doc
    assert "https://" not in html_doc
    assert "<link" not in html_doc
    assert "<script" not in html_doc
    # provenance stamp: generated-at timestamp + spark-mcp version
    assert "Generated at" in html_doc
    assert __version__ in html_doc
    # ISO-8601 UTC timestamp present in the stamp
    assert re.search(r"Generated at \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", html_doc)
    assert "Spark ledger report" in html_doc
    assert reg["episode_id"] in html_doc
    assert "## Trials" not in html_doc  # markdown heading must not leak into html
    assert "<h2>Trials</h2>" in html_doc


def test_html_hypothesis_report(tmp_path):
    hyp, reg = _seed_episode(tmp_path)
    payload = json.loads(
        _call(tmp_path, "report_write", {"hypothesis_id": hyp["hypothesis_id"], "format": "html"})
    )
    assert payload["format"] == "html"
    html_doc = payload["html"]
    assert hyp["hypothesis_id"] in html_doc
    assert reg["episode_id"] in html_doc
    assert "Episodes" in html_doc


def test_html_escapes_interpolated_text(tmp_path):
    payload_mechanism = '<script>alert("xss")</script> & <b>bold</b>'
    _hyp, reg = _seed_episode(tmp_path, mechanism=payload_mechanism)
    payload = json.loads(
        _call(tmp_path, "report_write", {"episode_id": reg["episode_id"], "format": "html"})
    )
    html_doc = payload["html"]
    # raw payload must not appear; escaped form must
    assert "<script>alert" not in html_doc
    assert "&lt;script&gt;" in html_doc
    assert "&lt;b&gt;bold&lt;/b&gt;" in html_doc
    assert "&amp;" in html_doc


def test_md_path_unchanged(tmp_path):
    _hyp, reg = _seed_episode(tmp_path)
    payload = json.loads(
        _call(tmp_path, "report_write", {"episode_id": reg["episode_id"], "format": "md"})
    )
    assert payload["format"] == "md"
    assert "Spark ledger report" in payload["markdown"]
    assert "## Trials" in payload["markdown"]
    # default format is md
    payload_default = json.loads(
        _call(tmp_path, "report_write", {"episode_id": reg["episode_id"]})
    )
    assert payload_default["format"] == "md"
    assert payload_default["markdown"] == payload["markdown"]


def test_html_rejected_when_renderer_deferred(tmp_path):
    _hyp, reg = _seed_episode(tmp_path)

    async def run():
        async with Client(_server(tmp_path, report_html_renderer="deferred")) as client:
            with pytest.raises(ClientToolError) as excinfo:
                await client.call_tool(
                    "report_write", {"episode_id": reg["episode_id"], "format": "html"}
                )
            assert "report_html_renderer" in str(excinfo.value)
            assert "builtin" in str(excinfo.value)

    asyncio.run(run())


def test_unknown_format_rejected(tmp_path):
    _hyp, reg = _seed_episode(tmp_path)

    async def run():
        async with Client(_server(tmp_path)) as client:
            with pytest.raises(ClientToolError) as excinfo:
                await client.call_tool(
                    "report_write", {"episode_id": reg["episode_id"], "format": "pdf"}
                )
            assert "md" in str(excinfo.value)
            assert "html" in str(excinfo.value)

    asyncio.run(run())


def test_html_unknown_episode_still_errors(tmp_path):
    async def run():
        async with Client(_server(tmp_path)) as client:
            with pytest.raises(ClientToolError, match="unknown episode_id"):
                await client.call_tool(
                    "report_write", {"episode_id": "ep_missing", "format": "html"}
                )

    asyncio.run(run())
