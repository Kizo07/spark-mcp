"""MCP server shell (Phase 0): health + config tools over a real database.

Domain tools (hypothesis_propose, explore_methods, ...) land in Phases 1-3
per spark_v3_master_plan.md §6. This module owns server construction so the
CLI (`spark serve`) and tests share one factory.
"""

from __future__ import annotations

from pathlib import Path

from fastmcp import FastMCP

from . import __version__
from .config import Config
from .storage import open_db


def create_server(config: Config | None = None) -> FastMCP:
    cfg = config or Config.from_yaml()
    db_path = cfg.db_path()
    db = open_db(db_path)
    cfg.artifacts_dir().mkdir(parents=True, exist_ok=True)

    mcp = FastMCP("spark")

    @mcp.tool()
    def spark_health() -> dict:
        """Liveness + storage readiness."""
        return {
            "service": "spark",
            "version": __version__,
            "db": str(db_path),
            "db_exists": Path(db_path).exists(),
            "schema_version": db.conn.execute("PRAGMA user_version").fetchone()[0],
        }

    @mcp.tool()
    def spark_doctor() -> dict:
        """Config/capability report: degraded mode, FTS5, sqlite-vec, peers."""
        peers = {}
        for name in ("quantflow", "alpha_engine", "documents"):
            peer = getattr(cfg.peers, name)
            peers[name] = {
                "enabled": peer.enabled,
                "transport": peer.transport,
                "configured": bool(peer.command or peer.url),
            }
        return {
            "version": __version__,
            "mode": cfg.active_mode(),
            "llm_model": cfg.llm.model,
            "embedder_model": cfg.llm.embedder_model,
            "fts5": db.fts5_available(),
            "sqlite_vec": db.sqlite_vec_available(),
            "peers": peers,
            "artifacts_dir": str(cfg.artifacts_dir()),
        }

    return mcp


def main() -> None:  # `python -m spark_mcp` entrypoint (stdio MCP)
    create_server().run()


if __name__ == "__main__":
    main()
