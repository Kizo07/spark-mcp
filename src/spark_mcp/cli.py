"""Operator CLI: init-db, doctor, serve (plan §13/M10 — CLI and MCP tools call
the same service functions; no logic fork)."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print

from .config import Config
from .server import create_server
from .storage import open_db

app = typer.Typer(help="Spark — creativity & institutional-learning MCP")


def _load(config: str | None) -> Config:
    return Config.from_yaml(config) if config else Config.from_yaml()


@app.command()
def init_db(config: str | None = typer.Option(None, "--config", "-c")) -> None:
    """Create/upgrade the SQLite schema."""
    cfg = _load(config)
    db = open_db(cfg.db_path())
    version = db.conn.execute("PRAGMA user_version").fetchone()[0]
    db.close()
    print(f"[green]spark.db ready[/green] at {cfg.db_path()} (schema v{version})")


@app.command()
def doctor(config: str | None = typer.Option(None, "--config", "-c")) -> None:
    """Print connectivity and capability sanity (also an MCP tool)."""
    cfg = _load(config)
    db = open_db(cfg.db_path())
    report = {
        "db_path": str(cfg.db_path()),
        "db_exists": Path(cfg.db_path()).exists(),
        "mode": cfg.active_mode(),
        "fts5": db.fts5_available(),
        "sqlite_vec": db.sqlite_vec_available(),
        "alpha_engine_enabled": cfg.peers.alpha_engine.enabled,
    }
    db.close()
    print(report)


@app.command()
def serve(config: str | None = typer.Option(None, "--config", "-c")) -> None:
    """Run the MCP server over stdio."""
    create_server(_load(config)).run()


if __name__ == "__main__":
    app()
