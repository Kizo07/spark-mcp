"""Operator CLI: init-db, doctor, serve, seed import (plan §13/M10 — CLI and
MCP tools call the same service functions; no logic fork)."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print

from .config import Config
from .graph import import_seeds
from .llm import build_embedder
from .server import create_server
from .storage import open_db

app = typer.Typer(help="Spark — creativity & institutional-learning MCP")
seed_app = typer.Typer(help="Seed corpus management (plan §16.1)")
app.add_typer(seed_app, name="seed")


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


@seed_app.command("import")
def seed_import(
    config: str | None = typer.Option(None, "--config", "-c"),
    seeds_dir: str | None = typer.Option(None, "--seeds-dir"),
) -> None:
    """Import the seed corpus (technique cards + concept layer) into the graph."""
    cfg = _load(config)
    db = open_db(cfg.db_path())
    stats = import_seeds(
        db, build_embedder(cfg), seeds_dir=seeds_dir, as_of=None
    )
    db.close()
    print(stats)


if __name__ == "__main__":
    app()
