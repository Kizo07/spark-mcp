"""Operator CLI: init-db, doctor, serve, seed import, closeout (plan §13/M10 —
CLI and MCP tools call the same service functions; no logic fork)."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print

from . import hooks
from . import skills as skills_service
from .config import Config
from .graph import import_seeds
from .llm import build_embedder, build_llm
from .server import create_server
from .storage import open_db

app = typer.Typer(help="Spark — creativity & institutional-learning MCP")
seed_app = typer.Typer(help="Seed corpus management (plan §16.1)")
app.add_typer(seed_app, name="seed")
skills_app = typer.Typer(help="Skill evolution (plan §11: distill/challenger/replay/approve)")
app.add_typer(skills_app, name="skills")


def _load(config: str | None) -> Config:
    return Config.from_yaml(config) if config else Config.from_yaml()


def _csv(value: str | None) -> list[str] | None:
    """Comma-separated CLI option -> list (None when absent)."""
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _json_arg(value: str) -> list[str]:
    """JSON-array CLI option; '@path' reads the array from a file."""
    text = Path(value[1:]).read_text(encoding="utf-8") if value.startswith("@") else value
    parsed = json.loads(text)
    if not isinstance(parsed, list) or any(not isinstance(s, str) for s in parsed):
        raise typer.BadParameter("expected a JSON array of strings")
    return parsed


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


@app.command()
def closeout(
    payload: str = typer.Option(
        ..., "--payload", help="Path to a QuantFlow Tier-2 run-summary JSON file (plan §15.1)."
    ),
    config: str | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Ingest a QuantFlow Tier-2 run-summary payload from a JSON file.

    The cron/manual-operator close-out path: same hooks.ingest_run_summary the
    run_summary_ingest MCP tool calls (plan §13, no logic fork). Prints the
    returned dict as JSON; a schema violation exits non-zero with the
    HookSchemaError message.
    """
    cfg = _load(config)
    db = open_db(cfg.db_path())
    data = json.loads(Path(payload).read_text(encoding="utf-8"))
    result = hooks.ingest_run_summary(db, cfg, data)
    db.close()
    print(json.dumps(result, indent=2, sort_keys=True))


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


# -- skills subgroup (plan §11; same service functions the MCP tools call, N13) --


@skills_app.command("distill")
def skills_distill(
    episode_ids: str = typer.Option(..., "--episode-ids", help="Comma-separated episode ids."),
    name: str = typer.Option(..., "--name"),
    role: str | None = typer.Option(None, "--role", help="quant|ml|nlp|risk|lead"),
    procedure_steps: str = typer.Option(
        ...,
        "--procedure-steps",
        help="JSON array of step strings, or @path to a JSON file.",
    ),
    references: str | None = typer.Option(None, "--references", help="Comma-separated."),
    client_id: str | None = typer.Option(None, "--client-id"),
    config: str | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Distill a draft skill from episodes carrying trajectory evidence
    (plan §6.3 tool 16). Same skills.distill_skill the MCP tool calls."""
    cfg = _load(config)
    db = open_db(cfg.db_path())
    result = skills_service.distill_skill(
        db,
        cfg,
        _csv(episode_ids) or [],
        name,
        role,
        _json_arg(procedure_steps),
        references=_csv(references),
        client_id=client_id,
    )
    db.close()
    print(json.dumps(result, indent=2, sort_keys=True))


@skills_app.command("challenger")
def skills_challenger(
    skill_name: str = typer.Option(..., "--skill-name"),
    op: str = typer.Option(..., "--op", help="repair|split|merge|paraphrase|gepa"),
    source_episode_ids: str | None = typer.Option(None, "--source-episode-ids"),
    client_id: str | None = typer.Option(None, "--client-id"),
    config: str | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Propose a challenger via a mutation operator / mini-GEPA
    (plan §6.3 tool 17). Same skills.propose_skill_challenger the MCP tool calls."""
    cfg = _load(config)
    db = open_db(cfg.db_path())
    result = skills_service.propose_skill_challenger(
        db,
        cfg,
        skill_name,
        op,
        source_episode_ids=_csv(source_episode_ids),
        client_id=client_id,
        llm=build_llm(cfg),
    )
    db.close()
    print(json.dumps(result, indent=2, sort_keys=True))


@skills_app.command("replay")
def skills_replay(
    skill_name: str | None = typer.Option(None, "--skill-name"),
    suite: str | None = typer.Option(None, "--suite", help="Fixture family filter."),
    task_ids: str | None = typer.Option(None, "--task-ids", help="Comma-separated."),
    seed: int | None = typer.Option(None, "--seed"),
    config: str | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Replay-suite verdicts with the held-in/held-out split and heldout_gap
    (plan §6.3 tool 18). Same skills.run_replay the MCP tool calls."""
    cfg = _load(config)
    db = open_db(cfg.db_path())
    result = skills_service.run_replay(
        db,
        cfg,
        skill_name=skill_name,
        suite=suite,
        task_ids=_csv(task_ids),
        seed=seed,
        llm=build_llm(cfg),
    )
    db.close()
    print(json.dumps(result, indent=2, sort_keys=True))


@skills_app.command("approve")
def skills_approve(
    target_type: str = typer.Option(..., "--target-type", help="belief|skill|adoption"),
    target_id: str = typer.Option(..., "--target-id"),
    decision: str = typer.Option(..., "--decision", help="approved|rejected"),
    rationale: str = typer.Option(..., "--rationale"),
    decided_by: str = typer.Option("human:operator", "--decided-by"),
    config: str | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Human gate for pending_human beliefs, staged champions, and adoptions
    (plan §6.3 tool 22, M11). Same skills.approve the MCP tool calls."""
    cfg = _load(config)
    db = open_db(cfg.db_path())
    result = skills_service.approve(
        db, cfg, target_type, target_id, decision, rationale, decided_by=decided_by
    )
    db.close()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
