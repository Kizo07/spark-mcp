"""Tier-1 trajectory recorder (plan §15.1 C10): Spark's own tool-call and
observation events, append-only, keyed by run_id (hypothesis/episode id).
"""

from __future__ import annotations

import json
from typing import Any

from .storage import Database
from .timeutil import now_iso


def record(db: Database, run_id: str, actor: str, kind: str, payload: dict[str, Any]) -> None:
    """Append one trajectory event. Payload is JSON-serialized with sorted keys."""
    db.execute(
        "INSERT INTO trajectory_events (run_id, ts, actor, kind, payload)"
        " VALUES (?, ?, ?, ?, ?)",
        (run_id, now_iso(), actor, kind, json.dumps(payload, sort_keys=True)),
    )
