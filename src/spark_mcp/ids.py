"""ULID-ish unique ids with type prefixes (plan §5.0 N2/N8).

Time-ordered: a millisecond timestamp (13 hex chars) followed by 72 bits of
CSPRNG randomness (18 hex chars), so ids are roughly sortable by creation time
and collisions within the same millisecond are negligible. Never use Python's
builtin hash() for persisted identifiers — it is salted per process.
"""

from __future__ import annotations

import secrets
import time

_TIMESTAMP_HEX_DIGITS = 13  # milliseconds since epoch fits until year 2286
_RANDOM_BYTES = 9  # 72 bits


def _ulidish() -> str:
    ms = int(time.time() * 1000)
    return f"{ms:0{_TIMESTAMP_HEX_DIGITS}x}{secrets.token_hex(_RANDOM_BYTES)}"


def new_hypothesis_id() -> str:
    return f"hyp_{_ulidish()}"


def new_episode_id() -> str:
    return f"ep_{_ulidish()}"


def new_trial_id() -> str:
    return f"trial_{_ulidish()}"


def new_belief_id() -> str:
    return f"belief_{_ulidish()}"


def new_tombstone_id() -> str:
    return f"tomb_{_ulidish()}"


def new_node_id() -> str:
    return f"node_{_ulidish()}"


def new_edge_id() -> str:
    return f"edge_{_ulidish()}"


def new_skill_id() -> str:
    return f"skill_{_ulidish()}"


def new_candidate_id() -> str:
    return f"cand_{_ulidish()}"


def new_approval_id() -> str:
    return f"appr_{_ulidish()}"


def new_consolidation_id() -> str:
    return f"cons_{_ulidish()}"


def new_exercise_id() -> str:
    return f"exercise_{_ulidish()}"


def new_gap_id() -> str:
    return f"gap_{_ulidish()}"


def new_evolution_line_id() -> str:
    return f"elif_{_ulidish()}"


def new_evolution_run_id() -> str:
    return f"erun_{_ulidish()}"


def new_evolution_feedback_id() -> str:
    return f"efb_{_ulidish()}"
