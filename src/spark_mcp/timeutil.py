"""Time discipline: one ISO-8601 UTC format, one validation helper (plan §5.0).

All timestamps are `YYYY-MM-DDTHH:MM:SSZ` text so lexicographic comparison is
chronological comparison — PIT filters compare strings directly.
"""

from __future__ import annotations

from datetime import UTC, datetime

ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
MIN_ISO = "0000-01-01T00:00:00Z"
MAX_ISO = "9999-12-31T23:59:59Z"


def now_iso() -> str:
    """Current UTC time as ISO-8601 `YYYY-MM-DDTHH:MM:SSZ` text."""
    return datetime.now(UTC).strftime(ISO_FORMAT)


def validate_iso(s: str) -> str:
    """Return s if it is exactly `YYYY-MM-DDTHH:MM:SSZ`, else raise ValueError."""
    if not isinstance(s, str):
        raise ValueError(f"expected ISO-8601 UTC string, got {type(s).__name__}")
    try:
        parsed = datetime.strptime(s, ISO_FORMAT)  # raises ValueError if malformed
    except ValueError:
        raise ValueError(f"invalid ISO-8601 UTC timestamp: {s!r}") from None
    if parsed.strftime(ISO_FORMAT) != s:  # strptime tolerates non-zero-padded fields
        raise ValueError(f"invalid ISO-8601 UTC timestamp: {s!r}")
    return s
