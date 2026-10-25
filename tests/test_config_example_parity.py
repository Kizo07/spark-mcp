"""Parity guard: config.example.yaml must not drift from Config defaults.

Every uncommented (explicitly set) value in config.example.yaml is compared
against the code default in src/spark_mcp/config.py. A mismatch fails unless
the dotted path is in INTENTIONAL_OVERRIDES with a stated reason. Commented-out
lines (e.g. the evolution knobs) are not explicit values and are ignored here.
"""

from pathlib import Path

import yaml

from spark_mcp.config import Config

EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "config.example.yaml"

# Dotted path -> why the example deliberately differs from the code default.
INTENTIONAL_OVERRIDES: dict[str, str] = {
    "peers.alpha_engine.command": (
        "Illustrative launch path showing how to start the alpha_engine peer; "
        "the code default is None (peer unconfigured)."
    ),
    "peers.alpha_engine.args": (
        "Example module invocation paired with peers.alpha_engine.command; "
        "the code default is [] (no args until a command is configured)."
    ),
}


def _leaf_paths(node, prefix=""):
    """Yield (dotted_path, raw_value) for each leaf; empty dicts are leaves."""
    if isinstance(node, dict) and node:
        for key, value in node.items():
            dotted = f"{prefix}.{key}" if prefix else str(key)
            yield from _leaf_paths(value, dotted)
    else:
        yield prefix, node


def _lookup(dump, dotted):
    node = dump
    for part in dotted.split("."):
        node = node[part]
    return node


def _explicit_leaves():
    raw = yaml.safe_load(EXAMPLE_PATH.read_text(encoding="utf-8")) or {}
    return dict(_leaf_paths(raw))


def test_example_values_match_code_defaults():
    parsed = Config.from_yaml(EXAMPLE_PATH).model_dump()
    defaults = Config().model_dump()
    drift = []
    for dotted in _explicit_leaves():
        if dotted in INTENTIONAL_OVERRIDES:
            continue
        example_value = _lookup(parsed, dotted)
        default_value = _lookup(defaults, dotted)
        if example_value != default_value:
            drift.append(
                f"{dotted}: example={example_value!r} != default={default_value!r}"
            )
    assert not drift, (
        "config.example.yaml drifted from src/spark_mcp/config.py defaults "
        "(fix the example, or allowlist the path in INTENTIONAL_OVERRIDES "
        "with a reason):\n" + "\n".join(drift)
    )


def test_intentional_overrides_are_live_and_justified():
    parsed = Config.from_yaml(EXAMPLE_PATH).model_dump()
    defaults = Config().model_dump()
    explicit = _explicit_leaves()
    for dotted, reason in INTENTIONAL_OVERRIDES.items():
        assert reason.strip(), f"{dotted}: override needs a reason"
        assert dotted in explicit, (
            f"{dotted}: allowlisted but not set in config.example.yaml "
            "(stale allowlist entry)"
        )
        assert _lookup(parsed, dotted) != _lookup(defaults, dotted), (
            f"{dotted}: allowlisted override now equals the default; "
            "remove the allowlist entry"
        )
