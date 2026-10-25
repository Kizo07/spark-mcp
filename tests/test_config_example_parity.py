"""Parity guard: config.example.yaml must not drift from Config defaults.

Every uncommented (explicitly set) value in config.example.yaml is compared
against the code default in src/spark_mcp/config.py. A mismatch fails unless
the dotted path is in INTENTIONAL_OVERRIDES with a stated reason. Commented-out
lines (e.g. the evolution knobs) are not explicit values and are ignored by
the drift test — but they DO count as coverage: every ResearchConfig field
must be present, commented out, or listed in INTENTIONAL_OMISSIONS with a
reason, so newly added config fields cannot slip past the example silently.
"""

import re
from pathlib import Path

import yaml

from spark_mcp.config import Config, ResearchConfig

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

# ResearchConfig field -> why it is deliberately absent from config.example.yaml
# (neither set nor commented out). New config fields without an example entry
# fail test_research_fields_covered_by_example until they are added here.
INTENTIONAL_OMISSIONS: dict[str, str] = {
    "selection_recall_pad_to": (
        "G1 recall floor (30) is a stable calibrated default; the example "
        "keeps the selection block minimal and the knob is documented in "
        "src/spark_mcp/config.py."
    ),
    "scoring_full_field": (
        "Full-field LLM scoring (True) is the post-round-5 architecture; "
        "disabling it is a degraded-mode escape hatch, not operator tuning."
    ),
    "scoring_workers": (
        "Thread count (8) is machine-dependent; operators tune per host and "
        "the default suits local runs."
    ),
    "informed_prior_after_episodes": (
        "Consolidation warmup threshold (20) is internal exploration tuning, "
        "rarely operator-tuned."
    ),
    "contradiction_overlap": (
        "Memory contradiction token-overlap threshold (0.6); stable default "
        "alongside contradiction_cos, documented in src/spark_mcp/config.py."
    ),
    "qd_reward_type_map": (
        "Hypothesis-type to reward-set wiring covers all six types by "
        "default; remapping is a Phase-5 internals change, not operator "
        "tuning."
    ),
    "hook_tool_calls_threshold": (
        "Tier-2 hook difficulty heuristic (50); only bites once QuantFlow "
        "hook emission lands (plan §15.1 fallback row 1)."
    ),
    "hook_tokens_threshold": (
        "Tier-2 hook difficulty heuristic (200_000 tokens); same "
        "fallback-tier rationale as hook_tool_calls_threshold."
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


def _research_coverage():
    """Return (present_fields, text) for the research section of the example."""
    raw = yaml.safe_load(EXAMPLE_PATH.read_text(encoding="utf-8")) or {}
    research = raw.get("research", {}) or {}
    return set(research), EXAMPLE_PATH.read_text(encoding="utf-8")


def _is_commented_out(field, text):
    return re.search(rf"^\s*#\s*{re.escape(field)}\s*:", text, re.M) is not None


def test_research_fields_covered_by_example():
    present, text = _research_coverage()
    uncovered = [
        field
        for field in ResearchConfig.model_fields
        if field not in present
        and not _is_commented_out(field, text)
        and field not in INTENTIONAL_OMISSIONS
    ]
    assert not uncovered, (
        "ResearchConfig fields with no config.example.yaml entry "
        "(fix the example, comment the field out, or allowlist it in "
        "INTENTIONAL_OMISSIONS with a reason):\n" + "\n".join(uncovered)
    )


def test_intentional_omissions_are_live_and_justified():
    present, text = _research_coverage()
    for field, reason in INTENTIONAL_OMISSIONS.items():
        assert reason.strip(), f"{field}: omission needs a reason"
        assert field in ResearchConfig.model_fields, (
            f"{field}: allowlisted but no such ResearchConfig field "
            "(stale allowlist entry)"
        )
        assert field not in present and not _is_commented_out(field, text), (
            f"{field}: allowlisted omission now has an example entry; "
            "remove the allowlist entry"
        )
