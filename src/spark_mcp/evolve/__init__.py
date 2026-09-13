"""Population-based procedure evolution, native and minimal (plan §11).

The loop and metric discipline are mined from dspy's GEPA (NOTICE §22), but
implemented natively per the independence contract (no dspy import, ever).
An individual is a candidate procedure variant::

    {"name": str, "procedure_steps": [str, ...], "references": [str, ...]}

Each generation: score every individual against the replay fixtures, take the
pareto front over (held_in_score, simplicity = -len(procedure_steps)), keep
the top-k parents, and refresh the population with deterministic mutation
operators. The whole run is driven by one seeded `random.Random`, so a run is
bit-reproducible given (seeds, fixtures, caps, seed).

Deliberately honest and small: deterministic rubric coverage is the only
scorer here. An LLM judge inside the loop would multiply calls by
population x generations x fixtures (§21 cost-blowup row); the LLM judge
lives in skills.run_replay at promotion time instead.
"""

from .gepa import fixture_score, pareto_front, run_gepa

__all__ = ["fixture_score", "pareto_front", "run_gepa"]
