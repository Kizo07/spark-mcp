"""LLM and embedder protocols (plan §4.3 / M4).

All model interaction goes through these two protocols. Implementations:
OpenAICompatibleLLM/Embedder (Phase 5 wiring; key required), fakes for tests.
`model_id()` must be versioned — it enters every embedding/score cache key so
a model change can never silently re-scale novelty or cached scores.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, prompt: str, *, json_schema: dict | None = None) -> str: ...
    def model_id(self) -> str: ...


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...
    def model_id(self) -> str: ...


class FakeLLM:
    """Deterministic scripted LLM for tests and degraded-mode development."""

    def __init__(self, replies: dict[str, str] | None = None, model: str = "fake-llm/0.1"):
        self._replies = replies or {}
        self._model = model
        self.calls: list[str] = []

    def complete(self, prompt: str, *, json_schema: dict | None = None) -> str:
        self.calls.append(prompt)
        for key, reply in self._replies.items():
            if key in prompt:
                return reply
        return "fake-llm: no scripted reply matched"

    def model_id(self) -> str:
        return self._model


class FakeEmbedder:
    """Deterministic embedding producer: seeded RNG projected to unit sphere.

    Same text + same model id -> same vector (stability tests rely on this).
    """

    def __init__(self, dim: int = 64, model: str = "fake-embedder/0.1"):
        self._dim = dim
        self._model = model

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.empty((len(texts), self._dim), dtype=np.float64)
        for i, text in enumerate(texts):
            seed = abs(hash((text, self._model))) % (2**32)
            rng = np.random.default_rng(seed)
            v = rng.normal(size=self._dim)
            out[i] = v / np.linalg.norm(v)
        return out

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def model_id(self) -> str:
        return self._model
