"""LLM and embedder protocols (plan §4.3 / M4).

All model interaction goes through these two protocols. Implementations:
OpenAICompatibleLLM/Embedder (Phase 5 wiring; key required), fakes for tests.
`model_id()` must be versioned — it enters every embedding/score cache key so
a model change can never silently re-scale novelty or cached scores.
"""

from __future__ import annotations

import hashlib
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
            # sha256, never builtin hash(): embeddings are persisted, and
            # hash() is salted per process (would break cross-restart N).
            digest = hashlib.sha256(f"{text}\x00{self._model}".encode()).digest()
            seed = int.from_bytes(digest[:8], "little")
            rng = np.random.default_rng(seed)
            v = rng.normal(size=self._dim)
            out[i] = v / np.linalg.norm(v)
        return out

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def model_id(self) -> str:
        return self._model


def split_model_id(model_id: str) -> tuple[str, str]:
    """Canonical (model, model_version) split. Accepts 'provider/model@snap'
    and 'name/version' (version = last path segment when no '@')."""
    if "@" in model_id:
        model, _, version = model_id.partition("@")
        return model, version
    if "/" in model_id:
        model, _, version = model_id.rpartition("/")
        return model, version or model_id
    return model_id, ""


class OpenAICompatibleLLM:
    """Chat-completions client for any OpenAI-compatible endpoint (plan §4.3/M4).

    httpx is already a project dependency (fastmcp stack). temperature=0 for
    score determinism; transient transport/5xx/429 errors retry with linear
    backoff, anything else raises — callers (e.g. explore._llm_scores) treat a
    raise as "fall back to the deterministic scorer" for that candidate.
    `model_id()` is the configured model string; it enters every score-cache
    key so a model swap can never silently re-scale cached scores.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout: float = 90.0,
        max_retries: int = 2,
    ):
        import httpx

        self._base = base_url.rstrip("/")
        self._key = api_key
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries
        self._client = httpx.Client(timeout=timeout)

    def model_id(self) -> str:
        return self._model

    def complete(self, prompt: str, *, json_schema: dict | None = None) -> str:
        import time

        import httpx

        payload: dict = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        if json_schema is not None:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self._key}"}
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = self._client.post(
                    f"{self._base}/chat/completions", json=payload, headers=headers
                )
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(
                        f"transient {resp.status_code}",
                        request=resp.request,
                        response=resp,
                    )
                resp.raise_for_status()
                data = resp.json()
                return str(data["choices"][0]["message"]["content"])
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    time.sleep(0.5 * (attempt + 1))
        assert last_exc is not None
        raise last_exc


class OpenAICompatibleEmbedder:
    """Embeddings client for any OpenAI-compatible endpoint (plan §4.3/M4).

    POST {base}/embeddings with Bearer auth; transient transport/5xx/429
    errors retry with linear backoff like OpenAICompatibleLLM, anything else
    raises. `model_id()` is the configured embedder model string; it enters
    every embedding cache key so a model swap can never silently re-scale
    cached vectors. `transport`/`client` are injectable for tests
    (httpx.MockTransport).
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout: float = 90.0,
        max_retries: int = 2,
        transport=None,
        client=None,
    ):
        import httpx

        self._base = base_url.rstrip("/")
        self._key = api_key
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries
        if client is not None:
            self._client = client
        elif transport is not None:
            self._client = httpx.Client(timeout=timeout, transport=transport)
        else:
            self._client = httpx.Client(timeout=timeout)

    def model_id(self) -> str:
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        import time

        import httpx

        if not texts:
            return np.empty((0, 0), dtype=np.float64)
        payload: dict = {"model": self._model, "input": list(texts)}
        headers = {"Authorization": f"Bearer {self._key}"}
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = self._client.post(
                    f"{self._base}/embeddings", json=payload, headers=headers
                )
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(
                        f"transient {resp.status_code}",
                        request=resp.request,
                        response=resp,
                    )
                resp.raise_for_status()
                data = resp.json()
                items = sorted(data["data"], key=lambda x: x["index"])
                vectors = [item["embedding"] for item in items]
                return np.array(vectors, dtype=np.float64)
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    time.sleep(0.5 * (attempt + 1))
        assert last_exc is not None
        raise last_exc

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


def build_embedder(cfg) -> Embedder | None:
    """Embedder for the given config; None when degraded (plan §4.3/M4).

    The z.ai coding-plan endpoint exposes no embeddings route (chat models
    only), so a chat `base_url` + key alone does not produce an embedder:
    a distinct `embeddings_base_url` is required. FTS5-only retrieval and
    lexical novelty remain the degraded path until an embedding endpoint
    (or local sentence-transformers extra) is configured.
    """
    from .config import Config

    assert isinstance(cfg, Config)
    if cfg.llm.degraded_mode == "never-llm":
        return None
    if not cfg.llm.embeddings_base_url:
        return None
    if not cfg.llm_api_key():
        return None
    return OpenAICompatibleEmbedder(
        cfg.llm.embeddings_base_url, cfg.llm_api_key(), cfg.llm.embedder_model
    )


def build_llm(cfg) -> LLMClient | None:
    """LLM client for the given config; None when degraded (deterministic
    scoring path, §8.1). Wired as soon as base_url + key exist — the
    OpenAI-compatible contract is stable across providers."""
    from .config import Config

    assert isinstance(cfg, Config)
    if cfg.llm.degraded_mode == "never-llm":
        return None
    if not cfg.llm_api_key():
        return None
    if not cfg.llm.base_url:
        return None  # no endpoint to talk to: degrade, don't guess
    return OpenAICompatibleLLM(cfg.llm.base_url, cfg.llm_api_key(), cfg.llm.model)
