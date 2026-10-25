"""OpenAICompatibleEmbedder: HTTP embeddings + build_embedder gating."""

import httpx
import numpy as np
import pytest

from spark_mcp.config import Config
from spark_mcp.llm import OpenAICompatibleEmbedder, build_embedder


def _handler_factory(dim=3, seen=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen["url"] = str(request.url)
            seen["auth"] = request.headers.get("authorization")
            import json

            seen["body"] = json.loads(request.content)
        # echo deterministic vectors based on input length order
        import json

        body = json.loads(request.content)
        inputs = body["input"]
        data = [
            {"index": i, "embedding": [float(i + 1)] * dim, "object": "embedding"}
            for i in range(len(inputs))
        ]
        return httpx.Response(200, json={"data": data, "model": body["model"]})

    return handler


def test_embed_success_shape_and_ordering():
    transport = httpx.MockTransport(_handler_factory(dim=4))
    emb = OpenAICompatibleEmbedder(
        "https://embed.test/v1", "sk-test", "embed-model", transport=transport
    )
    out = emb.embed(["hello", "world", "again"])
    assert isinstance(out, np.ndarray)
    assert out.shape == (3, 4)
    assert out.dtype == np.float64
    # index ordering preserved
    assert out[0, 0] == 1.0
    assert out[1, 0] == 2.0
    assert out[2, 0] == 3.0
    assert emb.model_id() == "embed-model"


def test_embed_auth_header_and_payload():
    seen = {}
    transport = httpx.MockTransport(_handler_factory(seen=seen))
    api_key = "testkey123"
    emb = OpenAICompatibleEmbedder(
        "https://embed.test/v1/", api_key, "my-embedder", transport=transport
    )
    emb.embed(["a", "b"])
    assert seen["url"] == "https://embed.test/v1/embeddings"
    assert seen["auth"] == f"Bearer {api_key}"
    assert seen["body"]["model"] == "my-embedder"
    assert seen["body"]["input"] == ["a", "b"]


def test_embed_sorts_by_index():
    def handler(request: httpx.Request) -> httpx.Response:
        # return out-of-order indices
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [2.0, 2.0]},
                    {"index": 0, "embedding": [1.0, 1.0]},
                ]
            },
        )

    emb = OpenAICompatibleEmbedder(
        "https://e.test/v1", "k", "m", transport=httpx.MockTransport(handler)
    )
    out = emb.embed(["x", "y"])
    assert out.shape == (2, 2)
    assert list(out[0]) == [1.0, 1.0]
    assert list(out[1]) == [2.0, 2.0]


def test_embed_retries_transient_then_succeeds():
    calls = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= 2:
            return httpx.Response(503, json={"error": "busy"})
        return httpx.Response(
            200, json={"data": [{"index": 0, "embedding": [0.5, 0.5]}]}
        )

    emb = OpenAICompatibleEmbedder(
        "https://e.test/v1", "k", "m", transport=httpx.MockTransport(flaky)
    )
    out = emb.embed(["hi"])
    assert out.shape == (1, 2)
    assert calls["n"] == 3


def test_embed_raises_after_retries_exhausted():
    def always_down(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "down"})

    emb = OpenAICompatibleEmbedder(
        "https://e.test/v1",
        "k",
        "m",
        transport=httpx.MockTransport(always_down),
        max_retries=1,
    )
    with pytest.raises(httpx.HTTPStatusError):
        emb.embed(["hi"])


def test_embed_via_client_replacement():
    # mirrors test_llm.py _mocked_llm pattern: replace _client after construction
    emb = OpenAICompatibleEmbedder("https://e.test/v1", "k", "m")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]}
        )

    emb._client = httpx.Client(
        transport=httpx.MockTransport(handler), timeout=5.0
    )
    out = emb.embed(["x"])
    assert out.shape == (1, 2)


# --- build_embedder gating ---


def test_build_embedder_no_endpoint_returns_none(monkeypatch):
    monkeypatch.setenv("SPARK_EMB_KEY", "sk-x")
    # no embeddings_base_url at all
    cfg = Config(llm={"api_key_env": "SPARK_EMB_KEY"})
    assert build_embedder(cfg) is None
    # chat base_url alone must NOT enable embeddings (z.ai lesson)
    cfg = Config(
        llm={"api_key_env": "SPARK_EMB_KEY", "base_url": "https://chat.test/v1"}
    )
    assert build_embedder(cfg) is None


def test_build_embedder_endpoint_without_key_returns_none(monkeypatch):
    monkeypatch.delenv("SPARK_EMB_MISSING", raising=False)
    cfg = Config(
        llm={
            "api_key_env": "SPARK_EMB_MISSING",
            "embeddings_base_url": "https://embed.test/v1",
        }
    )
    assert build_embedder(cfg) is None


def test_build_embedder_endpoint_and_key_returns_embedder(monkeypatch):
    monkeypatch.setenv("SPARK_EMB_KEY2", "sk-live")
    cfg = Config(
        llm={
            "api_key_env": "SPARK_EMB_KEY2",
            "embeddings_base_url": "https://embed.test/v1",
            "embedder_model": "text-embedding-3",
            "base_url": "https://chat.test/v1",
        }
    )
    emb = build_embedder(cfg)
    assert isinstance(emb, OpenAICompatibleEmbedder)
    assert emb.model_id() == "text-embedding-3"


def test_build_embedder_never_llm_returns_none(monkeypatch):
    monkeypatch.setenv("SPARK_EMB_KEY3", "sk-live")
    cfg = Config(
        llm={
            "api_key_env": "SPARK_EMB_KEY3",
            "embeddings_base_url": "https://embed.test/v1",
            "degraded_mode": "never-llm",
        }
    )
    assert build_embedder(cfg) is None
