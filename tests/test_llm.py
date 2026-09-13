"""LLM/embedder protocols and fakes: determinism, model_id keys."""

import httpx
import numpy as np
import pytest

from spark_mcp.config import Config
from spark_mcp.llm import FakeEmbedder, FakeLLM, OpenAICompatibleLLM, build_llm


def test_fake_llm_scripted_and_records_calls():
    llm = FakeLLM(replies={"score": "7/10"})
    assert llm.complete("please score this") == "7/10"
    assert llm.complete("unmatched prompt") != "7/10"
    assert len(llm.calls) == 2
    assert llm.model_id() == "fake-llm/0.1"


def test_fake_embedder_deterministic_unit_norm():
    emb = FakeEmbedder(dim=32)
    a = emb.embed(["cross-sectional momentum"])
    b = emb.embed(["cross-sectional momentum"])
    assert np.allclose(a, b)
    assert abs(np.linalg.norm(a[0]) - 1.0) < 1e-9


def test_fake_embedder_model_id_in_key_semantics():
    e1 = FakeEmbedder(model="fake-embedder/0.1")
    e2 = FakeEmbedder(model="fake-embedder/0.2")
    v1 = e1.embed_one("same text")
    v2 = e2.embed_one("same text")
    # different model ids must be treated as different embedding spaces
    assert not np.allclose(v1, v2)


def _mocked_llm(handler, model: str = "test-model"):
    llm = OpenAICompatibleLLM("https://example.test/v1", "sk-test", model)
    llm._client = httpx.Client(transport=httpx.MockTransport(handler), timeout=5.0)
    return llm


def test_openai_compatible_llm_posts_and_returns_content():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        body = httpx.Response(200, json={"choices": [{"message": {"content": "pong"}}]})
        return body

    llm = _mocked_llm(handler)
    assert llm.complete("ping") == "pong"
    assert seen["url"] == "https://example.test/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-test"
    assert llm.model_id() == "test-model"


def test_openai_compatible_llm_retries_transient_then_raises():
    calls = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= 2:
            return httpx.Response(503, json={"error": "busy"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    llm = _mocked_llm(flaky)
    assert llm.complete("ping") == "ok"
    assert calls["n"] == 3

    def always_down(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "down"})

    llm2 = _mocked_llm(always_down, model="m2")
    with pytest.raises(httpx.HTTPStatusError):
        llm2.complete("ping")


def test_build_llm_wiring(monkeypatch):
    # degraded modes stay deterministic
    cfg = Config(llm={"degraded_mode": "never-llm", "base_url": "https://x.test/v1"})
    assert build_llm(cfg) is None
    cfg = Config(llm={"api_key_env": "SPARK_TEST_MISSING_KEY", "base_url": "https://x.test/v1"})
    monkeypatch.delenv("SPARK_TEST_MISSING_KEY", raising=False)
    assert build_llm(cfg) is None
    # key but no endpoint: degrade rather than guess
    monkeypatch.setenv("SPARK_TEST_KEY", "sk-x")
    cfg = Config(llm={"api_key_env": "SPARK_TEST_KEY"})
    assert build_llm(cfg) is None
    # key + endpoint -> live client
    cfg = Config(
        llm={
            "api_key_env": "SPARK_TEST_KEY",
            "base_url": "https://x.test/v1",
            "model": "some-model/3",
        }
    )
    llm = build_llm(cfg)
    assert isinstance(llm, OpenAICompatibleLLM)
    assert llm.model_id() == "some-model/3"
