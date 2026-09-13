"""LLM/embedder protocols and fakes: determinism, model_id keys."""

import numpy as np

from spark_mcp.llm import FakeEmbedder, FakeLLM


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
