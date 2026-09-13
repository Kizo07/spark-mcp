"""Shared fixtures for Phase-2 tests: temp db + deterministic embedder + seeds."""

import pytest

from spark_mcp.config import Config
from spark_mcp.graph import import_seeds
from spark_mcp.llm import FakeEmbedder
from spark_mcp.storage import open_db


@pytest.fixture
def cfg(tmp_path):
    return Config(
        storage={"db_path": f"{tmp_path}/spark.db", "artifacts_dir": f"{tmp_path}/art"}
    )


@pytest.fixture
def db(cfg):
    handle = open_db(cfg.db_path())
    yield handle
    handle.close()


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def seeded(db, embedder):
    """A graph populated with the full seed corpus + embeddings."""
    import_seeds(db, embedder)
    return db
