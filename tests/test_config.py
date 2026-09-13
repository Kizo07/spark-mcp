"""Config loading: defaults, YAML overrides, env key detection."""

from spark_mcp.config import Config


def test_defaults_load():
    cfg = Config()
    assert cfg.research.dsr_alpha == 0.05
    assert cfg.research.kill_rule_failures == 5
    assert cfg.research.novelty_weights == [0.5, 0.3, 0.2]
    assert cfg.peers.alpha_engine.enabled is False


def test_yaml_override(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        "research:\n  dsr_alpha: 0.01\nllm:\n  model: test-model\n",
        encoding="utf-8",
    )
    cfg = Config.from_yaml(p)
    assert cfg.research.dsr_alpha == 0.01
    assert cfg.llm.model == "test-model"
    assert cfg.research.kill_rule_failures == 5  # unset keys keep defaults


def test_missing_file_gives_defaults(tmp_path):
    cfg = Config.from_yaml(tmp_path / "nope.yaml")
    assert cfg.research.min_T == 30


def test_mode_reporting(monkeypatch):
    cfg = Config()
    monkeypatch.delenv(cfg.llm.api_key_env, raising=False)
    assert "degraded" in cfg.active_mode()
    monkeypatch.setenv(cfg.llm.api_key_env, "sk-test")
    assert cfg.active_mode() == "full (llm key present)"
