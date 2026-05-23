"""Tests for configuration loading."""

from src.utils.config import load_config


def test_load_config_reads_yaml_defaults() -> None:
    cfg = load_config()
    assert cfg.batch_size == 32
    assert cfg.max_seq_length == 128
    assert cfg.seed == 42
    assert cfg.model_name == "ProsusAI/finbert"
