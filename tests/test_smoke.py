"""Minimal CPU-safe tests for the config utilities (no GPU required)."""

from src import config


def test_load_config():
    cfg = config.load_config()
    assert "seed" in cfg
    assert "paths" in cfg
    assert "data_dir" in cfg["paths"]
    assert "output_dir" in cfg["paths"]


def test_set_seed_and_device():
    config.set_seed(0)
    device = config.get_device()
    assert device.type in ("cuda", "cpu")
