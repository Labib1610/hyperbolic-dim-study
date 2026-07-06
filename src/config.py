"""Configuration loading, seeding, and device selection utilities."""

import random

import numpy as np
import torch
import yaml


def load_config(path: str = "configs/default.yaml") -> dict:
    """Load a YAML config file into a plain dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Seed random, numpy, and torch (CPU + CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return a CUDA device if available (and preferred), else CPU."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
