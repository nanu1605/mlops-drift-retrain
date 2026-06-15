"""Centralized determinism. Every place randomness occurs calls ``set_seed``."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> int:
    """Seed Python, numpy, and hash randomization. Returns the seed for logging."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed


def rng(seed: int) -> np.random.Generator:
    """A dedicated numpy Generator — preferred over global state for data gen."""
    return np.random.default_rng(seed)
