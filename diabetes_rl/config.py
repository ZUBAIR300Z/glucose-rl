"""
Run configuration: what train_pooled.py saves next to a model as config.json,
and how evaluation scripts rebuild the exact same environment from it.

Models trained before config.json existed are treated as the original setup
(``LEGACY_CONFIG``: v1 env, SAC, Magni reward).
"""
from __future__ import annotations

import json
import os

from diabetes_rl.rewards import REWARD_FUNCTIONS

LEGACY_CONFIG = dict(algo="sac", env_version="v1", reward="magni", history=None,
                     terminal_penalty=None, max_multiplier=5.0, gamma=0.99, shield=False,
                     residual=0.0)


def env_kwargs(cfg: dict) -> dict:
    """``make_glucose_env`` keyword arguments for a run config."""
    return dict(env_version=cfg["env_version"], history_len=cfg["history"],
                reward_fun=REWARD_FUNCTIONS[cfg["reward"]], terminal_penalty=cfg["terminal_penalty"],
                max_multiplier=cfg["max_multiplier"], gamma=cfg["gamma"], shield=cfg["shield"],
                residual=cfg.get("residual", 0.0) or 0.0)


def find_run_config(model_path: str) -> dict:
    """config.json next to the model or up to two directories above it."""
    d = os.path.dirname(os.path.abspath(model_path))
    for _ in range(3):
        f = os.path.join(d, "config.json")
        if os.path.exists(f):
            with open(f) as fh:
                return {**LEGACY_CONFIG, **json.load(fh)}
        d = os.path.dirname(d)
    return dict(LEGACY_CONFIG)
