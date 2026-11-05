"""
A rule-based safety shield that can sit in front of ANY controller.

Hypoglycemia is the failure mode that matters most, and it is exactly where the
learned policies in this project are weakest (2 of 3 pooled SAC seeds spend
~30% of the day below 70 mg/dL). Commercial pumps solve the same problem with a
*predictive low-glucose suspend* (PLGS): stop insulin when glucose is low, or is
projected to go low soon. This module implements that rule so it can be applied
identically to the PID, the baselines and the RL agent (fair comparison), at
evaluation time and optionally during training (shielded RL). Use it via
``ShieldWrapper`` or ``make_glucose_env(shield=True)``.

Rule (in the style of PLGS systems such as Tandem Basal-IQ):
    suspend insulin if   CGM < suspend_below
                     or  CGM projected ``horizon_min`` ahead < predicted_below
The projection is a least-squares line through the last ``window`` readings,
which is far less noise-sensitive than a two-point difference.
"""
from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np


class SafetyShield:
    def __init__(self, suspend_below: float = 70.0, predicted_below: float = 80.0,
                 horizon_min: float = 30.0, sample_time: float = 3.0, window: int = 5):
        self.suspend_below = suspend_below
        self.predicted_below = predicted_below
        self.horizon_min = horizon_min
        self.sample_time = sample_time
        self.window = window
        self.reset()

    def reset(self):
        self.cgm = deque(maxlen=self.window)
        self.n_suspended = 0

    def observe(self, cgm: float):
        """Feed every new CGM reading (including the one returned by ``reset``)."""
        self.cgm.append(float(cgm))

    def slope(self) -> float:
        """Least-squares CGM trend in mg/dL per minute over the window (0 if too short)."""
        if len(self.cgm) < 2:
            return 0.0
        y = np.asarray(self.cgm, dtype=float)
        t = np.arange(len(y)) * self.sample_time
        return float(np.polyfit(t, y, 1)[0])

    def predicted(self) -> float:
        return self.cgm[-1] + self.slope() * self.horizon_min if self.cgm else np.inf

    def should_suspend(self) -> bool:
        if not self.cgm:
            return False
        return self.cgm[-1] < self.suspend_below or self.predicted() < self.predicted_below

    def filter(self, insulin):
        """Return ``(safe_insulin, suspended)`` for a proposed dose (any array/scalar)."""
        if self.should_suspend():
            self.n_suspended += 1
            return np.zeros_like(np.asarray(insulin, dtype=np.float32)), True
        return np.asarray(insulin, dtype=np.float32), False


class ShieldWrapper(gym.Wrapper):
    """Apply a SafetyShield directly on the simulator's insulin action (U/min).

    Sits *below* any agent wrapper, so the same shield protects baselines, v1
    agents and v2 agents alike. Adds ``info["insulin_delivered"]`` and
    ``info["shield_suspended"]`` so upper wrappers can track the real dose.
    """

    def __init__(self, env, shield: SafetyShield | None = None):
        super().__init__(env)
        self.shield = shield or SafetyShield()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.shield.reset()
        self.shield.observe(float(np.asarray(obs).ravel()[0]))
        return obs, info

    def step(self, action):
        safe, suspended = self.shield.filter(np.asarray(action, dtype=np.float32).reshape(1))
        obs, reward, terminated, truncated, info = self.env.step(safe)
        self.shield.observe(float(np.asarray(obs).ravel()[0]))
        info = dict(info, insulin_delivered=float(safe[0]), shield_suspended=suspended)
        return obs, reward, terminated, truncated, info
