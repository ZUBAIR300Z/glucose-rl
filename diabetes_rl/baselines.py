"""
Baseline (non-learning) controllers to benchmark the RL agent against.
"""
from __future__ import annotations

import numpy as np


class PIDController:
    """A simple PID controller on glucose error (CGM - target).

    Positive error (glucose too high) -> deliver insulin. Output is clipped to
    [0, action_high], so this is a *correction-only* controller: it never tries
    to "remove" glucose, which makes it a safe, conservative baseline that
    rarely causes insulin-induced lows.
    """

    def __init__(self, target=120.0, kp=3e-4, ki=1e-6, kd=1e-4, action_high=None):
        self.target = target
        self.kp, self.ki, self.kd = kp, ki, kd
        self.action_high = action_high
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def act(self, cgm: float) -> np.ndarray:
        error = cgm - self.target
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error

        u = self.kp * error + self.ki * self.integral + self.kd * derivative
        high = self.action_high if self.action_high is not None else np.inf
        u = float(np.clip(u, 0.0, high))
        return np.array([u], dtype=np.float32)
