"""
Baseline (non-learning) controllers to benchmark the RL agent against.

All controllers expose ``reset()`` and ``act(cgm, info=None) -> np.ndarray`` with
the dose in U/min (the raw simglucose action unit).

Information each one uses (keep this in mind when comparing):
    PIDController         CGM only (same as the RL agent)
    BasalController       the patient's prescribed basal rate (open loop)
    BasalBolusController  prescribed basal + CR/CF + **announced meals**: it gets
                          meal carbs the RL agent never sees, so it is an upper
                          reference ("standard of care with perfect carb
                          counting"), not a like-for-like competitor.
"""
from __future__ import annotations

import numpy as np

from diabetes_rl.patients import therapy_params


class PIDController:
    """A simple PID controller on glucose error (CGM - target).

    Positive error (glucose too high) -> deliver insulin. Output is clipped to
    [0, action_high], so this is a *correction-only* controller: it never tries
    to "remove" glucose, which makes it a safe, conservative baseline that
    rarely causes insulin-induced lows.

    ``integral_limit`` (optional) clamps the integral term (anti-windup). Without
    it the integral keeps growing through a long post-meal high and keeps dosing
    after glucose has started to fall, which is a classic cause of late lows.
    The default (None) reproduces the original, un-clamped controller.
    """

    def __init__(self, target=120.0, kp=3e-4, ki=1e-6, kd=1e-4, action_high=None,
                 integral_limit=None):
        self.target = target
        self.kp, self.ki, self.kd = kp, ki, kd
        self.action_high = action_high
        self.integral_limit = integral_limit
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def act(self, cgm: float, info=None) -> np.ndarray:
        error = cgm - self.target
        self.integral += error
        if self.integral_limit is not None:
            self.integral = float(np.clip(self.integral, -self.integral_limit, self.integral_limit))
        derivative = error - self.prev_error
        self.prev_error = error

        u = self.kp * error + self.ki * self.integral + self.kd * derivative
        high = self.action_high if self.action_high is not None else np.inf
        u = float(np.clip(u, 0.0, high))
        return np.array([u], dtype=np.float32)


class BasalController:
    """Deliver the patient's prescribed basal rate, ignoring CGM (open loop).

    The simplest clinically meaningful baseline: what the pump does with no
    closed-loop algorithm at all. Any learned controller should beat it.
    """

    def __init__(self, patient_name: str):
        self.basal = therapy_params(patient_name)["basal"]

    def reset(self):
        pass

    def act(self, cgm: float, info=None) -> np.ndarray:
        return np.array([self.basal], dtype=np.float32)


class BasalBolusController:
    """Prescribed basal + meal bolus from carb ratio + correction (simglucose's BB logic).

    Needs ``info["meal"]`` (g/min of carbs this step) and ``info["sample_time"]``
    -- i.e. *announced meals*. See the module docstring for why this is a
    reference point rather than a fair competitor to a CGM-only agent.
    """

    def __init__(self, patient_name: str, target: float = 140.0, correction_above: float = 150.0):
        p = therapy_params(patient_name)
        self.basal, self.cr, self.cf = p["basal"], p["cr"], p["cf"]
        self.target, self.correction_above = target, correction_above

    def reset(self):
        pass

    def act(self, cgm: float, info=None) -> np.ndarray:
        info = info or {}
        meal = float(info.get("meal", 0.0))            # g/min over the last step
        st = float(info.get("sample_time", 3.0))       # minutes per step
        bolus = 0.0
        if meal > 0:
            bolus = meal * st / self.cr                 # carbs -> units
            if cgm > self.correction_above:
                bolus += (cgm - self.target) / self.cf  # correction units
        return np.array([self.basal + bolus / st], dtype=np.float32)  # U -> U/min
