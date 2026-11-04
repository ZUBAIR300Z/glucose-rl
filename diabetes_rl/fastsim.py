"""
Make simglucose ~10x faster without changing its results.

Profiling shows ~90% of simulation time is not physics: simglucose's ODE
right-hand side (``T1DPatient.model``) reads ~40 patient parameters as
attributes of a pandas Series (``params.kp1``, ``params.Vg``, ...), and every
such lookup goes through pandas' slow ``__getattr__``. The ODE solver calls the
right-hand side many times per simulated minute, so this dominates. Most of
the remainder is numpy overhead from carrying the insulin dose as a
1-element array through the right-hand side.

``enable()`` patches ``T1DPatient.step`` so that, only while the solver
integrates:
  * the patient's parameters are a plain namespace of Python floats
    (the same float64 values -- bit-for-bit identical results), and
  * the insulin rate is a Python float instead of a 1-element float32 array.
    The dose arithmetic then runs in float64 rather than float32; over 12
    full simulated days this moved blood glucose by at most 2e-4 mg/dL
    (CGM noise is ~10 mg/dL) and left every TIR/TBR value unchanged.
Everything else in simglucose still sees the original objects.

Measured: ~31 -> ~350 simulator steps/s per core. Enabled automatically by
``diabetes_rl.envs``; set the environment variable ``DIABETES_RL_FASTSIM=0``
to run the unmodified simulator.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np

_ENABLED = False
ORIGINAL_STEP = None  # simglucose's own T1DPatient.step, kept for tests / opt-out


def _to_namespace(series) -> SimpleNamespace:
    values = {}
    for key, value in series.items():
        try:
            values[key] = float(value)
        except (TypeError, ValueError):
            values[key] = value  # e.g. the patient's Name
    return SimpleNamespace(**values)


def enable() -> bool:
    """Patch simglucose once per process. Returns True if the patch is active."""
    global _ENABLED, ORIGINAL_STEP
    if _ENABLED:
        return True
    if os.environ.get("DIABETES_RL_FASTSIM", "1") == "0":
        return False
    from simglucose.patient.t1dpatient import T1DPatient

    ORIGINAL_STEP = original_step = T1DPatient.step

    def step(self, action):
        insulin = action.insulin
        if isinstance(insulin, np.ndarray) and insulin.size == 1:
            action = action._replace(insulin=float(insulin.reshape(-1)[0]))
        cache = self.__dict__.get("_fast_params")
        if cache is None or cache[0] is not self._params:
            cache = (self._params, _to_namespace(self._params))
            self._fast_params = cache
        series = self._params
        self._params = cache[1]
        try:
            return original_step(self, action)
        finally:
            self._params = series

    T1DPatient.step = step
    _ENABLED = True
    return True
