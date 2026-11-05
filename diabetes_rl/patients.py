"""
Per-patient clinical parameters for the 30 simglucose virtual patients.

Real insulin pumps are configured per person: every pump user has a prescribed
basal rate, a carb ratio (CR) and a correction factor (CF). These are known
*before* any controller runs, so a controller may legitimately use them.

They matter a lot here: the steady-state basal rate varies ~6x across the
population (child#003 ~0.005 U/min vs adult#006 ~0.029 U/min). A single fixed
action range in absolute units therefore means very different things for
different patients -- the same dose that is gentle for an adult is a large
overdose for a child.
"""
from __future__ import annotations

from functools import lru_cache
from importlib import resources

import pandas as pd


@lru_cache(maxsize=1)
def _tables():
    params_dir = resources.files("simglucose") / "params"
    with resources.as_file(params_dir / "vpatient_params.csv") as f:
        vp = pd.read_csv(f).set_index("Name")
    with resources.as_file(params_dir / "Quest.csv") as f:
        quest = pd.read_csv(f).set_index("Name")
    return vp, quest


def basal_rate(patient_name: str) -> float:
    """Steady-state basal insulin rate in U/min (simglucose: u2ss * BW / 6000)."""
    vp, _ = _tables()
    return float(vp.loc[patient_name, "u2ss"] * vp.loc[patient_name, "BW"] / 6000.0)


def therapy_params(patient_name: str) -> dict:
    """Prescribed therapy settings: basal (U/min), CR (g/U), CF (mg/dL per U), TDI (U/day)."""
    _, quest = _tables()
    return {
        "basal": basal_rate(patient_name),
        "cr": float(quest.loc[patient_name, "CR"]),
        "cf": float(quest.loc[patient_name, "CF"]),
        "tdi": float(quest.loc[patient_name, "TDI"]),
    }
