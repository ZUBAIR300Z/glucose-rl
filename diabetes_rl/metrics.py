"""
Glycemic metrics for evaluating a control policy.

The headline number for this project is **Time in Range (TIR)**: the percentage
of time blood glucose stays within 70-180 mg/dL. It is the standard clinical
outcome measure, and "my agent beats the baseline on TIR without increasing
time spent hypo" is exactly the result you want to be able to show.
"""
from __future__ import annotations

import numpy as np

TARGET_LOW = 70
TARGET_HIGH = 180
SEVERE_LOW = 54
SEVERE_HIGH = 250

# International consensus CGM targets (Battelino et al., Diabetes Care 2019),
# for adults with T1D. Used to flag whether a controller is clinically acceptable.
CONSENSUS_TARGETS = {
    "time_in_range_pct": (">", 70.0),
    "time_hypo_pct": ("<", 4.0),
    "time_severe_hypo_pct": ("<", 1.0),
    "time_hyper_pct": ("<", 25.0),
    "time_severe_hyper_pct": ("<", 5.0),
    "cv_pct": ("<=", 36.0),
}


def _kovatchev_risk(bg):
    """Kovatchev symmetrized BG risk -> (LBGI, HBGI)."""
    f = 1.509 * (np.log(np.maximum(bg, 1.0)) ** 1.084 - 5.381)
    r = 10.0 * f * f
    return float(np.mean(np.where(f < 0, r, 0.0))), float(np.mean(np.where(f > 0, r, 0.0)))


def glycemic_metrics(bg_series) -> dict:
    """Compute summary glycemic metrics from a sequence of BG values."""
    bg = np.asarray(bg_series, dtype=float)
    n = len(bg)
    if n == 0:
        return {}
    lbgi, hbgi = _kovatchev_risk(bg)
    pct = lambda mask: float(np.mean(mask) * 100)
    vlow, low = pct(bg < SEVERE_LOW), pct((bg >= SEVERE_LOW) & (bg < TARGET_LOW))
    vhigh, high = pct(bg > SEVERE_HIGH), pct((bg > TARGET_HIGH) & (bg <= SEVERE_HIGH))
    return {
        "n_steps": int(n),
        "mean_bg": float(np.mean(bg)),
        "min_bg": float(np.min(bg)),
        "max_bg": float(np.max(bg)),
        "time_in_range_pct": pct((bg >= TARGET_LOW) & (bg <= TARGET_HIGH)),
        "time_hypo_pct": pct(bg < TARGET_LOW),
        "time_hyper_pct": pct(bg > TARGET_HIGH),
        "time_severe_hypo_pct": vlow,
        "time_severe_hyper_pct": vhigh,
        "cv_pct": float(np.std(bg) / np.mean(bg) * 100),
        "lbgi": lbgi,
        "hbgi": hbgi,
        # Glycemia Risk Index (Klonoff et al., 2023): 0 (best) .. 100 (worst).
        "gri": float(min(3.0 * vlow + 2.4 * low + 1.6 * vhigh + 0.8 * high, 100.0)),
    }


def format_metrics(m: dict) -> str:
    """Pretty multi-line string for printing to the console."""
    if not m:
        return "  (no data)"
    return (
        f"  steps={m['n_steps']}  mean BG={m['mean_bg']:.1f} mg/dL "
        f"(min {m['min_bg']:.0f}, max {m['max_bg']:.0f})\n"
        f"  Time in range (70-180): {m['time_in_range_pct']:.1f}%\n"
        f"  Time hypo  (<70):       {m['time_hypo_pct']:.1f}%\n"
        f"  Time hyper (>180):      {m['time_hyper_pct']:.1f}%"
    )
