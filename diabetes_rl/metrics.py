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


def glycemic_metrics(bg_series) -> dict:
    """Compute summary glycemic metrics from a sequence of BG values."""
    bg = np.asarray(bg_series, dtype=float)
    n = len(bg)
    if n == 0:
        return {}
    return {
        "n_steps": int(n),
        "mean_bg": float(np.mean(bg)),
        "min_bg": float(np.min(bg)),
        "max_bg": float(np.max(bg)),
        "time_in_range_pct": float(np.mean((bg >= TARGET_LOW) & (bg <= TARGET_HIGH)) * 100),
        "time_hypo_pct": float(np.mean(bg < TARGET_LOW) * 100),
        "time_hyper_pct": float(np.mean(bg > TARGET_HIGH) * 100),
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
