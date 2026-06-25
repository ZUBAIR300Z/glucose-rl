"""
Visualize the multi-seed generalization study (results/study/seed*.csv).

Produces docs/study_results.png with three panels:
  1. Controller comparison (TIR vs hypoglycemia) on held-out patients
  2. SAC training-seed variance (the headline finding)
  3. Per held-out patient Time-in-Range (PID vs SAC)
"""
from __future__ import annotations

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDY = os.path.join(ROOT, "results", "study")
DOCS = os.path.join(ROOT, "docs")

CTRLS = ["random", "pid", "sac"]
LBL = {"random": "Random", "pid": "PID", "sac": "SAC"}


def load():
    files = sorted(glob.glob(os.path.join(STUDY, "seed*.csv")))
    if not files:
        sys.exit("No results/study/seed*.csv found — run scripts/run_study.py first.")
    frames = [pd.read_csv(f).assign(train_seed=int(os.path.basename(f)[4:-4])) for f in files]
    return pd.concat(frames, ignore_index=True)


def main():
    df = load()
    os.makedirs(DOCS, exist_ok=True)
    n_seeds = df.train_seed.nunique()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Panel 1: controller comparison (TIR vs TBR) ---
    ax = axes[0]
    x = np.arange(len(CTRLS)); w = 0.38
    tir = [df[df.controller == c].time_in_range_pct.mean() for c in CTRLS]
    tir_s = [df[df.controller == c].time_in_range_pct.std() for c in CTRLS]
    tbr = [df[df.controller == c].time_hypo_pct.mean() for c in CTRLS]
    tbr_s = [df[df.controller == c].time_hypo_pct.std() for c in CTRLS]
    ax.bar(x - w / 2, tir, w, yerr=tir_s, capsize=4, color="#2e7d32", label="Time in Range (70-180)")
    ax.bar(x + w / 2, tbr, w, yerr=tbr_s, capsize=4, color="#c62828", label="Time below range (<70)")
    ax.set_xticks(x); ax.set_xticklabels([LBL[c] for c in CTRLS])
    ax.set_ylabel("% of day"); ax.set_ylim(0, 80)
    ax.set_title(f"Held-out patients: range vs safety\n({n_seeds} seeds x 6 patients x 5 eval seeds, mean+/-std)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.2)

    # --- Panel 2: SAC training-seed variance (the headline) ---
    ax = axes[1]
    seeds = sorted(df.train_seed.unique())
    sac = df[df.controller == "sac"]
    stir = [sac[sac.train_seed == s].time_in_range_pct.mean() for s in seeds]
    stbr = [sac[sac.train_seed == s].time_hypo_pct.mean() for s in seeds]
    x = np.arange(len(seeds)); w = 0.38
    ax.bar(x - w / 2, stir, w, color="#2e7d32", label="SAC TIR")
    ax.bar(x + w / 2, stbr, w, color="#c62828", label="SAC hypo")
    pid_tir = df[df.controller == "pid"].time_in_range_pct.mean()
    pid_tbr = df[df.controller == "pid"].time_hypo_pct.mean()
    ax.axhline(pid_tir, ls="--", c="#1f77b4", lw=1.5, label=f"PID TIR ({pid_tir:.0f}%)")
    ax.axhline(pid_tbr, ls="--", c="#ff9800", lw=1.5, label=f"PID hypo ({pid_tbr:.0f}%)")
    ax.set_xticks(x); ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("% of day")
    ax.set_title("SAC training-seed variance\n(seed 0 good; others unsafe = unreliable)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.2)

    # --- Panel 3: per held-out patient TIR (PID vs SAC) ---
    ax = axes[2]
    patients = sorted(df.patient.unique())
    short = [p.replace("adolescent", "ado").replace("adult", "adu").replace("child", "chi") for p in patients]
    pid_p = [df[(df.controller == "pid") & (df.patient == p)].time_in_range_pct.mean() for p in patients]
    sac_p = [df[(df.controller == "sac") & (df.patient == p)].time_in_range_pct.mean() for p in patients]
    x = np.arange(len(patients)); w = 0.38
    ax.bar(x - w / 2, pid_p, w, color="#1f77b4", label="PID")
    ax.bar(x + w / 2, sac_p, w, color="#d62828", label="SAC (avg over seeds)")
    ax.set_xticks(x); ax.set_xticklabels(short, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Time in Range (%)")
    ax.set_title("Per held-out patient: Time in Range")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    out = os.path.join(DOCS, "study_results.png")
    fig.savefig(out, dpi=120)
    print(f"saved {out}  ({n_seeds} seeds, {df.patient.nunique()} held-out patients)")


if __name__ == "__main__":
    main()
