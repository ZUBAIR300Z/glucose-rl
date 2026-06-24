"""
Run the full generalization study end-to-end with ONE command.

For each training seed: train a pooled multi-patient SAC, benchmark its held-out
model, then aggregate every seed into a single mean +/- std table with a paired
SAC-vs-PID significance test. Runs SEQUENTIALLY (each training run uses all your
cores; running them in parallel would thrash the CPU and be slower).

Run from the project root (Anaconda Prompt; long — hours):
    python scripts/run_study.py --seeds 0 1 2 --timesteps 500000 --n-envs 6
    python scripts/run_study.py --skip-train      # only benchmark + aggregate existing models
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from diabetes_rl.cohorts import stratified_holdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
STUDY_DIR = os.path.join(ROOT, "results", "study")
POOLED = os.path.join(ROOT, "models", "pooled")
LABEL = {"random": "Random", "pid": "PID", "sac": "SAC (pooled)"}


def run(cmd):
    print("\n>>> " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--n-envs", type=int, default=6)
    p.add_argument("--holdout-per-cohort", type=int, default=2)
    p.add_argument("--eval-freq", type=int, default=25_000)
    p.add_argument("--benchmark-seeds", type=int, default=5)
    p.add_argument("--skip-train", action="store_true")
    args = p.parse_args()

    os.makedirs(STUDY_DIR, exist_ok=True)
    _, test_patients = stratified_holdout(args.holdout_per_cohort, seed=0)
    print(f"Held-out test patients ({len(test_patients)}): {test_patients}")

    frames = []
    for s in args.seeds:
        if not args.skip_train:
            run([PY, os.path.join(ROOT, "scripts", "train_pooled.py"),
                 "--timesteps", args.timesteps, "--n-envs", args.n_envs,
                 "--holdout-per-cohort", args.holdout_per_cohort,
                 "--eval-freq", args.eval_freq, "--seed", s])

        best = os.path.join(POOLED, f"best_seed{s}", "best_model")
        if not os.path.exists(best + ".zip"):
            print(f"!! no model for seed {s} at {best}.zip -- skipping")
            continue

        run([PY, os.path.join(ROOT, "scripts", "benchmark.py"),
             "--model", best, "--patients", *test_patients,
             "--seeds", args.benchmark_seeds])

        dst = os.path.join(STUDY_DIR, f"seed{s}.csv")
        shutil.copyfile(os.path.join(ROOT, "results", "benchmark.csv"), dst)
        df = pd.read_csv(dst)
        df["train_seed"] = s
        frames.append(df)

    if not frames:
        sys.exit("No results to aggregate (no trained models found).")

    allrows = pd.concat(frames, ignore_index=True)
    allrows.to_csv(os.path.join(STUDY_DIR, "all_runs.csv"), index=False)

    n_seeds = allrows["train_seed"].nunique()
    print(f"\n{'='*72}\nGENERALIZATION STUDY -- HELD-OUT patients "
          f"({n_seeds} train seeds x {args.benchmark_seeds} eval seeds)\n{'='*72}")
    print(f"{'Controller':<13}{'TIR 70-180':>13}{'TBR <70':>12}{'TAR >180':>12}{'surv':>8}")
    for c in [c for c in ("random", "pid", "sac") if c in set(allrows.controller)]:
        d = allrows[allrows.controller == c]
        print(f"{LABEL[c]:<13}"
              f"{d.time_in_range_pct.mean():6.1f}+/-{d.time_in_range_pct.std():<4.1f}"
              f"{d.time_hypo_pct.mean():7.1f}+/-{d.time_hypo_pct.std():<3.1f}"
              f"{d.time_hyper_pct.mean():7.1f}+/-{d.time_hyper_pct.std():<3.1f}"
              f"{100 * d.survived.mean():6.0f}%")

    # paired Wilcoxon SAC vs PID on TIR (matched by patient + eval seed + train seed)
    try:
        from scipy.stats import wilcoxon
        key = ["patient", "seed", "train_seed"]
        sac = allrows[allrows.controller == "sac"].set_index(key).time_in_range_pct
        pid = allrows[allrows.controller == "pid"].set_index(key).time_in_range_pct
        common = sac.index.intersection(pid.index)
        diffs = (sac.loc[common] - pid.loc[common]).to_numpy()
        if len(diffs) >= 6 and np.any(diffs != 0):
            _, pval = wilcoxon(diffs)
            tag = ("SAC > PID" if np.median(diffs) > 0 else "PID > SAC") if pval < 0.05 else "n.s."
            print(f"\nPaired Wilcoxon SAC vs PID (TIR, n={len(diffs)}): "
                  f"median {np.median(diffs):+.1f} pts, p={pval:.3f} ({tag})")
    except Exception as e:
        print(f"\n(significance test skipped: {e})")

    print(f"\nSaved: {os.path.join(STUDY_DIR, 'all_runs.csv')}")


if __name__ == "__main__":
    main()
