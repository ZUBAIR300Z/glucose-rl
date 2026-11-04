"""
Run the full generalization study end-to-end with ONE command.

For each training seed: train a pooled multi-patient agent, benchmark its
validation-selected model on the held-out TEST patients, then aggregate every
seed into a single mean +/- std table with a paired agent-vs-PID significance
test. Optionally also benchmark the median-action ENSEMBLE of all seeds.
Runs SEQUENTIALLY (each training run uses all your cores; running them in
parallel would thrash the CPU and be slower).

Run from the project root (Anaconda Prompt; long -- hours):
    python scripts/run_study.py --seeds 0 1 2 --timesteps 500000 --n-envs 6
    python scripts/run_study.py --env-version v2 --algo tqc --tag v2-tqc --ensemble --shield-eval
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

from diabetes_rl.algos import ALGOS
from diabetes_rl.cohorts import train_val_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
STUDY_DIR = os.path.join(ROOT, "results", "study")
POOLED = os.path.join(ROOT, "models", "pooled")
LABEL = {"random": "Random", "pid": "PID", "basal": "Basal only", "bb": "BB (meals)",
         "agent": "Agent", "ensemble": "Ensemble"}


def run(cmd):
    print("\n>>> " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def summarize(df: pd.DataFrame, title: str):
    print(f"\n{'=' * 84}\n{title}\n{'=' * 84}")
    print(f"{'Controller':<13}{'TIR 70-180':>13}{'TBR <70':>12}{'TBR <54':>9}{'TAR >180':>12}{'GRI':>6}{'surv':>7}")
    for c in [c for c in LABEL if c in set(df.controller)]:
        d = df[df.controller == c]
        sev = d.time_severe_hypo_pct.mean() if "time_severe_hypo_pct" in d else float("nan")
        gri = d.gri.mean() if "gri" in d else float("nan")
        print(f"{LABEL[c]:<13}"
              f"{d.time_in_range_pct.mean():6.1f}+/-{d.time_in_range_pct.std():<4.1f}"
              f"{d.time_hypo_pct.mean():7.1f}+/-{d.time_hypo_pct.std():<3.1f}{sev:8.1f}"
              f"{d.time_hyper_pct.mean():7.1f}+/-{d.time_hyper_pct.std():<3.1f}{gri:6.0f}"
              f"{100 * d.survived.mean():6.0f}%")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--algo", default="sac", choices=ALGOS)
    p.add_argument("--env-version", default="v1", choices=["v1", "v2"])
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--n-envs", type=int, default=6)
    p.add_argument("--holdout-per-cohort", type=int, default=2)
    p.add_argument("--val-per-cohort", type=int, default=1)
    p.add_argument("--reward", default="magni")
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--shield-train", action="store_true", help="v2: train with the safety shield in the loop")
    p.add_argument("--shield-eval", action="store_true", help="evaluate every controller behind the safety shield")
    p.add_argument("--with-bb", action="store_true", help="add the meal-announced basal-bolus reference")
    p.add_argument("--ensemble", action="store_true", help="also benchmark the median-action ensemble of all seeds")
    p.add_argument("--tag", default="default",
                   help="experiment name; models + results are namespaced by it")
    p.add_argument("--eval-freq", type=int, default=25_000)
    p.add_argument("--benchmark-seeds", type=int, default=5)
    p.add_argument("--skip-train", action="store_true")
    args = p.parse_args()

    study_dir = os.path.join(STUDY_DIR, args.tag)
    pooled_dir = os.path.join(POOLED, args.tag)
    os.makedirs(study_dir, exist_ok=True)
    _, _, test_patients = train_val_test_split(args.holdout_per_cohort, args.val_per_cohort, seed=0)
    print(f"[tag={args.tag}] algo={args.algo} env={args.env_version} reward={args.reward}")
    print(f"Held-out test patients ({len(test_patients)}): {test_patients}")
    bench_extra = (["--shield"] if args.shield_eval else []) + (["--with-bb"] if args.with_bb else [])

    frames, best_models = [], []
    for s in args.seeds:
        if not args.skip_train:
            cmd = [PY, os.path.join(ROOT, "scripts", "train_pooled.py"),
                   "--algo", args.algo, "--env-version", args.env_version,
                   "--timesteps", args.timesteps, "--n-envs", args.n_envs,
                   "--holdout-per-cohort", args.holdout_per_cohort,
                   "--val-per-cohort", args.val_per_cohort, "--reward", args.reward,
                   "--tag", args.tag, "--eval-freq", args.eval_freq, "--seed", s]
            if args.learning_rate is not None:
                cmd += ["--learning-rate", args.learning_rate]
            if args.shield_train:
                cmd += ["--shield"]
            run(cmd)

        best = os.path.join(pooled_dir, f"best_seed{s}", "best_model")
        if not os.path.exists(best + ".zip"):
            print(f"!! no model for seed {s} at {best}.zip -- skipping")
            continue
        best_models.append(best)

        dst = os.path.join(study_dir, f"seed{s}.csv")
        run([PY, os.path.join(ROOT, "scripts", "benchmark.py"),
             "--model", best, "--patients", *test_patients,
             "--seeds", args.benchmark_seeds, "--out", dst, *bench_extra])
        df = pd.read_csv(dst)
        df["train_seed"] = s
        frames.append(df)

    if not frames:
        sys.exit("No results to aggregate (no trained models found).")

    allrows = pd.concat(frames, ignore_index=True)
    allrows["controller"] = allrows["controller"].replace({"sac": "agent"})  # legacy CSVs
    allrows.to_csv(os.path.join(study_dir, "all_runs.csv"), index=False)

    n_seeds = allrows["train_seed"].nunique()
    summarize(allrows, f"GENERALIZATION STUDY -- HELD-OUT patients ({n_seeds} train seeds x "
                       f"{args.benchmark_seeds} eval seeds){'  [shielded]' if args.shield_eval else ''}")
    print("\nPer training seed (agent):")
    for s, d in allrows[allrows.controller == "agent"].groupby("train_seed"):
        print(f"  seed {s}: TIR {d.time_in_range_pct.mean():5.1f}%  TBR {d.time_hypo_pct.mean():5.1f}%  "
              f"survived {100 * d.survived.mean():3.0f}%")

    # paired Wilcoxon agent vs PID on TIR (matched by patient + eval seed + train seed)
    try:
        from scipy.stats import wilcoxon
        key = ["patient", "seed", "train_seed"]
        ag = allrows[allrows.controller == "agent"].set_index(key).time_in_range_pct
        pid = allrows[allrows.controller == "pid"].set_index(key).time_in_range_pct
        common = ag.index.intersection(pid.index)
        diffs = (ag.loc[common] - pid.loc[common]).to_numpy()
        if len(diffs) >= 6 and np.any(diffs != 0):
            _, pval = wilcoxon(diffs)
            tag = ("Agent > PID" if np.median(diffs) > 0 else "PID > Agent") if pval < 0.05 else "n.s."
            print(f"\nPaired Wilcoxon Agent vs PID (TIR, n={len(diffs)}): "
                  f"median {np.median(diffs):+.1f} pts, p={pval:.3f} ({tag})")
    except Exception as e:
        print(f"\n(significance test skipped: {e})")

    if args.ensemble and len(best_models) > 1:
        dst = os.path.join(study_dir, "ensemble.csv")
        run([PY, os.path.join(ROOT, "scripts", "benchmark.py"),
             "--model", *best_models, "--patients", *test_patients,
             "--seeds", args.benchmark_seeds, "--out", dst, *bench_extra])
        summarize(pd.read_csv(dst), f"ENSEMBLE of {len(best_models)} seeds (median action) -- HELD-OUT patients")

    print(f"\nSaved: {os.path.join(study_dir, 'all_runs.csv')}")


if __name__ == "__main__":
    main()
