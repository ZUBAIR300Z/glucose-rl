"""
Rigorous benchmark: Random vs PID vs trained SAC agent, across multiple virtual
patients and seeds. This is the script that produces the portfolio numbers.

Outputs:
  * a results table (mean +/- std) printed to the console,
  * results/benchmark.csv  (one row per run, for your own analysis),
  * results/benchmark_tir.png  (Time-in-Range bar chart for the README).

Run from the project root:
    python scripts/benchmark.py                         # random + PID + SAC
    python scripts/benchmark.py --no-agent              # before you've trained
    python scripts/benchmark.py --patients adolescent#001 adult#001 child#001 --seeds 5
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diabetes_rl.envs import make_env, STEPS_PER_DAY
from diabetes_rl.baselines import PIDController
from diabetes_rl.metrics import glycemic_metrics
from diabetes_rl.wrappers import GlucoseTrendWrapper

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

DEFAULT_PATIENTS = ["adolescent#001", "adult#001", "child#001"]
LABELS = {"random": "Random", "pid": "PID", "sac": "SAC agent"}


def _pid(name, seed):  # registration-safe env ids (unique per seed)
    return f"simglucose/bench-{name.replace('#','')}-s{seed}-v0"


def _agent(name, seed):
    return f"simglucose/bench-{name.replace('#','')}-agent-s{seed}-v0"


def roll_out(env, policy_fn, n_steps, seed):
    """Run one episode; return (bg_trajectory, survived_full_day)."""
    obs, info = env.reset(seed=seed)
    bg, survived = [], True
    for _ in range(n_steps):
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        bg.append(float(info.get("bg", obs[0] if np.ndim(obs) else obs)))
        if terminated:        # patient crashed (BG out of safe range)
            survived = False
            break
        if truncated:         # reached end of the day -> success
            break
    return bg, survived


def aggregate(rows, controller, key):
    vals = [r[key] for r in rows if r["controller"] == controller]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (0.0, 0.0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--patients", nargs="+", default=DEFAULT_PATIENTS)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--history", type=int, default=4, help="must match training")
    p.add_argument("--model", default=os.path.join(MODELS_DIR, "sac_glucose"))
    p.add_argument("--no-agent", action="store_true", help="skip the SAC agent")
    args = p.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    controllers = ["random", "pid"]
    model = None
    if not args.no_agent:
        model_path = args.model if args.model.endswith(".zip") else args.model + ".zip"
        if os.path.exists(model_path):
            from stable_baselines3 import SAC
            model = SAC.load(args.model)
            controllers.append("sac")
        else:
            print(f"(no model at {model_path} -- benchmarking random + PID only)\n")

    rows = []
    for patient in args.patients:
        for seed in range(args.seeds):
            # Fresh env per seed -> a genuinely different meal scenario.
            # Same seed for the raw and agent envs -> identical conditions,
            # so PID vs agent is a fair head-to-head on the very same day.
            raw_env = make_env(env_id=_pid(patient, seed), patient_name=patient, env_seed=seed)
            pid = PIDController(target=120.0, action_high=float(raw_env.action_space.high[0]))

            for ctrl in controllers:
                if ctrl == "random":
                    bg, surv = roll_out(raw_env, lambda o: raw_env.action_space.sample(), STEPS_PER_DAY, seed)
                elif ctrl == "pid":
                    pid.reset()
                    bg, surv = roll_out(raw_env, lambda o: pid.act(float(o[0])), STEPS_PER_DAY, seed)
                else:
                    agent_env = GlucoseTrendWrapper(
                        make_env(env_id=_agent(patient, seed), patient_name=patient, env_seed=seed),
                        history_len=args.history,
                    )
                    bg, surv = roll_out(agent_env, lambda o: model.predict(o, deterministic=True)[0], STEPS_PER_DAY, seed)
                    agent_env.close()

                m = glycemic_metrics(bg)
                m.update(controller=ctrl, patient=patient, seed=seed, survived=int(surv))
                rows.append(m)
                print(f"  {patient:>15} | seed {seed} | {LABELS[ctrl]:<10} "
                      f"TIR={m['time_in_range_pct']:5.1f}%  hypo={m['time_hypo_pct']:4.1f}%  "
                      f"{'survived' if surv else 'CRASHED'}")

            raw_env.close()

    # ---- write CSV ----
    csv_path = os.path.join(RESULTS_DIR, "benchmark.csv")
    fields = ["controller", "patient", "seed", "survived", "n_steps", "mean_bg",
              "min_bg", "max_bg", "time_in_range_pct", "time_hypo_pct", "time_hyper_pct"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})

    # ---- summary table ----
    n = len(args.patients) * args.seeds
    print(f"\n{'='*78}\nMEAN +/- STD over {n} runs "
          f"({len(args.patients)} patients x {args.seeds} seeds)\n{'='*78}")
    print(f"{'Controller':<12}{'TIR 70-180':>14}{'TBR <70':>12}{'TAR >180':>12}"
          f"{'mean BG':>9}{'surv':>7}")
    for ctrl in controllers:
        tir_m, tir_s = aggregate(rows, ctrl, "time_in_range_pct")
        tbr_m, tbr_s = aggregate(rows, ctrl, "time_hypo_pct")
        tar_m, tar_s = aggregate(rows, ctrl, "time_hyper_pct")
        bg_m, _ = aggregate(rows, ctrl, "mean_bg")
        surv = 100.0 * np.mean([r["survived"] for r in rows if r["controller"] == ctrl])
        print(f"{LABELS[ctrl]:<12}{tir_m:6.1f}±{tir_s:<4.1f}{tbr_m:7.1f}±{tbr_s:<3.1f}"
              f"{tar_m:7.1f}±{tar_s:<3.1f}{bg_m:9.0f}{surv:6.0f}%")

    # ---- paired significance test: SAC vs PID on TIR (same patient+seed pairs) ----
    if "sac" in controllers:
        try:
            from scipy.stats import wilcoxon
            k = lambda r: (r["patient"], r["seed"])
            sac = {k(r): r["time_in_range_pct"] for r in rows if r["controller"] == "sac"}
            pid = {k(r): r["time_in_range_pct"] for r in rows if r["controller"] == "pid"}
            common = sorted(set(sac) & set(pid))
            diffs = np.array([sac[c] - pid[c] for c in common])
            if len(common) >= 6 and np.any(diffs != 0):
                _, pval = wilcoxon(diffs)
                verdict = "SAC > PID" if np.median(diffs) > 0 else "PID > SAC"
                print(f"\nPaired Wilcoxon SAC vs PID (TIR, n={len(common)}): "
                      f"median diff {np.median(diffs):+.1f} pts, p={pval:.3f} "
                      f"({verdict if pval < 0.05 else 'not significant'})")
            else:
                print(f"\n(significance test needs >=6 paired, non-tied samples; have {len(common)})")
        except Exception as e:
            print(f"\n(significance test skipped: {e})")

    # ---- bar chart ----
    means = [aggregate(rows, c, "time_in_range_pct")[0] for c in controllers]
    stds = [aggregate(rows, c, "time_in_range_pct")[1] for c in controllers]
    colors = {"random": "#9e9e9e", "pid": "#1f77b4", "sac": "#d62728"}
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar([LABELS[c] for c in controllers], means, yerr=stds, capsize=6,
                  color=[colors[c] for c in controllers], alpha=0.85)
    ax.set_ylabel("Time in Range 70-180 mg/dL (%)")
    ax.set_title(f"Glycemic control: Time-in-Range\n({len(args.patients)} patients x {args.seeds} seeds, mean +/- std)")
    ax.set_ylim(0, 100)
    ax.bar_label(bars, fmt="%.0f%%", padding=3)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "benchmark_tir.png")
    fig.savefig(out, dpi=120)

    print(f"\nSaved: {csv_path}\nSaved: {out}")


if __name__ == "__main__":
    main()
