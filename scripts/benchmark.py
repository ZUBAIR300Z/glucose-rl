"""
Rigorous benchmark: baselines vs trained RL agent(s), across multiple virtual
patients and seeds. This is the script that produces the portfolio numbers.

Controllers:
    random     uniform random pump rate (sanity floor)
    pid        correction-only PID on CGM (same information as the agent)
    basal      the patient's prescribed basal rate, open loop
    bb         basal-bolus with ANNOUNCED meals (--with-bb; extra information,
               so a reference point -- "standard of care" -- not a fair rival)
    agent      the trained policy (--model PATH)
    ensemble   median action of several policies (--model A B C ...)

Every controller faces the exact same simulated day for a given (patient, seed).
``--shield`` puts the same predictive-low-glucose-suspend shield in front of
every controller.

The agent env is rebuilt from the ``config.json`` saved by train_pooled.py next
to the model; models without one (trained before it existed) are treated as
the original v1/SAC setup.

Outputs:
  * a results table (mean +/- std) printed to the console,
  * results/benchmark.csv  (one row per run, for your own analysis),
  * results/benchmark_tir.png  (Time-in-Range bar chart for the README).

Run from the project root:
    python scripts/benchmark.py                         # random + PID + basal + agent
    python scripts/benchmark.py --no-agent --with-bb    # baselines only
    python scripts/benchmark.py --model models/pooled/v2-tqc/best_seed{0,1,2}/best_model --shield
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

from diabetes_rl.algos import RECURRENT, PolicyRunner, load_model
from diabetes_rl.baselines import BasalBolusController, BasalController, BasalPIDController, PIDController
from diabetes_rl.config import env_kwargs, find_run_config
from diabetes_rl.envs import make_env, STEPS_PER_DAY
from diabetes_rl.metrics import CONSENSUS_TARGETS, glycemic_metrics
from diabetes_rl.safety import ShieldWrapper
from diabetes_rl.wrappers import make_glucose_env

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

DEFAULT_PATIENTS = ["adolescent#001", "adult#001", "child#001"]
LABELS = {"random": "Random", "pid": "PID", "bpid": "Basal PID", "basal": "Basal only",
          "bb": "BB (meals)", "agent": "Agent", "ensemble": "Ensemble"}
COLORS = {"random": "#9e9e9e", "pid": "#1f77b4", "bpid": "#17becf", "basal": "#8c564b",
          "bb": "#2ca02c", "agent": "#d62728", "ensemble": "#9467bd"}


def env_id(kind, patient, seed):  # registration-safe env ids (unique per seed)
    return f"simglucose/bench-{kind}-{patient.replace('#', '')}-s{seed}-v0"


def roll_out(env, policy_fn, seed):
    """Run one episode; return (bg_trajectory, survived_full_day, n_shield_suspensions)."""
    obs, info = env.reset(seed=seed)
    bg, survived, n_susp = [], True, 0
    for _ in range(STEPS_PER_DAY):
        action = policy_fn(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        bg.append(float(info["bg"]))
        n_susp += int(bool(info.get("shield_suspended", False)))
        if terminated:        # patient crashed (BG out of safe range)
            survived = False
            break
        if truncated:         # reached end of the day -> success
            break
    return bg, survived, n_susp


def baseline_env(patient, seed, shield):
    env = make_env(env_id=env_id("raw", patient, seed), patient_name=patient, env_seed=seed)
    return ShieldWrapper(env) if shield else env


def agent_env(cfg, patient, seed, shield, history_override):
    kw = env_kwargs(cfg)
    kw["history_len"] = history_override or kw["history_len"]
    kw["shield"] = shield or kw["shield"]
    return make_glucose_env(env_id=env_id(f"agent-{cfg['env_version']}", patient, seed),
                            patient_name=patient, env_seed=seed, **kw)


def aggregate(rows, controller, key):
    vals = [r[key] for r in rows if r["controller"] == controller]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (0.0, 0.0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patients", nargs="+", default=DEFAULT_PATIENTS)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--history", type=int, default=None,
                   help="override the agent's history length (legacy models without config.json)")
    p.add_argument("--model", nargs="+", default=[os.path.join(MODELS_DIR, "sac_glucose")],
                   help="one model -> 'agent'; several -> also a median-action 'ensemble'")
    p.add_argument("--algo", default=None, help="override the algorithm stored in config.json")
    p.add_argument("--no-agent", action="store_true", help="skip the RL agent(s)")
    p.add_argument("--with-bb", action="store_true", help="add the meal-announced basal-bolus reference")
    p.add_argument("--ensemble-only", action="store_true",
                   help="with several --model paths: score only their median-action ensemble "
                        "(skip baselines and the individual models)")
    p.add_argument("--no-baselines", action="store_true",
                   help="score only the agent model(s) (+ their ensemble); e.g. to add a configuration "
                        "to a benchmark whose baselines were already run with --no-agent")
    p.add_argument("--with-basal-pid", action="store_true",
                   help="add the basal-scaled PID (BasalPIDController, tuned gains)")
    p.add_argument("--shield", action="store_true", help="put the PLGS safety shield in front of every controller")
    p.add_argument("--out", default=os.path.join(RESULTS_DIR, "benchmark.csv"))
    args = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    controllers = [] if args.no_baselines else (
        ["random", "pid"] + (["bpid"] if args.with_basal_pid else []) + ["basal"]
        + (["bb"] if args.with_bb else []))
    runners, cfg = [], None
    if not args.no_agent:
        for path in args.model:
            zip_path = path if path.endswith(".zip") else path + ".zip"
            if not os.path.exists(zip_path):
                print(f"(no model at {zip_path} -- skipping it)")
                continue
            c = find_run_config(path)
            if args.algo:
                c["algo"] = args.algo
            key = lambda k: (k["env_version"], k["history"], k["max_multiplier"], k.get("residual", 0.0),
                             k.get("residual_base", "pid"))
            if cfg is not None and key(c) != key(cfg):
                sys.exit("all --model paths must share the same env config to be ensembled")
            cfg = c
            runners.append(PolicyRunner(load_model(c["algo"], path), recurrent=c["algo"] in RECURRENT))
        if runners:
            controllers.append("agent")
            if len(runners) > 1:
                controllers.append("ensemble")
            if args.ensemble_only:
                if len(runners) < 2:
                    sys.exit("--ensemble-only needs at least two --model paths")
                controllers = ["ensemble"]
            print(f"agent: {len(runners)} model(s), algo={cfg['algo']}, env={cfg['env_version']}")
        else:
            print("(no models found -- benchmarking baselines only)\n")

    rows = []
    for patient in args.patients:
        for seed in range(args.seeds):
            for ctrl in controllers:
                # A fresh env per controller, built with the same seed -> the
                # identical meal day for every controller (a fair head-to-head).
                if ctrl in ("agent", "ensemble"):
                    env = agent_env(cfg, patient, seed, args.shield, args.history)
                    if ctrl == "agent":
                        # each model separately, so seed-to-seed variance is visible
                        for k, r in enumerate(runners):
                            r.reset()
                            bg, surv, ns = roll_out(env, lambda o, i: r.act(o), seed)
                            rows.append(_row(bg, surv, ns, ctrl, patient, seed, k, cfg))
                            _print(rows[-1], f"{LABELS[ctrl]} #{k}")
                        env.close()
                        continue
                    for r in runners:
                        r.reset()
                    policy = lambda o, i: np.median(np.stack([r.act(o) for r in runners]), axis=0)
                else:
                    env = baseline_env(patient, seed, args.shield)
                    if ctrl == "random":
                        policy = lambda o, i, e=env: e.action_space.sample()
                    elif ctrl == "pid":
                        pid = PIDController(target=120.0, action_high=float(env.action_space.high[0]))
                        policy = lambda o, i, c=pid: c.act(float(o[0]))
                    elif ctrl == "bpid":
                        policy = lambda o, i, c=BasalPIDController(patient): c.act(float(o[0]))
                    elif ctrl == "basal":
                        policy = lambda o, i, c=BasalController(patient): c.act(float(o[0]))
                    else:  # bb
                        policy = lambda o, i, c=BasalBolusController(patient): c.act(float(o[0]), i)
                bg, surv, ns = roll_out(env, policy, seed)
                env.close()
                rows.append(_row(bg, surv, ns, ctrl, patient, seed, 0, cfg))
                _print(rows[-1], LABELS[ctrl])

    # ---- write CSV ----
    fields = ["controller", "model_idx", "algo", "patient", "seed", "survived", "shield_steps",
              "n_steps", "mean_bg", "min_bg", "max_bg", "time_in_range_pct", "time_hypo_pct",
              "time_hyper_pct", "time_severe_hypo_pct", "time_severe_hyper_pct", "cv_pct",
              "lbgi", "hbgi", "gri"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})

    # ---- summary table ----
    n = len(args.patients) * args.seeds
    print(f"\n{'=' * 92}\nMEAN +/- STD over {n} days ({len(args.patients)} patients x {args.seeds} seeds)"
          f"{'  [shielded]' if args.shield else ''}\n{'=' * 92}")
    print(f"{'Controller':<12}{'TIR 70-180':>14}{'TBR <70':>12}{'TBR <54':>9}{'TAR >180':>12}"
          f"{'CV':>6}{'GRI':>6}{'surv':>7}")
    for ctrl in controllers:
        tir_m, tir_s = aggregate(rows, ctrl, "time_in_range_pct")
        tbr_m, tbr_s = aggregate(rows, ctrl, "time_hypo_pct")
        tar_m, tar_s = aggregate(rows, ctrl, "time_hyper_pct")
        surv = 100.0 * np.mean([r["survived"] for r in rows if r["controller"] == ctrl])
        print(f"{LABELS[ctrl]:<12}{tir_m:6.1f}+/-{tir_s:<4.1f}{tbr_m:7.1f}+/-{tbr_s:<3.1f}"
              f"{aggregate(rows, ctrl, 'time_severe_hypo_pct')[0]:8.1f}"
              f"{tar_m:7.1f}+/-{tar_s:<3.1f}{aggregate(rows, ctrl, 'cv_pct')[0]:6.0f}"
              f"{aggregate(rows, ctrl, 'gri')[0]:6.0f}{surv:6.0f}%")
    print("Consensus targets: " + ", ".join(f"{k.replace('_pct', '')} {op} {v:g}%"
                                             for k, (op, v) in CONSENSUS_TARGETS.items()))

    # ---- paired significance test: agent vs PID on TIR (same patient+seed pairs) ----
    best = "ensemble" if "ensemble" in controllers else "agent"
    if best in controllers:
        try:
            from scipy.stats import wilcoxon
            k = lambda r: (r["patient"], r["seed"])
            ag = {}
            for r in rows:
                if r["controller"] == best:
                    ag.setdefault(k(r), []).append(r["time_in_range_pct"])
            ag = {key: float(np.mean(v)) for key, v in ag.items()}
            pid = {k(r): r["time_in_range_pct"] for r in rows if r["controller"] == "pid"}
            common = sorted(set(ag) & set(pid))
            diffs = np.array([ag[c] - pid[c] for c in common])
            if len(common) >= 6 and np.any(diffs != 0):
                _, pval = wilcoxon(diffs)
                verdict = f"{LABELS[best]} > PID" if np.median(diffs) > 0 else f"PID > {LABELS[best]}"
                print(f"\nPaired Wilcoxon {LABELS[best]} vs PID (TIR, n={len(common)}): "
                      f"median diff {np.median(diffs):+.1f} pts, p={pval:.3f} "
                      f"({verdict if pval < 0.05 else 'not significant'})")
            else:
                print(f"\n(significance test needs >=6 paired, non-tied samples; have {len(common)})")
        except Exception as e:
            print(f"\n(significance test skipped: {e})")

    # ---- bar chart ----
    means = [aggregate(rows, c, "time_in_range_pct")[0] for c in controllers]
    stds = [aggregate(rows, c, "time_in_range_pct")[1] for c in controllers]
    fig, ax = plt.subplots(figsize=(1.6 * len(controllers) + 2, 5))
    bars = ax.bar([LABELS[c] for c in controllers], means, yerr=stds, capsize=6,
                  color=[COLORS[c] for c in controllers], alpha=0.85)
    ax.set_ylabel("Time in Range 70-180 mg/dL (%)")
    ax.set_title(f"Glycemic control: Time-in-Range{' (shielded)' if args.shield else ''}\n"
                 f"({len(args.patients)} patients x {args.seeds} seeds, mean +/- std)")
    ax.set_ylim(0, 100)
    ax.bar_label(bars, fmt="%.0f%%", padding=3)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(args.out)), "benchmark_tir.png")
    fig.savefig(out, dpi=120)

    print(f"\nSaved: {args.out}\nSaved: {out}")


def _row(bg, surv, n_susp, ctrl, patient, seed, model_idx, cfg):
    m = glycemic_metrics(bg)
    m.update(controller=ctrl, patient=patient, seed=seed, survived=int(surv), shield_steps=n_susp,
             model_idx=model_idx, algo=(cfg["algo"] if ctrl in ("agent", "ensemble") else ""))
    return m


def _print(m, label):
    print(f"  {m['patient']:>15} | seed {m['seed']} | {label:<12} "
          f"TIR={m['time_in_range_pct']:5.1f}%  hypo={m['time_hypo_pct']:4.1f}%  "
          f"{'survived' if m['survived'] else 'CRASHED'}")


if __name__ == "__main__":
    main()
