"""
Tune the patient-scaled BasalPIDController by random search on TRAINING patients.

The test patients are never touched here. Each candidate gain set is scored
on fixed training-patient days by the Glycemia Risk Index (GRI, lower is
better; a day that crashes the patient scores the maximum, 100). The top
candidates are re-scored on extra days to avoid picking a lucky one, and the
winner is checked on the validation patients.

Run from the project root (~30-60 min on one core):
    python scripts/tune_pid.py --candidates 60
Prints the winning gains as BasalPIDController keyword arguments and saves
results/pid_tuning.csv.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from diabetes_rl.baselines import BasalPIDController
from diabetes_rl.cohorts import train_val_test_split
from diabetes_rl.envs import STEPS_PER_DAY, make_env
from diabetes_rl.metrics import glycemic_metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAY_SEED0 = 2000  # tuning days are distinct from validation (1000+) and test (0..4) days


def score_day(patient, seed, gains):
    env = make_env(env_id=f"simglucose/tune-{patient.replace('#', '')}-s{seed}-v0",
                   patient_name=patient, env_seed=seed)
    ctrl = BasalPIDController(patient, **gains)
    obs, info = env.reset()
    bg, crashed = [], False
    for _ in range(STEPS_PER_DAY):
        obs, _, term, trunc, info = env.step(ctrl.act(float(obs[0]), info))
        bg.append(info["bg"])
        if term:
            crashed = True
            break
        if trunc:
            break
    env.close()
    m = glycemic_metrics(bg)
    return (100.0 if crashed else m["gri"]), m, crashed


def evaluate(patients, days, gains):
    rows = [score_day(p, DAY_SEED0 + d, gains) for p in patients for d in days]
    return dict(gri=float(np.mean([r[0] for r in rows])),
                tir=float(np.mean([r[1]["time_in_range_pct"] for r in rows])),
                tbr=float(np.mean([r[1]["time_hypo_pct"] for r in rows])),
                surv=float(np.mean([not r[2] for r in rows])))


def sample(rng):
    return dict(target=float(rng.uniform(100, 160)),
                kp=float(10 ** rng.uniform(-3, -1)),
                ki=float(10 ** rng.uniform(-6, -3)),
                kd=float(rng.uniform(0.0, 3.0)),
                max_multiplier=float(rng.choice([3.0, 5.0, 8.0])))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidates", type=int, default=60)
    p.add_argument("--top", type=int, default=5, help="re-score the best N on extra days")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    train, val, _test = train_val_test_split(2, 1, seed=0)
    rng = np.random.default_rng(args.seed)
    defaults = dict(target=120.0, kp=0.01, ki=1e-4, kd=0.5, max_multiplier=5.0)
    cands = [defaults] + [sample(rng) for _ in range(args.candidates)]

    # Results are saved after every candidate, so an interrupted search resumes
    # where it stopped (the candidate list is deterministic for a given --seed).
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    partial = os.path.join(ROOT, "results", f"pid_tuning_partial_seed{args.seed}.csv")
    done = {}
    if os.path.exists(partial):
        for row in pd.read_csv(partial).to_dict("records"):
            i = int(row["idx"])
            if i < len(cands) and all(abs(row[k] - cands[i][k]) < 1e-9 for k in cands[i]):
                done[i] = row
    results = []
    for i, g in enumerate(cands):
        if i in done:
            results.append(done[i])
            continue
        r = evaluate(train, [0], g)
        results.append({"idx": i, **g, **{f"train1_{k}": v for k, v in r.items()}})
        pd.DataFrame(results).to_csv(partial, index=False)
        print(f"[{i + 1:3d}/{len(cands)}] GRI {r['gri']:5.1f}  TIR {r['tir']:5.1f}  TBR {r['tbr']:4.1f}  "
              f"surv {100 * r['surv']:3.0f}%  {json.dumps({k: round(v, 6) for k, v in g.items()})}", flush=True)
    if done:
        print(f"(reused {len(done)} candidates from {partial})")

    df = pd.DataFrame(results).sort_values("train1_gri")
    finalists = df.head(args.top).copy()
    keys = list(defaults)
    for idx, row in finalists.iterrows():
        g = {k: row[k] for k in keys}
        r = evaluate(train, [1, 2], g)
        for k, v in r.items():
            df.loc[idx, f"train3_{k}"] = (row[f"train1_{k}"] + 2 * v) / 3  # mean over 3 days/patient
    best_idx = df["train3_gri"].idxmin()
    best = {k: float(df.loc[best_idx, k]) for k in keys}
    val_r = evaluate(val, [0, 1, 2], best)

    df.to_csv(os.path.join(ROOT, "results", "pid_tuning.csv"), index=False)
    print("\nBest on training patients (3 days each): "
          f"GRI {df.loc[best_idx, 'train3_gri']:.1f}, TIR {df.loc[best_idx, 'train3_tir']:.1f}%, "
          f"TBR {df.loc[best_idx, 'train3_tbr']:.1f}%")
    print(f"Validation patients: GRI {val_r['gri']:.1f}, TIR {val_r['tir']:.1f}%, TBR {val_r['tbr']:.1f}%, "
          f"survived {100 * val_r['surv']:.0f}%")
    print("BasalPIDController(**" + json.dumps({k: round(v, 6) for k, v in best.items()}) + ")")


if __name__ == "__main__":
    main()
