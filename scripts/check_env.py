"""
Sanity check: run the simglucose environment with RANDOM insulin actions and
save a plot. This verifies the whole stack works end-to-end before we touch RL.

Random dosing is a deliberately terrible policy -- expect poor time-in-range
and some dangerous lows/highs. That's the point: it is the performance *floor*
that any real controller must beat, and running it proves the plumbing works.

Run from the project root:
    python scripts/check_env.py
"""
from __future__ import annotations

import os
import sys

# Make the `diabetes_rl` package importable when run as a plain script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")  # headless: render straight to a file, no GUI window
import matplotlib.pyplot as plt
import numpy as np

from diabetes_rl.envs import make_env, STEPS_PER_DAY
from diabetes_rl.metrics import glycemic_metrics, format_metrics, TARGET_LOW, TARGET_HIGH

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def run_episode(env, n_steps, seed=0):
    """Step the env with random actions; return per-step trajectories."""
    obs, info = env.reset(seed=seed)
    t0 = info.get("time")

    hours, cgm, bg, insulin, meals = [], [], [], [], []
    for step in range(n_steps):
        action = env.action_space.sample()  # random basal insulin
        obs, reward, terminated, truncated, info = env.step(action)

        t = info.get("time")
        minutes = (t - t0).total_seconds() / 60.0 if (t0 and t) else step * 3
        hours.append(minutes / 60.0)
        cgm.append(float(obs[0]))
        bg.append(float(info.get("bg", obs[0])))
        insulin.append(float(np.asarray(action).ravel()[0]))
        meals.append(float(info.get("meal", 0.0)))

        if terminated or truncated:
            why = "terminated (BG out of safe range)" if terminated else "truncated (day ended)"
            print(f"Episode ended at step {step}: {why}.")
            break

    return hours, cgm, bg, insulin, meals


def plot_trajectory(hours, cgm, bg, insulin, meals, metrics, out_path):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax1.axhspan(TARGET_LOW, TARGET_HIGH, color="green", alpha=0.12, label="target 70-180")
    ax1.axhline(TARGET_LOW, color="green", lw=0.8, ls="--")
    ax1.axhline(TARGET_HIGH, color="green", lw=0.8, ls="--")
    ax1.plot(hours, bg, color="C3", lw=1.3, label="blood glucose (true)")
    ax1.plot(hours, cgm, color="C0", lw=0.9, alpha=0.6, label="CGM (sensor)")
    ax1.set_ylabel("glucose (mg/dL)")
    ax1.set_title(
        f"simglucose sanity check - random insulin policy "
        f"(Time-in-Range {metrics['time_in_range_pct']:.0f}%)"
    )
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_ylim(0, max(400.0, (max(bg) * 1.05) if bg else 400.0))

    # meals as orange bars on a twin axis
    meal_h = [h for h, m in zip(hours, meals) if m > 0]
    meal_v = [m for m in meals if m > 0]
    if meal_h:
        axm = ax1.twinx()
        axm.bar(meal_h, meal_v, width=0.15, color="orange", alpha=0.5)
        axm.set_ylabel("carbs (g)", color="orange")
        axm.set_ylim(0, max(meal_v) * 4)

    ax2.plot(hours, insulin, color="C4", lw=1.0)
    ax2.set_ylabel("insulin\n(basal)")
    ax2.set_xlabel("time (hours)")
    ax2.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)


def main(n_steps: int = STEPS_PER_DAY, seed: int = 0):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Creating simglucose environment (patient adolescent#001)...")
    env = make_env()
    print(f"  observation_space = {env.observation_space}")
    print(f"  action_space      = {env.action_space}")

    print(f"\nRunning {n_steps} random steps (~{n_steps * 3 / 60:.0f} simulated hours)...")
    hours, cgm, bg, insulin, meals = run_episode(env, n_steps, seed=seed)
    env.close()

    metrics = glycemic_metrics(bg)
    print("\nRandom-policy results (this is the floor any controller must beat):")
    print(format_metrics(metrics))

    out = os.path.join(RESULTS_DIR, "check_env_random_policy.png")
    plot_trajectory(hours, cgm, bg, insulin, meals, metrics, out)
    print(f"\nSaved plot -> {out}")
    print("\nStack works end-to-end. Next: a PID baseline, then an RL agent.")


if __name__ == "__main__":
    main()
