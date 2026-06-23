"""
PID baseline controller for glucose control.

This is the benchmark your RL agent must beat. It is a classic feedback
controller that sees exactly what the RL agent will see -- the current CGM
reading -- so the comparison is fair.

Design choice: this is a *correction-only* controller. It delivers insulin
when glucose is above target and nothing when at/below target. That makes it a
safe, conservative baseline (it will rarely cause insulin-induced lows), which
is a sensible bar to clear. The gains are intentionally modest; tuning them is
left as an exercise -- and "the RL agent beat a tuned PID" is a stronger story.

Run from the project root:
    python scripts/pid_baseline.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diabetes_rl.envs import make_env, STEPS_PER_DAY
from diabetes_rl.baselines import PIDController
from diabetes_rl.metrics import glycemic_metrics, format_metrics, TARGET_LOW, TARGET_HIGH

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def run_episode(env, controller, n_steps, seed=0):
    obs, info = env.reset(seed=seed)
    controller.reset()
    t0 = info.get("time")

    hours, cgm, bg, insulin, meals = [], [], [], [], []
    for step in range(n_steps):
        action = controller.act(float(obs[0]))
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
        f"PID baseline (Time-in-Range {metrics['time_in_range_pct']:.0f}%, "
        f"hypo {metrics['time_hypo_pct']:.0f}%)"
    )
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_ylim(0, max(400.0, (max(bg) * 1.05) if bg else 400.0))

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
    action_high = float(env.action_space.high[0])
    print(f"  action range = [0, {action_high:.4f}]")

    controller = PIDController(target=120.0, action_high=action_high)
    print(f"\nRunning PID baseline for {n_steps} steps "
          f"(~{n_steps * 3 / 60:.0f} simulated hours)...")
    hours, cgm, bg, insulin, meals = run_episode(env, controller, n_steps, seed=seed)
    env.close()

    metrics = glycemic_metrics(bg)
    print("\nPID baseline results:")
    print(format_metrics(metrics))

    out = os.path.join(RESULTS_DIR, "pid_baseline.png")
    plot_trajectory(hours, cgm, bg, insulin, meals, metrics, out)
    print(f"\nSaved plot -> {out}")
    print("\nThis is the number to beat with RL. Tune the PID gains to make it harder.")


if __name__ == "__main__":
    main()
