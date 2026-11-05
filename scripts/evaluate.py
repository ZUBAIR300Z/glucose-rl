"""
Evaluate a trained RL agent (any supported algorithm) against the PID baseline.

Runs several episodes of each controller, reports mean glycemic metrics
(Time-in-Range etc.), and saves a head-to-head glucose plot. Each episode is a
different simulated day (a freshly seeded env -- simglucose's reset() alone
would replay the same day), and PID and agent face the identical day per
episode -- a fair statistical comparison rather than a single cherry-picked
day. Increase --episodes for tighter estimates.

Run from the project root (after training):
    python scripts/evaluate.py
    python scripts/evaluate.py --episodes 10 --history 4
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diabetes_rl.algos import RECURRENT, PolicyRunner, load_model
from diabetes_rl.envs import make_env, STEPS_PER_DAY
from diabetes_rl.baselines import PIDController
from diabetes_rl.config import env_kwargs, find_run_config
from diabetes_rl.metrics import glycemic_metrics, format_metrics, TARGET_LOW, TARGET_HIGH
from diabetes_rl.wrappers import make_glucose_env

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def _pid(patient, seed):
    return f"simglucose/{patient.replace('#', '')}-eval-pid-s{seed}-v0"


def _agent(patient, seed):
    return f"simglucose/{patient.replace('#', '')}-eval-agent-s{seed}-v0"


# NOTE: simglucose's reset() replays the same meal day, so each episode gets a
# freshly built env with its own seed (= a different day). PID and agent use the
# same seed per episode -> they face the identical day.
def make_agent_env(patient, cfg, seed, history=None):
    """The agent's env, rebuilt from the run's config.json (v1/SAC if absent)."""
    kw = env_kwargs(cfg)
    kw["history_len"] = history or kw["history_len"]
    return make_glucose_env(_agent(patient, seed), patient_name=patient, env_seed=seed, **kw)


def make_pid_env(patient, seed):
    return make_env(env_id=_pid(patient, seed), patient_name=patient, env_seed=seed,
                    max_episode_steps=STEPS_PER_DAY)


def roll_out(env, policy_fn, n_steps, seed):
    """Run one episode; policy_fn(obs, info) -> action. Returns trajectories."""
    obs, info = env.reset(seed=seed)
    t0 = info.get("time")
    hours, bg, insulin = [], [], []
    for step in range(n_steps):
        action = policy_fn(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        t = info.get("time")
        minutes = (t - t0).total_seconds() / 60.0 if (t0 and t) else step * 3
        hours.append(minutes / 60.0)
        bg.append(float(info.get("bg", obs[0] if np.ndim(obs) else obs)))
        # v2 agents act in basal-relative units; plot the delivered U/min.
        insulin.append(float(info.get("insulin", np.asarray(action).ravel()[0])))
        if terminated or truncated:
            break
    return {"hours": hours, "bg": bg, "insulin": insulin}


def evaluate(env_fn, policy_fn, episodes, n_steps, on_reset=None):
    """``env_fn(seed)`` builds the env for one episode (one distinct day per seed)."""
    runs, mets = [], []
    for ep in range(episodes):
        if on_reset is not None:
            on_reset()  # controller state (PID integral, recurrent memory)
        env = env_fn(ep)
        run = roll_out(env, policy_fn, n_steps, seed=ep)
        env.close()
        runs.append(run)
        mets.append(glycemic_metrics(run["bg"]))
    # average each metric across episodes
    keys = mets[0].keys()
    mean = {k: float(np.mean([m[k] for m in mets])) for k in keys}
    std = {k: float(np.std([m[k] for m in mets])) for k in keys}
    return runs, mean, std


def comparison_plot(agent_run, pid_run, agent_mean, pid_mean, out_path):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax1.axhspan(TARGET_LOW, TARGET_HIGH, color="green", alpha=0.12, label="target 70-180")
    ax1.axhline(TARGET_LOW, color="green", lw=0.8, ls="--")
    ax1.axhline(TARGET_HIGH, color="green", lw=0.8, ls="--")
    ax1.plot(pid_run["hours"], pid_run["bg"], color="C0", lw=1.4,
             label=f"PID  (TIR {pid_mean['time_in_range_pct']:.0f}%)")
    ax1.plot(agent_run["hours"], agent_run["bg"], color="C3", lw=1.6,
             label=f"RL agent  (TIR {agent_mean['time_in_range_pct']:.0f}%)")
    ax1.set_ylabel("blood glucose (mg/dL)")
    ax1.set_title("RL agent vs PID baseline (one representative day)")
    ax1.legend(loc="upper right", fontsize=9)
    allbg = (pid_run["bg"] + agent_run["bg"]) or [400]
    ax1.set_ylim(0, max(400.0, max(allbg) * 1.05))

    ax2.plot(pid_run["hours"], pid_run["insulin"], color="C0", lw=1.0, label="PID")
    ax2.plot(agent_run["hours"], agent_run["insulin"], color="C3", lw=1.0, label="RL agent")
    ax2.set_ylabel("insulin\n(basal)")
    ax2.set_xlabel("time (hours)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=os.path.join(MODELS_DIR, "sac_glucose"))
    p.add_argument("--patient", default="adolescent#001")
    p.add_argument("--history", type=int, default=None,
                   help="override history length (legacy models without config.json)")
    p.add_argument("--episodes", type=int, default=5)
    args = p.parse_args()

    model_path = args.model if args.model.endswith(".zip") else args.model + ".zip"
    if not os.path.exists(model_path):
        sys.exit(f"No model at {model_path}. Train one first: python scripts/train_sac.py")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg = find_run_config(args.model)
    model = load_model(cfg["algo"], args.model)
    runner = PolicyRunner(model, recurrent=cfg["algo"] in RECURRENT)

    # --- PID baseline ---
    pid = PIDController(target=120.0, action_high=30.0)  # raw simglucose pump limit

    def pid_policy(obs, info):
        return pid.act(float(obs[0]))

    pid_runs, pid_mean, pid_std = evaluate(lambda s: make_pid_env(args.patient, s), pid_policy,
                                           args.episodes, STEPS_PER_DAY, on_reset=pid.reset)

    # --- RL agent ---

    def agent_policy(obs, info):
        return runner.act(obs)

    agent_runs, agent_mean, agent_std = evaluate(
        lambda s: make_agent_env(args.patient, cfg, s, args.history), agent_policy,
        args.episodes, STEPS_PER_DAY, on_reset=runner.reset)

    # --- report ---
    print(f"\n=== Mean over {args.episodes} episodes (patient {args.patient}) ===\n")
    print("PID baseline:")
    print(format_metrics(pid_mean))
    print(f"\nRL agent ({cfg['algo']}, env {cfg['env_version']}):")
    print(format_metrics(agent_mean))

    dtir = agent_mean["time_in_range_pct"] - pid_mean["time_in_range_pct"]
    dhypo = agent_mean["time_hypo_pct"] - pid_mean["time_hypo_pct"]
    print("\nDelta (agent - PID):")
    print(f"  Time in range: {dtir:+.1f} pts   (higher is better)")
    print(f"  Time hypo:     {dhypo:+.1f} pts   (lower is better)")

    out = os.path.join(RESULTS_DIR, "agent_vs_pid.png")
    comparison_plot(agent_runs[0], pid_runs[0], agent_mean, pid_mean, out)
    print(f"\nSaved plot -> {out}")


if __name__ == "__main__":
    main()
