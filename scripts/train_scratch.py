"""
Train the FROM-SCRATCH SAC (diabetes_rl/sac_scratch.py) on the glucose env.

This is the hand-written counterpart to scripts/train_sac.py (which uses
Stable-Baselines3). Same environment, same reward, same state — so the two are
directly comparable. Single environment (no parallelism) to keep the loop
simple and readable; it is therefore slower than the SB3 version.

Run from the project root:
    python scripts/train_scratch.py --timesteps 100000
    python scripts/train_scratch.py --timesteps 100000 --reward zone
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from diabetes_rl.envs import make_env, STEPS_PER_DAY
from diabetes_rl.metrics import glycemic_metrics, format_metrics
from diabetes_rl.rewards import REWARD_FUNCTIONS
from diabetes_rl.sac_scratch import SAC, ReplayBuffer
from diabetes_rl.wrappers import GlucoseTrendWrapper

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def evaluate(agent, env, n_steps):
    """One deterministic episode on the held-out env -> glycemic metrics."""
    obs, info = env.reset()
    bg = []
    for _ in range(n_steps):
        action = agent.select_action(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        bg.append(float(info["bg"]))
        if terminated or truncated:
            break
    return glycemic_metrics(bg)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--patient", default="adolescent#001")
    p.add_argument("--reward", default="magni", choices=list(REWARD_FUNCTIONS))
    p.add_argument("--history", type=int, default=4)
    p.add_argument("--learning-starts", type=int, default=1_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--eval-every", type=int, default=5_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(MODELS_DIR, "sac_scratch.pt"))
    args = p.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    reward_fun = REWARD_FUNCTIONS[args.reward]
    train_env = GlucoseTrendWrapper(
        make_env(env_id="simglucose/scratch-train-v0", patient_name=args.patient,
                 reward_fun=reward_fun, env_seed=args.seed),
        history_len=args.history)
    eval_env = GlucoseTrendWrapper(
        make_env(env_id="simglucose/scratch-eval-v0", patient_name=args.patient,
                 env_seed=12345),  # fixed held-out day
        history_len=args.history)

    obs_dim = train_env.observation_space.shape[0]
    act_dim = train_env.action_space.shape[0]
    agent = SAC(obs_dim, act_dim, train_env.action_space.low, train_env.action_space.high)
    buffer = ReplayBuffer(args.buffer_size, obs_dim, act_dim, agent.device)

    print(f"obs_dim={obs_dim}  act_dim={act_dim}  reward={args.reward}  "
          f"device={agent.device}")
    print(f"Training from-scratch SAC for {args.timesteps:,} steps...\n")

    obs, info = train_env.reset(seed=args.seed)
    curve, last_log = [], {}
    t0 = time.time()

    for step in range(1, args.timesteps + 1):
        # 1) act (random warmup, then policy)
        if step < args.learning_starts:
            action = train_env.action_space.sample()
        else:
            action = agent.select_action(obs)

        # 2) step the env and store the transition
        next_obs, reward, terminated, truncated, info = train_env.step(action)
        # IMPORTANT: only `terminated` (a real failure) stops bootstrapping.
        # `truncated` (hit the day's time limit) is NOT a true end, so we still
        # want to bootstrap from next_obs -> store done=0 in that case.
        buffer.add(obs, action, reward, next_obs, float(terminated))
        obs = next_obs
        if terminated or truncated:
            obs, info = train_env.reset()

        # 3) learn
        if step >= args.learning_starts:
            last_log = agent.update(buffer.sample(args.batch_size))

        # 4) periodic held-out evaluation
        if step % args.eval_every == 0:
            m = evaluate(agent, eval_env, STEPS_PER_DAY)
            curve.append((step, m["time_in_range_pct"], m["time_hypo_pct"]))
            fps = step / (time.time() - t0)
            print(f"step {step:>7,} | eval TIR={m['time_in_range_pct']:5.1f}%  "
                  f"hypo={m['time_hypo_pct']:4.1f}%  | alpha={last_log.get('alpha',0):.3f}  "
                  f"q_loss={last_log.get('q_loss',0):8.1f}  | {fps:.0f} steps/s")

    agent.save(args.out)
    train_env.close()
    eval_env.close()
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. Saved -> {args.out}")

    # learning curve plot
    if curve:
        steps, tir, hypo = zip(*curve)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(steps, tir, "-o", color="C2", label="Time in Range")
        ax.plot(steps, hypo, "-o", color="C3", label="Time hypo")
        ax.set_xlabel("training steps")
        ax.set_ylabel("%")
        ax.set_title("From-scratch SAC — held-out learning curve")
        ax.legend()
        ax.grid(alpha=0.2)
        ax.set_ylim(0, 100)
        out = os.path.join(RESULTS_DIR, "scratch_learning_curve.png")
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        print(f"Saved learning curve -> {out}")

    final = evaluate(agent, eval_env, STEPS_PER_DAY)
    print("\nFinal held-out evaluation:")
    print(format_metrics(final))


if __name__ == "__main__":
    main()
