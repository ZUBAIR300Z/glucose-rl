"""
Pooled multi-patient SAC training with HELD-OUT evaluation.

Trains on a cohort of patients (a fresh patient is sampled each episode) and
selects the best model by performance on a DISJOINT set of held-out patients.
This is the protocol required to claim *generalizable* glucose control rather
than single-patient overfitting — the central weakness of the first study.

Default split: stratified hold-out of `--holdout-per-cohort` patients from each
of {adolescent, adult, child} (2 each -> 24 train / 6 unseen test).

Run from the project root:
    python scripts/train_pooled.py --timesteps 500000 --n-envs 6 --seed 0
    python scripts/train_pooled.py --holdout-per-cohort 2 --seed 1
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from diabetes_rl.cohorts import stratified_holdout
from diabetes_rl.envs import STEPS_PER_DAY
from diabetes_rl.rewards import REWARD_FUNCTIONS
from diabetes_rl.wrappers import make_glucose_env

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "pooled")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")


def make_train_fn(idx, train_patients, reward_name, history, terminal_penalty, seed):
    """Factory for one training worker (its own seed -> diverse patient sampling)."""
    def _init():
        return Monitor(make_glucose_env(
            env_id=f"simglucose/pooled-train-s{seed}-w{idx}-v0",
            patient_name=train_patients,
            reward_fun=REWARD_FUNCTIONS[reward_name],
            env_seed=seed * 1000 + idx,
            history_len=history,
            terminal_penalty=terminal_penalty,
        ))
    return _init


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--n-envs", type=int, default=6)
    p.add_argument("--holdout-per-cohort", type=int, default=2)
    p.add_argument("--reward", default="magni", choices=list(REWARD_FUNCTIONS))
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--history", type=int, default=4)
    p.add_argument("--terminal-penalty", type=float, default=100.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="default",
                   help="experiment name; outputs go to models/pooled/<tag>/")
    p.add_argument("--checkpoint-freq", type=int, default=50_000)
    p.add_argument("--eval-freq", type=int, default=25_000)
    args = p.parse_args()

    out_dir = os.path.join(MODELS_DIR, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # NOTE: the split is keyed on a FIXED seed (0) so the held-out patients are
    # identical across training seeds -> a fair, comparable generalization test.
    train_patients, test_patients = stratified_holdout(args.holdout_per_cohort, seed=0)
    print(f"TRAIN ({len(train_patients)}): {train_patients}")
    print(f"HELD-OUT TEST ({len(test_patients)}): {test_patients}")

    split_path = os.path.join(out_dir, "split.txt")
    with open(split_path, "w") as f:
        f.write("train," + ",".join(train_patients) + "\n")
        f.write("test," + ",".join(test_patients) + "\n")

    fns = [make_train_fn(i, train_patients, args.reward, args.history,
                         args.terminal_penalty, args.seed) for i in range(args.n_envs)]
    env = SubprocVecEnv(fns) if args.n_envs > 1 else DummyVecEnv(fns)

    eval_env = Monitor(make_glucose_env(
        env_id=f"simglucose/pooled-eval-s{args.seed}-v0",
        patient_name=test_patients, reward_fun=REWARD_FUNCTIONS[args.reward],
        env_seed=99999, history_len=args.history, terminal_penalty=args.terminal_penalty))

    model = SAC(
        "MlpPolicy", env,
        learning_rate=args.learning_rate, buffer_size=300_000, learning_starts=5_000,
        batch_size=256, gamma=0.99, tau=0.005, train_freq=1, gradient_steps=1,
        verbose=1, seed=args.seed, tensorboard_log=LOGS_DIR,
    )

    best_dir = os.path.join(out_dir, f"best_seed{args.seed}")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    callbacks = [
        CheckpointCallback(save_freq=max(args.checkpoint_freq // args.n_envs, 1),
                           save_path=ckpt_dir, name_prefix=f"pooled_s{args.seed}"),
        EvalCallback(eval_env, best_model_save_path=best_dir, log_path=LOGS_DIR,
                     eval_freq=max(args.eval_freq // args.n_envs, 1),
                     n_eval_episodes=6, deterministic=True, verbose=0),
    ]

    print(f"\nPooled SAC [tag={args.tag}]: {args.timesteps:,} steps, seed {args.seed}, "
          f"reward={args.reward}, lr={args.learning_rate}, {len(train_patients)} patients. "
          f"Best model chosen on held-out patients.\n")
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=False)
    final_out = os.path.join(out_dir, f"sac_pooled_seed{args.seed}")
    model.save(final_out)
    env.close()
    eval_env.close()

    dt = (time.time() - t0) / 60
    best_model = os.path.join(best_dir, "best_model")
    print(f"\nDone in {dt:.1f} min. Final -> {final_out}.zip | Best(held-out) -> {best_model}.zip")
    print("\nEvaluate on the HELD-OUT patients (the headline number):")
    print(f"  python scripts/benchmark.py --model {best_model} "
          f"--patients {' '.join(test_patients)} --seeds 5")


if __name__ == "__main__":
    main()
