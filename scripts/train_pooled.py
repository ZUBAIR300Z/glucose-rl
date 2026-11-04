"""
Pooled multi-patient RL training with leakage-free model selection.

Trains on a cohort of patients (a fresh patient AND a fresh meal day are sampled
each episode) and keeps the checkpoint that scores best on a VALIDATION set of
patients that is disjoint from the TEST patients used for the headline numbers.

Default split (stratified by cohort, test set identical to earlier studies):
    21 train / 3 validation / 6 held-out test patients.
Validation uses a fixed list of (patient, day) episodes, so every evaluation
scores the policy on the same days and "best checkpoint" is not seed noise.

Run from the project root:
    # original setup (v1 env, SAC), now without test-set leakage
    python scripts/train_pooled.py --timesteps 500000 --n-envs 6 --seed 0
    # redesigned env + a stronger off-policy algorithm
    python scripts/train_pooled.py --env-version v2 --algo tqc --tag v2-tqc --seed 0
    python scripts/train_pooled.py --env-version v2 --algo crossq --shield --tag v2-crossq-shield
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from diabetes_rl.algos import ALGOS, make_model
from diabetes_rl.cohorts import train_val_test_split
from diabetes_rl.config import env_kwargs
from diabetes_rl.rewards import REWARD_FUNCTIONS
from diabetes_rl.wrappers import make_glucose_env

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "pooled")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
VAL_SEED_OFFSET = 1000  # validation days never coincide with benchmark days (seeds 0..k)


def make_train_fn(idx, train_patients, cfg, seed):
    """Factory for one training worker (its own seed -> diverse patients and days)."""
    def _init():
        return Monitor(make_glucose_env(
            env_id=f"simglucose/pooled-train-{cfg['tag']}-s{seed}-w{idx}-v0",
            patient_name=train_patients, env_seed=seed * 1000 + idx,
            fresh_episodes=True, **env_kwargs(cfg)))
    return _init


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", default="sac", choices=ALGOS)
    p.add_argument("--env-version", default="v1", choices=["v1", "v2"],
                   help="v1 = original wrapper; v2 = basal-relative action, IOB, time of day")
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--n-envs", type=int, default=6)
    p.add_argument("--holdout-per-cohort", type=int, default=2, help="TEST patients per cohort")
    p.add_argument("--val-per-cohort", type=int, default=1,
                   help="VALIDATION patients per cohort (0 = validate on training patients, unseen days)")
    p.add_argument("--val-days", type=int, default=3, help="fixed validation days per validation patient")
    p.add_argument("--reward", default="magni", choices=list(REWARD_FUNCTIONS))
    p.add_argument("--learning-rate", type=float, default=None, help="default: per-algorithm")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--history", type=int, default=None, help="CGM history length (v1: 4, v2: 20)")
    p.add_argument("--terminal-penalty", type=float, default=None,
                   help="v1 default 100 (raw reward units); v2 default 1/(1-gamma) (normalized units)")
    p.add_argument("--max-multiplier", type=float, default=5.0, help="v2: max dose as a multiple of basal")
    p.add_argument("--shield", action="store_true", help="v2: train with the PLGS safety shield in the loop")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="default",
                   help="experiment name; outputs go to models/pooled/<tag>/")
    p.add_argument("--checkpoint-freq", type=int, default=50_000)
    p.add_argument("--eval-freq", type=int, default=25_000)
    args = p.parse_args()

    out_dir = os.path.join(MODELS_DIR, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # The split is keyed on a FIXED seed (0) so the patients are identical across
    # training seeds -> a fair, comparable generalization test.
    train_patients, val_patients, test_patients = train_val_test_split(
        args.holdout_per_cohort, args.val_per_cohort, seed=0)
    select_on = val_patients or train_patients
    val_episodes = [(pt, VAL_SEED_OFFSET + d) for pt in select_on for d in range(args.val_days)]
    print(f"TRAIN ({len(train_patients)}): {train_patients}")
    print(f"VALIDATION ({len(val_patients)}): {val_patients or '(training patients, unseen days)'}")
    print(f"HELD-OUT TEST ({len(test_patients)}): {test_patients}  <- never used during training")

    cfg = dict(algo=args.algo, env_version=args.env_version, reward=args.reward,
               history=args.history, terminal_penalty=args.terminal_penalty,
               max_multiplier=args.max_multiplier, gamma=args.gamma, shield=args.shield,
               learning_rate=args.learning_rate, timesteps=args.timesteps, tag=args.tag,
               train_patients=train_patients, val_patients=val_patients, test_patients=test_patients)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(out_dir, "split.txt"), "w") as f:
        f.write("train," + ",".join(train_patients) + "\n")
        f.write("val," + ",".join(val_patients) + "\n")
        f.write("test," + ",".join(test_patients) + "\n")

    fns = [make_train_fn(i, train_patients, cfg, args.seed) for i in range(args.n_envs)]
    env = SubprocVecEnv(fns) if args.n_envs > 1 else DummyVecEnv(fns)

    eval_env = Monitor(make_glucose_env(
        env_id=f"simglucose/pooled-val-{args.tag}-s{args.seed}-v0",
        patient_name=select_on[0], fixed_episodes=val_episodes, **env_kwargs(cfg)))

    model = make_model(args.algo, env, seed=args.seed, learning_rate=args.learning_rate,
                       gamma=args.gamma, tensorboard_log=LOGS_DIR)

    best_dir = os.path.join(out_dir, f"best_seed{args.seed}")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    callbacks = [
        CheckpointCallback(save_freq=max(args.checkpoint_freq // args.n_envs, 1),
                           save_path=ckpt_dir, name_prefix=f"pooled_s{args.seed}"),
        EvalCallback(eval_env, best_model_save_path=best_dir,
                     log_path=os.path.join(LOGS_DIR, f"{args.tag}_s{args.seed}"),
                     eval_freq=max(args.eval_freq // args.n_envs, 1),
                     n_eval_episodes=len(val_episodes), deterministic=True, verbose=0),
    ]

    print(f"\n{args.algo.upper()} [tag={args.tag}, env={args.env_version}]: {args.timesteps:,} steps, "
          f"seed {args.seed}, reward={args.reward}, shield={args.shield}. "
          f"Best checkpoint chosen on {len(val_episodes)} fixed validation days.\n")
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=False)
    final_out = os.path.join(out_dir, f"{args.algo}_pooled_seed{args.seed}")
    model.save(final_out)
    env.close()
    eval_env.close()

    dt = (time.time() - t0) / 60
    best_model = os.path.join(best_dir, "best_model")
    print(f"\nDone in {dt:.1f} min. Final -> {final_out}.zip | Best(validation) -> {best_model}.zip")
    print("\nEvaluate on the HELD-OUT test patients (the headline number):")
    print(f"  python scripts/benchmark.py --model {best_model} "
          f"--patients {' '.join(test_patients)} --seeds 5")


if __name__ == "__main__":
    main()
