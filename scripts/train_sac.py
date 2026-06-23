"""
Train a SAC agent to control basal insulin on the simglucose environment.

SAC (Soft Actor-Critic) is an off-policy, continuous-control deep RL algorithm
-- a good default for this problem because the action (insulin rate) is
continuous. We use Stable-Baselines3's implementation; you do NOT implement the
algorithm yourself. Your effort goes into the env, the state, and the reward.

Run from the project root (examples):
    python scripts/train_sac.py                          # default 50k steps
    python scripts/train_sac.py --timesteps 200000 --n-envs 4
    python scripts/train_sac.py --reward zone --history 6

--n-envs > 1 runs several simulators in parallel processes (much faster on a
multi-core CPU). Watch training live in another terminal:
    tensorboard --logdir logs
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

from diabetes_rl.envs import make_env, STEPS_PER_DAY
from diabetes_rl.rewards import REWARD_FUNCTIONS
from diabetes_rl.wrappers import GlucoseTrendWrapper

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")


def env_id_for(patient: str, reward_name: str) -> str:
    """A unique, registration-safe env id per (patient, reward) combo."""
    pid = patient.replace("#", "")
    return f"simglucose/{pid}-train-{reward_name}-v0"


def make_env_fn(patient, reward_name, history, terminal_penalty):
    """Return a zero-arg factory that builds one wrapped, monitored env.

    Used by the vectorized env. Note we pass `reward_name` (a string), not the
    function object, so it is looked up inside each worker process -- clean and
    pickle-safe.
    """
    def _init():
        env = make_env(
            env_id=env_id_for(patient, reward_name),
            patient_name=patient,
            reward_fun=REWARD_FUNCTIONS[reward_name],
            max_episode_steps=STEPS_PER_DAY,
        )
        env = GlucoseTrendWrapper(env, history_len=history, terminal_penalty=terminal_penalty)
        return Monitor(env)

    return _init


def build_vec_env(patient, reward_name, history, terminal_penalty, n_envs):
    fns = [make_env_fn(patient, reward_name, history, terminal_penalty) for _ in range(n_envs)]
    if n_envs > 1:
        return SubprocVecEnv(fns)
    return DummyVecEnv(fns)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timesteps", type=int, default=50_000)
    p.add_argument("--n-envs", type=int, default=1, help="parallel simulators (try 4-6)")
    p.add_argument("--patient", default="adolescent#001")
    p.add_argument("--reward", default="magni", choices=list(REWARD_FUNCTIONS))
    p.add_argument("--history", type=int, default=4)
    p.add_argument("--terminal-penalty", type=float, default=100.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(MODELS_DIR, "sac_glucose"))
    p.add_argument("--checkpoint-freq", type=int, default=25_000,
                   help="save a checkpoint every N steps (crash insurance)")
    p.add_argument("--eval-freq", type=int, default=10_000,
                   help="evaluate on a held-out day every N steps; saves best model")
    p.add_argument("--no-callbacks", action="store_true",
                   help="disable checkpoint + eval callbacks")
    args = p.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    env = build_vec_env(args.patient, args.reward, args.history,
                        args.terminal_penalty, args.n_envs)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space:      {env.action_space}")
    print(f"Reward: {args.reward} | history: {args.history} | parallel envs: {args.n_envs}")

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=1_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        seed=args.seed,
        tensorboard_log=LOGS_DIR,
    )

    # Callbacks: periodic checkpoints (crash insurance) + eval on a held-out day
    # to save the BEST model, not just the final one (RL can degrade late).
    callbacks = []
    if not args.no_callbacks:
        ckpt_dir = os.path.join(MODELS_DIR, "checkpoints")
        best_dir = os.path.join(MODELS_DIR, "best")
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(best_dir, exist_ok=True)
        callbacks.append(CheckpointCallback(
            save_freq=max(args.checkpoint_freq // args.n_envs, 1),
            save_path=ckpt_dir, name_prefix="sac",
        ))
        eval_raw = make_env(
            env_id=env_id_for(args.patient, args.reward) + "-eval",
            patient_name=args.patient, reward_fun=REWARD_FUNCTIONS[args.reward],
            env_seed=12345,  # fixed held-out day for a consistent learning curve
        )
        eval_env = Monitor(GlucoseTrendWrapper(
            eval_raw, history_len=args.history, terminal_penalty=args.terminal_penalty))
        callbacks.append(EvalCallback(
            eval_env, best_model_save_path=best_dir, log_path=LOGS_DIR,
            eval_freq=max(args.eval_freq // args.n_envs, 1),
            n_eval_episodes=3, deterministic=True, verbose=0,
        ))

    print(f"\nTraining SAC for {args.timesteps:,} timesteps "
          f"(~{args.timesteps / STEPS_PER_DAY:.0f} simulated days)...")
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps,
                callback=callbacks if callbacks else None, progress_bar=False)
    dt = time.time() - t0

    model.save(args.out)
    env.close()
    print(f"\nDone in {dt/60:.1f} min ({args.timesteps/max(dt,1e-9):.0f} steps/s).")
    print(f"Saved final model -> {args.out}.zip")
    if not args.no_callbacks:
        print(f"Best model      -> {os.path.join(MODELS_DIR, 'best', 'best_model.zip')}")
        print(f"Checkpoints     -> {os.path.join(MODELS_DIR, 'checkpoints')}")
    print("Next: python scripts/benchmark.py   (or evaluate.py)")


if __name__ == "__main__":
    main()
