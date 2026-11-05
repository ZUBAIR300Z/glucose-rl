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
    # residual RL: SAC learns corrections on top of the tuned basal PID
    python scripts/train_pooled.py --env-version v2 --residual 0.5 --tag v2-res-sac --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from diabetes_rl.algos import ALGOS, OFF_POLICY, algo_class, make_model, zero_init_policy_head
from diabetes_rl.cohorts import train_val_test_split
from diabetes_rl.config import env_kwargs
from diabetes_rl.rewards import REWARD_FUNCTIONS
from diabetes_rl.wrappers import make_glucose_env

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "pooled")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
VAL_SEED_OFFSET = 1000  # validation days never coincide with benchmark days (seeds 0..k)


def make_train_fn(idx, train_patients, cfg, seed, round_=0):
    """Factory for one training worker (its own seed -> diverse patients and days).

    ``round_`` > 0 after a resume: shifts the env seeds so the resumed part of
    training does not replay the days the first part already saw.
    """
    def _init():
        return Monitor(make_glucose_env(
            env_id=f"simglucose/pooled-train-{cfg['tag']}-s{seed}-r{round_}-w{idx}-v0",
            patient_name=train_patients, env_seed=seed * 1000 + idx + 100_000 * round_,
            fresh_episodes=True, **env_kwargs(cfg)))
    return _init


class ResumeSnapshot(BaseCallback):
    """Crash-safe snapshot every ``save_freq`` calls: the model, the replay buffer
    (off-policy algorithms) and the step count, written to a temp folder and then
    swapped in, so a killed run continues where it stopped instead of restarting."""

    def __init__(self, path, save_freq, round_, save_buffer):
        super().__init__()
        self.path, self.save_freq, self.round_, self.save_buffer = path, save_freq, round_, save_buffer

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.snapshot()
        return True

    def snapshot(self):
        tmp, old = self.path + ".tmp", self.path + ".old"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        self.model.save(os.path.join(tmp, "model.zip"))
        if self.save_buffer:
            self.model.save_replay_buffer(os.path.join(tmp, "replay.pkl"))
        with open(os.path.join(tmp, "state.json"), "w") as f:
            json.dump({"round": self.round_, "num_timesteps": int(self.model.num_timesteps)}, f)
        shutil.rmtree(old, ignore_errors=True)
        if os.path.exists(self.path):
            os.rename(self.path, old)
        os.rename(tmp, self.path)
        shutil.rmtree(old, ignore_errors=True)


def find_snapshot(path):
    """-> (folder, state) of the newest complete snapshot, or (None, None)."""
    for d in (path, path + ".old"):
        if os.path.exists(os.path.join(d, "state.json")) and os.path.exists(os.path.join(d, "model.zip")):
            with open(os.path.join(d, "state.json")) as f:
                return d, json.load(f)
    return None, None


def restore_eval_history(eval_cb, npz_path, upto):
    """Reload EvalCallback history so a resumed run keeps its best-so-far score
    (and does not overwrite best_model.zip with a worse model)."""
    if not os.path.exists(npz_path):
        return
    d = np.load(npz_path)
    keep = d["timesteps"] <= upto
    eval_cb.evaluations_timesteps = [int(t) for t in d["timesteps"][keep]]
    eval_cb.evaluations_results = [list(r) for r in d["results"][keep]]
    eval_cb.evaluations_length = [list(l) for l in d["ep_lengths"][keep]]
    # best over ALL recorded evaluations: best_model.zip may come from after the snapshot
    eval_cb.best_mean_reward = float(np.max(np.mean(d["results"], axis=1)))


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
    p.add_argument("--max-multiplier", type=float, default=None,
                   help="v2: max dose as a multiple of basal (default 5; 8 with --residual, the tuned PID's cap)")
    p.add_argument("--residual", type=float, default=0.0,
                   help="v2: residual RL around the tuned basal PID with this residual scale "
                        "(final action = PID action + scale * agent action; 0 = off)")
    p.add_argument("--shield", action="store_true", help="v2: train with the PLGS safety shield in the loop")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="default",
                   help="experiment name; outputs go to models/pooled/<tag>/")
    p.add_argument("--checkpoint-freq", type=int, default=50_000)
    p.add_argument("--eval-freq", type=int, default=25_000)
    p.add_argument("--resume-freq", type=int, default=25_000,
                   help="crash-safe snapshot every N steps; a rerun continues from it (0 = off)")
    p.add_argument("--no-resume", action="store_true", help="ignore an existing snapshot and start over")
    p.add_argument("--vec-env", default="auto", choices=["auto", "subproc", "dummy"],
                   help="subproc = one process per env (fastest for a single run); dummy = all envs "
                        "in-process (far less memory; best when several runs share the CPU)")
    args = p.parse_args()

    if args.residual and args.env_version != "v2":
        p.error("--residual needs --env-version v2")
    if args.max_multiplier is None:
        args.max_multiplier = 8.0 if args.residual else 5.0

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
               residual=args.residual, learning_rate=args.learning_rate, timesteps=args.timesteps, tag=args.tag,
               train_patients=train_patients, val_patients=val_patients, test_patients=test_patients)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(out_dir, "split.txt"), "w") as f:
        f.write("train," + ",".join(train_patients) + "\n")
        f.write("val," + ",".join(val_patients) + "\n")
        f.write("test," + ",".join(test_patients) + "\n")

    snap_path = os.path.join(out_dir, f"resume_seed{args.seed}")
    snap_dir, state = (None, None) if args.no_resume else find_snapshot(snap_path)
    round_ = state["round"] + 1 if state else 0

    fns = [make_train_fn(i, train_patients, cfg, args.seed, round_) for i in range(args.n_envs)]
    use_subproc = args.vec_env == "subproc" or (args.vec_env == "auto" and args.n_envs > 1)
    env = SubprocVecEnv(fns) if use_subproc else DummyVecEnv(fns)

    eval_env = Monitor(make_glucose_env(
        env_id=f"simglucose/pooled-val-{args.tag}-s{args.seed}-v0",
        patient_name=select_on[0], fixed_episodes=val_episodes, **env_kwargs(cfg)))

    if state:
        model = algo_class(args.algo).load(os.path.join(snap_dir, "model.zip"), env=env,
                                           tensorboard_log=LOGS_DIR, verbose=1)
        buffer_file = os.path.join(snap_dir, "replay.pkl")
        if args.algo in OFF_POLICY and os.path.exists(buffer_file):
            model.load_replay_buffer(buffer_file)
        model.set_random_seed(args.seed + 1000 * round_)
        print(f"RESUMING from step {state['num_timesteps']:,} (resume #{round_})")
    else:
        model = make_model(args.algo, env, seed=args.seed, learning_rate=args.learning_rate,
                           gamma=args.gamma, tensorboard_log=LOGS_DIR)
        if args.residual:
            zero_init_policy_head(model)  # untrained agent == the tuned PID

    best_dir = os.path.join(out_dir, f"best_seed{args.seed}")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    eval_log = os.path.join(LOGS_DIR, f"{args.tag}_s{args.seed}")
    eval_cb = EvalCallback(eval_env, best_model_save_path=best_dir, log_path=eval_log,
                           eval_freq=max(args.eval_freq // args.n_envs, 1),
                           n_eval_episodes=len(val_episodes), deterministic=True, verbose=0)
    if state:
        restore_eval_history(eval_cb, os.path.join(eval_log, "evaluations.npz"), state["num_timesteps"])
    callbacks = [CheckpointCallback(save_freq=max(args.checkpoint_freq // args.n_envs, 1),
                                    save_path=ckpt_dir, name_prefix=f"pooled_s{args.seed}")]
    if args.resume_freq > 0:
        callbacks.append(ResumeSnapshot(snap_path, max(args.resume_freq // args.n_envs, 1), round_,
                                        save_buffer=args.algo in OFF_POLICY))
    callbacks.append(eval_cb)

    print(f"\n{args.algo.upper()} [tag={args.tag}, env={args.env_version}"
          f"{f', residual on tuned PID x{args.residual:g}' if args.residual else ''}]: {args.timesteps:,} steps, "
          f"seed {args.seed}, reward={args.reward}, shield={args.shield}. "
          f"Best checkpoint chosen on {len(val_episodes)} fixed validation days.\n")
    t0 = time.time()
    done_steps = state["num_timesteps"] if state else 0
    model.learn(total_timesteps=args.timesteps - done_steps, callback=callbacks, progress_bar=False,
                reset_num_timesteps=not state, tb_log_name=f"{args.tag}_s{args.seed}")
    final_out = os.path.join(out_dir, f"{args.algo}_pooled_seed{args.seed}")
    model.save(final_out)
    for d in (snap_path, snap_path + ".old", snap_path + ".tmp"):
        shutil.rmtree(d, ignore_errors=True)  # finished: the snapshot is no longer needed
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
