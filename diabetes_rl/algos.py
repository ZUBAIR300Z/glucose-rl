"""
One place to build / load every RL algorithm the project supports.

    sac           Soft Actor-Critic (SB3) -- the original choice.
    td3           Twin Delayed DDPG (SB3) -- deterministic actor, a useful contrast to SAC.
    tqc           Truncated Quantile Critics (sb3-contrib): SAC with distributional
                  critics and the top quantiles dropped. Directly attacks Q-value
                  *overestimation*, which in this task shows up as over-dosing.
    crossq        CrossQ (sb3-contrib): batch-renormalized critics, no target
                  networks; markedly more sample-efficient than SAC -- valuable
                  because the simulator (~100-200 steps/s/core) is the bottleneck.
    ppo           PPO (SB3): on-policy; typically lower seed variance, more samples.
    recurrentppo  PPO with an LSTM policy (sb3-contrib): memory for the partial
                  observability (unobserved meals, insulin still acting).
"""
from __future__ import annotations

import numpy as np

from diabetes_rl.envs import STEPS_PER_DAY

OFF_POLICY = ("sac", "td3", "tqc", "crossq")
ON_POLICY = ("ppo", "recurrentppo")
ALGOS = OFF_POLICY + ON_POLICY
RECURRENT = ("recurrentppo",)

DEFAULT_LR = {"sac": 3e-4, "td3": 3e-4, "tqc": 3e-4, "crossq": 1e-3, "ppo": 3e-4, "recurrentppo": 3e-4}


def algo_class(name: str):
    name = name.lower()
    if name == "sac":
        from stable_baselines3 import SAC as cls
    elif name == "td3":
        from stable_baselines3 import TD3 as cls
    elif name == "ppo":
        from stable_baselines3 import PPO as cls
    elif name in ("tqc", "crossq", "recurrentppo"):
        try:
            import sb3_contrib
        except ImportError as e:  # pragma: no cover - depends on the install
            raise ImportError(f"--algo {name} needs sb3-contrib: pip install sb3-contrib") from e
        cls = {"tqc": sb3_contrib.TQC, "crossq": sb3_contrib.CrossQ,
               "recurrentppo": sb3_contrib.RecurrentPPO}[name]
    else:
        raise ValueError(f"unknown algo {name!r}; choose from {ALGOS}")
    return cls


def make_model(name: str, env, seed: int = 0, learning_rate: float | None = None,
               gamma: float = 0.99, buffer_size: int = 300_000, learning_starts: int = 5_000,
               tensorboard_log: str | None = None, verbose: int = 1):
    """Construct an untrained model with sensible defaults for this task."""
    name = name.lower()
    cls = algo_class(name)
    lr = DEFAULT_LR[name] if learning_rate is None else learning_rate
    common = dict(learning_rate=lr, gamma=gamma, seed=seed, verbose=verbose,
                  tensorboard_log=tensorboard_log)

    if name in OFF_POLICY:
        kw = dict(buffer_size=buffer_size, learning_starts=learning_starts, batch_size=256,
                  train_freq=1, gradient_steps=1, **common)
        if name in ("sac", "td3", "tqc"):
            kw["tau"] = 0.005
        if name == "td3":
            from stable_baselines3.common.noise import NormalActionNoise
            n = env.action_space.shape[0]
            kw["action_noise"] = NormalActionNoise(np.zeros(n), 0.1 * np.ones(n))
        if name == "tqc":
            kw["top_quantiles_to_drop_per_net"] = 2
        return cls("MlpPolicy", env, **kw)

    # On-policy: one simulated day per env per rollout.
    policy = "MlpLstmPolicy" if name in RECURRENT else "MlpPolicy"
    return cls(policy, env, n_steps=STEPS_PER_DAY, batch_size=240, n_epochs=10,
               gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, **common)


def zero_init_policy_head(model) -> None:
    """Zero the layer that outputs the (mean) action, so the untrained policy's
    deterministic action is exactly 0 for every observation.

    For residual RL that makes the untrained agent identical to the base
    controller (action 0 = "no correction"); exploration noise (SAC's learned
    std, TD3's action noise, PPO's log-std) is untouched, and the zeroed layer
    still receives gradients, so learning proceeds normally.
    """
    import torch as th

    def zero(layer):
        with th.no_grad():
            layer.weight.zero_()
            layer.bias.zero_()

    pol = model.policy
    if hasattr(pol, "action_net"):                      # PPO / RecurrentPPO
        zero(pol.action_net)
        return
    for actor in (pol.actor, getattr(pol, "actor_target", None)):
        if actor is None:
            continue
        mu = actor.mu
        if isinstance(mu, th.nn.Linear):                 # SAC / TQC / CrossQ
            zero(mu)
        else:                                            # TD3: Sequential(..., Linear, Tanh)
            zero([m for m in mu if isinstance(m, th.nn.Linear)][-1])


def load_model(name: str, path: str):
    return algo_class(name).load(path)


class PolicyRunner:
    """Uniform ``reset()`` / ``act(obs)`` over feed-forward and recurrent policies."""

    def __init__(self, model, recurrent: bool = False):
        self.model, self.recurrent = model, recurrent
        self.reset()

    def reset(self):
        self.state, self.start = None, True

    def act(self, obs):
        if self.recurrent:
            a, self.state = self.model.predict(obs, state=self.state,
                                               episode_start=np.array([self.start]),
                                               deterministic=True)
        else:
            a, _ = self.model.predict(obs, deterministic=True)
        self.start = False
        return np.asarray(a, dtype=np.float32)
