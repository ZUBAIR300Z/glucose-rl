"""
Environment setup for the simglucose Type-1 Diabetes simulator.

simglucose ships a Gymnasium-compatible environment class
(``T1DSimGymnaisumEnv`` -- the spelling is theirs, not a typo on our side).
We register it under a clean id and expose a ``make_env`` factory so the rest
of the project never has to remember the entry-point string.

The control problem
-------------------
    Observation : a single CGM (continuous glucose monitor) reading, in mg/dL.
    Action      : a single basal insulin rate, in [0, max_basal].
    Reward      : by default simglucose's ``risk_diff`` (the reduction in a
                  glycemic risk index). Pass a custom function via
                  ``reward_fun`` -- see diabetes_rl/rewards.py.

Patient cohorts available in simglucose:
    adolescent#001..010, adult#001..010, child#001..010
"""
from __future__ import annotations

import gymnasium as gym
from gymnasium.envs.registration import register, registry

DEFAULT_PATIENT = "adolescent#001"

# simglucose's CGM (Dexcom) samples every 3 minutes, so one simulated day is
# 24 * 60 / 3 = 480 steps. We cap an episode at one day by default.
STEPS_PER_DAY = 480


def register_env(
    env_id: str = "simglucose/adolescent1-v0",
    patient_name: str = DEFAULT_PATIENT,
    max_episode_steps: int = STEPS_PER_DAY,
    reward_fun=None,
    env_seed=None,
) -> str:
    """Register the simglucose gymnasium env under ``env_id`` (idempotent).

    Note: registration bakes ``reward_fun`` and ``env_seed`` into the env spec.
    For a different reward or seed, register under a different ``env_id``.

    ``env_seed`` is important: simglucose draws its random meal scenario at
    *construction* time (not on ``reset``), so the only way to get a genuinely
    different day per run is to build the env with a different seed.
    """
    if env_id in registry:
        return env_id

    kwargs = {"patient_name": patient_name}
    if reward_fun is not None:
        kwargs["reward_fun"] = reward_fun
    if env_seed is not None:
        kwargs["seed"] = env_seed

    register(
        id=env_id,
        entry_point="simglucose.envs:T1DSimGymnaisumEnv",
        max_episode_steps=max_episode_steps,
        kwargs=kwargs,
    )
    return env_id


def make_env(
    env_id: str = "simglucose/adolescent1-v0",
    patient_name: str = DEFAULT_PATIENT,
    max_episode_steps: int = STEPS_PER_DAY,
    reward_fun=None,
    env_seed=None,
    render_mode=None,
):
    """Register (if needed) and construct the environment.

    Pass ``env_seed`` to get a distinct meal scenario (see ``register_env``).
    """
    register_env(
        env_id=env_id,
        patient_name=patient_name,
        max_episode_steps=max_episode_steps,
        reward_fun=reward_fun,
        env_seed=env_seed,
    )
    return gym.make(env_id, render_mode=render_mode)
