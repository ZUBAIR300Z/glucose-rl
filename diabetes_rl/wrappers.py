"""
Observation / reward wrappers that turn the raw simglucose env into something
an RL agent can actually learn from.

The raw environment gives the agent a single number: the current CGM reading.
That is not enough -- with one number the agent cannot tell whether glucose is
*rising* or *falling*, so it cannot anticipate. This is the single most
important piece of feature engineering in the project.

``GlucoseTrendWrapper`` fixes that by giving the agent a short history of:
  * the last ``history_len`` CGM readings  (so it can see the trend), and
  * the last ``history_len`` insulin doses (a proxy for "insulin on board",
    since injected insulin keeps lowering glucose for a while).

It also adds a **terminal penalty**: simglucose ends an episode early when the
patient crashes (BG < 10 or > 600). Because the per-step reward is negative
(a cost), a shorter episode would otherwise look *better* to the agent -- it
could learn to kill the patient quickly to stop accumulating penalty. The
terminal penalty removes that perverse incentive.
"""
from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np

from diabetes_rl.envs import make_env, STEPS_PER_DAY

CGM_SCALE = 400.0  # normalize glucose (mg/dL) into roughly [0, 1.5]

# The pump's max basal (~30 U) is enormous vs. a useful dose (~0.01-0.05 U), so
# the good region is a tiny sliver of [0, 30] that SAC cannot find -- and its
# neutral output (the midpoint, 15 U) is already a catastrophic overdose. We
# expose a clinically sane action range [0, MAX_BASAL] to the agent instead.
# This single change is the difference between crashing every patient and
# actually learning to control glucose.
DEFAULT_MAX_BASAL = 0.1


class GlucoseTrendWrapper(gym.Wrapper):
    def __init__(self, env, history_len: int = 4, terminal_penalty: float = 100.0,
                 max_basal: float = DEFAULT_MAX_BASAL):
        super().__init__(env)
        self.history_len = history_len
        self.terminal_penalty = terminal_penalty

        # Rescale the action space to a clinically sane basal range so the agent
        # explores where the good doses actually are (see DEFAULT_MAX_BASAL note).
        self.max_basal = float(max_basal)
        self.action_space = gym.spaces.Box(
            low=0.0, high=self.max_basal, shape=(1,), dtype=np.float32)
        self.max_insulin = self.max_basal  # normalize the insulin history by this

        # obs = [history_len normalized CGM] + [history_len normalized insulin]
        self.observation_space = gym.spaces.Box(
            low=0.0, high=2.0, shape=(2 * history_len,), dtype=np.float32
        )
        self.cgm_hist = deque(maxlen=history_len)
        self.ins_hist = deque(maxlen=history_len)

    def _build_obs(self) -> np.ndarray:
        cgm = np.asarray(self.cgm_hist, dtype=np.float32) / CGM_SCALE
        ins = np.asarray(self.ins_hist, dtype=np.float32) / self.max_insulin
        return np.concatenate([cgm, ins]).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        cgm0 = float(obs[0])
        self.cgm_hist.clear()
        self.ins_hist.clear()
        # pad history with the initial reading / no insulin
        for _ in range(self.history_len):
            self.cgm_hist.append(cgm0)
            self.ins_hist.append(0.0)
        return self._build_obs(), info

    def step(self, action):
        # Agent acts in [0, max_basal]; clip for safety, then pass the (small,
        # valid) dose straight through to the underlying [0, 30] pump.
        action = np.clip(np.asarray(action, dtype=np.float32), 0.0, self.max_basal)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.cgm_hist.append(float(obs[0]))
        self.ins_hist.append(float(action.ravel()[0]))

        if terminated:
            reward = float(reward) - self.terminal_penalty

        return self._build_obs(), reward, terminated, truncated, info


class ResamplePatientWrapper(gym.Wrapper):
    """Resample the simglucose patient from a list on every reset.

    simglucose fixes the patient at construction and does NOT change it on
    reset, so pooled multi-patient training needs this: before each episode it
    rebuilds the underlying simulator with a freshly sampled patient from the
    list passed as ``patient_name``. Single-patient envs are unaffected.

    (All 30 patients use the same pump/sensor hardware, so the observation and
    action spaces are identical across patients — swapping the patient is safe.)
    """

    def reset(self, **kwargs):
        base = self.env.unwrapped              # T1DSimGymnaisumEnv
        inner = getattr(base, "env", None)     # T1DSimEnv (gym-API wrapper)
        if inner is not None and isinstance(inner.patient_name, (list, tuple)):
            inner.env, _, _, _ = inner._create_env()  # sample new patient + rebuild
        return self.env.reset(**kwargs)


def make_glucose_env(env_id, patient_name="adolescent#001", reward_fun=None,
                     env_seed=None, history_len=4, max_basal=DEFAULT_MAX_BASAL,
                     terminal_penalty=100.0, max_episode_steps=STEPS_PER_DAY):
    """Build the full agent environment, identically for training and evaluation.

    Stacks: simglucose env -> (patient resampling, if ``patient_name`` is a
    list) -> GlucoseTrendWrapper (trend state + clinically-scaled action). Using
    one factory everywhere guarantees train/eval use the exact same setup.
    """
    env = make_env(env_id=env_id, patient_name=patient_name, reward_fun=reward_fun,
                   env_seed=env_seed, max_episode_steps=max_episode_steps)
    if isinstance(patient_name, (list, tuple)):
        env = ResamplePatientWrapper(env)
    env = GlucoseTrendWrapper(env, history_len=history_len,
                              terminal_penalty=terminal_penalty, max_basal=max_basal)
    return env
