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

``GlucoseControlWrapper`` ("v2") is the redesigned version: patient-relative
actions, insulin-on-board, time of day, a normalized reward and an optional
safety shield. ``make_glucose_env(env_version=...)`` selects between them;
v1 is kept unchanged so the original results stay reproducible.
"""
from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np

from diabetes_rl.envs import make_env, STEPS_PER_DAY
from diabetes_rl.patients import basal_rate
from diabetes_rl.safety import ShieldWrapper

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


def _sim(env):
    """The simglucose gym-API ``T1DSimEnv`` under all gymnasium wrappers."""
    return env.unwrapped.env


class FreshEpisodeWrapper(gym.Wrapper):
    """Rebuild the simulator on every reset: new meal scenario, start hour and
    sensor noise -- plus a freshly sampled patient if ``patient_name`` is a list.

    Why this is needed: simglucose draws the meal scenario when the simulator is
    *built*, and its ``reset()`` replays the exact same day. Without this
    wrapper a training worker sees one meal day over and over (so
    ``train_sac.py`` with 6 workers trained on only 6 distinct days).

    (All 30 patients use the same pump/sensor hardware, so the observation and
    action spaces are identical across patients — swapping the patient is safe.)
    """

    def reset(self, **kwargs):
        inner = getattr(self.env.unwrapped, "env", None)   # T1DSimEnv (gym-API wrapper)
        if inner is not None:
            inner.env, _, _, _ = inner._create_env()      # new day (+ new patient)
        return self.env.reset(**kwargs)


# Backward-compatible name: it was only ever applied to list-of-patient envs,
# where the behavior is unchanged.
ResamplePatientWrapper = FreshEpisodeWrapper


class FixedEpisodesWrapper(gym.Wrapper):
    """Cycle deterministically through a fixed list of ``(patient, seed)`` days.

    Used for validation / model selection: every evaluation scores the policy on
    the *same* set of days, so "best checkpoint" reflects the policy, not which
    random days happened to be drawn. Day ``(p, s)`` is identical to the day a
    benchmark env built with ``patient_name=p, env_seed=s`` produces.
    """

    def __init__(self, env, episodes):
        super().__init__(env)
        self.episodes = list(episodes)
        self._i = 0

    def reset(self, **kwargs):
        from gym.utils import seeding  # the legacy gym that simglucose itself uses

        patient, seed = self.episodes[self._i % len(self.episodes)]
        self._i += 1
        inner = _sim(self.env)
        inner.patient_name = patient
        inner.np_random, _ = seeding.np_random(seed=seed)
        inner.env, _, _, _ = inner._create_env()
        return self.env.reset(**kwargs)


# --------------------------------------------------------------------------
# v2 environment
# --------------------------------------------------------------------------

def reward_scale_for(reward_fun, bg_range=(40.0, 400.0)) -> float:
    """1 / (worst per-step |reward| over ``bg_range``) so rewards land in ~[-1, 1].

    Magni risk spans ~0 (in range) to ~80 (BG 40) to ~400 (BG 10). Unscaled,
    the old terminal penalty of 100 was worth about ONE step of severe hypo, so
    crashing the patient was barely discouraged. Normalizing first lets the
    terminal penalty be set in a principled way (see GlucoseControlWrapper).
    """
    if reward_fun is None:
        return 1.0
    grid = np.linspace(bg_range[0], bg_range[1], 200)
    worst = max(abs(float(reward_fun([bg]))) for bg in grid)
    return 1.0 / worst if worst > 0 else 1.0


def action_to_multiplier(a, max_multiplier: float):
    """Map a policy output in [-1, 1] to a multiple of the patient's basal rate.

    Piecewise linear, chosen so the network's *neutral* output is safe:
        a = -1 -> 0x basal (suspend)    a = 0 -> 1x basal (prescribed rate)
        a = +1 -> max_multiplier x basal
    With the old absolute [0, 0.1] U/min range SAC's neutral output (0.05) was
    ~10x a child's basal rate -- an overdose before learning even started.
    """
    a = float(np.clip(np.asarray(a, dtype=np.float64).ravel()[0], -1.0, 1.0))
    return 1.0 + a if a <= 0.0 else 1.0 + a * (max_multiplier - 1.0)


class GlucoseControlWrapper(gym.Wrapper):
    """v2 state/action/reward design. Differences from GlucoseTrendWrapper:

    * **Patient-relative action.** The agent outputs a multiple of the patient's
      prescribed basal rate (known to any real pump) instead of absolute U/min,
      so one policy means the same thing for a 24 kg child and a 111 kg adult.
    * **Longer CGM history** (default 20 readings = 1 h) + least-squares trend.
    * **Insulin on board** from a 2-compartment subcutaneous insulin model
      (time constant ``iob_tau`` min, ~peak at tau, ~5*tau duration), expressed
      relative to the basal steady state. 4 past doses (12 min) cannot capture
      insulin that keeps acting for ~5 h.
    * **Time of day** (sin/cos): simglucose meals cluster around breakfast,
      lunch and dinner, so the clock lets the policy anticipate them.
    * **Normalized reward + dominant terminal penalty.** Per-step reward is
      scaled into ~[-1, 1] and clipped at -1, so the discounted cost of
      surviving can never exceed 1/(1-gamma); the terminal penalty defaults to
      exactly that, making a crash strictly worse than any survivable day.
    If a ``ShieldWrapper`` sits below this wrapper, the insulin history / IOB
    track the dose actually *delivered* (after any safety suspension).
    """

    CGM_SCALE = CGM_SCALE
    N_INSULIN_HIST = 4

    def __init__(self, env, history_len: int = 20, max_multiplier: float = 5.0,
                 reward_scale: float = 1.0, terminal_penalty: float | None = None,
                 gamma: float = 0.99, iob_tau: float = 55.0):
        super().__init__(env)
        self.history_len = int(history_len)
        self.max_multiplier = float(max_multiplier)
        self.reward_scale = float(reward_scale)
        self.terminal_penalty = (round(1.0 / (1.0 - gamma), 6) if terminal_penalty is None
                                 else float(terminal_penalty))
        self.iob_tau = float(iob_tau)

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        obs_dim = self.history_len + 1 + self.N_INSULIN_HIST + 1 + 2
        self.observation_space = gym.spaces.Box(low=-5.0, high=5.0, shape=(obs_dim,), dtype=np.float32)

        self.cgm_hist = deque(maxlen=self.history_len)
        self.mult_hist = deque(maxlen=self.N_INSULIN_HIST)
        self.basal = 0.0
        self.iob = np.zeros(2)
        self.minute_of_day = 0.0

    # -- helpers -------------------------------------------------------------
    def _iob_rel(self) -> float:
        return float(self.iob.sum() / (2.0 * self.iob_tau * self.basal)) if self.basal > 0 else 0.0

    def _update_iob(self, insulin_u_per_min: float, minutes: float):
        for _ in range(int(round(minutes))):  # 1-min Euler steps, stable for tau >> 1
            x1, x2 = self.iob
            self.iob = np.array([x1 + insulin_u_per_min - x1 / self.iob_tau,
                                 x2 + (x1 - x2) / self.iob_tau])

    def _slope(self) -> float:
        y = np.asarray(list(self.cgm_hist)[-5:], dtype=float)
        if len(y) < 2:
            return 0.0
        return float(np.polyfit(np.arange(len(y)) * 3.0, y, 1)[0])  # mg/dL per min

    def _set_clock(self, info):
        t = info.get("time") if info else None
        if t is not None:
            self.minute_of_day = t.hour * 60 + t.minute + t.second / 60.0

    def _build_obs(self) -> np.ndarray:
        ang = 2 * np.pi * self.minute_of_day / 1440.0
        obs = np.concatenate([
            np.asarray(self.cgm_hist, dtype=np.float32) / self.CGM_SCALE,
            [np.clip(self._slope() / 4.0, -2.0, 2.0)],
            np.asarray(self.mult_hist, dtype=np.float32) / self.max_multiplier,
            [self._iob_rel() / self.max_multiplier],
            [np.sin(ang), np.cos(ang)],
        ]).astype(np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    # -- gym API -------------------------------------------------------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.patient_name = _sim(self.env).env.patient.name
        self.basal = basal_rate(self.patient_name)
        # simglucose patients start at steady state under their basal rate.
        self.iob = np.full(2, self.basal * self.iob_tau)
        cgm0 = float(obs[0])
        self.cgm_hist.clear(); self.mult_hist.clear()
        for _ in range(self.history_len):
            self.cgm_hist.append(cgm0)
        for _ in range(self.N_INSULIN_HIST):
            self.mult_hist.append(1.0)
        self._set_clock(info)
        info = dict(info, patient_basal=self.basal)
        return self._build_obs(), info

    def step(self, action):
        proposed = action_to_multiplier(action, self.max_multiplier) * self.basal
        obs, reward, terminated, truncated, info = self.env.step(np.array([proposed], dtype=np.float32))
        insulin = float(info.get("insulin_delivered", proposed))

        self.cgm_hist.append(float(obs[0]))
        self.mult_hist.append(insulin / self.basal)
        self._update_iob(insulin, info.get("sample_time", 3.0))
        self._set_clock(info)

        reward = max(float(reward) * self.reward_scale, -1.0)
        if terminated:
            reward -= self.terminal_penalty
        info = dict(info, insulin=insulin, iob_rel=self._iob_rel())
        return self._build_obs(), reward, terminated, truncated, info


def make_glucose_env(env_id, patient_name="adolescent#001", reward_fun=None,
                     env_seed=None, history_len=None, max_basal=DEFAULT_MAX_BASAL,
                     terminal_penalty=None, max_episode_steps=STEPS_PER_DAY,
                     env_version="v1", fresh_episodes=None, fixed_episodes=None,
                     max_multiplier=5.0, gamma=0.99, shield=False):
    """Build the full agent environment, identically for training and evaluation.

    Stacks: simglucose env -> episode control -> state/action/reward wrapper.

    * ``env_version="v1"`` reproduces the original setup exactly
      (GlucoseTrendWrapper, absolute [0, max_basal] action, history 4,
      terminal penalty 100). ``"v2"`` uses GlucoseControlWrapper.
    * ``fresh_episodes``: rebuild the simulator every reset (new meal day). Default:
      on for a list of patients (as before); pass True for single-patient training.
    * ``fixed_episodes``: list of ``(patient, seed)`` days to cycle through
      deterministically (validation). Overrides ``fresh_episodes``.

    Using one factory everywhere guarantees train/eval use the exact same setup.
    """
    env = make_env(env_id=env_id, patient_name=patient_name, reward_fun=reward_fun,
                   env_seed=env_seed, max_episode_steps=max_episode_steps)
    if fixed_episodes:
        env = FixedEpisodesWrapper(env, fixed_episodes)
    elif fresh_episodes or (fresh_episodes is None and isinstance(patient_name, (list, tuple))):
        env = FreshEpisodeWrapper(env)
    if shield:
        env = ShieldWrapper(env)   # acts on raw U/min, below the agent wrapper

    if env_version == "v1":
        env = GlucoseTrendWrapper(env, history_len=history_len or 4,
                                  terminal_penalty=100.0 if terminal_penalty is None else terminal_penalty,
                                  max_basal=max_basal)
    elif env_version == "v2":
        env = GlucoseControlWrapper(env, history_len=history_len or 20,
                                    max_multiplier=max_multiplier,
                                    reward_scale=reward_scale_for(reward_fun),
                                    terminal_penalty=terminal_penalty, gamma=gamma)
    else:
        raise ValueError(f"unknown env_version {env_version!r} (use 'v1' or 'v2')")
    return env
