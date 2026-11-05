"""Fast checks for the env plumbing, splits, safety shield and metrics.

Run from the project root:  python -m pytest -q
(A few tests step the simulator for a couple of simulated hours; the whole
file takes well under a minute.)
"""
from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import pytest

from diabetes_rl.baselines import BasalBolusController, BasalController
from diabetes_rl.cohorts import ALL_PATIENTS, stratified_holdout, train_val_test_split
from diabetes_rl.envs import make_env
from diabetes_rl.metrics import glycemic_metrics
from diabetes_rl.patients import basal_rate
from diabetes_rl.rewards import REWARD_FUNCTIONS
from diabetes_rl.safety import SafetyShield
from diabetes_rl.wrappers import (FreshEpisodeWrapper, GlucoseControlWrapper, GlucoseTrendWrapper,
                                  action_to_multiplier, make_glucose_env, reward_scale_for)

_uid = iter(range(10_000))


def _id():
    return f"simglucose/test-{next(_uid)}-v0"


def _cgm_trace(env, action, n=40):
    obs, info = env.reset()
    out = [float(np.ravel(obs)[0])]
    for _ in range(n):
        obs, *_ = env.step(action)
        out.append(float(np.ravel(obs)[0]))
    return np.array(out), info


# ---------------------------------------------------------------- splits
def test_three_way_split_is_disjoint_and_keeps_test_set():
    train, val, test = train_val_test_split(2, 1, seed=0)
    assert (len(train), len(val), len(test)) == (21, 3, 6)
    assert not (set(train) & set(val)) and not (set(train) & set(test)) and not (set(val) & set(test))
    assert sorted(train + val + test) == sorted(ALL_PATIENTS)
    assert test == stratified_holdout(2, seed=0)[1]  # comparable with earlier studies


def test_split_without_validation_patients():
    train, val, test = train_val_test_split(2, 0, seed=0)
    assert val == [] and len(train) == 24


# ---------------------------------------------------------------- metrics
def test_metrics_basic_and_gri():
    bg = [50] * 10 + [100] * 80 + [300] * 10
    m = glycemic_metrics(bg)
    assert m["time_in_range_pct"] == pytest.approx(80)
    assert m["time_severe_hypo_pct"] == pytest.approx(10)
    assert m["time_severe_hyper_pct"] == pytest.approx(10)
    assert m["gri"] == pytest.approx(3.0 * 10 + 1.6 * 10)
    assert m["lbgi"] > 0 and m["hbgi"] > 0


# ---------------------------------------------------------------- action / reward design
def test_action_mapping_is_safe_at_neutral():
    assert action_to_multiplier(-1.0, 5.0) == 0.0          # suspend
    assert action_to_multiplier(0.0, 5.0) == 1.0           # prescribed basal
    assert action_to_multiplier(1.0, 5.0) == 5.0
    xs = np.linspace(-1, 1, 41)
    ys = [action_to_multiplier(x, 5.0) for x in xs]
    assert np.all(np.diff(ys) > 0)


@pytest.mark.parametrize("name", list(REWARD_FUNCTIONS))
def test_reward_scale_normalizes_into_unit_range(name):
    f = REWARD_FUNCTIONS[name]
    s = reward_scale_for(f)
    vals = [f([bg]) * s for bg in np.linspace(40, 400, 50)]
    assert min(vals) >= -1.0 - 1e-9 and max(vals) <= 1.0 + 1e-9


# ---------------------------------------------------------------- safety shield
def test_shield_suspends_when_low_or_falling_fast():
    sh = SafetyShield()
    for v in (150, 150, 150, 150, 150):
        sh.observe(v)
    assert sh.filter(0.02)[1] is False                     # stable, in range
    sh.reset()
    for v in (68, 68, 68):
        sh.observe(v)
    dose, suspended = sh.filter(0.02)
    assert suspended and float(dose) == 0.0                # already low
    sh.reset()
    for v in (130, 120, 110, 100, 90):                     # -3.3 mg/dL/min -> ~-10 in 30 min
        sh.observe(v)
    assert sh.filter(0.02)[1] is True                      # projected low


# ---------------------------------------------------------------- baselines
def test_basal_and_basal_bolus_controllers():
    assert float(BasalController("child#008").act(120.0)[0]) == pytest.approx(basal_rate("child#008"))
    bb = BasalBolusController("adult#001")
    assert float(bb.act(120.0, {"meal": 0.0})[0]) == pytest.approx(basal_rate("adult#001"))
    assert float(bb.act(120.0, {"meal": 10.0, "sample_time": 3.0})[0]) > basal_rate("adult#001")


def test_basal_pid_doses_around_prescribed_basal():
    from diabetes_rl.baselines import BasalPIDController
    b = basal_rate("child#008")
    c = BasalPIDController("child#008", target=120.0, kp=0.01, ki=0.0, kd=0.0, max_multiplier=5.0)
    assert float(c.act(120.0)[0]) == pytest.approx(b, rel=1e-6)      # at target -> basal
    c.reset()
    assert float(c.act(220.0)[0]) == pytest.approx(2.0 * b, rel=1e-6)  # +100 mg/dL -> 2x basal
    c.reset()
    assert float(c.act(10.0)[0]) == 0.0                                # far below -> suspend
    c.reset()
    assert float(c.act(5000.0)[0]) == pytest.approx(5.0 * b, rel=1e-6)  # capped at max_multiplier


def test_basal_rates_vary_across_population():
    rates = [basal_rate(p) for p in ALL_PATIENTS]
    assert max(rates) / min(rates) > 5  # why absolute action ranges are a problem


# ---------------------------------------------------------------- episode control
def test_plain_reset_replays_the_same_day_but_fresh_wrapper_does_not():
    a = np.array([basal_rate("child#008")], np.float32)
    env = make_env(env_id=_id(), patient_name="child#008", env_seed=1)
    t1, i1 = _cgm_trace(env, a)
    t2, i2 = _cgm_trace(env, a)
    assert np.allclose(t1, t2) and i1["time"] == i2["time"]  # simglucose quirk
    fresh = FreshEpisodeWrapper(make_env(env_id=_id(), patient_name="child#008", env_seed=1))
    starts = {str(fresh.reset()[1]["time"]) for _ in range(4)}
    assert len(starts) > 1


def test_fixed_episode_matches_benchmark_day():
    a = np.array([basal_rate("adult#005")], np.float32)
    bench = make_env(env_id=_id(), patient_name="adult#005", env_seed=7)
    ref, _ = _cgm_trace(bench, a)
    fixed = make_glucose_env(_id(), patient_name="child#001",
                             fixed_episodes=[("adult#005", 7)], env_version="v1")
    # v1 wrapper normalizes CGM; undo it and compare to the raw benchmark day
    obs, _ = fixed.reset()
    out = [float(obs[3]) * 400.0]
    for _ in range(40):
        obs, *_ = fixed.step(np.array([a[0]], np.float32))
        out.append(float(obs[3]) * 400.0)
    assert np.allclose(ref, out, atol=1e-3)


# ---------------------------------------------------------------- fast simulator
def test_fastsim_matches_original_simulator():
    from simglucose.patient.t1dpatient import T1DPatient
    from diabetes_rl import fastsim
    from diabetes_rl.baselines import BasalBolusController

    assert fastsim.enable() and fastsim.ORIGINAL_STEP is not None

    def day(step_fn):
        T1DPatient.step, patched = step_fn, T1DPatient.step
        try:
            env = make_env(env_id=_id(), patient_name="child#009", env_seed=0)
            ctrl = BasalBolusController("child#009")
            obs, info = env.reset()
            bg = []
            for _ in range(160):  # 8 h incl. a meal + bolus
                obs, _, term, _, info = env.step(ctrl.act(float(obs[0]), info))
                bg.append(info["bg"])
                if term:
                    break
            return np.array(bg)
        finally:
            T1DPatient.step = patched

    original = day(fastsim.ORIGINAL_STEP)
    fast = day(T1DPatient.step)
    assert len(original) == len(fast)
    assert np.max(np.abs(original - fast)) < 1e-3  # mg/dL


# ---------------------------------------------------------------- env versions
def test_v1_default_is_unchanged():
    env = make_glucose_env(_id(), patient_name="adolescent#001")
    assert isinstance(env, GlucoseTrendWrapper)
    assert env.observation_space.shape == (8,)
    assert float(env.action_space.high[0]) == pytest.approx(0.1)
    assert env.terminal_penalty == 100.0


def test_v2_neutral_action_delivers_basal_and_iob_is_steady():
    env = make_glucose_env(_id(), patient_name="child#008", env_seed=2, env_version="v2",
                           reward_fun=REWARD_FUNCTIONS["magni"])
    assert isinstance(env, GlucoseControlWrapper)
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    for _ in range(40):
        obs, r, term, trunc, info = env.step(np.zeros(1, np.float32))
        assert -1.0 - env.terminal_penalty <= r <= 1.0
    assert info["insulin"] == pytest.approx(basal_rate("child#008"), rel=1e-3)
    assert info["iob_rel"] == pytest.approx(1.0, abs=1e-3)
    assert env.observation_space.contains(obs)


def test_v2_shield_blocks_an_overdose_when_glucose_is_low():
    env = make_glucose_env(_id(), patient_name="child#009", env_seed=0, env_version="v2",
                           reward_fun=REWARD_FUNCTIONS["magni"], shield=True)
    env.reset()
    suspended = False
    for _ in range(200):  # max dose until the shield intervenes
        _, _, term, _, info = env.step(np.ones(1, np.float32))
        if info["shield_suspended"]:
            suspended = True
            assert info["insulin"] == 0.0
            break
        assert not term
    assert suspended


# ---------------------------------------------------------------- residual RL
def test_multiplier_to_action_inverts_action_to_multiplier():
    from diabetes_rl.wrappers import multiplier_to_action
    for m in (0.0, 0.3, 1.0, 2.5, 8.0):
        assert action_to_multiplier(multiplier_to_action(m, 8.0), 8.0) == pytest.approx(m)
    assert multiplier_to_action(12.0, 8.0) == 1.0 and multiplier_to_action(-1.0, 8.0) == -1.0


def test_residual_zero_action_reproduces_the_tuned_pid():
    """Agent action 0 = no correction: the BG trace equals the plain tuned PID's."""
    from diabetes_rl.baselines import BasalPIDController
    from diabetes_rl.wrappers import ResidualPIDWrapper
    patient, seed, n = "adult#005", 1, 120
    raw = make_env(env_id=_id(), patient_name=patient, env_seed=seed)
    pid = BasalPIDController(patient)
    obs, _ = raw.reset()
    pid_bg = []
    for _ in range(n):
        obs, *_rest, info = raw.step(pid.act(float(obs[0])))
        pid_bg.append(info["bg"])
    env = make_glucose_env(_id(), patient_name=patient, env_seed=seed, env_version="v2",
                           reward_fun=REWARD_FUNCTIONS["magni"], residual=0.5, max_multiplier=8.0)
    assert isinstance(env, ResidualPIDWrapper)
    obs, _ = env.reset()
    res_bg = []
    for _ in range(n):
        obs, r, term, trunc, info = env.step(np.zeros(1, np.float32))
        res_bg.append(info["bg"])
        assert info["final_action"] == pytest.approx(info["pid_action"])
    assert np.allclose(res_bg, pid_bg, rtol=1e-4)
    assert env.observation_space.contains(obs)
    assert obs.shape[0] == GlucoseControlWrapper.N_INSULIN_HIST + 20 + 4 + 2  # v2 obs + 2 PID features


def test_residual_scale_bounds_the_correction():
    env = make_glucose_env(_id(), patient_name="child#008", env_seed=0, env_version="v2",
                           reward_fun=REWARD_FUNCTIONS["magni"], residual=0.25, max_multiplier=8.0)
    env.reset()
    for a in (1.0, -1.0, 0.4):
        _, _, _, _, info = env.step(np.array([a], np.float32))
        expected = float(np.clip(info["pid_action"] + 0.25 * a, -1, 1))
        assert info["final_action"] == pytest.approx(expected)


def test_zero_init_policy_head_makes_the_untrained_agent_output_zero():
    from diabetes_rl.algos import make_model, zero_init_policy_head
    env = make_glucose_env(_id(), patient_name="child#008", env_seed=0, env_version="v2",
                           reward_fun=REWARD_FUNCTIONS["magni"], residual=0.5, max_multiplier=8.0)
    for algo in ("sac", "td3", "ppo"):
        model = make_model(algo, env, verbose=0)
        zero_init_policy_head(model)
        obs, _ = env.reset()
        for _ in range(3):
            a, _ = model.predict(obs, deterministic=True)
            assert float(np.abs(a).max()) < 1e-6, algo
            obs, *_ = env.step(np.array([0.7], np.float32))


def test_residual_bb_zero_action_reproduces_basal_bolus():
    """Hybrid closed loop: action 0 = plain meal-announced basal-bolus."""
    from diabetes_rl.wrappers import ResidualBBWrapper
    patient, seed, n = "child#009", 0, 200          # covers at least one meal bolus
    raw = make_env(env_id=_id(), patient_name=patient, env_seed=seed)
    bb = BasalBolusController(patient)
    obs, info = raw.reset()
    bb_bg, boluses = [], 0
    for _ in range(n):
        dose = bb.act(float(obs[0]), info)
        boluses += int(float(dose[0]) > 2 * basal_rate(patient))
        obs, _, _, _, info = raw.step(dose)
        bb_bg.append(info["bg"])
    assert boluses > 0
    env = make_glucose_env(_id(), patient_name=patient, env_seed=seed, env_version="v2",
                           reward_fun=REWARD_FUNCTIONS["magni"], residual=1.0, residual_base="bb")
    assert isinstance(env, ResidualBBWrapper)
    obs, _ = env.reset()
    res_bg = []
    for _ in range(n):
        obs, r, term, trunc, info = env.step(np.zeros(1, np.float32))
        res_bg.append(info["bg"])
        assert info["dose_mult"] == pytest.approx(info["base_mult"], rel=1e-6)
        assert env.observation_space.contains(obs)
    assert np.allclose(res_bg, bb_bg, rtol=1e-4)


def test_residual_bb_scales_the_dose():
    env = make_glucose_env(_id(), patient_name="adult#001", env_seed=0, env_version="v2",
                           reward_fun=REWARD_FUNCTIONS["magni"], residual=1.0, residual_base="bb")
    env.reset()
    _, _, _, _, info = env.step(np.array([-1.0], np.float32))
    assert info["dose_mult"] == 0.0 and info["insulin"] == 0.0            # full suspend
    _, _, _, _, info = env.step(np.array([0.5], np.float32))
    assert info["dose_mult"] == pytest.approx(1.5 * info["base_mult"])
