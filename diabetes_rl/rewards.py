"""
Reward functions for glucose control.

A reward function takes ``BG_last_hour`` -- a list of the most recent CGM
readings (mg/dL) within the last hour -- and returns a scalar reward.
simglucose calls it once per environment step.

Why reward design is THE core of this project
---------------------------------------------
Hypoglycemia (low blood glucose, < 70 mg/dL) is *acutely* dangerous: it can
cause seizures or loss of consciousness within minutes. Hyperglycemia (high,
> 180 mg/dL) is harmful mostly over the long term. So a good reward must be
**asymmetric** -- punish lows much harder than highs. A naive "stay near 100"
reward tends to produce agents that over-deliver insulin and cause lows.

Tuning these functions, and the state you feed the agent, is where almost all
of the real work (and the portfolio story) lives.
"""
from __future__ import annotations

import numpy as np

# Standard clinical glucose targets (mg/dL).
TARGET_LOW = 70
TARGET_HIGH = 180
SEVERE_HYPO = 54


def magni_risk(bg: float) -> float:
    """Magni et al. (2007) glycemic risk for a single BG value.

    Symmetric in a transformed (log) space, which makes the risk grow fast
    toward hypoglycemia. Higher = more dangerous. Always >= 0.
    """
    bg = max(float(bg), 1.0)  # guard against log(0)
    f = 3.5506 * (np.log(bg) ** 0.8353 - 3.7932)
    return 10.0 * f * f


def magni_reward(BG_last_hour) -> float:
    """Negative Magni risk of the most recent reading (maximize => safer)."""
    return -magni_risk(BG_last_hour[-1])


HYPO_WEIGHT = 3.0  # extra multiplier on the risk of readings below TARGET_LOW


def magni_hypo_reward(BG_last_hour) -> float:
    """Magni risk with the hypoglycemia region penalized ``HYPO_WEIGHT`` x harder.

    The plain Magni curve already grows faster toward lows, but our pooled agents
    still over-dose into hypoglycemia on 2 of 3 seeds. Multiplying the risk below
    70 mg/dL makes a low unambiguously the worst place to be, which should
    discourage the over-dosing that drives the seed-to-seed variance.
    """
    bg = float(BG_last_hour[-1])
    risk = magni_risk(bg)
    if bg < TARGET_LOW:
        risk *= HYPO_WEIGHT
    return -risk


def zone_reward(BG_last_hour) -> float:
    """Interpretable asymmetric piecewise reward.

    +1 inside the target range, steep penalties for hypoglycemia, milder
    penalties for hyperglycemia. Easy to explain when presenting results.
    """
    bg = float(BG_last_hour[-1])
    if TARGET_LOW <= bg <= TARGET_HIGH:
        return 1.0
    if bg < TARGET_LOW:
        if bg < SEVERE_HYPO:
            return -3.0 - (SEVERE_HYPO - bg) / 10.0
        return -1.0 - (TARGET_LOW - bg) / 15.0
    # hyperglycemia: penalize, but more gently than lows
    return -(bg - TARGET_HIGH) / 100.0


# Convenient lookup so scripts can select a reward by name.
REWARD_FUNCTIONS = {
    "magni": magni_reward,
    "magni_hypo": magni_hypo_reward,
    "zone": zone_reward,
}
