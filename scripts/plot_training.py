"""
Plot the training learning curve from the EvalCallback log (logs/evaluations.npz).

Produces docs/learning_curve.png: held-out mean reward and episode length vs.
training steps. Episode length climbing to 480 means the agent learned to keep
the patient alive for the full day; rising reward means tighter glucose control.

Run from the project root (after / during training):
    python scripts/plot_training.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(PROJECT_ROOT, "logs", "evaluations.npz")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")


def main():
    if not os.path.exists(LOG):
        sys.exit(f"No eval log at {LOG} -- train first (it appears once the first "
                 f"evaluation runs, ~10k steps in).")

    data = np.load(LOG)
    steps = data["timesteps"]
    reward_mean = data["results"].mean(axis=1)
    reward_std = data["results"].std(axis=1)
    ep_len_mean = data["ep_lengths"].mean(axis=1)

    os.makedirs(DOCS_DIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax1.plot(steps, reward_mean, "-o", color="C0", ms=4, label="held-out mean reward")
    ax1.fill_between(steps, reward_mean - reward_std, reward_mean + reward_std,
                     color="C0", alpha=0.15)
    ax1.set_ylabel("mean episode reward")
    ax1.set_title("GlucoRL — training learning curve (held-out evaluation)")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.2)

    ax2.plot(steps, ep_len_mean, "-o", color="C2", ms=4, label="episode length")
    ax2.axhline(480, ls="--", color="gray", lw=1, label="full day (480 steps)")
    ax2.set_ylabel("steps survived")
    ax2.set_xlabel("training timesteps")
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.2)

    fig.tight_layout()
    out = os.path.join(DOCS_DIR, "learning_curve.png")
    fig.savefig(out, dpi=120)
    print(f"Saved {out}  ({len(steps)} evaluations, up to step {int(steps[-1]):,})")


if __name__ == "__main__":
    main()
