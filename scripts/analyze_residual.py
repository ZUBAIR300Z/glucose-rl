"""
What did the residual agent learn? Runs a residual RL agent (``--residual``) and
its base controller alone (the tuned basal PID, or meal-announced basal-bolus
with ``--residual-base bb``) on the same held-out test days and shows *where*
the agent overrides it.

    a) an example day (--example: the median TIR gain [default], the largest
       drop in time below range, or the largest TIR loss; the panel title says which)
    b) the same day: insulin dose as a multiple of the prescribed basal rate
    c) mean correction = agent dose - PID proposal (x basal), by hour of day
    d) mean correction by the CGM reading at the time of the decision

Several --model paths are combined as a median-action ensemble.

Run from the project root:
    python scripts/analyze_residual.py --model models/pooled/v2-res-sac/best_seed0/best_model
    python scripts/analyze_residual.py --model models/pooled/v2-res-sac/best_seed{0,1,2,3,4}/best_model \\
        --out docs/residual_analysis.png
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from diabetes_rl.algos import RECURRENT, PolicyRunner, load_model
from diabetes_rl.cohorts import train_val_test_split
from diabetes_rl.config import env_kwargs, find_run_config
from diabetes_rl.envs import STEPS_PER_DAY
from diabetes_rl.metrics import glycemic_metrics
from diabetes_rl.wrappers import make_glucose_env

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# validated categorical slots 1-2 (dataviz reference palette) + recessive ink/surface
PID_C, AGENT_C = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CGM_BINS = [0, 70, 100, 140, 180, 250, 1000]
EXAMPLES = {"median": "median TIR gain", "max-tbr-drop": "largest TBR drop", "worst": "largest TIR loss"}


def run_day(cfg, patient, seed, runners=None):
    """One test day; ``runners=None`` -> zero correction (= the base controller alone)."""
    env = make_glucose_env(env_id=f"simglucose/resid-{patient.replace('#', '')}-s{seed}-{bool(runners)}-v0",
                           patient_name=patient, env_seed=seed, **env_kwargs(cfg))
    obs, info = env.reset(seed=seed)
    for r in runners or []:
        r.reset()
    rec = {k: [] for k in ("minute", "cgm", "bg", "base_mult", "dose_mult")}
    survived = True
    for _ in range(STEPS_PER_DAY):
        minute, cgm = env.minute_of_day, float(env.cgm_hist[-1])
        a = (np.median(np.stack([r.act(obs) for r in runners]), axis=0) if runners
             else np.zeros(1, np.float32))
        obs, _, term, trunc, info = env.step(a)
        rec["minute"].append(minute)
        rec["cgm"].append(cgm)
        rec["bg"].append(float(info["bg"]))
        rec["base_mult"].append(info["base_mult"])
        rec["dose_mult"].append(info["dose_mult"])
        if term:
            survived = False
            break
        if trunc:
            break
    env.close()
    out = {k: np.asarray(v) for k, v in rec.items()}
    out["metrics"], out["survived"] = glycemic_metrics(rec["bg"]), survived
    return out


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", nargs="+", required=True)
    p.add_argument("--patients", nargs="+", default=None, help="default: the 6 held-out test patients")
    p.add_argument("--seeds", type=int, default=5, help="days per patient (benchmark seeds 0..k-1)")
    p.add_argument("--example", default="median", choices=list(EXAMPLES),
                   help="which day panels a-b show (panels c-d always pool every day)")
    p.add_argument("--out", default=os.path.join(ROOT, "results", "residual_analysis.png"))
    args = p.parse_args()

    cfg = find_run_config(args.model[0])
    if not cfg.get("residual"):
        sys.exit("not a residual model (config.json has no 'residual' > 0)")
    base = "Tuned PID" if (cfg.get("residual_base") or "pid") == "pid" else "Basal-bolus"
    runners = [PolicyRunner(load_model(cfg["algo"], m), recurrent=cfg["algo"] in RECURRENT)
               for m in args.model]
    patients = args.patients or train_val_test_split(2, 1, seed=0)[2]

    days = []
    for pt in patients:
        for s in range(args.seeds):
            pid, ag = run_day(cfg, pt, s), run_day(cfg, pt, s, runners)
            days.append((pt, s, pid, ag))
            print(f"{pt:>15} day {s}: base TIR {pid['metrics']['time_in_range_pct']:5.1f}% "
                  f"TBR {pid['metrics']['time_hypo_pct']:4.1f}% | agent TIR "
                  f"{ag['metrics']['time_in_range_pct']:5.1f}% TBR {ag['metrics']['time_hypo_pct']:4.1f}%"
                  f"{'' if ag['survived'] else '  (agent day ended early)'}"
                  f"{'' if pid['survived'] else '  (PID day ended early)'}", flush=True)

    def mean(key, who):
        return float(np.mean([(d[2] if who == "pid" else d[3])["metrics"][key] for d in days]))
    gains = np.array([d[3]["metrics"]["time_in_range_pct"] - d[2]["metrics"]["time_in_range_pct"] for d in days])
    ins_ratio = np.mean([d[3]["dose_mult"].mean() / max(d[2]["dose_mult"].mean(), 1e-9) for d in days])
    print(f"\n{len(days)} days, {len(runners)} model(s): {base} TIR {mean('time_in_range_pct', 'pid'):.1f}% "
          f"TBR {mean('time_hypo_pct', 'pid'):.1f}% | residual agent TIR {mean('time_in_range_pct', 'agent'):.1f}% "
          f"TBR {mean('time_hypo_pct', 'agent'):.1f}% | TIR gain per day: median {np.median(gains):+.1f}, "
          f"better on {int((gains > 0).sum())}/{len(days)} days | mean insulin vs PID: {ins_ratio:.2f}x")

    # pooled corrections over every agent step
    minute = np.concatenate([d[3]["minute"] for d in days])
    cgm = np.concatenate([d[3]["cgm"] for d in days])
    corr = np.concatenate([d[3]["dose_mult"] - d[3]["base_mult"] for d in days])

    tbr_drop = np.array([d[2]["metrics"]["time_hypo_pct"] - d[3]["metrics"]["time_hypo_pct"] for d in days])
    pick = {"median": int(np.argmin(np.abs(gains - np.median(gains)))),
            "max-tbr-drop": int(np.argmax(tbr_drop)), "worst": int(np.argmin(gains))}[args.example]
    pt, s, pid, ag = days[pick]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), facecolor=SURFACE)
    (a1, a2), (a3, a4) = axes
    hrs = lambda d: d["minute"][0] / 60.0 + np.arange(len(d["bg"])) * 3 / 60.0  # clock time, unwrapped
    clock = matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(round(v)) % 24:02d}:00")
    style(a1, f"a) Example day: {pt}, day {s} ({EXAMPLES[args.example]})")
    a1.axhspan(70, 180, color="#e9f3ea", zorder=0)
    a1.text(hrs(pid)[-1], 72, "target 70-180 ", color=INK2, fontsize=8, va="bottom", ha="right")
    for d, c, lab in ((pid, PID_C, base), (ag, AGENT_C, "Residual agent")):
        m = d["metrics"]
        a1.plot(hrs(d), d["bg"], color=c, lw=1.8,
                label=f"{lab}: TIR {m['time_in_range_pct']:.0f}%, TBR {m['time_hypo_pct']:.1f}%")
    a1.set_ylabel("blood glucose (mg/dL)")
    a1.set_xlabel("time of day")
    a1.set_ylim(min(50, min(pid["bg"].min(), ag["bg"].min()) - 10),
                1.3 * max(pid["bg"].max(), ag["bg"].max()))
    a1.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left", ncol=1)

    style(a2, "b) Same day: insulin dose")
    for d, c, lab in ((pid, PID_C, base), (ag, AGENT_C, "Residual agent")):
        a2.step(hrs(d), d["dose_mult"], where="post", color=c, lw=1.4, label=lab)
    a2.axhline(1.0, color=INK2, lw=0.8, ls=":")
    if base != "Tuned PID":
        a2.set_yscale("symlog", linthresh=2.0)  # meal boluses are ~50-150x basal
    a2.set_ylabel("dose (x prescribed basal; dotted = 1x)")
    a2.set_xlabel("time of day")
    a2.set_ylim(0, (1.3 if base == "Tuned PID" else 3.0) * max(pid["dose_mult"].max(), ag["dose_mult"].max()))
    a2.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left", ncol=2)
    for ax in (a1, a2):
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(3))
        ax.xaxis.set_major_formatter(clock)

    style(a3, f"c) Mean correction by time of day ({len(days)} test days)")
    hour = (minute // 60).astype(int) % 24
    by_hour = [corr[hour == h].mean() if np.any(hour == h) else 0.0 for h in range(24)]
    a3.bar(np.arange(24), by_hour, width=0.78, color=AGENT_C)
    a3.axhline(0, color=INK2, lw=0.8)
    a3.set_xticks(range(0, 24, 3))
    a3.set_xlabel("hour of day (simglucose meals cluster at ~7, ~12 and ~18-20 h)")
    a3.set_ylabel(f"agent dose - {base} dose (x basal)")

    style(a4, "d) Mean correction by current glucose (CGM)")
    labels, vals, ns = [], [], []
    for lo, hi in zip(CGM_BINS[:-1], CGM_BINS[1:]):
        sel = (cgm >= lo) & (cgm < hi)
        labels.append(f"<{hi}" if lo == 0 else (f">{lo}" if hi == 1000 else f"{lo}-{hi}"))
        vals.append(corr[sel].mean() if sel.any() else 0.0)
        ns.append(int(sel.sum()))
    bars = a4.bar(range(len(vals)), vals, width=0.7, color=AGENT_C)
    a4.axhline(0, color=INK2, lw=0.8)
    a4.set_xticks(range(len(vals)), labels)
    for b, v, n in zip(bars, vals, ns):
        a4.annotate(f"{v:+.2f}\n{100 * n / len(cgm):.0f}% of steps", (b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 4 if v >= 0 else -4), textcoords="offset points", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=7.5, color=INK2)
    lo_, hi_ = min(vals + [0]), max(vals + [0])
    pad = 0.35 * (hi_ - lo_ or 1)
    a4.set_ylim(lo_ - pad, hi_ + pad)
    a4.set_xlabel("CGM at decision time (mg/dL)")
    a4.set_ylabel(f"agent dose - {base} dose (x basal)")

    fig.suptitle(f"What the residual agent changes on top of {base.lower()} "
                 f"({len(runners)} model{'s, median dose' if len(runners) > 1 else ''}; held-out test patients)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=130, facecolor=SURFACE)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
