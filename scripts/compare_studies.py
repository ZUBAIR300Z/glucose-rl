"""
Compare several generalization studies (configurations) side by side.

Reads what run_study.py / benchmark.py write:
    results/study/<tag>/seed<k>.csv              plain evaluation of training seed k
    results/study/<tag>__shield/seed<k>.csv      same model behind the safety shield
    results/study/<tag>[__shield]/ensemble.csv   median-action ensemble of the seeds

Seeds are the unit for an agent's spread: each seed's metrics are averaged over
the held-out test days first, then summarized across seeds (mean, std, worst
seed, and how many seeds meet the consensus hypoglycemia target). Baselines
have no training seed, so they are averaged over the test days directly.

Outputs a Markdown table (stdout + results/study/comparison.md) and a figure
(docs/study_comparison.png by default).

Run from the project root:
    python scripts/compare_studies.py
    python scripts/compare_studies.py --tags v1-sac v2-sac v2-tqc --out docs/study_comparison.png
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDY = os.path.join(ROOT, "results", "study")
SHIELD = "__shield"

ORDER = ["v1-sac", "v2-sac", "v2-sac-m8", "v2-tqc", "v2-crossq", "v2-ppo", "v2-rppo", "v2-tqc-shield",
         "v2-res-sac", "v2-res1-sac", "v2-res-sac-zone", "v2-res-ppo", "v2-res-bb"]
TAG_LABELS = {
    "v1-sac": "SAC, v1 env (original setup)",
    "v2-sac": "SAC, v2 env",
    "v2-sac-m8": "SAC, v2 env, dose cap 8x basal",
    "v2-tqc": "TQC, v2 env",
    "v2-crossq": "CrossQ, v2 env",
    "v2-ppo": "PPO, v2 env",
    "v2-rppo": "RecurrentPPO, v2 env",
    "v2-tqc-shield": "TQC, v2 env, shield in training",
    "v2-res-sac": "Residual SAC on tuned PID (scale 0.5)",
    "v2-res1-sac": "Residual SAC on tuned PID (scale 1.0)",
    "v2-res-sac-zone": "Residual SAC on tuned PID, zone reward",
    "v2-res-bb": "Residual SAC on basal-bolus, meals announced*",
    "v2-res-ppo": "Residual PPO on tuned PID (scale 0.5)",
}
BASELINES = {"pid": "PID (repo)", "bpid": "PID scaled to basal, tuned",
             "basal": "Prescribed basal only", "bb": "Basal-bolus, meals announced*"}
EXTRA_BASELINES = "_baselines"  # results/study/_baselines[__shield]/*.csv (benchmark.py --no-agent)
METRICS = {"tir": "time_in_range_pct", "tbr": "time_hypo_pct", "tbr54": "time_severe_hypo_pct",
           "tar": "time_hyper_pct", "gri": "gri", "surv": "survived"}

# Chart tokens (validated categorical slots 1-2; chrome/ink from the reference palette).
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SEED_C, ENS_C = "#2a78d6", "#eb6834"


def _seed_files(d):
    out = []
    for f in sorted(glob.glob(os.path.join(d, "seed*.csv"))):
        m = re.match(r"seed(\d+)\.csv$", os.path.basename(f))
        if m:
            out.append((int(m.group(1)), f))
    return out


def load_tag(tag, shield=False):
    """-> (agent rows with train_seed, baseline rows, ensemble rows or None)."""
    d = os.path.join(STUDY, tag + (SHIELD if shield else ""))
    frames = []
    for seed, f in _seed_files(d):
        df = pd.read_csv(f)
        df["controller"] = df["controller"].replace({"sac": "agent"})
        frames.append(df.assign(train_seed=seed))
    if not frames:
        return None, None, None
    df = pd.concat(frames, ignore_index=True)
    ens_f = os.path.join(d, "ensemble.csv")
    ens = pd.read_csv(ens_f) if os.path.exists(ens_f) else None
    if ens is not None:
        ens = ens[ens.controller == "ensemble"]
    return df[df.controller == "agent"], df[df.controller.isin(BASELINES)], ens


def per_day(df):
    """Mean over training seeds for each (patient, day) -> one value per test day."""
    cols = list(METRICS.values())
    return df.groupby(["patient", "seed"])[cols].mean()


def wilcoxon_vs(a_days, b_days, col):
    from scipy.stats import wilcoxon
    common = a_days.index.intersection(b_days.index)
    d = (a_days.loc[common, col] - b_days.loc[common, col]).to_numpy()
    if len(d) < 6 or not np.any(d != 0):
        return float("nan"), float("nan")
    return float(np.mean(d)), float(wilcoxon(d).pvalue)  # mean paired difference, Wilcoxon p


def summarize(tags, shield):
    rows, plot = [], []
    baseline_rows, pid_days = None, None
    for tag in tags:
        agent, base, ens = load_tag(tag, shield)
        if agent is None or agent.empty:
            continue
        if baseline_rows is None and base is not None and not base.empty:
            baseline_rows = base.drop_duplicates(["controller", "patient", "seed"])
        rows.append((tag, agent, ens))
    extra = sorted(glob.glob(os.path.join(STUDY, EXTRA_BASELINES + (SHIELD if shield else ""), "*.csv")))
    if extra:
        more = pd.concat([pd.read_csv(f) for f in extra], ignore_index=True)
        more = more[more.controller.isin(BASELINES)]
        baseline_rows = more if baseline_rows is None else pd.concat([baseline_rows, more], ignore_index=True)
    if baseline_rows is None:
        return None, None
    baseline_rows = baseline_rows.drop_duplicates(["controller", "patient", "seed"])
    pid_days = per_day(baseline_rows[baseline_rows.controller == "pid"])
    bpid_rows = baseline_rows[baseline_rows.controller == "bpid"]
    bpid_days = per_day(bpid_rows) if not bpid_rows.empty else None

    def vs_bpid(days):
        if bpid_days is None:
            return "-"
        m1, p1 = wilcoxon_vs(days, bpid_days, METRICS["tir"])
        m2, p2 = wilcoxon_vs(days, bpid_days, METRICS["tbr"])
        return _fmt_test(m1, p1, m2, p2)

    table, fig_rows = [], []
    for c, label in BASELINES.items():
        b = baseline_rows[baseline_rows.controller == c]
        if b.empty:
            continue
        v = {k: b[col].mean() for k, col in METRICS.items()}
        table.append(dict(config=label, seeds="-", tir=f"{v['tir']:.1f}", tbr=f"{v['tbr']:.1f}",
                          tbr54=f"{v['tbr54']:.1f}", gri=f"{v['gri']:.0f}", surv=f"{100 * v['surv']:.0f}%",
                          target="-", vs_pid="-", vs_bpid="-"))
        fig_rows.append(dict(label=label, kind="baseline", tir=[v["tir"]], tbr=[v["tbr"]]))

    for tag, agent, ens in rows:
        seeds = agent.groupby("train_seed")[list(METRICS.values())].mean()
        tir, tbr = seeds[METRICS["tir"]], seeds[METRICS["tbr"]]
        med_tir, p_tir = wilcoxon_vs(per_day(agent), pid_days, METRICS["tir"])
        med_tbr, p_tbr = wilcoxon_vs(per_day(agent), pid_days, METRICS["tbr"])
        table.append(dict(
            config=TAG_LABELS.get(tag, tag), seeds=str(len(seeds)),
            tir=f"{tir.mean():.1f} ± {tir.std(ddof=0):.1f}", tbr=f"{tbr.mean():.1f} ± {tbr.std(ddof=0):.1f}",
            tbr54=f"{seeds[METRICS['tbr54']].mean():.1f}", gri=f"{seeds[METRICS['gri']].mean():.0f}",
            surv=f"{100 * seeds[METRICS['surv']].mean():.0f}%",
            target=f"{int((tbr < 4.0).sum())}/{len(seeds)}",
            vs_pid=_fmt_test(med_tir, p_tir, med_tbr, p_tbr), vs_bpid=vs_bpid(per_day(agent))))
        fig_rows.append(dict(label=TAG_LABELS.get(tag, tag), kind="agent",
                             tir=list(tir), tbr=list(tbr)))
        if ens is not None and not ens.empty:
            e = {k: ens[col].mean() for k, col in METRICS.items()}
            e_days = per_day(ens)
            m1, p1 = wilcoxon_vs(e_days, pid_days, METRICS["tir"])
            m2, p2 = wilcoxon_vs(e_days, pid_days, METRICS["tbr"])
            label = f"  \u21b3 ensemble of {len(seeds)} seeds (median dose)"
            table.append(dict(config=label, seeds="-", tir=f"{e['tir']:.1f}", tbr=f"{e['tbr']:.1f}",
                              tbr54=f"{e['tbr54']:.1f}", gri=f"{e['gri']:.0f}",
                              surv=f"{100 * e['surv']:.0f}%", target="-", vs_pid=_fmt_test(m1, p1, m2, p2),
                              vs_bpid=vs_bpid(e_days)))
            fig_rows.append(dict(label=label.strip(), kind="ensemble", tir=[e["tir"]], tbr=[e["tbr"]]))
    return pd.DataFrame(table), fig_rows


def by_cohort(tags, shield):
    """Markdown table: mean TIR / TBR per patient cohort (agents averaged over seeds)."""
    groups = []
    base = None
    for tag in tags:
        agent, b, ens = load_tag(tag, shield)
        if agent is None or agent.empty:
            continue
        base = b if base is None else base
        groups.append((TAG_LABELS.get(tag, tag), agent))
        if ens is not None and not ens.empty:
            groups.append((f"{TAG_LABELS.get(tag, tag)}: ensemble", ens))
    extra = sorted(glob.glob(os.path.join(STUDY, EXTRA_BASELINES + (SHIELD if shield else ""), "*.csv")))
    frames = ([base] if base is not None else []) + [pd.read_csv(f) for f in extra]
    if frames:
        allb = pd.concat(frames, ignore_index=True).drop_duplicates(["controller", "patient", "seed"])
        groups = [(lbl, allb[allb.controller == c]) for c, lbl in BASELINES.items()
                  if (allb.controller == c).any()] + groups
    if not groups:
        return ""
    cohorts = ["child", "adolescent", "adult"]
    head = ["Controller"] + [f"{c} TIR / TBR" for c in cohorts]
    lines = ["| " + " | ".join(head) + " |", "|" + "----|" * len(head)]
    for label, df in groups:
        df = df.assign(cohort=df.patient.str.split("#").str[0])
        days = df.groupby(["cohort", "patient", "seed"])[[METRICS["tir"], METRICS["tbr"]]].mean()
        cells = []
        for c in cohorts:
            d = days.loc[c] if c in days.index.get_level_values(0) else None
            cells.append("-" if d is None else f"{d[METRICS['tir']].mean():.0f}% / {d[METRICS['tbr']].mean():.1f}%")
        lines.append("| " + " | ".join([label.replace("*", "")] + cells) + " |")
    return "\n".join(lines)


def _fmt_test(m_tir, p_tir, m_tbr, p_tbr):
    if np.isnan(p_tir):
        return "-"
    star = lambda p: "*" if p < 0.05 else ""
    return f"TIR {m_tir:+.1f} (p={p_tir:.3f}){star(p_tir)}; TBR {m_tbr:+.1f} (p={p_tbr:.3f}){star(p_tbr)}"


def to_markdown(df, title):
    head = ["Controller", "Seeds", "TIR 70-180 % ↑", "TBR <70 % ↓", "TBR <54 % ↓", "GRI ↓",
            "Survived", "Seeds with TBR <4%", "vs PID (repo): mean paired diff",
            "vs tuned basal PID: mean paired diff"]
    lines = [f"**{title}**", "", "Paired over the same patient-days (agents averaged over seeds); "
             "p from a Wilcoxon signed-rank test, * = p < 0.05.", "",
             "| " + " | ".join(head) + " |", "|" + "----|" * len(head)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join([r.config, r.seeds, r.tir, r.tbr, r.tbr54, r.gri, r.surv,
                                         r.target, r.vs_pid, r.vs_bpid]) + " |")
    return "\n".join(lines)


def figure(fig_rows, out, title):
    from matplotlib.lines import Line2D

    n = len(fig_rows)
    height = 0.42 * n + 1.8                       # inches
    fig, axes = plt.subplots(1, 2, figsize=(12.5, height), sharey=True, facecolor=SURFACE)
    fig.subplots_adjust(left=0.225, right=0.955, wspace=0.16,
                        top=1 - 1.05 / height, bottom=0.82 / height)
    ys = np.arange(n)[::-1]
    n_base = sum(r["kind"] == "baseline" for r in fig_rows)
    panels = [("tir", "Time in range 70-180 mg/dL (% of day)  \u2191", 70.0, "target > 70%"),
              ("tbr", "Time below 70 mg/dL (% of day)  \u2193", 4.0, "target < 4%")]
    for ax, (key, xlabel, target, tlabel) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        hi = max(max(r[key]) for r in fig_rows)
        xmax = 100 if key == "tir" else max(10.0, np.ceil(hi / 5) * 5 + 5)
        ax.axvline(target, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.text(target, n - 0.45, f" {tlabel}", color=INK_2, fontsize=8, va="bottom", ha="left")
        if 0 < n_base < n:                         # divider: baselines above, learned below
            ax.axhline(ys[n_base - 1] - 0.5, color=AXIS, lw=0.8, zorder=1)
        lab = ax.get_yaxis_transform()             # x in axes units, y in data units
        for y, r in zip(ys, fig_rows):
            vals = np.asarray(r[key], dtype=float)
            if r["kind"] == "agent":
                ax.plot([vals.min(), vals.max()], [y, y], color=SEED_C, lw=2, alpha=0.35, zorder=2)
                ax.scatter(vals, np.full(len(vals), y), s=46, color=SEED_C, edgecolor=SURFACE,
                           linewidth=1.5, zorder=3)
                m = vals.mean()
                ax.plot([m, m], [y - 0.3, y + 0.3], color=INK, lw=2, solid_capstyle="round", zorder=4)
            else:
                m = float(vals[0])
                color, marker = (ENS_C, "D") if r["kind"] == "ensemble" else (MUTED, "s")
                ax.scatter([m], [y], s=52, color=color, marker=marker, edgecolor=SURFACE,
                           linewidth=1.5, zorder=3)
            ax.text(1.012, y, f"{m:.1f}", transform=lab, color=INK_2, fontsize=8.5,
                    va="center", ha="left", clip_on=False)
        ax.set_xlim(0, xmax)
        ax.set_xlabel(xlabel, color=INK_2, fontsize=9.5)
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([r["label"] for r in fig_rows], color=INK, fontsize=9)
    axes[0].set_ylim(-0.6, n - 0.1)
    fig.text(0.01, 1 - 0.22 / height, title, color=INK, fontsize=11.5, ha="left", va="top")
    handles = [Line2D([], [], ls="", marker="o", ms=7, color=SEED_C, label="one training seed"),
               Line2D([], [], color=INK, lw=2, label="mean over seeds"),
               Line2D([], [], ls="", marker="D", ms=6.5, color=ENS_C, label="ensemble of seeds"),
               Line2D([], [], ls="", marker="s", ms=6.5, color=MUTED, label="baseline (no learning)")]
    fig.legend(handles=handles, loc="upper left", ncol=4, frameon=False, fontsize=9,
               labelcolor=INK_2, bbox_to_anchor=(0.003, 1 - 0.46 / height))
    fig.text(0.01, 0.12 / height, "* told meal sizes (announced meals); every other controller sees CGM only. "
             "Numbers beside each panel: mean.", color=MUTED, fontsize=8, va="bottom")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {out}")


def main():
    global STUDY
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tags", nargs="+", default=None, help="default: every study folder found")
    p.add_argument("--out", default=os.path.join(ROOT, "docs", "study_comparison.png"))
    p.add_argument("--shield-out", default=os.path.join(ROOT, "docs", "study_comparison_shield.png"))
    p.add_argument("--study-dir", default=STUDY)
    args = p.parse_args()
    STUDY = args.study_dir

    found = sorted(d for d in os.listdir(STUDY) if os.path.isdir(os.path.join(STUDY, d))
                   and SHIELD not in d and not d.startswith("_")) if os.path.isdir(STUDY) else []
    tags = args.tags or sorted(found, key=lambda t: (ORDER.index(t) if t in ORDER else len(ORDER), t))

    md = []
    for shield, out in ((False, args.out), (True, args.shield_out)):
        table, fig_rows = summarize(tags, shield)
        if table is None:
            continue
        title = ("Held-out test patients (6 patients x 5 days), "
                 + ("every controller behind the safety shield" if shield else "no shield"))
        md.append(to_markdown(table, title))
        md.append("**By patient cohort** (" + ("shielded" if shield else "no shield") + ")\n\n"
                  + by_cohort(tags, shield))
        figure(fig_rows, out, title)
    if not md:
        sys.exit(f"no study results under {STUDY}")
    text = "\n\n".join(md)
    print(text)
    with open(os.path.join(STUDY, "comparison.md"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
