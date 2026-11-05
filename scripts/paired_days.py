"""
Paired, day-by-day comparison of controllers from benchmark.py CSVs.

Every controller in a benchmark faces the identical simulated day for a given
(patient, seed), so differences can be tested per day (Wilcoxon signed-rank).
Reads a directory with ``_baselines.csv`` (benchmark.py --no-agent ...) and one
CSV per agent configuration (benchmark.py --ensemble-only ...; the ensemble
rows are used, or the per-model mean if there are none), and prints a
Markdown summary table plus paired tests against reference controllers.

Run from the project root, e.g. for the 120-day extended test in the README:
    python scripts/paired_days.py --dir docs/extended_results --ref bpid bb
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METRICS = [("time_in_range_pct", "TIR", True), ("time_hypo_pct", "TBR<70", False),
           ("time_severe_hypo_pct", "TBR<54", False), ("gri", "GRI", False)]
NAMES = {"bpid": "PID scaled to basal, tuned", "bb": "Basal-bolus, meals announced", "pid": "PID (repo)",
         "basal": "Prescribed basal only", "random": "Random"}


def load(directory):
    ctrl = {}
    base = pd.read_csv(os.path.join(directory, "_baselines.csv"))
    for c in base.controller.unique():
        ctrl[c] = base[base.controller == c].set_index(["patient", "seed"])
    for f in sorted(glob.glob(os.path.join(directory, "*.csv"))):
        name = os.path.basename(f)[:-4]
        if name.startswith("_"):
            continue
        df = pd.read_csv(f)
        rows = df[df.controller == "ensemble"]
        if rows.empty:
            rows = df[df.controller == "agent"].groupby(["patient", "seed"]).mean(numeric_only=True).reset_index()
        ctrl[name] = rows.set_index(["patient", "seed"])
    return ctrl


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True)
    p.add_argument("--ref", nargs="+", default=["bpid"], help="reference controllers for the paired tests")
    args = p.parse_args()
    ctrl = load(args.dir)
    n_days = len(next(iter(ctrl.values())))

    print(f"**{n_days} days per controller**\n")
    print("| Controller | " + " | ".join(l for _, l, _ in METRICS) + " | Survived |")
    print("|----" * (len(METRICS) + 2) + "|")
    for k, d in ctrl.items():
        if k == "random":
            continue
        print(f"| {NAMES.get(k, k)} | " + " | ".join(f"{d[m].mean():.1f}" for m, _, _ in METRICS)
              + f" | {100 * d.survived.mean():.0f}% |")

    for ref in args.ref:
        print(f"\n**Paired vs {NAMES.get(ref, ref)}**: mean difference (Wilcoxon p; days better / worse)\n")
        print("| Controller | " + " | ".join(l for _, l, _ in METRICS) + " |")
        print("|----" * (len(METRICS) + 1) + "|")
        for k, d in ctrl.items():
            if k == ref or k in NAMES:
                continue
            idx = d.index.intersection(ctrl[ref].index)
            cells = []
            for m, _, higher_better in METRICS:
                diff = (d.loc[idx, m] - ctrl[ref].loc[idx, m]).values
                pval = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
                better, worse = ((diff > 0).sum(), (diff < 0).sum()) if higher_better else \
                    ((diff < 0).sum(), (diff > 0).sum())
                cells.append(f"{diff.mean():+.2f} (p={pval:.3f}; {better}/{worse})")
            print(f"| {k} (n={len(idx)}) | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
