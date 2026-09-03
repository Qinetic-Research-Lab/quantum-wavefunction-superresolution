"""Generate Figure 1 of the paper from the canonical per-sample metrics.

Reads `outputs/analysis/test_metrics_canonical.csv` (written by
`analysis/baseline_comparison.py`) and writes the two-panel error-structure
figure at native column width for a NeurIPS-style 5.5 in text block.

Usage (from the repository root):

    python analysis/baseline_comparison.py   # if the CSV does not exist yet
    python paper/make_figure1.py

Outputs:
    outputs/analysis/figure1.png  (300 dpi)
    outputs/analysis/figure1.pdf  (vector)
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "outputs" / "analysis" / "test_metrics_canonical.csv"
OUT = REPO / "outputs" / "analysis"

BIN_EDGES = [0.5, 1.4, 2.3, 3.2, 4.1, 5.001]
SPLINE_C, CNN_C = "#B2182B", "#2166AC"


def main() -> None:
    rows = list(csv.DictReader(open(CSV)))
    om = np.array([float(r["omega"]) for r in rows])
    ci = 1 - np.array([float(r["F_cnn"]) for r in rows])
    si = 1 - np.array([float(r["F_spline"]) for r in rows])
    order = np.argsort(om)
    om, ci, si = om[order], ci[order], si[order]

    wins = ci < si
    last_spline_win = om[np.where(~wins)[0][-1]]
    i = len(wins) - 1
    while i > 0 and wins[i - 1]:
        i -= 1
    print(f"CNN wins {int(wins.sum())}/{len(wins)}; "
          f"last spline win omega={last_spline_win:.3f}; "
          f"unbroken CNN run from omega={om[i]:.3f}")

    ctr, bc, bs = [], [], []
    for a, b in zip(BIN_EDGES, BIN_EDGES[1:]):
        m = (om >= a) & (om < b)
        if m.sum():
            ctr.append(om[m].mean()); bc.append(ci[m].mean()); bs.append(si[m].mean())

    plt.rcParams.update({"font.size": 6.5, "axes.linewidth": 0.8,
                         "font.family": "DejaVu Sans"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.0),
                                   gridspec_kw={"width_ratios": [1.35, 1]})

    ax1.axvspan(1.95, 2.15, color="0.88", zorder=0)
    ax1.scatter(om, si, s=7, marker="o", facecolors="none",
                edgecolors=SPLINE_C, linewidths=0.5, alpha=.65, zorder=2)
    ax1.scatter(om, ci, s=7, marker="^", facecolors="none",
                edgecolors=CNN_C, linewidths=0.5, alpha=.65, zorder=2)
    ax1.plot(ctr, bs, "-o", color=SPLINE_C, lw=1.3, ms=3.4, zorder=4,
             label="Cubic spline (bin mean)")
    ax1.plot(ctr, bc, "--^", color=CNN_C, lw=1.3, ms=3.4, zorder=4,
             label="CNN (bin mean)")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"oscillator frequency  $\omega$")
    ax1.set_ylabel(r"infidelity  $1-F$")
    ax1.set_xlim(0.3, 5.2)
    ax1.text(2.05, ax1.get_ylim()[1] * 0.55, "crossover", ha="center",
             fontsize=6, color="0.35")
    ax1.legend(frameon=False, fontsize=6, loc="lower right")
    ax1.grid(True, which="major", alpha=.25, lw=.5)
    ax1.set_title("(a) error vs. frequency", fontsize=7.5, loc="left")

    ax2.axhline(1.0, color="0.3", lw=1.0, ls=":")
    ax2.scatter(om, si / ci, s=7, marker="s", facecolors="none",
                edgecolors="#4D4D4D", linewidths=.5, alpha=.7)
    ax2.set_yscale("log")
    ax2.set_xlabel(r"oscillator frequency  $\omega$")
    ax2.set_ylabel(r"spline infidelity / CNN infidelity")
    ax2.set_xlim(0.3, 5.2)
    ax2.text(4.9, 1.25, "CNN lower error", ha="right", fontsize=6, color="0.3")
    ax2.text(4.9, 0.66, "spline lower error", ha="right", fontsize=6, color="0.3")
    ax2.grid(True, which="major", alpha=.25, lw=.5)
    ax2.set_title("(b) paired ratio", fontsize=7.5, loc="left")

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figure1.png", dpi=300)
    fig.savefig(OUT / "figure1.pdf")
    print("wrote", OUT / "figure1.png", "and", OUT / "figure1.pdf")


if __name__ == "__main__":
    main()
