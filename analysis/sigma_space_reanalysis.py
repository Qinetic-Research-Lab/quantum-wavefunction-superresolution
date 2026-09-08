"""Task C: re-analyze canonical per-sample CNN infidelity in σ and log-ω space.

Reads existing saved metrics only. Does not retrain or regenerate data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment_common import (  # noqa: E402
    CANONICAL_CSV,
    CANONICAL_JSON,
    TABLE1_BIN_EDGES,
    spearman,
    write_json,
    write_text,
)

OUT_MD = REPO / "outputs" / "sigma_space_reanalysis.md"
OUT_JSON = REPO / "outputs" / "sigma_space_reanalysis.json"


def main() -> None:
    rows = np.genfromtxt(CANONICAL_CSV, delimiter=",", names=True)
    om = rows["omega"]
    inf_c = 1.0 - rows["F_cnn"]
    inf_s = 1.0 - rows["F_spline"]

    with open(CANONICAL_JSON, encoding="utf-8") as f:
        canon = json.load(f)
    mid = float(canon["omega_midpoint"])

    sigma = 1.0 / np.sqrt(om)
    log_om = np.log(om)

    results = {
        "n": int(len(om)),
        "omega_midpoint_source": "outputs/analysis/summary_canonical.json",
        "omega_midpoint": mid,
        "spearman_cnn_infidelity": {
            "vs_omega": spearman(om, inf_c),
            "vs_abs_omega_minus_mid": spearman(np.abs(om - mid), inf_c),
            "vs_sigma": spearman(sigma, inf_c),
            "vs_log_omega": spearman(log_om, inf_c),
        },
        "spearman_spline_infidelity": {
            "vs_omega": spearman(om, inf_s),
            "vs_abs_omega_minus_mid": spearman(np.abs(om - mid), inf_s),
            "vs_sigma": spearman(sigma, inf_s),
            "vs_log_omega": spearman(log_om, inf_s),
        },
    }

    lo = (om >= TABLE1_BIN_EDGES[0]) & (om < TABLE1_BIN_EDGES[1])
    hi = (om >= TABLE1_BIN_EDGES[-2]) & (om < TABLE1_BIN_EDGES[-1])
    inf_lo = float(inf_c[lo].mean())
    inf_hi = float(inf_c[hi].mean())
    ratio = inf_lo / inf_hi
    results["table1_edge_bins"] = {
        "low_range": [TABLE1_BIN_EDGES[0], TABLE1_BIN_EDGES[1]],
        "high_range": [TABLE1_BIN_EDGES[-2], 5.0],
        "n_low": int(lo.sum()),
        "n_high": int(hi.sum()),
        "cnn_infid_low": inf_lo,
        "cnn_infid_high": inf_hi,
        "ratio_low_over_high": ratio,
        "paper_low": 3.20e-4,
        "paper_high": 2.23e-4,
        "paper_ratio_approx": 1.4,
    }
    write_json(OUT_JSON, results)

    sc = results["spearman_cnn_infidelity"]
    ss = results["spearman_spline_infidelity"]
    lines = [
        "# Sigma-space reanalysis of the canonical test set",
        "",
        "Source: `outputs/analysis/test_metrics_canonical.csv` (200 samples). "
        "No retraining, no data generation. ω midpoint taken from "
        f"`summary_canonical.json` ({mid:.10f}), matching the already-reported "
        "Spearman vs `|ω − ω_mid|`.",
        "",
        "## Spearman correlation with CNN infidelity",
        "",
        "| predictor | ρ | p |",
        "|---|---|---|",
        f"| ω | {sc['vs_omega']['rho']:.6f} | {sc['vs_omega']['p']:.6e} |",
        f"| \\|ω − ω_mid\\| | {sc['vs_abs_omega_minus_mid']['rho']:.6f} | "
        f"{sc['vs_abs_omega_minus_mid']['p']:.6e} |",
        f"| σ = 1/√ω | {sc['vs_sigma']['rho']:.6f} | {sc['vs_sigma']['p']:.6e} |",
        f"| log(ω) | {sc['vs_log_omega']['rho']:.6f} | {sc['vs_log_omega']['p']:.6e} |",
        "",
        "## Spearman correlation with spline infidelity (context)",
        "",
        "| predictor | ρ | p |",
        "|---|---|---|",
        f"| ω | {ss['vs_omega']['rho']:.6f} | {ss['vs_omega']['p']:.6e} |",
        f"| \\|ω − ω_mid\\| | {ss['vs_abs_omega_minus_mid']['rho']:.6f} | "
        f"{ss['vs_abs_omega_minus_mid']['p']:.6e} |",
        f"| σ = 1/√ω | {ss['vs_sigma']['rho']:.6f} | {ss['vs_sigma']['p']:.6e} |",
        f"| log(ω) | {ss['vs_log_omega']['rho']:.6f} | {ss['vs_log_omega']['p']:.6e} |",
        "",
        "## Table 1 edge-bin CNN infidelity asymmetry",
        "",
        f"- Low bin [0.5, 1.4): mean CNN infidelity = {inf_lo:.6e} "
        f"(n = {int(lo.sum())}; paper 3.20e-4)",
        f"- High bin [4.1, 5.0): mean CNN infidelity = {inf_hi:.6e} "
        f"(n = {int(hi.sum())}; paper 2.23e-4)",
        f"- Ratio low/high = {ratio:.4f} (paper ~1.4×)",
        "",
        "The already-reported vs-ω and vs-|ω−ω_mid| CNN Spearman values are "
        "reproduced from the same CSV; any tiny difference vs the JSON would "
        "indicate a read error (none expected).",
        "",
    ]
    write_text(OUT_MD, "\n".join(lines))
    print(f"wrote {OUT_MD}")
    print(f"sigma rho = {sc['vs_sigma']['rho']:.6f}")
    print(f"log-omega rho = {sc['vs_log_omega']['rho']:.6f}")
    print(f"low/high infid ratio = {ratio:.4f}")


if __name__ == "__main__":
    main()
