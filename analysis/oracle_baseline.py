"""Task B: oracle-family baseline on the canonical 200-sample test set.

Fits ω from the 64-point input only, then evaluates the analytic HO ground
state on the 1024-point grid. Does not peek at stored ω except for binning.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment_common import (  # noqa: E402
    TABLE1_BIN_EDGES,
    analytic_ho_ground_state,
    bin_stats,
    fidelity_summary,
    fit_omega_from_lr,
    grids,
    load_canonical_test_wavefunctions,
    score_fidelities,
    write_json,
    write_text,
)

OUT_MD = REPO / "outputs" / "oracle_baseline_report.md"
OUT_JSON = REPO / "outputs" / "oracle_baseline_report.json"


def main() -> None:
    lr_t, hr_t, om_t, _ = load_canonical_test_wavefunctions()
    x_lr, x_hr, dx_lr, dx_hr = grids()
    n_test = len(om_t)

    hats = []
    preds = np.empty_like(hr_t)
    methods = []
    for i in range(n_test):
        fit = fit_omega_from_lr(lr_t[i], x_lr, dx_lr)
        hats.append(fit["omega_hat"])
        methods.append(fit["method"])
        preds[i] = analytic_ho_ground_state(x_hr, fit["omega_hat"])
    hats = np.array(hats)
    f_or = score_fidelities(preds, hr_t, dx_hr)

    # Oracle that cheats with true ω (diagnostic only; not the baseline).
    cheat = np.array([analytic_ho_ground_state(x_hr, float(om_t[i])) for i in range(n_test)])
    f_cheat = score_fidelities(cheat, hr_t, dx_hr)

    summary = {
        "n_test": n_test,
        "fit_method_counts": {m: int(sum(1 for x in methods if x == m)) for m in sorted(set(methods))},
        "omega_fit_abs_err": {
            "mean": float(np.mean(np.abs(hats - om_t))),
            "median": float(np.median(np.abs(hats - om_t))),
            "max": float(np.max(np.abs(hats - om_t))),
        },
        "oracle": fidelity_summary(f_or),
        "analytic_at_true_omega_diagnostic": fidelity_summary(f_cheat),
        "bins": bin_stats(om_t, {"oracle": f_or}, TABLE1_BIN_EDGES, cap_hi=5.0),
    }
    write_json(OUT_JSON, summary)

    o = summary["oracle"]
    lines = [
        "# Oracle-family baseline",
        "",
        "For each 64-point test input, ω is **fitted from the coarse wavefunction** "
        "(second-moment identity `⟨x²⟩ = 1/(2ω)` for the HO ground state; "
        "nonlinear least-squares Gaussian fallback if that estimate is unstable). "
        "The analytic ground state `ψ(x) = (ω/π)^{1/4} exp(−ω x² / 2)` is then "
        "evaluated on the 1024-point grid and scored with `utils.quantum_fidelity` "
        "against the **numerical** eigsh high-resolution state.",
        "",
        "Stored ω values are used only to bin results, never to build the prediction.",
        "",
        f"Fit methods used: {summary['fit_method_counts']}",
        f"|Δω| mean/median/max = "
        f"{summary['omega_fit_abs_err']['mean']:.6e} / "
        f"{summary['omega_fit_abs_err']['median']:.6e} / "
        f"{summary['omega_fit_abs_err']['max']:.6e}",
        "",
        "## Overall fidelity (n = 200)",
        "",
        "| method | mean | median | min | max |",
        "|---|---|---|---|---|",
        f"| oracle (fitted ω) | {o['mean']:.6f} | {o['median']:.6f} | "
        f"{o['min']:.6f} | {o['max']:.6f} |",
        f"| analytic @ true ω (diagnostic, not a baseline) | "
        f"{summary['analytic_at_true_omega_diagnostic']['mean']:.6f} | "
        f"{summary['analytic_at_true_omega_diagnostic']['median']:.6f} | "
        f"{summary['analytic_at_true_omega_diagnostic']['min']:.6f} | "
        f"{summary['analytic_at_true_omega_diagnostic']['max']:.6f} |",
        "",
        "## Per-ω-bin (Table 1 bins)",
        "",
        "| ω bin | n | mean F | median F | min F | max F | mean infid |",
        "|---|---|---|---|---|---|---|",
    ]
    # per-bin min/max need a second pass
    for a, b, row in zip(
        TABLE1_BIN_EDGES[:-1], TABLE1_BIN_EDGES[1:], summary["bins"]
    ):
        m = (om_t >= a) & (om_t < b)
        f = f_or[m]
        hi = min(b, 5.0)
        lines.append(
            f"| [{a:.1f}, {hi:.1f}) | {int(m.sum())} | "
            f"{f.mean():.6f} | {np.median(f):.6f} | {f.min():.6f} | "
            f"{f.max():.6f} | {(1 - f).mean():.6e} |"
        )
        row["oracle_median"] = float(np.median(f))
        row["oracle_min"] = float(f.min())
        row["oracle_max"] = float(f.max())
    write_json(OUT_JSON, summary)

    lines += [
        "",
        "CNN/spline Table 1 numbers are unchanged; see `outputs/analysis/summary_canonical.json`.",
        "",
    ]
    write_text(OUT_MD, "\n".join(lines))
    print(f"wrote {OUT_MD}")
    print(f"oracle mean F = {o['mean']:.6f}")


if __name__ == "__main__":
    main()
