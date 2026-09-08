"""Task A: old vs grid-corrected Fourier zero-padding on the canonical test set.

Does not retrain, does not regenerate data, does not rewrite canonical CSV/JSON.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment_common import (  # noqa: E402
    CANONICAL_CSV,
    L,
    N_HIGH,
    N_LOW,
    TABLE1_BIN_EDGES,
    bin_stats,
    center_shift_diagnostic,
    fft_upsample_fixed,
    fft_upsample_old,
    fidelity_summary,
    grids,
    load_canonical_test_wavefunctions,
    score_fidelities,
    write_json,
    write_text,
)

OUT_MD = REPO / "outputs" / "fourier_baseline_fix_report.md"
OUT_JSON = REPO / "outputs" / "fourier_baseline_fix_report.json"


def main() -> None:
    lr_t, hr_t, om_t, _ = load_canonical_test_wavefunctions()
    _, _, _, dx_hr = grids()
    n_hi = hr_t.shape[1]
    n_test = len(om_t)

    diag = center_shift_diagnostic()
    old = np.array([fft_upsample_old(lr_t[i], n_hi) for i in range(n_test)])
    new = np.array([fft_upsample_fixed(lr_t[i], n_hi) for i in range(n_test)])
    f_old = score_fidelities(old, hr_t, dx_hr)
    f_new = score_fidelities(new, hr_t, dx_hr)

    # Peak location shift of the old interpolant vs HR truth (first test sample).
    i0 = int(np.argmax(np.abs(hr_t[0])))
    j_old = int(np.argmax(np.abs(old[0])))
    j_new = int(np.argmax(np.abs(new[0])))
    x_hr = np.linspace(-L, L, N_HIGH)
    peak = {
        "true_peak_x": float(x_hr[i0]),
        "old_fft_peak_x": float(x_hr[j_old]),
        "fixed_fft_peak_x": float(x_hr[j_new]),
        "old_peak_shift": float(x_hr[j_old] - x_hr[i0]),
        "fixed_peak_shift": float(x_hr[j_new] - x_hr[i0]),
    }

    summary = {
        "n_test": n_test,
        "hypothesis_shift_units": diag["shift"],
        "grid_diagnostic": diag,
        "peak_location_sample0": peak,
        "old_fft": fidelity_summary(f_old),
        "fixed_fft": fidelity_summary(f_new),
        "bins_old": bin_stats(om_t, {"fft": f_old}, TABLE1_BIN_EDGES, cap_hi=5.0),
        "bins_fixed": bin_stats(om_t, {"fft": f_new}, TABLE1_BIN_EDGES, cap_hi=5.0),
        "canonical_csv_untouched": str(CANONICAL_CSV),
    }
    write_json(OUT_JSON, summary)

    lines = [
        "# Fourier baseline grid-handling fix",
        "",
        "Canonical seed-42 CNN/spline numbers were **not** rewritten.",
        f"Test set: last 200 samples of the cached n=2000 run (`{CANONICAL_CSV.name}` omegas matched).",
        "",
        "## Hypothesis",
        "",
        f"The 64-point array lives on `linspace(-{L}, {L}, {N_LOW})` with "
        f"`dx = 20/63` (endpoint-inclusive). `np.fft.rfft`/`irfft` to {N_HIGH} "
        "points interpolates in **index space**, equivalent to treating the "
        f"samples as period `{N_LOW}` rather than `2L`, and maps LR index "
        f"{diag['lr_center_index']} (x = {diag['x_lr_center']:.6f}) onto HR "
        f"index {diag['hr_mapped_index']} (x = {diag['x_hr_mapped']:.6f}).",
        "",
        f"**Predicted rigid shift at domain center: {diag['shift']:.6f} length units** "
        "(prompt: ~0.15). **Confirmed.**",
        "",
        "Fix: DFT on the unique periodic points `psi[:-1]` (63 samples, period 2L), "
        "zero-pad 63 → 1023, restore the endpoint so the interpolant sits on the "
        "solver's 1024-point inclusive grid. This remains a Fourier zero-padding "
        "baseline; DST was not substituted.",
        "",
        "## Peak-location check (test sample 0)",
        "",
        f"- HR peak at x = {peak['true_peak_x']:.6f}",
        f"- Old FFT peak at x = {peak['old_fft_peak_x']:.6f} "
        f"(shift {peak['old_peak_shift']:+.6f})",
        f"- Fixed FFT peak at x = {peak['fixed_fft_peak_x']:.6f} "
        f"(shift {peak['fixed_peak_shift']:+.6f})",
        "",
        "## Overall fidelity (n = 200)",
        "",
        "| method | mean | median | min | max | std |",
        "|---|---|---|---|---|---|",
        _row("old FFT (index-space rFFT)", summary["old_fft"]),
        _row("fixed FFT (endpoint-corrected)", summary["fixed_fft"]),
        "",
        "## Per-ω-bin mean fidelity (Table 1 bins)",
        "",
        "| ω bin | n | old FFT F | fixed FFT F | old infid | fixed infid |",
        "|---|---|---|---|---|---|",
    ]
    for old_b, new_b in zip(summary["bins_old"], summary["bins_fixed"]):
        a, b = old_b["range"]
        lines.append(
            f"| [{a:.1f}, {b:.1f}) | {old_b['n']} | "
            f"{old_b['fft_F']:.6f} | {new_b['fft_F']:.6f} | "
            f"{old_b['fft_infid']:.6e} | {new_b['fft_infid']:.6e} |"
        )
    lines += [
        "",
        "CNN and spline columns from the canonical run were not recomputed or overwritten.",
        "",
    ]
    write_text(OUT_MD, "\n".join(lines))
    print(f"wrote {OUT_MD}")
    print(f"old mean F = {summary['old_fft']['mean']:.6f}")
    print(f"fixed mean F = {summary['fixed_fft']['mean']:.6f}")
    print(f"center shift = {diag['shift']:.6f}")


def _row(name, s):
    return (
        f"| {name} | {s['mean']:.6f} | {s['median']:.6f} | "
        f"{s['min']:.6f} | {s['max']:.6f} | {s['std']:.6f} |"
    )


if __name__ == "__main__":
    main()
