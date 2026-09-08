"""Score CNN / spline / FFT baselines for an isolated training run.

Writes metrics into --output-dir only. Refuses canonical seed-42 paths.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import QuantumResNet  # noqa: E402
from experiment_common import (  # noqa: E402
    TABLE1_BIN_EDGES,
    WIDE_BIN_EDGES,
    assert_safe_write,
    bin_stats,
    crossover_omega,
    cubic_spline_upsample,
    fft_upsample_fixed,
    fft_upsample_old,
    fidelity_summary,
    grids,
    score_fidelities,
    test_slice,
    write_json,
    write_text,
)


def parse_args():
    p = argparse.ArgumentParser(description="Score one isolated training run")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--n-total", type=int, required=True)
    p.add_argument(
        "--bins",
        choices=["table1", "wide"],
        default="table1",
        help="table1 = [0.5,5]; wide = extra [0.3,0.5) and [5,7] edges",
    )
    p.add_argument(
        "--fft",
        choices=["old", "fixed", "both"],
        default="fixed",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output_dir
    assert_safe_write(out / "test_metrics.csv")
    assert_safe_write(out / "baseline_summary.json")

    data = np.load(args.cache)
    psi_lr, psi_hr, omegas = data["psi_lr"], data["psi_hr"], data["omegas"]
    if len(omegas) != args.n_total:
        raise RuntimeError(
            f"cache has {len(omegas)} samples, expected n-total={args.n_total}"
        )
    te = test_slice(args.n_total)
    lr_t, hr_t, om_t = psi_lr[te], psi_hr[te], omegas[te]
    n_test = len(om_t)
    x_lr, x_hr, _, dx_hr = grids()
    n_hi = hr_t.shape[1]

    net = QuantumResNet()
    net.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    net.eval()
    with torch.no_grad():
        pred = net(torch.tensor(lr_t, dtype=torch.float32).unsqueeze(1))
    pred = pred.squeeze(1).numpy()

    f_cnn = score_fidelities(pred, hr_t, dx_hr)
    f_spl = score_fidelities(
        [cubic_spline_upsample(lr_t[i], x_lr, x_hr) for i in range(n_test)],
        hr_t, dx_hr,
    )
    f_fft_old = f_fft_fixed = None
    if args.fft in ("old", "both"):
        f_fft_old = score_fidelities(
            [fft_upsample_old(lr_t[i], n_hi) for i in range(n_test)], hr_t, dx_hr,
        )
    if args.fft in ("fixed", "both"):
        f_fft_fixed = score_fidelities(
            [fft_upsample_fixed(lr_t[i], n_hi) for i in range(n_test)], hr_t, dx_hr,
        )
    f_fft = f_fft_fixed if f_fft_fixed is not None else f_fft_old

    edges = WIDE_BIN_EDGES if args.bins == "wide" else TABLE1_BIN_EDGES
    cap = None if args.bins == "wide" else 5.0
    fid_map = {"cnn": f_cnn, "spline": f_spl}
    if f_fft is not None:
        fid_map["fft"] = f_fft

    summary = {
        "n_test": n_test,
        "cache": str(args.cache),
        "checkpoint": str(args.checkpoint),
        "cnn": fidelity_summary(f_cnn),
        "spline": fidelity_summary(f_spl),
        "crossover": crossover_omega(om_t, f_cnn, f_spl),
        "bins": bin_stats(om_t, fid_map, edges, cap_hi=cap),
    }
    if f_fft_old is not None:
        summary["fft_old"] = fidelity_summary(f_fft_old)
    if f_fft_fixed is not None:
        summary["fft_fixed"] = fidelity_summary(f_fft_fixed)

    csv_path = out / "test_metrics.csv"
    assert_safe_write(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    hdr = "sample_idx,omega,F_cnn,F_spline"
    extras = []
    if f_fft_old is not None:
        extras.append(("F_fft_old", f_fft_old))
        hdr += ",F_fft_old"
    if f_fft_fixed is not None:
        extras.append(("F_fft_fixed", f_fft_fixed))
        hdr += ",F_fft_fixed"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(hdr + "\n")
        for i in range(n_test):
            line = f"{i},{om_t[i]:.6f},{f_cnn[i]:.10f},{f_spl[i]:.10f}"
            for _, arr in extras:
                line += f",{arr[i]:.10f}"
            f.write(line + "\n")

    write_json(out / "baseline_summary.json", summary)

    lines = [
        f"# Baseline scores for `{args.output_dir.as_posix()}`",
        "",
        f"n_test = {n_test}",
        f"CNN mean F = {summary['cnn']['mean']:.6f}",
        f"spline mean F = {summary['spline']['mean']:.6f}",
        f"crossover unbroken CNN-win from ω = {summary['crossover']['unbroken_cnn_from']}",
        f"last spline win ω = {summary['crossover']['last_spline_win']}",
        "",
        "## Per-bin mean infidelity",
        "",
    ]
    for row in summary["bins"]:
        a, b = row["range"]
        lines.append(
            f"- [{a:.2f}, {b:.2f}) n={row['n']}: "
            f"CNN infid={row.get('cnn_infid', float('nan')):.6e}, "
            f"spline infid={row.get('spline_infid', float('nan')):.6e}"
        )
    write_text(out / "baseline_report.md", "\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
