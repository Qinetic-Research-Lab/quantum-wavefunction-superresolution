"""Reproduce every number in the paper from the released pipeline.

Loads the cached dataset (regenerating it if absent), scores the trained
checkpoint against cubic-spline and Fourier zero-padding baselines on the
held-out test split, and writes per-sample metrics plus a summary.

Fidelity is computed with `utils.quantum_fidelity` (the same discrete-sum
overlap used by `main.py`), so values here match `benchmark_results.txt`.

Usage (from the repository root):

    python analysis/baseline_comparison.py

Outputs:
    outputs/analysis/test_metrics_canonical.csv
    outputs/analysis/summary_canonical.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from scipy.interpolate import CubicSpline

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from model import QuantumResNet          # noqa: E402
from utils import quantum_fidelity       # noqa: E402

CACHE = REPO / "data_cache" / "harmonic_n2000_omega0.5-5.0_seed42.npz"
CHECKPOINT = REPO / "outputs" / "models" / "quantum_model.pth"
OUT_DIR = REPO / "outputs" / "analysis"

N_TOTAL, TEST_FRAC = 2000, 0.10          # matches main.py split_three_way
L = 10.0
BIN_EDGES = [0.5, 1.4, 2.3, 3.2, 4.1, 5.001]


def load_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not CACHE.exists():
        # Regenerate deterministically (same parameters as the paper run).
        from physics_engine import QuantumSolver
        rng = np.random.default_rng(42)
        solver = QuantumSolver()
        lr, hr, om = [], [], []
        for _ in range(N_TOTAL):
            s = solver.solve_random_omega(0.5, 5.0, rng)
            lr.append(s["psi_lr"]); hr.append(s["psi_hr"]); om.append(s["omega"])
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(CACHE, psi_lr=np.array(lr), psi_hr=np.array(hr),
                 omegas=np.array(om))
    d = np.load(CACHE)
    return d["psi_lr"], d["psi_hr"], d["omegas"]


def fft_upsample(psi_lr: np.ndarray, n_hi: int) -> np.ndarray:
    spec = np.fft.rfft(psi_lr)
    padded = np.zeros(n_hi // 2 + 1, dtype=complex)
    padded[: len(spec)] = spec
    return np.fft.irfft(padded, n_hi) * (n_hi / len(psi_lr))


def main() -> None:
    psi_lr, psi_hr, omegas = load_dataset()
    n_hi = psi_hr.shape[1]
    x_lr = np.linspace(-L, L, psi_lr.shape[1])
    x_hr = np.linspace(-L, L, n_hi)
    dx = 2 * L / (n_hi - 1)

    n_test = max(1, int(round(N_TOTAL * TEST_FRAC)))
    te = slice(N_TOTAL - n_test, N_TOTAL)   # test = final slice, as in main.py
    lr_t, hr_t, om_t = psi_lr[te], psi_hr[te], omegas[te]

    net = QuantumResNet()
    net.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    net.eval()
    with torch.no_grad():
        pred = net(torch.tensor(lr_t, dtype=torch.float32).unsqueeze(1))
    pred = pred.squeeze(1).numpy()

    f_cnn = np.array([quantum_fidelity(pred[i], hr_t[i], dx)
                      for i in range(n_test)])
    f_spl = np.array([quantum_fidelity(CubicSpline(x_lr, lr_t[i])(x_hr),
                                       hr_t[i], dx) for i in range(n_test)])
    f_fft = np.array([quantum_fidelity(fft_upsample(lr_t[i], n_hi),
                                       hr_t[i], dx) for i in range(n_test)])

    inf_c, inf_s = 1 - f_cnn, 1 - f_spl
    wins = int((inf_c < inf_s).sum())
    lo, hi = om_t < 1.5, om_t >= 3.5
    mid = float((omegas.min() + omegas.max()) / 2)

    summary = {
        "n_test": n_test,
        "cnn": {"mean": f_cnn.mean(), "median": float(np.median(f_cnn)),
                "min": f_cnn.min(), "max": f_cnn.max(),
                "std": f_cnn.std(ddof=0)},
        "spline": {"mean": f_spl.mean(), "median": float(np.median(f_spl)),
                   "min": f_spl.min(), "std": f_spl.std(ddof=0)},
        "fft_mean": f_fft.mean(),
        "wins_overall": wins,
        "wins_low_omega_lt1.5": [int((inf_c[lo] < inf_s[lo]).sum()),
                                 int(lo.sum())],
        "wins_high_omega_ge3.5": [int((inf_c[hi] < inf_s[hi]).sum()),
                                  int(hi.sum())],
        "binomial_p_one_sided": stats.binomtest(
            wins, n_test, 0.5, alternative="greater").pvalue,
        "wilcoxon": dict(zip(("W", "p_two_sided"),
                             stats.wilcoxon(inf_s, inf_c))),
        "infidelity_ratio_overall": inf_s.mean() / inf_c.mean(),
        "infidelity_ratio_high_omega": inf_s[hi].mean() / inf_c[hi].mean(),
        "spearman_cnn_vs_omega": dict(zip(("rho", "p"),
                                          stats.spearmanr(om_t, inf_c))),
        "spearman_cnn_vs_dist": dict(zip(("rho", "p"),
                                         stats.spearmanr(np.abs(om_t - mid),
                                                         inf_c))),
        "spearman_spline_vs_omega": dict(zip(("rho", "p"),
                                             stats.spearmanr(om_t, inf_s))),
        "spearman_spline_vs_dist": dict(zip(("rho", "p"),
                                            stats.spearmanr(np.abs(om_t - mid),
                                                            inf_s))),
        "omega_midpoint": mid,
        "bins": [],
    }
    for a, b in zip(BIN_EDGES, BIN_EDGES[1:]):
        m = (om_t >= a) & (om_t < b)
        summary["bins"].append({
            "range": [a, min(b, 5.0)], "n": int(m.sum()),
            "spline_F": f_spl[m].mean(), "fft_F": f_fft[m].mean(),
            "cnn_F": f_cnn[m].mean(),
            "spline_infid": inf_s[m].mean(), "cnn_infid": inf_c[m].mean(),
            "cnn_wins": int((inf_c[m] < inf_s[m]).sum()),
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "test_metrics_canonical.csv", "w") as f:
        f.write("sample_idx,omega,F_cnn,F_spline,F_fft\n")
        for i in range(n_test):
            f.write(f"{i},{om_t[i]:.6f},{f_cnn[i]:.8f},"
                    f"{f_spl[i]:.8f},{f_fft[i]:.8f}\n")
    with open(OUT_DIR / "summary_canonical.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
