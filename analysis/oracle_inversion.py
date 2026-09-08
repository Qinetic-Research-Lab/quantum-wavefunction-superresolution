"""Recover ω by matching the 64-point FD ground state to each canonical test input.

Does not modify oracle_baseline.py, canonical artifacts, or the manuscript.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from physics_engine import QuantumSolver  # noqa: E402
from utils import quantum_fidelity  # noqa: E402
from experiment_common import (  # noqa: E402
    TABLE1_BIN_EDGES,
    fidelity_summary,
    fit_omega_from_lr,
    grids,
    load_canonical_test_wavefunctions,
    score_fidelities,
    write_text,
)

OUT_MD = REPO / "outputs" / "oracle_inversion_report.md"
BOUNDS = (0.4, 5.1)
XATOL = 1e-8


def _moment_seed(psi_lr: np.ndarray, x_lr: np.ndarray, dx_lr: float) -> float:
    fit = fit_omega_from_lr(psi_lr, x_lr, dx_lr)
    w0 = fit["omega_moment"] if fit["omega_moment"] is not None else fit["omega_hat"]
    return float(np.clip(w0, BOUNDS[0], BOUNDS[1]))


def recover_omega(solver: QuantumSolver, psi_lr: np.ndarray, dx_lr: float,
                  w_seed: float) -> tuple[float, int, bool, float]:
    """Minimize r(w) = 1 - F(psi_lr, coarse FD ground state at w)."""

    def r(w: float) -> float:
        _, psi_w, _ = solver._solve(solver.n_low, float(w))
        return 1.0 - quantum_fidelity(psi_lr, psi_w, dx_lr)

    r_seed = float(r(w_seed))
    res = minimize_scalar(
        r,
        bounds=BOUNDS,
        method="bounded",
        options={"xatol": XATOL},
    )
    if (not res.success) or (not np.isfinite(res.fun)):
        return w_seed, int(getattr(res, "nfev", 0)) + 1, False, r_seed
    return float(res.x), int(res.nfev) + 1, True, r_seed


def main() -> None:
    lr_t, hr_t, om_t, _ = load_canonical_test_wavefunctions()
    x_lr, _, dx_lr, dx_hr = grids()
    n_test = len(om_t)
    solver = QuantumSolver(L=10.0, n_high=1024, n_low=64, potential="harmonic")

    hats = np.empty(n_test)
    seeds = np.empty(n_test)
    nfevs = np.empty(n_test, dtype=int)
    preds = np.empty_like(hr_t)
    n_fail = 0
    for i in range(n_test):
        w_seed = _moment_seed(lr_t[i], x_lr, dx_lr)
        w_hat, nfev, ok, _ = recover_omega(solver, lr_t[i], dx_lr, w_seed)
        if not ok:
            n_fail += 1
        seeds[i] = w_seed
        hats[i] = w_hat
        nfevs[i] = nfev
        _, psi_hr_hat, _ = solver._solve(solver.n_high, w_hat)
        preds[i] = psi_hr_hat
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{n_test}")

    f_inv = score_fidelities(preds, hr_t, dx_hr)
    abs_err = np.abs(hats - om_t)
    s = fidelity_summary(f_inv)
    seed_shift = np.abs(hats - seeds)

    lines = [
        "# Oracle inversion baseline",
        "",
        "Same canonical 200-sample test slice as `analysis/oracle_baseline.py` "
        "(last 10% of `data_cache/harmonic_n2000_omega0.5-5.0_seed42.npz`). "
        "Stored ω is used only to bin results and to compute |Δω|; it is never "
        "the inversion target.",
        "",
        "For each 64-point input, recover ω by 1-D minimization of",
        "`r(w) = 1 − fidelity(ψ_LR, coarse FD ground state at w)`,",
        "where the coarse state is `QuantumSolver._solve(n_low=64, w)`.",
        f"`scipy.optimize.minimize_scalar` is called with `method='bounded'`, "
        f"`bounds={BOUNDS}`, `xatol={XATOL}`. Bounded Brent has no `x0`; the "
        "existing second-moment estimate is clipped into the interval, evaluated "
        "first, and used as a fallback if the optimizer fails. The 1024-point "
        "FD ground state at the recovered ω is then scored against numerical "
        "`psi_hr` with `utils.quantum_fidelity`.",
        "",
        f"Optimizer failures (fell back to moment seed): {n_fail}/{n_test}",
        f"Function evaluations per sample: mean {nfevs.mean():.1f}, "
        f"max {int(nfevs.max())}",
        "",
        "## ω recovery vs true ω",
        "",
        f"|Δω| mean = {abs_err.mean():.6e}",
        f"|Δω| median = {np.median(abs_err):.6e}",
        f"|Δω| max = {abs_err.max():.6e}",
        f"mean |recovered ω − moment seed| = {seed_shift.mean():.6e}",
        "",
        "## Overall fidelity (n = 200)",
        "",
        "| method | mean | median | min | max |",
        "|---|---|---|---|---|",
        f"| inversion (FD @ recovered ω) | {s['mean']:.6f} | {s['median']:.6f} | "
        f"{s['min']:.6f} | {s['max']:.6f} |",
        "",
        "## Per-ω-bin (Table 1 bins)",
        "",
        "| ω bin | n | mean F | median F | min F | max F | mean infid |",
        "|---|---|---|---|---|---|---|",
    ]
    for a, b in zip(TABLE1_BIN_EDGES, TABLE1_BIN_EDGES[1:]):
        m = (om_t >= a) & (om_t < b)
        f = f_inv[m]
        hi = min(b, 5.0)
        lines.append(
            f"| [{a:.1f}, {hi:.1f}) | {int(m.sum())} | "
            f"{f.mean():.6f} | {np.median(f):.6f} | {f.min():.6f} | "
            f"{f.max():.6f} | {(1.0 - f).mean():.6e} |"
        )
    lines += [
        "",
        "`analysis/oracle_baseline.py` and all canonical seed-42 artifacts were not modified.",
        "",
    ]
    write_text(OUT_MD, "\n".join(lines))
    print(f"wrote {OUT_MD}")
    print(f"inversion mean F = {s['mean']:.6f}")
    print(f"|d omega| mean = {abs_err.mean():.6e}")


if __name__ == "__main__":
    main()
