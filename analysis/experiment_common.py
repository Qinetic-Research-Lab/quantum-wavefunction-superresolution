"""Shared helpers for the addendum experiments (Tasks A–E).

Does not modify the paper-reproduction script `baseline_comparison.py`.
Refuses to write any canonical seed-42 artifact.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from utils import quantum_fidelity  # noqa: E402

L = 10.0
N_LOW, N_HIGH = 64, 1024
TABLE1_BIN_EDGES = [0.5, 1.4, 2.3, 3.2, 4.1, 5.001]
WIDE_BIN_EDGES = [0.3, 0.5, 1.4, 2.3, 3.2, 4.1, 5.0, 7.001]
TEST_FRAC = 0.10

CANONICAL_CACHE = REPO / "data_cache" / "harmonic_n2000_omega0.5-5.0_seed42.npz"
CANONICAL_CSV = REPO / "outputs" / "analysis" / "test_metrics_canonical.csv"
CANONICAL_JSON = REPO / "outputs" / "analysis" / "summary_canonical.json"
CANONICAL_CKPT = REPO / "outputs" / "models" / "quantum_model.pth"

PROTECTED_FILES = frozenset(
    {
        CANONICAL_CSV.resolve(),
        CANONICAL_JSON.resolve(),
        CANONICAL_CKPT.resolve(),
        (REPO / "outputs" / "benchmark_results.txt").resolve(),
        (REPO / "outputs" / "training_log.csv").resolve(),
        CANONICAL_CACHE.resolve(),
    }
)
PROTECTED_DIRS = (
    (REPO / "outputs" / "plots").resolve(),
    (REPO / "outputs" / "legacy_from_results").resolve(),
    (REPO / "outputs" / "models").resolve(),
    (REPO / "outputs" / "analysis").resolve(),
    (REPO / "paper").resolve(),
)


def assert_safe_write(path: Path) -> Path:
    """Raise if *path* would clobber a canonical / manuscript artifact."""
    path = path.resolve()
    if path in PROTECTED_FILES:
        raise RuntimeError(f"Refusing to overwrite canonical file: {path}")
    for d in PROTECTED_DIRS:
        try:
            path.relative_to(d)
        except ValueError:
            continue
        raise RuntimeError(f"Refusing to write inside protected directory {d}: {path}")
    return path


def grids(n_low: int = N_LOW, n_high: int = N_HIGH, L: float = L):
    x_lr = np.linspace(-L, L, n_low)
    x_hr = np.linspace(-L, L, n_high)
    dx_lr = 2.0 * L / (n_low - 1)
    dx_hr = 2.0 * L / (n_high - 1)
    return x_lr, x_hr, dx_lr, dx_hr


def test_slice(n_total: int) -> slice:
    n_test = max(1, int(round(n_total * TEST_FRAC)))
    return slice(n_total - n_test, n_total)


def fft_upsample_old(psi_lr: np.ndarray, n_hi: int) -> np.ndarray:
    """Index-space rFFT zero-padding (the paper's original baseline)."""
    spec = np.fft.rfft(psi_lr)
    padded = np.zeros(n_hi // 2 + 1, dtype=complex)
    padded[: len(spec)] = spec
    return np.fft.irfft(padded, n_hi) * (n_hi / len(psi_lr))


def fft_upsample_fixed(psi_lr: np.ndarray, n_hi: int) -> np.ndarray:
    """Fourier zero-padding on the unique periodic points of an inclusive grid.

    ``linspace(-L, L, N)`` identifies the endpoints under a 2L-periodic
    extension, so the DFT lives on the first N-1 samples. Those are
    zero-padded 63 → 1023 and the endpoint is restored, landing on the
    same 1024-point inclusive HR grid used by the solver.
    """
    n_per_lr = len(psi_lr) - 1
    n_per_hr = n_hi - 1
    spec = np.fft.rfft(psi_lr[:-1])
    padded = np.zeros(n_per_hr // 2 + 1, dtype=complex)
    n_copy = min(len(spec), len(padded))
    padded[:n_copy] = spec[:n_copy]
    interior = np.fft.irfft(padded, n_per_hr) * (n_per_hr / n_per_lr)
    out = np.empty(n_hi, dtype=float)
    out[:-1] = np.real(interior)
    out[-1] = out[0]
    return out


def center_shift_diagnostic(n_low: int = N_LOW, n_high: int = N_HIGH, L: float = L):
    """Rigid index-space shift (length units) at the domain center."""
    x_lr, x_hr, _, _ = grids(n_low, n_high, L)
    i_c = n_low // 2
    j_c = i_c * (n_high // n_low)
    return {
        "lr_center_index": i_c,
        "hr_mapped_index": j_c,
        "x_lr_center": float(x_lr[i_c]),
        "x_hr_mapped": float(x_hr[j_c]),
        "shift": float(x_lr[i_c] - x_hr[j_c]),
        "dx_lr_inclusive": 2.0 * L / (n_low - 1),
        "dx_lr_periodic_wrong": 2.0 * L / n_low,
        "dx_hr_inclusive": 2.0 * L / (n_high - 1),
    }


def load_canonical_test_wavefunctions():
    """Load the canonical 200-sample test split. Never overwrites the cache."""
    if not CANONICAL_CACHE.exists():
        raise FileNotFoundError(
            f"Canonical cache missing: {CANONICAL_CACHE}. "
            "Regenerate only by an explicit, verified path."
        )
    data = np.load(CANONICAL_CACHE)
    psi_lr = data["psi_lr"]
    psi_hr = data["psi_hr"]
    omegas = data["omegas"]
    if len(omegas) != 2000:
        raise RuntimeError(f"Expected 2000 cached samples, got {len(omegas)}")
    te = test_slice(2000)
    lr_t, hr_t, om_t = psi_lr[te], psi_hr[te], omegas[te]

    rows = np.genfromtxt(CANONICAL_CSV, delimiter=",", names=True)
    csv_om = rows["omega"]
    max_diff = float(np.max(np.abs(om_t - csv_om)))
    if max_diff > 1e-6:
        raise RuntimeError(
            f"Cache test omegas do not match {CANONICAL_CSV.name} "
            f"(max |Δω| = {max_diff:.3e}). Stopping."
        )
    return lr_t, hr_t, om_t, omegas


def bin_stats(omegas, fidelities, bin_edges, cap_hi: float | None = None):
    """Per-bin mean fidelity / infidelity. *fidelities* is a dict of arrays."""
    rows = []
    for a, b in zip(bin_edges, bin_edges[1:]):
        m = (omegas >= a) & (omegas < b)
        hi = b if cap_hi is None else min(b, cap_hi)
        row = {"range": [float(a), float(hi)], "n": int(m.sum())}
        if row["n"] == 0:
            for k in fidelities:
                row[f"{k}_F"] = float("nan")
                row[f"{k}_infid"] = float("nan")
            rows.append(row)
            continue
        for k, arr in fidelities.items():
            f = arr[m]
            row[f"{k}_F"] = float(f.mean())
            row[f"{k}_infid"] = float((1.0 - f).mean())
        rows.append(row)
    return rows


def fidelity_summary(arr: np.ndarray) -> dict:
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "std": float(arr.std(ddof=0)),
    }


def crossover_omega(omegas: np.ndarray, f_cnn: np.ndarray, f_spl: np.ndarray) -> dict:
    """Headline crossover = start of the unbroken high-ω CNN-win suffix."""
    order = np.argsort(omegas)
    om = omegas[order]
    inf_c = 1.0 - f_cnn[order]
    inf_s = 1.0 - f_spl[order]
    wins = inf_c < inf_s
    spline_idx = np.where(~wins)[0]
    last_spline_win = float(om[spline_idx[-1]]) if len(spline_idx) else None
    i = len(wins) - 1
    while i > 0 and wins[i - 1]:
        i -= 1
    unbroken = float(om[i]) if (len(wins) and wins[i]) else None
    return {
        "last_spline_win": last_spline_win,
        "unbroken_cnn_from": unbroken,
        "cnn_wins": int(wins.sum()),
        "n": int(len(wins)),
    }


def sign_convention(psi: np.ndarray) -> np.ndarray:
    out = np.asarray(psi, dtype=float).copy()
    if out[np.argmax(np.abs(out))] < 0:
        out = -out
    return out


def analytic_ho_ground_state(x: np.ndarray, omega: float) -> np.ndarray:
    psi = (omega / np.pi) ** 0.25 * np.exp(-0.5 * omega * x**2)
    return sign_convention(psi)


def fit_omega_from_lr(psi_lr: np.ndarray, x_lr: np.ndarray, dx_lr: float) -> dict:
    """Estimate ω from a coarse ground-state sample; never peeks at true ω."""
    psi = sign_convention(psi_lr)
    dens = psi**2
    dens_sum = np.sum(dens) * dx_lr
    if dens_sum <= 0:
        raise RuntimeError("vanishing LR density")
    x2 = float(np.sum(dens * x_lr**2) * dx_lr / dens_sum)
    omega_moment = 1.0 / (2.0 * x2) if x2 > 0 else np.nan
    used = "moment"
    omega = omega_moment

    def _gauss(x, om, amp):
        return amp * np.exp(-0.5 * om * x**2)

    unstable = (
        not np.isfinite(omega_moment)
        or omega_moment <= 0.05
        or omega_moment >= 40.0
    )
    if unstable:
        used = "least_squares"
        p0 = (max(omega_moment, 1.0) if np.isfinite(omega_moment) else 2.0, float(np.max(np.abs(psi))))
        try:
            popt, _ = curve_fit(
                _gauss, x_lr, psi, p0=p0,
                bounds=([1e-3, 0.0], [50.0, np.inf]),
                maxfev=5000,
            )
            omega = float(popt[0])
        except Exception:
            mask = np.abs(psi) > 0.05 * np.max(np.abs(psi))
            if mask.sum() >= 4:
                slope, _ = np.polyfit(x_lr[mask] ** 2, np.log(np.abs(psi[mask])), 1)
                omega = float(-2.0 * slope)
                used = "log_linear"
            else:
                omega = 1.0 if not np.isfinite(omega_moment) else omega_moment
                used = "fallback_default"
    return {
        "omega_hat": float(omega),
        "omega_moment": float(omega_moment) if np.isfinite(omega_moment) else None,
        "method": used,
    }


def cubic_spline_upsample(psi_lr, x_lr, x_hr):
    return CubicSpline(x_lr, psi_lr)(x_hr)


def score_fidelities(pred_list, psi_hr, dx_hr):
    return np.array([quantum_fidelity(pred_list[i], psi_hr[i], dx_hr)
                     for i in range(len(psi_hr))])


def spearman(x, y):
    rho, p = stats.spearmanr(x, y)
    return {"rho": float(rho), "p": float(p)}


def write_json(path: Path, obj) -> None:
    path = assert_safe_write(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path = assert_safe_write(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def fmt(x, digits=6):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{digits}g}" if abs(x) < 1e-3 or abs(x) >= 1e3 else f"{x:.{digits}f}"
