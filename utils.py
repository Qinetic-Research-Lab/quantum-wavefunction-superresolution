"""
Utility functions for the quantum wavefunction upsampling project.

* L²-consistent normalisation
* Quantum fidelity (squared overlap integral)
* Side-by-side comparison plots saved to disk
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# Normalisation
# -------------------------------------------------------------------

def normalize_wavefunction(psi: np.ndarray, dx: float) -> np.ndarray:
    """Return a copy of *psi* normalised so that ∫|ψ|² dx ≈ Σ|ψ_i|² Δx = 1."""
    norm = np.sqrt(np.sum(psi**2) * dx)
    if norm == 0.0:
        return psi.copy()
    return psi / norm


# -------------------------------------------------------------------
# Quantum fidelity
# -------------------------------------------------------------------

def quantum_fidelity(psi_pred: np.ndarray, psi_hr: np.ndarray,
                     dx: float) -> float:
    """Squared overlap integral between two wavefunctions on the same grid.

    Both inputs are assumed to live on the same uniform grid with spacing
    *dx*.  They are re-normalised internally so that small numerical drift
    in the model output does not artificially deflate the fidelity.

    F = |⟨ψ_pred | ψ_hr⟩|² = |Σ ψ_pred_i · ψ_hr_i · Δx|²
    """
    psi_pred = normalize_wavefunction(psi_pred, dx)
    psi_hr = normalize_wavefunction(psi_hr, dx)
    overlap = np.sum(psi_pred * psi_hr) * dx
    return float(overlap**2)


# -------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------

def save_comparison_plot(
    x_hr: np.ndarray,
    psi_hr: np.ndarray,
    x_lr: np.ndarray,
    psi_lr: np.ndarray,
    psi_pred: np.ndarray,
    omega: float,
    save_path: str | Path,
    fidelity: float | None = None,
) -> None:
    """Plot HR ground truth, LR (interpolated onto HR grid), and ML
    prediction side-by-side and save to *save_path*.

    Parameters
    ----------
    x_hr, psi_hr : high-resolution grid and wavefunction (length 1024)
    x_lr, psi_lr : low-resolution grid and wavefunction (length 64)
    psi_pred     : model prediction on the HR grid (length 1024)
    omega        : harmonic oscillator frequency (for the title)
    save_path    : destination file path (png)
    fidelity     : optional pre-computed fidelity to display on the plot
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    axes[0].plot(x_hr, psi_hr, color="tab:blue", linewidth=1.2)
    axes[0].set_title("High-Res (1024 pts)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel(r"$\psi(x)$")

    axes[1].plot(x_lr, psi_lr, "o-", color="tab:orange", markersize=2, linewidth=0.8)
    axes[1].set_title("Low-Res (64 pts)")
    axes[1].set_xlabel("x")

    pred_label = "ML Prediction"
    if fidelity is not None:
        pred_label += f"  (F = {fidelity:.6f})"
    axes[2].plot(x_hr, psi_pred, color="tab:green", linewidth=1.2, label=pred_label)
    axes[2].plot(x_hr, psi_hr, "--", color="tab:blue", linewidth=0.8,
                 alpha=0.5, label="HR ground truth")
    axes[2].legend(fontsize=8)
    axes[2].set_title("ML vs HR")
    axes[2].set_xlabel("x")

    fig.suptitle(rf"$\omega = {omega:.3f}$", fontsize=13)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def save_error_plot(
    x_hr: np.ndarray,
    psi_hr: np.ndarray,
    psi_pred: np.ndarray,
    omega: float,
    fidelity: float,
    save_path: str | Path,
) -> None:
    """Plot the pointwise absolute error |psi_pred - psi_hr| alongside
    the prediction and ground truth.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    error = np.abs(psi_pred - psi_hr)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(x_hr, psi_hr, color="tab:blue", linewidth=1.2, label="HR truth")
    ax1.plot(x_hr, psi_pred, "--", color="tab:green", linewidth=1.0,
             label="ML prediction")
    ax1.set_ylabel(r"$\psi(x)$")
    ax1.legend(fontsize=9)
    ax1.set_title(rf"$\omega = {omega:.3f}$   |   F = {fidelity:.6f}")

    ax2.fill_between(x_hr, 0, error, color="tab:red", alpha=0.4)
    ax2.plot(x_hr, error, color="tab:red", linewidth=0.8)
    ax2.set_xlabel("x")
    ax2.set_ylabel(r"$|\Delta\psi|$")
    ax2.set_title(f"Max error = {error.max():.4e}   |   "
                  f"Mean error = {error.mean():.4e}")

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def save_fidelity_vs_omega_plot(
    omegas: np.ndarray,
    fidelities: np.ndarray,
    save_path: str | Path,
) -> None:
    """Scatter plot of quantum fidelity vs. omega for the test set."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(omegas, fidelities, s=12, alpha=0.6, edgecolors="none",
               c=fidelities, cmap="RdYlGn", vmin=0.99, vmax=1.0)
    ax.axhline(y=fidelities.mean(), color="tab:blue", linestyle="--",
               linewidth=0.8, label=f"Mean F = {fidelities.mean():.6f}")
    ax.set_xlabel(r"$\omega$", fontsize=12)
    ax.set_ylabel("Quantum Fidelity", fontsize=12)
    ax.set_title("Fidelity vs. Harmonic Frequency")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
