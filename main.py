"""
Main training and evaluation pipeline for the ML-accelerated quantum
wavefunction upsampling project.

Steps
-----
1. Generate (or load cached) (LR, HR) wavefunction pairs with randomised omega.
2. Train QuantumResNet with validation tracking, LR scheduling, and early stopping.
3. Benchmark: HR eigsh vs. (LR eigsh + model inference).
4. Evaluate quantum fidelity on a held-out test set.
5. Save plots, training log, and benchmark notes to outputs/.
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from physics_engine import QuantumSolver
from model import QuantumResNet
from utils import (
    quantum_fidelity,
    save_comparison_plot,
    save_error_plot,
)

# -----------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------
DEFAULTS = dict(
    seed=42,
    n_samples=2000,
    omega_min=0.5,
    omega_max=5.0,
    potential="harmonic",
    epochs=30,
    batch_size=64,
    lr=1e-3,
    test_split=0.1,
    base_channels=64,
    patience=7,
    cache_dir="data_cache",
)


def parse_args():
    p = argparse.ArgumentParser(
        description="ML-accelerated quantum wavefunction upsampling",
    )
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--n-samples", type=int, default=DEFAULTS["n_samples"])
    p.add_argument("--omega-min", type=float, default=DEFAULTS["omega_min"])
    p.add_argument("--omega-max", type=float, default=DEFAULTS["omega_max"])
    p.add_argument(
        "--potential", choices=["harmonic", "double_well", "anharmonic"],
        default=DEFAULTS["potential"],
        help="Potential type for the Schrödinger equation",
    )
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--test-split", type=float, default=DEFAULTS["test_split"])
    p.add_argument("--base-channels", type=int, default=DEFAULTS["base_channels"])
    p.add_argument("--patience", type=int, default=DEFAULTS["patience"],
                    help="Early stopping patience (0 to disable)")
    p.add_argument("--cache-dir", type=str, default=DEFAULTS["cache_dir"],
                    help="Directory for cached datasets")
    p.add_argument("--no-cache", action="store_true",
                    help="Disable dataset caching (regenerate every run)")
    p.add_argument("--load", type=str, default=None, metavar="PATH",
                    help="Load a trained model and skip training (inference only)")
    p.add_argument("--output-dir", type=str, default="outputs",
                    help="Root directory for plots, logs, and benchmark notes")
    return p.parse_args()


# -----------------------------------------------------------------
# Dataset generation / caching
# -----------------------------------------------------------------

def _cache_path(args) -> str:
    """Deterministic filename based on dataset parameters."""
    tag = (f"{args.potential}_n{args.n_samples}_"
           f"omega{args.omega_min}-{args.omega_max}_seed{args.seed}")
    return os.path.join(args.cache_dir, f"{tag}.npz")


def generate_dataset(solver: QuantumSolver, n_samples: int,
                     omega_min: float, omega_max: float,
                     rng: np.random.Generator):
    """Build parallel arrays of LR and HR wavefunctions."""
    lr_list, hr_list, omegas = [], [], []
    print(f"Generating {n_samples} samples ...")
    for i in range(n_samples):
        result = solver.solve_random_omega(omega_min, omega_max, rng=rng)
        lr_list.append(result["psi_lr"])
        hr_list.append(result["psi_hr"])
        omegas.append(result["omega"])
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n_samples}")
    return np.stack(lr_list), np.stack(hr_list), np.array(omegas)


def load_or_generate(solver, args, rng):
    """Load dataset from cache or generate and save it."""
    cache = _cache_path(args)
    if not args.no_cache and os.path.exists(cache):
        print(f"Loading cached dataset from {cache}")
        data = np.load(cache)
        return data["psi_lr"], data["psi_hr"], data["omegas"]

    psi_lr, psi_hr, omegas = generate_dataset(
        solver, args.n_samples, args.omega_min, args.omega_max, rng,
    )

    if not args.no_cache:
        os.makedirs(args.cache_dir, exist_ok=True)
        np.savez_compressed(cache, psi_lr=psi_lr, psi_hr=psi_hr, omegas=omegas)
        print(f"Dataset cached to {cache}")

    return psi_lr, psi_hr, omegas


# -----------------------------------------------------------------
# Training with validation + scheduler + early stopping
# -----------------------------------------------------------------

def train(model, train_loader, val_loader, epochs, lr, device, patience=7):
    """Train with MSE loss, cosine-annealing LR, and optional early stopping.

    Returns
    -------
    history : dict with keys 'train_loss', 'val_loss', 'lr' (per-epoch lists)
    """
    model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val = float("inf")
    wait = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item() * x_batch.size(0)
        train_avg = epoch_loss / len(train_loader.dataset)

        # --- validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                pred = model(x_batch)
                val_loss += criterion(pred, y_batch).item() * x_batch.size(0)
        val_avg = val_loss / len(val_loader.dataset)

        current_lr = scheduler.get_last_lr()[0]
        history["train_loss"].append(train_avg)
        history["val_loss"].append(val_avg)
        history["lr"].append(current_lr)
        scheduler.step()

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"train={train_avg:.6e}  val={val_avg:.6e}  lr={current_lr:.2e}")

        # --- early stopping ---
        if val_avg < best_val:
            best_val = val_avg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if patience > 0 and wait >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return history


# -----------------------------------------------------------------
# Benchmarking
# -----------------------------------------------------------------

def benchmark(solver: QuantumSolver, model, omegas_test, device, n_repeat: int = 50):
    """Compare wall-clock time: full HR solve vs. (LR solve + model inference)."""
    model.eval()

    t0 = time.perf_counter()
    for _ in range(n_repeat):
        omega = float(np.random.choice(omegas_test))
        solver._solve(solver.n_high, omega)
    hr_time = (time.perf_counter() - t0) / n_repeat

    t0 = time.perf_counter()
    for _ in range(n_repeat):
        omega = float(np.random.choice(omegas_test))
        _, psi_lr, _ = solver._solve(solver.n_low, omega)
        inp = (torch.tensor(psi_lr, dtype=torch.float32)
               .unsqueeze(0).unsqueeze(0).to(device))
        with torch.no_grad():
            _ = model(inp)
    lr_ml_time = (time.perf_counter() - t0) / n_repeat

    return hr_time, lr_ml_time


# =================================================================
# Main
# =================================================================

def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)
    rng = np.random.default_rng(args.seed)
    solver = QuantumSolver(L=10.0, n_high=1024, n_low=64,
                           potential=args.potential)

    # ---- 1. Dataset --------------------------------------------------
    psi_lr_all, psi_hr_all, omegas_all = load_or_generate(solver, args, rng)

    n_test = max(1, int(args.n_samples * args.test_split))
    n_train = args.n_samples - n_test
    psi_lr_train, psi_lr_test = psi_lr_all[:n_train], psi_lr_all[n_train:]
    psi_hr_train, psi_hr_test = psi_hr_all[:n_train], psi_hr_all[n_train:]
    omegas_test = omegas_all[n_train:]

    X_train = torch.tensor(psi_lr_train, dtype=torch.float32).unsqueeze(1)
    Y_train = torch.tensor(psi_hr_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(psi_lr_test, dtype=torch.float32).unsqueeze(1)
    Y_test = torch.tensor(psi_hr_test, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(
        TensorDataset(X_train, Y_train), batch_size=args.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_test, Y_test), batch_size=args.batch_size,
    )
    print(f"Train: {n_train}  |  Test: {n_test}  |  Device: {device}\n")

    # ---- 2. Train or load --------------------------------------------
    model = QuantumResNet(base_channels=args.base_channels)

    if args.load:
        print(f"Loading model from {args.load}")
        model.load_state_dict(torch.load(args.load, map_location=device,
                                         weights_only=True))
        model.to(device)
        history = None
    else:
        print("Training ...")
        history = train(model, train_loader, val_loader,
                        args.epochs, args.lr, device, args.patience)

        torch.save(model.state_dict(), "quantum_model.pth")
        print("\nModel saved -> quantum_model.pth")

        _save_training_artifacts(history, args.output_dir)

    # ---- 3. Benchmark ------------------------------------------------
    print("\nBenchmarking (50 runs each) ...")
    hr_t, lr_ml_t = benchmark(solver, model, omegas_test, device)
    speedup = hr_t / lr_ml_t if lr_ml_t > 0 else float("inf")
    print(f"  HR eigsh          : {hr_t*1e3:8.2f} ms / sample")
    print(f"  LR eigsh + model  : {lr_ml_t*1e3:8.2f} ms / sample")
    print(f"  Speed-up factor   : {speedup:.2f}x")

    # ---- 4. Evaluation: fidelity & plots -----------------------------
    model.eval()
    dx_hr = 2.0 * solver.L / (solver.n_high - 1)
    x_hr = np.linspace(-solver.L, solver.L, solver.n_high)
    x_lr = np.linspace(-solver.L, solver.L, solver.n_low)

    with torch.no_grad():
        preds = model(X_test.to(device)).cpu().numpy()[:, 0, :]

    fidelities = np.array([
        quantum_fidelity(preds[i], psi_hr_test[i], dx_hr)
        for i in range(n_test)
    ])

    print(f"\nQuantum Fidelity on test set ({n_test} samples):")
    print(f"  Mean   = {fidelities.mean():.6f}")
    print(f"  Median = {np.median(fidelities):.6f}")
    print(f"  Min    = {fidelities.min():.6f}")
    print(f"  Max    = {fidelities.max():.6f}")

    # ---- 5. Canonical plots ------------------------------------------
    plots_dir = os.path.join(args.output_dir, "plots")
    median_idx = int(np.argsort(fidelities)[len(fidelities) // 2])
    save_comparison_plot(
        x_hr, psi_hr_test[median_idx],
        x_lr, psi_lr_test[median_idx],
        preds[median_idx], omegas_test[median_idx],
        save_path=os.path.join(plots_dir, "wavefunction_comparison.png"),
        fidelity=fidelities[median_idx],
    )

    worst_idx = int(np.argmin(fidelities))
    save_error_plot(
        x_hr, psi_hr_test[worst_idx], preds[worst_idx],
        omegas_test[worst_idx], fidelities[worst_idx],
        save_path=os.path.join(plots_dir, "pointwise_residuals.png"),
    )
    print(f"\nPlots saved to {plots_dir}/")

    # ---- 6. Benchmark notes ------------------------------------------
    _save_benchmark_results(
        args.output_dir, speedup, hr_t, lr_ml_t, fidelities, history,
    )


# -----------------------------------------------------------------
# Helpers for saving artefacts
# -----------------------------------------------------------------

def _save_training_artifacts(history, output_dir):
    """Save loss curves plot and per-epoch training log CSV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.semilogy(epochs, history["train_loss"], label="Train")
    ax1.semilogy(epochs, history["val_loss"], label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("Training Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["lr"])
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("LR Schedule (Cosine Annealing)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "loss_curves.png"), dpi=150)
    plt.close(fig)

    log_path = os.path.join(output_dir, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr"])
        for i, epoch in enumerate(epochs):
            writer.writerow([
                epoch,
                f"{history['train_loss'][i]:.8e}",
                f"{history['val_loss'][i]:.8e}",
                f"{history['lr'][i]:.2e}",
            ])

    print("Training artefacts saved -> outputs/plots/loss_curves.png, outputs/training_log.csv")


def _save_benchmark_results(output_dir, speedup, hr_t, lr_ml_t, fidelities, history):
    """Write human-readable benchmark and evaluation notes."""
    path = os.path.join(output_dir, "benchmark_results.txt")
    lines = [
        "Quantum Wavefunction Upsampling — Benchmark Results",
        "=" * 52,
        "",
        "Timing (50-run average per sample)",
        f"  HR eigsh          : {hr_t * 1e3:8.2f} ms",
        f"  LR eigsh + model  : {lr_ml_t * 1e3:8.2f} ms",
        f"  Speed-up factor   : {speedup:.2f}x",
        "",
        f"Quantum fidelity ({len(fidelities)} test samples)",
        f"  Mean   : {fidelities.mean():.6f}",
        f"  Median : {np.median(fidelities):.6f}",
        f"  Min    : {fidelities.min():.6f}",
        f"  Max    : {fidelities.max():.6f}",
        f"  Std    : {fidelities.std():.6f}",
    ]
    if history is not None:
        lines.extend([
            "",
            "Training summary",
            f"  Epochs trained : {len(history['train_loss'])}",
            f"  Best val loss  : {min(history['val_loss']):.8e}",
            f"  Final train    : {history['train_loss'][-1]:.8e}",
            f"  Final val      : {history['val_loss'][-1]:.8e}",
        ])

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Benchmark notes saved -> {path}")


if __name__ == "__main__":
    main()
