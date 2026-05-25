"""
Main training and evaluation pipeline for the ML-accelerated quantum
wavefunction upsampling project.

Steps
-----
1. Generate (or load cached) data; split 72%/18%/10% train/val/test.
2. Train QuantumResNet with validation tracking, cosine LR, early stopping.
3. Benchmark timing; fidelity and plots use the held-out **test** set only.
4. Inference (--load PATH) skips caching and uses fresh mini-samples for eval.
"""

import argparse
import csv
import os


from pathlib import Path

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

REPO_ROOT = Path(__file__).resolve().parent

TRAIN_FRAC = 0.72
VAL_FRAC = 0.18
TEST_FRAC = 0.10
N_COMPARE_ERROR_PLOT = 8  # paired comparison + residual per index

FID_THRESHOLD = 0.99

DEFAULTS = dict(
    seed=42,
    n_samples=2000,
    omega_min=0.5,
    omega_max=5.0,
    potential="harmonic",
    epochs=30,
    batch_size=64,
    lr=1e-3,
    base_channels=64,
    patience=7,
    cache_dir="data_cache",
    infer_samples=20,
)


def split_three_way(n_total: int) -> tuple[int, int, int, int, int]:
    """Indices and counts for 72%/18%/10% train / val / test split.

    Returns
    -------
    i_train_end :
        Exclusive end index of the train slice (length = ``n_train``).
    i_val_end :
        Exclusive end index of validation (test starts here).
    """
    n_test = max(1, int(round(n_total * TEST_FRAC)))
    n_val = max(1, int(round(n_total * VAL_FRAC)))
    n_train = max(1, n_total - n_val - n_test)
    # absorb rounding slack into train slice
    if n_train + n_val + n_test != n_total:
        n_train = n_total - n_val - n_test
        assert n_train >= 1

    i_tr = n_train
    i_val_end = i_tr + n_val
    return i_tr, i_val_end, n_train, n_val, n_test


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
    p.add_argument("--base-channels", type=int, default=DEFAULTS["base_channels"])
    p.add_argument("--patience", type=int, default=DEFAULTS["patience"],
                    help="Early stopping patience (0 to disable)")
    p.add_argument("--cache-dir", type=str, default=DEFAULTS["cache_dir"],
                    help="Directory for cached datasets")
    p.add_argument("--no-cache", action="store_true",
                    help="Disable dataset caching (regenerate every run)")
    p.add_argument("--load", type=str, default=None, metavar="PATH",
                    help="Load checkpoint and skip training (no cached dataset)")
    p.add_argument("--infer-samples", type=int, default=DEFAULTS["infer_samples"],
                    help="Fresh solves for benchmarking/plots when using --load")
    p.add_argument("--output-dir", type=str, default="outputs",
                    help="Root directory for plots, logs, checkpoints, benchmark notes")
    p.add_argument(
        "--model-path", type=str, default=None,
        help=(
            "Full path for quantum_model.pth. "
            "Default: OUTPUT_DIR/models/quantum_model.pth"
        ),
    )
    p.add_argument("--paper-git-stage", action="store_true",
                    help="After run: git add OUTPUT_DIR (paper snapshot staging)")
    p.add_argument("--paper-git-commit", action="store_true",
                    help="Also git commit (--paper-commit-msg required)")
    p.add_argument("--paper-git-push", action="store_true",
                    help="Also git push after commit")
    p.add_argument("--paper-commit-msg", type=str, default=None,
                    help='Commit message when using --paper-git-commit')

    args = p.parse_args()
    if (args.paper_git_commit or args.paper_git_push) and args.paper_commit_msg is None:
        p.error("--paper-git-commit and --paper-git-push require --paper-commit-msg")

    return args


def checkpoint_path(args) -> str:
    if args.model_path:
        return os.path.abspath(args.model_path)
    return os.path.abspath(
        os.path.join(args.output_dir, "models", "quantum_model.pth"),
    )


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


def generate_infer_mini(solver: QuantumSolver, n: int,
                        omega_min: float, omega_max: float,
                        rng: np.random.Generator):
    """Fresh random solves for inference-only benchmarking and plots."""
    print(f"Inference-only: generating {n} fresh random samples ...")
    return generate_dataset(solver, n, omega_min, omega_max, rng)


# -----------------------------------------------------------------
# Training with validation + scheduler + early stopping
# -----------------------------------------------------------------

def train(model, train_loader, val_loader, epochs, lr, device, patience=7):
    """Train with MSE loss, cosine-annealing LR, and optional early stopping."""
    model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val = float("inf")
    wait = 0
    best_state = None

    for epoch in range(1, epochs + 1):
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

def benchmark(solver: QuantumSolver, model, omegas_eval, device, n_repeat: int = 50):
    """Compare wall-clock time: full HR solve vs. LR solve plus model inference."""
    model.eval()
    rng_pool = np.asarray(omegas_eval, dtype=np.float64)

    t0 = time.perf_counter()
    for _ in range(n_repeat):
        omega = float(np.random.choice(rng_pool))
        solver._solve(solver.n_high, omega)
    hr_time = (time.perf_counter() - t0) / n_repeat

    t0 = time.perf_counter()
    for _ in range(n_repeat):
        omega = float(np.random.choice(rng_pool))
        _, psi_lr, _ = solver._solve(solver.n_low, omega)
        inp = (torch.tensor(psi_lr, dtype=torch.float32)
               .unsqueeze(0).unsqueeze(0).to(device))
        with torch.no_grad():
            _ = model(inp)
    lr_ml_time = (time.perf_counter() - t0) / n_repeat

    return hr_time, lr_ml_time


# -----------------------------------------------------------------
# Git helpers (paper snapshots; warn-only on failure)
# -----------------------------------------------------------------

def _git_repo_root() -> str | None:
    """Return filesystem root for the git repo, or None."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=20,
        )
        if r.returncode != 0:
            return None
        root = r.stdout.strip()
        return root if root else None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _posix_path(p: str) -> str:
    return Path(p).as_posix()


def _maybe_paper_git(args):
    flags = (
        getattr(args, "paper_git_stage", False)
        or getattr(args, "paper_git_commit", False)
        or getattr(args, "paper_git_push", False)
    )
    if not flags:
        return

    repo = _git_repo_root()
    if not repo:
        print("Paper git: not inside a Git repository — skipping.")
        return

    out_abs = Path(args.output_dir).resolve()
    try:
        rel_out = Path(os.path.relpath(out_abs, Path(repo))).as_posix()
    except ValueError:
        print("Paper git: --output-dir is outside repo root — skipping.")
        return
    if rel_out.startswith("../"):
        print("Paper git: --output-dir escapes repo tree — skipping.")
        return

    def _git_run(cmd):
        proc = subprocess.run(
            cmd, cwd=repo, capture_output=True, text=True, timeout=120,
            shell=False,
        )
        if proc.returncode != 0:
            print(f"Paper git: command failed ({cmd[1]}):\n{proc.stderr or proc.stdout}")
        return proc.returncode == 0

    print(f"\nPaper git: staging {rel_out}")
    ok = _git_run(["git", "add", rel_out])
    ckpt = checkpoint_path(args)
    ckpt_abs = Path(ckpt).resolve()
    try:
        rel_ckpt = Path(os.path.relpath(ckpt_abs, Path(repo))).as_posix()
        if ckpt_abs.is_file() and not rel_ckpt.startswith("../"):
            ok = ok and _git_run(["git", "add", rel_ckpt])
    except ValueError:
        pass

    _git_run(["git", "status", "-s"])

    stage_only = getattr(args, "paper_git_stage", False) and not (
        getattr(args, "paper_git_commit", False)
        or getattr(args, "paper_git_push", False)
    )
    if stage_only:
        print("\nSuggested next commands:")
        msg = getattr(args, "paper_commit_msg", None) or "paper snapshot"
        print(f'  git commit -m "{msg}"')
        print("  git push")
        return

    if getattr(args, "paper_git_commit", False) or getattr(args, "paper_git_push", False):
        msg = getattr(args, "paper_commit_msg", "")
        ok = ok and _git_run(["git", "commit", "-m", msg])
        if getattr(args, "paper_git_push", False) and ok:
            _git_run(["git", "push"])


# =================================================================
# Main
# =================================================================

def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "models"), exist_ok=True)

    rng = np.random.default_rng(args.seed)
    solver = QuantumSolver(L=10.0, n_high=1024, n_low=64,
                           potential=args.potential)

    ck_path = checkpoint_path(args)
    history = None
    fidelity_label = ""

    if args.load:
        # Inference: no cached / full dataset pipeline
        psi_lr_ev, psi_hr_ev, omega_ev = generate_infer_mini(
            solver,
            args.infer_samples,
            args.omega_min,
            args.omega_max,
            np.random.default_rng(args.seed + 9177),
        )
        model = QuantumResNet(base_channels=args.base_channels)
        print(f"Loading model from {args.load}")
        model.load_state_dict(torch.load(
            args.load, map_location=device, weights_only=True,
        ))
        model.to(device)

        X_ev = torch.tensor(psi_lr_ev, dtype=torch.float32).unsqueeze(1)
        n_eval = len(omega_ev)
        fidelity_label = (
            f"inference-only evaluation ({n_eval} freshly generated samples)"
        )
    else:
        psi_lr_all, psi_hr_all, omega_all = load_or_generate(solver, args, rng)

        i_tr, i_va_end, n_tr, n_val, n_te = split_three_way(args.n_samples)
        psi_lr_train = psi_lr_all[:i_tr]
        psi_hr_train = psi_hr_all[:i_tr]
        psi_lr_val = psi_lr_all[i_tr:i_va_end]
        psi_hr_val = psi_hr_all[i_tr:i_va_end]
        psi_lr_ev = psi_lr_all[i_va_end:]
        psi_hr_ev = psi_hr_all[i_va_end:]
        omega_ev = omega_all[i_va_end:]
        n_eval = len(omega_ev)

        X_train = torch.tensor(psi_lr_train, dtype=torch.float32).unsqueeze(1)
        Y_train = torch.tensor(psi_hr_train, dtype=torch.float32).unsqueeze(1)
        X_val = torch.tensor(psi_lr_val, dtype=torch.float32).unsqueeze(1)
        Y_val = torch.tensor(psi_hr_val, dtype=torch.float32).unsqueeze(1)
        X_ev = torch.tensor(psi_lr_ev, dtype=torch.float32).unsqueeze(1)

        train_loader = DataLoader(
            TensorDataset(X_train, Y_train),
            batch_size=args.batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(X_val, Y_val),
            batch_size=args.batch_size,
        )

        print(
            f"Train: {n_tr}  |  Val: {n_val}  |  Test: {n_te} "
            f" |  Device: {device}\n",
        )

        model = QuantumResNet(base_channels=args.base_channels)
        print("Training ...")
        history = train(model, train_loader, val_loader,
                        args.epochs, args.lr, device, args.patience)

        os.makedirs(os.path.dirname(ck_path), exist_ok=True)
        torch.save(model.state_dict(), ck_path)
        print(f"\nModel saved -> {ck_path}")

        _save_training_artifacts(history, args.output_dir)

        fidelity_label = f"held-out test set ({n_eval} samples)"

    # ---- Benchmark ----------------------------------------------
    print("\nBenchmarking (50 runs each) ...")
    hr_t, lr_ml_t = benchmark(solver, model, omega_ev, device)
    speedup = hr_t / lr_ml_t if lr_ml_t > 0 else float("inf")
    print(f"  HR eigsh          : {hr_t * 1e3:8.2f} ms / sample")
    print(f"  LR eigsh + model  : {lr_ml_t * 1e3:8.2f} ms / sample")
    print(f"  Speed-up factor   : {speedup:.2f}x")

    model.eval()
    dx_hr = 2.0 * solver.L / (solver.n_high - 1)
    x_hr = np.linspace(-solver.L, solver.L, solver.n_high)
    x_lr = np.linspace(-solver.L, solver.L, solver.n_low)

    with torch.no_grad():
        preds = model(X_ev.to(device)).cpu().numpy()[:, 0, :]

    fidelities = np.array([
        quantum_fidelity(preds[i], psi_hr_ev[i], dx_hr)
        for i in range(n_eval)
    ])

    print(f"\nQuantum Fidelity ({fidelity_label}):")
    print(f"  Mean   = {fidelities.mean():.6f}")
    print(f"  Median = {np.median(fidelities):.6f}")
    print(f"  Min    = {fidelities.min():.6f}")
    print(f"  Max    = {fidelities.max():.6f}")
    print(f"  Std    = {fidelities.std():.6f}")
    bad = int(np.sum(fidelities < FID_THRESHOLD))
    print(
        f"WARNING: {bad}/{n_eval} samples below fidelity threshold "
        f"(F < {FID_THRESHOLD})\n",
    )

    plots_dir = os.path.join(args.output_dir, "plots")
    plot_indices = np.linspace(
        0,
        max(0, n_eval - 1),
        num=min(N_COMPARE_ERROR_PLOT, n_eval),
        dtype=int,
    )
    for idx in plot_indices:
        iw = omega_ev[idx]
        base = (
            f"comparison_plotidx{idx:d}_omega{iw:.3f}"
            .replace(".", "p")
        )
        eb = (
            f"error_plotidx{idx:d}_omega{iw:.3f}"
            .replace(".", "p")
        )
        save_comparison_plot(
            x_hr, psi_hr_ev[idx],
            x_lr, psi_lr_ev[idx],
            preds[idx], iw,
            save_path=os.path.join(plots_dir, f"{base}.png"),
            fidelity=fidelities[idx],
        )
        save_error_plot(
            x_hr, psi_hr_ev[idx],
            preds[idx],
            iw, fidelities[idx],
            save_path=os.path.join(plots_dir, f"{eb}.png"),
        )
    print(f"\nPlots saved to {plots_dir}/")

    _save_benchmark_results(
        args.output_dir,
        speedup,
        hr_t,
        lr_ml_t,
        fidelities,
        history,
        fidelity_label=fidelity_label,
        infer_mode=bool(args.load),
    )

    _maybe_paper_git(args)


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

    out_base = Path(output_dir).as_posix()
    print(f"Training artefacts saved -> {out_base}/plots/loss_curves.png, "
          f"{out_base}/training_log.csv")


def _save_benchmark_results(
        output_dir, speedup, hr_t, lr_ml_t,
        fidelities, history,
        fidelity_label="", infer_mode=False,
):
    """Write human-readable benchmark and evaluation notes."""
    path = os.path.join(output_dir, "benchmark_results.txt")
    nf = len(fidelities)
    bad = int(np.sum(fidelities < FID_THRESHOLD))
    warn_line = (
        f"WARNING: {bad}/{nf} samples below fidelity threshold "
        f"(F < {FID_THRESHOLD})"
    )

    lines = [
        "Quantum Wavefunction Upsampling — Benchmark Results",
        "=" * 52,
        "",
        ("Evaluation mode : inference-only (fresh solves)"
            if infer_mode else "Evaluation mode : training pipeline (held-out test)"),
        f"Dataset description : {fidelity_label}",
        "",
        "Timing (50-run average per sample)",
        f"  HR eigsh          : {hr_t * 1e3:8.2f} ms",
        f"  LR eigsh + model  : {lr_ml_t * 1e3:8.2f} ms",
        f"  Speed-up factor   : {speedup:.2f}x",
        "",
        f"Quantum fidelity ({nf} samples)",
        f"  Mean   : {fidelities.mean():.6f}",
        f"  Median : {np.median(fidelities):.6f}",
        f"  Min    : {fidelities.min():.6f}",
        f"  Max    : {fidelities.max():.6f}",
        f"  Std    : {fidelities.std():.6f}",
        warn_line,
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
