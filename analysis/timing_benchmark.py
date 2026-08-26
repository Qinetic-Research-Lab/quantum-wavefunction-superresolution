"""Repeated timing benchmark: median and range over independent repetitions.

Runs the same measurement as `main.py`'s `benchmark()` (50-run average of the
1024-point eigensolve versus the 64-point eigensolve plus model inference,
omegas drawn from the test split) N_REPS times in one process, and reports the
median and range of the per-repetition speedup. Wall-clock figures are
machine- and load-dependent.

Usage (from the repository root):

    python analysis/timing_benchmark.py

Output: outputs/analysis/timing_summary.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from model import QuantumResNet          # noqa: E402
from physics_engine import QuantumSolver  # noqa: E402

N_REPS = 10
N_REPEAT = 50   # matches main.py benchmark()

CACHE = REPO / "data_cache" / "harmonic_n2000_omega0.5-5.0_seed42.npz"
CHECKPOINT = REPO / "outputs" / "models" / "quantum_model.pth"
OUT = REPO / "outputs" / "analysis" / "timing_summary.json"


def one_rep(solver, model, pool):
    t0 = time.perf_counter()
    for _ in range(N_REPEAT):
        solver._solve(solver.n_high, float(np.random.choice(pool)))
    hr = (time.perf_counter() - t0) / N_REPEAT

    t0 = time.perf_counter()
    for _ in range(N_REPEAT):
        _, psi_lr, _ = solver._solve(solver.n_low, float(np.random.choice(pool)))
        inp = torch.tensor(psi_lr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            model(inp)
    lr_ml = (time.perf_counter() - t0) / N_REPEAT
    return hr, lr_ml


def main() -> None:
    d = np.load(CACHE)
    pool = d["omegas"][-200:]            # test split, as in main.py
    solver = QuantumSolver()
    model = QuantumResNet()
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    model.eval()

    hr_ms, lr_ml_ms, speedups = [], [], []
    for i in range(N_REPS):
        hr, lr_ml = one_rep(solver, model, pool)
        hr_ms.append(hr * 1e3)
        lr_ml_ms.append(lr_ml * 1e3)
        speedups.append(hr / lr_ml)
        print(f"rep {i + 1}/{N_REPS}: HR {hr * 1e3:.2f} ms, "
              f"LR+ML {lr_ml * 1e3:.2f} ms, speedup {hr / lr_ml:.2f}x")

    summary = {
        "n_reps": N_REPS,
        "n_repeat_per_rep": N_REPEAT,
        "hr_ms": {"median": float(np.median(hr_ms)),
                  "min": min(hr_ms), "max": max(hr_ms)},
        "lr_ml_ms": {"median": float(np.median(lr_ml_ms)),
                     "min": min(lr_ml_ms), "max": max(lr_ml_ms)},
        "speedup": {"median": float(np.median(speedups)),
                    "min": min(speedups), "max": max(speedups)},
        "note": "CPU wall-clock; machine- and load-dependent.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
