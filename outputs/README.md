# outputs/

Artifacts from [`main.py`](../main.py):

- **`plots/`** — Current-run comparison and residual plots (paired per index).
- **`models/`** — Trained checkpoints (`quantum_model.pth` by default).
- **`training_log.csv`**, **`benchmark_results.txt`** — Training curves and timings/fidelity summary.
- **`legacy_from_results/`** — Frozen copy of plots and tables from when the repo used a top-level **`results/`** directory.

For Git: commit only deliberate **paper** snapshots of this tree (plus optional checkpoint under **`models/`**).
