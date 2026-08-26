# Quantum wavefunction upsampling (ML-assisted)

Harmonic oscillator (and alternatives) ground states on a coarse grid upsampled by a ConvTranspose residual network (`QuantumResNet`: 64 -> 1024). Physics uses `scipy.sparse.linalg.eigsh` on sparse finite-difference Hamiltonians ([`physics_engine.py`](physics_engine.py)).

## Environment

```bash
python -m pip install -r requirements.txt
```

### Development checks

Install dev tools and lint for **undefined names** (`F821`) so missing imports (e.g. `time`, `subprocess` in [`main.py`](main.py)) are caught before runtime. `python -m py_compile` alone does **not** load function bodies fully enough to warn about this.

```bash
python -m pip install -r requirements-dev.txt
python -m ruff check main.py physics_engine.py model.py utils.py
```

## Training (default)

Writes artifacts under **`outputs/`** (`plots/`, **`models/quantum_model.pth`**, `training_log.csv`, `benchmark_results.txt`):

```bash
python main.py
```

Data split: **72% train**, **18% validation** (early stopping only), **10% test** (never seen during training). Dataset cache stays in **`data_cache/`** (gitignored unless you configure otherwise).

## Reproducing the paper's numbers

`python main.py` (defaults: seed 42, n=2000, 30 epochs) retrains the model and writes
`outputs/benchmark_results.txt`, the timing and fidelity summary reported in the paper.
Then:

```bash
python analysis/baseline_comparison.py
```

scores the trained checkpoint against cubic-spline and Fourier zero-padding baselines on
the held-out test split, using the same fidelity function as the training pipeline, and
writes per-sample metrics (`outputs/analysis/test_metrics_canonical.csv`) plus every
statistic in the paper (`outputs/analysis/summary_canonical.json`).

Timing numbers in `benchmark_results.txt` are machine- and load-dependent.

**Note:** `outputs/legacy_from_results/` archives an earlier, superseded run from an older
repository layout; its values differ from the paper's and are not used by it.

### Inference-only (no dataset cache / loading)

Loads weights from **`PATH`**, skips `load_or_generate`, evaluates on **fresh random solves** (default **20** samples):

```bash
python main.py --load PATH/TO/checkpoint.pth
python main.py --load outputs/models/quantum_model.pth --infer-samples 20
```

## Paper snapshots and GitHub

Routine runs overwrite local **`outputs/`**. Do **not** commit after every experiment; keep the repo tidy with **paper runs only**.

After a curated run:

```bash
python main.py --paper-git-stage
python main.py --paper-git-commit --paper-commit-msg "paper Fig 2; harmonic 2026-05"
python main.py --paper-git-commit --paper-git-push --paper-commit-msg "paper reproducibility checkpoint"
```

- **`--paper-git-stage`** --- `git add` **`--output-dir`** (covers `plots/`, **`models/`**, CSV/txt).
- **`--paper-git-commit`** --- also `git commit` (requires **`--paper-commit-msg`**).
- **`--paper-git-push`** --- also `git push` (ensure auth and upstream are configured).

Failures in Git steps do **not** fail the physics/training pipeline (warnings printed).

### Cursor workflows

Plans in Cursor describe work; execution is explicit: Switch to **Agent mode** (or edit locally/push git yourself). There is **no automatic Task 2** &mdash; you define further experiments (alternative potentials, ablations, plots, CI) after the baseline pipeline.

## Outputs layout

| Path | Role |
|------|------|
| `outputs/plots/` | Comparison + error PNGs from the current run |
| `outputs/models/` | Default checkpoint `quantum_model.pth` unless `--model-path` |
| `outputs/training_log.csv` | Epoch train/val loss (training mode only) |
| `outputs/benchmark_results.txt` | Timing + fidelity summary |
| `outputs/legacy_from_results/` | Archived figures/metrics from an older **`results/`** layout |

See [`outputs/README.md`](outputs/README.md).

## Checkpoint path

Default save/load location: **`{--output-dir}/models/quantum_model.pth`**. Override with **`--model-path`** if needed.
