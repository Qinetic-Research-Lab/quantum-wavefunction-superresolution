# RESULTS_SUMMARY.md

These are new experimental results. The canonical run (seed=42, outputs/) and the current manuscript are UNCHANGED. Incorporating any of these numbers into the paper is a separate decision to be made after review.

All five `python main.py` jobs ran **sequentially** (one process at a time) after cheap Tasks A–C: widerange → seed 43 → seed 44 → seed 45 → seed 46.

---

## Task A — Fourier baseline (grid-handling fix)

**Hypothesis confirmed.** Index-space `rfft`/`irfft` maps LR index 32 (`x = 0.158730`) onto HR index 512 (`x = 0.009775`): a rigid shift of **0.148955** length units at the domain center (~0.15 as predicted).

Fix used: DFT on the unique periodic points `psi[:-1]` (63 samples, period `2L`), zero-pad 63 → 1023, restore the endpoint onto the solver’s 1024-point inclusive grid. This is still Fourier zero-padding, not a DST substitution.

Old FFT mean F on the canonical 200-sample test set is **0.969008**, matching `outputs/analysis/summary_canonical.json` (`fft_mean`). After the fix, mean F rises to **0.999770**.

| method | mean F | median F | min F | max F |
|---|---|---|---|---|
| old FFT (index-space rFFT) | 0.969008 | 0.969256 | 0.944092 | 0.994355 |
| fixed FFT (endpoint-corrected) | 0.999770 | 0.999821 | 0.999361 | 0.999995 |

Per Table 1 bin, mean fidelity:

| ω bin | n | old FFT F | fixed FFT F |
|---|---|---|---|
| [0.5, 1.4) | 36 | 0.989648 | 0.999979 |
| [1.4, 2.3) | 36 | 0.980681 | 0.999930 |
| [2.3, 3.2) | 46 | 0.969822 | 0.999826 |
| [3.2, 4.1) | 42 | 0.959795 | 0.999682 |
| [4.1, 5.0) | 40 | 0.948664 | 0.999466 |

CNN and spline numbers in existing outputs were not rewritten. Details: `outputs/fourier_baseline_fix_report.md`.

---

## Task B — Oracle-family baseline

ω is fitted from each 64-point input only (HO identity `⟨x²⟩ = 1/(2ω)`; all 200 samples used the moment estimator; no least-squares fallback). The analytic ground state is scored against the **numerical** HR eigenvector with `utils.quantum_fidelity`.

Mean |Δω| (fitted vs true) = 0.130. Analytic wavefunction at the *true* ω has F ≈ 1 against numerical HR (discretization is not the error source); the oracle’s residual is the ω-fit error.

| method (canonical test, n=200) | mean F | median F | min F | max F |
|---|---|---|---|---|
| CNN (existing Table 1 / canonical) | 0.999832 | 0.999889 | 0.999328 | 0.999909 |
| cubic spline (existing) | 0.999790 | 0.999831 | 0.999442 | 0.999995 |
| old FFT (existing) | 0.969008 | — | — | — |
| **oracle (fitted ω)** | **0.999789** | **0.999835** | **0.999416** | **0.999995** |
| fixed FFT (Task A, new) | 0.999770 | 0.999821 | 0.999361 | 0.999995 |

Per Table 1 bin, oracle vs existing CNN / spline mean F:

| ω bin | n | CNN F | spline F | oracle F |
|---|---|---|---|---|
| [0.5, 1.4) | 36 | 0.999680 | 0.999980 | 0.999981 |
| [1.4, 2.3) | 36 | 0.999894 | 0.999932 | 0.999936 |
| [2.3, 3.2) | 46 | 0.999896 | 0.999836 | 0.999840 |
| [3.2, 4.1) | 42 | 0.999892 | 0.999708 | 0.999709 |
| [4.1, 5.0) | 40 | 0.999777 | 0.999527 | 0.999511 |

The oracle tracks cubic spline, not CNN: it inherits the same ω-growing error of a correctly located Gaussian with a slightly wrong width. Details: `outputs/oracle_baseline_report.md`.

---

## Task C — Sigma-space reanalysis

Source: existing `outputs/analysis/test_metrics_canonical.csv` only. ω midpoint = 2.7498664701 from `summary_canonical.json`.

Spearman correlation with **CNN infidelity**:

| predictor | ρ | p |
|---|---|---|
| ω | 0.026086 | 7.14×10⁻¹ |
| \|ω − ω_mid\| | 0.861259 | 4.04×10⁻⁶⁰ |
| σ = 1/√ω | −0.026086 | 7.14×10⁻¹ |
| log(ω) | 0.026086 | 7.14×10⁻¹ |

σ = 1/√ω is a strictly decreasing function of ω, and log(ω) is strictly increasing, so Spearman vs σ / log(ω) **must** equal −ρ(ω) / +ρ(ω). CNN infidelity is **not** tracking wavefunction width; it tracks distance from the training-range midpoint.

Spline infidelity vs ω is ρ = 1 (monotone), vs σ is ρ = −1, vs \|ω − ω_mid\| is ρ = 0.031 (n.s.).

Table 1 edge-bin CNN infidelity:

- Low [0.5, 1.4): 3.1998×10⁻⁴ (n=36; paper 3.20×10⁻⁴)
- High [4.1, 5.0): 2.2333×10⁻⁴ (n=40; paper 2.23×10⁻⁴)
- Ratio low/high = **1.433** (paper ~1.4×)

Details: `outputs/sigma_space_reanalysis.md`.

---

## Task D — Widened-ω ablation (ω ∈ [0.3, 7.0], N=3000, seed=42)

Pre-flight (passed; did not stop):

- dx_LR = 20/63 = 0.31746
- points-per-width σ/dx_LR: **5.751** at ω=0.3, **1.191** at ω=7.0
- 5 σ_ψ(0.3) = **9.129 < 10**

Independent cache: `data_cache_widerange/harmonic_n3000_omega0.3-7.0_seed42.npz`. Split 2160/540/300. Canonical `data_cache/harmonic_n2000_*` and `outputs/` were not written.

| method | mean F | median F | min F |
|---|---|---|---|
| CNN | 0.999923 | 0.999945 | 0.999613 |
| cubic spline | 0.999639 | 0.999721 | 0.998950 |
| fixed FFT | 0.999579 | 0.999698 | 0.998681 |
| old FFT (diagnostic) | 0.960381 | 0.960609 | 0.921210 |

Crossover (unbroken CNN-win suffix): ω = **1.528** (last spline win 1.516). CNN wins 237/300.

Per-bin CNN / spline / fixed-FFT mean infidelity:

| ω bin | n | CNN infid | spline infid | fixed FFT infid |
|---|---|---|---|---|
| [0.3, 0.5) *new edge* | 9 | 3.15×10⁻⁴ | 3.32×10⁻⁶ | 3.34×10⁻⁶ |
| [0.5, 1.4) *old low edge, now interior* | 48 | 1.09×10⁻⁴ | 2.24×10⁻⁵ | 2.28×10⁻⁵ |
| [1.4, 2.3) | 43 | 4.94×10⁻⁵ | 7.48×10⁻⁵ | 7.75×10⁻⁵ |
| [2.3, 3.2) | 35 | 6.26×10⁻⁵ | 1.70×10⁻⁴ | 1.80×10⁻⁴ |
| [3.2, 4.1) | 40 | 5.56×10⁻⁵ | 3.05×10⁻⁴ | 3.32×10⁻⁴ |
| [4.1, 5.0) *old high edge, now interior* | 45 | 5.45×10⁻⁵ | 4.71×10⁻⁴ | 5.32×10⁻⁴ |
| [5.0, 7.0] *new edge* | 80 | 7.40×10⁻⁵ | 8.10×10⁻⁴ | 9.79×10⁻⁴ |

### Edge-following prediction — not a full yes

Two-part claim from the prompt:

1. **CNN infidelity at the old edges, now interior, is lower than in the canonical run: YES.**
   - [0.5, 1.4): 3.20×10⁻⁴ (canonical) → 1.09×10⁻⁴ (widerange)
   - [4.1, 5.0): 2.23×10⁻⁴ (canonical) → 5.45×10⁻⁵ (widerange)
2. **CNN infidelity is now worst at the new edges ω≈0.3 and ω≈7.0: NO.**
   - Worst bin is the new *low* edge [0.3, 0.5) at 3.15×10⁻⁴ (n=9, small).
   - New high edge [5.0, 7.0] is only mildly elevated (7.40×10⁻⁵) relative to the mid-range (~5–6×10⁻⁵) and is **not** the worst CNN bin.

**Overall: partially confirmed.** Interiorizing the old ω edges reduced CNN error there. Error still piles up at the *low*-ω (large-σ) edge of whatever range is trained; it does not equally pile up at the new high-ω edge.

---

## Task E — Multi-seed replication (canonical config)

Headline crossover = start of the unbroken high-ω CNN-win suffix (same definition as `paper/make_figure1.py`). Seed 42 is the existing canonical run (not retrained).

| seed | CNN mean F | spline mean F | crossover ω | last spline win |
|---|---|---|---|---|
| 42 (canonical) | 0.999832 | 0.999790 | 2.085 | 2.029 |
| 43 | 0.999875 | 0.999776 | 2.103 | 2.074 |
| 44 | 0.999914 | 0.999769 | 1.732 | 1.660 |
| 45 | 0.999847 | 0.999811 | 2.013 | 2.007 |
| 46 | 0.999878 | 0.999818 | 1.819 | 1.806 |
| **mean ± range** | **0.999869 ± 8.15×10⁻⁵** | **0.999793 ± 4.96×10⁻⁵** | **1.950 ± 0.371** | **1.915 ± 0.414** |

Range = max − min over the five seeds (not a standard error).

Per-bin **CNN** infidelity, mean ± range:

| ω bin | mean infid | min | max | range |
|---|---|---|---|---|
| [0.5, 1.4) | 2.42×10⁻⁴ | 1.24×10⁻⁴ | 3.20×10⁻⁴ | 1.96×10⁻⁴ |
| [1.4, 2.3) | 8.65×10⁻⁵ | 6.74×10⁻⁵ | 1.06×10⁻⁴ | 3.89×10⁻⁵ |
| [2.3, 3.2) | 8.76×10⁻⁵ | 7.30×10⁻⁵ | 1.04×10⁻⁴ | 3.06×10⁻⁵ |
| [3.2, 4.1) | 8.71×10⁻⁵ | 6.41×10⁻⁵ | 1.08×10⁻⁴ | 4.41×10⁻⁵ |
| [4.1, 5.0) | 1.48×10⁻⁴ | 1.09×10⁻⁴ | 2.23×10⁻⁴ | 1.15×10⁻⁴ |

Per-bin **spline** infidelity, mean ± range:

| ω bin | mean infid | min | max | range |
|---|---|---|---|---|
| [0.5, 1.4) | 1.97×10⁻⁵ | 1.88×10⁻⁵ | 2.09×10⁻⁵ | 2.10×10⁻⁶ |
| [1.4, 2.3) | 7.38×10⁻⁵ | 6.75×10⁻⁵ | 7.74×10⁻⁵ | 9.85×10⁻⁶ |
| [2.3, 3.2) | 1.72×10⁻⁴ | 1.64×10⁻⁴ | 1.79×10⁻⁴ | 1.47×10⁻⁵ |
| [3.2, 4.1) | 2.99×10⁻⁴ | 2.91×10⁻⁴ | 3.13×10⁻⁴ | 2.18×10⁻⁵ |
| [4.1, 5.0) | 4.63×10⁻⁴ | 4.55×10⁻⁴ | 4.73×10⁻⁴ | 1.77×10⁻⁵ |

CNN overall fidelity is stable near 0.99987 (span 8×10⁻⁵). Crossover sits in ~1.73–2.10. Spline per-bin infidelity is far more stable across seeds than CNN, as expected for a deterministic interpolant on independently drawn test omegas.

---

## File inventory

### Existing files modified

**None.** SHA-256 prefixes of canonical artifacts were identical before Tasks A–C and after all five training jobs:

- `outputs/analysis/test_metrics_canonical.csv` `98c1643168c6a96e`
- `outputs/analysis/summary_canonical.json` `daae9c56824ee6b5`
- `outputs/models/quantum_model.pth` `5b1fd0e70d5f1434`
- `outputs/benchmark_results.txt` `718e0a4389354770`
- `outputs/training_log.csv` `6aa55205a5b5165e`

`main.py`, `model.py`, `physics_engine.py`, `utils.py`, `analysis/baseline_comparison.py`, `paper/`, and all manuscript / context drafts were not edited. `data_cache/harmonic_n2000_omega0.5-5.0_seed42.npz` was loaded, never overwritten.

### New analysis code

| file | description |
|---|---|
| `analysis/__init__.py` | Package marker so analysis scripts can share helpers |
| `analysis/experiment_common.py` | Shared grids, old/fixed FFT, oracle fit, crossover, write-guards |
| `analysis/fourier_baseline_fix.py` | Task A: old vs fixed FFT on the canonical test set |
| `analysis/oracle_baseline.py` | Task B: fitted-ω analytic HO baseline |
| `analysis/sigma_space_reanalysis.py` | Task C: Spearman vs σ and log(ω) from the saved CSV |
| `analysis/sanity_widerange.py` | Task D pre-flight truncation / points-per-width checks |
| `analysis/score_run.py` | Isolated-run CNN/spline/FFT scorer (refuses canonical paths) |

### New reports under `outputs/` (additive; did not replace existing files)

| file | description |
|---|---|
| `outputs/fourier_baseline_fix_report.md` | Task A human-readable report |
| `outputs/fourier_baseline_fix_report.json` | Task A machine-readable numbers |
| `outputs/oracle_baseline_report.md` | Task B human-readable report |
| `outputs/oracle_baseline_report.json` | Task B machine-readable numbers |
| `outputs/sigma_space_reanalysis.md` | Task C human-readable report |
| `outputs/sigma_space_reanalysis.json` | Task C machine-readable numbers |

### New experiment directories

| path | description |
|---|---|
| `outputs_widerange_ablation/` | Task D training artifacts (`models/quantum_model.pth`, plots, `training_log.csv`, `benchmark_results.txt`) plus `test_metrics.csv`, `baseline_summary.json`, `baseline_report.md` |
| `outputs_seed43/` | Seed-43 canonical-config run + baseline scores |
| `outputs_seed44/` | Seed-44 canonical-config run + baseline scores |
| `outputs_seed45/` | Seed-45 canonical-config run + baseline scores |
| `outputs_seed46/` | Seed-46 canonical-config run + baseline scores |
| `data_cache_widerange/harmonic_n3000_omega0.3-7.0_seed42.npz` | Task D dataset (separate cache dir) |
| `data_cache/harmonic_n2000_omega0.5-5.0_seed43.npz` | Seed-43 dataset (does not collide with seed 42) |
| `data_cache/harmonic_n2000_omega0.5-5.0_seed44.npz` | Seed-44 dataset |
| `data_cache/harmonic_n2000_omega0.5-5.0_seed45.npz` | Seed-45 dataset |
| `data_cache/harmonic_n2000_omega0.5-5.0_seed46.npz` | Seed-46 dataset |
| `RESULTS_SUMMARY.md` | This file |
